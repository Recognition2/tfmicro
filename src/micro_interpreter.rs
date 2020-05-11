//! Micro interpreter
//!
//! # Usage
//!
//! ```rust
//! # use tfmicro::{
//! #     micro_interpreter::MicroInterpreter, micro_op_resolver::MicroOpResolver,
//! #     model::Model,
//! # };
//! // model
//! let model = include_bytes!("../examples/models/hello_world.tflite");
//! let model = Model::from_buffer(&model[..]).unwrap();
//!
//! // resolver
//! let all_op_resolver = MicroOpResolver::new_with_all_ops();
//!
//! // arena
//! const TENSOR_ARENA_SIZE: usize = 4 * 1024;
//! let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];
//!
//! let _ = MicroInterpreter::new(
//!     &model,
//!     all_op_resolver,
//!     &mut tensor_arena[..],
//! );
//! ```
//!
//! Remember that once once you have instantiated the `MicroInterpreter`,
//! the references you provided for `model`, `tensor_arena` must remain in
//! scope. This is because the underlying C++ microinterpreter
//! contains pointers to these objects.
//!
//! For example, the following will not compile:
//!
//! ```compile_fail
//! # use tfmicro::{
//! #     micro_interpreter::MicroInterpreter, micro_op_resolver::MicroOpResolver,
//! #     model::Model,
//! # };
//! let mut interpreter = {
//!     let model = include_bytes!("../examples/models/hello_world.tflite");
//!     let model = Model::from_buffer(&model[..]).unwrap();
//!
//!     // ...
//! # let all_op_resolver = MicroOpResolver::new_with_all_ops();
//! # const TENSOR_ARENA_SIZE: usize = 4 * 1024;
//! # let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];
//!
//!     MicroInterpreter::new(
//!         &model,
//!         all_op_resolver,
//!         &mut tensor_arena[..],
//!     )
//! }; // Error [model, ..] dropped here whilst still borrowed
//!
//! // interpreter used here
//! # interpreter.input(0);
//! ```

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::micro_error_reporter::MicroErrorReporter;
use crate::micro_op_resolver::MicroOpResolver;
use crate::{model::Model, tensor::Tensor, Status};
use managed::ManagedSlice;

use crate::bindings;
use crate::bindings::tflite;

cpp! {{
    #include "tensorflow/lite/micro/micro_interpreter.h"
    #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
    #include "tensorflow/lite/micro/kernels/micro_ops.h"
    #include "tensorflow/lite/micro/micro_error_reporter.h"
    #include "tensorflow/lite/micro/testing/micro_test.h"
    #include "tensorflow/lite/schema/schema_generated.h"
    #include "tensorflow/lite/version.h"
}}

static mut ERROR_REPORTER: MaybeUninit<MicroErrorReporter> =
    MaybeUninit::uninit();

pub struct MicroInterpreter<'a> {
    // bindgen types
    micro_interpreter: tflite::MicroInterpreter,

    // See https://doc.rust-lang.org/std/marker/struct.PhantomData.html#unused-lifetime-parameters
    _phantom: PhantomData<&'a ()>,
}

impl<'a> MicroInterpreter<'a> {
    // From tensorflow source:
    // tensorflow/lite/micro/micro_interpreter.h
    //
    // "The lifetime of the model, op resolver, tensor arena, and error
    // reporter must be at least as long as that of the interpreter object,
    // since the interpreter may need to access them at any time. This
    // means that you should usually create them with the same scope as
    // each other, for example having them all allocated on the stack as
    // local variables through a top-level function.  The interpreter
    // doesn't do any deallocation of any of the pointed-to objects,
    // ownership remains with the caller."

    /// Create a new micro_interpreter from a Model, a MicroOpResolver and
    /// a tensor arena (scratchpad).
    ///
    pub fn new<'m: 'a, 't: 'a, TArena>(
        model: &'m Model,
        resolver: MicroOpResolver,
        tensor_arena: TArena,
    ) -> Self
    where
        TArena: Into<ManagedSlice<'t, u8>>,
    {
        let mut tensor_arena = tensor_arena.into();

        let tensor_arena_size = tensor_arena.len();
        let tensor_arena = tensor_arena.as_mut_ptr();

        // Idempotent block to get a pointer to a MicroErrorReporter
        let micro_error_reporter_ref = unsafe {
            // Initialise MicroErrorReporter. We assume that `new` is a pure
            // function that only fills in the MicroErrorReporter vtable
            let micro_error_reporter = MicroErrorReporter::new();
            *&mut ERROR_REPORTER = MaybeUninit::new(micro_error_reporter);

            &ERROR_REPORTER // return reference with 'static lifetime
        };

        // Create interpreter
        let micro_interpreter = unsafe {
            cpp! ([
                model as "const tflite::Model*",
                resolver as "tflite::MicroMutableOpResolver",
                tensor_arena as "uint8_t*",
                tensor_arena_size as "size_t",
                micro_error_reporter_ref as "tflite::MicroErrorReporter*"
            ] -> tflite::MicroInterpreter as "tflite::MicroInterpreter"
              {
                  tflite::ErrorReporter* error_reporter = micro_error_reporter_ref;
                  // Build an interpreter to run the model with.
                  tflite::MicroInterpreter interpreter(model,
                                                       resolver,
                                                       tensor_arena,
                                                       tensor_arena_size,
                                                       error_reporter);

                  interpreter.AllocateTensors();

                  return interpreter;
              })
        };

        // Create self
        Self {
            micro_interpreter,
            _phantom: PhantomData,
        }
    }

    /// Returns a mutable reference to the nth input tensor
    ///
    pub fn input(&mut self, n: usize) -> &'a mut Tensor {
        let interpreter = &self.micro_interpreter;
        unsafe {
            // Call method on micro_interpreter
            let inp = cpp!([
                interpreter as "tflite::MicroInterpreter*",
                n as "size_t"]
                -> *mut bindings::TfLiteTensor as "TfLiteTensor*" {
                return interpreter->input(n);
            });

            // Check result
            assert!(!inp.is_null(), "Obtained nullptr from TensorFlow");

            // From bindgen type to Rust type
            inp.into()
        }
    }

    /// Invoke runs the Tensorflow operation to transform inputs to outputs
    ///
    pub fn invoke(&mut self) -> Result<(), Status> {
        let interpreter = &self.micro_interpreter;

        let status = unsafe {
            cpp!([interpreter as "tflite::MicroInterpreter*"]
                  -> bindings::TfLiteStatus as "TfLiteStatus" {
                return interpreter->Invoke();
            })
        };

        // Return result
        match status.into() {
            Status::Ok => Ok(()),
            e => Err(e),
        }
    }

    /// Returns an immutable reference to the nth output tensor
    ///
    pub fn output(&self, n: usize) -> &'a Tensor {
        let interpreter = &self.micro_interpreter;
        unsafe {
            // Call method on micro_interpreter
            let out = cpp!([
                interpreter as "tflite::MicroInterpreter*",
                n as "size_t"]
                    -> *mut bindings::TfLiteTensor as "TfLiteTensor*" {
                return interpreter->output(n);
            });

            // Check result
            assert!(!out.is_null(), "Obtained nullptr from Tensorflow!");

            // From bindgen type to Rust type
            out.into()
        }
    }

    /// Returns the actual number of bytes required for the arena
    ///
    pub fn arena_used_bytes(&self) -> usize {
        let interpreter = &self.micro_interpreter;
        unsafe {
            cpp!([interpreter as "tflite::MicroInterpreter*"]
                  -> usize as "size_t" {
                return interpreter->arena_used_bytes();
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_interpreter_static_arena() {
        // model
        let model = include_bytes!("../examples/models/hello_world.tflite");
        let model = Model::from_buffer(&model[..]).unwrap();

        // resolver
        let all_op_resolver = MicroOpResolver::new_with_all_ops();

        // arena
        const TENSOR_ARENA_SIZE: usize = 4 * 1024;
        let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];

        let _ = MicroInterpreter::new(
            &model,
            all_op_resolver,
            &mut tensor_arena[..],
        );
    }

    #[cfg(feature = "alloc")]
    extern crate alloc;

    #[cfg(feature = "alloc")]
    use alloc::{vec, vec::Vec};

    #[test]
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn new_interpreter_alloc_arena() {
        // model
        let model = include_bytes!("../examples/models/hello_world.tflite");
        let model = Model::from_buffer(&model[..]).unwrap();

        // resolver
        let all_op_resolver = MicroOpResolver::new_with_all_ops();

        // arena
        let tensor_arena: Vec<u8> = vec![0u8; 4 * 1024];

        let _ = MicroInterpreter::new(&model, all_op_resolver, tensor_arena);
    }
}
