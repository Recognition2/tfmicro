//! A wrapper around the Tensorflow `MicroInterpreter`. Much like in the C
//! API, this type takes a model, a set of working memory and the input
//! tensors. It is then used to access the output tensors.
//!
//! See [`MicroInterpreter`](MicroInterpreter).
//!
//! # Usage
//!
//! ```rust
//! # use tfmicro::{
//! #     MicroInterpreter, AllOpResolver, Model,
//! # };
//! // model
//! let model = include_bytes!("../examples/models/hello_world.tflite");
//! let model = Model::from_buffer(&model[..]).unwrap();
//!
//! // resolver
//! let all_op_resolver = AllOpResolver::new();
//!
//! // arena
//! const TENSOR_ARENA_SIZE: usize = 4 * 1024;
//! let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];
//!
//! let _ = MicroInterpreter::new(
//!     &model,
//!     all_op_resolver,
//!     &mut tensor_arena[..],
//! ).unwrap();
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
//! #     MicroInterpreter, AllOpResolver, Model,
//! # };
//! let mut interpreter = {
//!     let model = include_bytes!("../examples/models/hello_world.tflite");
//!     let model = Model::from_buffer(&model[..]).unwrap();
//!
//!     // ...
//! # let all_op_resolver = AllOpResolver::new();
//! # const TENSOR_ARENA_SIZE: usize = 4 * 1024;
//! # let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];
//!
//!     MicroInterpreter::new(
//!         &model,
//!         all_op_resolver,
//!         &mut tensor_arena[..],
//!     ).unwrap()
//! }; // Error [model, ..] dropped here whilst still borrowed
//!
//! // interpreter used here
//! interpreter.input_info(0);
//! ```

use core::convert::TryInto;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::micro_error_reporter::MicroErrorReporter;
use crate::micro_op_resolver::OpResolverRepr;
use crate::tensor::{ElemTypeOf, Tensor, TensorInfo};
use crate::Error;
use crate::{model::Model, Status};
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
    // #include "tensorflow/lite/version.h"
}}

static mut ERROR_REPORTER: MaybeUninit<MicroErrorReporter> =
    MaybeUninit::uninit();

/// An interpreter for TensorFlow models
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

    /// Create a new micro_interpreter from a Model, a MicroOpResolver and a
    /// tensor arena (scratchpad).
    ///
    /// # Errors
    ///
    /// Returns `Error::InterpreterInitError` if there is an error creating
    /// the interpreter.
    ///
    /// Returns `Error::AllocateTensors` if there is error in the call to
    /// `AllocateTensors`.
    pub fn new<'m: 'a, 't: 'a, TArena, OpResolver>(
        model: &'m Model,
        resolver: OpResolver,
        tensor_arena: TArena,
    ) -> Result<Self, Error>
    where
        OpResolver: OpResolverRepr,
        TArena: Into<ManagedSlice<'t, u8>>,
    {
        let resolver = resolver.to_inner();

        let mut tensor_arena = tensor_arena.into();

        let tensor_arena_size = tensor_arena.len();
        let tensor_arena = tensor_arena.as_mut_ptr();

        // Idempotent block to get a pointer to a MicroErrorReporter
        let micro_error_reporter_ref = unsafe {
            // Initialise MicroErrorReporter. We assume that `new` is a pure
            // function that only fills in the MicroErrorReporter vtable
            let micro_error_reporter = MicroErrorReporter::new();
            ERROR_REPORTER = MaybeUninit::new(micro_error_reporter);

            &ERROR_REPORTER // return reference with 'static lifetime
        };

        let mut status = bindings::TfLiteStatus::kTfLiteError;

        // Create interpreter
        let mut micro_interpreter = unsafe {
            let status_ref = &mut status;

            cpp! ([
                model as "const tflite::Model*",
                resolver as "tflite::MicroMutableOpResolver<128>",
                tensor_arena as "uint8_t*",
                tensor_arena_size as "size_t",
                micro_error_reporter_ref as "tflite::MicroErrorReporter*",
                status_ref as "TfLiteStatus*"
            ] -> tflite::MicroInterpreter as "tflite::MicroInterpreter"
              {
                  tflite::ErrorReporter* error_reporter = micro_error_reporter_ref;
                  // Build an interpreter to run the model with.
                  tflite::MicroInterpreter interpreter(model,
                                                       resolver,
                                                       tensor_arena,
                                                       tensor_arena_size,
                                                       error_reporter);

                  // Get status
                  *status_ref = interpreter.initialization_status();

                  return interpreter;
              })
        };
        if status != bindings::TfLiteStatus::kTfLiteOk {
            return Err(Error::InterpreterInitError);
        }

        // Allocate tensors
        let allocate_tensors_status = unsafe {
            let interpreter_ref = &mut micro_interpreter;

            cpp! ([interpreter_ref as "tflite::MicroInterpreter*"]
                   -> bindings::TfLiteStatus as "TfLiteStatus" {
                return interpreter_ref->AllocateTensors();
            })
        };
        if allocate_tensors_status != bindings::TfLiteStatus::kTfLiteOk {
            return Err(Error::AllocateTensorsError);
        }

        // Create self
        Ok(Self {
            micro_interpreter,
            _phantom: PhantomData,
        })
    }

    /// Returns a [`TensorInfo`](crate::tensor::TensorInfo) that describes
    /// the `n`th input tensor.
    ///
    /// # Panics
    ///
    /// Panics if the underlying tensor cannot be represented by a
    /// [`TensorInfo`](crate::tensor::TensorInfo).
    pub fn input_info(&self, n: usize) -> TensorInfo {
        let interpreter = &self.micro_interpreter;
        let input_tensor: &'a Tensor = unsafe {
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
        };

        // Panics if tensor cannot be repesented (eg. unimplemented type)
        input_tensor.info()
    }

    /// Clones data into the `n`th input tensor.
    ///
    /// # Errors
    ///
    /// Returns `Error::InputDataLenMismatch` if the length of slice `data`
    /// does not match the flat length of the `n`th input tensor.
    ///
    /// Returns an Error if the underlying tensor cannot be represented by a
    /// [`TensorInfo`](crate::tensor::TensorInfo).
    ///
    /// # Panics
    ///
    /// Panics if the attempt to get a pointer from TensorFlow returns a
    /// nullptr. This can occour if the tensor index `n` is invalid.
    pub fn input<T: ElemTypeOf + core::clone::Clone>(
        &mut self,
        n: usize,
        data: &[T],
    ) -> Result<(), Error> {
        let interpreter = &self.micro_interpreter;
        let input_tensor: &mut Tensor = unsafe {
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
        };

        // Returns Err if tensor cannot be repesented (eg. unimplemented
        // type)
        let tensor_info: TensorInfo = input_tensor.inner().try_into()?;

        // Length is the product of all dimensions
        let tensor_len = tensor_info.dims.iter().product::<i32>();

        if tensor_len != data.len().try_into().unwrap() {
            Err(Error::InputDataLenMismatch)
        } else {
            input_tensor.as_data_mut().clone_from_slice(data);
            Ok(())
        }
    }

    /// Runs the Tensorflow operation to transform input tensors to output
    /// tensors
    ///
    /// # Errors
    ///
    /// Returns a TensorFlow [`Status`](crate::Status) if an error occours in
    /// TensorFlow.
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
    /// # Panics
    ///
    /// Panics if the attempt to get a pointer from TensorFlow returns a
    /// nullptr. This can occour if the tensor index `n` is invalid.
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
    use crate::micro_op_resolver::AllOpResolver;
    use crate::tensor::ElementType;

    #[test]
    fn new_interpreter_static_arena() {
        // model
        let model = include_bytes!("../examples/models/hello_world.tflite");
        let model = Model::from_buffer(&model[..]).unwrap();

        // resolver
        let all_op_resolver = AllOpResolver::new();

        // arena
        const TENSOR_ARENA_SIZE: usize = 4 * 1024;
        let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];

        let _ = MicroInterpreter::new(
            &model,
            all_op_resolver,
            &mut tensor_arena[..],
        )
        .unwrap();
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
        let all_op_resolver = AllOpResolver::new();

        // arena
        let tensor_arena: Vec<u8> = vec![0u8; 4 * 1024];

        let _ = MicroInterpreter::new(&model, all_op_resolver, tensor_arena)
            .unwrap();
    }

    #[test]
    fn input_info() {
        // model
        let model = include_bytes!("../examples/models/hello_world.tflite");
        let model = Model::from_buffer(&model[..]).unwrap();

        // resolver
        let all_op_resolver = AllOpResolver::new();

        // arena
        const TENSOR_ARENA_SIZE: usize = 4 * 1024;
        let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];

        let interpreter = MicroInterpreter::new(
            &model,
            all_op_resolver,
            &mut tensor_arena[..],
        )
        .unwrap();

        let info = interpreter.input_info(0);

        // input tensor properties for hello_world example
        assert_eq!(info.name, "dense_2_input");
        assert_eq!(info.element_type, ElementType::Float32);
        assert_eq!(info.dims, [1, 1]);
    }
}
