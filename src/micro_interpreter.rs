//! Micro interpreter

use core::marker::PhantomData;

use crate::interpreter::Tensor;
use crate::micro_error_reporter::MicroErrorReporter;
use crate::micro_op_resolver::MicroOpResolver;
use crate::model::Model;

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

pub struct MicroInterpreter<'a> {
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

    pub fn new<'m: 'a, 't: 'a, 'e: 'a>(
        model: &'m Model,
        resolver: MicroOpResolver,
        tensor_arena: &'t mut [u8],
        tensor_arena_size: usize,
        micro_error_reporter: &'e MicroErrorReporter,
    ) -> Self {
        let tensor_arena = tensor_arena.as_ptr();

        // Create interpreter
        let micro_interpreter = unsafe {
            cpp! ([
                model as "const tflite::Model*",
                resolver as "tflite::MicroMutableOpResolver",
                tensor_arena as "uint8_t*",
                tensor_arena_size as "size_t",
                micro_error_reporter as "tflite::MicroErrorReporter*"
            ] -> tflite::MicroInterpreter as "tflite::MicroInterpreter"
              {
                  tflite::ErrorReporter* error_reporter = micro_error_reporter;
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

    pub fn input(&self, n: usize) -> &'a Tensor {
        let micro_interpreter = &self.micro_interpreter;
        unsafe {
            let inp = cpp!([micro_interpreter as "tflite::MicroInterpreter*",
                n as "size_t"] -> *mut bindings::TfLiteTensor as "TfLiteTensor*" {

                return micro_interpreter->input(n);
            });
            inp.into()
        }
    }
}
