//! Tensorflow Lite Op Resolver
//!

use crate::bindings::tflite;

cpp! {{
    #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
    #include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
}}

#[repr(transparent)]
pub struct MicroOpResolver(tflite::MicroMutableOpResolver);

impl MicroOpResolver {
    /// Create a new MicroOpResolver, populated with all available
    /// operators (`AllOpsResolver`)
    pub fn new() -> Self {
        // The C++ compiler fills in the MicroMutableOpResolver with the
        // operators enumerated in AllOpsResolver
        let micro_op_resolver = unsafe {
            cpp!([] -> tflite::MicroMutableOpResolver as "tflite::MicroMutableOpResolver" {
                // All ops
                tflite::ops::micro::AllOpsResolver resolver;

                return resolver;
            })
        };

        Self(micro_op_resolver)
    }

    /// Create a new MicroOpResolver
    pub fn new_for_microspeech() -> Self {
        // Select only the operations needed for the micro_speech example.
        //
        // We still need to take the full memory footprint of
        // `MicroMutableOpResolver`, in order to be layout
        // compatible. However the unreferenced operations themselves will
        // be optimised away
        let micro_op_resolver = unsafe {
            cpp!([] -> tflite::MicroMutableOpResolver as
                 "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>" {

                // ops for microspeech
                tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX> resolver;
                resolver.AddBuiltin(
                    tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                    tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

                resolver.AddBuiltin(
                    tflite::BuiltinOperator_FULLY_CONNECTED,
                    tflite::ops::micro::Register_FULLY_CONNECTED());

                resolver.AddBuiltin(
                    tflite::BuiltinOperator_SOFTMAX,
                    tflite::ops::micro::Register_SOFTMAX());

                return resolver;
            })
        };

        Self(micro_op_resolver)
    }
}
