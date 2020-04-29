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
}
