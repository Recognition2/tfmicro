//! Tensorflow Lite Op Resolver
//!

use crate::bindings::tflite;

cpp! {{
    #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
    #include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
}}

/// Marker trait for types that have the memory representation of a
/// `tflite::MicroMutableOpResolver`
pub trait MicroMutableOpResolver {
    fn to_inner(self) -> tflite::MicroMutableOpResolver;
}

/// An Op Resolver populated with all available operators
#[derive(Default)]
pub struct AllOpResolver(tflite::MicroMutableOpResolver);
impl MicroMutableOpResolver for AllOpResolver {
    fn to_inner(self) -> tflite::MicroMutableOpResolver {
        self.0
    }
}

/// An Op Resolver that has no operators by default, but can be added by
/// calling methods in a builder pattern
#[derive(Default)]
pub struct MutableOpResolver {
    pub(crate) inner: tflite::MicroMutableOpResolver,
    capacity: usize,
    len: usize,
}
impl MicroMutableOpResolver for MutableOpResolver {
    fn to_inner(self) -> tflite::MicroMutableOpResolver {
        self.inner
    }
}

impl AllOpResolver {
    /// Create a new Op Resolver, populated with all available
    /// operators
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

impl MutableOpResolver {
    /// Check the number of operators is OK
    pub(crate) fn check_then_inc_len(&mut self) {
        assert!(self.len < self.capacity,
              "Tensorflow micro does not support more than {} operators. See TFLITE_REGISTRATIONS_MAX",
                self.capacity);

        self.len += 1;
    }

    /// Returns the current number of operators in this resolver
    pub fn len(&self) -> usize {
        self.len
    }

    /// Create a new MutableOpResolver, initially empty
    pub fn empty() -> Self {
        // Get the maximum number of registrations
        let tflite_registrations_max = cpp!(unsafe [] -> usize as "size_t" {
            return TFLITE_REGISTRATIONS_MAX;
        });

        let micro_op_resolver = unsafe {
            // Create resolver object
            //
            // We still need to take the full memory footprint of
            // `MicroMutableOpResolver`, in order to be layout
            // compatible. However the unreferenced operations themselves will
            // be optimised away
            cpp!([] -> tflite::MicroMutableOpResolver as
                 "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>" {

                tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX> resolver;
                return resolver;
            })
        };

        Self {
            inner: micro_op_resolver,
            capacity: tflite_registrations_max,
            len: 0,
        }
    }
}
