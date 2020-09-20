//! Tensorflow Lite Op Resolvers
//!

use crate::bindings::tflite;

use core::fmt;

cpp! {{
    #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
    #include "tensorflow/lite/micro/all_ops_resolver.h"
}}

// AllOpsResolver has the same memory representation as
// MicroMutableOpResolver<128>.
//
// That is:
// class AllOpsResolver : public MicroMutableOpResolver<128> { ... }
//
// Thus we can cast between the two types.

type OpResolverT = tflite::AllOpsResolver;

/// Marker trait for types that have the memory representation of a
/// `OpResolver`
pub trait OpResolverRepr {
    fn to_inner(self) -> OpResolverT;
}

/// An Op Resolver populated with all available operators
#[derive(Default)]
pub struct AllOpResolver(OpResolverT);
impl OpResolverRepr for AllOpResolver {
    fn to_inner(self) -> OpResolverT {
        self.0
    }
}

/// An Op Resolver that has no operators by default, but can be added by
/// calling methods in a builder pattern
#[derive(Default)]
pub struct MutableOpResolver {
    pub(crate) inner: OpResolverT,
    capacity: usize,
    len: usize,
}
impl OpResolverRepr for MutableOpResolver {
    fn to_inner(self) -> OpResolverT {
        self.inner
    }
}
impl fmt::Debug for MutableOpResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("MutableOpResolver (ops = {})", self.len))
    }
}

impl AllOpResolver {
    /// Create a new Op Resolver, populated with all available
    /// operators
    pub fn new() -> Self {
        // The C++ compiler fills in the MicroMutableOpResolver with the
        // operators enumerated in AllOpsResolver
        let micro_op_resolver = unsafe {
            cpp!([] -> OpResolverT as "tflite::AllOpsResolver" {
                // All ops
                tflite::AllOpsResolver resolver;

                return resolver;
            })
        };

        Self(micro_op_resolver)
    }
}

impl MutableOpResolver {
    /// Check the number of operators is OK
    pub(crate) fn check_then_inc_len(&mut self) {
        assert!(
            self.len < self.capacity,
            "Tensorflow micro does not support more than {} operators.",
            self.capacity
        );

        self.len += 1;
    }

    /// Returns the current number of operators in this resolver
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether there are zero operators
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Create a new MutableOpResolver, initially empty
    pub fn empty() -> Self {
        // Maximum number of registrations
        //
        // tensorflow/lite/micro/kernels/all_ops_resolver.h:L27
        let tflite_registrations_max = 128;

        let micro_op_resolver = unsafe {
            // Create resolver object
            //
            // We still need to take the full memory footprint of
            // `MicroMutableOpResolver`, in order to be layout
            // compatible. However the unreferenced operations themselves will
            // be optimised away
            cpp!([] -> OpResolverT as
                 "tflite::MicroMutableOpResolver<128>" {

                tflite::MicroMutableOpResolver<128> resolver;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ops_resolver() {
        let _ = AllOpResolver::new();
    }

    #[test]
    fn mutable_op_resolver() {
        let _ = MutableOpResolver::empty()
            .depthwise_conv_2d()
            .fully_connected()
            .softmax();
    }
}
