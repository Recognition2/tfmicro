//! Tensorflow Lite Op Resolver
//!

use crate::bindings::tflite;

/// Operators for Tensorflow micro
///
/// See lite/micro/kernels/all_ops_resolver.cc
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Operators {
    FullyConnected,
    MaxPool2d,
    Softmax,
    Logistic,
    Svdf,
    Conv2d,
    Concatenation,
    DepthwiseConv2d,
    AveragePool2d,
    Abs,
    Sin,
    Cos,
    Log,
    Sqrt,
    Rsqrt,
    Square,
    Prelu,
    Floor,
    Maximum,
    Minimum,
    ArgMax,
    ArgMin,
    LogicalOr,
    LogicalAnd,
    LogicalNot,
    Reshape,
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Ceil,
    Round,
    StridedSlice,
    Pack,
    Pad,
    Padv2,
    Split,
    Unpack,
    Neg,
    Add,
    Mul,
    Sub,
    Quantize,
    Dequantize,
    Relu,
    Relu6,
    Mean,
    ResizeNearestNeighbor,
    L2Normalization,
    Tanh,
}

cpp! {{
    #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
    #include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
}}

#[repr(transparent)]
pub struct MicroOpResolver(tflite::MicroMutableOpResolver);

impl MicroOpResolver {
    /// Create a new MicroOpResolver, populated with all available
    /// operators (`AllOpsResolver`)
    pub fn new_with_all_ops() -> Self {
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

    /// Create a new MicroOpResolver, populated only with the operators
    /// specified.
    pub fn new(operators: &[Operators]) -> Self {
        let tflite_registrations_max = cpp!(unsafe [] -> usize as "size_t" {
            return TFLITE_REGISTRATIONS_MAX;
        });

        // Check the number of operators is OK
        assert!(operators.len() < tflite_registrations_max,
                "Tensorflow micro does not support more than {} operators. See TFLITE_REGISTRATIONS_MAX",
                tflite_registrations_max);

        let micro_op_resolver = unsafe {
            // Create resolver object
            //
            // We still need to take the full memory footprint of
            // `MicroMutableOpResolver`, in order to be layout
            // compatible. However the unreferenced operations themselves will
            // be optimised away
            let mut resolver = cpp!([] -> tflite::MicroMutableOpResolver as
                                    "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>" {
                tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX> resolver;
                return resolver;
            });

            // Add each builtin
            for op in operators {
                add_builtin!(op, resolver);
            }

            resolver
        };

        Self(micro_op_resolver)
    }

    pub fn new_for_magic_wand() -> Self {
        let micro_op_resolver = unsafe {
            cpp!([] -> tflite::MicroMutableOpResolver as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>" {

                tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX> resolver;  // NOLINT
                resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                    tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
                resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                    tflite::ops::micro::Register_MAX_POOL_2D());
                resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                    tflite::ops::micro::Register_CONV_2D());
                resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                    tflite::ops::micro::Register_FULLY_CONNECTED());
                resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                    tflite::ops::micro::Register_SOFTMAX());
                return resolver;

            })
        };
        Self(micro_op_resolver)
    }
}
