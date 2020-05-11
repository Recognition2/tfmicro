//! Builtin ops

macro_rules! add_builtin {
    ($op:ident, $resolver:ident) => {
        let resolver_ref: &mut tflite::MicroMutableOpResolver = &mut $resolver;
        use Operators::*;

        match $op {
            FullyConnected => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_FULLY_CONNECTED,
                        tflite::ops::micro::Register_FULLY_CONNECTED()
                    );
                })
            },
            MaxPool2d => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_MAX_POOL_2D,
                        tflite::ops::micro::Register_MAX_POOL_2D()
                    );
                })
            },
            Softmax => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SOFTMAX,
                        tflite::ops::micro::Register_SOFTMAX()
                    );
                })
            },
            Logistic => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LOGISTIC,
                        tflite::ops::micro::Register_LOGISTIC()
                    );
                })
            },
            Svdf => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SVDF,
                        tflite::ops::micro::Register_SVDF()
                    );
                })
            },
            Conv2d => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_CONV_2D,
                        tflite::ops::micro::Register_CONV_2D()
                    );
                })
            },
            Concatenation => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_CONCATENATION,
                        tflite::ops::micro::Register_CONCATENATION()
                    );
                })
            },
            DepthwiseConv2d => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                        tflite::ops::micro::Register_DEPTHWISE_CONV_2D()
                    );
                })
            },
            AveragePool2d => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_AVERAGE_POOL_2D,
                        tflite::ops::micro::Register_AVERAGE_POOL_2D()
                    );
                })
            },
            Abs => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_ABS,
                        tflite::ops::micro::Register_ABS()
                    );
                })
            },
            Sin => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SIN,
                        tflite::ops::micro::Register_SIN()
                    );
                })
            },
            Cos => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_COS,
                        tflite::ops::micro::Register_COS()
                    );
                })
            },
            Log => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LOG,
                        tflite::ops::micro::Register_LOG()
                    );
                })
            },
            Sqrt => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SQRT,
                        tflite::ops::micro::Register_SQRT()
                    );
                })
            },
            Rsqrt => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_RSQRT,
                        tflite::ops::micro::Register_RSQRT()
                    );
                })
            },
            Square => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SQUARE,
                        tflite::ops::micro::Register_SQUARE()
                    );
                })
            },
            Prelu => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_PRELU,
                        tflite::ops::micro::Register_PRELU()
                    );
                })
            },
            Floor => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_FLOOR,
                        tflite::ops::micro::Register_FLOOR()
                    );
                })
            },
            Maximum => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_MAXIMUM,
                        tflite::ops::micro::Register_MAXIMUM()
                    );
                })
            },
            Minimum => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_MINIMUM,
                        tflite::ops::micro::Register_MINIMUM()
                    );
                })
            },
            ArgMax => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_ARG_MAX,
                        tflite::ops::micro::Register_ARG_MAX()
                    );
                })
            },
            ArgMin => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_ARG_MIN,
                        tflite::ops::micro::Register_ARG_MIN()
                    );
                })
            },
            LogicalOr => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LOGICAL_OR,
                        tflite::ops::micro::Register_LOGICAL_OR()
                    );
                })
            },
            LogicalAnd => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LOGICAL_AND,
                        tflite::ops::micro::Register_LOGICAL_AND()
                    );
                })
            },
            LogicalNot => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LOGICAL_NOT,
                        tflite::ops::micro::Register_LOGICAL_NOT()
                    );
                })
            },
            Reshape => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_RESHAPE,
                        tflite::ops::micro::Register_RESHAPE()
                    );
                })
            },
            Equal => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_EQUAL,
                        tflite::ops::micro::Register_EQUAL()
                    );
                })
            },
            NotEqual => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_NOT_EQUAL,
                        tflite::ops::micro::Register_NOT_EQUAL()
                    );
                })
            },
            Greater => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_GREATER,
                        tflite::ops::micro::Register_GREATER()
                    );
                })
            },
            GreaterEqual => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_GREATER_EQUAL,
                        tflite::ops::micro::Register_GREATER_EQUAL()
                    );
                })
            },
            Less => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LESS,
                        tflite::ops::micro::Register_LESS()
                    );
                })
            },
            LessEqual => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_LESS_EQUAL,
                        tflite::ops::micro::Register_LESS_EQUAL()
                    );
                })
            },
            Ceil => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_CEIL,
                        tflite::ops::micro::Register_CEIL()
                    );
                })
            },
            Round => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_ROUND,
                        tflite::ops::micro::Register_ROUND()
                    );
                })
            },
            StridedSlice => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_STRIDED_SLICE,
                        tflite::ops::micro::Register_STRIDED_SLICE()
                    );
                })
            },
            Pack => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_PACK,
                        tflite::ops::micro::Register_PACK()
                    );
                })
            },
            Pad => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_PAD,
                        tflite::ops::micro::Register_PAD()
                    );
                })
            },
            Padv2 => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_PADV2,
                        tflite::ops::micro::Register_PADV2()
                    );
                })
            },
            Split => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SPLIT,
                        tflite::ops::micro::Register_SPLIT()
                    );
                })
            },
            Unpack => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_UNPACK,
                        tflite::ops::micro::Register_UNPACK()
                    );
                })
            },
            Neg => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_NEG,
                        tflite::ops::micro::Register_NEG()
                    );
                })
            },
            Add => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_ADD,
                        tflite::ops::micro::Register_ADD()
                    );
                })
            },
            Mul => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_MUL,
                        tflite::ops::micro::Register_MUL()
                    );
                })
            },
            Sub => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_SUB,
                        tflite::ops::micro::Register_SUB()
                    );
                })
            },
            Quantize => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_QUANTIZE,
                        tflite::ops::micro::Register_QUANTIZE()
                    );
                })
            },
            Dequantize => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_DEQUANTIZE,
                        tflite::ops::micro::Register_DEQUANTIZE()
                    );
                })
            },
            Relu => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_RELU,
                        tflite::ops::micro::Register_RELU()
                    );
                })
            },
            Relu6 => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_RELU6,
                        tflite::ops::micro::Register_RELU6()
                    );
                })
            },
            Mean => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_MEAN,
                        tflite::ops::micro::Register_MEAN()
                    );
                })
            },
            ResizeNearestNeighbor => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                        tflite::ops::micro::Register_RESIZE_NEAREST_NEIGHBOR()
                    );
                })
            },
            L2Normalization => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_L2_NORMALIZATION,
                        tflite::ops::micro::Register_L2_NORMALIZATION()
                    );
                })
            },
            Tanh => {
                cpp!([resolver_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
                    resolver_ref->AddBuiltin(
                        tflite::BuiltinOperator_TANH,
                        tflite::ops::micro::Register_TANH()
                    );
                })
            },
        }
    };
}
