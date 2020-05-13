/// Operators for Tensorflow micro
///
/// See lite/micro/kernels/all_ops_resolver.cc
use crate::micro_op_resolver::MutableOpResolver;

impl MutableOpResolver {
    /// Use the FULLY_CONNECTED operator in this op resolver
    pub fn fully_connected(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_FULLY_CONNECTED,
                tflite::ops::micro::Register_FULLY_CONNECTED()
            );
        });

        self
    }
    /// Use the MAX_POOL_2D operator in this op resolver
    pub fn max_pool_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_MAX_POOL_2D,
                tflite::ops::micro::Register_MAX_POOL_2D()
            );
        });

        self
    }
    /// Use the SOFTMAX operator in this op resolver
    pub fn softmax(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SOFTMAX,
                tflite::ops::micro::Register_SOFTMAX()
            );
        });

        self
    }
    /// Use the LOGISTIC operator in this op resolver
    pub fn logistic(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LOGISTIC,
                tflite::ops::micro::Register_LOGISTIC()
            );
        });

        self
    }
    /// Use the SVDF operator in this op resolver
    pub fn svdf(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SVDF,
                tflite::ops::micro::Register_SVDF()
            );
        });

        self
    }
    /// Use the CONV_2D operator in this op resolver
    pub fn conv_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_CONV_2D,
                tflite::ops::micro::Register_CONV_2D()
            );
        });

        self
    }
    /// Use the CONCATENATION operator in this op resolver
    pub fn concatenation(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_CONCATENATION,
                tflite::ops::micro::Register_CONCATENATION()
            );
        });

        self
    }
    /// Use the DEPTHWISE_CONV_2D operator in this op resolver
    pub fn depthwise_conv_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                tflite::ops::micro::Register_DEPTHWISE_CONV_2D()
            );
        });

        self
    }
    /// Use the AVERAGE_POOL_2D operator in this op resolver
    pub fn average_pool_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_AVERAGE_POOL_2D,
                tflite::ops::micro::Register_AVERAGE_POOL_2D()
            );
        });

        self
    }
    /// Use the ABS operator in this op resolver
    pub fn abs(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_ABS,
                tflite::ops::micro::Register_ABS()
            );
        });

        self
    }
    /// Use the SIN operator in this op resolver
    pub fn sin(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SIN,
                tflite::ops::micro::Register_SIN()
            );
        });

        self
    }
    /// Use the COS operator in this op resolver
    pub fn cos(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_COS,
                tflite::ops::micro::Register_COS()
            );
        });

        self
    }
    /// Use the LOG operator in this op resolver
    pub fn log(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LOG,
                tflite::ops::micro::Register_LOG()
            );
        });

        self
    }
    /// Use the SQRT operator in this op resolver
    pub fn sqrt(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SQRT,
                tflite::ops::micro::Register_SQRT()
            );
        });

        self
    }
    /// Use the RSQRT operator in this op resolver
    pub fn rsqrt(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_RSQRT,
                tflite::ops::micro::Register_RSQRT()
            );
        });

        self
    }
    /// Use the SQUARE operator in this op resolver
    pub fn square(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SQUARE,
                tflite::ops::micro::Register_SQUARE()
            );
        });

        self
    }
    /// Use the PRELU operator in this op resolver
    pub fn prelu(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_PRELU,
                tflite::ops::micro::Register_PRELU()
            );
        });

        self
    }
    /// Use the FLOOR operator in this op resolver
    pub fn floor(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_FLOOR,
                tflite::ops::micro::Register_FLOOR()
            );
        });

        self
    }
    /// Use the MAXIMUM operator in this op resolver
    pub fn maximum(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_MAXIMUM,
                tflite::ops::micro::Register_MAXIMUM()
            );
        });

        self
    }
    /// Use the MINIMUM operator in this op resolver
    pub fn minimum(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_MINIMUM,
                tflite::ops::micro::Register_MINIMUM()
            );
        });

        self
    }
    /// Use the ARG_MAX operator in this op resolver
    pub fn arg_max(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_ARG_MAX,
                tflite::ops::micro::Register_ARG_MAX()
            );
        });

        self
    }
    /// Use the ARG_MIN operator in this op resolver
    pub fn arg_min(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_ARG_MIN,
                tflite::ops::micro::Register_ARG_MIN()
            );
        });

        self
    }
    /// Use the LOGICAL_OR operator in this op resolver
    pub fn logical_or(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LOGICAL_OR,
                tflite::ops::micro::Register_LOGICAL_OR()
            );
        });

        self
    }
    /// Use the LOGICAL_AND operator in this op resolver
    pub fn logical_and(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LOGICAL_AND,
                tflite::ops::micro::Register_LOGICAL_AND()
            );
        });

        self
    }
    /// Use the LOGICAL_NOT operator in this op resolver
    pub fn logical_not(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LOGICAL_NOT,
                tflite::ops::micro::Register_LOGICAL_NOT()
            );
        });

        self
    }
    /// Use the RESHAPE operator in this op resolver
    pub fn reshape(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_RESHAPE,
                tflite::ops::micro::Register_RESHAPE()
            );
        });

        self
    }
    /// Use the EQUAL operator in this op resolver
    pub fn equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_EQUAL,
                tflite::ops::micro::Register_EQUAL()
            );
        });

        self
    }
    /// Use the NOT_EQUAL operator in this op resolver
    pub fn not_equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_NOT_EQUAL,
                tflite::ops::micro::Register_NOT_EQUAL()
            );
        });

        self
    }
    /// Use the GREATER operator in this op resolver
    pub fn greater(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_GREATER,
                tflite::ops::micro::Register_GREATER()
            );
        });

        self
    }
    /// Use the GREATER_EQUAL operator in this op resolver
    pub fn greater_equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_GREATER_EQUAL,
                tflite::ops::micro::Register_GREATER_EQUAL()
            );
        });

        self
    }
    /// Use the LESS operator in this op resolver
    pub fn less(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LESS,
                tflite::ops::micro::Register_LESS()
            );
        });

        self
    }
    /// Use the LESS_EQUAL operator in this op resolver
    pub fn less_equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_LESS_EQUAL,
                tflite::ops::micro::Register_LESS_EQUAL()
            );
        });

        self
    }
    /// Use the CEIL operator in this op resolver
    pub fn ceil(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_CEIL,
                tflite::ops::micro::Register_CEIL()
            );
        });

        self
    }
    /// Use the ROUND operator in this op resolver
    pub fn round(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_ROUND,
                tflite::ops::micro::Register_ROUND()
            );
        });

        self
    }
    /// Use the STRIDED_SLICE operator in this op resolver
    pub fn strided_slice(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_STRIDED_SLICE,
                tflite::ops::micro::Register_STRIDED_SLICE()
            );
        });

        self
    }
    /// Use the PACK operator in this op resolver
    pub fn pack(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_PACK,
                tflite::ops::micro::Register_PACK()
            );
        });

        self
    }
    /// Use the PAD operator in this op resolver
    pub fn pad(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_PAD,
                tflite::ops::micro::Register_PAD()
            );
        });

        self
    }
    /// Use the PADV2 operator in this op resolver
    pub fn padv2(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_PADV2,
                tflite::ops::micro::Register_PADV2()
            );
        });

        self
    }
    /// Use the SPLIT operator in this op resolver
    pub fn split(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SPLIT,
                tflite::ops::micro::Register_SPLIT()
            );
        });

        self
    }
    /// Use the UNPACK operator in this op resolver
    pub fn unpack(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_UNPACK,
                tflite::ops::micro::Register_UNPACK()
            );
        });

        self
    }
    /// Use the NEG operator in this op resolver
    pub fn neg(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_NEG,
                tflite::ops::micro::Register_NEG()
            );
        });

        self
    }
    /// Use the ADD operator in this op resolver
    pub fn add(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_ADD,
                tflite::ops::micro::Register_ADD()
            );
        });

        self
    }
    /// Use the MUL operator in this op resolver
    pub fn mul(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_MUL,
                tflite::ops::micro::Register_MUL()
            );
        });

        self
    }
    /// Use the SUB operator in this op resolver
    pub fn sub(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_SUB,
                tflite::ops::micro::Register_SUB()
            );
        });

        self
    }
    /// Use the QUANTIZE operator in this op resolver
    pub fn quantize(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_QUANTIZE,
                tflite::ops::micro::Register_QUANTIZE()
            );
        });

        self
    }
    /// Use the DEQUANTIZE operator in this op resolver
    pub fn dequantize(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_DEQUANTIZE,
                tflite::ops::micro::Register_DEQUANTIZE()
            );
        });

        self
    }
    /// Use the RELU operator in this op resolver
    pub fn relu(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_RELU,
                tflite::ops::micro::Register_RELU()
            );
        });

        self
    }
    /// Use the RELU6 operator in this op resolver
    pub fn relu6(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_RELU6,
                tflite::ops::micro::Register_RELU6()
            );
        });

        self
    }
    /// Use the MEAN operator in this op resolver
    pub fn mean(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_MEAN,
                tflite::ops::micro::Register_MEAN()
            );
        });

        self
    }
    /// Use the RESIZE_NEAREST_NEIGHBOR operator in this op resolver
    pub fn resize_nearest_neighbor(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                tflite::ops::micro::Register_RESIZE_NEAREST_NEIGHBOR()
            );
        });

        self
    }
    /// Use the L2_NORMALIZATION operator in this op resolver
    pub fn l2_normalization(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_L2_NORMALIZATION,
                tflite::ops::micro::Register_L2_NORMALIZATION()
            );
        });

        self
    }
    /// Use the TANH operator in this op resolver
    pub fn tanh(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroOpResolver<TFLITE_REGISTRATIONS_MAX>*"] {
            inner_ref->AddBuiltin(
                tflite::BuiltinOperator_TANH,
                tflite::ops::micro::Register_TANH()
            );
        });

        self
    }
}
