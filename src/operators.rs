/// Operators for Tensorflow micro
///
/// See lite/micro/all_ops_resolver.cc
use crate::micro_op_resolver::MutableOpResolver;

impl MutableOpResolver {
    /// Use the FULLY_CONNECTED operator in this op resolver
    pub fn fully_connected(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddFullyConnected();
        });

        self
    }
    /// Use the MAX_POOL_2D operator in this op resolver
    pub fn max_pool_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddMaxPool2D();
        });

        self
    }
    /// Use the SOFTMAX operator in this op resolver
    pub fn softmax(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSoftmax();
        });

        self
    }
    /// Use the LOGISTIC operator in this op resolver
    pub fn logistic(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLogistic();
        });

        self
    }
    /// Use the SVDF operator in this op resolver
    pub fn svdf(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSvdf(); 
        });

        self
    }
    /// Use the CONV_2D operator in this op resolver
    pub fn conv_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddConv2D();
        });

        self
    }
    /// Use the CONCATENATION operator in this op resolver
    pub fn concatenation(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddConcatenation();
        });

        self
    }
    /// Use the DEPTHWISE_CONV_2D operator in this op resolver
    pub fn depthwise_conv_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddDepthwiseConv2D();
        });

        self
    }
    /// Use the AVERAGE_POOL_2D operator in this op resolver
    pub fn average_pool_2d(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddAveragePool2D();
        });

        self
    }
    /// Use the ABS operator in this op resolver
    pub fn abs(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddAbs();
        });

        self
    }
    /// Use the SIN operator in this op resolver
    pub fn sin(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSin();
        });

        self
    }
    /// Use the COS operator in this op resolver
    pub fn cos(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddCos();
        });

        self
    }
    /// Use the LOG operator in this op resolver
    pub fn log(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLog();
        });

        self
    }
    /// Use the SQRT operator in this op resolver
    pub fn sqrt(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSqrt();
        });

        self
    }
    /// Use the RSQRT operator in this op resolver
    pub fn rsqrt(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddRsqrt();
        });

        self
    }
    /// Use the SQUARE operator in this op resolver
    pub fn square(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSquare();
        });

        self
    }
    /// Use the PRELU operator in this op resolver
    pub fn prelu(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddPrelu();
        });

        self
    }
    /// Use the FLOOR operator in this op resolver
    pub fn floor(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddFloor();
        });

        self
    }
    /// Use the MAXIMUM operator in this op resolver
    pub fn maximum(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddMaximum();
        });

        self
    }
    /// Use the MINIMUM operator in this op resolver
    pub fn minimum(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddMinimum();
        });

        self
    }
    /// Use the ARG_MAX operator in this op resolver
    pub fn arg_max(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddArgMax();
        });

        self
    }
    /// Use the ARG_MIN operator in this op resolver
    pub fn arg_min(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddArgMin();
        });

        self
    }
    /// Use the LOGICAL_OR operator in this op resolver
    pub fn logical_or(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLogicalOr();
        });

        self
    }
    /// Use the LOGICAL_AND operator in this op resolver
    pub fn logical_and(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLogicalAnd();
        });

        self
    }
    /// Use the LOGICAL_NOT operator in this op resolver
    pub fn logical_not(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLogicalNot();
        });

        self
    }
    /// Use the RESHAPE operator in this op resolver
    pub fn reshape(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddReshape();
        });

        self
    }
    /// Use the EQUAL operator in this op resolver
    pub fn equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddEqual();
        });

        self
    }
    /// Use the NOT_EQUAL operator in this op resolver
    pub fn not_equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddNotEqual();
        });

        self
    }
    /// Use the GREATER operator in this op resolver
    pub fn greater(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddGreater();
        });

        self
    }
    /// Use the GREATER_EQUAL operator in this op resolver
    pub fn greater_equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddGreaterEqual();
        });

        self
    }
    /// Use the LESS operator in this op resolver
    pub fn less(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLess();
        });

        self
    }
    /// Use the LESS_EQUAL operator in this op resolver
    pub fn less_equal(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddLessEqual();
        });

        self
    }
    /// Use the CEIL operator in this op resolver
    pub fn ceil(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddCeil();
        });

        self
    }
    /// Use the ROUND operator in this op resolver
    pub fn round(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddRound();
        });

        self
    }
    /// Use the STRIDED_SLICE operator in this op resolver
    pub fn strided_slice(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddStridedSlice();
        });

        self
    }
    /// Use the PACK operator in this op resolver
    pub fn pack(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddPack();
        });

        self
    }
    /// Use the PAD operator in this op resolver
    pub fn pad(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddPad();
        });

        self
    }
    /// Use the PADV2 operator in this op resolver
    pub fn padv2(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddPadV2();
        });

        self
    }
    /// Use the SPLIT operator in this op resolver
    pub fn split(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSplit();
        });

        self
    }
    /// Use the UNPACK operator in this op resolver
    pub fn unpack(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddUnpack();
        });

        self
    }
    #[allow(clippy::should_implement_trait)]
    /// Use the NEG operator in this op resolver
    pub fn neg(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddNeg();
        });

        self
    }
    /// Use the ADD operator in this op resolver
    pub fn add(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddAdd();
        });

        self
    }
    /// Use the MUL operator in this op resolver
    pub fn mul(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddMul();
        });

        self
    }
    /// Use the SUB operator in this op resolver
    pub fn sub(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddSub();
        });

        self
    }
    /// Use the QUANTIZE operator in this op resolver
    pub fn quantize(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddQuantize();
        });

        self
    }
    /// Use the DEQUANTIZE operator in this op resolver
    pub fn dequantize(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddDequantize();
        });

        self
    }
    /// Use the RELU operator in this op resolver
    pub fn relu(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddRelu();
        });

        self
    }
    /// Use the RELU6 operator in this op resolver
    pub fn relu6(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddRelu6();
        });

        self
    }
    /// Use the MEAN operator in this op resolver
    pub fn mean(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddMean();
        });

        self
    }
    /// Use the RESIZE_NEAREST_NEIGHBOR operator in this op resolver
    pub fn resize_nearest_neighbor(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddResizeNearestNeighbor();
        });

        self
    }
    /// Use the L2_NORMALIZATION operator in this op resolver
    pub fn l2_normalization(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddL2Normalization();
        });

        self
    }
    /// Use the TANH operator in this op resolver
    pub fn tanh(mut self) -> Self {
        self.check_then_inc_len();
        let inner_ref = &mut self.inner;

        cpp!(unsafe [inner_ref as "tflite::MicroMutableOpResolver<128>*"] {
            inner_ref->AddTanh();
        });

        self
    }
}
