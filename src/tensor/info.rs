//! Extract information about a Tensor into a TensorInfo

use core::fmt;

use crate::bindings;

pub type ElementType = bindings::TfLiteType;
pub type QuantizationParams = bindings::TfLiteQuantizationParams;

pub trait ElemTypeOf {
    fn elem_type_of() -> ElementType;
}

impl ElemTypeOf for f32 {
    fn elem_type_of() -> ElementType {
        bindings::TfLiteType::kTfLiteFloat32
    }
}

impl ElemTypeOf for u8 {
    fn elem_type_of() -> ElementType {
        bindings::TfLiteType::kTfLiteUInt8
    }
}

impl ElemTypeOf for i32 {
    fn elem_type_of() -> ElementType {
        bindings::TfLiteType::kTfLiteInt32
    }
}

pub struct TensorInfo<'a> {
    //pub name: &'a str,
    pub element_type: ElementType,
    pub dims: &'a [i32],
}

impl fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorInfo")
            //.field("name", &self.name)
            .field("element_type", &self.element_type)
            .field("dims", &self.dims)
            .finish()
    }
}

impl<'a> From<&'a bindings::TfLiteTensor> for TensorInfo<'a> {
    fn from(t: &'a bindings::TfLiteTensor) -> Self {
        Self {
            element_type: t.type_,
            dims: {
                let slice = unsafe {
                    let dims = &*t.dims;
                    dims.data.as_slice(dims.size as usize)
                };
                slice
            },
        }
    }
}
