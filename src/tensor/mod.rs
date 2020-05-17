//! Rust Bindings for Tensor type

use core::convert::{TryFrom, TryInto};
use core::mem::size_of;
use core::slice;
use cty::c_int;
use ordered_float::NotNan;

use crate::bindings;

mod info;
pub use info::TensorInfo;

pub type TensorIndex = c_int;

/// A TensorFlow Tensor
#[repr(transparent)]
#[derive(Default)]
pub struct Tensor(bindings::TfLiteTensor);

/// Tensor element data types supported
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ElementType {
    Float32,
    UInt8,
    Int32,
}
impl TryFrom<bindings::TfLiteType> for ElementType {
    type Error = bindings::TfLiteType;

    fn try_from(status: bindings::TfLiteType) -> Result<Self, Self::Error> {
        use ElementType::*;

        match status {
            bindings::TfLiteType::kTfLiteFloat32 => Ok(Float32),
            bindings::TfLiteType::kTfLiteUInt8 => Ok(UInt8),
            bindings::TfLiteType::kTfLiteInt32 => Ok(Int32),
            t => Err(t),
        }
    }
}

/// Marker trait for those intristic types we support
pub trait ElemTypeOf {
    fn elem_type_of() -> ElementType;
}

impl ElemTypeOf for NotNan<f32> {
    fn elem_type_of() -> ElementType {
        ElementType::Float32
    }
}
impl ElemTypeOf for f32 {
    fn elem_type_of() -> ElementType {
        ElementType::Float32
    }
}
impl ElemTypeOf for u8 {
    fn elem_type_of() -> ElementType {
        ElementType::UInt8
    }
}
impl ElemTypeOf for i32 {
    fn elem_type_of() -> ElementType {
        ElementType::Int32
    }
}

/// From implementations
impl From<*mut bindings::TfLiteTensor> for &Tensor {
    fn from(t: *mut bindings::TfLiteTensor) -> Self {
        unsafe { &*(t as *mut Tensor) }
    }
}
impl From<*mut bindings::TfLiteTensor> for &mut Tensor {
    fn from(t: *mut bindings::TfLiteTensor) -> Self {
        unsafe { &mut *(t as *mut Tensor) }
    }
}

impl Tensor {
    /// Return the element type of this tensor
    pub fn get_type(&self) -> Option<ElementType> {
        self.0.type_.try_into().ok()
    }

    fn inner(&self) -> &bindings::TfLiteTensor {
        &self.0
    }

    /// Return a TensorInfo that lives as long as this Tensor
    pub fn tensor_info<'a>(&'a self) -> TensorInfo<'a> {
        self.inner().into()
    }

    pub fn tensor_data<T>(&self) -> &[T]
    where
        T: ElemTypeOf,
    {
        let tensor_info: TensorInfo = self.inner().into();

        assert!(
            tensor_info.element_type == T::elem_type_of(),
            "Invalid type reference of `{:?}` to the original type `{:?}`",
            T::elem_type_of(),
            tensor_info.element_type
        );

        unsafe {
            slice::from_raw_parts(
                self.0.data.raw_const as *const T,
                self.0.bytes / size_of::<T>(),
            )
        }
    }

    pub fn tensor_data_mut<T>(&mut self) -> &mut [T]
    where
        T: ElemTypeOf,
    {
        let tensor_info: TensorInfo = self.inner().into();
        assert!(
            tensor_info.element_type == T::elem_type_of(),
            "Invalid type reference of `{:?}` to the original type `{:?}`",
            T::elem_type_of(),
            tensor_info.element_type
        );
        unsafe {
            slice::from_raw_parts_mut(
                self.0.data.raw as *mut T,
                self.0.bytes / size_of::<T>(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}
