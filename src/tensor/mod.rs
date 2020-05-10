//! Rust Bindings for Tensor type

use core::mem::size_of;
use core::slice;
use cty::c_int;

use crate::bindings;

mod info;
use info::{ElemTypeOf, TensorInfo};

pub type TensorIndex = c_int;

#[repr(transparent)]
#[derive(Default)]
pub struct Tensor(bindings::TfLiteTensor);

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
    pub fn get_type(&self) -> &bindings::TfLiteType {
        &self.0.type_
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
