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

/// Implement From raw types to Tensor
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
    /// The element type of this tensor.
    ///
    /// Returns `Some(element_type)` if the element type annotated on this
    /// tensor matches a member of
    /// [`ElementType`](crate::tensor::ElementType). Otherwise returns `None`.
    pub fn element_type(&self) -> Option<ElementType> {
        self.0.type_.try_into().ok()
    }

    /// A [`TensorInfo`](crate::tensor::TensorInfo) that describes this tensor
    ///
    /// # Panics
    ///
    /// Panics if the underlying tensor cannot be represented by a TensorInfo.
    pub fn info(&self) -> TensorInfo {
        self.inner().try_into().unwrap()
    }

    fn inner(&self) -> &bindings::TfLiteTensor {
        &self.0
    }

    /// Extracts the tensor's data as a flat slice.
    ///
    /// Call the [info](#method.info) method to check the dimensionality of
    /// the tensor.
    ///
    /// # Panics
    ///
    /// This method will panic if `T` does not match the data type
    /// annotated on this tensor. Call
    /// [element_type()](#method.element_type) to discover the data type.
    pub fn as_data<T>(&self) -> &[T]
    where
        T: ElemTypeOf,
    {
        assert!(
            self.element_type().unwrap() == T::elem_type_of(),
            "Type `{:?}` does not match the original type `{:?}`",
            T::elem_type_of(),
            self.0.type_
        );

        unsafe {
            slice::from_raw_parts(
                self.0.data.raw_const as *const T,
                self.0.bytes / size_of::<T>(),
            )
        }
    }

    /// Extracts the tensor's data as a mutable flat slice.
    ///
    /// Call the [info](#method.info) method to check the dimensionality of
    /// the tensor.
    ///
    /// # Panics
    ///
    /// This method will panic if `T` does not match the data type
    /// annotated on this tensor. Call
    /// [element_type()](#method.element_type) to discover the data type.
    pub fn as_data_mut<T>(&mut self) -> &mut [T]
    where
        T: ElemTypeOf,
    {
        assert!(
            self.element_type().unwrap() == T::elem_type_of(),
            "Type `{:?}` does not match the original type `{:?}`",
            T::elem_type_of(),
            self.0.type_
        );

        unsafe {
            slice::from_raw_parts_mut(
                self.0.data.raw as *mut T,
                self.0.bytes / size_of::<T>(),
            )
        }
    }
}
