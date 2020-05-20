//! Extract information about a Tensor into a TensorInfo

use core::convert::TryInto;
use core::fmt;

use crate::bindings;

use super::ElementType;

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
            element_type: t.type_.try_into().unwrap(),
            dims: unsafe {
                let dims = &*t.dims;
                dims.data.as_slice(dims.size as usize)
            },
        }
    }
}
