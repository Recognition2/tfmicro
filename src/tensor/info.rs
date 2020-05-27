//! Extract information about a Tensor into a TensorInfo

use core::convert::{TryFrom, TryInto};
use core::fmt;
use core::slice;
use core::str;

use crate::bindings;
use crate::interop;
use crate::Error;

use super::ElementType;

/// Metadata describing a tensor
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub element_type: ElementType,
    pub dims: &'a [i32],
}

impl fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorInfo")
            .field("name", &self.name)
            .field("element_type", &self.element_type)
            .field("dims", &self.dims)
            .finish()
    }
}

impl<'a> TryFrom<&'a bindings::TfLiteTensor> for TensorInfo<'a> {
    type Error = Error;

    fn try_from(t: &'a bindings::TfLiteTensor) -> Result<Self, Self::Error> {
        let name_as_slice = unsafe {
            let len = interop::strlen::strlen(t.name);
            let ptr = t.name as *const u8;
            slice::from_raw_parts(ptr, len as usize)
        };

        // Parse name as utf8
        let name = str::from_utf8(name_as_slice).or(Err(Error::Utf8Error))?;

        // Attempt to match type_ as a ElementType
        let element_type = t
            .type_
            .try_into()
            .or(Err(Error::ElementTypeUnimplemented))?;

        Ok(Self {
            name,
            element_type,
            dims: unsafe {
                let dims = &*t.dims;
                dims.data.as_slice(dims.size as usize)
            },
        })
    }
}
