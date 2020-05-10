//! TensorFlow model

use core::ops::Deref;

use crate::bindings::tflite;
use crate::Error;

#[repr(transparent)]
#[derive(Default)]
pub struct Model(tflite::Model);

impl Deref for Model {
    type Target = tflite::Model;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Model {
    /// Create a tensorflow model that lives as long as the underlying buffer
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidModel` if the buffer failed verification
    pub fn from_buffer<'a>(buffer: &'a [u8]) -> Result<&'a Self, Error> {
        let len = buffer.len();
        let buffer = buffer.as_ptr();

        let model = unsafe {
            cpp!([buffer as "const void*", len as "size_t"]
                  -> *const tflite::Model as "const tflite::Model*" {

                auto verifier = flatbuffers::Verifier((const uint8_t *)buffer, len);
                if (!::tflite::VerifyModelBuffer(verifier)) {
                    return NULL;
                }

                return ::tflite::GetModel(buffer);
            })
        };

        if !model.is_null() {
            Ok(unsafe { &*(model as *const Self) })
        } else {
            Err(Error::InvalidModel)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_from_buffer() {
        let model = include_bytes!("../examples/models/hello_world.tflite");

        // Instantiate the model
        let _ = Model::from_buffer(&model[..]).unwrap();
    }

    #[test]
    #[should_panic]
    fn bad_model_from_buffer() {
        let model =
            include_bytes!("../examples/models/hello_world.tflite").clone();

        let _ = Model::from_buffer(&model[..88]).unwrap();
        //                                  ^^
    }
}
