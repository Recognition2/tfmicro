//! TensorFlow model

use core::ops::Deref;

use crate::bindings::tflite;

#[repr(transparent)]
#[derive(Default)]
pub struct Model(tflite::Model);

// impl Clone for Model {
//     fn clone(&self) -> Self {
//         Self::from_buffer(&self.to_buffer()).unwrap()
//     }
// }

impl Deref for Model {
    type Target = tflite::Model;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Model {
    /// Create a tensorflow model that lives as long as the underlying buffer
    ///
    pub fn from_buffer<'a>(buffer: &'a [u8]) -> Option<&'a Self> {
        let len = buffer.len();
        let buffer = buffer.as_ptr();

        let r = unsafe {
            cpp!([buffer as "const void*", len as "size_t"]
                  -> *const tflite::Model as "const tflite::Model*" {
                (void) len;
                // auto verifier = flatbuffers::Verifier((const uint8_t *)buffer, len);
                // if (!VerifyModelBuffer(verifier)) {
                //     return false;
                // }

                return ::tflite::GetModel(buffer);
            })
        };

        Some(unsafe { &*(r as *const Self) })
    }
}
