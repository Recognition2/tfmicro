//!
//! Tensorflow + `no_std` + Rust = ❤️
//!
//! The crate contains Rust bindings for the [TensorFlow Micro][]
//! project. TensorFlow Micro is a version of TensorFlow Lite designed to
//! run without a standard library, for use on microcontrollers, wasm and
//! more.
//!
//! Thanks to [Cargo][] and the [CC crate][], there's no porting required for
//! new platforms - just drop `tfmicro` into your `Cargo.toml` and
//! go. You will need a C++ compiler behind the scenes, including for
//! cross-compiling targets, but in most cases this will be present
//! already.
//!
//! # Getting started
//!
//! Add `tfmicro` in the dependencies section of your `Cargo.toml`
//!
//! ```text
//! [dependencies]
//! tfmicro = 0.1.0
//! ```
//!
//! To understand how the [TensorFlow Micro C examples][c_examples] map to
//! idiomatic Rust code, see the [Examples](#Examples) section. Otherwise
//! for a more general description see [Usage](#Usage).
//!
//! # Examples
//!
//!
//!
//! # Usage
//!
//! # Developing
//!
//! See [DEVELOP.md](DEVELOP.md)
//!
//! # License
//!
//! [rust-embedded]: https://www.rust-lang.org/what/embedded
//! [TensorFlow Micro]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro
//! [Cargo]: https://doc.rust-lang.org/stable/cargo/
//! [CC crate]: https://crates.io/crates/cc
//! [c_examples]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples
#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate cpp;

mod bindings;
mod interop;

/// Error type for tfmicro
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Error {
    InvalidModel,
}

/// The status resulting from a TensorFlow operation
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Ok,
    Error,
    DelegateError,
}
impl From<bindings::TfLiteStatus> for Status {
    fn from(status: bindings::TfLiteStatus) -> Self {
        use Status::*;

        match status {
            bindings::TfLiteStatus::kTfLiteOk => Ok,
            bindings::TfLiteStatus::kTfLiteError => Error,
            bindings::TfLiteStatus::kTfLiteDelegateError => DelegateError,
        }
    }
}

mod micro_error_reporter;

pub mod micro_interpreter;
pub mod micro_op_resolver;
pub mod model;
pub mod tensor;

pub use micro_interpreter::MicroInterpreter;
pub use micro_op_resolver::MicroOpResolver;
