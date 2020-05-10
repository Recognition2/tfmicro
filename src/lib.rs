//! tfmicro
//! ======
//!
//! Tensorflow + `no_std` + Rust = ❤️
//!
//! The crate contains Rust bindings for the [TensorFlow Micro][] project. TensorFlow Micro is
//! a version of TensorFlow Lite designed to run without a standard
//! library, for use on microcontrollers, wasm and more.
//!
//! Thanks to [Cargo][] and the [CC crate][], there's no porting required for
//! new platforms - just drop `tfmicro = 0.1.0` into your `Cargo.toml` and
//! go. You will need a C++ compiler behind the scenes, including for
//! cross-compiling targets, but in most cases this will be present
//! already.
//!
//! # Getting started
//!
//! # Examples
//!
//! # Usage
//!
//!
//! [rust-embedded]: https://www.rust-lang.org/what/embedded
//! [TensorFlow Micro]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro
//! [Cargo]: https://doc.rust-lang.org/stable/cargo/
//! [CC crate]: https://crates.io/crates/cc
#![no_std]

#[macro_use]
extern crate log;
#[macro_use]
extern crate cpp;

pub mod bindings;
mod interop;
pub mod interpreter;
pub mod micro_error_reporter;
pub mod micro_interpreter;
pub mod micro_op_resolver;
pub mod model;

use interpreter::Tensor;
pub use micro_error_reporter::MicroErrorReporter;
pub use micro_interpreter::MicroInterpreter;
pub use micro_op_resolver::MicroOpResolver;
