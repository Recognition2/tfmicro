#![recursion_limit = "8192"]
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
