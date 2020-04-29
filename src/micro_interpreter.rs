use crate::error_reporter::ErrorReporter;
use crate::micro_op_resolver::MicroOpResolver;
use crate::model::Model;

pub struct MicroInterpreter {
    //
}

impl MicroInterpreter {
    pub fn new<'a>(
        model: &Model,
        micro_op_resolver: &MicroOpResolver,
        tensor_arena: &'a [u8],
        tensor_arena_size: usize,
        error_reporter: &ErrorReporter,
    ) -> Self {
        Self {}
    }
}
