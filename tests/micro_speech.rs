//! micro_speech example

use tfmicro::{
    micro_interpreter::MicroInterpreter, micro_op_resolver::MicroOpResolver,
    model::Model,
};

use log::info;

#[test]
fn micro_speech() {
    env_logger::init();
    info!("---- Starting tensorflow micro example: micro_speech");

    let model = include_bytes!("../examples/models/micro_speech.tflite");
    let no =
        include_bytes!("../examples/models/no_micro_f9643d42_nohash_4.data");
    let yes =
        include_bytes!("../examples/models/yes_micro_f2e59fea_nohash_1.data");

    // Map the model into a usable data structure. This doesn't involve
    // any copying or parsing, it's a very lightweight operation.
    let model = Model::from_buffer(&model[..]).unwrap();

    // Create an area of memory to use for input, output, and
    // intermediate arrays.
    const TENSOR_ARENA_SIZE: usize = 10 * 1024;
    let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];

    // Pull in all needed operation implementations
    let micro_op_resolver = MicroOpResolver::new_for_microspeech();

    // Build an interpreter to run the model with
    let mut interpreter =
        MicroInterpreter::new(&model, micro_op_resolver, &mut tensor_arena[..])
            .unwrap();

    // Check properties of the input sensor
    let input = interpreter.input(0);
    assert_eq!([1, 49, 40, 1], input.tensor_info().dims);

    // -------- 'yes' example --------

    input.tensor_data_mut().clone_from_slice(yes);

    interpreter.invoke().unwrap();

    // Get output for 'yes'
    let output = interpreter.output(0);
    assert_eq!([1, 4], output.tensor_info().dims);

    dbg!(output.tensor_data::<u8>());
    let silence_score: u8 = output.tensor_data()[0];
    let unknown_score: u8 = output.tensor_data()[1];
    let yes_score: u8 = output.tensor_data()[2];
    let no_score: u8 = output.tensor_data()[3];

    assert!(yes_score > silence_score);
    assert!(yes_score > unknown_score);
    assert!(yes_score > no_score);

    // -------- 'no' example --------

    input.tensor_data_mut().clone_from_slice(no);

    interpreter.invoke().unwrap();

    // Get output for 'no'
    let output = interpreter.output(0);
    assert_eq!([1, 4], output.tensor_info().dims);

    dbg!(output.tensor_data::<u8>());
    let silence_score: u8 = output.tensor_data()[0];
    let unknown_score: u8 = output.tensor_data()[1];
    let yes_score: u8 = output.tensor_data()[2];
    let no_score: u8 = output.tensor_data()[3];

    assert!(no_score > silence_score);
    assert!(no_score > unknown_score);
    assert!(no_score > yes_score);

    interpreter.arena_used_bytes();

    info!("---- Done");
}
