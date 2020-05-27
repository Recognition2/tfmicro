//! micro_speech example

use tfmicro::{MicroInterpreter, Model, MutableOpResolver};

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
    let micro_op_resolver = MutableOpResolver::empty()
        .depthwise_conv_2d()
        .fully_connected()
        .softmax();

    info!("Resolver: {:?}", micro_op_resolver);

    // Build an interpreter to run the model with
    let mut interpreter =
        MicroInterpreter::new(&model, micro_op_resolver, &mut tensor_arena[..])
            .unwrap();

    // Check properties of the input sensor
    assert_eq!([1, 49, 40, 1], interpreter.input_info(0).dims);

    // -------- 'yes' example --------
    interpreter.input(0, yes).unwrap();
    interpreter.invoke().unwrap();

    // Get output for 'yes'
    let output_tensor = interpreter.output(0);
    assert_eq!([1, 4], output_tensor.info().dims);

    dbg!(output_tensor.as_data::<u8>());
    let silence_score: u8 = output_tensor.as_data()[0];
    let unknown_score: u8 = output_tensor.as_data()[1];
    let yes_score: u8 = output_tensor.as_data()[2];
    let no_score: u8 = output_tensor.as_data()[3];

    assert!(yes_score > silence_score);
    assert!(yes_score > unknown_score);
    assert!(yes_score > no_score);

    // -------- 'no' example --------

    interpreter.input(0, no).unwrap();
    interpreter.invoke().unwrap();

    // Get output for 'no'
    let output_tensor = interpreter.output(0);
    assert_eq!([1, 4], output_tensor.info().dims);

    dbg!(output_tensor.as_data::<u8>());
    let silence_score: u8 = output_tensor.as_data()[0];
    let unknown_score: u8 = output_tensor.as_data()[1];
    let yes_score: u8 = output_tensor.as_data()[2];
    let no_score: u8 = output_tensor.as_data()[3];

    assert!(no_score > silence_score);
    assert!(no_score > unknown_score);
    assert!(no_score > yes_score);

    interpreter.arena_used_bytes();
    info!("Output Info: {:?}", output_tensor.info());

    info!("---- Done");
}
