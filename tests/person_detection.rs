//! person_detection example
//!
use tfmicro::{
    micro_interpreter::MicroInterpreter, micro_op_resolver::MutableOpResolver,
    model::Model,
};

use itertools::Itertools;
use log::info;

#[test]
fn person_detection() {
    env_logger::init();

    info!("---- Starting tensorflow micro example: pesron_detection");

    // Include trained model and test datasets
    let model =
        include_bytes!("../examples/models/person_detection_grayscale.tflite");
    let no_person = include_bytes!(
        "../examples/models/no_person_image_data_grayscale.data"
    );
    let person =
        include_bytes!("../examples/models/person_image_data_grayscale.data");

    // Map the model into a usable data structure. This doesn't involve
    // any copying or parsing, it's a very lightweight operation.
    let model = Model::from_buffer(&model[..]).unwrap();

    // Create memory area for input, output and intermediate arrays
    const TENSOR_ARENA_SIZE: usize = 93 * 1024;
    let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];

    let micro_op_resolver = MutableOpResolver::empty()
        .depthwise_conv_2d()
        .conv_2d()
        .average_pool_2d();

    // Build an interpreter to run the model with
    let mut interpreter =
        MicroInterpreter::new(&model, micro_op_resolver, &mut tensor_arena[..])
            .unwrap();

    // Check properties of the input sensor
    let input = interpreter.input(0);
    assert_eq!([1, 96, 96, 1], input.tensor_info().dims);

    info!("Created setup");

    // -------- 'person' example ------------
    input.tensor_data_mut().clone_from_slice(person);

    interpreter.invoke().unwrap();

    // get output for 'person'
    let output = interpreter.output(0);
    assert_eq!(
        [1, 1, 1, 3],
        output.tensor_info().dims,
        "Dimensions of output tensor"
    );

    assert_eq!(1, output.tensor_data::<u8>().iter().position_max().unwrap());
    info!("---- Person output correct!");

    // ------- 'no person' example ----------
    input.tensor_data_mut().clone_from_slice(no_person);

    interpreter.invoke().unwrap();

    // get output for 'no person'
    let output = interpreter.output(0);
    assert_eq!(
        [1, 1, 1, 3],
        output.tensor_info().dims,
        "Dimensions of output tensor"
    );

    assert_eq!(2, output.tensor_data::<u8>().iter().position_max().unwrap());
    info!("---- No-person output correct!");

    info!("---- Done");
}
