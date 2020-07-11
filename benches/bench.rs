#![feature(test)]

extern crate test;
use test::Bencher;

use tfmicro::{MicroInterpreter, Model, MutableOpResolver};

use itertools::Itertools;
use log::info;

#[bench]
fn person_detection(b: &mut Bencher) {
    env_logger::init();

    info!("---- Starting tensorflow micro example: person_detection");

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

    info!("Created setup");

    // -------- 'person' example ------------
    b.iter(|| {
        // set input tensor
        interpreter.input(0, person).unwrap();

        interpreter.invoke().unwrap();

        // get output tensor
        let output_tensor = interpreter.output(0);
        assert_eq!(
            [1, 1, 1, 3],
            output_tensor.info().dims,
            "Dimensions of output tensor"
        );

        assert_eq!(
            1,
            output_tensor.as_data::<u8>().iter().position_max().unwrap()
        );
    });

    info!("---- Done");
}
