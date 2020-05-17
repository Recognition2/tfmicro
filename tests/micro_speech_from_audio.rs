//! micro_speech example, from audio files

extern crate itertools;
use itertools::Itertools;

use tfmicro::{
    frontend::Frontend, micro_interpreter::MicroInterpreter,
    micro_op_resolver::MutableOpResolver, model::Model,
};

use log::info;

/// Returns 40 elements of micro_feature from an audio slice
fn micro_speech_frontend(
    frontend: &mut Frontend,
    audio_slice: &[i16],
) -> [u8; 40] {
    // Run generate_micro_features
    let mut output: [u16; 40] = [0; 40];
    frontend.generate_micro_features(audio_slice, &mut output);

    // Scaling, values dervied from using in the training pipeline. See
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.cc
    let mut scaled_features: [u8; 40] = [0; 40];
    for m in 0..40 {
        let scaled = ((output[m] as f32 * 2550.) + 3328.) / 6656.;
        scaled_features[m] = match scaled {
            x if x < 0. => 0u8,
            x if x > 255. => 255u8,
            x => x as u8,
        };
    }

    scaled_features
}

#[test]
fn micro_speech_with_audio() {
    env_logger::init();
    info!("---- Starting tensorflow micro example: micro_speech_from_audio");

    let model = include_bytes!("../examples/models/micro_speech.tflite");
    let no_1000ms = &include_bytes!("../examples/models/no_1000ms_sample.data")
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect_vec();
    let yes_1000ms =
        &include_bytes!("../examples/models/yes_1000ms_sample.data")
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect_vec();

    // Frontend for creating micro_features
    let mut frontend = Frontend::new().unwrap();

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

    // Build an interpreter to run the model with
    let mut interpreter =
        MicroInterpreter::new(&model, micro_op_resolver, &mut tensor_arena[..])
            .unwrap();

    // Check properties of the input sensor
    assert_eq!([1, 49, 40, 1], interpreter.input_tensor_info(0).dims);

    // -------- 'yes' example --------
    info!("Begin 'yes' example");
    let mut yes_count = 0;
    let mut no_count = 0;

    // Run the front end on 30ms slices every 10ms
    let micro_feature_slices = (0..97)
        .map(|n| &yes_1000ms[n * 160..(n * 160) + 480])
        .map(|audio_slice| micro_speech_frontend(&mut frontend, audio_slice))
        .collect_vec();

    for n in 0..48 {
        // Collect 49 elements from a Vec<[u8; 40]> into a Vec<u8>
        let micro_feature: Vec<u8> = micro_feature_slices[n..n + 49]
            .into_iter()
            .fold(vec![], |mut acc: Vec<u8>, slice| {
                acc.extend(&slice[..]);
                acc
            });

        assert_eq!(micro_feature.len(), 1960);

        // Invoke interpreter
        interpreter.input(0, &micro_feature).unwrap();
        interpreter.invoke().unwrap();

        // Get the output tensor
        let output = interpreter.output(0);
        assert_eq!([1, 4], output.tensor_info().dims);

        info!("{:?}", output.tensor_data::<u8>());

        // Count the number of `yes` values
        let max_index =
            output.tensor_data::<u8>().iter().position_max().unwrap();
        if max_index == 2 && output.tensor_data::<u8>()[2] >= 220 {
            yes_count += 1;
        }

        // Count the number of `no` values
        let max_index =
            output.tensor_data::<u8>().iter().position_max().unwrap();
        if max_index == 3 && output.tensor_data::<u8>()[3] >= 220 {
            no_count += 1;
        }
    }

    info!("Counted {} 'yes' tensors", yes_count);
    assert!(yes_count > 10);
    assert_eq!(no_count, 0, "Counted a 'no' during the 'yes' example");

    // -------- 'no' example --------
    info!("Begin 'no' example");
    let mut yes_count = 0;
    let mut no_count = 0;

    // Run the front end on 30ms slices every 10ms
    let micro_feature_slices = (0..97)
        .map(|n| &no_1000ms[n * 160..(n * 160) + 480])
        .map(|audio_slice| micro_speech_frontend(&mut frontend, audio_slice))
        .collect_vec();

    for n in 0..48 {
        // Collect 49 elements from a Vec<[u8; 40]> into a Vec<u8>
        let micro_feature = micro_feature_slices[n..n + 49].into_iter().fold(
            vec![],
            |mut acc: Vec<u8>, slice| {
                acc.extend(&slice[..]);
                acc
            },
        );

        assert_eq!(micro_feature.len(), 1960);

        // Invoke interpreter
        interpreter.input(0, &micro_feature).unwrap();
        interpreter.invoke().unwrap();

        // Get the output tensor
        let output = interpreter.output(0);
        assert_eq!([1, 4], output.tensor_info().dims);

        info!("{:?}", output.tensor_data::<u8>());

        // Count the number of `yes` values
        let max_index =
            output.tensor_data::<u8>().iter().position_max().unwrap();
        if max_index == 2 && output.tensor_data::<u8>()[2] >= 220 {
            yes_count += 1;
        }

        // Count the number of `no` values
        let max_index =
            output.tensor_data::<u8>().iter().position_max().unwrap();
        if max_index == 3 && output.tensor_data::<u8>()[3] >= 220 {
            no_count += 1;
        }
    }

    info!("Counted {} 'no' tensors", no_count);
    assert!(no_count > 10);
    assert_eq!(yes_count, 0, "Counted a 'yes' during the 'no' example");

    info!("---- Done");
}
