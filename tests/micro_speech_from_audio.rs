//! micro_speech example, from audio files

use tfmicro::{Frontend, MicroInterpreter, Model, MutableOpResolver};

use itertools::Itertools;
use log::info;

/// Returns 40 elements of micro_feature from an audio slice
fn micro_speech_frontend(
    frontend: &mut Frontend,
    audio_slice: &[i16],
) -> [u8; 40] {
    // Run generate_micro_features
    let mut output: [u16; 40] = [0; 40];
    frontend.generate_micro_features(audio_slice, &mut output);

    // Scaling, values derived from those used in the training pipeline. See
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
    assert_eq!([1, 49, 40, 1], interpreter.input_info(0).dims);

    // -------- 'yes' example --------

    // Run the front end on 30ms slices every 20ms
    let micro_feature = (0..49)
        .map(|n| &yes_1000ms[n * 320..(n * 320) + 480])
        .map(|audio_slice| micro_speech_frontend(&mut frontend, audio_slice))
        .fold(vec![], |mut acc: Vec<u8>, slice| {
            acc.extend(&slice[..]);
            acc
        });

    assert_eq!(micro_feature.len(), 1960);

    // Invoke interpreter
    interpreter.input(0, &micro_feature).unwrap();
    interpreter.invoke().unwrap();

    // Get the output tensor
    let output_tensor = interpreter.output(0);
    assert_eq!([1, 4], output_tensor.info().dims);

    info!("{:?}", output_tensor.as_data::<u8>());

    // Result must be 'yes'
    assert_eq!(Some(2), output_tensor.as_data::<u8>().iter().position_max());
    assert!(output_tensor.as_data::<u8>()[2] > 220);

    // -------- 'no' example --------

    // Run the front end on 30ms slices every 20ms
    let micro_feature = (0..49)
        .map(|n| &no_1000ms[n * 320..(n * 320) + 480])
        .map(|audio_slice| micro_speech_frontend(&mut frontend, audio_slice))
        .fold(vec![], |mut acc: Vec<u8>, slice| {
            acc.extend(&slice[..]);
            acc
        });

    assert_eq!(micro_feature.len(), 1960);

    // Invoke interpreter
    interpreter.input(0, &micro_feature).unwrap();
    interpreter.invoke().unwrap();

    // Get the output tensor
    let output_tensor = interpreter.output(0);
    assert_eq!([1, 4], output_tensor.info().dims);

    info!("{:?}", output_tensor.as_data::<u8>());

    // Result must be 'no'
    assert_eq!(Some(3), output_tensor.as_data::<u8>().iter().position_max());
    assert!(output_tensor.as_data::<u8>()[3] > 220);

    info!("---- Done");
}
