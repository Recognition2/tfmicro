use tfmicro::{
    bindings, micro_error_reporter::MicroErrorReporter,
    micro_interpreter::MicroInterpreter, micro_op_resolver::MicroOpResolver,
    model::Model,
};

#[test]
pub fn micro_speech() {
    println!("Starting test micro_speech_rust");
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

    // Pull in all operation implementations
    let micro_op_resolver = MicroOpResolver::new_for_microspeech();

    // Build an interpreter to run the model with
    let error_reporter = MicroErrorReporter::new();
    let mut interpreter = MicroInterpreter::new(
        &model,
        micro_op_resolver,
        &mut tensor_arena,
        TENSOR_ARENA_SIZE,
        &error_reporter,
    );
    let mut inp = interpreter.input(0);

    // Assert input properties
    assert_eq!([1, 49, 40, 1], inp.tensor_info().dims);
    assert_eq!(&bindings::TfLiteType::kTfLiteUInt8, inp.get_type());

    inp.tensor_data_mut().clone_from_slice(yes);

    let status = interpreter.Invoke();
    assert_eq!(bindings::TfLiteStatus::kTfLiteOk, status, "Invoke failed!");

    let output = interpreter.output(0);
    assert_eq!([1, 4], output.tensor_info().dims);
    assert_eq!(&bindings::TfLiteType::kTfLiteUInt8, output.get_type());

    let silence_score: u8 = output.tensor_data()[0];
    let unknown_score: u8 = output.tensor_data()[1];
    let yes_score: u8 = output.tensor_data()[2];
    let no_score: u8 = output.tensor_data()[3];

    assert!(yes_score > silence_score);
    assert!(yes_score > unknown_score);
    assert!(yes_score > no_score);

    inp.tensor_data_mut().clone_from_slice(no);

    let status = interpreter.Invoke();
    assert_eq!(bindings::TfLiteStatus::kTfLiteOk, status);

    let output = interpreter.output(0);
    assert_eq!([1, 4], output.tensor_info().dims);
    assert_eq!(&bindings::TfLiteType::kTfLiteUInt8, output.get_type());

    let silence_score: u8 = output.tensor_data()[0];
    let unknown_score: u8 = output.tensor_data()[1];
    let yes_score: u8 = output.tensor_data()[2];
    let no_score: u8 = output.tensor_data()[3];

    assert!(no_score > silence_score);
    assert!(no_score > unknown_score);
    assert!(no_score > yes_score);
}
