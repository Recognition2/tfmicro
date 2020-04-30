#![recursion_limit = "8192"]
#![no_std]

#[macro_use]
extern crate log;
#[macro_use]
extern crate cpp;

mod bindings;
mod interop;
mod interpreter;
mod micro_error_reporter;
mod micro_interpreter;
mod micro_op_resolver;

pub mod model;

use interpreter::Tensor;
use micro_interpreter::MicroInterpreter;
use micro_error_reporter::MicroErrorReporter;
use micro_op_resolver::MicroOpResolver;

pub fn do_it(model: &[u8; 18288], yes: &[u8; 1960], no: &[u8; 1960]) -> bool {
    info!("Starting test...");

    // Map the model into a usable data structure. This doesn't involve
    // any copying or parsing, it's a very lightweight operation.
    let model = model::Model::from_buffer(&model[..]).unwrap();

    // Create an area of memory to use for input, output, and
    // intermediate arrays.
    const TENSOR_ARENA_SIZE: usize = 10 * 1024;
    let mut tensor_arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];

    // Pull in all operation implementations
    let micro_op_resolver = MicroOpResolver::new();

    // Build an interpreter to run the model with
    let error_reporter = MicroErrorReporter::new();
    let rusty_interpreter = MicroInterpreter::new(&model,
                                                  micro_op_resolver,
                                                  &mut tensor_arena,
                                                  TENSOR_ARENA_SIZE,
                                                  &error_reporter);

    unsafe {

        cpp! {{
            #include "tensorflow/lite/micro/kernels/micro_ops.h"
            #include "tensorflow/lite/micro/micro_error_reporter.h"
            #include "tensorflow/lite/micro/micro_interpreter.h"
            #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
            #include "tensorflow/lite/micro/testing/micro_test.h"
            #include "tensorflow/lite/schema/schema_generated.h"
            #include "tensorflow/lite/version.h"
        }}

        cpp! {{
            namespace micro_test {
                int tests_passed;
                int tests_failed;
                bool is_test_complete;
                bool did_test_fail;
                tflite::ErrorReporter* reporter;
            }
        }}

        let yes_features_data = yes.as_ptr();
        let no_features_data = no.as_ptr();

        let result = cpp! ([rusty_interpreter as "tflite::MicroInterpreter",
                            yes_features_data as "const uint8_t*",
                            no_features_data as "const uint8_t*"] -> bool as "bool" {

            // Example from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc

            micro_test::tests_passed = 0;
            micro_test::tests_failed = 0;

            // Set up logging.
            tflite::MicroErrorReporter micro_error_reporter;
            tflite::ErrorReporter* error_reporter = &micro_error_reporter;
            micro_test::reporter = &micro_error_reporter;

            // // Check model version
            // if (model->version() != TFLITE_SCHEMA_VERSION) {
            //     TF_LITE_REPORT_ERROR(error_reporter,
            //                          "Model provided is schema version %d not equal "
            //                          "to supported version %d.\n",
            //                          model->version(), TFLITE_SCHEMA_VERSION);
            // }

            // // Pull in only the operation implementations we need.
            // // This relies on a complete list of all the ops needed by this graph.
            // // An easier approach is to just use the AllOpsResolver, but this will
            // // incur some penalty in code space for op implementations that are not
            // // needed by this graph.
            // //
            // // tflite::ops::micro::AllOpsResolver resolver;
            // tflite::MicroOpResolver<3> micro_op_resolver;
            // micro_op_resolver.AddBuiltin(
            //     tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
            //     tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
            // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
            //                              tflite::ops::micro::Register_FULLY_CONNECTED());
            // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
            //                              tflite::ops::micro::Register_SOFTMAX());

            // // Create an area of memory to use for input, output, and intermediate arrays.
            // const int tensor_arena_size = 10 * 1024;
            // uint8_t tensor_arena[tensor_arena_size];

            // // Build an interpreter to run the model with.
            // tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
            //                                      tensor_arena_size, error_reporter);
            // interpreter.AllocateTensors();

            tflite::MicroInterpreter interpreter = rusty_interpreter;

            // Get information about the memory area to use for the model's input.
            TfLiteTensor* input = interpreter.input(0);

            // Make sure the input has the properties we expect.
            TF_LITE_MICRO_EXPECT_NE(nullptr, input);
            TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
            TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
            TF_LITE_MICRO_EXPECT_EQ(49, input->dims->data[1]);
            TF_LITE_MICRO_EXPECT_EQ(40, input->dims->data[2]);
            TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[3]);
            TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

            // Copy a spectrogram created from a .wav audio file of someone saying "Yes",
            // into the memory area used for the input.
            //const uint8_t* yes_features_data = g_yes_micro_f2e59fea_nohash_1_data;
            for (int i = 0; i < input->bytes; ++i) {
                input->data.uint8[i] = yes_features_data[i];
            }

            // Run the model on this input and make sure it succeeds.
            TfLiteStatus invoke_status = interpreter.Invoke();
            if (invoke_status != kTfLiteOk) {
                TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
            }
            TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

            // Get the output from the model, and make sure it's the expected size and
            // type.
            TfLiteTensor* output = interpreter.output(0);
            TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
            TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
            TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
            TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

            // There are four possible classes in the output, each with a score.
            const int kSilenceIndex = 0;
            const int kUnknownIndex = 1;
            const int kYesIndex = 2;
            const int kNoIndex = 3;

            // Make sure that the expected "Yes" score is higher than the other classes.
            uint8_t silence_score = output->data.uint8[kSilenceIndex];
            uint8_t unknown_score = output->data.uint8[kUnknownIndex];
            uint8_t yes_score = output->data.uint8[kYesIndex];
            uint8_t no_score = output->data.uint8[kNoIndex];
            TF_LITE_MICRO_EXPECT_GT(yes_score, silence_score);
            TF_LITE_MICRO_EXPECT_GT(yes_score, unknown_score);
            TF_LITE_MICRO_EXPECT_GT(yes_score, no_score);

            // Now test with a different input, from a recording of "No".
            //const uint8_t* no_features_data = g_no_micro_f9643d42_nohash_4_data;
            for (int i = 0; i < input->bytes; ++i) {
                input->data.uint8[i] = no_features_data[i];
            }

            // Run the model on this "No" input.
            invoke_status = interpreter.Invoke();
            if (invoke_status != kTfLiteOk) {
                TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
            }
            TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

            // Get the output from the model, and make sure it's the expected size and
            // type.
            output = interpreter.output(0);
            TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
            TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
            TF_LITE_MICRO_EXPECT_EQ(4, output->dims->data[1]);
            TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

            // Make sure that the expected "No" score is higher than the other classes.
            silence_score = output->data.uint8[kSilenceIndex];
            unknown_score = output->data.uint8[kUnknownIndex];
            yes_score = output->data.uint8[kYesIndex];
            no_score = output->data.uint8[kNoIndex];
            TF_LITE_MICRO_EXPECT_GT(no_score, silence_score);
            TF_LITE_MICRO_EXPECT_GT(no_score, unknown_score);
            TF_LITE_MICRO_EXPECT_GT(no_score, yes_score);

            TF_LITE_REPORT_ERROR(error_reporter, "Ran successfully\n");

            return micro_test::did_test_fail;
        });

        info!("Done LIB. Did_Test_Fail = {:?}", result);
        result
    }
}
