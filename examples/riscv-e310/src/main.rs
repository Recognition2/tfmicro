//! tensorflow micro (tfmicro) example for the E310
//!
//! micro-speech example, see
//! https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc

#![deny(warnings)]
#![deny(unsafe_code)]
#![no_main]
#![no_std]

use panic_halt as _;
use riscv_rt;

use e310x_hal::e310x as pac;

use e310x_hal::{
    clock::{AonExt, Clocks, PrciExt},
    prelude::*,
    time::Hertz,
};

extern crate tfmicro;
use tfmicro::{MicroInterpreter, Model, MutableOpResolver};

#[riscv_rt::entry]
fn main() -> ! {
    let p = pac::Peripherals::take().unwrap();

    let coreclk = p.PRCI.constrain();
    let coreclk = coreclk
        .use_external(Hertz(16_000_000))
        .coreclk(16_000_000_u32.hz());

    let aonclk = p.AONCLK.constrain();

    Clocks::freeze(coreclk, aonclk);

    let model = include_bytes!("../../models/micro_speech.tflite");
    let no = include_bytes!("../../models/no_micro_f9643d42_nohash_4.data");
    let yes = include_bytes!("../../models/yes_micro_f2e59fea_nohash_1.data");

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
    interpreter.input(0, yes).unwrap();
    interpreter.invoke().unwrap();

    // Get output for 'yes'
    let output_tensor = interpreter.output(0);
    assert_eq!([1, 4], output_tensor.info().dims);

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

    let silence_score: u8 = output_tensor.as_data()[0];
    let unknown_score: u8 = output_tensor.as_data()[1];
    let yes_score: u8 = output_tensor.as_data()[2];
    let no_score: u8 = output_tensor.as_data()[3];

    assert!(no_score > silence_score);
    assert!(no_score > unknown_score);
    assert!(no_score > yes_score);

    loop {}
}
