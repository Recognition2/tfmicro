//! micro_speech example

extern crate env_logger;
extern crate tfmicro;

pub fn main() {
    env_logger::init();

    let model = include_bytes!("models/micro_speech.tflite");
    let no = include_bytes!("models/no_micro_f9643d42_nohash_4.data");
    let yes = include_bytes!("models/yes_micro_f2e59fea_nohash_1.data");

    tfmicro::do_it(model, yes, no);
}
