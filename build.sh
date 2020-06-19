#!/bin/bash

# A script for repeatably, cleanly, slowly re-building the crate from scratch

cd submodules/tensorflow

rm -rf tensorflow/lite/micro/tools/make/downloads

make -f tensorflow/lite/micro/tools/make/Makefile clean
make -f tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test
cd ../..

cargo clean
cargo test || false
