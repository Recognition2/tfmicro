#!/bin/bash

TF_HASH=$(git submodule status submodules/tensorflow | awk '{print $1}')

echo "Currently pinned to TensorFlow [${TF_HASH:0:10}](https://github.com/tensorflow/tflite-micro/tree/${TF_HASH})"
