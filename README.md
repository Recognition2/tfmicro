[![docs.rs](https://docs.rs/tfmicro/badge.svg)](https://docs.rs/tfmicro)
[![Crates.io](https://img.shields.io/crates/v/tfmicro.svg)](https://crates.io/crates/tfmicro)

# tfmicro


TensorFlow + `no_std` + Rust = ❤️

The crate contains Rust bindings for the [TensorFlow Lite for Microcontrollers][]
project. TensorFlow Lite for Microcontrollers is a version of TensorFlow Lite designed to run
without a standard library, for use on microcontrollers, wasm and more.

Thanks to [Cargo][] and the [CC crate][], there's no porting required for
new platforms - just drop `tfmicro` into your `Cargo.toml` and go. You will
need a C++ compiler behind the scenes, including for cross-compiling
targets, but in most cases this will be present already.

The aim of this implementation is to provide usability alongside all the
usual Rust guarantees. Nonetheless this crate calls into a very significant
quantity of C/C++ code, and you should employ the usual methods (static
analysis, memory guards, fuzzing) as you would with any other legacy C/C++
codebase.

Currently pinned to TensorFlow
[33689c48ad](https://github.com/tensorflow/tensorflow/tree/33689c48ad5e00908cd59089ef1956e1478fda78)

## Getting started

Add `tfmicro` in the dependencies section of your `Cargo.toml`

```
[dependencies]
tfmicro = 0.1.0
```

To understand how the [TensorFlow Micro C examples][c_examples] map to
idiomatic Rust code, see the [Tests](tests/) directory. Otherwise
for a more general description see [Usage](#Usage).

## Usage

#### Creating a model

Typically a model is exported from the TensorFlow training framework in a
binary file format with extension `.tflite`. You can import this straight
into Rust with the
[`include_bytes!`](https://doc.rust-lang.org/core/macro.include_bytes.html)
macro.

Then we can use the [`Model::from_buffer`](crate::Model::from_buffer) method
to perform a zero-copy conversion into a `Model`.

```rust
let model_array = include_bytes!("../examples/models/hello_world.tflite");
let model = Model::from_buffer(&model_array[..]).unwrap();
```

#### Creating a tensor arena

The TensorFlow interpreter requires a working area called a "Tensor
Arena". You can use an array on the stack for this, although it must remain
in scope whilst you use it. Alternatively if you have a `std` or `alloc`
environment, you can pass a heap-allocated `Vec` instead.

```rust
const TENSOR_ARENA_SIZE: usize = 4 * 1024;
let mut arena: [u8; TENSOR_ARENA_SIZE] = [0; TENSOR_ARENA_SIZE];
```

TensorFlow requires that the size of the arena is determined before creating
the interpreter. However, once you have created the interpreter, you can get
the number of bytes actually used by the model by calling
[`arena_used_bytes`](crate::MicroInterpreter::arena_used_bytes).


#### Instantiating an Interpreter and Input Tensors

To run the model, an interpreter is [built](crate::MicroInterpreter::new) in
much the same way as in the C API. Note that unlike the C API no
`error_reporter` is required. Error reports from TensorFlow are always
passed to the standard Rust [`log`](https://crates.io/crates/log) framework
at the `info` log level. This allows any compatible log implementation to be
used.

A op_resolver is required for the interpreter. The simplest option is to
pass an [`AllOpResolver`](crate::AllOpResolver), but to save memory use a
[`MutableOpResolver`](crate::MutableOpResolver) with the required operations
only.

```rust
let op_resolver = AllOpResolver::new();

let mut interpreter =
    MicroInterpreter::new(&model, op_resolver, &mut arena[..]).unwrap();

interpreter.input(0, &[0.0]).unwrap(); // Input tensor of length 1
```

The input tensor is set with the [`input`](crate::MicroInterpreter::input)
method. Simple models use only a single tensor, which can be specified with
index `0`.

#### Running the model and Output Tensors

The model is run by calling the [`invoke`](crate::MicroInterpreter::invoke)
method on the interpreter. The resulting output tensor is available by
calling the [`output`](crate::MicroInterpreter::output) method on the
interpreter.

```rust
interpreter.invoke().unwrap();

dbg!(interpreter.output(0).as_data::<f32>());
```

And that's it for a minimal use case! See the [Tests](tests/) folder
for more advanced use cases.

## Developing

See [DEVELOP.md](DEVELOP.md)

## License (this crate)

[Apache 2.0](LICENSE-APACHE)

[rust-embedded]: https://www.rust-lang.org/what/embedded
[TensorFlow Lite for Microcontrollers]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro
[Cargo]: https://doc.rust-lang.org/stable/cargo/
[CC crate]: https://crates.io/crates/cc
[c_examples]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples

Copyright 2020 Kevin Hill and Richard Meadows

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
