tfmicro
======

## Developing

Update submodules:

```
git submodule init
git submodule update
```

Then we need to tell tensorflow to download its own dependencies. Actually
building one of the tensorflow micro examples does this, and as a side
effect checks that tensorflow is working.

```
cd submodules/tensorflow
make -f tensorflow/lite/micro/tools/make/Makefile test_micro_speech_test
cd ../..
```

Finally we need patched versions of the `rust-cpp` crates. The
`[patches.crates-io]` statement in Cargo.toml expects you arranged the
crates like this

```
tfmicro
  - README.md
rust-cpp
  - cpp_build
  - cpp_macros
```

So you need to

```
cd ..
git clone https://github.com/mystor/rust-cpp
```

Then we can build!

```
cargo run --example minimal
```

We use the `env_logger` crate for log output, try

```
RUST_LOG=info cargo run --example minimal
```

Changes in the tensorflow source tree aren't tracked by cargo. If the
tensorflow source has changed, use the `build` feature gate to force a
re-build. You probably want `-j5` or similar for that.

```
cargo build -j5 --features build
```

To debug `build.rs` itself, try `cargo build -vv`

## TODO

* Build a better safe abstraction around `Model`
* Build a safe abstraction around all the other tflite types
* Replace all the `TF_LITE_MICRO_EXPECT_` macros with `assert!`
