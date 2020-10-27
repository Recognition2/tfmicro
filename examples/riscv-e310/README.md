# SiFive E310x example

tfmicro rust example for the SiFive E310x

## Building

Note: When cross-compiling with cargo, you need to prevent features enabled on
build dependencies from being enabled for normal dependencies. See this
[tracking issue](https://github.com/rust-lang/cargo/issues/7915).

```
cargo build -Z features=build_dep
```

## C++ compiler

Error message:

`Error setting up bindgen for cross compiling: Couldn't find target GCC executable.`

You need to have `riscv32imac-unknown-elf-g++` installed.

Configure and compile GNU toolchain:

```
../configure --prefix=/opt/riscv32 --with-arch=rv32imac --with-abi=ilp32
```

```
make
```

And add the new compiler to your path.

## License (this example)

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
