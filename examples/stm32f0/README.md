# STM32F0xx example

tfmicro rust example for the STM32F098 (ARM Cortex M0)

## Building

Note: When cross-compiling with cargo, you need to prevent features enabled on
build dependencies from being enabled for normal dependencies. See this
[tracking issue](https://github.com/rust-lang/cargo/issues/7915).

```
cargo build -Z features=build_dep
```

## Binary Size

Built with `rustc 1.46.0-nightly (a8cf39911 2020-06-21)`

```
************************************************************
MCU:  STM32F098 debug

.text:    158380 / 262144 = 60.4172 %
.rodata:  49908 / 262144 = 19.0384 %

************************************************************
************************************************************
MCU:  STM32F098 release

.text:    75816 / 262144 = 28.9215 %
.rodata:  39072 / 262144 = 14.9048 %

************************************************************
```

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
