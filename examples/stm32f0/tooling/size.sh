#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

arm-none-eabi-size -Ad target/thumbv6m-none-eabi/debug/tfmicro-stm32f0-example \
    | awk -v PROFILE=debug -f $DIR/size.awk -

arm-none-eabi-size -Ad target/thumbv6m-none-eabi/release/tfmicro-stm32f0-example \
    | awk -v PROFILE=release -f $DIR/size.awk -
