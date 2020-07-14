// See build.rs:global_cmath_for_target

#ifndef TFMICRO_INJECT_CMATH_STD_HPP
#define TFMICRO_INJECT_CMATH_STD_HPP

#include <cmath>

// TODO: Extend with any more functions that show up during compilation

namespace std
{
    inline int round(float x)
    {
        return ::round(x);
    }
}

#endif
