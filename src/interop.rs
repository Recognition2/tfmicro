//! Symbol definitions those missing from building tensorflow without a C++
//! (or C) standard library

pub(crate) mod strlen {
    cpp! {{
        #include <string.h>
        #include <stdint.h>
        #include <limits.h>

        #define ALIGN (sizeof(size_t))
        #define ONES ((size_t)-1/UCHAR_MAX)
        #define HIGHS (ONES * (UCHAR_MAX/2+1))
        #define HASZERO(x) (((x)-ONES) & ~(x) & HIGHS)
    }}

    // A strlen implementation
    pub unsafe fn strlen(string: *const cty::c_char) -> usize {
        cpp! ([string as "char *"] -> usize as "size_t" {
            const char *s = string;
            const char *a = s;
            const size_t *w;
            for (; (uintptr_t)s % ALIGN; s++) if (!*s) return s-a;
            for (w = (const size_t *)s; !HASZERO(*w); w++);
            for (s = (const char *)w; *s; s++);
            return s-a;
        })
    }
}

// private module
mod tensorflow {
    use core::slice;
    use core::str;

    #[no_mangle]
    // __cxa_pure_virtual is a function, address of which compiler writes
    // in the virtual table when the function is pure virtual. It may be
    // called due to some unnatural pointer abuse or when trying to invoke
    // pure virtual function in the destructor of the abstract base
    // class. The call to this function should never happen in the normal
    // application run. If it happens it means there is a bug.
    pub extern "C" fn __cxa_pure_virtual() {
        loop {}
    }

    #[no_mangle]
    // A cleanup must return control to the unwinding code by tail calling
    // __cxa_end_cleanup. The routine performs whatever housekeeping is
    // required and resumes the exception propagation by calling
    // _Unwind_Resume. This routine does not return.
    pub extern "C" fn __cxa_end_cleanup() {
        loop {}
    }

    #[no_mangle]
    pub extern "C" fn __gxx_personality_v0() {}

    // Simple implementation of errno
    static ERRNO: cty::c_int = 0;
    #[no_mangle]
    pub extern "C" fn __errno() -> *const cty::c_int {
        &ERRNO
    }

    // Despite -fno-rtti, these symbols are still generated. Define them
    // here, in a way that would likely have horrific consequences at
    // runtime
    cpp! {{
        namespace __cxxabiv1 {
            class __class_type_info {
                virtual void dummy();
            };
            void __class_type_info::dummy() { }
        };
        namespace __cxxabiv1 {
            class __si_class_type_info {
                virtual void dummy();
            };
            void __si_class_type_info::dummy() { }
        };
    }}

    // A strcmp implementation, for flatbuffers to use
    cpp! {{
        #include <string.h>

        int strcmp(const char *l, const char *r)
        {
            for (; *l==*r && *l; l++, r++);
            return *(unsigned char *)l - *(unsigned char *)r;
        }
    }}
    cpp! {{
        #include <string.h>

        int strncmp(const char *l, const char *r, size_t n)
        {
            if (!n--) return 0;
            for (; *l && *r && n && *l == *r ; l++, r++, n--);
            return *l - *r;
        }
    }}

    #[no_mangle]
    // Repalcement for implementation in debug_log.cc
    pub extern "C" fn DebugLog(s: *const cty::c_char) {
        let slice = unsafe {
            let len = super::strlen::strlen(s);
            let ptr = s as *const u8;
            slice::from_raw_parts(ptr, len as usize + 1)
        };
        info!("{}", str::from_utf8(slice).unwrap().trim());
    }

    // Underlying assert function for tensorflow to use
    #[no_mangle]
    pub extern "C" fn __assert_func(
        _expr: *const cty::c_char,
        _line: cty::c_int,
        _file: *const cty::c_char,
        _function: *const cty::c_char,
    ) {
        panic!("__assert_func ASSERTED"); // __noreturn__
    }

    // Don't deallocate memory - tensorflow micro should be stack-based
    cpp! {{
        void operator delete(void * p) {}
    }}
}
