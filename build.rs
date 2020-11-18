//! Build

#[macro_use]
extern crate error_chain;
error_chain! {
    foreign_links {
        Io(::std::io::Error);
        EnvVar(::std::env::VarError);
        StringFromUtf8(::std::string::FromUtf8Error);
    }
}

use glob::glob;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use std::borrow::Borrow;
use std::ffi::OsStr;

fn run_command_or_fail<P, S>(dir: &str, cmd: P, args: &[S])
where
    P: AsRef<Path>,
    S: Borrow<str> + AsRef<OsStr>,
{
    let cmd = cmd.as_ref();
    let cmd = if cmd.components().count() > 1 && cmd.is_relative() {
        // If `cmd` is a relative path (and not a bare command that should be
        // looked up in PATH), absolutize it relative to `dir`, as otherwise the
        // behavior of std::process::Command is undefined.
        // https://github.com/rust-lang/rust/issues/37868
        PathBuf::from(dir)
            .join(cmd)
            .canonicalize()
            .expect("canonicalization failed")
    } else {
        PathBuf::from(cmd)
    };
    eprintln!(
        "Running command: \"{} {}\" in dir: {}",
        cmd.display(),
        args.join(" "),
        dir
    );
    let ret = Command::new(cmd).current_dir(dir).args(args).status();
    match ret.map(|status| (status.success(), status.code())) {
        Ok((true, _)) => {}
        Ok((false, Some(c))) => panic!("Command failed with error code {}", c),
        Ok((false, None)) => panic!("Command got killed"),
        Err(e) => panic!("Command failed with error: {}", e),
    }
}

fn check_submodules_or_checkout_from_git() {
    if !Path::new("submodules/tensorflow/LICENSE").exists() {
        eprintln!("Setting up submodules");
        run_command_or_fail(".", "git", &["submodule", "update", "--init"]);
    }

    if !Path::new("submodules/tensorflow/tensorflow/lite/micro/tools/make/downloads/flatbuffers/CONTRIBUTING.md").exists() {
        eprintln!("Building tensorflow micro example to fetch Tensorflow dependencies");
        run_command_or_fail("submodules/tensorflow", "make", &["-f", "tensorflow/lite/micro/tools/make/Makefile", "test_micro_speech_test"]);
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn submodules() -> PathBuf {
    manifest_dir().join("submodules")
}

fn flatbuffers_include_dir() -> PathBuf {
    submodules().join("tensorflow/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include")
}

fn is_cross_compiling() -> Result<bool> {
    Ok(env::var("TARGET")? != env::var("HOST")?)
}

fn get_command_result(command: &mut Command) -> Result<String> {
    command
        .output()
        .chain_err(|| "Couldn't find target GCC executable.")
        .and_then(|output| {
            if output.status.success() {
                Ok(String::from_utf8(output.stdout)?)
            } else {
                panic!("Couldn't read output from GCC.")
            }
        })
}

/// Move tensorflow source to $OUT_DIR
fn prepare_tensorflow_source() -> PathBuf {
    println!("Moving tensorflow micro source");
    let start = Instant::now();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let tf_src_dir = out_dir.join("tensorflow/tensorflow");
    let submodules = submodules();

    let copy_dir = fs_extra::dir::CopyOptions {
        content_only: false,
        overwrite: true,
        skip_exist: false,
        buffer_size: 65536,
        copy_inside: false,
        depth: 0,
    };

    if !tf_src_dir.exists() || cfg!(feature = "build") {
        // Copy directory
        println!("Copying TF from {:?}", submodules.join("tensorflow"));
        println!("Copying TF to {:?}", out_dir);
        fs_extra::dir::copy(submodules.join("tensorflow"), &out_dir, &copy_dir)
            .expect("Unable to copy tensorflow");
    }

    println!("Moving source took {:?}", start.elapsed());

    tf_src_dir
}

/// Return a Vec of all *.cc files in `path`, excluding those that have a
/// name containing 'test.cc'
fn get_files_glob(path: PathBuf) -> Vec<String> {
    let mut paths: Vec<String> = vec![];

    for entry in glob(&path.to_string_lossy()).unwrap() {
        let p: PathBuf = entry.unwrap();
        paths.push(p.to_string_lossy().to_string());
    }

    paths
        .into_iter()
        .filter(|p| !p.contains("test.cc"))
        .filter(|p| !p.contains("debug_log.cc"))
        .filter(|p| !p.contains("frontend_memmap"))
        .filter(|p| !p.contains("frontend_main"))
        .collect()
}

trait CompilationBuilder {
    fn flag(&mut self, s: &str) -> &mut Self;
    fn define(&mut self, var: &str, val: Option<&str>) -> &mut Self;

    /// Build flags for tensorflow micro sources
    fn tensorflow_build_setup(&mut self) -> &mut Self {
        let target = env::var("TARGET").unwrap_or_else(|_| "".to_string());

        let build = self
            .flag("-fno-rtti") // No Runtime type information
            .flag("-fmessage-length=0")
            .flag("-fno-exceptions")
            .flag("-fno-unwind-tables")
            .flag("-ffunction-sections")
            .flag("-fdata-sections")
            .flag("-funsigned-char")
            .flag("-MMD")
            .flag("-std=c++11")
            .flag("-fno-delete-null-pointer-checks")
            .flag("-fomit-frame-pointer")
            .flag("-fpermissive")
            .flag("-fno-use-cxa-atexit")
            // use a full word for enums, this should match clang's behaviour
            .flag("-fno-short-enums")
            .define("NDEBUG", None)
            .define("TF_LITE_STRIP_ERROR_STRINGS", None)
            .define("TF_LITE_STATIC_MEMORY", None)
            .define("TF_LITE_MCU_DEBUG_LOG", None)
            .define("GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK", None);

        // warnings on by default
        let build = if cfg!(feature = "no-c-warnings") {
            build.flag("-w")
        } else {
            build
                .flag("-Wvla")
                .flag("-Wall")
                .flag("-Wextra")
                .flag("-Wno-unused-parameter")
                .flag("-Wno-missing-field-initializers")
                .flag("-Wno-write-strings")
                .flag("-Wno-sign-compare")
                .flag("-Wunused-function")
        };

        if target.starts_with("thumb") {
            // unaligned accesses are usually a poor idea on ARM cortex-m
            build.flag("-mno-unaligned-access")
        } else {
            build
        }
    }
}
impl CompilationBuilder for cpp_build::Config {
    fn flag(&mut self, s: &str) -> &mut Self {
        self.flag(s)
    }
    fn define(&mut self, var: &str, val: Option<&str>) -> &mut Self {
        self.define(var, val)
    }
}
impl CompilationBuilder for cc::Build {
    fn flag(&mut self, s: &str) -> &mut Self {
        self.flag(s)
    }
    fn define(&mut self, var: &str, val: Option<&str>) -> &mut Self {
        self.define(var, val)
    }
}

fn cc_tensorflow_library() {
    let tflite = prepare_tensorflow_source();
    let out_dir = env::var("OUT_DIR").unwrap();
    let tf_lib_name =
        Path::new(&out_dir).join("libtensorflow-microlite.a".to_string());

    if is_cross_compiling().unwrap() {
        // Find include directory used by the crosscompiler for libm
        let mut gcc = cc::Build::new().get_compiler().to_command();
        let libm_location = PathBuf::from(
            get_command_result(gcc.arg("--print-file-name=libm.a"))
                .expect("Error querying gcc for libm location"),
        );
        let libm_path = libm_location.parent().unwrap();

        // Pass this to the linker
        println!(
            "cargo:rustc-link-search=native={}",
            libm_path.to_string_lossy()
        );
        println!("cargo:rustc-link-lib=static=m");
    }

    if !tf_lib_name.exists() || cfg!(feature = "build") {
        println!("Building tensorflow micro");
        let target = env::var("TARGET").unwrap_or_else(|_| "".to_string());
        let tfmicro_mdir = tflite.join("lite/micro/tools/make/");
        let start = Instant::now();

        let mut builder = cc::Build::new();
        let builder_ref = builder
            .cpp(true)
            .tensorflow_build_setup()
            .cpp_link_stdlib(None)
            //
            .include(tflite.parent().unwrap())
            .include(tfmicro_mdir.join("downloads"))
            .include(tfmicro_mdir.join("downloads/gemmlowp"))
            .include(tfmicro_mdir.join("downloads/flatbuffers/include"))
            .include(tfmicro_mdir.join("downloads/ruy"))
            //
            .files(get_files_glob(tflite.join("lite/micro/*.cc")))
            .files(get_files_glob(tflite.join("lite/micro/kernels/*.cc")))
            .files(get_files_glob(
                tflite.join("lite/micro/memory_planner/*.cc"),
            ))
            .files(get_files_glob(
                tflite.join("lite/experimental/microfrontend/lib/*.c"),
            ))
            .file(tflite.join("lite/c/common.c"))
            .file(tflite.join("lite/core/api/error_reporter.cc"))
            .file(tflite.join("lite/core/api/flatbuffer_conversions.cc"))
            .file(tflite.join("lite/core/api/op_resolver.cc"))
            .file(tflite.join("lite/core/api/tensor_utils.cc"))
            .file(tflite.join("lite/kernels/internal/quantization_util.cc"))
            .file(tflite.join("lite/kernels/kernel_util.cc"));

        // CMSIS-NN for ARM Cortex-M targets
        if target.starts_with("thumb")
            && target.contains("m-none-")
            && cfg!(feature = "cmsis-nn")
        {
            println!("Build includes CMSIS-NN.");
            let cmsis = tflite.join("lite/micro/tools/make/downloads/cmsis");

            builder_ref
                .files(get_files_glob(cmsis.join("CMSIS/NN/Source/*.c")))
                .include(cmsis.join("CMSIS/NN/Include"))
                .include(cmsis.join("CMSIS/DSP/Include"))
                .include(cmsis.join("CMSIS/Core/Include"));
        }

        // micro frontend
        builder_ref
            .include(tfmicro_mdir.join("downloads/kissfft"))
            .include(tfmicro_mdir.join("downloads/kissfft/tools"))
            .include(tflite.join("lite/experimental/microfrontend/lib"))
            .files(get_files_glob(
                tflite.join("lite/experimental/microfrontend/lib/*.cc"),
            ))
            .file(tfmicro_mdir.join("downloads/kissfft/kiss_fft.c"))
            .file(tfmicro_mdir.join("downloads/kissfft/tools/kiss_fftr.c"));

        // Compile
        builder_ref.compile("tensorflow-microlite");

        println!(
            "Building tensorflow micro from source took {:?}",
            start.elapsed()
        );
    } else {
        println!("Didn't rebuild tensorflow micro, using {:?}", tf_lib_name);

        println!("cargo:rustc-link-lib=static=tensorflow-microlite");
        println!("cargo:rustc-link-search=native={}", out_dir);
    }
}

/// Configure bindgen for cross-compiling
fn bindgen_cross_builder() -> Result<bindgen::Builder> {
    let builder = bindgen::Builder::default().clang_arg("--verbose");

    if is_cross_compiling()? {
        // Setup target triple
        let target = env::var("TARGET")?;
        let builder = builder.clang_arg(format!("--target={}", target));
        println!("Setting bindgen to cross compile to {}", target);

        // Find the sysroot used by the crosscompiler, and pass this to clang
        let mut gcc = cc::Build::new().get_compiler().to_command();
        let path = get_command_result(gcc.arg("--print-sysroot"))?;
        let builder = builder.clang_arg(format!("--sysroot={}", path.trim()));

        // Add a path to the system headers for the target
        // compiler. Possibly we end up using a gcc header with clang
        // frontend, which is sketchy.
        let search_paths = cc::Build::new()
            .cpp(true)
            .get_compiler()
            .to_command()
            .arg("-E")
            .arg("-Wp,-v")
            .arg("-xc++")
            .arg(".")
            .output()
            .chain_err(|| "Couldn't find target GCC executable.")
            .and_then(|output| {
                // We have to scrape the gcc console output to find where
                // the c++ headers are. If we only needed the c headers we
                // could use `--print-file-name=include` but that's not
                // possible.
                let gcc_out = String::from_utf8(output.stderr)?;

                // Scrape the search paths
                let search_start = gcc_out.find("search starts here").unwrap();
                let search_paths: Vec<PathBuf> = gcc_out[search_start..]
                    .split('\n')
                    .map(|p| PathBuf::from(p.trim()))
                    .filter(|path| path.exists())
                    .collect();

                Ok(search_paths)
            })?;

        // Add scraped paths to builder
        let mut builder = builder.detect_include_paths(false);
        for path in search_paths {
            builder =
                builder.clang_arg(format!("-I{}", path.to_string_lossy()));
        }
        Ok(builder)
    } else {
        Ok(builder)
    }
}

/// This generates "tflite_types.rs" containing structs and enums which are
/// inter-operable with rust
fn bindgen_tflite_types() {
    use bindgen::*;

    let submodules = submodules();
    let submodules_str = submodules.to_string_lossy();
    let out_dir = env::var("OUT_DIR").unwrap();
    let tflite_types_name = Path::new(&out_dir).join("tflite_types.rs");

    if !tflite_types_name.exists() || cfg!(feature = "build") {
        println!("Running bindgen");
        let start = Instant::now();

        let bindings = bindgen_cross_builder()
            .expect("Error setting up bindgen for cross compiling")
            .whitelist_recursively(true)
            .prepend_enum_name(false)
            .impl_debug(true)
            .with_codegen_config(CodegenConfig::TYPES)
            .layout_tests(false)
            .enable_cxx_namespaces()
            .derive_default(true)
            .size_t_is_usize(true)
            .use_core()
            .ctypes_prefix("cty")
            // Types
            .whitelist_type("tflite::MicroErrorReporter")
            .opaque_type("tflite::MicroErrorReporter")
            .whitelist_type("tflite::Model")
            .opaque_type("tflite::Model")
            .whitelist_type("tflite::MicroInterpreter")
            .opaque_type("tflite::MicroInterpreter")
            .whitelist_type("tflite::ops::micro::AllOpsResolver")
            .opaque_type("tflite::ops::micro::AllOpsResolver")
            .whitelist_type("TfLiteTensor")
            .whitelist_type("FrontendState")
            .whitelist_type("FrontendConfig")
            .whitelist_type("FrontendOutput")
            // Types - blacklist
            .blacklist_type("std")
            .blacklist_type("tflite::Interpreter_TfLiteDelegatePtr")
            .blacklist_type("tflite::Interpreter_State")
            .default_enum_style(EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_partialeq(true)
            .derive_eq(true)
            .header("csrc/tflite_wrapper.hpp")
            .clang_arg(format!("-I{}/tensorflow", submodules_str))
            .clang_arg(format!(
                // -> flatbuffers/flatbuffers.h
                "-I{}",
                flatbuffers_include_dir().to_string_lossy()
            ))
            .clang_arg("-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK")
            .clang_arg("-xc++")
            .clang_arg("-std=c++11");

        let bindings =
            bindings.generate().expect("Unable to generate bindings");

        // Write the bindings to $OUT_DIR/tflite_types.rs
        let out_path = PathBuf::from(out_dir).join("tflite_types.rs");
        bindings
            .write_to_file(out_path)
            .expect("Couldn't write bindings!");

        println!("Running bindgen took {:?}", start.elapsed());
    } else {
        println!("Didn't regenerate bindings");
    }
}

fn build_inline_cpp() {
    let submodules = submodules();

    println!("Building inline cpp");
    let start = Instant::now();

    cpp_build::Config::new()
        .include(submodules.join("tensorflow"))
        .include(flatbuffers_include_dir())
        .tensorflow_build_setup()
        .cpp_link_stdlib(None)
        //.flag("-std=c++14")
        .build("src/lib.rs");

    println!("Building inline cpp took {:?}", start.elapsed());
}

fn main() {
    check_submodules_or_checkout_from_git();
    bindgen_tflite_types();
    build_inline_cpp();
    cc_tensorflow_library();
}
