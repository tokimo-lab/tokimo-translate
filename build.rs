//! Build script — detects GPU backend and emits `cargo:rustc-cfg=gguf_backend_X`
//! so that gguf_translate.rs can choose GPU layers at compile time.
//!
//! Backend priority: cuda > metal > vulkan > rocm > cpu
//! Features (--features cuda/metal/vulkan/rocm) must still be set explicitly;
//! this script only DETECTS what is available and warns if a GPU is found but
//! no matching feature was enabled.  The `Makefile` auto-selects the feature.

fn main() {
    // Register custom cfg names so rustc doesn't warn about unknown cfgs
    println!("cargo::rustc-check-cfg=cfg(gguf_backend_cuda)");
    println!("cargo::rustc-check-cfg=cfg(gguf_backend_metal)");
    println!("cargo::rustc-check-cfg=cfg(gguf_backend_vulkan)");
    println!("cargo::rustc-check-cfg=cfg(gguf_backend_rocm)");
    println!("cargo::rustc-check-cfg=cfg(gguf_backend_cpu)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let feat_cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    let feat_metal = std::env::var("CARGO_FEATURE_METAL").is_ok();
    let feat_vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();
    let feat_rocm = std::env::var("CARGO_FEATURE_ROCM").is_ok();

    // Detect available hardware
    let is_apple_silicon =
        target_os == "macos" && (target_arch == "aarch64" || target_arch == "arm64");
    let is_intel_mac = target_os == "macos" && target_arch == "x86_64";
    let cuda_available = std::path::Path::new("/usr/local/cuda").exists()
        || std::path::Path::new("/usr/cuda").exists()
        || std::env::var("CUDA_PATH").is_ok()
        || std::env::var("CUDA_HOME").is_ok()
        || which("nvcc");
    let rocm_available = std::path::Path::new("/opt/rocm").exists()
        || std::env::var("ROCM_PATH").is_ok();

    // Emit cfg flags for ALL enabled backends (can be multiple, e.g. cuda+vulkan)
    let mut any_gpu = false;
    if feat_cuda {
        println!("cargo:rustc-cfg=gguf_backend_cuda");
        any_gpu = true;
    }
    if feat_metal {
        println!("cargo:rustc-cfg=gguf_backend_metal");
        any_gpu = true;
    }
    if feat_vulkan {
        println!("cargo:rustc-cfg=gguf_backend_vulkan");
        any_gpu = true;
    }
    if feat_rocm {
        println!("cargo:rustc-cfg=gguf_backend_rocm");
        any_gpu = true;
    }

    if any_gpu {
        let backends: Vec<&str> = [
            feat_cuda.then_some("cuda"),
            feat_metal.then_some("metal"),
            feat_vulkan.then_some("vulkan"),
            feat_rocm.then_some("rocm"),
        ].into_iter().flatten().collect();
        println!("cargo:warning=🟢 llama.cpp backends compiled in: {}", backends.join(", "));
    } else {
        println!("cargo:rustc-cfg=gguf_backend_cpu");
        // Helpful hints about available but unused GPU
        if cuda_available {
            println!("cargo:warning=⚠️  CUDA toolkit detected — build with --features cuda for GPU");
            println!("cargo:warning=    or run: make  (auto-detects and adds the right feature)");
        } else if is_apple_silicon {
            println!("cargo:warning=⚠️  Apple Silicon detected — build with --features metal for GPU");
            println!("cargo:warning=    or run: make");
        } else if rocm_available {
            println!("cargo:warning=⚠️  ROCm detected — build with --features rocm for GPU");
        } else if is_intel_mac {
            println!("cargo:warning=ℹ️  Intel Mac — using CPU (no Metal GPU support for x86)");
        } else {
            println!("cargo:warning=ℹ️  llama.cpp backend: CPU");
        }
    }
}

fn which(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
