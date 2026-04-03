# Auto-detect GPU backend and build with the right Cargo feature.
#
# For binary distribution (compile once, users run directly):
#   make release-linux   — CUDA + Vulkan (covers NVIDIA + AMD/Intel GPU)
#   make release-mac     — Metal (Apple Silicon)
#   make release-windows — CUDA + Vulkan
#
# For local development:
#   make            — auto-detect and build (debug)
#   make run        — auto-detect, build, and run

.PHONY: all run release release-linux release-mac release-windows cuda metal vulkan rocm cpu detect clean

OS   := $(shell uname -s)
ARCH := $(shell uname -m)

FEATURE := $(shell \
  if [ "$(OS)" = "Darwin" ] && [ "$(ARCH)" = "arm64" ]; then \
    echo "metal"; \
  elif command -v nvcc >/dev/null 2>&1 || [ -d /usr/local/cuda ] || [ -n "$$CUDA_PATH" ]; then \
    echo "cuda,vulkan"; \
  elif [ -d /opt/rocm ] || [ -n "$$ROCM_PATH" ]; then \
    echo "rocm"; \
  elif command -v vulkaninfo >/dev/null 2>&1; then \
    echo "vulkan"; \
  else \
    echo "cpu"; \
  fi)

CARGO_FLAGS := $(if $(filter cpu,$(FEATURE)),,--features $(FEATURE))

all: detect
	cargo build --bin gguf_translate $(CARGO_FLAGS)

run: detect
	cargo run --bin gguf_translate $(CARGO_FLAGS)

detect:
	@echo "🔍 Detected backend: $(FEATURE)"

# --- Release builds for binary distribution ---

# Linux: CUDA + Vulkan → runtime picks best GPU (NVIDIA, AMD, Intel, or CPU)
release-linux:
	cargo build --release --bin gguf_translate --features cuda,vulkan
	@echo "✅ Linux binary: target/release/gguf_translate  (CUDA + Vulkan, runtime auto-select)"

# macOS Apple Silicon: Metal → uses GPU if M-chip, CPU otherwise
release-mac:
	cargo build --release --bin gguf_translate --features metal
	@echo "✅ Mac binary: target/release/gguf_translate  (Metal)"

# Windows (run in PowerShell/WSL)
release-windows:
	cargo build --release --bin gguf_translate --features cuda,vulkan
	@echo "✅ Windows binary built (CUDA + Vulkan)"

release: detect
	cargo build --release --bin gguf_translate $(CARGO_FLAGS)
	@echo "✅ Release binary: target/release/gguf_translate"

# --- Explicit backend targets ---
cuda:
	cargo build --bin gguf_translate --features cuda

metal:
	cargo build --bin gguf_translate --features metal

vulkan:
	cargo build --bin gguf_translate --features vulkan

rocm:
	cargo build --bin gguf_translate --features rocm

cpu:
	cargo build --bin gguf_translate

clean:
	cargo clean
