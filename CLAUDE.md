# AGENTS.md

我们在给这个 rust 推理工具做 python-binding，让 python 开发者可以在 python 中用 deepseek-ocr.rs 库，对 deepseek-ocr, paddle ocr, dots-cor 还有未来的其他 ocr 模型进行推理。

在合适的时候将改动提交到 git 中，遵守 conventional commits。

所有工作的标准:
- 标准库级别的严谨程度，test cov 100%, mypy strict, ruff
- 2025 年最佳实践，支持版本 python 3.10+，完全是用 python 3.10+ 的最佳实践，避免所有 deprecated 的语法。
- 顶级的 DX 和用户体验
- 完善，好读，尊重 HCI 和心理学理论的文档和快速开始和 cookbook sample。

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

deepseek-ocr.rs is a Rust implementation of the DeepSeek-OCR inference stack featuring a CLI and OpenAI-compatible HTTP server. The project provides multiple OCR backends (DeepSeek-OCR, PaddleOCR-VL, DotsOCR) with support for CPU, Apple Metal, and (alpha) CUDA execution.

## Common Commands

### Building

**Always use `--release` for builds** - debug builds are 10x slower and unusable for inference workloads.

```bash
# Build CLI for CPU
cargo build --release -p deepseek-ocr-cli

# Build CLI with Metal (macOS only)
cargo build --release -p deepseek-ocr-cli --features metal,accelerate

# Build CLI with CUDA (Linux/Windows, alpha)
cargo build --release -p deepseek-ocr-cli --features cuda

# Build server for CPU
cargo build --release -p deepseek-ocr-server

# Build server with Metal (macOS only)
cargo build --release -p deepseek-ocr-server --features metal,accelerate

# Build entire workspace
cargo build --release --workspace
```

### Running

```bash
# Run CLI (always use --release)
cargo run -p deepseek-ocr-cli --release -- \
  --prompt "<image>\n<|grounding|>Convert this to markdown." \
  --image baselines/sample/images/test.png \
  --device cpu --max-new-tokens 512

# Run server (always use --release)
cargo run -p deepseek-ocr-server --release -- \
  --host 0.0.0.0 --port 8000 \
  --device cpu --max-new-tokens 512

# Run with Metal acceleration (macOS)
cargo run -p deepseek-ocr-cli --release --features metal,accelerate -- \
  --device metal --dtype f16 \
  --prompt "<image> Extract text" --image test.png

# Run with different model
cargo run -p deepseek-ocr-cli --release -- \
  --model paddleocr-vl \
  --prompt "<image> Extract text" --image test.png
```

### Testing

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p deepseek-ocr-core

# Run tests with specific features
cargo test -p deepseek-ocr-py --features mock-engine
```

### Python Binding Development

```bash
cd bindings/python

# Sync dependencies
uv sync --dev

# Build extension with mock engine (faster, no weights needed)
uv run maturin develop --locked --features mock-engine

# Build extension with real engines
uv run maturin develop --locked --no-default-features

# Run tests
uv run pytest

# Run type checker
uv run mypy src tests

# Run both tests and type checking
uv run pytest && uv run mypy src tests

# Build wheels for release
uv run maturin build --release
```

### DSQ (Quantization) Tools

```bash
# Run quantization CLI
cargo run -p deepseek-ocr-dsq-cli --release -- --help
```

## Architecture

### Workspace Structure

The project is a Cargo workspace with the following key crates:

**Core Infrastructure:**
- `crates/core` - Shared inference pipeline, model loaders, conversation templates, sampling, and streaming logic
- `crates/config` - Configuration management and TOML handling
- `crates/assets` - Asset management with dual-source (Hugging Face + ModelScope) download, caching, and verification

**Model Inference Engines:**
- `crates/infer-deepseek` - DeepSeek-OCR implementation (SAM+CLIP vision tower + MoE DeepSeek-V2 decoder)
- `crates/infer-paddleocr` - PaddleOCR-VL implementation (SigLIP + Ernie decoder)
- `crates/infer-dots` - DotsOCR implementation (DotsVision + Qwen2 VLM)

**User-Facing Entrypoints:**
- `crates/cli` - Command-line frontend (`deepseek-ocr-cli`)
- `crates/server` - Rocket-based HTTP server with OpenAI-compatible endpoints
- `bindings/python` - PyO3 Python bindings (`deepseek-ocr-rs` package)

**Quantization (DSQ):**
- `crates/dsq` - Core DSQ format reader/writer
- `crates/dsq-runtime` - Runtime dequantization for inference
- `crates/dsq-writer` - Quantization writer implementation
- `crates/dsq-models` - Model-specific quantization configs
- `crates/dsq-cli` - CLI tool for quantizing safetensors

### Configuration System

The CLI and server share a unified `config.toml` that is auto-generated on first launch:

**Platform-Specific Locations:**
- Linux: `~/.config/deepseek-ocr/config.toml`
- macOS: `~/Library/Application Support/deepseek-ocr/config.toml`
- Windows: `%APPDATA%\deepseek-ocr\config.toml`

**Model Cache Locations:**
- Linux: `~/.cache/deepseek-ocr/models/<id>/...`
- macOS: `~/Library/Caches/deepseek-ocr/models/<id>/...`
- Windows: `%LOCALAPPDATA%\deepseek-ocr\models\<id>\...`

**Resolution Order:**
Command-line flags → `config.toml` → built-in defaults → (server only) request payload fields

### Model System

Three base models are supported, each with quantized variants (q4k, q6k, q8k):

1. **deepseek-ocr** (default) - Highest accuracy, SAM+CLIP + MoE decoder, ~6.3GB FP16, ~13GB runtime
2. **paddleocr-vl** - Lighter 0.9B Ernie + SigLIP, ~4.7GB FP16, ~9GB runtime, best for 16GB systems
3. **dots-ocr** - DotsVision + Qwen2 VLM, ~9GB FP16, 30-50GB runtime for high-res docs

Models are specified via `--model <id>` and configured in `config.toml` under `[models.entries.<id>]`. Each entry specifies:
- `kind`: Engine type (`deepseek`, `paddle_ocr_vl`, `dots_ocr`)
- Optional `config`, `tokenizer`, `weights` paths (auto-downloaded if omitted)

### Asset Download Strategy

The `crates/assets` package implements dual-source downloads:
- Attempts Hugging Face and ModelScope in parallel
- Uses whichever responds fastest
- Handles verification and caching
- Respects `HF_TOKEN` and `HF_HOME` environment variables

### Inference Pipeline

**Shared Pipeline (in `crates/core`):**
1. **Prompt Processing** - Template rendering, tokenization, `<image>` placeholder handling
2. **Vision Encoding** - Model-specific vision tower (SAM+CLIP, SigLIP, or DotsVision)
3. **Decode Generation** - Autoregressive generation with KV-cache, sampling controls
4. **Streaming** - Optional SSE streaming for server responses

**Model-Specific Implementations:**
Each `crates/infer-*` crate implements the `OcrEngine` trait defined in `crates/core`:
- `load()` - Load model weights and config
- `encode_images()` - Vision tower processing
- `decode()` - Text generation

### Python Binding Architecture

The Python binding (`bindings/python`) is a thin PyO3 wrapper:

1. **Native Layer** (`src/lib.rs`):
   - Exposes `deepseek_ocr._native` module
   - Wraps Rust engines in thread-safe `EngineHandle`
   - Uses the same `ModelLoadArgs` as CLI/server

2. **Python Layer** (`src/deepseek_ocr/`):
   - `_api.py` - High-level API with type hints
   - `_native.pyi` - Type stubs for native module
   - Light validation and marshalling only

3. **Mock Engine Feature**:
   - Default feature for testing without downloading 6GB+ weights
   - Disable with `--no-default-features` for real inference

### Device & Acceleration

**CPU (default):**
- Works everywhere
- Optional Intel MKL support via `--features mkl` (install Intel oneMKL first)

**Metal (macOS Apple Silicon):**
- Compile with `--features metal,accelerate`
- Run with `--device metal --dtype f16`
- Best performance on M-series chips

**CUDA (Linux/Windows, alpha):**
- Compile with `--features cuda`
- Run with `--device cuda --dtype f16`
- Requires CUDA 12.2+ toolkit

### Sampling & Decoding

**Deterministic (default):**
- `do_sample=false`, `temperature=0.0`, `no_repeat_ngram_size=20`
- Reproducible outputs

**Stochastic:**
- Enable with `--do-sample true --temperature 0.8`
- Control with `--top-p`, `--top-k`, `--repetition-penalty`, `--seed`

## Development Workflow

### Adding New Model Support

1. Create new crate in `crates/infer-<model>/`
2. Implement `OcrEngine` trait from `crates/core`
3. Add model entry to `crates/config` defaults
4. Update `crates/assets` for download locations
5. Add model ID to CLI/server argument parsers
6. Document in README.md model matrix

### Modifying Inference Parameters

When adding new parameters to `DecodeParameters` or `VisionSettings` in `crates/core`:

1. Update the struct definition
2. Add CLI flags in `crates/cli/src/main.rs`
3. Add server flags in `crates/server/src/main.rs`
4. Update `config.toml` schema in `crates/config`
5. For Python bindings:
   - Add PyO3 struct field in `bindings/python/src/lib.rs`
   - Add Python dataclass field in `_api.py`
   - Update type stubs in `_native.pyi` and `__init__.pyi`

### Testing Changes

1. Compile workspace: `cargo build --release --workspace`
2. Run unit tests: `cargo test --workspace`
3. Test CLI locally with sample images from `baselines/`
4. For Python changes:
   - Test with mock engine: `cargo test -p deepseek-ocr-py --features mock-engine`
   - Rebuild extension: `cd bindings/python && uv run maturin develop --locked`
   - Run Python tests: `uv run pytest && uv run mypy src tests`

### CI/CD

**GitHub Actions workflows:**
- `.github/workflows/build-binaries.yml` - Builds macOS (with Metal) and Windows binaries
- `.github/workflows/python-binding.yml` - Tests Python binding on Python 3.10 and 3.12
- `.github/workflows/build-extended-artifacts.yml` - Extended build configurations

**Python CI:**
- Runs on `push` to `main` or PRs affecting `bindings/python/**` or `crates/**`
- Tests with mock engine only (no weight downloads)
- Validates both pytest and mypy

## Important Notes

### Performance

- **ALWAYS build with `--release`** - debug builds are too slow for any real inference work
- First request is slower (model load + GPU warmup)
- Subsequent requests benefit from KV-cache

### Memory Requirements

- DeepSeek-OCR: ~6.3GB weights, ~13GB total runtime
- PaddleOCR-VL: ~4.7GB weights, ~9GB total runtime
- DotsOCR: ~9GB weights, ~30-50GB total runtime (high-res docs)
- Quantized variants (q4k, q6k, q8k) reduce weight size

### Python Binding GIL Handling

The Python binding releases the GIL during inference to allow concurrent Python execution. Ensure all PyO3 code properly uses `py.allow_threads()` for long-running operations.

### OpenAI Compatibility

The server implements `/v1/responses`, `/v1/chat/completions`, and `/v1/models` endpoints. Multi-turn chat history is automatically collapsed to the latest user message for OCR-friendly prompts.

### Asset Downloads

- First run downloads ~6GB+ of model weights
- Set `HF_TOKEN` for Hugging Face private repos
- Set `HF_HOME` to customize cache location
- Downloads automatically choose fastest source (HF or ModelScope)

## Benchmarking & Validation

Python comparison scripts in `scripts/`:
- `python_bench.py` - Benchmark Rust CLI against Python reference
- `capture_baseline.py` - Capture intermediate tensors for validation
- `compare_bench.py` - Compare benchmark results
- `paddleocr_vl_fixture.py` - Generate test fixtures for PaddleOCR-VL

Test fixtures and baselines in `baselines/` directory.
