# DeepSeek OCR Python Binding - Technical Reference

This document provides technical details about the Python binding architecture and API.

For quick start and examples, see:
- **[README.md](../README.md)** - Quick start guide with examples
- **[examples/](../examples/)** - Ready-to-run example scripts

## Quick Setup

```bash
# For users - install from GitHub - reference bindings/python/README.md for more details
pip install git+https://github.com/TimmyOVO/deepseek-ocr.rs.git#subdirectory=bindings/python

# For developers - build from source
cd bindings/python
uv sync --dev
uv run maturin develop --locked --features mock-engine  # Mock (fast, but it's fake)
uv run maturin develop --locked --no-default-features   # Real engines
```

## Model Files and Cache

Model files are automatically downloaded to platform-specific cache directories:

| Platform | Cache Location |
|----------|---------------|
| Linux    | `~/.cache/deepseek-ocr/models/<model-id>` |
| macOS    | `~/Library/Caches/deepseek-ocr/models/<model-id>` |
| Windows  | `%LOCALAPPDATA%\deepseek-ocr\models\<model-id>` |

Each model requires:
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer data
- Model weights (varies by model: `.safetensors` files)

## API Reference

### Core Classes

**`OcrEngine`** - Main OCR inference engine

Methods:
- `from_pretrained(model_id="deepseek-ocr", device=None, dtype=None, template="plain", system_prompt="", cache_dir=None)` - Load model with auto-download
- `from_files(model_id=None, engine="deepseek", config_path=None, tokenizer_path=None, weights_path=None, snapshot_path=None, device="cpu", dtype=None, template="plain", system_prompt="", auto_download=False, cache_dir=None)` - Load from custom paths
- `generate(prompt, images, generation=None, vision=None, stream=None)` - Run OCR inference

**`GenerationConfig`** - Controls text generation behavior

Parameters:
- `max_new_tokens: int = 512` - Maximum output length
- `temperature: float = 0.0` - Sampling temperature (0.0 = deterministic)
- `do_sample: bool = False` - Enable sampling vs greedy decoding
- `top_p: float | None` - Nucleus sampling threshold
- `top_k: int | None` - Top-k sampling limit
- `repetition_penalty: float = 1.0` - Penalty for repetition
- `no_repeat_ngram_size: int | None` - Prevent n-gram repetition
- `use_cache: bool = True` - Enable KV-cache
- `seed: int | None` - Random seed for reproducibility

**`VisionConfig`** - Controls image processing

Parameters:
- `base_size: int = 1536` - Maximum base dimension
- `image_size: int = 1024` - Target patch size
- `crop_mode: bool = False` - Use cropping vs padding

**`DecodeResult`** - OCR output

Attributes:
- `text: str` - Extracted text
- `prompt_tokens: int` - Input token count
- `response_tokens: int` - Generated token count
- `generated_tokens: tuple[int, ...]` - Token IDs

### Helper Functions

- `get_default_cache_dir(model_id: str) -> Path` - Get cache directory for model
- `render_prompt(template: str, system_prompt: str, raw_prompt: str) -> str` - Apply prompt template
- `normalize_text(text: str) -> str` - Clean output text
- `download_model(model_id: str, cache_dir: str | Path | None = None) -> DownloadResultHandle` - Download model assets

### Type Aliases

```python
ModelKind = Literal["deepseek", "paddle", "dots", "mock"]
DeviceLiteral = Literal["cpu", "cuda", "metal"]
DTypeLiteral = Literal["f32", "f16", "bf16"]
StreamCallback = Callable[[int, Sequence[int]], None]
```

## Key Concepts

### Image Inputs

Accepts multiple input formats:
- `PIL.Image` objects (converted to RGB internally)
- File paths as `str` or `Path`
- Raw `bytes` or `bytearray`

**Important:** Number of `<image>` tokens in prompt must match number of images in the `images` list.

### Prompting

```python
# Single image
result = engine.generate(
    prompt="<image> Extract text",
    images=[image1]
)

# Multiple images
result = engine.generate(
    prompt="<image> <image> Extract text from both pages",
    images=[image1, image2]
)
```

### Conversation Templates

The `template` parameter controls how prompts are formatted before being sent to the model. Available options:

- **`"plain"`** (default) - No special formatting, prompt is passed directly to the model
- **`"deepseek"`** - DeepSeek conversation format with `<|User|>:` and `<|Assistant|>:` roles
- **`"deepseekv2"`** - DeepSeek V2 specific format
- **`"alignment"`** - Specialized format for alignment tasks

**For most OCR use cases, use the default `"plain"` template.** Other templates are primarily for specific model training formats.

```python
# Default - recommended for OCR
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr", template="plain")

# DeepSeek conversation format (if needed for specific use cases)
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr", template="deepseek")
```

**Note:** The `template` parameter controls conversation format, not output format. To get markdown or other formatted output, specify it in your prompt:

```python
result = engine.generate(
    prompt="<image> Convert this document to Markdown format",
    images=[image]
)
```

### Streaming

Pass a callback function to receive tokens as they're generated:

```python
def callback(count: int, token_ids: Sequence[int]) -> None:
    print(f"Generated {count} tokens...")

result = engine.generate(..., stream=callback)
```

### Thread Safety

- `OcrEngine` is thread-safe (protected by internal mutex)
- Releases Python GIL during inference
- Decode calls are serialized per engine instance
- For parallel processing, create multiple engine instances

### Device Support

| Device | Description | dtype |
|--------|-------------|-------|
| `cpu` | CPU-only (works everywhere) | `f32` |
| `metal` | macOS Apple Silicon GPU | `f16` |
| `cuda` | NVIDIA GPU (CUDA 12.2+) | `f16` |

**Note:** GPU support requires building from source with appropriate features. See [README.md](../README.md#gpu-installation).

## Model Comparison

See the root README for comparison table with detailed memory requirements, hardware recommendations, and use case guidance.

## Examples

See the [examples/](../examples/) directory for ready-to-run code:
- [`deepseek_example.py`](../examples/deepseek_example.py) - Highest accuracy model with streaming
- [`paddle_example.py`](../examples/paddle_example.py) - Lighter, faster processing
- [`dots_example.py`](../examples/dots_example.py) - Complex layouts and multi-image support

Also check [README.md](../README.md) for inline examples covering GPU acceleration, custom prompts, and more.

## Testing

```bash
# Unit tests with mock engine (fast)
uv run pytest --cov=deepseek_ocr_rs --cov-report=term-missing

# Type checking
uv run mypy src tests

# End-to-end with real models (requires model download)
DEEPSEEK_OCR_E2E=1 uv run pytest -m e2e
```

## Development

### Adding New Parameters

When adding new inference parameters:

1. Update PyO3 structs in `src/lib.rs`
2. Mirror in `GenerationConfig`/`VisionConfig` in `_api.py` (use `kw_only=True` for dataclasses)
3. Update `.pyi` type stubs
4. Update documentation (README.md, this file)
5. Add tests

### Adding New Models

When adding support for a new model:

1. Add to Rust asset registry first
2. Update `ModelKind` in `_api.py`
3. Update `EngineKind` in `src/lib.rs`
4. Document in README.md and add example to `examples/`
5. Add test cases

## Architecture

The Python binding is a thin wrapper around Rust:

- **Heavy work in Rust:** Model loading, inference, tensor operations
- **Python layer:** Type-safe API, convenience functions
- **GIL handling:** Released during inference for concurrency
- **Thread safety:** Mutex protection around engine state

All model metadata comes from the Rust asset registry, ensuring single source of truth across CLI, server, and Python binding.

---

For practical examples and quick start, see:
- **[README.md](../README.md)** - Installation and quick start
- **[examples/](../examples/)** - Ready-to-run example scripts
