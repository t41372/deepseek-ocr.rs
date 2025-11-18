# DeepSeek OCR - Python Bindings

High-performance Python bindings for the [deepseek-ocr.rs](https://github.com/t41372/deepseek-ocr.rs) Rust library, providing fast and accurate optical character recognition using state-of-the-art models.

## Features

- 🚀 **High Performance**: Native Rust implementation with Python bindings via PyO3
- 🎯 **Multiple Models**: Support for DeepSeek-OCR, PaddleOCR-VL, and DotsOCR
- ⚡ **Hardware Acceleration**: CPU, CUDA (NVIDIA GPU), and Metal (Apple Silicon) support
- 📦 **Quantization**: Q4K, Q6K, Q8K quantized models for reduced memory usage
- 🔒 **Type Safe**: Full type annotations with strict mypy compliance
- 🐍 **Modern Python**: Python 3.10+ with best practices (pyproject.toml, type hints)

## Installation

### Prerequisites

- Python >= 3.10
- Rust >= 1.70 (for building from source)

### Install from PyPI (when published)

```bash
pip install deepseek-ocr
```

### Install from source

```bash
# Install uv (recommended) or pip-tools
pip install uv

# Clone the repository
git clone https://github.com/t41372/deepseek-ocr.rs
cd deepseek-ocr.rs/crates/python-bindings

# Install with uv
uv pip install -e .

# Or with maturin directly
pip install maturin
maturin develop --release
```

### Platform-specific installation

#### macOS (Metal acceleration)

```bash
maturin develop --release --features metal
```

#### Linux/Windows (CUDA acceleration)

```bash
maturin develop --release --features cuda
```

#### Intel CPU (MKL acceleration)

```bash
maturin develop --release --features mkl
```

## Quick Start

### Basic OCR

```python
from deepseek_ocr import OcrModel, Device

# Load model
model = OcrModel("deepseek-ocr", device=Device.Cpu)

# Perform OCR
result = model.infer(
    prompt="<image> Extract all text from this image",
    images=["path/to/image.png"]
)

print(result.text)
```

### Advanced Configuration

```python
from deepseek_ocr import (
    OcrModel,
    Device,
    Precision,
    VisionConfig,
    DecodeParams
)

# Load model with custom configuration
model = OcrModel(
    model_name="deepseek-ocr-q6k",  # Quantized model
    device=Device.Cuda,  # Use NVIDIA GPU
    precision=Precision.F16,  # 16-bit precision
)

# Configure vision preprocessing
vision_config = VisionConfig(
    base_size=1024,  # Base resolution
    image_size=640,  # Target size
    crop_mode=True,  # Enable cropping
)

# Configure text generation
decode_params = DecodeParams(
    max_new_tokens=1024,  # Max output length
    do_sample=False,  # Greedy decoding
    temperature=0.0,
    use_cache=True,  # Enable KV cache
)

# Run OCR
result = model.infer(
    prompt="<image> Extract text and convert to markdown",
    images=["document.png"],
    vision_config=vision_config,
    decode_params=decode_params,
)

print(f"Extracted text ({result.response_tokens} tokens):")
print(result.text)
```

### Multi-Image OCR

```python
from pathlib import Path
from deepseek_ocr import OcrModel, Device

model = OcrModel("paddleocr-vl", device=Device.Cpu)

# Process multiple images at once
result = model.infer(
    prompt="""<image>
Page 1 content above.

<image>
Page 2 content above.

Summarize both pages.""",
    images=[Path("page1.png"), Path("page2.png")],
)

print(result.text)
```

## Available Models

### DeepSeek-OCR

High-quality OCR with MoE architecture (~3B params, ~570M active):

- `deepseek-ocr` - Full precision (FP16, ~6.3GB weights)
- `deepseek-ocr-q4k` - 4-bit quantization (smallest size)
- `deepseek-ocr-q6k` - 6-bit quantization (recommended)
- `deepseek-ocr-q8k` - 8-bit quantization (near full quality)

### PaddleOCR-VL

Lightweight OCR model (~0.9B params, ~4.7GB weights):

- `paddleocr-vl` - Full precision
- `paddleocr-vl-q4k` - 4-bit quantization
- `paddleocr-vl-q6k` - 6-bit quantization (recommended)
- `paddleocr-vl-q8k` - 8-bit quantization

### DotsOCR

High-accuracy layout and multi-language OCR (~9GB weights):

- `dots-ocr` - Full precision (FP16/BF16)
- `dots-ocr-q4k` - 4-bit quantization
- `dots-ocr-q6k` - 6-bit quantization (recommended)
- `dots-ocr-q8k` - 8-bit quantization

**List available models:**

```python
from deepseek_ocr import list_available_models

print(list_available_models())
```

## API Reference

### Classes

#### `OcrModel`

Main class for loading and running OCR models.

**Constructor:**

```python
OcrModel(
    model_name: str,
    device: Device = Device.Cpu,
    precision: Precision = Precision.Auto,
    config_path: Optional[Path] = None,
    weights_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
)
```

**Methods:**

- `infer(prompt, images, vision_config=None, decode_params=None, template=None, system_prompt=None) -> OcrResult`
  - Perform OCR on image(s) with text prompt
- `info() -> str`
  - Get model information string

#### `OcrResult`

OCR inference result.

**Attributes:**

- `text: str` - Extracted text
- `prompt_tokens: int` - Number of prompt tokens
- `response_tokens: int` - Number of generated tokens

#### `Device`

Compute device enumeration.

**Values:**

- `Device.Cpu` - CPU device
- `Device.Cuda` - CUDA device (NVIDIA GPU)
- `Device.Metal` - Metal device (Apple Silicon)

#### `Precision`

Model precision/dtype enumeration.

**Values:**

- `Precision.F16` - 16-bit floating point
- `Precision.F32` - 32-bit floating point
- `Precision.BF16` - Brain floating point 16
- `Precision.Auto` - Auto-select based on device

#### `VisionConfig`

Vision preprocessing configuration.

**Constructor:**

```python
VisionConfig(
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
)
```

#### `DecodeParams`

Text generation parameters.

**Constructor:**

```python
DecodeParams(
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: Optional[int] = None,
    seed: Optional[int] = None,
    use_cache: bool = True,
)
```

## Examples

See the [`examples/`](./examples/) directory for complete examples:

- [`basic_ocr.py`](./examples/basic_ocr.py) - Basic OCR on a single image
- [`multi_image_ocr.py`](./examples/multi_image_ocr.py) - Multi-page document processing
- [`batch_processing.py`](./examples/batch_processing.py) - Batch processing with JSON output
- [`model_comparison.py`](./examples/model_comparison.py) - Compare different models

## Development

### Building

```bash
cd crates/python-bindings

# Development build
maturin develop

# Release build
maturin develop --release

# With specific features
maturin develop --release --features metal  # For macOS
maturin develop --release --features cuda   # For NVIDIA GPU
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=deepseek_ocr --cov-report=html

# Type checking
mypy python/deepseek_ocr

# Linting
ruff check python/
```

### Type Checking

This package has strict type checking enabled. All public APIs have complete type annotations.

```bash
# Run mypy in strict mode
mypy python/deepseek_ocr
```

## Performance Tips

1. **Use quantized models** (Q6K recommended) for lower memory usage with minimal quality loss
2. **Enable GPU acceleration** (CUDA/Metal) for faster inference
3. **Reuse model instances** when processing multiple images to avoid reloading
4. **Enable KV cache** (`use_cache=True`) for faster text generation
5. **Choose the right model**:
   - `paddleocr-vl`: Fastest, lowest memory (good for simple documents)
   - `deepseek-ocr`: Best quality-performance balance
   - `dots-ocr`: Highest accuracy, best for complex layouts

## Troubleshooting

### Model Download Issues

Models are automatically downloaded on first use. If downloads fail:

```python
# Models are cached in:
# - Linux: ~/.cache/deepseek-ocr/
# - macOS: ~/Library/Caches/deepseek-ocr/
# - Windows: %LOCALAPPDATA%\deepseek-ocr\

# Manually download models from HuggingFace:
# https://huggingface.co/deepseek-ai/DeepSeek-VL-1.3B-OCR
```

### Memory Issues

If you encounter OOM errors:

1. Use quantized models (Q4K, Q6K)
2. Reduce `image_size` in `VisionConfig`
3. Reduce `max_new_tokens` in `DecodeParams`
4. Process images one at a time instead of batching

### Import Errors

If you see `ImportError: _core module not found`:

```bash
# Rebuild the extension module
maturin develop --release
```

## License

MIT OR Apache-2.0

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for Python binding development guidelines.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{deepseek-ocr-rs,
  title = {DeepSeek OCR Rust},
  author = {DeepSeek OCR Contributors},
  year = {2025},
  url = {https://github.com/t41372/deepseek-ocr.rs}
}
```
