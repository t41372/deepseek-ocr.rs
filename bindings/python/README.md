# deepseek-ocr.rs Python Binding 🚀

![Python >=3.10](https://img.shields.io/badge/python->=3.10-blue?logo=python&logoColor=white)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

Python binding for deepseek-ocr.rs: call the Rust implementation of the DeepSeek-OCR inference stack in Python code.

## Installation

This package is distributed as source code. You must have the **[Rust toolchain](https://rustup.rs/)** installed to build it.

### Quick Start - CPU Installation

```bash
# Install directly from GitHub (CPU-only, works everywhere)
pip install "git+https://github.com/TimmyOVO/deepseek-ocr.rs.git#subdirectory=bindings/python"

# Or with uv (faster)
uv add "git+https://github.com/TimmyOVO/deepseek-ocr.rs.git#subdirectory=bindings/python"
```

### GPU Acceleration (Metal/CUDA)

For GPU-accelerated builds, use the `MATURIN_FEATURES` environment variable to enable hardware-specific features:

```bash
# Example: macOS Metal + Accelerate
MATURIN_NO_DEFAULT_FEATURES=1 \
MATURIN_FEATURES="deepseek-ocr-core/metal,deepseek-ocr-core/accelerate,deepseek-ocr-infer-deepseek/metal,deepseek-ocr-infer-deepseek/accelerate,deepseek-ocr-infer-paddleocr/metal,deepseek-ocr-infer-paddleocr/accelerate,deepseek-ocr-infer-dots/metal,deepseek-ocr-infer-dots/accelerate" \
pip install "git+https://github.com/TimmyOVO/deepseek-ocr.rs.git#subdirectory=bindings/python"

# Example: NVIDIA CUDA (Linux/Windows, requires CUDA 12.x)
MATURIN_NO_DEFAULT_FEATURES=1 \
MATURIN_FEATURES="deepseek-ocr-core/cuda,deepseek-ocr-infer-deepseek/cuda,deepseek-ocr-infer-paddleocr/cuda,deepseek-ocr-infer-dots/cuda" \
pip install "git+https://github.com/TimmyOVO/deepseek-ocr.rs.git#subdirectory=bindings/python"
```

#### How Build Features Work

The `MATURIN_FEATURES` environment variable passes Cargo features directly to the Rust build system:
- Features are specified as `<crate-name>/<feature-name>` (e.g., `deepseek-ocr-core/metal`)
- Enable features for each OCR engine crate you want to accelerate: `deepseek-ocr-infer-{deepseek,paddleocr,dots}`
- Common hardware features: `metal` (macOS GPU), `cuda` (NVIDIA GPU), `accelerate` (macOS CPU BLAS), `mkl` (Intel CPU)

#### Determining Which Features to Enable

**The authoritative source for all available features is the Rust codebase:**

1. **Check [`crates/cli/Cargo.toml`](../../crates/cli/Cargo.toml) `[features]` section** - This shows all available feature combinations
   - Example: `metal = ["deepseek-ocr-core/metal", "deepseek-ocr-infer-deepseek/metal", ...]`
   - Copy the pattern: List all crates for the feature you want to enable

2. **Reference the main project documentation:**
   - 📖 [Main README - GPU Acceleration](../../README.md) - Hardware support overview

**Why look at Cargo.toml?**
- The `[features]` section is the **source of truth** for what features exist and what they enable
- When new OCR engines or acceleration backends are added, they appear there first
- Python binding uses the exact same features as the Rust CLI/Server

**Example workflow:**
```bash
# 1. Check what features are available
cat ../../crates/cli/Cargo.toml | grep -A 5 "^metal ="

# Output shows:
# metal = [
#     "deepseek-ocr-core/metal",
#     "deepseek-ocr-infer-deepseek/metal",
#     "deepseek-ocr-infer-paddleocr/metal",
#     "deepseek-ocr-infer-dots/metal",
# ]

# 2. Use those exact features with MATURIN_FEATURES
MATURIN_FEATURES="deepseek-ocr-core/metal,deepseek-ocr-infer-deepseek/metal,deepseek-ocr-infer-paddleocr/metal,deepseek-ocr-infer-dots/metal"
```

### For Developers
For contributing to the project or local development.

```bash
# Clone the repository
git clone https://github.com/TimmyOVO/deepseek-ocr.rs.git
cd deepseek-ocr.rs/bindings/python

# Install dependencies with uv
uv sync --dev

# Fast development mode (mock engine, no model weights needed)
uv run maturin develop --locked --features mock-engine

# Or build with real engines (will download model weights on first use)
uv run maturin develop --locked --no-default-features
```

## 🚀 Quick Start - Your First OCR in 30 Seconds

**Step 1:** Create a simple Python script:

```python
# my_first_ocr.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine

# Load the model (automatically downloads ~6.3GB weights, requires ~13GB RAM at runtime)
# This uses DeepSeek-OCR, the most accurate model
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

# Open your image (replace with your own image path)
image = Image.open("bindings/python/tests/assets/sample.jpg")

# Extract text - it's that simple!
result = engine.generate(
    prompt="<image> Extract all text from this image",
    images=[image]
)

print(result.text)
```

**Step 2:** Run it:

```bash
python my_first_ocr.py
```

**That's it!** The model will download automatically on first run and then extract text from your image.

## 📚 What Just Happened?

Let's break down the code:

1. **`OcrEngine.from_pretrained(model_id="deepseek-ocr")`** - Loads the OCR model
   - Downloads model files automatically if not cached (~6GB, one-time)
   - Caches models in a system directory for reuse
   - Choose from: `"deepseek-ocr"` (best quality), `"paddleocr-vl"` (lighter), or `"dots-ocr"` (complex layouts)

2. **`Image.open(...)`** - Loads your image
   - Supports PNG, JPG, TIFF, and other common formats
   - Can also pass file paths directly to `generate()` instead of PIL Image objects

3. **`engine.generate(prompt="<image> ...", images=[...])`** - Performs OCR
   - The `<image>` token tells the model where to insert the image
   - You can customize the prompt for different tasks (see [Examples](#examples) below)
   - Returns a result object with `.text` containing the extracted text

## 💾 Where Are Models Stored?

Models are cached automatically in platform-specific locations:

| Platform | Cache Directory |
|----------|----------------|
| Linux    | `~/.cache/deepseek-ocr/models/<model-id>` |
| macOS    | `~/Library/Caches/deepseek-ocr/models/<model-id>` |
| Windows  | `%LOCALAPPDATA%\deepseek-ocr\models\<model-id>` |

**Tip:** Set `HF_TOKEN` environment variable if you need access to private model repositories on Hugging Face.

## 🎯 Examples - Different Use Cases

### Example 1: Extract Text with Custom Formatting

```python
# extract_formatted.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine, GenerationConfig

# Load the model
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

# Open your image - replace this with your own image!
image = Image.open("bindings/python/tests/assets/sample.jpg")

# Customize the prompt to get Markdown output
result = engine.generate(
    prompt="<image>\n<|grounding|>Convert this document to Markdown format, preserving structure and formatting.",
    images=[image],
    # Control generation behavior (using defaults for deterministic output)
    generation=GenerationConfig(
        max_new_tokens=512,      # Maximum length of output (increase for longer documents)
        temperature=0.0,         # 0.0 = deterministic, >0 for creative variation
        do_sample=False,         # False = greedy decoding, True = sampling
    )
)

print(result.text)
```

### Example 2: Process Receipt with Vision Settings

```python
# process_receipt.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine, GenerationConfig, VisionConfig

# Load the model
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

# Open receipt image - replace with your receipt!
receipt = Image.open("bindings/python/tests/assets/sample.jpg")

# Extract structured data from receipt
result = engine.generate(
    prompt="<image> Extract items, prices, and total from this receipt. Format as a table.",
    images=[receipt],
    generation=GenerationConfig(
        max_new_tokens=512,
        temperature=0.0,        # Deterministic for accurate extraction
    ),
    # Optionally adjust vision settings (these are the defaults, shown for reference)
    vision=VisionConfig(
        base_size=1024,         # Base resolution (increase to 1536 for more detail, slower)
        image_size=640,         # Crop size (increase to 1024 for higher quality)
        crop_mode=True,         # Dynamic crop mode (set False for padding mode)
    )
)

print(result.text)
```

### Example 3: GPU Acceleration (Metal on macOS)

```python
# gpu_ocr.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine

# Use GPU for faster processing
# Note: Requires building with --features metal (macOS) or cuda (Linux/Windows)
engine = OcrEngine.from_pretrained(
    model_id="deepseek-ocr",
    device="metal",             # Use "cuda" on Linux/Windows, "cpu" for CPU-only
    dtype="f16",               # f16 uses less memory on GPU, f32 for CPU
)

# Open your image - replace with your document!
image = Image.open("bindings/python/tests/assets/sample.jpg")

# Process with GPU acceleration (uses default generation settings for deterministic output)
result = engine.generate(
    prompt="<image> Extract all text",
    images=[image],
)

print(result.text)
```

### Example 4: Streaming Output (See Progress in Real-Time)

```python
# streaming_ocr.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine, GenerationConfig

# Load model
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

# Open your image
image = Image.open("bindings/python/tests/assets/sample.jpg")

# Track streaming progress
token_count = 0
def on_token_generated(count: int, token_ids: tuple[int, ...]) -> None:
    """
    Callback function called for each generated token.

    Args:
        count: Total number of tokens generated so far
        token_ids: Tuple of token IDs in this batch (usually just 1)
    """
    global token_count
    token_count = count
    print(f"\rGenerated {count} tokens...", end="", flush=True)

# Generate with streaming
result = engine.generate(
    prompt="<image> Convert to Markdown",
    images=[image],
    generation=GenerationConfig(
        max_new_tokens=512,
        temperature=0.0,
    ),
    stream=on_token_generated  # Pass callback function to get real-time updates
)

print(f"\n\nFinal output ({token_count} tokens):\n{result.text}")
```

### Example 5: Stochastic Sampling (Creative/Variable Output)

```python
# stochastic_sampling.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine, GenerationConfig

# Load model
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

# Open your image
image = Image.open("bindings/python/tests/assets/sample.jpg")

# Generate with stochastic sampling for creative/variable output
result = engine.generate(
    prompt="<image> Describe this image creatively",
    images=[image],
    generation=GenerationConfig(
        max_new_tokens=512,
        do_sample=True,          # Enable stochastic sampling
        temperature=0.7,         # Controls randomness (0.1=focused, 1.0=creative)
        top_p=0.9,              # Nucleus sampling: consider top 90% probability mass
        top_k=50,               # Top-k sampling: consider only top 50 tokens
        repetition_penalty=1.1,  # Penalize repetition (>1.0 = less repetition)
        seed=42,                # Set seed for reproducible stochastic generation
    )
)

print(result.text)
```

### Example 6: Choosing Different Models

```python
# model_comparison.py
from PIL import Image
from deepseek_ocr_rs import OcrEngine

# Open your image
image = Image.open("bindings/python/tests/assets/sample.jpg")

# Model 1: DeepSeek-OCR (best accuracy, ~13GB runtime memory)
deepseek_engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")
result1 = deepseek_engine.generate(prompt="<image> Extract text", images=[image])
print("DeepSeek-OCR:", result1.text[:100], "...")

# Model 2: PaddleOCR-VL (lighter, ~9GB runtime memory, good for 16GB RAM systems)
paddle_engine = OcrEngine.from_pretrained(model_id="paddleocr-vl")
result2 = paddle_engine.generate(prompt="<image> Extract text", images=[image])
print("PaddleOCR-VL:", result2.text[:100], "...")

# Model 3: DotsOCR (best for complex layouts, ~30-50GB runtime memory)
# Uncomment if you have enough RAM:
# dots_engine = OcrEngine.from_pretrained(model_id="dots-ocr")
# result3 = dots_engine.generate(prompt="<image> Extract text", images=[image])
# print("DotsOCR:", result3.text[:100], "...")
```

### Example 7: Advanced - Custom Model Paths (Air-Gapped Environments)


```python
# custom_paths.py
from pathlib import Path
from PIL import Image
from deepseek_ocr_rs import OcrEngine, get_default_cache_dir

# First, check where models are cached by default
cache_dir = get_default_cache_dir("deepseek-ocr")
print(f"Default cache location: {cache_dir}")

# Load from custom paths (useful for air-gapped systems or custom model locations)
engine = OcrEngine.from_files(
    engine="deepseek",                           # Engine type: "deepseek", "paddle", or "dots"
    config_path="/custom/path/config.json",      # Path to model config
    tokenizer_path="/custom/path/tokenizer.json", # Path to tokenizer
    weights_path="/custom/path/model.safetensors", # Path to model weights
    device="cpu",                                # Device: "cpu", "cuda", "metal"
    dtype="f32",                                 # Data type: "f32", "f16", "bf16"
    template="plain",                            # Template: "plain", "deepseek", "deepseekv2", "alignment"
)

# Use the engine normally
image = Image.open("bindings/python/tests/assets/sample.jpg")
result = engine.generate(prompt="<image> Extract text", images=[image])
print(result.text)
```

## 🧠 Which Model Should I Use?

For detailed model specifications, memory requirements, and hardware recommendations, see the **[Model Comparison Table](../../README.md#choosing-a-model-)** in the root README.

**Quick recommendations:**
- **16GB RAM systems**: Use `paddleocr-vl` (lighter, fast, good quality)
- **32GB+ RAM systems**: Use `deepseek-ocr` (highest accuracy)
- **Complex layouts/tables**: Consider `dots-ocr` if you have 64GB+ RAM

### Hardware Acceleration Support

Refer to the main [README](../../README.md) for models available and their hardware acceleration support.

## 🔄 Concurrency - Processing Multiple Images

The `OcrEngine` is thread-safe and releases Python's GIL during inference, so you can process multiple images concurrently:

```python
# concurrent_processing.py
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from deepseek_ocr_rs import OcrEngine

# Create one engine instance (shared across threads)
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

# List of images to process - replace with your images!
image_paths = [
    "bindings/python/tests/assets/sample.jpg",
    "bindings/python/tests/assets/sample.jpg",  # Using same for demo
    "bindings/python/tests/assets/sample.jpg",
]

def process_image(image_path: str) -> str:
    """Process a single image and return extracted text."""
    image = Image.open(image_path)
    result = engine.generate(
        prompt="<image> Extract all text",
        images=[image]
    )
    return result.text

# Process images concurrently (but they'll be serialized internally due to mutex)
# Note: For true parallel processing, create multiple engine instances
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_image, image_paths))

for i, text in enumerate(results, 1):
    print(f"Image {i}: {text[:50]}...")
```

**Note:** Inference calls are serialized per engine instance (one at a time). For true parallel processing, create multiple `OcrEngine` instances.

## 📖 More Examples

Check out these additional resources:

- **[Example Scripts](examples/)** - Ready-to-run examples demonstrating each model:
  - [`deepseek_example.py`](examples/deepseek_example.py) - DeepSeek-OCR with streaming support (~13GB RAM)
  - [`paddle_example.py`](examples/paddle_example.py) - PaddleOCR-VL for faster processing (~9GB RAM)
  - [`dots_example.py`](examples/dots_example.py) - DotsOCR for complex layouts with multi-image support (~30-50GB RAM)
- **[Full API Documentation](docs/python-binding.md)** - Complete technical reference

## 🧪 Testing

For developers working on this package:

```bash
# Run tests with mock engine (fast, no model download needed)
uv run pytest --cov=deepseek_ocr_rs --cov-report=term-missing

# Type checking
uv run mypy src tests

# Run both
uv run pytest --cov=deepseek_ocr_rs --cov-report=term-missing && uv run mypy src tests
```

For end-to-end tests with real models (requires downloading models):

```bash
uv run python -m deepseek_ocr_rs.download --model deepseek-ocr
DEEPSEEK_OCR_E2E=1 DEEPSEEK_OCR_E2E_MODEL_HOME=$(uv run python -m deepseek_ocr_rs.download --model deepseek-ocr --print-cache) uv run pytest -m e2e
```

## 🤝 Contributing

Contributions are welcome! The python binding follows these standards:

- ✅ Type checked with mypy (strict mode)
- ✅ Linted with ruff
- ✅ Python 3.10+ best practices

## 📝 License

See the root repository for license information.

---

**Need help?** Check the [Full Documentation](docs/python-binding.md) or open an issue on GitHub.
