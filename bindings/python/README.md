# DeepSeek OCR Python bindings

Typed, minimal Python access to the Rust DeepSeek OCR engines (DeepSeek-OCR,
PaddleOCR-VL, DotsOCR) with the same decoding pipeline as the CLI/server.

## Install

Prereqs: Python 3.10+, Rust toolchain, Pillow, and [UV](https://github.com/astral-sh/uv).

```bash
uv sync --dev
# Fast path for development (mock engine, no weights)
uv run maturin develop --locked --features mock-engine
# Real engines (downloads/weights required)
uv run maturin develop --locked --no-default-features
# Build wheels for distribution
uv run maturin build --release

# Download model assets (paths match Rust CLI caches)
uv run python -m deepseek_ocr_rs.download --model deepseek-ocr

### GPU-enabled builds

Enable Candle features to match your platform when building wheels or editable installs:

- Apple Metal / Accelerate (macOS): `uv run maturin develop --locked --no-default-features --features deepseek-ocr-infer-deepseek/metal,deepseek-ocr-infer-paddleocr/metal`
- CUDA 12.x (Linux/Windows): `uv run maturin develop --locked --no-default-features --features deepseek-ocr-infer-deepseek/cuda,deepseek-ocr-infer-paddleocr/cuda`

If you ship CPU-only wheels, document that GPU users must rebuild from source with the flags above.
```

## Get model assets

Use the helper above or the Rust CLI to download once. Assets land in the same
cache layout as the Rust tools:

| Model ID        | Config          | Tokenizer       | Weights                                  |
| --------------- | --------------- | --------------- | ---------------------------------------- |
| deepseek-ocr    | `config.json`   | `tokenizer.json`| `model-00001-of-000001.safetensors`      |
| paddleocr-vl    | `config.json`   | `tokenizer.json`| `model.safetensors`                      |
| dots-ocr        | `config.json`   | `tokenizer.json`| `model.safetensors.index.json` (+ shards)|
| Quantized \*    | `config.json`   | `tokenizer.json`| `<id>.dsq` snapshot                      |

Default cache roots after the CLI downloads:

| OS      | Path |
| ------- | ---- |
| Linux   | `~/.cache/deepseek-ocr/models/<model-id>` |
| macOS   | `~/Library/Caches/deepseek-ocr/models/<model-id>` |
| Windows | `%LOCALAPPDATA%\\deepseek-ocr\\models\\<model-id>` |

\* Quantized snapshots live under the corresponding baseline model directory.
Set `HF_TOKEN` if you need access to private mirrors.

## Quick start (mock engine)

```python
from PIL import Image
from deepseek_ocr_rs import OcrEngine

engine = OcrEngine.from_files(engine="mock")
result = engine.generate(prompt="<image> hi", images=[Image.new("RGB", (8, 8))])
print(result.text)
```

## Quick start (real model, auto-download)

```python
from deepseek_ocr_rs import OcrEngine
from PIL import Image

# Downloads missing assets into the same cache used by the Rust CLI/server.
# Works for "deepseek-ocr", "paddleocr-vl", and "dots-ocr".
engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

img = Image.new("RGB", (8, 8), color="white")  # replace with your document
print(engine.generate(prompt="<image> Hello", images=[img]).text)
```

## Run real models (advanced configuration)

```python
import os
from pathlib import Path
from PIL import Image
from deepseek_ocr_rs import GenerationConfig, OcrEngine, VisionConfig, get_default_cache_dir

# You can inspect where assets are stored:
print(f"Cache: {get_default_cache_dir('deepseek-ocr')}")

# Or use explicit paths (useful for air-gapped environments):
engine = OcrEngine.from_files(
    engine="deepseek",                  # "deepseek", "paddle", or "dots"
    config_path="/path/to/config.json",
    tokenizer_path="/path/to/tokenizer.json",
    weights_path="/path/to/model.safetensors",
    device="cpu",                       # "cpu", "cuda", "metal"
    dtype="f32",                        # "f32", "f16", "bf16"
    template="plain",
)

image = Image.open("your-doc.png")
generation = GenerationConfig(max_new_tokens=512, temperature=0.0)
vision = VisionConfig(base_size=1536, image_size=1024, crop_mode=False)
outcome = engine.generate(prompt="<image> Extract text", images=[image], generation=generation, vision=vision)
print(outcome.text)
```

Tips:
- Match the `model` literal to the engine: `deepseek`, `paddle`, `dots`.
- Use quantized snapshots on lower-memory machines (see cache table above).
- Keep `temperature=0` for deterministic runs; enable sampling for creative tasks.
- Multi-image prompts must include one `<image>` token per image.

## Prompting, streaming, and concurrency

- Templates: `render_prompt(template, system_prompt, raw_prompt)` mirrors the CLI.
- Streaming: pass `stream=callable` and receive incremental token IDs.
- Concurrency: the binding releases the GIL during decode; a single `OcrEngine`
  is mutex-protected so decode calls are serialized. Create multiple engines for
  parallel inferences.

Example streaming callback:

```python
tokens = []
def on_stream(count: int, ids: tuple[int, ...]) -> None:
    tokens.append(ids[-1])

engine.generate(prompt="<image> stream me", images=[image], stream=on_stream)
```

## Model selection and memory hints

- **deepseek-ocr**: best accuracy; ~13 GB runtime; prefer Metal/CUDA when available.
- **paddleocr-vl**: good quality on 16 GB RAM; smaller 0.9B model.
- **dots-ocr**: excels at complex layouts; needs 30–50 GB runtime; consider
  quantized variants on limited hardware.

## Cookbook and examples

- Recipes live in `docs/cookbook.md`.
- Runnable samples under `examples/`:
  - `deepseek_receipt.py` – Markdown extraction with DeepSeek-OCR
  - `paddle_lightweight.py` – Fast pass on limited RAM
  - `dots_layout.py` – Layout-heavy documents with multi-image prompts

Run any script with `uv run python examples/<script>.py --help` for flags.

## Tests

- Fast path (mock only): `uv run pytest --cov=deepseek_ocr_rs --cov-report=term-missing`
- Opt-in end-to-end (real weights): set `DEEPSEEK_OCR_E2E=1` and point
  `DEEPSEEK_OCR_E2E_MODEL_HOME` at a cache directory, then run:
  ```
  uv run python -m deepseek_ocr_rs.download --model deepseek-ocr
  DEEPSEEK_OCR_E2E=1 DEEPSEEK_OCR_E2E_MODEL_HOME=$(uv run python -m deepseek_ocr_rs.download --model deepseek-ocr --print-cache) uv run pytest -m e2e
  ```

## Forward compatibility and single source of truth

- The binding defers model metadata to the Rust asset registry (`deepseek_ocr_assets`)
  via `_native.engine_for_model` and `download_model`, avoiding duplicated model tables.
- Environment overrides (`DEEPSEEK_OCR_ENGINE_<MODEL_ID>`) still work for brand new
  models that the binding does not yet know about.
- Quantized snapshots are detected via the Rust registry; no magic suffix matching in
  Python. If you add a model in Rust, updating the assets list keeps the Python binding
  aligned automatically.
