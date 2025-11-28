# DeepSeek OCR Python bindings

This package exposes a thin, fully typed Python interface on top of the Rust
DeepSeek OCR engines.  It ships as a standard `pyproject.toml` project named
`deepseek-ocr-rs` that can be built locally with
[maturin](https://github.com/PyO3/maturin) and managed via
[Astral UV](https://github.com/astral-sh/uv).

## Quick start

```bash
uv sync --dev
uv run maturin develop --locked
```

The default features include a tiny mock engine so you can run the test suite
or a smoke check without downloading model weights.

Once the native module is compiled you can use the friendly wrapper API:

```python
from pathlib import Path
from PIL import Image
from deepseek_ocr import (
    GenerationConfig,
    OcrEngine,
    VisionConfig,
)

engine = OcrEngine.from_files(
    config_path=Path("~/.cache/deepseek-ocr/config.json").expanduser(),
    tokenizer_path=Path("~/.cache/deepseek-ocr/tokenizer.json").expanduser(),
    weights_path=Path("~/Downloads/DeepSeek-OCR/model-00001-of-000001.safetensors")
        .expanduser(),
    model="deepseek",
)

image = Image.open("tests/assets/document.png")
prompt = "<image> Extract the table and answer the questions."
result = engine.generate(prompt=prompt, images=[image])
print(result.text)
```

For a dependency-free smoke test you can swap in the mock engine:

```python
from PIL import Image
from deepseek_ocr import OcrEngine

engine = OcrEngine.from_files(model="mock")
print(engine.generate(prompt="<image> hi", images=[Image.new("RGB", (8, 8))]).text)
```

## Package layout

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Python packaging metadata and tooling configuration (package name `deepseek-ocr-rs`). |
| `Cargo.toml` | PyO3 crate that produces the `deepseek_ocr._native` extension. |
| `src/deepseek_ocr` | The typed user-facing Python API and stubs. |
| `tests/` | Unit and end-to-end tests that exercise the binding surface. |
| `DEVELOPING.md` | Contribution guide dedicated to the Python binding. |

## Runtime requirements

* Python **3.10+**
* A Rust toolchain capable of building the workspace (stable recommended)
* Pillow for image conversion on the Python side

GPU acceleration follows the exact same rules as the Rust CLI.  As long as the
underlying system exposes CUDA or Metal to Candle, the Python wrapper can use
those devices.

## Documentation

The developer guide in `DEVELOPING.md` explains how the Python code mirrors the
Rust API and what needs to happen when the engines change.  Refer to the
repository README for dataset download instructions and CLI usage.
