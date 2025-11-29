# Python binding cookbook

Practical recipes you can copy into your own projects.

## How to load engines

All three real engines share the **same Python API**:

- `model_id="deepseek-ocr"`  → DeepSeek-OCR
- `model_id="paddleocr-vl"`  → PaddleOCR-VL
- `model_id="dots-ocr"`      → DotsOCR

You can load them in two ways:

```python
from deepseek_ocr import OcrEngine

# 1) Auto-download into the standard cache (recommended for most users).
# This works for ALL supported models.
deepseek = OcrEngine.from_pretrained(model_id="deepseek-ocr")
paddle = OcrEngine.from_pretrained(model_id="paddleocr-vl")
dots   = OcrEngine.from_pretrained(model_id="dots-ocr")

# 2) Manual paths (air-gapped / custom cache layouts).
# See "Advanced: Manual Path Control" below.
```

All recipes below work with **any** backend – they just pick one that best demonstrates the use case.
Feel free to swap `model_id=` between `"deepseek-ocr"`, `"paddleocr-vl"`, and `"dots-ocr"`.

## DeepSeek-OCR – receipt to Markdown with streaming (auto-download)

```python
import os
from PIL import Image
from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig

image = Image.open("receipt.png")

# Easiest path: pull assets if missing, keep cache in sync with the Rust CLI.
# Note: device/dtype are automatically picked up from DEEPSEEK_OCR_DEVICE / DEEPSEEK_OCR_DTYPE env vars if set.
engine = OcrEngine.from_pretrained(
    model_id="deepseek-ocr",
    template="plain",
    system_prompt="You are an OCR model that returns clean Markdown.",
)

tokens: list[int] = []
def on_stream(count: int, ids):  # ids is a sequence of token IDs
    if ids:
        tokens.append(ids[-1])

result = engine.generate(
    prompt="<image>\n<|grounding|>Convert this receipt to Markdown with totals.",
    images=[image],
    generation=GenerationConfig(max_new_tokens=512, temperature=0.0),
    vision=VisionConfig(base_size=1536, image_size=1024, crop_mode=False),
    stream=on_stream,
)
print(result.text)
print(f"streamed {len(tokens)} tokens")
```

## PaddleOCR-VL – fast pass on CPU or low-memory machines

```python
import os
from PIL import Image
from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig

image = Image.open("form.png")

# Works exactly the same way!
engine = OcrEngine.from_pretrained(model_id="paddleocr-vl")

result = engine.generate(
    prompt="<image> Extract the key-value pairs as YAML.",
    images=[image],
    generation=GenerationConfig(max_new_tokens=256, temperature=0.0),
    vision=VisionConfig(base_size=1024, image_size=768, crop_mode=True),
)
print(result.text)
```

## DotsOCR – layout-heavy pages to Markdown (multi-image)

```python
import os
from PIL import Image
from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig

images = [Image.open("page1.png"), Image.open("page2.png")]

engine = OcrEngine.from_pretrained(
    model_id="dots-ocr",
    template="markdown",
)

prompt = "<image> Page 1\n<image> Page 2\nCombine into Markdown preserving headings and tables."
result = engine.generate(
    prompt=prompt,
    images=images,
    generation=GenerationConfig(max_new_tokens=1024, temperature=0.0),
    vision=VisionConfig(base_size=1536, image_size=1280, crop_mode=True),
)
print(result.text)
```

## Batch concurrent decode with one shared engine

```python
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from deepseek_ocr import OcrEngine

engine = OcrEngine.from_pretrained(model_id="deepseek-ocr")

images = [Image.open(p) for p in ["doc1.png", "doc2.png", "doc3.png"]]

def run(prompt: str, image):
    return engine.generate(prompt=prompt, images=[image]).text

with ThreadPoolExecutor(max_workers=3) as pool:
    # The engine handle is thread-safe and shares the underlying model state
    outputs = list(pool.map(run, ["<image> OCR this"] * len(images), images))

for idx, text in enumerate(outputs, 1):
    print(f"Doc {idx}: {text[:80]}…")
```

## Advanced: Manual Path Control

If you need to load models from a custom location (e.g. air-gapped environment) without auto-download logic:

```python
from deepseek_ocr import OcrEngine

# You can use the helper to get the default cache path if you just want to inspect it
from deepseek_ocr import get_default_cache_dir
print(f"Default cache for deepseek: {get_default_cache_dir('deepseek-ocr')}")

# Or specify paths manually
engine = OcrEngine.from_files(
    engine="deepseek", # "deepseek", "paddle", or "dots"
    config_path="/path/to/config.json",
    tokenizer_path="/path/to/tokenizer.json",
    weights_path="/path/to/model.safetensors",
    device="cpu",
)
```
