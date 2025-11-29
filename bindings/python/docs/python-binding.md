# DeepSeek OCR Python binding guide

Everything you need to load models, stream tokens, and keep the Python layer in
sync with the Rust engines.

## Setup recap

```bash
uv sync --dev
# Mock engine (fast, no weights)
uv run maturin develop --locked --features mock-engine
# Real engines (provide weights)
uv run maturin develop --locked --no-default-features
# Download assets (same cache as Rust CLI/server)
uv run python -m deepseek_ocr_rs.download --model deepseek-ocr
```

Artifacts required by each model:

| Model ID        | Files (inside the model cache directory)                          |
| --------------- | ----------------------------------------------------------------- |
| deepseek-ocr    | `config.json`, `tokenizer.json`, `model-00001-of-000001.safetensors` |
| paddleocr-vl    | `config.json`, `tokenizer.json`, `model.safetensors`             |
| dots-ocr        | `config.json`, `tokenizer.json`, `model.safetensors.index.json` (+ shards), `preprocessor_config.json` |
| quantized *     | Same config/tokenizer as the baseline + `<id>.dsq` snapshot       |

Default cache roots after the CLI downloads:

- Linux: `~/.cache/deepseek-ocr/models/<model-id>`
- macOS: `~/Library/Caches/deepseek-ocr/models/<model-id>`
- Windows: `%LOCALAPPDATA%\\deepseek-ocr\\models\\<model-id>`

\* Quantized snapshots share the baseline directory.

## API surface

| Class / function | Purpose |
| --- | --- |
| `OcrEngine.from_files` | Build a thread-safe handle from explicit paths and device/dtype hints. |
| `auto_download` flag | Optional lazy download of missing assets into the standard cache. |
| `OcrEngine.from_pretrained` | Convenience wrapper that defaults to `auto_download=True` and infers engine from `model_id`. |
| `get_default_cache_dir` | Helper to inspect the default cache location for a given model ID. |
| `OcrEngine.generate` | Run inference with optional streaming, `GenerationConfig`, and `VisionConfig`. |
| `GenerationConfig` | Mirrors `DecodeParameters` in Rust (sampling, penalties, cache use). |
| `VisionConfig` | Mirrors `VisionSettings` (base size, crop mode, resize). |
| `render_prompt` | Apply the template + system prompt just like the CLI/server. |
| `normalize_text` | Strip trailing markers the engines emit. |
| `download_model` / `python -m deepseek_ocr_rs.download` | Fetch model assets and print resolved cache paths. |

All heavy work happens in Rust; the GIL is released during decode.

## Inputs and prompting

- Acceptable images: `PIL.Image`, filesystem paths (strings or `Path`), and raw
  `bytes`/`bytearray`. PIL inputs are converted to RGB PPM for fast, lossless
  transfer into Rust.
- The prompt must contain exactly one `<image>` token per image passed to
  `generate`. A mismatch raises `ValueError` before any model work begins.
- Templates are the same as the CLI (`plain` by default). Use `render_prompt`
  if you need to preview the final text before calling `generate`.

## Streaming and concurrency

- Pass `stream=callable` to `generate` to receive incremental token IDs. Errors
  inside the callback are printed but do not abort decoding.
- `OcrEngine` can be shared across threads; the binding holds a mutex around the
  engine state while keeping the GIL released so Python threads keep running.
  Decode calls are serialized by that mutex (single-flight), so threads remain
  responsive but inferences themselves do not run in parallel per engine
  instance. Create multiple engines if you need concurrent decodes.

## Device and dtype rules

- `device`: `cpu`, `metal` (macOS), `cuda` (CUDA 12.2+). The binding delegates
  to the same `prepare_device_and_dtype` helper as the Rust runtime.
- `dtype`: `f32`, `f16`, or `bf16`. When omitted we choose the default for the
  device (f32 on CPU, f16 on Metal/CUDA where available).
- Quantized snapshots still require a baseline `config.json` and `tokenizer.json`
  plus the `.dsq` file; set `weights_path` to the snapshot and leave
  `snapshot_path` unset.
- GPU builds: enable the matching Candle features when building wheels/editable
  installs so `device="cuda"/"metal"` actually works:
  - macOS Metal/Accelerate: `--features deepseek-ocr-infer-deepseek/metal,deepseek-ocr-infer-paddleocr/metal`
  - CUDA 12.x: `--features deepseek-ocr-infer-deepseek/cuda,deepseek-ocr-infer-paddleocr/cuda`
  CPU-only wheels accept the device strings but will fail at runtime; document the
  need to rebuild with GPU features for those users.

## Model quick picks

- **deepseek-ocr** – highest accuracy; ~13 GB runtime; choose Metal/CUDA when possible.
- **paddleocr-vl** – best on 16 GB systems; smaller 0.9B model; good general quality.
- **dots-ocr** – excels at complex layouts; heavy (30–50 GB runtime); consider
  quantized variants on RAM-limited hosts.

## Testing matrix

- Unit + type checks (mock engine):  
  `uv run pytest --cov=deepseek_ocr_rs --cov-report=term-missing && uv run mypy src tests`
- End-to-end with real weights (opt-in):  
  ```
  uv run python -m deepseek_ocr_rs.download --model deepseek-ocr
  DEEPSEEK_OCR_E2E=1 DEEPSEEK_OCR_E2E_MODEL_HOME=$(uv run python -m deepseek_ocr_rs.download --model deepseek-ocr --print-cache) uv run pytest -m e2e
  ```

## Forward compatibility with new Rust models

Single source of truth: model ids, engine kinds, and required files come from the
Rust asset registry (`deepseek_ocr_assets`) via `_native.engine_for_model` and
`download_model`. Python no longer hard-codes suffixes or file names.

If a new Rust release adds a model the current Python package doesn't recognize:

1. Set `DEEPSEEK_OCR_ENGINE_<MODEL_ID>` to `deepseek` / `paddle` / `dots` to force an engine.
2. If no override is set and the model isn't in the registry, the binding falls back to a
   simple heuristic (`paddle` → `paddle`, `dots` → `dots`, else `deepseek`).
3. Asset downloads always run through the Rust registry. Unknown ids will error early,
   prompting you to update the asset list in Rust.

Manual path loading remains supported via `from_files`; for quantized variants, omit
`snapshot_path` and let `download_model` place the `.dsq` beside the baseline assets.

## Maintenance checklist

When inference parameters or model options change upstream:

1. Add fields to the PyO3 structs in `src/lib.rs`.
2. Mirror them in `GenerationConfig` / `VisionConfig` in `_api.py`.
3. Update the `.pyi` stubs and the user-facing docs (README + cookbook).
4. Extend tests to cover the new knobs (use the mock engine where possible).

When new model types are added to Rust:

1. Update `ModelKind` literal in `_api.py` (line ~17).
2. Update `EngineKind` enum and `parse()` in `src/lib.rs` (lines ~367-390).
3. Add loader function call in `model_from_args()` (lines ~325-333).
4. Document the new model in README.md and this guide.
5. Add test cases for model ID inference.
