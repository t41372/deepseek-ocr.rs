# Developing the Python binding

This document explains how the Python layer maps onto the Rust workspace and
what to do when engine-facing changes land.

## Architecture overview

* `bindings/python/src/lib.rs` implements a small PyO3 crate that exposes
  `deepseek_ocr._native`.
* All inference happens inside the existing Rust engines.  The binding loads a
  model via the same `ModelLoadArgs` structure used by the CLI and wraps it in
  a thread-safe `EngineHandle`.
* The Python package (`src/deepseek_ocr`) only performs light validation and
  marshals user input (images, prompts, config objects) into the native layer.

## Rebuilding after Rust changes

1. Ensure the workspace compiles via `cargo test -p deepseek-ocr-py`.
2. Rebuild the extension: `uv run maturin develop --locked`.
3. Run Python tests and the strict type checker:
   ```bash
   uv run pytest
   uv run mypy src tests
   ```

Whenever the Rust inference APIs gain new knobs (for example, a new field on
`DecodeParameters`), add a matching property to both the PyO3 binding structs
and the Python data classes in `_api.py`.  Keep the `.pyi` files in sync so IDEs
surface the changes immediately.

## Updating for new engines or devices

* Add a new `EngineKind` variant in `src/lib.rs` and wire it into
  `model_from_args` so that the correct Rust crate is used for loading.
* Document the new option in `bindings/python/README.md` and expose a friendly
  literal in the `OcrEngine.from_files` factory.
* Update the tests to cover the new engine-specific behaviour using the
  `mock-engine` feature if the real weights are too large to ship.

## When the tokenizer format changes

The binding expects a HuggingFace tokenizer JSON file, matching the CLI.  If the
format or location changes, make sure the Python `OcrEngine.from_files`
parameters and error messages mirror whatever the Rust CLI requires.  The
`DEEPSEEK_TOKENIZER` helper in `_api.py` centralises path handling.

## Publishing

1. Build wheels: `uv run maturin build --release`.
2. Verify the resulting wheels via `uv pip install dist/deepseek_ocr-*.whl` and
   re-run the test suite against the installed package.
3. Upload to PyPI or your internal index once the GitHub Actions workflow
   finishes successfully.
