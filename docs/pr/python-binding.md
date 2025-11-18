# Pull request summary – Python binding

## Highlights

- Added a PyO3 crate (`bindings/python`) that exports the `deepseek_ocr._native`
  extension module.  The binding wraps the existing Rust `OcrEngine` trait and
  forwards all inference work to the same implementations used by the CLI.
- Introduced a modern Python package with strict typing, `.pyi` stubs, dev
  tooling powered by Astral UV, and Pillow-powered image helpers for developers.
- Documented usage (README), maintenance processes (DEVELOPING.md), and wired the
  new flow into CI via a dedicated GitHub Actions workflow that builds the
  extension and runs pytest + mypy in strict mode.
- Added regression tests with a mock engine so Python contributors can run the
  suite locally without downloading multi-gigabyte weights.
- Updated the top-level README with the new binding details and how to build it.

## Tests

- `uv run maturin develop --locked --features mock-engine`
- `uv run pytest`
- `uv run mypy src tests`
