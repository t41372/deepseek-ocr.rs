# Changelog

All notable changes to the Python bindings will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial Python bindings for deepseek-ocr.rs
- Support for all OCR models (DeepSeek, PaddleOCR, DotsOCR)
- Hardware acceleration support (CPU, CUDA, Metal)
- Quantized model support (Q4K, Q6K, Q8K)
- Full type annotations with strict mypy compliance
- Comprehensive documentation and examples
- Unit and integration tests
- GitHub Actions CI/CD pipeline
- Developer guide (CONTRIBUTING.md)

### Python API Classes
- `OcrModel`: Main class for model loading and inference
- `OcrResult`: OCR result with text and token counts
- `DecodeParams`: Text generation parameters
- `VisionConfig`: Image preprocessing configuration
- `Device`: Hardware device enumeration
- `Precision`: Model precision enumeration
- `list_available_models()`: Function to list available models

### Examples
- `basic_ocr.py`: Single image OCR
- `multi_image_ocr.py`: Multi-page document processing
- `batch_processing.py`: Batch processing with JSON output
- `model_comparison.py`: Benchmark different models

### Documentation
- Complete README.md with installation and usage guide
- CONTRIBUTING.md for maintainers
- Type stubs (.pyi) for IDE support
- Docstrings for all public APIs

### Testing
- Unit tests for all Python classes
- Integration tests for model loading and inference
- Type checking with mypy in strict mode
- Linting with ruff

### CI/CD
- Multi-platform testing (Linux, macOS, Windows)
- Multi-Python version testing (3.10, 3.11, 3.12)
- Wheel building for all platforms
- Source distribution building
- Automated GitHub Releases

## [0.1.0] - TBD

First release of Python bindings.

[Unreleased]: https://github.com/t41372/deepseek-ocr.rs/compare/python-v0.1.0...HEAD
[0.1.0]: https://github.com/t41372/deepseek-ocr.rs/releases/tag/python-v0.1.0
