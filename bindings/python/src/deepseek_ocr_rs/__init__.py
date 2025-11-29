"""Python bindings for deepseek-ocr.rs - High-performance OCR inference library.

This package provides Python bindings to the deepseek-ocr.rs Rust library,
enabling fast and accurate OCR (Optical Character Recognition) with multiple
model backends including DeepSeek-OCR, PaddleOCR-VL, and DotsOCR.

Key Features:
    - Multiple OCR models with automatic weight downloads
    - CPU, CUDA, and Metal (Apple Silicon) acceleration
    - Quantized model variants (Q4K, Q6K, Q8K) for reduced memory usage
    - Thread-safe inference with GIL release for concurrent processing
    - Type-safe API with comprehensive type hints

Quick Start:
    >>> from deepseek_ocr_rs import OcrEngine
    >>>
    >>> # Create engine with automatic downloads
    >>> engine = OcrEngine.from_pretrained("deepseek-ocr", device="cpu")
    >>>
    >>> # Perform OCR
    >>> result = engine.generate(
    ...     prompt="<image> Extract all text from this image",
    ...     images=["document.jpg"]
    ... )
    >>> print(result.text)

Supported Models:
    - deepseek-ocr: Full-precision DeepSeek-OCR (~6.3GB weights, ~13GB RAM runtime, highest accuracy)
    - paddleocr-vl: Lighter PaddleOCR-VL model (~4.7GB weights, ~9GB RAM runtime, faster)
    - dots-ocr: DotsOCR for high-resolution documents (~9GB weights, ~30-50GB RAM runtime)
    - Quantized variants: *-q4k, *-q6k, *-q8k for reduced memory footprint

Main Classes and Functions:
    OcrEngine: Main inference engine
    GenerationConfig: Text generation parameters
    VisionConfig: Image preprocessing settings
    DecodeResult: Inference result container
    download_model: Download model assets
    get_default_cache_dir: Get platform cache directory
    normalize_text: Text normalization utility
    render_prompt: Prompt template rendering

See Also:
    Documentation: https://github.com/deepseek-ai/deepseek-ocr.rs
    Examples: examples/ directory in the source distribution
"""

from ._api import (
    DecodeResult,
    GenerationConfig,
    OcrEngine,
    VisionConfig,
    download_model,
    get_default_cache_dir,
)
from ._native import normalize_text, render_prompt

__all__ = [
    "DecodeResult",
    "GenerationConfig",
    "OcrEngine",
    "VisionConfig",
    "download_model",
    "get_default_cache_dir",
    "normalize_text",
    "render_prompt",
]
