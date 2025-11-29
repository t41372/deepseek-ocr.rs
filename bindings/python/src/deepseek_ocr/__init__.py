"""Typed Python bindings for DeepSeek OCR."""

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
