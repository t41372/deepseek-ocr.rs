"""
DeepSeek OCR - High-performance OCR with DeepSeek, PaddleOCR, and DotsOCR models.

This package provides Python bindings for the deepseek-ocr.rs Rust library,
enabling fast and accurate optical character recognition using state-of-the-art models.

Basic usage:
    ```python
    from deepseek_ocr import OcrModel, Device, DecodeParams

    # Load model
    model = OcrModel("deepseek-ocr", device=Device.Cpu)

    # Perform OCR
    result = model.infer(
        prompt="<image> Extract all text from this image",
        images=["path/to/image.png"]
    )
    print(result.text)
    ```

For more information, see the documentation at:
https://github.com/t41372/deepseek-ocr.rs/blob/master/crates/python-bindings/README.md
"""

from typing import TYPE_CHECKING

from ._core import (
    DecodeParams,
    Device,
    OcrModel,
    OcrResult,
    Precision,
    VisionConfig,
    list_available_models,
)

__version__ = "0.1.0"

__all__ = [
    "OcrModel",
    "OcrResult",
    "DecodeParams",
    "VisionConfig",
    "Device",
    "Precision",
    "list_available_models",
    "__version__",
]


# Type checking stubs
if TYPE_CHECKING:
    from .py.typed import *
