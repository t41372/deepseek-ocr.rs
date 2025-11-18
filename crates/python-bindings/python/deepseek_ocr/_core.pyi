"""
Type stubs for deepseek_ocr._core module.

This file provides type annotations for the Rust-implemented core module.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

class Device(Enum):
    """Compute device for model inference."""

    Cpu: Device
    """CPU device"""

    Cuda: Device
    """CUDA device (NVIDIA GPU)"""

    Metal: Device
    """Metal device (Apple Silicon)"""

class Precision(Enum):
    """Precision/dtype for model weights."""

    F16: Precision
    """16-bit floating point"""

    F32: Precision
    """32-bit floating point"""

    BF16: Precision
    """Brain floating point 16"""

    Auto: Precision
    """Auto-select based on device"""

class VisionConfig:
    """
    Vision preprocessing configuration.

    Args:
        base_size: Base resolution for image processing (default: 1024)
        image_size: Target image size (default: 640)
        crop_mode: Enable crop mode (default: True)
    """

    base_size: int
    image_size: int
    crop_mode: bool

    def __init__(
        self, base_size: int = 1024, image_size: int = 640, crop_mode: bool = True
    ) -> None: ...
    def __repr__(self) -> str: ...

class DecodeParams:
    """
    Decoding/generation parameters for OCR inference.

    Args:
        max_new_tokens: Maximum number of new tokens to generate (default: 512)
        do_sample: Enable sampling vs greedy decoding (default: False)
        temperature: Sampling temperature (default: 0.0)
        top_p: Top-p (nucleus) sampling threshold (default: None)
        top_k: Top-k sampling limit (default: None)
        repetition_penalty: Repetition penalty (default: 1.0)
        no_repeat_ngram_size: No-repeat n-gram size (default: None)
        seed: Random seed for sampling (default: None)
        use_cache: Use KV cache (default: True)
    """

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: Optional[float]
    top_k: Optional[int]
    repetition_penalty: float
    no_repeat_ngram_size: Optional[int]
    seed: Optional[int]
    use_cache: bool

    def __init__(
        self,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ) -> None: ...
    def __repr__(self) -> str: ...

class OcrResult:
    """
    OCR inference result.

    Attributes:
        text: Extracted text
        prompt_tokens: Number of prompt tokens
        response_tokens: Number of generated tokens
    """

    text: str
    prompt_tokens: int
    response_tokens: int

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class OcrModel:
    """
    Main OCR model class for inference.

    This class handles loading OCR models and performing inference on images.

    Example:
        ```python
        from deepseek_ocr import OcrModel, Device

        # Load model
        model = OcrModel("deepseek-ocr", device=Device.Cpu)

        # Perform OCR
        result = model.infer(
            prompt="<image> Extract text",
            images=["image.png"]
        )
        print(result.text)
        ```
    """

    def __init__(
        self,
        model_name: str,
        device: Device = Device.Cpu,
        precision: Precision = Precision.Auto,
        config_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        """
        Load an OCR model.

        Args:
            model_name: Name of the model. Available models:
                - "deepseek-ocr", "deepseek-ocr-q4k", "deepseek-ocr-q6k", "deepseek-ocr-q8k"
                - "paddleocr-vl", "paddleocr-vl-q4k", "paddleocr-vl-q6k", "paddleocr-vl-q8k"
                - "dots-ocr", "dots-ocr-q4k", "dots-ocr-q6k", "dots-ocr-q8k"
            device: Compute device (Device.Cpu, Device.Cuda, or Device.Metal)
            precision: Model precision (Precision.F16, F32, BF16, or Auto)
            config_path: Optional path to model config.json
            weights_path: Optional path to model weights
            snapshot_path: Optional path to quantized snapshot

        Raises:
            RuntimeError: If model loading fails
        """
        ...

    def infer(
        self,
        prompt: str,
        images: list[Path | str],
        vision_config: Optional[VisionConfig] = None,
        decode_params: Optional[DecodeParams] = None,
        template: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> OcrResult:
        """
        Perform OCR on image(s) with a text prompt.

        Args:
            prompt: Text prompt. Use "<image>" as placeholder for images.
                Example: "<image> Extract all text from this document"
            images: List of image paths (as Path or str)
            vision_config: Optional vision preprocessing config
            decode_params: Optional decoding parameters
            template: Conversation template name (default: "plain")
            system_prompt: System prompt to prepend (default: "")

        Returns:
            OcrResult with extracted text and token counts

        Raises:
            ValueError: If number of <image> placeholders doesn't match number of images
            RuntimeError: If inference fails
        """
        ...

    def info(self) -> str:
        """Get model information string."""
        ...

    def __repr__(self) -> str: ...

def list_available_models() -> list[str]:
    """
    List all available OCR models.

    Returns:
        List of model names that can be used with OcrModel
    """
    ...
