"""Type stubs for the native Rust extension module.

This module provides the low-level native interface to the Rust OCR engine.
Users should typically use the high-level API in _api.py instead.
"""

from collections.abc import Callable, Sequence

class VisionSettingsInput:
    """Native vision preprocessing settings.

    Attributes:
        base_size: Maximum base dimension for image resizing.
        image_size: Target patch size for vision encoding.
        crop_mode: Whether to use center cropping.
    """

    base_size: int
    image_size: int
    crop_mode: bool
    def __init__(self, base_size: int, image_size: int, crop_mode: bool) -> None: ...

class DecodeParametersInput:
    """Native text generation parameters.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: Whether to use sampling or greedy decoding.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability threshold.
        top_k: Top-k sampling limit.
        repetition_penalty: Penalty for token repetition.
        no_repeat_ngram_size: N-gram size for repetition prevention.
        seed: Random seed for sampling.
        use_cache: Whether to use KV-cache.
    """

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float | None
    top_k: int | None
    repetition_penalty: float
    no_repeat_ngram_size: int | None
    seed: int | None
    use_cache: bool
    def __init__(
        self,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float | None = ...,
        top_k: int | None = ...,
        repetition_penalty: float = ...,
        no_repeat_ngram_size: int | None = ...,
        seed: int | None = ...,
        use_cache: bool = ...,
    ) -> None: ...

class DecodeOutcomeHandle:
    """Native inference result.

    Attributes:
        text: The decoded text output.
        prompt_tokens: Number of tokens in the prompt.
        response_tokens: Number of tokens in the response.
        generated_tokens: List of generated token IDs.
    """

    text: str
    prompt_tokens: int
    response_tokens: int
    generated_tokens: list[int]

class DownloadResultHandle:
    """Native model download result.

    Attributes:
        model_id: The requested model identifier.
        model_dir: Directory containing the model files.
        baseline_dir: Directory containing baseline (non-quantized) assets.
        config_path: Path to config.json.
        tokenizer_path: Path to tokenizer.json.
        weights_path: Path to model weights.
        snapshot_path: Path to .dsq quantization file (if applicable).
        preprocessor_path: Path to preprocessor_config.json (if applicable).
    """

    model_id: str
    model_dir: str
    baseline_dir: str
    config_path: str
    tokenizer_path: str
    weights_path: str
    snapshot_path: str | None
    preprocessor_path: str | None

class EngineHandle:
    """Native OCR engine handle.

    This is the low-level Rust engine handle. Use OcrEngine from _api.py
    for a higher-level interface.
    """

    def decode(
        self,
        prompt: str,
        images: Sequence[bytes],
        vision: VisionSettingsInput,
        decode: DecodeParametersInput,
        stream: Callable[[int, Sequence[int]], None] | None = ...,
    ) -> DecodeOutcomeHandle:
        """Perform OCR inference.

        Args:
            prompt: Rendered prompt string with image placeholders.
            images: Sequence of image byte payloads.
            vision: Vision preprocessing settings.
            decode: Text generation parameters.
            stream: Optional streaming callback.

        Returns:
            DecodeOutcomeHandle: Inference result.
        """
        ...

MOCK_MODEL_KIND: str
"""Mock engine kind constant (only available with mock-engine feature)."""

def create_engine(
    model_kind: str,
    *,
    config_path: str | None = ...,
    tokenizer_path: str | None = ...,
    weights_path: str | None = ...,
    snapshot_path: str | None = ...,
    device: str | None = ...,
    dtype: str | None = ...,
) -> EngineHandle:
    """Create a native OCR engine.

    Args:
        model_kind: Engine type ("deepseek", "paddle", "dots", or "mock").
        config_path: Path to config.json (required for non-mock engines).
        tokenizer_path: Path to tokenizer.json (required for non-mock engines).
        weights_path: Path to model weights (optional).
        snapshot_path: Path to quantization snapshot (optional).
        device: Target device ("cpu", "cuda", or "metal").
        dtype: Data type ("f32", "f16", or "bf16").

    Returns:
        EngineHandle: Native engine handle.
    """
    ...

def engine_for_model(model_id: str) -> str | None:
    """Get the engine type for a registered model ID.

    Args:
        model_id: Model identifier.

    Returns:
        str | None: Engine type label ("deepseek", "paddle", "dots") or None if unknown.
    """
    ...

def model_requires_snapshot(model_id: str) -> bool:
    """Check if a model requires a quantization snapshot.

    Args:
        model_id: Model identifier.

    Returns:
        bool: True if the model is quantized and requires a .dsq file.
    """
    ...

def download_model(model_id: str, cache_dir: str | None = ...) -> DownloadResultHandle:
    """Download model assets from remote sources.

    Args:
        model_id: Model identifier to download.
        cache_dir: Optional custom cache root directory.

    Returns:
        DownloadResultHandle: Download result with file paths.
    """
    ...

def render_prompt(template: str, system_prompt: str, raw_prompt: str) -> str:
    """Render a prompt using a conversation template.

    Args:
        template: Conversation template name. Options: "plain", "deepseek",
            "deepseekv2", "alignment".
        system_prompt: System prompt text.
        raw_prompt: User prompt text.

    Returns:
        str: Rendered prompt string.
    """
    ...

def normalize_text(text: str) -> str:
    """Normalize text output from the model.

    Applies standard text normalization rules (whitespace, punctuation, etc.).

    Args:
        text: Raw text to normalize.

    Returns:
        str: Normalized text.
    """
    ...
