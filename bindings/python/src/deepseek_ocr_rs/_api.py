"""High-level Python helpers for the DeepSeek OCR bindings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import platform
import warnings

from PIL import Image

from io import BytesIO
from . import _native

ImageInput = str | Path | bytes | bytearray | Image.Image
ModelKind = Literal["deepseek", "paddle", "dots", "mock"]
# Kept for backward compatibility, though we prefer ModelKind internally now
ModelLiteral = ModelKind
DeviceLiteral = Literal["cpu", "cuda", "metal"]
DTypeLiteral = Literal["f32", "f16", "bf16"]
StreamCallback = Callable[[int, Sequence[int]], None]


@dataclass(slots=True, kw_only=True)
class VisionConfig:
    """Vision pre-processing settings for image encoding.

    Attributes:
        base_size: Maximum base dimension for image resizing. Images are resized
            proportionally to fit within this dimension while preserving aspect ratio.
            Default is 1024 (matching Rust implementation).
        image_size: Target patch size for vision encoding. Resized images are further
            processed into patches of this size. Default is 640 (matching Rust implementation).
        crop_mode: Whether to use dynamic crop mode. If True, enables dynamic cropping
            for better quality; if False, uses padding mode. Default is True (matching Rust implementation).
    """

    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True

    def to_native(self) -> _native.VisionSettingsInput:
        return _native.VisionSettingsInput(
            int(self.base_size),
            int(self.image_size),
            bool(self.crop_mode),
        )


@dataclass(slots=True, kw_only=True)
class GenerationConfig:
    """Decoding configuration for text generation.

    Controls how the language model generates text tokens during OCR inference.
    For deterministic output, use defaults (do_sample=False, temperature=0.0).
    For stochastic generation, set do_sample=True and adjust temperature/sampling params.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate. Default is 512.
        do_sample: Whether to use sampling (True) or greedy decoding (False).
            Default is False for deterministic output.
        temperature: Sampling temperature. Higher values (e.g., 0.8) increase randomness,
            lower values (e.g., 0.2) make output more focused. Only used when do_sample=True.
            Default is 0.0.
        top_p: Nucleus sampling probability threshold. Only tokens with cumulative
            probability up to top_p are considered. Range: (0.0, 1.0]. Default is None.
        top_k: Only the top_k highest probability tokens are considered at each step.
            Default is None (no limit).
        repetition_penalty: Penalty for token repetition. Values > 1.0 discourage repetition,
            values < 1.0 encourage it. Default is 1.0 (no penalty).
        no_repeat_ngram_size: If set, prevents n-grams of this size from appearing more
            than once. Default is 20 (matching Rust implementation).
        seed: Random seed for reproducible sampling when do_sample=True. Default is None.
        use_cache: Whether to use KV-cache for faster generation. Default is True.
    """

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int | None = 20
    seed: int | None = None
    use_cache: bool = True

    def to_native(self) -> _native.DecodeParametersInput:
        return _native.DecodeParametersInput(
            int(self.max_new_tokens),
            bool(self.do_sample),
            float(self.temperature),
            self.top_p,
            self.top_k,
            float(self.repetition_penalty),
            self.no_repeat_ngram_size,
            self.seed,
            bool(self.use_cache),
        )


@dataclass(slots=True)
class DecodeResult:
    """Result from a single OCR generation call.

    Attributes:
        text: The decoded text output from the model.
        prompt_tokens: Number of tokens in the input prompt (including image embeddings).
        response_tokens: Number of tokens generated in the response.
        generated_tokens: Tuple of token IDs that were generated during decoding.
            Useful for debugging or advanced analysis.
    """

    text: str
    prompt_tokens: int
    response_tokens: int
    generated_tokens: tuple[int, ...]


class OcrEngine:
    """High-level OCR inference engine.

    This is the main entry point for performing OCR with deepseek-ocr.rs.
    Create an instance using `from_pretrained()` for automatic model downloads,
    or `from_files()` for manual path management.

    The engine is thread-safe and releases Python's GIL during inference,
    allowing concurrent execution with other Python code.
    """

    def __init__(
        self,
        handle: _native.EngineHandle,
        *,
        template: str,
        system_prompt: str,
    ) -> None:
        """Initialize an OcrEngine with a native handle.

        This constructor is typically not called directly. Use `from_pretrained()`
        or `from_files()` instead.

        Args:
            handle: Native Rust engine handle.
            template: Conversation template name. Options: "plain" (default), "deepseek",
                "deepseekv2", "alignment". Controls how prompts are formatted for the model.
            system_prompt: System prompt text to prepend to all generations.
        """
        self._handle = handle
        self._template = template
        self._system_prompt = system_prompt

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "deepseek-ocr",
        *,
        device: DeviceLiteral | None = None,
        dtype: DTypeLiteral | None = None,
        template: str = "plain",
        system_prompt: str = "",
        cache_dir: str | Path | None = None,
    ) -> "OcrEngine":
        """Build an OCR engine with automatic asset downloads.

        This is the recommended way to create an OcrEngine. Model files are
        automatically downloaded to a platform-specific cache directory if not
        already present. Subsequent calls reuse cached files.

        Args:
            model_id: Model identifier. Supported values include:
                - "deepseek-ocr" (default): Full-precision DeepSeek-OCR
                - "deepseek-ocr-q4k", "deepseek-ocr-q6k", "deepseek-ocr-q8k": Quantized variants
                - "paddleocr-vl": Full-precision PaddleOCR-VL
                - "paddleocr-vl-q4k", "paddleocr-vl-q6k", "paddleocr-vl-q8k": Quantized variants
                - "dots-ocr": Full-precision DotsOCR
                - "dots-ocr-q4k", "dots-ocr-q6k", "dots-ocr-q8k": Quantized variants
            device: Target device for inference. Options: "cpu", "cuda", "metal".
                Defaults to DEEPSEEK_OCR_DEVICE environment variable or "cpu".
            dtype: Data type for model weights. Options: "f32", "f16", "bf16".
                If None, automatically selects the best dtype for the device
                (f32 for CPU, f16 for Metal/CUDA).
            template: Conversation template name. Options: "plain" (default), "deepseek",
                "deepseekv2", "alignment". Controls how prompts are formatted for the model.
            system_prompt: Optional system prompt prepended to all user prompts. Default is "".
            cache_dir: Custom root cache directory. If None, uses platform default:
                - macOS: ~/Library/Caches/deepseek-ocr
                - Linux: ~/.cache/deepseek-ocr
                - Windows: %LOCALAPPDATA%/deepseek-ocr

        Returns:
            OcrEngine: Ready-to-use OCR engine instance.

        Raises:
            ValueError: If model_id is unknown or device/dtype combination is invalid.
            RuntimeError: If model download or loading fails.

        Example:
            >>> engine = OcrEngine.from_pretrained("deepseek-ocr", device="cpu")
            >>> result = engine.generate(
            ...     prompt="<image> Extract all text",
            ...     images=["receipt.jpg"]
            ... )
            >>> print(result.text)
        """
        if device is None:
            device = _resolve_device()

        # Validate device/dtype compatibility
        _validate_device_dtype(device, dtype)

        # Infer engine type from model_id
        engine = _engine_from_model_id(model_id)

        return cls.from_files(
            model_id=model_id,
            engine=engine,
            device=device,
            dtype=dtype,
            template=template,
            system_prompt=system_prompt,
            auto_download=True,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_files(
        cls,
        *,
        model_id: str | None = None,
        engine: ModelKind = "deepseek",
        config_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        snapshot_path: str | Path | None = None,
        device: DeviceLiteral = "cpu",
        dtype: DTypeLiteral | None = None,
        template: str = "plain",
        system_prompt: str = "",
        auto_download: bool = False,
        cache_dir: str | Path | None = None,
    ) -> "OcrEngine":
        """Build an OCR engine from explicit file paths or with auto-download.

        This method provides fine-grained control over model loading. Use it when you:
        - Have custom model files at non-standard locations
        - Want to specify individual asset paths explicitly
        - Need to override the auto-detected engine type

        For most use cases, prefer `from_pretrained()` instead.

        Args:
            model_id: Model identifier used for auto-download when auto_download=True.
                Not required when providing all paths manually.
            engine: Engine type to use. Options: "deepseek", "paddle", "dots", "mock".
                Default is "deepseek".
            config_path: Path to config.json file. Required unless auto_download=True or engine="mock".
            tokenizer_path: Path to tokenizer.json file. Required unless auto_download=True or engine="mock".
            weights_path: Path to model weights (safetensors). Optional; if None and auto_download=True,
                will be downloaded.
            snapshot_path: Path to quantization snapshot (.dsq file). Required for quantized models
                (e.g., deepseek-ocr-q4k), optional otherwise.
            device: Target device. Options: "cpu", "cuda", "metal". Default is "cpu".
            dtype: Data type for model weights. Options: "f32", "f16", "bf16".
                If None, automatically selects the best dtype for the device.
            template: Conversation template name. Options: "plain" (default), "deepseek",
                "deepseekv2", "alignment". Controls how prompts are formatted for the model.
            system_prompt: Optional system prompt prepended to all user prompts. Default is "".
            auto_download: If True and model_id is provided, downloads missing assets automatically.
                Default is False.
            cache_dir: Custom root cache directory for auto-download. If None, uses platform default.

        Returns:
            OcrEngine: Ready-to-use OCR engine instance.

        Raises:
            ValueError: If required paths are missing or engine type is invalid.
            FileNotFoundError: If specified paths don't exist.
            RuntimeError: If model loading fails.

        Example:
            >>> # Manual path specification
            >>> engine = OcrEngine.from_files(
            ...     engine="deepseek",
            ...     config_path="/path/to/config.json",
            ...     tokenizer_path="/path/to/tokenizer.json",
            ...     weights_path="/path/to/model.safetensors",
            ...     device="cpu"
            ... )
            >>>
            >>> # With auto-download
            >>> engine = OcrEngine.from_files(
            ...     model_id="deepseek-ocr-q4k",
            ...     engine="deepseek",
            ...     auto_download=True,
            ...     device="cpu"
            ... )
        """
        if auto_download:
            if model_id is None:
                raise ValueError(
                    "model_id is required when auto_download=True.\n"
                    "Hint: Use OcrEngine.from_pretrained(model_id='deepseek-ocr') "
                    "for automatic downloads, or provide explicit paths with from_files()."
                )

            config_path, tokenizer_path, weights_path, snapshot_path = _ensure_assets(
                model_id=model_id,
                config_path=config_path,
                tokenizer_path=tokenizer_path,
                weights_path=weights_path,
                snapshot_path=snapshot_path,
                cache_dir=cache_dir,
            )
        else:
            # Validate paths if not auto-downloading and not a mock
            if engine != "mock" and (config_path is None or tokenizer_path is None):
                raise ValueError(
                    f"config_path and tokenizer_path are required for engine '{engine}' "
                    f"when auto_download=False.\n"
                    f"Hint: Use OcrEngine.from_pretrained(model_id='{model_id or 'deepseek-ocr'}') "
                    f"to download assets automatically, or provide explicit paths.\n"
                    f"See docs/python-binding.md for details on model asset locations."
                )

        handle = _native.create_engine(
            model_kind=engine,
            config_path=_coerce_optional_path(config_path),
            tokenizer_path=_coerce_optional_path(tokenizer_path),
            weights_path=_coerce_optional_path(weights_path),
            snapshot_path=_coerce_optional_path(snapshot_path),
            device=device,
            dtype=dtype,
        )
        return cls(handle, template=template, system_prompt=system_prompt)

    def generate(
        self,
        *,
        prompt: str,
        images: Sequence[ImageInput],
        generation: GenerationConfig | None = None,
        vision: VisionConfig | None = None,
        stream: StreamCallback | None = None,
    ) -> DecodeResult:
        """Perform OCR inference on one or more images.

        This method encodes the provided images through the vision tower, renders the
        prompt with image placeholders, and generates text using the language model.
        The Python GIL is released during inference for better concurrency.

        Args:
            prompt: User prompt containing one or more "<image>" tokens. The number of
                "<image>" tokens must exactly match the number of images provided.
                Example: "<image> Extract all text from this receipt"
                Example (multi-image): "<image><image> Compare these two documents"
            images: Sequence of images to process. Each image can be:
                - str or Path: File path to an image (supports .jpg, .png, .webp, etc.)
                - bytes or bytearray: Raw image data in any format supported by PIL
                - PIL.Image.Image: PIL image object
                The number of images must match the number of "<image>" tokens in prompt.
            generation: Optional generation configuration. If None, uses defaults
                (deterministic greedy decoding with max_new_tokens=512).
            vision: Optional vision processing configuration. If None, uses defaults
                (base_size=1536, image_size=1024, crop_mode=False).
            stream: Optional callback for streaming token generation. Called as
                stream(token_count: int, token_ids: Sequence[int]) for each generated token.
                Useful for real-time display of results.

        Returns:
            DecodeResult: Object containing:
                - text: The generated text output
                - prompt_tokens: Number of tokens in the prompt
                - response_tokens: Number of tokens in the response
                - generated_tokens: Tuple of generated token IDs

        Raises:
            ValueError: If the number of "<image>" tokens doesn't match the number of images,
                or if image paths are invalid, or if image data is malformed.
            FileNotFoundError: If an image path doesn't exist.
            PermissionError: If an image file cannot be read due to permissions.
            RuntimeError: If inference fails (e.g., out of memory, device error).

        Example:
            >>> engine = OcrEngine.from_pretrained("deepseek-ocr")
            >>> result = engine.generate(
            ...     prompt="<image> Convert to markdown",
            ...     images=["document.jpg"]
            ... )
            >>> print(result.text)
            >>>
            >>> # With custom generation config
            >>> result = engine.generate(
            ...     prompt="<image> Extract text",
            ...     images=["receipt.png"],
            ...     generation=GenerationConfig(max_new_tokens=1024, do_sample=True, temperature=0.7)
            ... )
            >>>
            >>> # With streaming
            >>> def on_token(count, ids):
            ...     print(f"Generated {count} tokens so far")
            >>> result = engine.generate(
            ...     prompt="<image> Extract all text",
            ...     images=["page.jpg"],
            ...     stream=on_token
            ... )
        """
        rendered_prompt = _native.render_prompt(
            self._template,
            self._system_prompt,
            prompt,
        )
        image_slots = rendered_prompt.count("<image>")
        if image_slots != len(images):
            if image_slots == 0:
                hint = (
                    "\nHint: Your prompt must contain at least one '<image>' token. "
                    "Example: '<image> Extract all text from this image'"
                )
            elif len(images) == 0:
                hint = (
                    f"\nHint: Your prompt has {image_slots} <image> token(s) "
                    f"but you provided no images. Pass images=[...] to generate()."
                )
            else:
                hint = (
                    f"\nHint: {'Add' if image_slots > len(images) else 'Remove'} "
                    f"{abs(image_slots - len(images))} <image> token(s) "
                    f"{'to' if image_slots > len(images) else 'from'} your prompt "
                    f"to match the {len(images)} image(s) provided."
                )
            raise ValueError(
                f"Image count mismatch: prompt contains {image_slots} <image> token(s) "
                f"but {len(images)} image(s) were provided.{hint}"
            )
        payloads = [_normalise_image(image) for image in images]
        vision_settings = (vision or VisionConfig()).to_native()
        generation_settings = (generation or GenerationConfig()).to_native()
        outcome = self._handle.decode(
            rendered_prompt,
            payloads,
            vision_settings,
            generation_settings,
            stream,
        )
        return DecodeResult(
            text=outcome.text,
            prompt_tokens=outcome.prompt_tokens,
            response_tokens=outcome.response_tokens,
            generated_tokens=tuple(outcome.generated_tokens),
        )


def _normalise_image(image: ImageInput) -> bytes:
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if isinstance(image, Image.Image):
        return _buffer_ppm(image)

    # Handle path-like inputs
    path = Path(image)
    expanded_path = path.expanduser()

    if not expanded_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {expanded_path}\n"
            f"Hint: Check that the path is correct and the file exists. "
            f"You can also pass PIL.Image objects or raw bytes instead of paths."
        )

    if not expanded_path.is_file():
        raise ValueError(
            f"Path is not a file: {expanded_path}\n"
            f"Hint: Provide a path to an image file (e.g., .jpg, .png, .webp), "
            f"not a directory."
        )

    try:
        data = expanded_path.read_bytes()
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied reading image: {expanded_path}\n"
            f"Hint: Check file permissions or try loading with PIL.Image.open() first."
        ) from e

    return data


def _buffer_ppm(image: Image.Image) -> bytes:
    """Encode a PIL image to bytes using fast PPM format.

    PPM is effectively raw RGB with a tiny header, so it avoids the heavy
    compression/decompression cost of PNG while remaining compatible with the
    Rust-side `image` crate (with the `pnm` feature). The trade-off is larger
    payload size, which is acceptable for in-process transfer.
    """
    converted = image.convert("RGB")

    buf = BytesIO()
    converted.save(buf, format="PPM")
    data = buf.getvalue()
    converted.close()
    return data


def _coerce_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return os.fsdecode(Path(value).expanduser())


def download_model(
    model: str, cache_dir: str | Path | None = None
) -> _native.DownloadResultHandle:
    """Download OCR model assets to local cache.

    Downloads all required files for the specified model (config, tokenizer, weights,
    and optionally quantization snapshots) from remote sources (Hugging Face or ModelScope).
    Files are cached locally and reused on subsequent calls.

    Args:
        model: Model identifier to download. Examples:
            - "deepseek-ocr": Full-precision DeepSeek-OCR (~6.3GB)
            - "deepseek-ocr-q4k": 4-bit quantized variant (~2GB)
            - "paddleocr-vl": Full-precision PaddleOCR-VL (~4.7GB)
            - "dots-ocr": Full-precision DotsOCR (~9GB)
            See from_pretrained() for full list of supported models.
        cache_dir: Optional custom cache root directory. If None, uses platform default:
            - macOS: ~/Library/Caches/deepseek-ocr
            - Linux: ~/.cache/deepseek-ocr
            - Windows: %LOCALAPPDATA%/deepseek-ocr

    Returns:
        DownloadResultHandle: Object with the following attributes:
            - model_id (str): The requested model identifier
            - model_dir (str): Directory containing the model files
            - baseline_dir (str): Directory containing baseline (non-quantized) assets
            - config_path (str): Path to config.json
            - tokenizer_path (str): Path to tokenizer.json
            - weights_path (str): Path to model weights
            - snapshot_path (str | None): Path to .dsq quantization file (if applicable)
            - preprocessor_path (str | None): Path to preprocessor_config.json (if applicable)

    Raises:
        ValueError: If model identifier is unknown.
        RuntimeError: If download or verification fails.

    Example:
        >>> result = download_model("deepseek-ocr-q4k")
        >>> print(f"Model cached at: {result.model_dir}")
        >>> print(f"Config: {result.config_path}")
        >>> print(f"Weights: {result.weights_path}")
    """
    cache_str = str(cache_dir) if cache_dir is not None else None
    return _native.download_model(model, cache_dir=cache_str)


def _ensure_assets(
    *,
    model_id: str,
    config_path: str | Path | None,
    tokenizer_path: str | Path | None,
    weights_path: str | Path | None,
    snapshot_path: str | Path | None,
    cache_dir: str | Path | None,
) -> tuple[str | Path, str | Path, str | Path, str | Path | None]:
    """Ensure all assets are present using the Rust registry as the single source of truth."""

    # Fast path: everything explicitly supplied and present on disk.
    supplied_paths = [config_path, tokenizer_path, weights_path]
    all_present = all(
        p is not None and Path(p).expanduser().exists() for p in supplied_paths
    )

    # If the model is quantized, a snapshot is mandatory; otherwise snapshot_path is optional.
    needs_snapshot = False
    try:
        needs_snapshot = bool(_native.model_requires_snapshot(model_id))
    except Exception:
        needs_snapshot = False

    snapshot_present = not needs_snapshot or (
        snapshot_path is not None and Path(snapshot_path).expanduser().exists()
    )

    if all_present and snapshot_present:
        assert config_path is not None
        assert tokenizer_path is not None
        assert weights_path is not None
        return config_path, tokenizer_path, weights_path, snapshot_path

    # Delegate to the Rust asset helper; it will no-op if cache files already exist.
    result = download_model(model_id, cache_dir=cache_dir)
    return (
        result.config_path,
        result.tokenizer_path,
        result.weights_path,
        result.snapshot_path,
    )


def _engine_from_model_id(model_id: str) -> ModelKind:
    """Infer the engine type from the model ID.

    Supports environment variable override for forward compatibility:
    Set DEEPSEEK_OCR_ENGINE_<MODEL_ID>=<engine_type> to manually specify
    the engine for models not yet recognized by this version.

    Example: DEEPSEEK_OCR_ENGINE_FOO_OCR=deepseek
    """
    # Check for explicit environment variable override
    env_key = f"DEEPSEEK_OCR_ENGINE_{model_id.upper().replace('-', '_')}"
    explicit_engine = os.environ.get(env_key)
    if explicit_engine in ("deepseek", "paddle", "dots", "mock"):
        return explicit_engine  # type: ignore[return-value]

    try:
        native_engine = _native.engine_for_model(model_id)
        if native_engine:
            return native_engine  # type: ignore[return-value]
    except Exception:
        # Fall through to heuristics for forward compatibility.
        pass

    # Heuristic-based inference
    lower = model_id.lower()
    if "paddle" in lower:
        return "paddle"
    if "dots" in lower:
        return "dots"

    # Default to deepseek for unknown models (most compatible fallback)
    return "deepseek"


def _validate_device_dtype(device: DeviceLiteral, dtype: DTypeLiteral | None) -> None:
    """Validate device/dtype compatibility and emit warnings for suboptimal configs."""
    if device == "cpu" and dtype == "f16":
        warnings.warn(
            "f16 dtype on CPU may have poor performance or compatibility issues. "
            "Consider using f32 for CPU, or switch to metal/cuda for f16 acceleration.",
            UserWarning,
            stacklevel=3,
        )
    elif device == "cpu" and dtype == "bf16":
        warnings.warn(
            "bf16 dtype on CPU may have poor performance or compatibility issues. "
            "Consider using f32 for CPU, or switch to metal/cuda for bf16 support.",
            UserWarning,
            stacklevel=3,
        )
    elif device == "metal" and dtype not in (None, "f16", "f32"):
        warnings.warn(
            f"Metal device works best with f16 or f32. {dtype} may not be supported.",
            UserWarning,
            stacklevel=3,
        )


def _resolve_device(
    env_var: str = "DEEPSEEK_OCR_DEVICE",
    default: DeviceLiteral = "cpu",
) -> DeviceLiteral:
    value = os.environ.get(env_var, default)
    if value not in ("cpu", "cuda", "metal"):
        return default
    return value  # type: ignore[return-value]


def get_default_cache_dir(model_id: str) -> Path:
    """Get the platform-specific default cache directory for a model.

    Returns the standard cache location where model files are stored. The path
    follows platform conventions and respects XDG Base Directory specification on Linux.

    Args:
        model_id: Model identifier (e.g., "deepseek-ocr", "paddleocr-vl").

    Returns:
        Path: Absolute path to the model's cache directory. The directory structure is:
            <platform_cache_root>/deepseek-ocr/models/<model_id>/

            Platform-specific roots:
            - macOS: ~/Library/Caches/
            - Linux: $XDG_CACHE_HOME/ (or ~/.cache/ if unset)
            - Windows: %LOCALAPPDATA%/ (or ~/AppData/Local/ if unset)

    Example:
        >>> cache_dir = get_default_cache_dir("deepseek-ocr")
        >>> print(cache_dir)
        # macOS: /Users/username/Library/Caches/deepseek-ocr/models/deepseek-ocr
        # Linux: /home/username/.cache/deepseek-ocr/models/deepseek-ocr
        # Windows: C:\\Users\\username\\AppData\\Local\\deepseek-ocr\\models\\deepseek-ocr
    """
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"

    return base / "deepseek-ocr" / "models" / model_id
