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
    """Vision pre-processing settings."""

    base_size: int = 1536
    image_size: int = 1024
    crop_mode: bool = False

    def to_native(self) -> _native.VisionSettingsInput:
        return _native.VisionSettingsInput(
            int(self.base_size),
            int(self.image_size),
            bool(self.crop_mode),
        )


@dataclass(slots=True, kw_only=True)
class GenerationConfig:
    """Decoding configuration used by the underlying language model."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int | None = None
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
    text: str
    prompt_tokens: int
    response_tokens: int
    generated_tokens: tuple[int, ...]


class OcrEngine:
    """User-facing helper that mirrors the Rust CLI API."""

    def __init__(
        self,
        handle: _native.EngineHandle,
        *,
        template: str,
        system_prompt: str,
    ) -> None:
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
        """Build an engine and download missing assets automatically.

        Args:
            model_id: The ID of the model to download/load (e.g. "deepseek-ocr", "paddleocr-vl").
            device: Execution device ("cpu", "cuda", "metal"). Defaults to DEEPSEEK_OCR_DEVICE env var or "cpu".
            dtype: Execution dtype. Defaults to DEEPSEEK_OCR_DTYPE env var or None.
            template: Prompt template name.
            system_prompt: System prompt to use.
            cache_dir: Custom cache directory.
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
        """Build an engine from local files or auto-downloaded assets."""
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
    """Download model assets into the standard cache, returning resolved paths."""
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
    missing = []
    for label, path in (
        ("config", config_path),
        ("tokenizer", tokenizer_path),
        ("weights", weights_path),
    ):
        if path is None or not Path(path).expanduser().exists():
            missing.append(label)

    # TODO: Remove magic string check by using a proper manifest/registry
    QUANTIZED_SUFFIXES = ("-q4k", "-q6k", "-q8k", "q4", "q6", "q8")
    wants_snapshot = any(model_id.endswith(suffix) for suffix in QUANTIZED_SUFFIXES)
    if wants_snapshot and (
        snapshot_path is None or not Path(snapshot_path).expanduser().exists()
    ):
        missing.append("snapshot")

    if not missing:
        # All required paths are present and valid
        assert config_path is not None
        assert tokenizer_path is not None
        assert weights_path is not None
        return config_path, tokenizer_path, weights_path, snapshot_path

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
    """Return the default model cache path for the current platform."""
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"

    return base / "deepseek-ocr" / "models" / model_id
