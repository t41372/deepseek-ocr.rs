"""High-level Python helpers for the DeepSeek OCR bindings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import platform

from PIL import Image

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
        # Backward compatibility
        model: ModelKind | None = None,
    ) -> "OcrEngine":
        """Build an engine and download missing assets automatically.

        Args:
            model_id: The ID of the model to download/load (e.g. "deepseek-ocr", "paddleocr-vl").
            device: Execution device ("cpu", "cuda", "metal"). Defaults to DEEPSEEK_OCR_DEVICE env var or "cpu".
            dtype: Execution dtype. Defaults to DEEPSEEK_OCR_DTYPE env var or None.
            template: Prompt template name.
            system_prompt: System prompt to use.
            cache_dir: Custom cache directory.
            model: [Deprecated] Explicitly set the engine kind. If None, inferred from model_id.
        """
        if device is None:
            device = _resolve_device()

        # If the user passed the old `model` arg but no `model_id` (or default),
        # we try to respect it, but `model_id` is the primary source of truth now.
        # Ideally, we infer engine from model_id.
        engine = model if model is not None else _engine_from_model_id(model_id)

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
        # Backward compatibility alias
        model: ModelKind | None = None,
    ) -> "OcrEngine":
        """Build an engine from local files or auto-downloaded assets."""
        if model is not None:
            engine = model

        if auto_download:
            if model_id is None:
                raise ValueError("model_id is required when auto_download=True")

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
                    "config_path and tokenizer_path are required when auto_download=False "
                    "for real models; see README 'Get model assets' for expected filenames."
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
            raise ValueError(
                f"prompt contains {image_slots} <image> tokens "
                f"but {len(images)} images were provided"
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
    path = Path(image)
    data = path.expanduser().read_bytes()
    return data


def _buffer_ppm(image: Image.Image) -> bytes:
    """Encode a PIL image to bytes using fast PPM format.

    PPM is effectively raw RGB with a tiny header, so it avoids the heavy
    compression/decompression cost of PNG while remaining compatible with the
    Rust-side `image` crate (with the `pnm` feature). The trade-off is larger
    payload size, which is acceptable for in-process transfer.
    """
    # Import locally to avoid top-level IO import if not needed,
    # though for high-throughput this might be better at module level.
    # Keeping as is for now, but could be optimized.
    from io import BytesIO

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
        root = "~/Library/Caches/deepseek-ocr/models"
    elif system == "Windows":
        root = "%LOCALAPPDATA%\\deepseek-ocr\\models"
    else:
        root = "~/.cache/deepseek-ocr/models"
    return Path(os.path.expandvars(root)).expanduser() / model_id
