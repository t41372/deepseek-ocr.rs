"""High-level Python helpers for the DeepSeek OCR bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

from PIL import Image

from . import _native

ImageInput = Union[str, Path, bytes, bytearray, Image.Image]
ModelLiteral = Literal["deepseek", "paddle", "dots", "mock"]
DeviceLiteral = Literal["cpu", "cuda", "metal"]
DTypeLiteral = Literal["f32", "f16", "bf16"]
StreamCallback = Callable[[int, Sequence[int]], None]


@dataclass(slots=True)
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


@dataclass(slots=True)
class GenerationConfig:
    """Decoding configuration used by the underlying language model."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: Optional[int] = None
    seed: Optional[int] = None
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
    generated_tokens: Tuple[int, ...]


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
    def from_files(
        cls,
        *,
        config_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        weights_path: Optional[Union[str, Path]] = None,
        snapshot_path: Optional[Union[str, Path]] = None,
        model: ModelLiteral = "deepseek",
        device: DeviceLiteral = "cpu",
        dtype: Optional[DTypeLiteral] = None,
        template: str = "plain",
        system_prompt: str = "",
    ) -> "OcrEngine":
        handle = _native.create_engine(
            model_kind=model,
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
        generation: Optional[GenerationConfig] = None,
        vision: Optional[VisionConfig] = None,
        stream: Optional[StreamCallback] = None,
    ) -> DecodeResult:
        rendered_prompt = _native.render_prompt(
            self._template,
            self._system_prompt,
            prompt,
        )
        image_slots = rendered_prompt.count("<image>")
        if image_slots != len(images):
            raise ValueError(
                "prompt contains %d <image> tokens but %d images were provided"
                % (image_slots, len(images))
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
        buffer = _buffer_png(image)
        return buffer
    path = Path(image)
    data = path.expanduser().read_bytes()
    return data


def _buffer_png(image: Image.Image) -> bytes:
    with image.convert("RGB") as converted:
        from io import BytesIO

        buf = BytesIO()
        converted.save(buf, format="PNG")
        return buf.getvalue()


def _coerce_optional_path(value: Optional[Union[str, Path]]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value)
    return str(path.expanduser())
