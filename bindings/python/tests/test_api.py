from __future__ import annotations

import concurrent.futures
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from collections.abc import Sequence

import pytest
from PIL import Image

from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig, render_prompt


def _sample_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(128, 64, 32))


def test_mock_engine_roundtrip() -> None:
    engine = OcrEngine.from_files(model="mock")
    prompt = "<image> Describe the receipt."
    result = engine.generate(prompt=prompt, images=[_sample_image()])
    assert "mock-response" in result.text
    assert result.response_tokens > 0


def test_prompt_image_count_validation() -> None:
    engine = OcrEngine.from_files(model="mock")
    with pytest.raises(ValueError):
        engine.generate(prompt="<image> <image> ...", images=[_sample_image()])


def test_generation_config_maps_to_native() -> None:
    engine = OcrEngine.from_files(model="mock")
    generation = GenerationConfig(max_new_tokens=8, do_sample=True, temperature=0.7)
    vision = VisionConfig(base_size=768, image_size=512, crop_mode=True)
    result = engine.generate(
        prompt="<image> Explain.",
        images=[_sample_image()],
        generation=generation,
        vision=vision,
    )
    assert "sampled" in result.text


def test_streaming_callback_receives_tokens() -> None:
    engine = OcrEngine.from_files(model="mock")
    seen: list[int] = []

    def _cb(count: int, tokens: Sequence[int]) -> None:
        seen.append(count)
        assert len(tokens) == count

    engine.generate(prompt="<image> stream", images=[_sample_image()], stream=_cb)
    assert seen, "callback should be invoked"


def test_render_prompt_is_accessible() -> None:
    prompt = render_prompt("plain", "", "<image> hello")
    assert "<image>" in prompt


def test_pil_image_modes() -> None:
    """Ensure various PIL modes are converted to RGB correctly."""

    engine = OcrEngine.from_files(model="mock")

    for mode in ["RGB", "RGBA", "L"]:
        img = Image.new(mode, (16, 16), color=(100, 100, 100) if mode != "L" else 100)
        result = engine.generate(prompt="<image> test", images=[img])
        assert "mock-response" in result.text
        assert result.response_tokens > 0


def test_path_like_support() -> None:
    """Path objects and ~ expansion should be accepted."""

    engine = OcrEngine.from_files(model="mock")

    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", (8, 8))
        img.save(tmp.name)
        tmp.flush()

    try:
        result = engine.generate(prompt="<image> test", images=[Path(tmp.name)])
        assert "mock-response" in result.text

        result = engine.generate(prompt="<image> test", images=[tmp.name])
        assert "mock-response" in result.text
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def test_bytes_and_bytearray_support() -> None:
    """Bytes-like inputs should pass through unchanged."""

    engine = OcrEngine.from_files(model="mock")
    img = Image.new("RGB", (8, 8))

    buf = BytesIO()
    img.save(buf, format="PNG")

    result = engine.generate(prompt="<image> bytes", images=[buf.getvalue()])
    assert "mock-response" in result.text

    result = engine.generate(prompt="<image> bytearray", images=[bytearray(buf.getvalue())])
    assert "mock-response" in result.text


def test_multiple_images_roundtrip() -> None:
    engine = OcrEngine.from_files(model="mock")
    images = [_sample_image() for _ in range(3)]

    result = engine.generate(
        prompt="<image> First <image> Second <image> Third",
        images=images,
    )

    assert "mock-response" in result.text
    assert "images=3" in result.text


def test_concurrent_decode_safety() -> None:
    engine = OcrEngine.from_files(model="mock")

    def run_decode(i: int) -> str:
        return engine.generate(prompt=f"<image> Test {i}", images=[_sample_image()]).text

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_decode, range(10)))

    assert len(results) == 10
    assert all("mock-response" in text for text in results)


def test_generation_and_vision_config_passthrough() -> None:
    engine = OcrEngine.from_files(model="mock")

    generation = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    vision = VisionConfig(base_size=768, image_size=512, crop_mode=True)

    result = engine.generate(
        prompt="<image> test",
        images=[_sample_image()],
        generation=generation,
        vision=vision,
    )

    assert "mock-response" in result.text
    assert "sampled" in result.text
    assert "max_tokens=100" in result.text
    assert "base=768" in result.text
    assert "size=512" in result.text
