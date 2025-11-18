from __future__ import annotations

import pytest
from PIL import Image
from typing import Sequence

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
