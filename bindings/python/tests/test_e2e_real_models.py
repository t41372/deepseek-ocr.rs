from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, cast

import pytest
from PIL import Image

from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig

E2E_ENABLED = os.environ.get("DEEPSEEK_OCR_E2E") == "1"
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not E2E_ENABLED, reason="set DEEPSEEK_OCR_E2E=1 to run real-model tests"
    ),
]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SAMPLE_IMAGE = ASSETS_DIR / "sample.jpg"
EXPECTED_TEXT = (
    "Rust implementation of the DeepSeek-OCR inference stack with a fast CLI and an "
    "OpenAI-compatible HTTP server. The workspace packages multiple OCR backends, prompt "
    "tooling, and a serving layer so you can build document understanding pipelines that "
    "run locally on CPU, Apple Metal, or (alpha) NVIDIA CUDA GPUs."
)


def _engine_label(model_id: str) -> Literal["deepseek", "paddle", "dots"]:
    lower = model_id.lower()
    if "paddle" in lower:
        return "paddle"
    if "dots" in lower:
        return "dots"
    return "deepseek"


def _model_home() -> Path:
    root = os.environ.get("DEEPSEEK_OCR_E2E_MODEL_HOME")
    if not root:
        pytest.skip("DEEPSEEK_OCR_E2E_MODEL_HOME is unset")
    path = Path(root).expanduser()
    if not path.exists():
        pytest.skip(f"model home {path} does not exist")
    return path


def _prompt() -> str:
    return os.environ.get(
        "DEEPSEEK_OCR_E2E_PROMPT",
        "<image> Extract the visible text as Markdown.",
    )


def _select_file(home: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(home.glob(pattern))
        if matches:
            return matches[0]
    return None


def _resolve_assets(home: Path) -> tuple[Path, Path, Path, Path | None]:
    config = home / "config.json"
    tokenizer = home / "tokenizer.json"
    weights_override = os.environ.get("DEEPSEEK_OCR_E2E_WEIGHTS")
    weights = Path(weights_override).expanduser() if weights_override else None
    snapshot = _select_file(home, ["*.dsq"])
    if weights is None:
        weights = _select_file(home, ["*.safetensors.index.json", "*.safetensors"])
    if not config.exists() or not tokenizer.exists() or weights is None:
        pytest.skip("missing model artifacts under DEEPSEEK_OCR_E2E_MODEL_HOME")
    return config, tokenizer, weights, snapshot


def _load_images() -> list[Image.Image]:
    raw = os.environ.get("DEEPSEEK_OCR_E2E_IMAGES")
    if raw:
        return [Image.open(Path(p).expanduser()) for p in raw.split(",") if p]
    if not SAMPLE_IMAGE.exists():
        pytest.skip(f"sample asset missing at {SAMPLE_IMAGE}")
    return [Image.open(SAMPLE_IMAGE)]


def _strip_spaces(text: str) -> str:
    return "".join(ch for ch in text.lower() if not ch.isspace())


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def test_real_model_roundtrip() -> None:
    home = _model_home()
    config, tokenizer, weights, snapshot = _resolve_assets(home)
    model_id = os.environ.get("DEEPSEEK_OCR_E2E_MODEL_ID", "deepseek-ocr")

    device_str = os.environ.get("DEEPSEEK_OCR_E2E_DEVICE", "cpu")
    device = cast(Literal["cpu", "cuda", "metal"], device_str)

    dtype_str = os.environ.get("DEEPSEEK_OCR_E2E_DTYPE")
    dtype: Literal["f32", "f16", "bf16"] | None = None
    if dtype_str in ("f32", "f16", "bf16"):
        dtype = cast(Literal["f32", "f16", "bf16"], dtype_str)

    engine = OcrEngine.from_files(
        engine=_engine_label(model_id),
        config_path=config,
        tokenizer_path=tokenizer,
        weights_path=weights,
        snapshot_path=snapshot,
        device=device,
        dtype=dtype,
        template=os.environ.get("DEEPSEEK_OCR_E2E_TEMPLATE", "plain"),
        system_prompt=os.environ.get("DEEPSEEK_OCR_E2E_SYSTEM_PROMPT", ""),
    )

    result = engine.generate(
        prompt=_prompt(),
        images=_load_images(),
        generation=GenerationConfig(
            max_new_tokens=int(os.environ.get("DEEPSEEK_OCR_E2E_MAX_TOKENS", "256")),
            temperature=float(os.environ.get("DEEPSEEK_OCR_E2E_TEMPERATURE", "0")),
        ),
        vision=VisionConfig(
            base_size=int(os.environ.get("DEEPSEEK_OCR_E2E_BASE_SIZE", "1024")),
            image_size=int(os.environ.get("DEEPSEEK_OCR_E2E_IMAGE_SIZE", "768")),
            crop_mode=os.environ.get("DEEPSEEK_OCR_E2E_CROP_MODE", "0") == "1",
        ),
    )

    normalized = _strip_spaces(result.text)
    target = _strip_spaces(EXPECTED_TEXT)
    distance = _levenshtein(normalized, target)
    tolerance = max(5, len(target) // 20)  # allow ~5% edits

    assert normalized == target or distance <= tolerance, (
        f"levenshtein distance too high ({distance} > {tolerance}); "
        f"first 120 chars: {result.text[:120]!r}"
    )
    assert result.response_tokens > 0
