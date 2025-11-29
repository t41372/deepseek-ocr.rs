from __future__ import annotations

from typing import Any
import warnings

import pytest
from pathlib import Path
from deepseek_ocr_rs import (
    OcrEngine,
    GenerationConfig,
    VisionConfig,
    get_default_cache_dir,
)
from deepseek_ocr_rs._api import _engine_from_model_id, _resolve_device


def test_engine_from_model_id() -> None:
    assert _engine_from_model_id("deepseek-ocr") == "deepseek"
    assert _engine_from_model_id("paddleocr-vl") == "paddle"
    assert _engine_from_model_id("dots-ocr") == "dots"
    assert _engine_from_model_id("deepseek-ocr-q4k") == "deepseek"
    assert _engine_from_model_id("unknown") == "deepseek"


def test_generation_config_kw_only() -> None:
    with pytest.raises(TypeError):
        GenerationConfig(100)  # type: ignore

    # Should work with kwargs
    config = GenerationConfig(max_new_tokens=100)
    assert config.max_new_tokens == 100


def test_vision_config_kw_only() -> None:
    with pytest.raises(TypeError):
        VisionConfig(1024)  # type: ignore

    config = VisionConfig(base_size=1024)
    assert config.base_size == 1024


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _resolve_device() == "cpu"

    monkeypatch.setenv("DEEPSEEK_OCR_DEVICE", "cuda")
    assert _resolve_device() == "cuda"

    monkeypatch.setenv("DEEPSEEK_OCR_DEVICE", "invalid")
    assert _resolve_device() == "cpu"


def test_get_default_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    # Just check it returns a Path and ends with model_id
    path = get_default_cache_dir("my-model")
    assert isinstance(path, Path)
    assert path.name == "my-model"


def test_from_pretrained_infers_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock from_files to verify arguments
    called: dict[str, Any] = {}

    def fake_from_files(**kwargs: Any) -> str:
        called.update(kwargs)
        return "engine"

    monkeypatch.setattr(OcrEngine, "from_files", fake_from_files)

    OcrEngine.from_pretrained(model_id="paddleocr-vl")

    assert called["engine"] == "paddle"
    assert called["model_id"] == "paddleocr-vl"
    assert called["auto_download"] is True


def test_create_engine_validation() -> None:
    # Should raise ValueError if paths are missing and not auto_download
    with pytest.raises(ValueError, match="config_path and tokenizer_path are required"):
        OcrEngine.from_files(engine="deepseek", auto_download=False)


def test_auto_download_requires_model_id() -> None:
    """auto_download=True requires model_id parameter."""
    with pytest.raises(
        ValueError, match="model_id is required when auto_download=True"
    ):
        OcrEngine.from_files(auto_download=True)


def test_engine_from_model_id_with_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test environment variable override for unknown model IDs."""
    # Set env var for custom model
    monkeypatch.setenv("DEEPSEEK_OCR_ENGINE_CUSTOM_MODEL", "paddle")

    assert _engine_from_model_id("custom-model") == "paddle"

    # Test with different engine
    monkeypatch.setenv("DEEPSEEK_OCR_ENGINE_FOO_BAR", "dots")
    assert _engine_from_model_id("foo-bar") == "dots"

    # Invalid engine value should be ignored
    monkeypatch.setenv("DEEPSEEK_OCR_ENGINE_BAD", "invalid")
    assert _engine_from_model_id("bad") == "deepseek"  # Falls back to default


def test_error_message_for_missing_model_id() -> None:
    """Verify helpful error message when model_id is missing with auto_download."""
    with pytest.raises(ValueError) as exc_info:
        OcrEngine.from_files(auto_download=True)

    assert "model_id is required when auto_download=True" in str(exc_info.value)
    assert "Hint" in str(exc_info.value)
    assert "from_pretrained" in str(exc_info.value)


def test_error_message_for_missing_paths() -> None:
    """Verify helpful error message when paths are missing without auto_download."""
    with pytest.raises(ValueError) as exc_info:
        OcrEngine.from_files(engine="deepseek", auto_download=False)

    assert "config_path and tokenizer_path are required" in str(exc_info.value)
    assert "Hint" in str(exc_info.value)
    assert "from_pretrained" in str(exc_info.value)


def test_device_dtype_compatibility_warnings() -> None:
    """Test that incompatible device/dtype combinations emit warnings."""
    from deepseek_ocr_rs._api import _validate_device_dtype

    # CPU + f16 should warn
    with pytest.warns(UserWarning, match=r"f16 dtype on CPU.*poor performance"):
        _validate_device_dtype("cpu", "f16")

    # CPU + bf16 should warn
    with pytest.warns(UserWarning, match=r"bf16 dtype on CPU.*poor performance"):
        _validate_device_dtype("cpu", "bf16")

    # Metal + bf16 should warn
    with pytest.warns(UserWarning, match=r"Metal device works best with f16 or f32"):
        _validate_device_dtype("metal", "bf16")

    # CPU + f32 should not warn (optimal)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_device_dtype("cpu", "f32")

    # Metal + f16 should not warn (optimal)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_device_dtype("metal", "f16")
