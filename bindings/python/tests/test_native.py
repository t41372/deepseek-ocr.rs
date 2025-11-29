from __future__ import annotations

import pytest

from deepseek_ocr_rs import _native


def test_create_engine_requires_paths_for_real_models() -> None:
    with pytest.raises(RuntimeError):
        _native.create_engine("deepseek")


def test_normalize_text_strips_markers() -> None:
    text = _native.normalize_text("hello<｜end▁of▁sentence｜>\r\n")
    assert text == "hello"


def test_engine_for_model_known() -> None:
    assert _native.engine_for_model("deepseek-ocr") == "deepseek"


def test_model_requires_snapshot_flags_quantized() -> None:
    assert _native.model_requires_snapshot("deepseek-ocr-q4k") is True
    assert _native.model_requires_snapshot("deepseek-ocr") is False
