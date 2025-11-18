from __future__ import annotations

import pytest

from deepseek_ocr import _native


def test_create_engine_requires_paths_for_real_models() -> None:
    with pytest.raises(RuntimeError):
        _native.create_engine("deepseek")


def test_normalize_text_strips_markers() -> None:
    text = _native.normalize_text("hello<｜end▁of▁sentence｜>\r\n")
    assert text == "hello"
