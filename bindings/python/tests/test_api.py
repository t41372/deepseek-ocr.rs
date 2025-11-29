from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
from PIL import Image

from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig, render_prompt


def _sample_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(128, 64, 32))


def test_mock_engine_roundtrip() -> None:
    engine = OcrEngine.from_files(engine="mock")
    prompt = "<image> Describe the receipt."
    result = engine.generate(prompt=prompt, images=[_sample_image()])
    assert "mock-response" in result.text
    assert result.response_tokens > 0


def test_prompt_image_count_validation() -> None:
    engine = OcrEngine.from_files(engine="mock")
    with pytest.raises(ValueError, match=r"Image count mismatch"):
        engine.generate(prompt="<image> <image> ...", images=[_sample_image()])


def test_prompt_image_count_helpful_messages() -> None:
    """Verify helpful error messages for various image count mismatches."""
    engine = OcrEngine.from_files(engine="mock")

    # No <image> token in prompt
    with pytest.raises(
        ValueError, match=r"must contain at least one '<image>' token.*Example"
    ):
        engine.generate(prompt="Extract text", images=[_sample_image()])

    # <image> token but no images provided
    with pytest.raises(
        ValueError, match=r"you provided no images.*Pass images=\[\.\.\.\]"
    ):
        engine.generate(prompt="<image> Extract", images=[])

    # Too many <image> tokens
    with pytest.raises(ValueError, match=r"Add.*<image> token.*to match"):
        engine.generate(prompt="<image> <image> <image>", images=[_sample_image()])

    # Too many images provided
    with pytest.raises(ValueError, match=r"Remove.*<image> token.*from.*to match"):
        engine.generate(
            prompt="<image>", images=[_sample_image(), _sample_image(), _sample_image()]
        )


def test_generation_config_maps_to_native() -> None:
    engine = OcrEngine.from_files(engine="mock")
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
    engine = OcrEngine.from_files(engine="mock")
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

    engine = OcrEngine.from_files(engine="mock")

    for mode in ["RGB", "RGBA", "L"]:
        img = Image.new(mode, (16, 16), color=(100, 100, 100) if mode != "L" else 100)
        result = engine.generate(prompt="<image> test", images=[img])
        assert "mock-response" in result.text
        assert result.response_tokens > 0


def test_path_like_support() -> None:
    """Path objects and ~ expansion should be accepted."""

    engine = OcrEngine.from_files(engine="mock")

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


def test_image_file_not_found_error() -> None:
    """Verify helpful error message when image file doesn't exist."""
    engine = OcrEngine.from_files(engine="mock")

    with pytest.raises(FileNotFoundError) as exc_info:
        engine.generate(
            prompt="<image> test", images=["/nonexistent/path/to/image.jpg"]
        )

    assert "Image file not found" in str(exc_info.value)
    assert "Hint" in str(exc_info.value)


def test_image_path_is_directory_error(tmp_path: Path) -> None:
    """Verify helpful error message when path points to directory."""
    engine = OcrEngine.from_files(engine="mock")
    directory = tmp_path / "some_dir"
    directory.mkdir()

    with pytest.raises(ValueError) as exc_info:
        engine.generate(prompt="<image> test", images=[directory])

    assert "Path is not a file" in str(exc_info.value)
    assert "Hint" in str(exc_info.value)


def test_bytes_and_bytearray_support() -> None:
    """Bytes-like inputs should pass through unchanged."""

    engine = OcrEngine.from_files(engine="mock")
    img = Image.new("RGB", (8, 8))

    buf = BytesIO()
    img.save(buf, format="PNG")

    result = engine.generate(prompt="<image> bytes", images=[buf.getvalue()])
    assert "mock-response" in result.text

    result = engine.generate(
        prompt="<image> bytearray", images=[bytearray(buf.getvalue())]
    )
    assert "mock-response" in result.text


def test_multiple_images_roundtrip() -> None:
    engine = OcrEngine.from_files(engine="mock")
    images = [_sample_image() for _ in range(3)]

    result = engine.generate(
        prompt="<image> First <image> Second <image> Third",
        images=images,
    )

    assert "mock-response" in result.text
    assert "images=3" in result.text


def test_concurrent_decode_safety() -> None:
    engine = OcrEngine.from_files(engine="mock")

    def run_decode(i: int) -> str:
        return engine.generate(
            prompt=f"<image> Test {i}", images=[_sample_image()]
        ).text

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_decode, range(10)))

    assert len(results) == 10
    assert all("mock-response" in text for text in results)


def test_generation_and_vision_config_passthrough() -> None:
    engine = OcrEngine.from_files(engine="mock")

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


def test_coerce_optional_path_handles_home_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepseek_ocr import _api

    sample = tmp_path / "artifact.bin"
    sample.write_bytes(b"ok")
    monkeypatch.setenv("HOME", str(tmp_path))

    expanded = _api._coerce_optional_path(f"~/{sample.name}")
    assert expanded == str(sample)
    assert _api._coerce_optional_path(None) is None


def test_auto_download_invokes_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure auto_download wires through when files are missing."""

    from deepseek_ocr import _api

    calls: list[str] = []

    class DummyResult:
        config_path = str(tmp_path / "config.json")
        tokenizer_path = str(tmp_path / "tokenizer.json")
        weights_path = str(tmp_path / "weights.bin")
        snapshot_path = str(tmp_path / "snapshot.dsq")

    def fake_download(model: str, cache_dir: Any = None) -> type[DummyResult]:
        calls.append(model)
        # Touch files so downstream existence checks would pass if added later.
        Path(DummyResult.config_path).write_text("{}")
        Path(DummyResult.tokenizer_path).write_text("{}")
        Path(DummyResult.weights_path).write_text("{}")
        Path(DummyResult.snapshot_path).write_text("dsq")
        return DummyResult

    missing = tmp_path / "missing.json"
    monkeypatch.setattr("deepseek_ocr._api.download_model", fake_download)

    result = _api._ensure_assets(
        model_id="deepseek",
        config_path=missing,
        tokenizer_path=None,
        weights_path=None,
        snapshot_path=None,
        cache_dir=None,
    )

    assert calls == ["deepseek"]
    assert result[0] == DummyResult.config_path
    assert result[3] == DummyResult.snapshot_path


def test_auto_download_skips_when_present(tmp_path: Path) -> None:
    """If all paths exist, download helper should not be called."""

    from deepseek_ocr import _api

    cfg = tmp_path / "c.json"
    tok = tmp_path / "t.json"
    w = tmp_path / "w.bin"
    for path in (cfg, tok, w):
        path.write_text("{}")

    result = _api._ensure_assets(
        model_id="deepseek",
        config_path=cfg,
        tokenizer_path=tok,
        weights_path=w,
        snapshot_path=None,
        cache_dir=None,
    )

    assert result[0] == cfg
    assert result[1] == tok
    assert result[2] == w


def test_from_files_auto_download_calls_ensure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """from_files should call _ensure_assets when auto_download=True."""

    cfg = tmp_path / "c.json"
    tok = tmp_path / "t.json"
    w = tmp_path / "w.bin"

    def fake_ensure(**kwargs: Any) -> tuple[str, str, str, None]:
        return str(cfg), str(tok), str(w), None

    class DummyHandle:
        def decode(self, *a: Any, **k: Any) -> None:  # pragma: no cover - not used here
            raise RuntimeError("should not decode in this test")

    monkeypatch.setattr("deepseek_ocr._api._ensure_assets", fake_ensure)
    monkeypatch.setattr("deepseek_ocr._native.create_engine", lambda **k: DummyHandle())

    engine = OcrEngine.from_files(engine="mock", model_id="mock-id", auto_download=True)
    assert isinstance(engine, OcrEngine)


def test_from_pretrained_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """from_pretrained should invoke auto_download by default."""

    cfg = tmp_path / "c.json"
    tok = tmp_path / "t.json"
    w = tmp_path / "w.bin"

    monkeypatch.setattr(
        "deepseek_ocr._api._ensure_assets",
        lambda **kwargs: (str(cfg), str(tok), str(w), None),
    )
    monkeypatch.setattr("deepseek_ocr._native.create_engine", lambda **k: object())

    OcrEngine.from_pretrained(model_id="deepseek-ocr")


def test_download_cli_prints_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Smoke test the python -m deepseek_ocr.download CLI wrapper."""

    class Dummy:
        model_id = "deepseek-ocr"
        model_dir = str(tmp_path / "models" / "deepseek-ocr")
        baseline_dir = model_dir
        config_path = str(tmp_path / "config.json")
        tokenizer_path = str(tmp_path / "tokenizer.json")
        weights_path = str(tmp_path / "weights.safetensors")
        snapshot_path = str(tmp_path / "snap.dsq")
        preprocessor_path = str(tmp_path / "pre.json")

    monkeypatch.setattr("deepseek_ocr.download.download_model", lambda *a, **k: Dummy)

    from deepseek_ocr import download as dl

    dl.main(["--model", "deepseek-ocr", "--print-cache"])
    out = capsys.readouterr().out
    assert Dummy.model_dir in out


def test_download_cli_full(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Exercise the non --print-cache code path."""

    class Dummy:
        model_id = "deepseek-ocr"
        model_dir = str(tmp_path / "models" / "deepseek-ocr")
        baseline_dir = model_dir
        config_path = str(tmp_path / "config.json")
        tokenizer_path = str(tmp_path / "tokenizer.json")
        weights_path = str(tmp_path / "weights.safetensors")
        snapshot_path = str(tmp_path / "snap.dsq")
        preprocessor_path = str(tmp_path / "pre.json")

    monkeypatch.setattr("deepseek_ocr.download.download_model", lambda *a, **k: Dummy)

    from deepseek_ocr import download as dl

    dl.main(["--model", "deepseek-ocr"])
    out = capsys.readouterr().out
    assert "config:" in out and Dummy.config_path in out
    assert "snapshot:" in out and Dummy.snapshot_path in out
    assert "preprocessor:" in out and Dummy.preprocessor_path in out


def test_download_model_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """download_model should delegate to the native layer."""

    class Dummy:
        pass

    called: dict[str, Any] = {}

    def fake_native(model_id: str, cache_dir: Any = None) -> Dummy:
        called["model_id"] = model_id
        called["cache_dir"] = cache_dir
        return Dummy()

    import deepseek_ocr._native as native

    monkeypatch.setattr(native, "download_model", fake_native, raising=False)
    from deepseek_ocr._api import download_model

    result = download_model("deepseek-ocr", cache_dir="/tmp/cache")
    assert isinstance(result, Dummy)
    assert called == {"model_id": "deepseek-ocr", "cache_dir": "/tmp/cache"}


def test_ensure_assets_marks_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Quantized model should request snapshot when missing."""

    from deepseek_ocr import _api

    monkeypatch.setattr(
        "deepseek_ocr._api.download_model",
        lambda *a, **k: type(
            "R",
            (),
            {
                "config_path": "c",
                "tokenizer_path": "t",
                "weights_path": "w",
                "snapshot_path": "s",
            },
        ),
    )

    result = _api._ensure_assets(
        model_id="deepseek-ocr-q4k",
        config_path=None,
        tokenizer_path=None,
        weights_path=None,
        snapshot_path=None,
        cache_dir=None,
    )

    assert result[3] == "s"
