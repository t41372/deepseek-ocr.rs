"""
Unit tests for deepseek_ocr Python bindings.

These tests verify individual components work correctly without requiring
full model loading (which would be slow and resource-intensive).
"""

import pytest

from deepseek_ocr import (
    DecodeParams,
    Device,
    OcrResult,
    Precision,
    VisionConfig,
    list_available_models,
)


class TestVisionConfig:
    """Test VisionConfig class."""

    def test_default_constructor(self) -> None:
        """Test VisionConfig with default parameters."""
        config = VisionConfig()
        assert config.base_size == 1024
        assert config.image_size == 640
        assert config.crop_mode is True

    def test_custom_constructor(self) -> None:
        """Test VisionConfig with custom parameters."""
        config = VisionConfig(base_size=2048, image_size=1024, crop_mode=False)
        assert config.base_size == 2048
        assert config.image_size == 1024
        assert config.crop_mode is False

    def test_repr(self) -> None:
        """Test VisionConfig string representation."""
        config = VisionConfig(base_size=512, image_size=256, crop_mode=True)
        repr_str = repr(config)
        assert "VisionConfig" in repr_str
        assert "512" in repr_str
        assert "256" in repr_str

    def test_mutable_attributes(self) -> None:
        """Test that VisionConfig attributes can be modified."""
        config = VisionConfig()
        config.base_size = 512
        config.image_size = 256
        config.crop_mode = False
        assert config.base_size == 512
        assert config.image_size == 256
        assert config.crop_mode is False


class TestDecodeParams:
    """Test DecodeParams class."""

    def test_default_constructor(self) -> None:
        """Test DecodeParams with default parameters."""
        params = DecodeParams()
        assert params.max_new_tokens == 512
        assert params.do_sample is False
        assert params.temperature == 0.0
        assert params.top_p is None
        assert params.top_k is None
        assert params.repetition_penalty == 1.0
        assert params.no_repeat_ngram_size is None
        assert params.seed is None
        assert params.use_cache is True

    def test_custom_constructor(self) -> None:
        """Test DecodeParams with custom parameters."""
        params = DecodeParams(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            seed=42,
            use_cache=False,
        )
        assert params.max_new_tokens == 1024
        assert params.do_sample is True
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.repetition_penalty == 1.2
        assert params.no_repeat_ngram_size == 3
        assert params.seed == 42
        assert params.use_cache is False

    def test_repr(self) -> None:
        """Test DecodeParams string representation."""
        params = DecodeParams(max_new_tokens=256)
        repr_str = repr(params)
        assert "DecodeParams" in repr_str
        assert "256" in repr_str

    def test_mutable_attributes(self) -> None:
        """Test that DecodeParams attributes can be modified."""
        params = DecodeParams()
        params.max_new_tokens = 2048
        params.temperature = 0.8
        params.seed = 123
        assert params.max_new_tokens == 2048
        assert params.temperature == 0.8
        assert params.seed == 123


class TestDevice:
    """Test Device enum."""

    def test_device_values(self) -> None:
        """Test Device enum values exist."""
        assert Device.Cpu is not None
        assert Device.Cuda is not None
        assert Device.Metal is not None

    def test_device_equality(self) -> None:
        """Test Device enum equality."""
        assert Device.Cpu == Device.Cpu
        assert Device.Cuda == Device.Cuda
        assert Device.Metal == Device.Metal
        assert Device.Cpu != Device.Cuda


class TestPrecision:
    """Test Precision enum."""

    def test_precision_values(self) -> None:
        """Test Precision enum values exist."""
        assert Precision.F16 is not None
        assert Precision.F32 is not None
        assert Precision.BF16 is not None
        assert Precision.Auto is not None

    def test_precision_equality(self) -> None:
        """Test Precision enum equality."""
        assert Precision.F16 == Precision.F16
        assert Precision.F32 == Precision.F32
        assert Precision.Auto != Precision.F16


class TestListAvailableModels:
    """Test list_available_models function."""

    def test_returns_list(self) -> None:
        """Test that list_available_models returns a list."""
        models = list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_contains_expected_models(self) -> None:
        """Test that expected models are in the list."""
        models = list_available_models()
        # DeepSeek models
        assert "deepseek-ocr" in models
        assert "deepseek-ocr-q4k" in models
        assert "deepseek-ocr-q6k" in models
        assert "deepseek-ocr-q8k" in models
        # PaddleOCR models
        assert "paddleocr-vl" in models
        assert "paddleocr-vl-q4k" in models
        # DotsOCR models
        assert "dots-ocr" in models
        assert "dots-ocr-q6k" in models

    def test_all_strings(self) -> None:
        """Test that all model names are strings."""
        models = list_available_models()
        for model in models:
            assert isinstance(model, str)
            assert len(model) > 0


class TestOcrResult:
    """Test OcrResult class (without actual model loading)."""

    # Note: OcrResult is created by Rust code, so we can't directly
    # construct it in Python. These tests would need actual model
    # inference, which belongs in integration tests.

    pass


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_vision_config_extreme_values(self) -> None:
        """Test VisionConfig with extreme values."""
        # Very small values
        config = VisionConfig(base_size=64, image_size=32, crop_mode=False)
        assert config.base_size == 64
        assert config.image_size == 32

        # Very large values
        config = VisionConfig(base_size=8192, image_size=4096, crop_mode=True)
        assert config.base_size == 8192
        assert config.image_size == 4096

    def test_decode_params_edge_values(self) -> None:
        """Test DecodeParams with edge values."""
        # Zero tokens
        params = DecodeParams(max_new_tokens=0)
        assert params.max_new_tokens == 0

        # Very high temperature
        params = DecodeParams(temperature=2.0)
        assert params.temperature == 2.0

        # Zero temperature
        params = DecodeParams(temperature=0.0)
        assert params.temperature == 0.0

    def test_decode_params_optional_none(self) -> None:
        """Test DecodeParams with explicit None values."""
        params = DecodeParams(
            top_p=None,
            top_k=None,
            no_repeat_ngram_size=None,
            seed=None,
        )
        assert params.top_p is None
        assert params.top_k is None
        assert params.no_repeat_ngram_size is None
        assert params.seed is None
