"""
Integration tests for deepseek_ocr Python bindings.

These tests require actual model loading and are marked as slow.
They can be skipped in CI with: pytest -m "not slow"
"""

from pathlib import Path
from typing import Any

import pytest

from deepseek_ocr import DecodeParams, Device, OcrModel, VisionConfig

# Test configuration
TEST_MODEL = "paddleocr-vl"  # Use smaller model for faster tests
TEST_DEVICE = Device.Cpu

# Skip integration tests if model resources are not available
# (e.g., in CI environments without model downloads)
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def test_image(tmp_path_factory: Any) -> Path:
    """
    Create a simple test image.

    Args:
        tmp_path_factory: Pytest fixture for temporary directories

    Returns:
        Path to test image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        pytest.skip("Pillow not installed")

    # Create a simple image with text
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Add some text
    text = "Hello, OCR!\nThis is a test image.\n123456789"
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((50, 50), text, fill=(0, 0, 0), font=font)

    # Save image
    tmp_path = tmp_path_factory.mktemp("images")
    image_path = tmp_path / "test.png"
    img.save(str(image_path))

    return image_path


@pytest.fixture(scope="module")
def model() -> OcrModel:
    """
    Load OCR model for tests.

    Returns:
        Loaded OcrModel instance

    Raises:
        pytest.skip: If model loading fails (e.g., model not downloaded)
    """
    try:
        return OcrModel(
            model_name=TEST_MODEL,
            device=TEST_DEVICE,
        )
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_cpu_model(self) -> None:
        """Test loading model on CPU."""
        try:
            model = OcrModel(TEST_MODEL, device=Device.Cpu)
            assert model is not None
            info = model.info()
            assert TEST_MODEL in info
            assert "Cpu" in info
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")

    @pytest.mark.skipif(
        not hasattr(Device, "Cuda"),
        reason="CUDA not available"
    )
    def test_load_cuda_model_if_available(self) -> None:
        """Test loading model on CUDA if available."""
        try:
            model = OcrModel(TEST_MODEL, device=Device.Cuda)
            assert model is not None
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                pytest.skip("CUDA not available on this system")
            raise

    def test_model_info(self, model: OcrModel) -> None:
        """Test model info method."""
        info = model.info()
        assert isinstance(info, str)
        assert len(info) > 0
        assert TEST_MODEL in info.lower() or "paddle" in info.lower()


class TestBasicInference:
    """Test basic OCR inference."""

    def test_single_image_inference(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test OCR on a single image."""
        result = model.infer(
            prompt="<image> Extract all text",
            images=[test_image],
        )

        assert result is not None
        assert isinstance(result.text, str)
        assert result.prompt_tokens > 0
        assert result.response_tokens >= 0

    def test_inference_with_vision_config(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test inference with custom vision config."""
        vision_config = VisionConfig(
            base_size=512,
            image_size=384,
            crop_mode=False,
        )

        result = model.infer(
            prompt="<image> What text is in this image?",
            images=[test_image],
            vision_config=vision_config,
        )

        assert result is not None
        assert isinstance(result.text, str)

    def test_inference_with_decode_params(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test inference with custom decode parameters."""
        decode_params = DecodeParams(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )

        result = model.infer(
            prompt="<image> Extract text",
            images=[test_image],
            decode_params=decode_params,
        )

        assert result is not None
        assert isinstance(result.text, str)
        assert result.response_tokens <= 256  # Should respect max_new_tokens

    def test_inference_with_template(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test inference with conversation template."""
        result = model.infer(
            prompt="<image> Extract the text",
            images=[test_image],
            template="plain",
            system_prompt="You are an OCR assistant.",
        )

        assert result is not None
        assert isinstance(result.text, str)


class TestMultiImageInference:
    """Test multi-image OCR."""

    def test_two_images(
        self,
        model: OcrModel,
        test_image: Path,
        tmp_path: Path,
    ) -> None:
        """Test OCR on two images."""
        # Create second test image
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            pytest.skip("Pillow not installed")

        img2 = Image.new("RGB", (400, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(img2)
        draw.text((20, 20), "Second image", fill=(0, 0, 0))

        image2_path = tmp_path / "test2.png"
        img2.save(str(image2_path))

        # Test with two images
        result = model.infer(
            prompt="<image> First image. <image> Second image. Combine the text.",
            images=[test_image, image2_path],
        )

        assert result is not None
        assert isinstance(result.text, str)
        assert result.prompt_tokens > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_image_not_found(self, model: OcrModel) -> None:
        """Test error when image file doesn't exist."""
        with pytest.raises(RuntimeError, match="Failed to open image|No such file"):
            model.infer(
                prompt="<image> Extract text",
                images=[Path("nonexistent.png")],
            )

    def test_mismatched_image_count(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test error when image count doesn't match placeholders."""
        with pytest.raises(ValueError, match="image.*placeholder"):
            # Prompt has 1 <image>, but we provide 2 images
            model.infer(
                prompt="<image> Extract text",
                images=[test_image, test_image],
            )

        with pytest.raises(ValueError, match="image.*placeholder"):
            # Prompt has 2 <image>s, but we provide 1 image
            model.infer(
                prompt="<image> First. <image> Second.",
                images=[test_image],
            )

    def test_empty_images_list(self, model: OcrModel) -> None:
        """Test error with empty images list."""
        with pytest.raises(ValueError, match="image.*placeholder"):
            model.infer(
                prompt="Extract text",  # No <image> placeholder
                images=[],
            )


class TestOcrResult:
    """Test OcrResult properties."""

    def test_result_attributes(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test OcrResult has expected attributes."""
        result = model.infer(
            prompt="<image> Extract text",
            images=[test_image],
        )

        assert hasattr(result, "text")
        assert hasattr(result, "prompt_tokens")
        assert hasattr(result, "response_tokens")

        assert isinstance(result.text, str)
        assert isinstance(result.prompt_tokens, int)
        assert isinstance(result.response_tokens, int)

        assert result.prompt_tokens > 0
        assert result.response_tokens >= 0

    def test_result_str(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test OcrResult string conversion."""
        result = model.infer(
            prompt="<image> Extract text",
            images=[test_image],
        )

        # __str__ should return the text
        assert str(result) == result.text

    def test_result_repr(
        self,
        model: OcrModel,
        test_image: Path,
    ) -> None:
        """Test OcrResult representation."""
        result = model.infer(
            prompt="<image> Extract text",
            images=[test_image],
        )

        repr_str = repr(result)
        assert "OcrResult" in repr_str
        assert "prompt_tokens" in repr_str or str(result.prompt_tokens) in repr_str


@pytest.mark.parametrize(
    "model_name",
    [
        "paddleocr-vl",
        pytest.param("deepseek-ocr", marks=pytest.mark.slow),
        pytest.param("dots-ocr", marks=pytest.mark.slow),
    ],
)
def test_different_models(
    model_name: str,
    test_image: Path,
) -> None:
    """Test that different models can be loaded and run inference."""
    try:
        model = OcrModel(model_name, device=Device.Cpu)
        result = model.infer(
            prompt="<image> Extract text",
            images=[test_image],
        )
        assert isinstance(result.text, str)
    except Exception as e:
        pytest.skip(f"Model {model_name} not available: {e}")
