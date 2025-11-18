"""
Basic OCR example using DeepSeek OCR Python bindings.

This example demonstrates how to perform OCR on a single image using the
deepseek-ocr Python library.
"""

from pathlib import Path

from deepseek_ocr import DecodeParams, Device, OcrModel, Precision, VisionConfig


def main() -> None:
    """Run basic OCR example."""
    # Load the model
    print("Loading DeepSeek OCR model...")
    model = OcrModel(
        model_name="deepseek-ocr",
        device=Device.Cpu,  # Change to Device.Cuda or Device.Metal for GPU
        precision=Precision.Auto,
    )
    print(f"Model loaded: {model.info()}")

    # Configure vision preprocessing
    vision_config = VisionConfig(
        base_size=1024,  # Base resolution
        image_size=640,  # Target image size
        crop_mode=True,  # Enable crop mode
    )

    # Configure decoding parameters
    decode_params = DecodeParams(
        max_new_tokens=512,  # Maximum tokens to generate
        do_sample=False,  # Use greedy decoding
        temperature=0.0,  # Temperature (only used if do_sample=True)
        use_cache=True,  # Enable KV cache for faster inference
    )

    # Path to the image
    image_path = Path("path/to/your/image.png")

    # Perform OCR
    print(f"Performing OCR on {image_path}...")
    result = model.infer(
        prompt="<image> Extract all text from this image",
        images=[image_path],
        vision_config=vision_config,
        decode_params=decode_params,
    )

    # Print results
    print("\n" + "=" * 80)
    print("OCR RESULT")
    print("=" * 80)
    print(result.text)
    print("=" * 80)
    print(f"\nTokens - Prompt: {result.prompt_tokens}, Response: {result.response_tokens}")


if __name__ == "__main__":
    main()
