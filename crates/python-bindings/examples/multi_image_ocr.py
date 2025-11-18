"""
Multi-image OCR example.

This example demonstrates how to perform OCR on multiple images simultaneously,
useful for processing multi-page documents or combining information from
multiple sources.
"""

from pathlib import Path

from deepseek_ocr import DecodeParams, Device, OcrModel, VisionConfig


def main() -> None:
    """Run multi-image OCR example."""
    # Load the model
    print("Loading PaddleOCR-VL model...")
    model = OcrModel(
        model_name="paddleocr-vl",  # Using PaddleOCR for lighter memory footprint
        device=Device.Cpu,
    )

    # Image paths
    images = [
        Path("page1.png"),
        Path("page2.png"),
        Path("page3.png"),
    ]

    # Prompt with multiple image placeholders
    prompt = """<image>
Page 1 content above.

<image>
Page 2 content above.

<image>
Page 3 content above.

Combine all text from the three pages and create a summary."""

    # Perform OCR
    print(f"Performing OCR on {len(images)} images...")
    result = model.infer(
        prompt=prompt,
        images=images,
        vision_config=VisionConfig(base_size=1024, image_size=640, crop_mode=True),
        decode_params=DecodeParams(max_new_tokens=1024, use_cache=True),
    )

    # Print results
    print("\n" + "=" * 80)
    print("MULTI-IMAGE OCR RESULT")
    print("=" * 80)
    print(result.text)
    print("=" * 80)
    print(f"\nTotal tokens - Prompt: {result.prompt_tokens}, Response: {result.response_tokens}")


if __name__ == "__main__":
    main()
