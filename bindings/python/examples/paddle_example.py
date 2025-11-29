#!/usr/bin/env python3
"""
PaddleOCR-VL Model Example

Demonstrates how to use the PaddleOCR-VL model - a lighter, faster model
with ~9GB memory requirement. Good for systems with 16GB RAM.
"""

from __future__ import annotations

from PIL import Image

from deepseek_ocr_rs import GenerationConfig, OcrEngine, VisionConfig


def main() -> None:
    """Basic example using PaddleOCR-VL model."""

    # Load the PaddleOCR-VL model
    print("Loading PaddleOCR-VL model...")
    engine = OcrEngine.from_pretrained(
        model_id="paddleocr-vl-q6k",  # blends of accuracy and footprint.
        device="cpu",  # Options: "cpu", "metal" (macOS), "cuda" (Linux/Windows)
        dtype="f32",  # "f32" for CPU, "f16" for GPU
    )
    print("Model loaded!\n")

    # Load test image - REPLACE with your own image path
    image = Image.open("bindings/python/tests/assets/sample.jpg")
    print(f"Loaded image: {image.size[0]}x{image.size[1]} pixels\n")

    # Configure parameters
    generation = GenerationConfig(
        max_new_tokens=512,  # Maximum number of tokens to generate
        do_sample=False,  # Enable sampling during decoding (true/false)
        temperature=0.0,  # Softmax temperature for sampling
        top_p=1.0,  # Nucleus sampling probability mass
        top_k=None,  # Top-k sampling cutoff
        repetition_penalty=1.0,  # Repetition penalty (>1 decreases repetition)
        no_repeat_ngram_size=20,  # Enforce no-repeat n-gram constraint of the given size
        seed=None,  # RNG seed for sampling
        use_cache=True,  # Enable KV-cache usage during decoding
    )

    vision = VisionConfig(
        base_size=1024,  # Global view resolution (defaults to 1024)
        image_size=640,  # Local crop resolution (defaults to 640)
        crop_mode=True,  # Enable/disable dynamic crop mode (true/false)
    )

    # Run OCR
    print("Running OCR...")
    result = engine.generate(
        prompt="<image> Extract text",
        images=[image],
        generation=generation,
        vision=vision,
    )
    print("Result:")
    print("-" * 80)
    print(result.text)
    print("-" * 80)

    print(f"\nGenerated tokens: {result.response_tokens}")


if __name__ == "__main__":
    main()
