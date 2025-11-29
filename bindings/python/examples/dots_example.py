#!/usr/bin/env python3
"""
DotsOCR Model Example

Demonstrates how to use the DotsOCR model - specialized for complex layouts
with ~30-50GB memory requirement. Supports multi-image processing.
"""

from __future__ import annotations

from PIL import Image

from deepseek_ocr_rs import GenerationConfig, OcrEngine, VisionConfig


def main() -> None:
    """Basic example using DotsOCR model."""

    # Load the DotsOCR model
    print("Loading DotsOCR model...")
    print("⚠️  This model requires ~30-50GB RAM")
    engine = OcrEngine.from_pretrained(
        model_id="dots-ocr",  # Complex layout specialist
        device="cpu",  # Options: "cpu", "metal" (macOS), "cuda" (Linux/Windows)
        dtype="f32",  # "f32" for CPU, "f16" for GPU
        template="markdown",  # Markdown template for structured output
    )
    print("Model loaded!\n")

    # Load test images - REPLACE with your own image paths

    # DotsOCR supports multiple images in a single request
    # For demo, we use the same image twice
    images = [
        Image.open("bindings/python/tests/assets/sample.jpg"),
        Image.open("bindings/python/tests/assets/sample.jpg"),
    ]
    print(f"Loaded {len(images)} images\n")

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

    # Run OCR with multiple images
    # IMPORTANT: prompt must have one <image> token per image
    prompt = "<image> <image> Extract text from both images"

    print("Running OCR on multiple images...")
    result = engine.generate(
        prompt=prompt,
        images=images,  # List of images
        generation=generation,
        vision=vision,
    )

    print("\n" + "=" * 80)
    print(result.text)
    print("=" * 80)
    print(f"\nPrompt tokens: {result.prompt_tokens}")
    print(f"Generated tokens: {result.response_tokens}")


if __name__ == "__main__":
    main()
