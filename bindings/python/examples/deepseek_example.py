#!/usr/bin/env python3
"""
DeepSeek-OCR Model Example

Demonstrates how to use the DeepSeek-OCR model - the highest accuracy model
with ~13GB memory requirement. Shows basic usage with streaming support.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from deepseek_ocr_rs import GenerationConfig, OcrEngine, VisionConfig


def main() -> None:
    """Basic example using DeepSeek-OCR model."""

    # Load the DeepSeek-OCR model
    print("Loading DeepSeek-OCR model...")
    engine = OcrEngine.from_pretrained(
        model_id="deepseek-ocr-q6k",  # Day‑to‑day balance for mid‑range GPUs
        device="cpu",  # Options: "cpu", "metal" (macOS), "cuda" (Linux/Windows)
        dtype="f32",  # "f32" for CPU, "f16" for GPU
        template="plain",  # Template: "plain", "deepseek", "deepseekv2", "alignment"
    )
    print("Model loaded!\n")

    # Load test image - REPLACE with your own image path
    image = Image.open("bindings/python/tests/assets/sample.jpg")
    print(f"Loaded image: {image.size[0]}x{image.size[1]} pixels\n")

    # Configure generation parameters
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

    # Configure vision parameters
    vision = VisionConfig(
        base_size=1024,  # Global view resolution (defaults to 1024)
        image_size=640,  # Local crop resolution (defaults to 640)
        crop_mode=True,  # Enable/disable dynamic crop mode (true/false)
    )

    # Run OCR with streaming
    token_count = 0

    def on_stream(count: int, token_ids: Sequence[int]) -> None:
        """Callback for streaming - called for each generated token.

        Note: This callback receives token IDs, not the decoded text.
        To stream text, you would need to use a tokenizer to decode the IDs.
        For this example, we just show the progress.
        """
        nonlocal token_count
        token_count = count
        print(f"\rGenerating... {count} tokens", end="", flush=True)

    print("Running OCR...")
    result = engine.generate(
        prompt="<image> Extract all text from this image",
        images=[image],
        generation=generation,
        vision=vision,
        stream=on_stream,  # Optional: remove for no streaming
    )

    print("\n\n" + "=" * 80)
    print(result.text)
    print("=" * 80)
    print(f"\nGenerated {token_count} tokens")


if __name__ == "__main__":
    main()
