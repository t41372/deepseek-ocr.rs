"""
Model comparison example.

This example demonstrates how to compare different OCR models
(DeepSeek, PaddleOCR, DotsOCR) on the same image.
"""

import time
from pathlib import Path
from typing import Any

from deepseek_ocr import DecodeParams, Device, OcrModel, VisionConfig


def benchmark_model(
    model_name: str,
    image_path: Path,
    prompt: str,
    device: Device = Device.Cpu,
) -> dict[str, Any]:
    """
    Benchmark a single model on an image.

    Args:
        model_name: Name of the model to test
        image_path: Path to the image
        prompt: OCR prompt
        device: Compute device

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'=' * 80}")
    print(f"Testing {model_name}")
    print(f"{'=' * 80}")

    # Load model
    load_start = time.time()
    model = OcrModel(model_name=model_name, device=device)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Prepare configs
    vision_config = VisionConfig(base_size=1024, image_size=640, crop_mode=True)
    decode_params = DecodeParams(max_new_tokens=512, use_cache=True)

    # Run inference
    infer_start = time.time()
    result = model.infer(
        prompt=prompt,
        images=[image_path],
        vision_config=vision_config,
        decode_params=decode_params,
    )
    infer_time = time.time() - infer_start

    print(f"Inference completed in {infer_time:.2f}s")
    print(f"Tokens - Prompt: {result.prompt_tokens}, Response: {result.response_tokens}")
    print(f"\nExtracted text preview (first 200 chars):")
    print("-" * 80)
    print(result.text[:200])
    if len(result.text) > 200:
        print("...")
    print("-" * 80)

    return {
        "model": model_name,
        "load_time_seconds": load_time,
        "inference_time_seconds": infer_time,
        "prompt_tokens": result.prompt_tokens,
        "response_tokens": result.response_tokens,
        "text_length": len(result.text),
        "text": result.text,
    }


def main() -> None:
    """Run model comparison."""
    # Configuration
    image_path = Path("test_image.png")
    prompt = "<image> Extract all text from this document"
    device = Device.Cpu  # Change to Device.Cuda or Device.Metal for GPU

    # Models to compare
    models = [
        "deepseek-ocr",
        "paddleocr-vl",
        "dots-ocr",
    ]

    print(f"Comparing {len(models)} models on {image_path}")
    print(f"Device: {device}")

    # Run benchmarks
    results = []
    for model_name in models:
        try:
            result = benchmark_model(model_name, image_path, prompt, device)
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error with {model_name}: {e}")
            results.append(
                {
                    "model": model_name,
                    "load_time_seconds": 0,
                    "inference_time_seconds": 0,
                    "prompt_tokens": 0,
                    "response_tokens": 0,
                    "text_length": 0,
                    "text": "",
                    "error": str(e),
                }
            )

    # Print comparison summary
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Load (s)':<12} {'Infer (s)':<12} {'Tokens':<10} {'Text Len':<10}")
    print("-" * 80)

    for r in results:
        if "error" not in r:
            print(
                f"{r['model']:<20} {r['load_time_seconds']:<12.2f} "
                f"{r['inference_time_seconds']:<12.2f} {r['response_tokens']:<10} "
                f"{r['text_length']:<10}"
            )
        else:
            print(f"{r['model']:<20} ERROR: {r['error']}")

    print("=" * 80)

    # Find fastest
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        fastest = min(valid_results, key=lambda x: x["inference_time_seconds"])
        print(f"\nFastest inference: {fastest['model']} ({fastest['inference_time_seconds']:.2f}s)")


if __name__ == "__main__":
    main()
