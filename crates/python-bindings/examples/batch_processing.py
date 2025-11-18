"""
Batch processing example.

This example shows how to efficiently process multiple images in batch,
reusing the same model instance for better performance.
"""

import json
from pathlib import Path
from typing import Any

from deepseek_ocr import DecodeParams, Device, OcrModel, VisionConfig


def process_image_batch(
    image_paths: list[Path],
    model: OcrModel,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """
    Process a batch of images and save results.

    Args:
        image_paths: List of image paths to process
        model: Loaded OCR model instance
        output_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    vision_config = VisionConfig(base_size=1024, image_size=640, crop_mode=True)
    decode_params = DecodeParams(max_new_tokens=512, use_cache=True)

    for i, image_path in enumerate(image_paths, 1):
        print(f"Processing {i}/{len(image_paths)}: {image_path.name}...")

        try:
            result = model.infer(
                prompt="<image> Extract all text, maintaining the original layout and structure",
                images=[image_path],
                vision_config=vision_config,
                decode_params=decode_params,
            )

            result_dict = {
                "file": str(image_path),
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "response_tokens": result.response_tokens,
                "status": "success",
            }

            # Save individual result
            output_file = output_dir / f"{image_path.stem}_ocr.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)

            results.append(result_dict)
            print(f"  ✓ Extracted {result.response_tokens} tokens")

        except Exception as e:
            error_result = {
                "file": str(image_path),
                "text": "",
                "prompt_tokens": 0,
                "response_tokens": 0,
                "status": "error",
                "error": str(e),
            }
            results.append(error_result)
            print(f"  ✗ Error: {e}")

    return results


def main() -> None:
    """Run batch processing example."""
    # Define input/output paths
    input_dir = Path("input_images")
    output_dir = Path("ocr_results")

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = [
        p for p in input_dir.glob("*") if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process")

    # Load model once
    print("\nLoading model...")
    model = OcrModel(
        model_name="deepseek-ocr-q6k",  # Using quantized model for efficiency
        device=Device.Cpu,
    )
    print(f"Model loaded: {model.info()}\n")

    # Process batch
    results = process_image_batch(image_paths, model, output_dir)

    # Save summary
    summary = {
        "total_images": len(image_paths),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_tokens": sum(r["response_tokens"] for r in results),
        "results": results,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total images: {summary['total_images']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total tokens generated: {summary['total_tokens']}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
