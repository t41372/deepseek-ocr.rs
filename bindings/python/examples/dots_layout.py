#!/usr/bin/env python3
"""DotsOCR sample for layout-heavy pages with multi-image prompts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig


def _load_images(paths: Sequence[str | Path]) -> list[Image.Image]:
    """Load multiple images from the given paths."""
    return [Image.open(Path(p).expanduser()) for p in paths]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DotsOCR example for complex layouts with multiple images"
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Paths to input images"
    )
    parser.add_argument(
        "--prompt",
        default="<image> Extract text and preserve layout structure",
        help="Prompt template (must contain <image> for each input image)",
    )
    parser.add_argument(
        "--model-home", help="Directory containing config.json, tokenizer.json, weights"
    )
    parser.add_argument("--weights", help="Path to model weights file")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "metal"],
        help="Execution device",
    )
    parser.add_argument(
        "--dtype", choices=["f32", "f16", "bf16"], help="Data type (default: auto)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--base-size", type=int, default=1536, help="Vision base size")
    parser.add_argument(
        "--image-size", type=int, default=1024, help="Vision image size"
    )
    parser.add_argument("--crop-mode", action="store_true", help="Enable crop mode")

    args = parser.parse_args(argv)

    # Build engine from explicit paths or auto-download
    if args.model_home or args.weights:
        home = (
            Path(args.model_home).expanduser()
            if args.model_home
            else Path(args.weights).parent
        )
        config_path = home / "config.json"
        tokenizer_path = home / "tokenizer.json"
        weights_path = (
            Path(args.weights).expanduser()
            if args.weights
            else home / "model.safetensors.index.json"
        )

        engine = OcrEngine.from_files(
            model_id="dots-ocr",
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            weights_path=weights_path,
            device=args.device,
            dtype=args.dtype,
            template="markdown",
        )
    else:
        engine = OcrEngine.from_pretrained(
            model_id="dots-ocr",
            device=args.device,
            dtype=args.dtype,
            template="markdown",
        )

    images = _load_images(args.images)
    vision = VisionConfig(
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
    )
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    result = engine.generate(
        prompt=args.prompt,
        images=images,
        generation=generation,
        vision=vision,
    )

    print(result.text)


if __name__ == "__main__":
    main()
