#!/usr/bin/env python3
"""PaddleOCR-VL sample tuned for low-memory CPU runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from deepseek_ocr import GenerationConfig, OcrEngine, VisionConfig


def _load_image(path: str | Path) -> Image.Image:
    """Load and return a PIL Image from the given path."""
    return Image.open(Path(path).expanduser())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL lightweight example for low-memory systems"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        default="<image> Extract all text from this image",
        help="Prompt template (must contain <image>)",
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
            else home / "model.safetensors"
        )

        engine = OcrEngine.from_files(
            model_id="paddleocr-vl",
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            weights_path=weights_path,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        engine = OcrEngine.from_pretrained(
            model_id="paddleocr-vl",
            device=args.device,
            dtype=args.dtype,
        )

    image = _load_image(args.image)
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    vision = VisionConfig(
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
    )

    result = engine.generate(
        prompt=args.prompt,
        images=[image],
        generation=generation,
        vision=vision,
    )

    print(result.text)


if __name__ == "__main__":
    main()
