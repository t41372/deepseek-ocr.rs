#!/usr/bin/env python3
"""DeepSeek-OCR best-practice sample: receipt to Markdown with streaming."""

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
        description="DeepSeek-OCR receipt extraction with streaming demo"
    )
    parser.add_argument("--image", required=True, help="Path to receipt image")
    parser.add_argument(
        "--prompt",
        default="<image> Convert this receipt to Markdown format",
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
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling")
    parser.add_argument("--base-size", type=int, default=1536, help="Vision base size")
    parser.add_argument(
        "--image-size", type=int, default=1024, help="Vision image size"
    )
    parser.add_argument("--crop-mode", action="store_true", help="Enable crop mode")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")

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
            else home / "model-00001-of-000001.safetensors"
        )

        engine = OcrEngine.from_files(
            model_id="deepseek-ocr",
            config_path=config_path,
            tokenizer_path=tokenizer_path,
            weights_path=weights_path,
            device=args.device,
            dtype=args.dtype,
            template="plain",
            system_prompt="You are an OCR model that returns clean Markdown.",
        )
    else:
        engine = OcrEngine.from_pretrained(
            model_id="deepseek-ocr",
            device=args.device,
            dtype=args.dtype,
            template="plain",
            system_prompt="You are an OCR model that returns clean Markdown.",
        )

    image = _load_image(args.image)
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    vision = VisionConfig(
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=args.crop_mode,
    )

    tokens: list[int] = []

    def _on_stream(count: int, ids: Sequence[int]) -> None:
        if ids:
            tokens.append(ids[-1])
        print(f"\rstreamed {count} tokens", end="", flush=True)

    result = engine.generate(
        prompt=args.prompt,
        images=[image],
        generation=generation,
        vision=vision,
        stream=_on_stream if args.stream else None,
    )

    if args.stream:
        print()
    print(result.text)


if __name__ == "__main__":
    main()
