"""Convenience CLI to download DeepSeek OCR models and print resolved paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ._api import download_model


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download DeepSeek OCR model assets")
    parser.add_argument(
        "--model",
        required=True,
        help="model id (e.g. deepseek-ocr, paddleocr-vl, dots-ocr, deepseek-ocr-q4k)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="override cache root (models/<id> will be created inside)",
    )
    parser.add_argument(
        "--print-cache", action="store_true", help="print model cache dir path and exit"
    )
    args = parser.parse_args(argv)

    result = download_model(args.model, cache_dir=args.cache_dir)
    if args.print_cache:
        print(result.model_dir)
        return

    print(f"model: {result.model_id}")
    print(f"model_dir: {result.model_dir}")
    print(f"config: {result.config_path}")
    print(f"tokenizer: {result.tokenizer_path}")
    print(f"weights: {result.weights_path}")
    if result.snapshot_path:
        print(f"snapshot: {result.snapshot_path}")
    if result.preprocessor_path:
        print(f"preprocessor: {result.preprocessor_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
