"""Command-line interface for downloading DeepSeek OCR models.

This module provides a CLI tool to download model assets and print their
resolved paths. It can be invoked as a script or used via the main() function.

Usage:
    python -m deepseek_ocr_rs.download --model deepseek-ocr
    python -m deepseek_ocr_rs.download --model deepseek-ocr-q4k --cache-dir /custom/path
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ._api import download_model


def main(argv: Sequence[str] | None = None) -> None:
    """Download DeepSeek OCR model assets and print resolved paths.

    This function serves as the entry point for the download CLI. It downloads
    the specified model assets to the cache directory and prints the locations
    of all downloaded files.

    Args:
        argv: Optional command-line arguments. If None, uses sys.argv.
            Expected arguments:
            - --model MODEL_ID (required): Model identifier to download
            - --cache-dir PATH (optional): Custom cache root directory
            - --print-cache (optional): Print only the model cache directory path

    Returns:
        None. Prints download results to stdout.

    Example:
        >>> # Programmatic usage
        >>> main(["--model", "deepseek-ocr-q4k"])
        model: deepseek-ocr-q4k
        model_dir: /Users/user/Library/Caches/deepseek-ocr/models/deepseek-ocr-q4k
        config: /Users/user/Library/Caches/deepseek-ocr/models/deepseek-ocr/config.json
        ...

        >>> # Print only cache directory
        >>> main(["--model", "deepseek-ocr", "--print-cache"])
        /Users/user/Library/Caches/deepseek-ocr/models/deepseek-ocr
    """
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
