#!/usr/bin/env python3
"""Convert NuInsSeg tissue images to RGB PNG files.

The unified NuInsSeg annotations expect images at:

    <nuinsseg-root>/<organ>/tissue images/<image_name>.png

This script scans each organ directory for a `tissue images` folder and converts
common image formats inside it to RGB PNG while preserving the directory layout.
It is intended for release users who download/prep NuInsSeg themselves and need
to reproduce the PNG tissue-image paths used by the annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_NUINSSEG_ROOT = Path(__file__).resolve().parents[2] / "images" / "NuInsSeg"
SUPPORTED_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def convert_one(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as img:
        rgb = img.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rgb.save(tmp_path, format="PNG")
        tmp_path.replace(output_path)


def discover_records(source_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    records: list[tuple[Path, Path]] = []
    for tissue_dir in sorted(source_root.glob("*/tissue images")):
        if not tissue_dir.is_dir():
            continue
        organ = tissue_dir.parent.name
        for input_path in sorted(tissue_dir.iterdir()):
            if not input_path.is_file():
                continue
            if input_path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            output_path = output_root / organ / "tissue images" / f"{input_path.stem}.png"
            records.append((input_path, output_path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_NUINSSEG_ROOT,
        help="Root containing <organ>/tissue images/* source files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to --source-root for in-place PNG generation.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for convert_all_sources compatibility; outputs are rewritten when not in dry-run mode.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="After conversion, compare output RGB pixels with source RGB pixels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root: Path = args.source_root
    output_root: Path = args.output_root or source_root
    records = discover_records(source_root, output_root)

    converted = 0
    verified = 0
    failures: list[dict[str, Any]] = []
    ext_counts: dict[str, int] = {}
    examples: list[dict[str, str]] = []

    for input_path, output_path in records:
        ext = input_path.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

        source_hash = rgb_sha256(input_path) if args.verify_existing else None

        if not args.dry_run:
            convert_one(input_path, output_path)
            converted += 1

        if args.verify_existing and not args.dry_run:
            output_hash = rgb_sha256(output_path)
            if output_hash == source_hash:
                verified += 1
            else:
                failures.append(
                    {
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "source_rgb_sha256": source_hash,
                        "output_rgb_sha256": output_hash,
                    }
                )

        if len(examples) < 10:
            examples.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                }
            )

    report = {
        "format": "nuinsseg_tissue_images_to_png_conversion_v1",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "expected_layout": "<root>/<organ>/tissue images/<image_name>.png",
        "supported_extensions": sorted(SUPPORTED_EXTS),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "verify_existing": bool(args.verify_existing),
        "total_source_images": len(records),
        "source_extension_counts": ext_counts,
        "converted": converted,
        "verified_rgb_pixel_match": verified,
        "failure_count": len(failures),
        "failures": failures[:50],
        "examples": examples,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
