#!/usr/bin/env python3
"""Build the MoNuSeg PNG images referenced by CLOC annotations.

The official MoNuSeg packages provide TIFF images:

    images/MoNuSeg/MoNuSeg 2018 Training Data/Tissue Images/*.tif
    images/MoNuSeg/MoNuSegTestData/*.tif

CLOC annotations use PNG image paths.  This script converts each official TIFF
to an RGB PNG next to its source TIFF, without changing the original extracted
folder structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


# Use absolute() instead of resolve() so a symlinked `tools/` directory still
# treats the visible `data/` folder as the release root.
DEFAULT_MONUSEG_ROOT = Path(__file__).absolute().parents[2] / "images" / "MoNuSeg"


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def iter_tiffs(monuseg_root: Path) -> list[Path]:
    source_roots = [
        monuseg_root / "MoNuSeg 2018 Training Data" / "Tissue Images",
        monuseg_root / "MoNuSegTestData",
    ]
    records: list[Path] = []
    for source_root in source_roots:
        records.extend(sorted(source_root.glob("*.tif")))
        records.extend(sorted(source_root.glob("*.tiff")))
    return records


def convert_one(tif_path: Path, png_path: Path) -> None:
    with Image.open(tif_path) as img:
        rgb = img.convert("RGB")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(png_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monuseg-root", type=Path, default=DEFAULT_MONUSEG_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, compare decoded RGB pixels between TIFF and PNG.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    monuseg_root = args.monuseg_root.resolve()
    records = [(tif_path, tif_path.with_suffix(".png")) for tif_path in iter_tiffs(monuseg_root)]

    converted = 0
    verified = 0
    mismatches: list[dict[str, Any]] = []
    examples: list[dict[str, str]] = []

    for tif_path, png_path in records:
        if not args.dry_run:
            convert_one(tif_path, png_path)
            converted += 1
        if args.verify and png_path.exists():
            tif_hash = rgb_sha256(tif_path)
            png_hash = rgb_sha256(png_path)
            if tif_hash == png_hash:
                verified += 1
            elif len(mismatches) < 20:
                mismatches.append(
                    {
                        "tif_path": str(tif_path),
                        "png_path": str(png_path),
                        "tif_rgb_sha256": tif_hash,
                        "png_rgb_sha256": png_hash,
                    }
                )
        if len(examples) < 8:
            examples.append({"tif_path": str(tif_path), "png_path": str(png_path)})

    report = {
        "format": "monuseg_tif_to_png_conversion_v1",
        "monuseg_root": str(monuseg_root),
        "source_roots": [
            str(monuseg_root / "MoNuSeg 2018 Training Data" / "Tissue Images"),
            str(monuseg_root / "MoNuSegTestData"),
        ],
        "conversion": "PIL TIFF -> RGB -> PNG, saved next to source TIFF",
        "dry_run": bool(args.dry_run),
        "total_tif": len(records),
        "converted": converted,
        "verify_requested": bool(args.verify),
        "verified_rgb_match": verified,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "examples": examples,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
