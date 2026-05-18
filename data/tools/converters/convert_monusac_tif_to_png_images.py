#!/usr/bin/env python3
"""Build MoNuSAC png_images from the official TIFF files.

CLOC annotations reference:

    images/MoNuSAC/png_images/*.png

The official MoNuSAC archive provides TIFF images under
`MoNuSAC_images_and_annotations/*/*.tif`.  This script rebuilds the PNG folder
expected by the release annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MONUSAC_ROOT = Path(__file__).resolve().parents[2] / "images" / "MoNuSAC"


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def convert_one(tif_path: Path, output_path: Path, *, normal_tiff_rgb: bool) -> None:
    with Image.open(tif_path) as img:
        rgba = np.asarray(img.convert("RGBA"))

    rgb = rgba[:, :, :3]
    if not normal_tiff_rgb:
        # Historical CLOC MoNuSAC PNGs used the display-correct channel order.
        rgb = rgb[:, :, ::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def build_records(monusac_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    source_root = monusac_root / "MoNuSAC_images_and_annotations"
    return [
        (tif_path, output_root / f"{tif_path.stem}.png")
        for tif_path in sorted(source_root.rglob("*.tif"))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monusac-root", type=Path, default=DEFAULT_MONUSAC_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--normal-tiff-rgb",
        action="store_true",
        help="Save TIFF RGB channels directly; default reverses channels to match CLOC history.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    monusac_root = args.monusac_root.resolve()
    output_root = args.output_root or (monusac_root / "png_images")
    records = build_records(monusac_root, output_root)

    converted = 0
    examples: list[dict[str, Any]] = []
    for tif_path, output_path in records:
        if not args.dry_run:
            convert_one(tif_path, output_path, normal_tiff_rgb=args.normal_tiff_rgb)
            converted += 1
        if len(examples) < 5:
            examples.append({"tif_path": str(tif_path), "output_path": str(output_path)})

    report = {
        "format": "monusac_tif_to_png_images_conversion_v1",
        "monusac_root": str(monusac_root),
        "source_root": str(monusac_root / "MoNuSAC_images_and_annotations"),
        "output_root": str(output_root),
        "conversion": (
            "PIL RGBA -> RGB channels -> save PNG as RGB"
            if args.normal_tiff_rgb
            else "PIL RGBA -> RGB channels -> reverse channel order -> save PNG as RGB"
        ),
        "normal_tiff_rgb": bool(args.normal_tiff_rgb),
        "total_tif": len(records),
        "converted": converted,
        "examples": examples,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
