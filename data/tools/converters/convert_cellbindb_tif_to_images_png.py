#!/usr/bin/env python3
"""Convert CellBinDB TIFF images to the PNG layout used by CLOC annotations.

The release annotations expect CellBinDB source images at:

    images/CellBinDB/images_png/<sample_name>-img.png

The upstream CellBinDB archive extracts to nested sample folders containing:

    images/CellBinDB/CellBinDB/<platform>/<sample>/<sample>-img.tif

This script converts every `*-img.tif` to RGB PNG and writes it to
`images_png` while preserving only the image filename.  It does not modify
annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_RELEASE_ROOT = Path(os.environ.get("CLOC_RELEASE_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_CELLBINDB_ROOT = DEFAULT_RELEASE_ROOT / "images" / "CellBinDB"


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def image_size_mode(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as img:
        return img.size, img.mode


def convert_one(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as img:
        rgb = img.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rgb.save(tmp_path, format="PNG")
        tmp_path.replace(output_path)


def discover_records(cellbindb_root: Path, output_root: Path) -> list[tuple[Path, Path]]:
    source_root = cellbindb_root / "CellBinDB"
    records: list[tuple[Path, Path]] = []
    for tif_path in sorted(source_root.rglob("*-img.tif")):
        records.append((tif_path, output_root / f"{tif_path.stem}.png"))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cellbindb-root",
        type=Path,
        default=DEFAULT_CELLBINDB_ROOT,
        help="Root containing CellBinDB.zip, extracted CellBinDB/, and target images_png/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <cellbindb-root>/images_png.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing PNG files. By default existing outputs are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N discovered records, useful for smoke tests.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, verify output RGB pixels match input.convert('RGB').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cellbindb_root: Path = args.cellbindb_root
    output_root: Path = args.output_root or (cellbindb_root / "images_png")
    records = discover_records(cellbindb_root, output_root)
    if args.limit is not None:
        records = records[: args.limit]

    converted = 0
    skipped_existing = 0
    verified = 0
    failures: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for input_path, output_path in records:
        source_hash = rgb_sha256(input_path) if args.verify and not args.dry_run else None

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
        elif not args.dry_run:
            try:
                convert_one(input_path, output_path)
                converted += 1
            except Exception as exc:  # pragma: no cover - report operational failures.
                failures.append(
                    {
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "error": repr(exc),
                    }
                )
                continue

        if args.verify and not args.dry_run and output_path.exists():
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
            source_size, source_mode = image_size_mode(input_path)
            output_info: dict[str, Any] = {"exists": output_path.exists()}
            if output_path.exists():
                output_size, output_mode = image_size_mode(output_path)
                output_info.update({"size": output_size, "mode": output_mode})
            examples.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "source_size": source_size,
                    "source_mode": source_mode,
                    "output": output_info,
                }
            )

    report = {
        "format": "cellbindb_tif_to_images_png_conversion_v1",
        "cellbindb_root": str(cellbindb_root),
        "source_root": str(cellbindb_root / "CellBinDB"),
        "output_root": str(output_root),
        "expected_annotation_layout": "images/CellBinDB/images_png/<sample_name>-img.png",
        "conversion": "PIL Image.open(tif).convert('RGB') -> PNG",
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "limit": args.limit,
        "total_records_considered": len(records),
        "converted": converted,
        "skipped_existing": skipped_existing,
        "verify": bool(args.verify),
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
