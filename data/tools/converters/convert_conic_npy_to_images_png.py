#!/usr/bin/env python3
"""Export CoNIC image patches from images.npy to the PNG layout used by CLOC.

The release annotations expect CoNIC source images at:

    images/CoNIC/images_png/<patch_name>.png

The official CoNIC challenge package stores all image patches in:

    images/CoNIC/data/images.npy
    images/CoNIC/data/patch_info.csv

This script saves `images.npy[i]` as RGB PNG named after row `i` in
`patch_info.csv`. It does not modify annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_RELEASE_ROOT = Path(os.environ.get("CLOC_RELEASE_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_CONIC_ROOT = DEFAULT_RELEASE_ROOT / "images" / "CoNIC"


def rgb_sha256_from_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def rgb_sha256_from_png(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def load_patch_names(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].strip()
    names = [line.strip() for line in lines[1:] if line.strip()]
    if header != "patch_info":
        raise ValueError(f"Unexpected patch_info.csv header: {header!r}")
    return names


def save_png(arr: np.ndarray, output_path: Path) -> None:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape={arr.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conic-root", type=Path, default=DEFAULT_CONIC_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <conic-root>/images_png.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, verify output RGB pixels match images.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conic_root: Path = args.conic_root
    data_root = conic_root / "data"
    output_root: Path = args.output_root or (conic_root / "images_png")
    patch_names = load_patch_names(data_root / "patch_info.csv")
    images = np.load(data_root / "images.npy", mmap_mode="r")

    if len(patch_names) != images.shape[0]:
        raise ValueError(
            f"patch_info row count ({len(patch_names)}) does not match images.npy first dimension ({images.shape[0]})"
        )

    indices = range(len(patch_names))
    if args.limit is not None:
        indices = range(min(args.limit, len(patch_names)))

    converted = 0
    skipped_existing = 0
    verified = 0
    failures: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for i in indices:
        name = patch_names[i]
        output_path = output_root / f"{name}.png"

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
        elif not args.dry_run:
            try:
                save_png(images[i], output_path)
                converted += 1
            except Exception as exc:  # pragma: no cover - report operational failures.
                failures.append(
                    {
                        "index": i,
                        "patch_name": name,
                        "output_path": str(output_path),
                        "error": repr(exc),
                    }
                )
                continue

        if args.verify and not args.dry_run and output_path.exists():
            source_hash = rgb_sha256_from_array(images[i])
            output_hash = rgb_sha256_from_png(output_path)
            if source_hash == output_hash:
                verified += 1
            else:
                failures.append(
                    {
                        "index": i,
                        "patch_name": name,
                        "output_path": str(output_path),
                        "source_rgb_sha256": source_hash,
                        "output_rgb_sha256": output_hash,
                    }
                )

        if len(examples) < 10:
            output_info: dict[str, Any] = {"exists": output_path.exists()}
            if output_path.exists():
                with Image.open(output_path) as img:
                    output_info.update({"size": img.size, "mode": img.mode})
            examples.append(
                {
                    "index": i,
                    "patch_name": name,
                    "output_path": str(output_path),
                    "source_shape": list(images[i].shape),
                    "source_dtype": str(images.dtype),
                    "output": output_info,
                }
            )

    considered = len(list(indices))
    report = {
        "format": "conic_npy_to_images_png_conversion_v1",
        "conic_root": str(conic_root),
        "source_images_npy": str(data_root / "images.npy"),
        "source_patch_info": str(data_root / "patch_info.csv"),
        "output_root": str(output_root),
        "expected_annotation_layout": "images/CoNIC/images_png/<patch_name>.png",
        "conversion": "images.npy[i] uint8 RGB -> PNG named by patch_info.csv row i",
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "limit": args.limit,
        "images_shape": list(images.shape),
        "images_dtype": str(images.dtype),
        "total_patch_names": len(patch_names),
        "total_records_considered": considered,
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
