#!/usr/bin/env python3
"""Convert DeepBacs TIFF images to the PNG layout used by CLOC annotations.

The release annotations expect DeepBacs source images at:

    images/DeepBacs/PNGImages/DeepBacs_<subset>_<split>_<name>.png

The official DeepBacs archives extract several task-specific folders containing
TIFF files.  This script reads the release annotations, converts only the
referenced DeepBacs TIFF images, and writes the PNG files expected by the JSONs.
It does not modify annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_RELEASE_ROOT = Path(os.environ.get("CLOC_RELEASE_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_DEEPBACS_ROOT = DEFAULT_RELEASE_ROOT / "images" / "DeepBacs"
DEFAULT_ANNOTATIONS_ROOT = DEFAULT_RELEASE_ROOT / "annotations"

PATH_RE = re.compile(rb"images/DeepBacs/PNGImages/DeepBacs_[^\"\\\s]+\.png")


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def rgb_sha256_from_array(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def iter_deepbacs_annotation_paths(annotations_root: Path) -> list[str]:
    """Scan large JSON files without loading them fully into memory."""

    found: set[str] = set()
    for json_path in sorted(annotations_root.glob("*split*.json")):
        if json_path.name in {"seen_unseen_split.json", "split_summary.json"}:
            continue
        tail = b""
        with json_path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                block = tail + chunk
                for match in PATH_RE.finditer(block):
                    found.add(match.group(0).decode("utf-8"))
                tail = block[-512:]
    return sorted(found)


def normalize_to_uint8_rgb(input_path: Path, force_minmax: bool = False) -> np.ndarray:
    """Match the historical DeepBacs PNG generation.

    L-mode TIFFs were saved through ordinary RGB conversion.  S.aureus 16-bit
    TIFFs were min-max normalized per image to uint8 before saving.
    """

    with Image.open(input_path) as img:
        arr = np.asarray(img)
        if force_minmax or arr.dtype.itemsize > 1:
            arr64 = arr.astype(np.float64)
            min_value = float(arr64.min())
            max_value = float(arr64.max())
            if max_value <= min_value:
                gray = np.zeros(arr.shape, dtype=np.uint8)
            else:
                gray = np.rint((arr64 - min_value) * 255.0 / (max_value - min_value)).clip(0, 255).astype(np.uint8)
            return np.stack([gray] * 3, axis=-1)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def save_png(arr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    Image.fromarray(arr).save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def source_for_target(deepbacs_root: Path, release_path: str) -> tuple[Path, str]:
    name = Path(release_path).name
    if not name.startswith("DeepBacs_") or not name.endswith(".png"):
        raise ValueError(f"Unexpected DeepBacs target filename: {name}")
    stem = name.removeprefix("DeepBacs_").removesuffix(".png")

    prefixes = [
        (
            "E.coli_bf_train_",
            deepbacs_root / "train" / "brightfield",
            "pil_rgb",
        ),
        (
            "E.coli_bf_test_",
            deepbacs_root / "test" / "brightfield",
            "pil_rgb",
        ),
        (
            "B.subtilis_fl_test_",
            deepbacs_root / "StarDist_dataset" / "test" / "fluorescence",
            "pil_rgb",
        ),
        (
            "S.aureus_bf_train_",
            deepbacs_root / "brightfield_dataset" / "train" / "full_images" / "brightfield",
            "minmax_uint8_rgb",
        ),
        (
            "S.aureus_bf_test_",
            deepbacs_root / "brightfield_dataset" / "test" / "brightfield",
            "minmax_uint8_rgb",
        ),
        (
            "S.aureus_fl_train_",
            deepbacs_root / "fluorescence_dataset" / "train" / "full_images" / "fluorescence",
            "minmax_uint8_rgb",
        ),
        (
            "S.aureus_fl_test_",
            deepbacs_root / "fluorescence_dataset" / "test" / "fluorescence",
            "minmax_uint8_rgb",
        ),
    ]

    for prefix, source_dir, conversion in prefixes:
        if stem.startswith(prefix):
            source_stem = stem.removeprefix(prefix)
            return source_dir / f"{source_stem}.tif", conversion
    raise ValueError(f"Cannot map DeepBacs target filename to a source TIFF: {name}")


def image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        return {"size": list(img.size), "mode": img.mode}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepbacs-root", type=Path, default=DEFAULT_DEEPBACS_ROOT)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--verify-old-root",
        type=Path,
        default=None,
        help="Optional old PNGImages directory for exact RGB-pixel hash validation.",
    )
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help="Verify saved PNG RGB pixels against the generated array.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deepbacs_root: Path = args.deepbacs_root
    output_root: Path = args.output_root or (deepbacs_root / "PNGImages")
    target_paths = iter_deepbacs_annotation_paths(args.annotations_root)
    if args.limit is not None:
        target_paths = target_paths[: args.limit]

    converted = 0
    skipped_existing = 0
    verified_output = 0
    verified_old = 0
    failures: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for release_path in target_paths:
        output_path = output_root / Path(release_path).name
        try:
            source_path, conversion = source_for_target(deepbacs_root, release_path)
            if not source_path.exists():
                failures.append(
                    {
                        "release_path": release_path,
                        "output_path": str(output_path),
                        "source_path": str(source_path),
                        "error": "source TIFF does not exist",
                    }
                )
                continue
            generated = normalize_to_uint8_rgb(source_path, force_minmax=(conversion == "minmax_uint8_rgb"))
        except Exception as exc:
            failures.append({"release_path": release_path, "output_path": str(output_path), "error": repr(exc)})
            continue

        generated_hash = rgb_sha256_from_array(generated)
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
        elif not args.dry_run:
            try:
                save_png(generated, output_path)
                converted += 1
            except Exception as exc:  # pragma: no cover - report operational failures.
                failures.append(
                    {
                        "release_path": release_path,
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "error": repr(exc),
                    }
                )
                continue

        if args.verify_output and not args.dry_run and output_path.exists():
            output_hash = rgb_sha256(output_path)
            if output_hash == generated_hash:
                verified_output += 1
            else:
                failures.append(
                    {
                        "release_path": release_path,
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "generated_rgb_sha256": generated_hash,
                        "output_rgb_sha256": output_hash,
                    }
                )

        if args.verify_old_root is not None:
            old_path = args.verify_old_root / Path(release_path).name
            if old_path.exists():
                old_hash = rgb_sha256(old_path)
                if old_hash == generated_hash:
                    verified_old += 1
                else:
                    failures.append(
                        {
                            "release_path": release_path,
                            "source_path": str(source_path),
                            "old_path": str(old_path),
                            "generated_rgb_sha256": generated_hash,
                            "old_rgb_sha256": old_hash,
                        }
                    )
            else:
                failures.append(
                    {
                        "release_path": release_path,
                        "old_path": str(old_path),
                        "error": "old reference PNG does not exist",
                    }
                )

        if len(examples) < 12:
            output = {"exists": output_path.exists()}
            if output_path.exists():
                output.update(image_info(output_path))
            examples.append(
                {
                    "release_path": release_path,
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                    "conversion": conversion,
                    "source": image_info(source_path),
                    "generated_rgb_sha256": generated_hash,
                    "output": output,
                }
            )

    report = {
        "format": "deepbacs_tif_to_pngimages_conversion_v1",
        "deepbacs_root": str(deepbacs_root),
        "annotations_root": str(args.annotations_root),
        "output_root": str(output_root),
        "expected_annotation_layout": "images/DeepBacs/PNGImages/DeepBacs_<subset>_<split>_<name>.png",
        "conversion": {
            "E.coli_bf": "PIL Image.open(tif).convert('RGB')",
            "B.subtilis_fl": "PIL Image.open(tif).convert('RGB') using StarDist_dataset/test/fluorescence",
            "S.aureus_bf_and_fl": "per-image min-max normalization of 16-bit TIFF to uint8, then RGB PNG",
        },
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "limit": args.limit,
        "total_annotation_targets": len(target_paths),
        "converted": converted,
        "skipped_existing": skipped_existing,
        "verify_output": bool(args.verify_output),
        "verified_output_rgb_pixel_match": verified_output,
        "verify_old_root": str(args.verify_old_root) if args.verify_old_root else None,
        "verified_old_rgb_pixel_match": verified_old,
        "failure_count": len(failures),
        "failures": failures[:100],
        "examples": examples,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
