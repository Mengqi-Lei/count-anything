#!/usr/bin/env python3
"""Rebuild the Objects365 high-resolution derived images.

This script has two modes:

1. ``make-recipe``: inspect the current local high-resolution outputs and
   write a pixel-free recipe JSON. The recipe records only source image paths
   and deterministic crop/resize parameters.
2. ``generate``: given the recipe and a local Objects365-2020 image root,
   regenerate the high-resolution images without redistributing the originals
   or derived pixels.

The implementation follows the high-resolution processing rules in
``scripts_old/repair_high_resolution.py``. The recipe layer is intentionally
used because the final split JSON records no longer store ``original_idx`` or
explicit crop boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


STRICT_MAX_SIDE = 2048
GRID_TARGET_CELL_SIZE = 1024
GRID_ASPECT_RATIO_THRESHOLD_LOW = 0.5
GRID_ASPECT_RATIO_THRESHOLD_HIGH = 2.0
ZOOM_TARGET_SIZE = 1024

HIGH_RES_RE = re.compile(
    r"(?:^|/)high_resolution/Objects365_unified/"
    r"(cropped|zoomed|grid)/(Objects365_unified_(\d+)_(\d+)\.png)$"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_high_res_path(path: str) -> dict[str, Any] | None:
    m = HIGH_RES_RE.search(str(path).replace("\\", "/"))
    if not m:
        return None
    strategy, filename, source_idx, seq = m.groups()
    return {
        "strategy": strategy,
        "filename": filename,
        "source_idx": source_idx,
        "seq": int(seq),
        "output_relpath": f"high_resolution/Objects365_unified/{strategy}/{filename}",
    }


def path_under_objects365_root(image_path: str, objects365_root: Path) -> Path:
    """Map stale absolute paths to a caller-provided Objects365-2020 root."""
    p = str(image_path).replace("\\", "/")
    marker = "Objects365-2020/"
    if marker in p:
        return objects365_root / p.split(marker, 1)[1]
    candidate = Path(p)
    if candidate.exists():
        return candidate
    return objects365_root / p.lstrip("/")


def source_relpath(image_path: str) -> str:
    p = str(image_path).replace("\\", "/")
    marker = "Objects365-2020/"
    if marker in p:
        return p.split(marker, 1)[1]
    return p


def collect_targets(target_jsons: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Collect unique Objects365 high-res records from one or more JSON files."""
    targets: dict[str, dict[str, Any]] = {}
    for path in target_jsons:
        data = load_json(path)
        for record_key, record in data.items():
            parsed = parse_high_res_path(record.get("image_path", ""))
            if not parsed:
                continue
            out_rel = parsed["output_relpath"]
            if out_rel in targets:
                continue
            targets[out_rel] = {
                **parsed,
                "record_key": str(record_key),
                "record_classes": list(record.get("classes", [])),
                "record_annotation": record.get("annotation", {}),
                "reference_image_path": record.get("image_path", ""),
            }
    return targets


def grid_cells(width: int, height: int) -> list[tuple[int, int, int, int]]:
    num_x = max(1, math.ceil(width / STRICT_MAX_SIDE))
    num_y = max(1, math.ceil(height / STRICT_MAX_SIDE))
    aspect_ratio = width / height if height else 1.0

    if GRID_ASPECT_RATIO_THRESHOLD_LOW < aspect_ratio < GRID_ASPECT_RATIO_THRESHOLD_HIGH:
        num_x = max(num_x, 2)
        num_y = max(num_y, 2)
    elif aspect_ratio >= GRID_ASPECT_RATIO_THRESHOLD_HIGH:
        num_x = max(num_x, math.ceil(width / GRID_TARGET_CELL_SIZE))
    else:
        num_y = max(num_y, math.ceil(height / GRID_TARGET_CELL_SIZE))

    num_x = max(num_x, math.ceil(width / STRICT_MAX_SIDE))
    num_y = max(num_y, math.ceil(height / STRICT_MAX_SIDE))
    cell_w = math.ceil(width / num_x)
    cell_h = math.ceil(height / num_y)

    cells = []
    for j in range(num_y):
        for i in range(num_x):
            x1 = i * cell_w
            y1 = j * cell_h
            x2 = min((i + 1) * cell_w, width)
            y2 = min((j + 1) * cell_h, height)
            cells.append((x1, y1, x2, y2))
    return cells


def image_digest(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    h = hashlib.sha256()
    h.update(rgb.tobytes())
    h.update(str(rgb.size).encode("ascii"))
    return h.hexdigest()


def images_equal(a: Image.Image, b: Image.Image) -> bool:
    if a.size != b.size:
        return False
    return ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None


def iter_points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for details in annotation.values():
        for point in details.get("point", []) or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))
    return points


def infer_crop_box_from_points(
    raw_annotation: dict[str, Any],
    target_annotation: dict[str, Any],
    target_size: tuple[int, int],
    source_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Infer translation crop box by matching target points to raw points."""
    raw_points = iter_points(raw_annotation)
    target_points = iter_points(target_annotation)
    if not raw_points or not target_points:
        return None

    width, height = target_size
    source_w, source_h = source_size
    votes: Counter[tuple[int, int]] = Counter()

    for tx, ty in target_points:
        for rx, ry in raw_points:
            dx = rx - tx
            dy = ry - ty
            ix = int(round(dx))
            iy = int(round(dy))
            if abs(dx - ix) > 1e-3 or abs(dy - iy) > 1e-3:
                continue
            if ix < 0 or iy < 0 or ix + width > source_w or iy + height > source_h:
                continue
            votes[(ix, iy)] += 1

    if not votes:
        return None
    (x1, y1), _ = votes.most_common(1)[0]
    return (x1, y1, x1 + width, y1 + height)


def make_recipe(args: argparse.Namespace) -> None:
    raw = load_json(args.raw_objects365_json)
    targets = collect_targets(args.target_json)
    entries: list[dict[str, Any]] = []
    stats = Counter()
    failures: list[dict[str, Any]] = []

    print(f"Collected {len(targets)} target high-resolution records.")

    for i, target in enumerate(targets.values(), 1):
        if i % 250 == 0:
            print(f"  processed {i}/{len(targets)}")

        source_idx = target["source_idx"]
        raw_record = raw.get(source_idx)
        if raw_record is None:
            failures.append({**target, "reason": "source_idx_missing_in_raw_json"})
            continue

        source_path = path_under_objects365_root(raw_record.get("image_path", ""), args.objects365_root)
        reference_path = Path(target["reference_image_path"])
        if not source_path.exists():
            failures.append({**target, "reason": "source_image_missing", "source_path": str(source_path)})
            continue
        if not reference_path.exists():
            failures.append({**target, "reason": "reference_image_missing", "reference_path": str(reference_path)})
            continue

        with Image.open(source_path) as source_img, Image.open(reference_path) as ref_img:
            source_img = source_img.convert("RGB")
            ref_img = ref_img.convert("RGB")
            source_w, source_h = source_img.size
            ref_w, ref_h = ref_img.size
            strategy = target["strategy"]

            entry: dict[str, Any] = {
                "output_relpath": target["output_relpath"],
                "strategy": strategy,
                "source_idx": source_idx,
                "source_image_relpath": source_relpath(raw_record.get("image_path", "")),
                "reference_size": [ref_w, ref_h],
                "reference_sha256_rgb": image_digest(ref_img),
            }

            if strategy == "zoomed":
                scale = ZOOM_TARGET_SIZE / max(source_w, source_h)
                expected_size = (int(source_w * scale), int(source_h * scale))
                if expected_size != (ref_w, ref_h):
                    failures.append({
                        **target,
                        "reason": "zoom_size_mismatch",
                        "expected_size": expected_size,
                        "reference_size": (ref_w, ref_h),
                    })
                    continue
                entry["resize_size"] = [ref_w, ref_h]

            elif strategy == "grid":
                matched_box = None
                for box in grid_cells(source_w, source_h):
                    x1, y1, x2, y2 = box
                    if (x2 - x1, y2 - y1) != (ref_w, ref_h):
                        continue
                    if images_equal(source_img.crop(box), ref_img):
                        matched_box = box
                        break
                if matched_box is None:
                    matched_box = infer_crop_box_from_points(
                        raw_record.get("annotation", {}),
                        target["record_annotation"],
                        (ref_w, ref_h),
                        (source_w, source_h),
                    )
                if matched_box is None or not images_equal(source_img.crop(matched_box), ref_img):
                    failures.append({**target, "reason": "grid_crop_box_not_inferred"})
                    continue
                entry["crop_box"] = list(matched_box)

            elif strategy == "cropped":
                crop_box = infer_crop_box_from_points(
                    raw_record.get("annotation", {}),
                    target["record_annotation"],
                    (ref_w, ref_h),
                    (source_w, source_h),
                )
                if crop_box is None or not images_equal(source_img.crop(crop_box), ref_img):
                    failures.append({**target, "reason": "cropped_crop_box_not_inferred"})
                    continue
                entry["crop_box"] = list(crop_box)

            else:
                failures.append({**target, "reason": f"unknown_strategy:{strategy}"})
                continue

            entries.append(entry)
            stats[strategy] += 1

    recipe = {
        "format": "objects365_high_resolution_recipe",
        "version": 1,
        "description": "Pixel-free recipe for regenerating Objects365 high-resolution processed images.",
        "constants": {
            "strict_max_side": STRICT_MAX_SIDE,
            "grid_target_cell_size": GRID_TARGET_CELL_SIZE,
            "zoom_target_size": ZOOM_TARGET_SIZE,
        },
        "stats": {
            "targets": len(targets),
            "entries": len(entries),
            "failures": len(failures),
            "by_strategy": dict(stats),
        },
        "entries": entries,
        "failures": failures[: args.max_failures_in_recipe],
    }
    dump_json(recipe, args.output_recipe)
    print(f"Wrote recipe: {args.output_recipe}")
    print(json.dumps(recipe["stats"], ensure_ascii=False, indent=2))
    if failures:
        print(f"WARNING: {len(failures)} records failed. See recipe failures section.", file=sys.stderr)
        if not args.allow_failures:
            sys.exit(2)


def generate(args: argparse.Namespace) -> None:
    recipe = load_json(args.recipe)
    entries = recipe.get("entries", [])
    stats = Counter()

    for i, entry in enumerate(entries, 1):
        if i % 250 == 0:
            print(f"  generated {i}/{len(entries)}")
        source_path = args.objects365_root / entry["source_image_relpath"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        output_path = args.output_root / entry["output_relpath"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            stats["skipped_exists"] += 1
            continue

        with Image.open(source_path) as img:
            img = img.convert("RGB")
            if entry["strategy"] == "zoomed":
                out = img.resize(tuple(entry["resize_size"]), Image.Resampling.LANCZOS)
            else:
                out = img.crop(tuple(entry["crop_box"]))
            out.save(output_path, format="PNG")

        stats[entry["strategy"]] += 1

    print("Generation complete.")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    make = sub.add_parser("make-recipe", help="Create a pixel-free high-res recipe from local reference outputs.")
    make.add_argument("--target-json", type=Path, action="append", required=True, help="Final split/unified JSON containing high-res records. Repeatable.")
    make.add_argument("--raw-objects365-json", type=Path, required=True, help="Raw/pre-high-res Objects365_unified.json.")
    make.add_argument("--objects365-root", type=Path, required=True, help="Path to Objects365-2020 image root.")
    make.add_argument("--output-recipe", type=Path, required=True)
    make.add_argument("--allow-failures", action="store_true")
    make.add_argument("--max-failures-in-recipe", type=int, default=200)
    make.set_defaults(func=make_recipe)

    gen = sub.add_parser("generate", help="Regenerate high-res images from a recipe.")
    gen.add_argument("--recipe", type=Path, required=True)
    gen.add_argument("--objects365-root", type=Path, required=True)
    gen.add_argument("--output-root", type=Path, required=True, help="Root under which high_resolution/Objects365_unified/... is created.")
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=generate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
