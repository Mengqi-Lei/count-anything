#!/usr/bin/env python3
"""Convert LIVECell TIFF images to the PNG layout used by CLOC annotations.

The release annotations expect LIVECell images at:

    images/LIVECELL/images_png/<image_name>.png

The official image archive extracts TIFF files under:

    images/LIVECELL/images/livecell_train_val_images/*.tif
    images/LIVECELL/images/livecell_test_images/*.tif

This script scans the release annotations, converts only the referenced TIFF
files to RGB PNG, and writes them to `images_png`. It does not modify
annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_RELEASE_ROOT = Path(os.environ.get("CLOC_RELEASE_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_LIVECELL_ROOT = DEFAULT_RELEASE_ROOT / "images" / "LIVECELL"
DEFAULT_ANNOTATIONS_ROOT = DEFAULT_RELEASE_ROOT / "annotations"

PATH_RE = re.compile(rb"images/LIVECELL/images_png/[^\"\\\s]+\.png")


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def iter_livecell_annotation_paths(annotations_root: Path) -> list[str]:
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


def source_for_target(livecell_root: Path, release_path: str) -> Path:
    tif_name = Path(release_path).with_suffix(".tif").name
    candidates = [
        livecell_root / "images" / "livecell_train_val_images" / tif_name,
        livecell_root / "images" / "livecell_test_images" / tif_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No LIVECell source TIFF found for {release_path}: {candidates}")


def convert_one(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as img:
        rgb = img.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        rgb.save(tmp_path, format="PNG")
        tmp_path.replace(output_path)


def image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        return {"size": list(img.size), "mode": img.mode}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--livecell-root", type=Path, default=DEFAULT_LIVECELL_ROOT)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, verify output RGB pixels match input.convert('RGB').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    livecell_root: Path = args.livecell_root
    output_root: Path = args.output_root or (livecell_root / "images_png")
    target_paths = iter_livecell_annotation_paths(args.annotations_root)
    if args.limit is not None:
        target_paths = target_paths[: args.limit]

    converted = 0
    skipped_existing = 0
    verified = 0
    failures: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    source_split_counts = {"livecell_train_val_images": 0, "livecell_test_images": 0}

    for release_path in target_paths:
        output_path = output_root / Path(release_path).name
        try:
            source_path = source_for_target(livecell_root, release_path)
            source_split_counts[source_path.parent.name] = source_split_counts.get(source_path.parent.name, 0) + 1
        except Exception as exc:
            failures.append({"release_path": release_path, "output_path": str(output_path), "error": repr(exc)})
            continue

        source_hash = rgb_sha256(source_path) if args.verify and not args.dry_run else None

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
        elif not args.dry_run:
            try:
                convert_one(source_path, output_path)
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

        if args.verify and not args.dry_run and output_path.exists():
            output_hash = rgb_sha256(output_path)
            if output_hash == source_hash:
                verified += 1
            else:
                failures.append(
                    {
                        "release_path": release_path,
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "source_rgb_sha256": source_hash,
                        "output_rgb_sha256": output_hash,
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
                    "source": image_info(source_path),
                    "output": output,
                }
            )

    report = {
        "format": "livecell_tif_to_images_png_conversion_v1",
        "livecell_root": str(livecell_root),
        "annotations_root": str(args.annotations_root),
        "output_root": str(output_root),
        "expected_annotation_layout": "images/LIVECELL/images_png/<image_name>.png",
        "conversion": "PIL Image.open(tif).convert('RGB') -> PNG",
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "limit": args.limit,
        "total_annotation_targets": len(target_paths),
        "source_split_counts": source_split_counts,
        "converted": converted,
        "skipped_existing": skipped_existing,
        "verify": bool(args.verify),
        "verified_rgb_pixel_match": verified,
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
