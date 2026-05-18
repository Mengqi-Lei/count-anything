#!/usr/bin/env python3
"""Rebuild high-resolution derived images from a pixel-free recipe.

This tool rebuilds high-resolution derived images inside the CLOC workspace by
default. It reads source images from a local dataset root and writes under the
workspace `augmented/` directory unless an explicit `--output-root` is provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

WORKSPACE = Path(__file__).resolve().parents[2]
DATASET_ROOT = WORKSPACE / "images"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "augmented"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_path(path: Any) -> str:
    return str(path).replace("\\", "/")


def parse_prefix(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --path-prefix value, expected OLD=NEW: {value}")
        old, new = value.split("=", 1)
        pairs.append((norm_path(old).rstrip("/"), norm_path(new).rstrip("/")))
    return pairs


def resolve_path(path: Path, prefixes: list[tuple[str, str]]) -> Path:
    normalized = norm_path(path)
    for old, new in prefixes:
        if normalized == old or normalized.startswith(old + "/"):
            candidate = Path(new + normalized[len(old) :])
            if candidate.exists():
                return candidate
    if path.exists():
        return path
    return path


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def image_digest(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    h = hashlib.sha256()
    h.update(rgb.tobytes())
    h.update(str(rgb.size).encode("ascii"))
    return h.hexdigest()


def source_path(entry: dict[str, Any], dataset_root: Path, prefixes: list[tuple[str, str]]) -> Path:
    rel = entry["source_image_relpath"]
    p = Path(rel)
    if p.is_absolute():
        return resolve_path(p, prefixes)
    return resolve_path(dataset_root / rel, prefixes)


def rebuild_entry(entry: dict[str, Any], dataset_root: Path, prefixes: list[tuple[str, str]]) -> Image.Image:
    path = source_path(entry, dataset_root, prefixes)
    with Image.open(path) as img:
        img = img.convert("RGB")
        strategy = entry["strategy"]
        if strategy in {"grid", "cropped"}:
            return img.crop(tuple(entry["crop_box"]))
        if strategy == "zoomed":
            return img.resize(tuple(entry["resize_size"]), Image.Resampling.LANCZOS)
        raise ValueError(f"Unsupported strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=WORKSPACE / "metadata/high_resolution_rebuild_report.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--path-prefix",
        action="append",
        default=[],
        help="Rewrite source paths as OLD=NEW when rebuilding on another machine. Can be repeated.",
    )
    args = parser.parse_args()
    prefixes = parse_prefix(args.path_prefix)

    recipe = load_json(args.recipe)
    entries = recipe.get("entries", [])
    failures: list[dict[str, Any]] = []
    written = 0
    skipped_existing = 0
    verified = 0

    for i, entry in enumerate(entries, 1):
        if i == 1 or i % 500 == 0 or i == len(entries):
            print(f"rebuild {i}/{len(entries)}", flush=True)
        out_path = args.output_root / entry["output_relpath"]
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        try:
            rebuilt = rebuild_entry(entry, args.dataset_root, prefixes)
            if args.verify:
                got = image_digest(rebuilt)
                expected = entry.get("reference_sha256_rgb")
                if expected and got != expected:
                    failures.append({**entry, "reason": "sha256_mismatch", "rebuilt_sha256_rgb": got})
                    continue
                verified += 1
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rebuilt.save(out_path)
            written += 1
        except Exception as exc:
            failures.append({**entry, "reason": f"{type(exc).__name__}: {exc}"})

    summary = {
        "recipe": str(args.recipe),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "entries": len(entries),
        "written": written,
        "skipped_existing": skipped_existing,
        "verified": verified,
        "failures": len(failures),
    }
    write_json({"summary": summary, "failures": failures[:100]}, args.report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
