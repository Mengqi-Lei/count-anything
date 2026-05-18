#!/usr/bin/env python3
"""Rebuild stitched/mosaic images from an inferred reconstruction recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = (
    DEFAULT_WORKSPACE
    / "tools"
    / "recipes"
    / "stitched_reconstruction_recipe_restricted.json"
)


def norm_path(path: Any) -> str:
    return str(path).replace("\\", "/")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_prefix(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --path-prefix value, expected OLD=NEW: {value}")
        old, new = value.split("=", 1)
        pairs.append((norm_path(old).rstrip("/"), norm_path(new).rstrip("/")))
    return pairs


def resolve_path(path: str, prefixes: list[tuple[str, str]]) -> Path:
    normalized = norm_path(path)
    for old, new in prefixes:
        if normalized == old or normalized.startswith(old + "/"):
            candidate = Path(new + normalized[len(old) :])
            if candidate.exists():
                return candidate
    direct = Path(normalized)
    if direct.exists():
        return direct
    return direct


def rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def rebuild_one(recipe: dict[str, Any], output_root: Path, prefixes: list[tuple[str, str]]) -> Path:
    canvas_w, canvas_h = recipe["canvas_size"]
    canvas = Image.new("RGB", (int(canvas_w), int(canvas_h)))

    for patch in recipe["patches"]:
        source_path = resolve_path(patch["source_image_path"], prefixes)
        if not source_path.exists():
            raise FileNotFoundError(f"source image not found: {patch['source_image_path']}")
        crop_box = tuple(int(x) for x in patch["crop_box"])
        resize_size = tuple(int(x) for x in patch["resize_size"])
        paste_position = tuple(int(x) for x in patch["paste_position"])
        with Image.open(source_path) as img:
            patch_img = img.convert("RGB").crop(crop_box).resize(resize_size, Image.Resampling.BILINEAR)
        canvas.paste(patch_img, paste_position)

    output_path = output_root / recipe["output_relpath"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--path-prefix",
        action="append",
        default=[],
        help="Rewrite source paths as OLD=NEW when rebuilding on another machine. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verify", action="store_true", help="Compare rebuilt RGB hash with reference hash when present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.recipe)
    recipes = payload.get("recipes", {})
    prefixes = parse_prefix(args.path_prefix)

    total = 0
    verified = 0
    verification_failed = 0
    errors: list[dict[str, str]] = []

    for recipe_key, recipe in recipes.items():
        if args.limit and total >= args.limit:
            break
        try:
            output_path = rebuild_one(recipe, args.output_root, prefixes)
            total += 1
            if args.verify and recipe.get("reference_sha256_rgb"):
                got = rgb_sha256(output_path)
                if got == recipe["reference_sha256_rgb"]:
                    verified += 1
                else:
                    verification_failed += 1
                    errors.append({"recipe_key": recipe_key, "reason": "hash_mismatch", "output_path": str(output_path)})
        except Exception as exc:  # pragma: no cover - operational script
            errors.append({"recipe_key": recipe_key, "reason": repr(exc)})

    print(json.dumps(
        {
            "rebuilt": total,
            "verified": verified,
            "verification_failed": verification_failed,
            "errors": errors[:50],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
