#!/usr/bin/env python3
"""Rebuild cropped augmentation images from inferred crop recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]
MIN_PAD_DIM = 384


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
        h = hashlib.sha256()
        h.update(rgb.tobytes())
        h.update(str(rgb.size).encode("ascii"))
        return h.hexdigest()


def image_sha256(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    h = hashlib.sha256()
    h.update(rgb.tobytes())
    h.update(str(rgb.size).encode("ascii"))
    return h.hexdigest()


def output_path_for_entry(entry: dict[str, Any], workspace: Path) -> Path:
    old = str(entry["output_image_path"]).replace("\\", "/")
    if old == "augmented" or old.startswith("augmented/"):
        return workspace / old
    workspace_aug = (workspace / "augmented").as_posix()
    if old.startswith(workspace_aug + "/"):
        return Path(old)
    if "/augmented/" in old:
        # Some recipes were inferred after copying into the release workspace,
        # so their output paths already contain an augmented/ subtree under an
        # older absolute workspace root.
        return workspace / "augmented" / old.split("/augmented/", 1)[1]
    name = Path(old).name
    if "/General/cropped_img/" in old:
        return workspace / "augmented" / "General" / "cropped" / name
    if "/Remote_sensing/cropped_img/" in old:
        return workspace / "augmented" / "Remote_sensing" / "cropped" / name
    raise ValueError(f"cannot infer output directory from old output path: {old}")


def crop_from_source_image(source_img: Image.Image, entry: dict[str, Any]) -> Image.Image:
    crop_box = tuple(int(x) for x in entry["crop_box"])
    crop = source_img.crop(crop_box)
    target_w = max(crop.width, MIN_PAD_DIM)
    target_h = max(crop.height, MIN_PAD_DIM)
    if (target_w, target_h) != crop.size:
        padded = Image.new(crop.mode, (target_w, target_h), (0, 0, 0))
        padded.paste(crop, (0, 0))
        crop = padded
    return crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--recipe", type=Path, action="append", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--path-prefix",
        action="append",
        default=[],
        help="Rewrite source paths as OLD=NEW when rebuilding on another machine. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebuilt = 0
    skipped = 0
    verified = 0
    verification_failed = 0
    errors: list[dict[str, str]] = []
    prefixes = parse_prefix(args.path_prefix)

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for recipe_path in args.recipe:
        payload = load_json(recipe_path)
        for entry in payload.get("entries", []):
            source_path = resolve_path(str(entry["source_image_path"]), prefixes)
            grouped[source_path.as_posix()].append((recipe_path, entry))

    total = sum(len(v) for v in grouped.values())
    processed = 0
    for source_i, (source_path_str, rows) in enumerate(sorted(grouped.items()), 1):
        source_path = Path(source_path_str)
        if source_i == 1 or source_i % 200 == 0 or source_i == len(grouped):
            print(f"source {source_i}/{len(grouped)} rows={len(rows)} {source_path.name}", flush=True)
        if not source_path.exists():
            for recipe_path, entry in rows:
                errors.append(
                    {
                        "recipe": str(recipe_path),
                        "reason": f"missing source image: {source_path}",
                        "entry": str(entry.get("record_key")),
                    }
                )
            continue

        try:
            with Image.open(source_path) as src:
                source_img = src.convert("RGB")
                for recipe_path, entry in rows:
                    processed += 1
                    if processed == 1 or processed % 1000 == 0 or processed == total:
                        print(f"  rebuilt progress {processed}/{total}", flush=True)
                    try:
                        output_path = output_path_for_entry(entry, args.workspace)
                        if output_path.exists() and not args.overwrite:
                            skipped += 1
                            continue
                        crop = crop_from_source_image(source_img, entry)
                        if args.verify and entry.get("reference_sha256_rgb"):
                            digest = image_sha256(crop)
                            if digest == entry["reference_sha256_rgb"]:
                                verified += 1
                            else:
                                verification_failed += 1
                                errors.append(
                                    {
                                        "output_path": str(output_path),
                                        "reason": "hash_mismatch",
                                        "expected": str(entry["reference_sha256_rgb"]),
                                        "actual": digest,
                                    }
                                )
                                continue
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        crop.save(output_path)
                        rebuilt += 1
                    except Exception as exc:  # pragma: no cover - operational script
                        errors.append(
                            {
                                "recipe": str(recipe_path),
                                "reason": repr(exc),
                                "entry": str(entry.get("record_key")),
                            }
                        )
        except Exception as exc:  # pragma: no cover - operational script
            for recipe_path, entry in rows:
                errors.append(
                    {
                        "recipe": str(recipe_path),
                        "reason": f"source_open_failed:{type(exc).__name__}:{exc}",
                        "entry": str(entry.get("record_key")),
                    }
                )

    print(
        json.dumps(
            {
                "rebuilt": rebuilt,
                "skipped": skipped,
                "verified": verified,
                "verification_failed": verification_failed,
                "errors": errors[:50],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
