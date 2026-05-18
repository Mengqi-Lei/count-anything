#!/usr/bin/env python3
"""Check user-downloaded source dataset layouts for the release workflow.

This script does not download anything.  It verifies that the datasets placed
under --dataset-root look compatible with the paths used by our rebuild recipes.
Objects365 is skipped by default because it is handled separately.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
RECIPE_DIR = WORKSPACE / "tools" / "recipes"

DEFAULT_RECIPE_FILES = [
    "carpk_pucpr_cropped_recipe_probe.json",
    "cropped_recipe_remaining_permission_sources.json",
    "cropped_recipe_remaining_permission_sources_approx_11.json",
    "high_resolution_recipe_nwpu_crowd_by_pixels.json",
    "stitched_reconstruction_recipe_restricted.json",
    "stitched_reconstruction_recipe_restricted_failed_98_approx.json",
]

ROOT_ALIASES = {
    "Cityscapes": ["Cityscapes"],
    "FSCD_LVIS": ["FSCD_LVIS/FSCD-LVIS/FSCD_LVIS", "FSCD_LVIS", "FSCD-LVIS", "fscd_lvis"],
    "CARPK_PUCPR": ["CARPK_PUCPR/datasets", "CARPK_PUCPR", "CARPK_PUCPR+", "CARPK"],
    "NWPU-CROWD": ["NWPU-CROWD"],
    "NWPU-MOC": ["NWPU-MOC/NWPU-MOC", "NWPU-MOC"],
    "BCData": ["BCData/BCData", "BCData"],
    "MoNuSAC": ["MoNuSAC"],
    "NuInsSeg": ["NuInsSeg"],
    "soybean_pod": ["soybean_pod"],
    "VOCdevkit": ["VOCdevkit/VOCdevkit", "VOCdevkit"],
    "Objects365-2020": ["Objects365-2020", "Objects365"],
}


def norm_path(path: Any) -> str:
    return str(path).replace("\\", "/")


def render(value: str, context: dict[str, str]) -> str:
    rendered = value
    for key, replacement in context.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def load_prefixes(manifest_path: Path, workspace: Path, dataset_root: Path) -> list[tuple[str, str]]:
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    defaults = manifest.get("default_roots", {})
    context = {
        "workspace": norm_path(workspace),
        "dataset_root": norm_path(dataset_root),
        "objects365_root": norm_path(dataset_root / "Objects365-2020"),
    }
    if defaults.get("objects365_root"):
        context["objects365_root"] = render(str(defaults["objects365_root"]), context)

    prefixes: list[tuple[str, str]] = []
    for item in manifest.get("default_path_prefixes", []):
        old = render(str(item["old"]), context).rstrip("/")
        new = render(str(item["new"]), context).rstrip("/")
        prefixes.append((old, new))
    return prefixes


def apply_prefix(path: Path | str, prefixes: list[tuple[str, str]]) -> Path:
    normalized = norm_path(path)
    for old, new in prefixes:
        if normalized == old or normalized.startswith(old + "/"):
            return Path(new + normalized[len(old) :])
    return Path(normalized)

STRUCTURAL_CHECKS = {
    "Cityscapes": {
        "required_dirs": ["leftImg8bit"],
        "optional_dirs": ["gtFine"],
        "notes": "Official Cityscapes usually unpacks leftImg8bit and gtFine as separate archives; our recipes reference leftImg8bit images.",
    },
    "FSCD_LVIS": {
        "required_dirs": ["images/train", "images/test"],
        "optional_dirs": ["annotations", "masks/train", "masks/test"],
        "notes": "Counting-DETR FSCD-LVIS unpacks as images/{train,test} plus masks/{train,test}; release annotations reference the images subtree.",
    },
    "CARPK_PUCPR": {
        "required_dirs": [
            "CARPK_devkit/data/Images",
            "PUCPR+_devkit/data/Images",
        ],
        "optional_dirs": [
            "CARPK_devkit/data/Annotations",
            "PUCPR+_devkit/data/Annotations",
        ],
        "notes": "The LPN release unpacks CARPK_devkit and PUCPR+_devkit under one root in our workflow.",
    },
    "NWPU-CROWD": {
        "required_dirs": ["NWPU-CROWD"],
        "optional_dirs": [],
        "notes": "Our paths expect the official inner folder NWPU-CROWD under images/NWPU-CROWD.",
    },
    "NWPU-MOC": {
        "required_dirs": ["rgb"],
        "optional_dirs": ["annotations"],
        "notes": "Cropped recipes reference RGB images under NWPU-MOC/rgb.",
    },
    "BCData": {
        "required_dirs": ["images/train", "images/validation", "images/test"],
        "optional_dirs": ["annotations"],
        "notes": "Cropped recipes reference PNG images under BCData/images/{train,validation,test}.",
    },
    "MoNuSAC": {
        "required_dirs": ["MoNuSAC_images_and_annotations"],
        "optional_dirs": ["png_images"],
        "notes": "Users provide TIFF data; tools/converters/convert_monusac_tif_to_png_images.py creates png_images.",
    },
    "NuInsSeg": {
        "required_dirs": [],
        "required_globs": ["*/tissue images"],
        "optional_dirs": [],
        "notes": "Users provide organ subfolders; tools/converters/convert_nuinsseg_tissue_images_to_png.py normalizes tissue images to PNG.",
    },
    "soybean_pod": {
        "required_dirs": ["dataset"],
        "required_globs": ["dataset/*.bmp"],
        "optional_dirs": ["dataset_png"],
        "notes": "Users provide original BMP files; tools/converters/convert_soybean_pod_bmp_to_png.py rebuilds dataset_png/*.png expected by annotations.",
    },
    "VOCdevkit": {
        "required_dirs": ["VOC2007/JPEGImages"],
        "optional_dirs": ["VOC2007/Annotations"],
        "notes": "Only a small number of mixed stitched recipes reference VOC2007 JPEGImages.",
    },
    "Objects365-2020": {
        "required_dirs": ["train", "val"],
        "optional_dirs": [],
        "notes": "Skipped by default; include with --include-objects365 if needed.",
    },
}


def resolve_root(dataset_root: Path, canonical: str) -> Path:
    for alias in ROOT_ALIASES.get(canonical, [canonical]):
        candidate = dataset_root / alias
        if candidate.exists():
            return candidate
    return dataset_root / canonical


def iter_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {
                "source_image_path",
                "source_stitched_image_path",
                "image_path",
            }:
                yield value
            elif key == "source_paths" and isinstance(value, list):
                yield from value
            elif key == "source_image_relpath":
                yield value
            else:
                yield from iter_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_values(value)


def canonicalize_source_path(
    path_text: str,
    dataset_root: Path,
    prefixes: list[tuple[str, str]],
) -> tuple[str | None, Path | None]:
    if not isinstance(path_text, str) or not path_text:
        return None, None

    raw_candidate = Path(path_text)
    if not raw_candidate.is_absolute():
        raw_candidate = dataset_root / path_text
    candidate = apply_prefix(raw_candidate, prefixes)

    candidate_text = norm_path(candidate)
    dataset_root_text = norm_path(dataset_root).rstrip("/")
    if candidate_text == dataset_root_text:
        rel = ""
    elif candidate_text.startswith(dataset_root_text + "/"):
        rel = candidate_text[len(dataset_root_text) + 1 :]
    elif not Path(path_text).is_absolute():
        rel = path_text
    else:
        rel = candidate_text

    parts = Path(rel).parts
    if not parts:
        return None, None
    root = parts[0]
    if root not in ROOT_ALIASES:
        return None, None
    return root, candidate


def check_structure(
    dataset_root: Path, include_objects365: bool, only: set[str] | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for canonical in STRUCTURAL_CHECKS:
        if canonical == "Objects365-2020" and not include_objects365:
            continue
        if only is not None and canonical not in only:
            continue
        root_path = resolve_root(dataset_root, canonical)
        spec = STRUCTURAL_CHECKS[canonical]
        missing_dirs = [
            d for d in spec.get("required_dirs", []) if not (root_path / d).is_dir()
        ]
        missing_globs = []
        for pattern in spec.get("required_globs", []):
            if not list(root_path.glob(pattern)):
                missing_globs.append(pattern)
        optional_missing = [
            d for d in spec.get("optional_dirs", []) if not (root_path / d).is_dir()
        ]
        rows.append(
            {
                "dataset": canonical,
                "resolved_root": str(root_path),
                "root_exists": root_path.exists(),
                "required_missing": missing_dirs + missing_globs,
                "optional_missing": optional_missing,
                "ok": root_path.exists() and not missing_dirs and not missing_globs,
                "notes": spec.get("notes", ""),
            }
        )
    return rows


def check_recipe_paths(
    dataset_root: Path,
    include_objects365: bool,
    sample_limit: int,
    only: set[str] | None,
    prefixes: list[tuple[str, str]],
) -> dict[str, Any]:
    counts = Counter()
    missing_counts = Counter()
    missing_samples: dict[str, list[str]] = defaultdict(list)
    checked_samples: dict[str, list[str]] = defaultdict(list)

    for name in DEFAULT_RECIPE_FILES:
        path = RECIPE_DIR / name
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        for value in iter_values(obj):
            root, candidate = canonicalize_source_path(value, dataset_root, prefixes)
            if root is None or candidate is None:
                continue
            if root == "Objects365-2020" and not include_objects365:
                continue
            if only is not None and root not in only:
                continue
            counts[root] += 1
            if len(checked_samples[root]) < sample_limit:
                checked_samples[root].append(str(candidate))
            if not candidate.exists():
                missing_counts[root] += 1
                if len(missing_samples[root]) < sample_limit:
                    missing_samples[root].append(str(candidate))

    all_roots = sorted(set(counts) | set(missing_counts))
    rows = []
    for root in all_roots:
        total = counts[root]
        missing = missing_counts[root]
        rows.append(
            {
                "dataset": root,
                "referenced_paths": total,
                "missing_paths": missing,
                "ok": missing == 0,
                "checked_samples": checked_samples[root],
                "missing_samples": missing_samples[root],
            }
        )
    return {
        "recipe_files": DEFAULT_RECIPE_FILES,
        "rows": rows,
        "ok": all(row["ok"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=str(WORKSPACE / "images"),
        help="Root directory containing user-downloaded source datasets. In the release layout this is <workspace>/images.",
    )
    parser.add_argument(
        "--manifest",
        default=str(WORKSPACE / "manifests" / "restricted_derived_rebuild_manifest.json"),
        help="Rebuild manifest used for source path prefix rewrites.",
    )
    parser.add_argument(
        "--include-objects365",
        action="store_true",
        help="Also check Objects365-2020. It is skipped by default.",
    )
    parser.add_argument(
        "--report",
        default=str(WORKSPACE / "metadata" / "source_dataset_layout_check_report.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=(
            "Check only one canonical dataset root. Can be repeated. "
            "Examples: --only VOCdevkit --only NWPU-CROWD"
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    prefixes = load_prefixes(manifest_path, WORKSPACE, dataset_root)
    only = set(args.only) if args.only else None
    structure = check_structure(dataset_root, args.include_objects365, only)
    recipe_paths = check_recipe_paths(
        dataset_root, args.include_objects365, args.sample_limit, only, prefixes
    )
    report = {
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "path_prefix_count": len(prefixes),
        "include_objects365": args.include_objects365,
        "only": sorted(only) if only else None,
        "structure": structure,
        "recipe_paths": recipe_paths,
        "ok": all(row["ok"] for row in structure) and recipe_paths["ok"],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"Wrote report: {report_path}")
    print(f"Overall OK: {report['ok']}")
    print("\nStructure checks:")
    for row in structure:
        status = "OK" if row["ok"] else "MISSING"
        print(f"  {status:7s} {row['dataset']} -> {row['resolved_root']}")
        if row["required_missing"]:
            print(f"          missing required: {row['required_missing']}")
    print("\nRecipe path checks:")
    for row in recipe_paths["rows"]:
        status = "OK" if row["ok"] else "MISSING"
        print(
            f"  {status:7s} {row['dataset']}: "
            f"referenced={row['referenced_paths']} missing={row['missing_paths']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
