#!/usr/bin/env python3
"""Rebuild BriFiSeg unified RGB PNG images from the official 3channels.tar.

The CLOC annotations reference ``images/BriFiSeg/unified_images/*.png``. These
PNGs are derived from BriFiSeg's three-channel nnU-Net inputs
``*_0000.nii.gz``, ``*_0001.nii.gz``, and ``*_0002.nii.gz``. This script keeps
the official archive/extracted structure intact and only creates the derived
``unified_images`` directory needed by the release annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def sha256_rgb(path: Path) -> str:
    arr = np.asarray(Image.open(path).convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def render_three_channels(paths: list[Path]) -> np.ndarray:
    channels = [np.squeeze(nib.load(str(path)).get_fdata()) for path in paths]
    rgb = np.stack(channels, axis=-1)
    min_value = np.nanmin(rgb)
    max_value = np.nanmax(rgb)
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return np.zeros(rgb.shape, dtype=np.uint8)
    # Match the historical cv2.normalize(..., 0, 255, NORM_MINMAX).astype(uint8)
    return ((rgb - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)


def extract_sources(root: Path, tar_name: str, force: bool) -> dict:
    report: dict = {"tar": str(root / tar_name), "tar_extracted": False, "task_zips_extracted": []}
    tar_path = root / tar_name
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing BriFiSeg archive: {tar_path}")

    channels_dir = root / "3channels"
    if force or not channels_dir.exists():
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(root)
        report["tar_extracted"] = True

    task_zips = sorted(channels_dir.glob("Task*.zip"))
    if not task_zips:
        raise FileNotFoundError(f"No Task*.zip files found in {channels_dir}")

    for zip_path in task_zips:
        task_name = zip_path.stem
        extract_dir = root / task_name
        expected_dir = extract_dir / task_name
        if force or not expected_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            report["task_zips_extracted"].append(str(zip_path.relative_to(root)))

    return report


def find_task_dirs(root: Path) -> list[Path]:
    task_dirs = []
    for path in sorted(root.glob("Task*/Task*")):
        if path.is_dir() and ((path / "imagesTr").is_dir() or (path / "imagesTs").is_dir()):
            task_dirs.append(path)
    if not task_dirs:
        raise FileNotFoundError(f"No extracted BriFiSeg Task*/Task* image folders found under {root}")
    return task_dirs


def convert(root: Path, output_dir: Path, overwrite: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "root": str(root),
        "output_dir": str(output_dir),
        "task_dirs": [],
        "written": 0,
        "skipped_existing": 0,
        "duplicate_same": 0,
        "duplicate_different": [],
        "missing_channels": [],
    }

    seen_hashes: dict[str, str] = {}
    for task_dir in find_task_dirs(root):
        report["task_dirs"].append(str(task_dir.relative_to(root)))
        for split in ("Tr", "Ts"):
            image_dir = task_dir / f"images{split}"
            if not image_dir.is_dir():
                continue
            for ch0 in sorted(image_dir.glob("*_0000.nii.gz")):
                base = ch0.name.removesuffix("_0000.nii.gz")
                channels = [image_dir / f"{base}_000{i}.nii.gz" for i in range(3)]
                if not all(path.exists() for path in channels):
                    report["missing_channels"].append(str(ch0.relative_to(root)))
                    continue

                out_path = output_dir / f"{base}.png"
                rgb = render_three_channels(channels)
                rendered_hash = hashlib.sha256(rgb.tobytes()).hexdigest()

                if out_path.name in seen_hashes:
                    if seen_hashes[out_path.name] == rendered_hash:
                        report["duplicate_same"] += 1
                        continue
                    report["duplicate_different"].append(
                        {"image": out_path.name, "task": str(task_dir.relative_to(root))}
                    )
                    # Historical CLOC keeps the first generated image when
                    # BriFiSeg contains duplicate base names. In particular,
                    # Task019_A549 has 48 imagesTs files whose names collide
                    # with imagesTr, but the release annotations expect the
                    # first/training versions. Do not let later duplicates
                    # replace earlier ones.
                    continue

                if out_path.exists() and not overwrite:
                    report["skipped_existing"] += 1
                else:
                    Image.fromarray(rgb).save(out_path)
                    report["written"] += 1

                seen_hashes[out_path.name] = rendered_hash

    report["final_png_count"] = len(list(output_dir.glob("*.png")))
    return report


def compare_to_reference(output_dir: Path, reference_dir: Path) -> dict:
    generated = {path.name: path for path in output_dir.glob("*.png")}
    reference = {path.name: path for path in reference_dir.glob("*.png")}
    common = sorted(generated.keys() & reference.keys())
    mismatches = []
    for name in common:
        if sha256_rgb(generated[name]) != sha256_rgb(reference[name]):
            mismatches.append(name)
            if len(mismatches) >= 20:
                break
    return {
        "reference_dir": str(reference_dir),
        "generated_count": len(generated),
        "reference_count": len(reference),
        "common_count": len(common),
        "missing_from_generated": sorted(reference.keys() - generated.keys())[:20],
        "extra_in_generated": sorted(generated.keys() - reference.keys())[:20],
        "mismatch_count_capped_at_20": len(mismatches),
        "mismatch_examples": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).absolute().parents[2],
        help="data root",
    )
    parser.add_argument("--brifiseg-root", type=Path, default=None)
    parser.add_argument("--tar-name", default="3channels.tar")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--compare-reference-dir", type=Path, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.brifiseg_root or (args.workspace / "images" / "BriFiSeg")
    output_dir = args.output_dir or (root / "unified_images")
    report_path = args.report or (args.workspace / "metadata" / "brifiseg_3channels_conversion_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"extract": None, "convert": None, "compare": None}
    if not args.skip_extract:
        report["extract"] = extract_sources(root, args.tar_name, args.force_extract)
    report["convert"] = convert(root, output_dir, args.overwrite)
    if args.compare_reference_dir:
        report["compare"] = compare_to_reference(output_dir, args.compare_reference_dir)

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
