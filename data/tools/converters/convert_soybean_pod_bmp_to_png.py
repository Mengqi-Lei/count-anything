#!/usr/bin/env python3
"""Convert Soybean Pod BMP images into the PNG layout used by CLOC annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops


DEFAULT_SOYBEAN_ROOT = Path(__file__).resolve().parents[2] / "images" / "soybean_pod"


def image_equal(a: Path, b: Path) -> bool:
    with Image.open(a) as ia, Image.open(b) as ib:
        ra = ia.convert("RGB")
        rb = ib.convert("RGB")
        if ra.size != rb.size:
            return False
        return ImageChops.difference(ra, rb).getbbox() is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soybean-root", type=Path, default=DEFAULT_SOYBEAN_ROOT)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    soybean_root = args.soybean_root.resolve()
    source_dir = args.source_dir or (soybean_root / "dataset")
    output_dir = args.output_dir or (soybean_root / "dataset_png")
    report_path = args.report or (
        soybean_root.parents[1] / "metadata" / "soybean_pod_bmp_to_png_conversion_report.json"
    )

    bmp_files = sorted(source_dir.glob("*.bmp"))
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    verify_counts: dict[str, int] = {}

    for src in bmp_files:
        dst = output_dir / f"{src.stem}.png"
        if dst.exists() and not args.overwrite:
            status = "exists_skipped"
        elif args.dry_run:
            status = "dry_run"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as img:
                img.convert("RGB").save(dst, "PNG")
            status = "written"
        status_counts[status] = status_counts.get(status, 0) + 1

        verify_status = "not_requested"
        if args.verify and not args.dry_run:
            verify_status = (
                "pixel_match" if dst.exists() and image_equal(src, dst) else "pixel_mismatch"
            )
            verify_counts[verify_status] = verify_counts.get(verify_status, 0) + 1
        rows.append({"source": str(src), "output": str(dst), "status": status, "verify": verify_status})

    existing_png = sorted(output_dir.glob("*.png")) if output_dir.exists() else []
    expected_names = {f"{p.stem}.png" for p in bmp_files}
    missing_output = [name for name in expected_names if not (output_dir / name).exists()]
    extra_png = [p.name for p in existing_png if p.name not in expected_names]
    report = {
        "format": "soybean_pod_bmp_to_png_conversion_report_v1",
        "soybean_root": str(soybean_root),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_bmp_count": len(bmp_files),
        "output_png_count": len(existing_png),
        "status_counts": status_counts,
        "verify_counts": verify_counts,
        "missing_output_count": len(missing_output),
        "missing_output_samples": missing_output[:20],
        "extra_png_count": len(extra_png),
        "extra_png_samples": extra_png[:20],
        "rows": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if missing_output or verify_counts.get("pixel_mismatch", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
