#!/usr/bin/env python3
"""Verify Objects365 release paths against a partial or full patch tree.

This script does not require downloading the full Objects365 dataset. It maps
annotation paths such as
``images/Objects365-2020/train/patch0/foo.jpg`` to a user-provided
Objects365 root, checks existence by patch, and opens a small sample per patch.
Optionally it compares sampled image SHA256 hashes against a reference root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


KEY_RE = re.compile(r'^\s+"([^"]+)":\s+\{\s*$')
OBJECTS365_PREFIX = "images/Objects365-2020/"


def parse_json_value_from_line(line: str) -> Any:
    return json.loads(line.split(":", 1)[1].strip().rstrip(","))


def patch_key(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 2 and parts[0] == "train" and parts[1].startswith("patch"):
        return f"train/{parts[1]}"
    if len(parts) >= 5 and parts[0] == "val" and parts[1] == "images":
        return f"val/images/{parts[2]}/{parts[3]}"
    return "<unknown>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as exc:  # pragma: no cover - report detail only
        return False, str(exc)


def iter_objects365_paths(split_paths: list[Path]):
    current_key: str | None = None
    pending_path: str | None = None
    for split in split_paths:
        with split.open("r", encoding="utf-8") as f:
            for line in f:
                m = KEY_RE.match(line)
                if m:
                    current_key = m.group(1)
                    continue
                stripped = line.lstrip()
                if stripped.startswith('"image_path"'):
                    pending_path = parse_json_value_from_line(line)
                    continue
                if stripped.startswith('"image_from"') and pending_path is not None:
                    image_from = parse_json_value_from_line(line)
                    if image_from == "Objects365":
                        yield {
                            "split": str(split),
                            "record_key": current_key or "",
                            "image_path": pending_path,
                        }
                    pending_path = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).absolute().parents[2])
    parser.add_argument(
        "--objects365-root",
        type=Path,
        default=None,
        help="Candidate Objects365-2020 root. Defaults to <workspace>/images/Objects365-2020.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Optional reference Objects365-2020 root for sampled SHA256 comparison.",
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument(
        "--patch",
        action="append",
        default=[],
        help="Only verify this patch key, e.g. train/patch0, val/images/v1/patch0, val/images/v2/patch16. Can be repeated.",
    )
    parser.add_argument("--sample-per-patch", type=int, default=3)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    objects_root = args.objects365_root or (workspace / "images" / "Objects365-2020")
    split_paths = [Path(p) for p in args.split] if args.split else [
        workspace / "annotations" / "train_split.json",
        workspace / "annotations" / "val_split.json",
        workspace / "annotations" / "test_split.json",
    ]
    report_path = args.report or (workspace / "metadata" / "objects365_patch_sample_verify.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = existing = missing = 0
    skipped_non_source_prefix = 0
    by_patch: dict[str, Counter] = defaultdict(Counter)
    samples_by_patch: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_samples: list[dict[str, str]] = []
    patch_filter = set(args.patch)
    skipped_by_patch_filter = 0

    for item in iter_objects365_paths(split_paths):
        path = item["image_path"]
        if not path.startswith(OBJECTS365_PREFIX):
            skipped_non_source_prefix += 1
            continue
        rel = path[len(OBJECTS365_PREFIX):]
        key = patch_key(rel)
        if patch_filter and key not in patch_filter:
            skipped_by_patch_filter += 1
            continue
        candidate = objects_root / rel
        total += 1
        if candidate.exists():
            existing += 1
            by_patch[key]["existing"] += 1
            if len(samples_by_patch[key]) < args.sample_per_patch:
                samples_by_patch[key].append({**item, "relative_path": rel})
        else:
            missing += 1
            by_patch[key]["missing"] += 1
            if len(missing_samples) < 50:
                missing_samples.append({**item, "relative_path": rel})

    opened = []
    open_failures = []
    hash_mismatches = []
    for key, samples in sorted(samples_by_patch.items()):
        for sample in samples:
            rel = sample["relative_path"]
            candidate = objects_root / rel
            ok, error = open_image(candidate)
            row = {
                "patch": key,
                "relative_path": rel,
                "candidate_path": str(candidate),
                "open_ok": ok,
            }
            if error:
                row["open_error"] = error
            if ok and args.reference_root is not None:
                reference = args.reference_root / rel
                row["reference_path"] = str(reference)
                row["reference_exists"] = reference.exists()
                if reference.exists():
                    row["sha256_match"] = sha256_file(candidate) == sha256_file(reference)
                    if not row["sha256_match"]:
                        hash_mismatches.append(row)
            opened.append(row)
            if not ok:
                open_failures.append(row)

    by_patch_rows = []
    for key, counter in by_patch.items():
        by_patch_rows.append(
            {
                "patch": key,
                "total": int(counter["existing"] + counter["missing"]),
                "existing": int(counter["existing"]),
                "missing": int(counter["missing"]),
            }
        )
    by_patch_rows.sort(key=lambda r: (r["patch"]))

    report = {
        "workspace": str(workspace),
        "objects365_root": str(objects_root),
        "reference_root": str(args.reference_root) if args.reference_root else None,
        "splits": [str(p) for p in split_paths],
        "total": total,
        "existing": existing,
        "missing": missing,
        "skipped_non_source_prefix": skipped_non_source_prefix,
        "patch_filter": sorted(patch_filter),
        "skipped_by_patch_filter": skipped_by_patch_filter,
        "by_patch": by_patch_rows,
        "opened_sample_count": len(opened),
        "open_failure_count": len(open_failures),
        "hash_mismatch_count": len(hash_mismatches),
        "opened_samples": opened[:200],
        "missing_samples": missing_samples,
        "hash_mismatch_samples": hash_mismatches[:20],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Objects365 root: {objects_root}")
    print(f"total={total} existing={existing} missing={missing}")
    print(f"opened_samples={len(opened)} open_failures={len(open_failures)}")
    if args.reference_root:
        print(f"hash_mismatches={len(hash_mismatches)}")
    print(f"Wrote {report_path}")
    return 0 if missing == 0 and not open_failures and not hash_mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
