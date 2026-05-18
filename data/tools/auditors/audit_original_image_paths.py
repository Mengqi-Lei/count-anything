#!/usr/bin/env python3
"""Stream-check original image_path existence in split JSON files.

This audits records whose top-level ``status`` field is ``original``. The split
JSON files can be several GB, so this script intentionally avoids loading them
as JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).absolute().parents[2]
KEY_RE = re.compile(r'^  "([^"]+)":\s+\{\s*$')


def parse_json_value_from_line(line: str) -> Any:
    value = line.split(":", 1)[1].strip().rstrip(",")
    return json.loads(value)


def resolve_image_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return WORKSPACE / p


def dataset_root_for_path(path: str) -> str:
    markers = [f"{WORKSPACE.as_posix().rstrip('/')}/"]
    for marker in markers:
        if marker in path:
            rel = path.split(marker, 1)[1]
            return rel.split("/", 1)[0]
    p = Path(path)
    if not p.is_absolute():
        return path.split("/", 1)[0]
    return "<outside_dataset>"


def empty_record() -> dict[str, Any]:
    return {
        "record_key": "",
        "image_path": None,
        "image_from": None,
        "status": None,
    }


def scan_split(path: Path, exclude_image_from: set[str], sample_limit: int) -> dict[str, Any]:
    total_original_seen = 0
    total_checked = 0
    existing = 0
    missing = 0
    excluded = Counter()
    by_image_from: dict[str, Counter] = defaultdict(Counter)
    by_dataset_root: dict[str, Counter] = defaultdict(Counter)
    missing_samples: list[dict[str, str]] = []

    current = empty_record()

    def flush() -> None:
        nonlocal total_original_seen, total_checked, existing, missing
        if current["status"] != "original" or not current["image_path"]:
            return
        image_from = str(current["image_from"] or "")
        total_original_seen += 1
        if image_from in exclude_image_from:
            excluded[image_from] += 1
            return
        total_checked += 1
        image_path = str(current["image_path"])
        exists = resolve_image_path(image_path).exists()
        status = "existing" if exists else "missing"
        if exists:
            existing += 1
        else:
            missing += 1
            if len(missing_samples) < sample_limit:
                missing_samples.append(
                    {
                        "record_key": str(current["record_key"]),
                        "image_from": image_from,
                        "image_path": image_path,
                    }
                )
        by_image_from[image_from][status] += 1
        by_dataset_root[dataset_root_for_path(image_path)][status] += 1

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = KEY_RE.match(line)
            if m:
                flush()
                current.update(empty_record())
                current["record_key"] = m.group(1)
                continue

            stripped = line.lstrip()
            if stripped.startswith('"image_path"'):
                current["image_path"] = parse_json_value_from_line(line)
            elif stripped.startswith('"image_from"'):
                current["image_from"] = parse_json_value_from_line(line)
            elif stripped.startswith('"status"'):
                current["status"] = parse_json_value_from_line(line)

    flush()

    def counter_table(counter_map: dict[str, Counter]) -> list[dict[str, Any]]:
        rows = []
        for name, c in counter_map.items():
            rows.append(
                {
                    "name": name,
                    "total": int(c["existing"] + c["missing"]),
                    "existing": int(c["existing"]),
                    "missing": int(c["missing"]),
                }
            )
        rows.sort(key=lambda r: (r["missing"], r["total"], r["name"]), reverse=True)
        return rows

    return {
        "split_file": str(path),
        "total_original_seen": total_original_seen,
        "excluded": dict(excluded),
        "total_checked": total_checked,
        "existing": existing,
        "missing": missing,
        "by_image_from": counter_table(by_image_from),
        "by_dataset_root": counter_table(by_dataset_root),
        "missing_samples": missing_samples,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Original Image Path Audit", ""]
    lines.append(f"Workspace: `{report['workspace']}`")
    if report["exclude_image_from"]:
        lines.append(f"Excluded image_from: `{', '.join(report['exclude_image_from'])}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| split | original seen | checked | existing | missing | excluded |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in report["splits"]:
        name = Path(split["split_file"]).name
        excluded_total = sum(split["excluded"].values())
        lines.append(
            f"| {name} | {split['total_original_seen']} | {split['total_checked']} | "
            f"{split['existing']} | {split['missing']} | {excluded_total} |"
        )
    lines.append("")
    lines.append("## Missing By image_from")
    lines.append("")
    lines.append("| split | image_from | total | existing | missing |")
    lines.append("|---|---|---:|---:|---:|")
    for split in report["splits"]:
        name = Path(split["split_file"]).name
        for row in split["by_image_from"]:
            if row["missing"]:
                lines.append(
                    f"| {name} | {row['name']} | {row['total']} | {row['existing']} | {row['missing']} |"
                )
    lines.append("")
    lines.append("## Missing Samples")
    lines.append("")
    for split in report["splits"]:
        name = Path(split["split_file"]).name
        if not split["missing_samples"]:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for sample in split["missing_samples"]:
            lines.append(
                f"- `{sample['record_key']}` `{sample['image_from']}` `{sample['image_path']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global WORKSPACE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--exclude-image-from", action="append", default=[])
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    WORKSPACE = args.workspace.absolute()
    split_paths = (
        [Path(p) for p in args.split]
        if args.split
        else [
            WORKSPACE / "annotations" / "train_split.json",
            WORKSPACE / "annotations" / "val_split.json",
            WORKSPACE / "annotations" / "test_split.json",
        ]
    )
    exclude_image_from = set(args.exclude_image_from)
    report = {
        "workspace": str(WORKSPACE),
        "exclude_image_from": sorted(exclude_image_from),
        "splits": [
            scan_split(path, exclude_image_from, args.sample_limit)
            for path in split_paths
        ],
    }
    report_path = args.report or (WORKSPACE / "metadata" / "original_image_path_audit.json")
    markdown_path = args.markdown or (WORKSPACE / "metadata" / "original_image_path_audit.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, markdown_path)

    for split in report["splits"]:
        print(
            f"{Path(split['split_file']).name}: original_seen={split['total_original_seen']} "
            f"checked={split['total_checked']} existing={split['existing']} missing={split['missing']} "
            f"excluded={sum(split['excluded'].values())}"
        )
    print(f"Wrote {report_path}")
    print(f"Wrote {markdown_path}")
    return 0 if all(split["missing"] == 0 for split in report["splits"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
