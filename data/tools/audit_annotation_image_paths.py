#!/usr/bin/env python3
"""Stream-check image_path existence in split JSON files.

The split JSON files can be several GB, so this script intentionally avoids
loading them as JSON. It relies on the stable pretty-printed record layout where
`image_path` appears before `image_from` inside each record.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = [
    WORKSPACE / "annotations" / "train_split.json",
    WORKSPACE / "annotations" / "val_split.json",
    WORKSPACE / "annotations" / "test_split.json",
]

KEY_RE = re.compile(r'^\s+"([^"]+)":\s+\{\s*$')


def parse_json_value_from_line(line: str) -> Any:
    value = line.split(":", 1)[1].strip().rstrip(",")
    return json.loads(value)


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


def resolve_image_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return WORKSPACE / p


def scan_split(path: Path, sample_limit: int) -> dict[str, Any]:
    total = 0
    existing = 0
    missing = 0
    by_image_from: dict[str, Counter] = defaultdict(Counter)
    by_dataset_root: dict[str, Counter] = defaultdict(Counter)
    missing_samples: list[dict[str, str]] = []

    current_key: str | None = None
    pending_path: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = KEY_RE.match(line)
            if m and "__" not in m.group(1):
                current_key = m.group(1)
                continue

            stripped = line.lstrip()
            if stripped.startswith('"image_path"'):
                pending_path = parse_json_value_from_line(line)
                continue

            if stripped.startswith('"image_from"') and pending_path is not None:
                image_from = parse_json_value_from_line(line)
                total += 1
                exists = resolve_image_path(pending_path).exists()
                status = "existing" if exists else "missing"
                if exists:
                    existing += 1
                else:
                    missing += 1
                    if len(missing_samples) < sample_limit:
                        missing_samples.append(
                            {
                                "record_key": current_key or "",
                                "image_from": image_from,
                                "image_path": pending_path,
                            }
                        )
                by_image_from[image_from][status] += 1
                by_dataset_root[dataset_root_for_path(pending_path)][status] += 1
                pending_path = None

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
        rows.sort(key=lambda r: (r["missing"], r["total"]), reverse=True)
        return rows

    return {
        "split_file": str(path),
        "total": total,
        "existing": existing,
        "missing": missing,
        "by_image_from": counter_table(by_image_from),
        "by_dataset_root": counter_table(by_dataset_root),
        "missing_samples": missing_samples,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Annotation Image Path Audit", ""]
    lines.append(f"Workspace: `{WORKSPACE}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| split | total | existing | missing |")
    lines.append("|---|---:|---:|---:|")
    for split in report["splits"]:
        name = Path(split["split_file"]).name
        lines.append(
            f"| {name} | {split['total']} | {split['existing']} | {split['missing']} |"
        )
    lines.append("")
    lines.append("## Missing By Dataset Root")
    lines.append("")
    lines.append("| split | dataset root | total | existing | missing |")
    lines.append("|---|---|---:|---:|---:|")
    for split in report["splits"]:
        name = Path(split["split_file"]).name
        for row in split["by_dataset_root"]:
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
    parser.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE,
        help="Dataset release root. Defaults to the parent of this tools directory.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Split JSON to scan. Defaults to unexpanded train/val/test.",
    )
    parser.add_argument(
        "--report",
        default=None,
    )
    parser.add_argument(
        "--markdown",
        default=None,
    )
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    WORKSPACE = args.workspace.resolve()
    split_paths = [Path(p) for p in args.split] if args.split else [
        WORKSPACE / "annotations" / "train_split.json",
        WORKSPACE / "annotations" / "val_split.json",
        WORKSPACE / "annotations" / "test_split.json",
    ]
    report = {
        "workspace": str(WORKSPACE),
        "splits": [scan_split(path, args.sample_limit) for path in split_paths],
    }
    report_path = Path(args.report) if args.report else (
        WORKSPACE / "metadata" / "annotation_image_path_audit.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown) if args.markdown else (
        WORKSPACE / "metadata" / "annotation_image_path_audit.md"
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, markdown_path)

    for split in report["splits"]:
        print(
            f"{Path(split['split_file']).name}: total={split['total']} "
            f"existing={split['existing']} missing={split['missing']}"
        )
    print(f"Wrote {report_path}")
    print(f"Wrote {markdown_path}")
    return 0 if all(split["missing"] == 0 for split in report["splits"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
