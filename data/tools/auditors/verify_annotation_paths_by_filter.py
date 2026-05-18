#!/usr/bin/env python3
"""Stream-verify selected image_path entries in release annotation JSON files."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = [
    WORKSPACE / "annotations" / "train_split.json",
    WORKSPACE / "annotations" / "val_split.json",
    WORKSPACE / "annotations" / "test_split.json",
]


def parse_json_value_from_line(line: str) -> Any:
    return json.loads(line.split(":", 1)[1].strip().rstrip(","))


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else WORKSPACE / p


def match_any(value: str, patterns: list[str], *, prefix: bool) -> bool:
    if not patterns:
        return True
    if prefix:
        return any(value.startswith(pattern) for pattern in patterns)
    return value in patterns


def scan_split(
    split: Path,
    prefixes: list[str],
    sources: list[str],
    sample_limit: int,
) -> dict[str, Any]:
    total = existing = missing = 0
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    missing_samples: list[dict[str, str]] = []
    pending_path: str | None = None

    with split.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith('"image_path"'):
                pending_path = parse_json_value_from_line(line)
                continue
            if stripped.startswith('"image_from"') and pending_path is not None:
                image_from = parse_json_value_from_line(line)
                image_path = pending_path
                pending_path = None
                if not match_any(image_path, prefixes, prefix=True):
                    continue
                if not match_any(image_from, sources, prefix=False):
                    continue
                total += 1
                ok = resolve(image_path).exists()
                status = "existing" if ok else "missing"
                by_source[image_from][status] += 1
                if ok:
                    existing += 1
                else:
                    missing += 1
                    if len(missing_samples) < sample_limit:
                        missing_samples.append(
                            {
                                "split": split.name,
                                "image_from": image_from,
                                "image_path": image_path,
                                "resolved": str(resolve(image_path)),
                            }
                        )

    return {
        "split": str(split),
        "total": total,
        "existing": existing,
        "missing": missing,
        "by_source": {
            k: {"existing": int(v["existing"]), "missing": int(v["missing"])}
            for k, v in sorted(by_source.items())
        },
        "missing_samples": missing_samples,
    }


def main() -> int:
    global WORKSPACE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("CLOC_RELEASE_ROOT", str(WORKSPACE))),
        help="Dataset release root used to resolve relative image_path values.",
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    WORKSPACE = args.workspace.resolve()
    splits = [Path(p) for p in args.split] if args.split else [
        WORKSPACE / "annotations" / "train_split.json",
        WORKSPACE / "annotations" / "val_split.json",
        WORKSPACE / "annotations" / "test_split.json",
    ]
    report = {
        "workspace": str(WORKSPACE),
        "prefixes": args.prefix,
        "sources": args.source,
        "splits": [
            scan_split(split, args.prefix, args.source, args.sample_limit)
            for split in splits
        ],
    }
    report["total"] = sum(s["total"] for s in report["splits"])
    report["existing"] = sum(s["existing"] for s in report["splits"])
    report["missing"] = sum(s["missing"] for s in report["splits"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in report["splits"]:
        print(
            f"{Path(split['split']).name}: total={split['total']} "
            f"existing={split['existing']} missing={split['missing']}"
        )
    print(
        f"ALL: total={report['total']} existing={report['existing']} missing={report['missing']}"
    )
    print(f"Wrote {args.report}")
    return 0 if report["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
