#!/usr/bin/env python3
"""Audit stitched augmentation release closure for modified v17.

This script is intentionally streaming-friendly for train_split.json, which is
large.  It checks the current unexpanded train/val/test split files and reports
which stitched records are covered by strict or accepted-approximate rebuild
recipes, whether their output image exists under the release workspace, and
which records remain uncovered.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]


def brace_delta_outside_strings(line: str) -> int:
    delta = 0
    in_string = False
    escape = False
    for ch in line:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            delta += 1
        elif ch == "}":
            delta -= 1
    return delta


def iter_top_level_record_blobs(path: Path) -> Iterator[tuple[str, str]]:
    """Yield raw record JSON blobs from a pretty-printed top-level object.

    The split JSONs in this project are emitted as:
      {
        "key": {
          ...
        },
        ...
      }
    Reading the whole train split costs several GB.  We therefore emit raw
    record text first and let callers JSON-parse only records they care about.
    """

    with path.open("r", encoding="utf-8") as f:
        current_key: str | None = None
        buf: list[str] = []
        depth = 0
        for line in f:
            stripped = line.lstrip()
            if current_key is None:
                if not stripped.startswith('"') or '": {' not in stripped:
                    continue
                current_key = stripped.split('"', 2)[1]
                after_colon = stripped.split(":", 1)[1].strip()
                if after_colon.endswith(","):
                    after_colon = after_colon[:-1]
                buf = [after_colon + "\n"]
                depth = brace_delta_outside_strings(after_colon)
                if depth == 0:
                    yield current_key, "".join(buf)
                    current_key = None
                    buf = []
                continue

            payload_line = line
            depth += brace_delta_outside_strings(payload_line)
            if depth == 0:
                # Drop the trailing comma after the closing record brace.
                if payload_line.rstrip().endswith(","):
                    payload_line = payload_line.rstrip()[:-1] + "\n"
                buf.append(payload_line)
                yield current_key, "".join(buf)
                current_key = None
                buf = []
            else:
                buf.append(payload_line)


def iter_stitched_records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    for key, blob in iter_top_level_record_blobs(path):
        if '"status": "stitched"' not in blob:
            continue
        yield key, json.loads(blob)


TOP_LEVEL_KEY_RE = re.compile(r'^  "([^"]+)": \{$')
FIELD_RE = re.compile(r'^    "(image_path|image_from|modality|status)": (.+?)(,)?$')


def iter_stitched_record_summaries(path: Path) -> Iterator[tuple[str, dict[str, str]]]:
    """Fast line scanner for stitched record metadata.

    This avoids parsing huge point/bbox arrays.  It relies on the stable
    json.dump(indent=2) formatting used for the split files.
    """

    key: str | None = None
    fields: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if key is None:
                m = TOP_LEVEL_KEY_RE.match(line)
                if m:
                    key = m.group(1)
                    fields = {}
                continue

            if line in ("  }", "  },"):
                if fields.get("status") == "stitched":
                    yield key, fields
                key = None
                fields = {}
                continue

            m = FIELD_RE.match(line)
            if not m:
                continue
            field, raw_value, _comma = m.groups()
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value.strip('"')
            fields[field] = str(value)


def load_recipe_keys(path: Path) -> tuple[set[str], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    recipes = payload.get("recipes", {})
    return set(recipes), payload.get("metadata", {})


def norm_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def is_release_stitched_path(path: str, workspace: Path) -> bool:
    p = norm_path(path)
    root = norm_path(workspace / "augmented")
    return p == root or p.startswith(root + "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace
    ann_dir = workspace / "annotations"
    strict_recipe = workspace / "tools" / "recipes" / "stitched_reconstruction_recipe_restricted.json"
    approx_recipe = workspace / "tools" / "recipes" / "stitched_reconstruction_recipe_restricted_failed_98_approx.json"

    strict_keys, strict_meta = load_recipe_keys(strict_recipe)
    approx_keys, approx_meta = load_recipe_keys(approx_recipe)
    all_recipe_keys = strict_keys | approx_keys

    summary: dict[str, Any] = {
        "workspace": str(workspace),
        "strict_recipe": str(strict_recipe),
        "approx_recipe": str(approx_recipe),
        "strict_recipe_count": len(strict_keys),
        "approx_recipe_count": len(approx_keys),
        "strict_recipe_metadata": strict_meta,
        "approx_recipe_metadata": approx_meta,
        "splits": {},
        "overall": {},
    }

    overall = Counter()
    missing_images: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []
    outside_release_workspace: list[dict[str, Any]] = []
    by_image_from: Counter[str] = Counter()
    by_modality: Counter[str] = Counter()
    by_coverage: Counter[str] = Counter()
    current_stitched_keys: set[str] = set()

    for split_name in ["train_split.json", "val_split.json", "test_split.json"]:
        split_path = ann_dir / split_name
        counters = Counter()
        split_examples = defaultdict(list)
        for key, record in iter_stitched_record_summaries(split_path):
            full_key = f"{split_name}:{key}"
            current_stitched_keys.add(full_key)
            image_path = norm_path(record.get("image_path", ""))
            image_from = str(record.get("image_from", ""))
            modality = str(record.get("modality", ""))
            exists = Path(image_path).exists()
            in_release_workspace = is_release_stitched_path(image_path, workspace)
            in_strict = full_key in strict_keys
            in_approx = full_key in approx_keys

            counters["stitched"] += 1
            by_image_from[image_from] += 1
            by_modality[modality] += 1
            if exists:
                counters["image_exists"] += 1
            else:
                counters["image_missing"] += 1
                missing_images.append({"key": full_key, "image_path": image_path, "image_from": image_from})
            if in_release_workspace:
                counters["path_in_release_workspace"] += 1
            else:
                counters["path_outside_release_workspace"] += 1
                outside_release_workspace.append({"key": full_key, "image_path": image_path, "image_from": image_from})

            if in_strict:
                counters["covered_by_strict_recipe"] += 1
                by_coverage["strict"] += 1
            elif in_approx:
                counters["covered_by_approx_recipe"] += 1
                by_coverage["approx"] += 1
            else:
                counters["uncovered_by_release_recipes"] += 1
                by_coverage["uncovered"] += 1
                uncovered.append({"key": full_key, "image_path": image_path, "image_from": image_from, "modality": modality})

            if len(split_examples["stitched"]) < 5:
                split_examples["stitched"].append({"key": full_key, "image_path": image_path})

        summary["splits"][split_name] = {
            "counters": dict(counters),
            "examples": dict(split_examples),
        }
        overall.update(counters)

    summary["overall"] = {
        **dict(overall),
        "by_image_from": dict(by_image_from),
        "by_modality": dict(by_modality),
        "by_coverage": dict(by_coverage),
        "current_stitched_key_count": len(current_stitched_keys),
        "strict_recipe_keys_not_in_current_splits": len(strict_keys - current_stitched_keys),
        "approx_recipe_keys_not_in_current_splits": len(approx_keys - current_stitched_keys),
        "current_stitched_keys_without_release_recipe": len(current_stitched_keys - all_recipe_keys),
    }
    summary["missing_images"] = missing_images[:200]
    summary["outside_release_workspace"] = outside_release_workspace[:200]
    summary["uncovered"] = uncovered[:500]

    output = args.output or (workspace / "metadata" / "stitched_release_closure_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "output": str(output),
        "overall": summary["overall"],
        "missing_images_examples": len(missing_images),
        "uncovered_examples": len(uncovered),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
