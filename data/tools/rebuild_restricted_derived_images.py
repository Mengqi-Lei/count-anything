#!/usr/bin/env python3
"""Run the restricted-derived-image rebuild plan for modified v17.

The script is a thin orchestrator around the existing conversion/rebuild tools.
It intentionally keeps all derived outputs inside the release workspace.  Steps
that write into user-provided upstream dataset roots, such as MoNuSAC/NuInsSeg
PNG conversion, require --allow-external-writes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def render(value: str, context: dict[str, str]) -> str:
    rendered = value
    for key, replacement in context.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def render_default_path(defaults: dict[str, Any], key: str, fallback: Path, context: dict[str, str]) -> Path:
    if key not in defaults:
        return fallback
    return Path(render(str(defaults[key]), context))


def recipe_entry_count(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if isinstance(payload.get("entries"), list):
        return len(payload["entries"])
    if isinstance(payload.get("recipes"), dict):
        return len(payload["recipes"])
    return None


def format_path_prefixes(manifest: dict[str, Any], context: dict[str, str], user_prefixes: list[str]) -> list[str]:
    prefixes: list[str] = []
    for item in manifest.get("default_path_prefixes", []):
        old = render(str(item["old"]), context).rstrip("/")
        new = render(str(item["new"]), context).rstrip("/")
        prefixes.append(f"{old}={new}")
    prefixes.extend(user_prefixes)
    return prefixes


def expand_command(
    step: dict[str, Any],
    context: dict[str, str],
    *,
    overwrite: bool,
    verify: bool,
    path_prefixes: list[str],
) -> list[str]:
    expanded: list[str] = []
    for token in step["command"]:
        token = str(token)
        if token == "{overwrite_flag}":
            if overwrite and step.get("supports_overwrite"):
                expanded.append("--overwrite")
            continue
        if token == "{verify_flag}":
            if verify and step.get("supports_verify"):
                expanded.append("--verify")
            continue
        if token == "{path_prefix_args}":
            if step.get("uses_path_prefix"):
                for prefix in path_prefixes:
                    expanded.extend(["--path-prefix", prefix])
            continue
        rendered = render(token, context)
        if rendered:
            expanded.append(rendered)
    return expanded


def validate_step_files(step: dict[str, Any], workspace: Path) -> list[str]:
    issues: list[str] = []
    script = workspace / step["script"]
    if not script.exists():
        issues.append(f"missing script: {script}")
    for key in ("recipe",):
        if step.get(key):
            path = workspace / step[key]
            if not path.exists():
                issues.append(f"missing {key}: {path}")
    for recipe in step.get("recipes", []):
        path = workspace / recipe
        if not path.exists():
            issues.append(f"missing recipe: {path}")
    return issues


def validate_expected_counts(step: dict[str, Any], workspace: Path) -> dict[str, Any]:
    expected = step.get("expected", {})
    result: dict[str, Any] = {}
    if "recipe" in step:
        count = recipe_entry_count(workspace / step["recipe"])
        result["recipe_entries"] = count
        if expected.get("entries") is not None:
            result["expected_entries"] = expected["entries"]
            result["entries_match"] = count == expected["entries"]
    if "recipes" in step:
        counts: dict[str, int | None] = {}
        total = 0
        has_missing = False
        for recipe in step["recipes"]:
            count = recipe_entry_count(workspace / recipe)
            counts[recipe] = count
            if count is None:
                has_missing = True
            else:
                total += count
        result["recipe_entries_by_file"] = counts
        result["recipe_entries"] = None if has_missing else total
        if expected.get("entries") is not None:
            result["expected_entries"] = expected["entries"]
            result["entries_match"] = (not has_missing) and total == expected["entries"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--objects365-root", type=Path, default=None)
    parser.add_argument("--monusac-root", type=Path, default=None)
    parser.add_argument("--nuinsseg-root", type=Path, default=None)
    parser.add_argument("--soybean-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--allow-external-writes",
        action="store_true",
        help="Allow steps that write into user-provided upstream dataset roots, such as medical PNG conversion.",
    )
    parser.add_argument("--step", action="append", default=[], help="Run only selected step id(s). Repeatable.")
    parser.add_argument("--group", action="append", default=[], help="Run only selected group(s). Repeatable.")
    parser.add_argument("--path-prefix", action="append", default=[], help="Extra OLD=NEW source path rewrite. Repeatable.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    manifest_path = args.manifest or (workspace / "manifests" / "restricted_derived_rebuild_manifest.json")
    manifest = load_json(manifest_path)
    defaults = manifest.get("default_roots", {})

    initial_context = {"workspace": norm_path(workspace)}
    dataset_root = args.dataset_root or render_default_path(defaults, "dataset_root", workspace / "images", initial_context)
    root_context = {**initial_context, "dataset_root": norm_path(dataset_root)}
    objects365_root = args.objects365_root or render_default_path(
        defaults,
        "objects365_root",
        dataset_root / "Objects365-2020",
        root_context,
    )
    monusac_root = args.monusac_root or render_default_path(defaults, "monusac_root", dataset_root / "MoNuSAC", root_context)
    nuinsseg_root = args.nuinsseg_root or render_default_path(defaults, "nuinsseg_root", dataset_root / "NuInsSeg", root_context)
    soybean_root = args.soybean_root or render_default_path(defaults, "soybean_root", dataset_root / "soybean_pod", root_context)

    context = {
        "python": args.python,
        "workspace": norm_path(workspace),
        "dataset_root": norm_path(dataset_root),
        "objects365_root": norm_path(objects365_root),
        "monusac_root": norm_path(monusac_root),
        "nuinsseg_root": norm_path(nuinsseg_root),
        "soybean_root": norm_path(soybean_root),
    }
    path_prefixes = format_path_prefixes(manifest, context, args.path_prefix)

    selected_steps = set(args.step)
    selected_groups = set(args.group)
    report: dict[str, Any] = {
        "format": "restricted_derived_rebuild_run_report_v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "workspace": str(workspace),
        "manifest": str(manifest_path),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "verify": bool(args.verify),
        "allow_external_writes": bool(args.allow_external_writes),
        "context": context,
        "steps": [],
    }

    failed = False
    for step in manifest.get("steps", []):
        step_id = step["id"]
        group = step.get("group", "")
        if selected_steps and step_id not in selected_steps:
            continue
        if selected_groups and group not in selected_groups:
            continue

        step_report: dict[str, Any] = {
            "id": step_id,
            "group": group,
            "description": step.get("description", ""),
            "external_write_required": bool(step.get("external_write_required")),
            "accepted_approximate": bool(step.get("accepted_approximate")),
            "validation": {},
            "status": "pending",
        }
        step_report["validation"]["files"] = validate_step_files(step, workspace)
        step_report["validation"]["counts"] = validate_expected_counts(step, workspace)

        command = expand_command(
            step,
            context,
            overwrite=args.overwrite,
            verify=args.verify,
            path_prefixes=path_prefixes,
        )
        step_report["command"] = command

        if step_report["validation"]["files"]:
            step_report["status"] = "invalid_manifest_files"
            failed = True
            report["steps"].append(step_report)
            if not args.continue_on_error:
                break
            continue

        counts = step_report["validation"].get("counts", {})
        if counts.get("entries_match") is False:
            step_report["status"] = "invalid_manifest_counts"
            failed = True
            report["steps"].append(step_report)
            if not args.continue_on_error:
                break
            continue

        if step.get("external_write_required") and not args.allow_external_writes:
            step_report["status"] = "skipped_external_write_gate"
            report["steps"].append(step_report)
            continue

        if args.dry_run:
            step_report["status"] = "dry_run"
            report["steps"].append(step_report)
            continue

        print(f"[RUN] {step_id}", flush=True)
        print(" ".join(command), flush=True)
        proc = subprocess.run(command, cwd=workspace, text=True, capture_output=True)
        step_report["returncode"] = proc.returncode
        step_report["stdout_tail"] = proc.stdout[-4000:]
        step_report["stderr_tail"] = proc.stderr[-4000:]
        if proc.returncode == 0:
            step_report["status"] = "completed"
        else:
            step_report["status"] = "failed"
            failed = True
            report["steps"].append(step_report)
            if not args.continue_on_error:
                break
            continue
        report["steps"].append(step_report)

    report_path = args.report or (workspace / "metadata" / ("restricted_derived_rebuild_dry_run_report.json" if args.dry_run else "restricted_derived_rebuild_run_report.json"))
    write_json(report_path, report)

    summary = {
        "report": str(report_path),
        "dry_run": bool(args.dry_run),
        "steps": len(report["steps"]),
        "status_counts": {},
        "failed": failed,
    }
    for step in report["steps"]:
        status = step["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
