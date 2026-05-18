#!/usr/bin/env python3
"""Run or preview all source-image conversion steps for the CLOC release.

The script is intentionally conservative: by default it only validates inputs
and writes a dry-run report. Add --run when the downloaded source datasets are
ready and you want to create/verify converted image folders.
"""

from __future__ import annotations

import argparse
import json
import shlex
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


def render_token(token: str, context: dict[str, str], *, overwrite: bool, supports_overwrite: bool) -> list[str]:
    if token == "{overwrite_flag}":
        return ["--overwrite"] if overwrite and supports_overwrite else []

    rendered = token
    for key, value in context.items():
        rendered = rendered.replace("{" + key + "}", value)
    return [rendered] if rendered else []


def render_command(step: dict[str, Any], context: dict[str, str], *, overwrite: bool) -> list[str]:
    command: list[str] = []
    supports_overwrite = bool(step.get("supports_overwrite"))
    for token in step.get("command", []):
        command.extend(render_token(str(token), context, overwrite=overwrite, supports_overwrite=supports_overwrite))
    return command


def check_paths(workspace: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    result = []
    for rel_path in rel_paths:
        path = workspace / rel_path
        result.append(
            {
                "path": rel_path,
                "absolute_path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "is_file": path.is_file()
            }
        )
    return result


def missing_paths(path_checks: list[dict[str, Any]]) -> list[str]:
    return [item["path"] for item in path_checks if not item["exists"]]


def split_csv(values: list[str]) -> set[str]:
    selected: set[str] = set()
    for value in values:
        for piece in value.split(","):
            piece = piece.strip()
            if piece:
                selected.add(piece)
    return selected


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: <workspace>/manifests/conversion_manifest.json",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="Execute conversions. Default only validates and previews.")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to conversion scripts that support it.")
    parser.add_argument("--only", action="append", default=[], help="Run selected step id(s), comma separated or repeated.")
    parser.add_argument("--skip", action="append", default=[], help="Skip selected step id(s), comma separated or repeated.")
    parser.add_argument("--list", action="store_true", help="List available conversion steps and exit.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Default: metadata/conversion_run_report.json or metadata/conversion_dry_run_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    manifest_path = (args.manifest or (workspace / "manifests" / "conversion_manifest.json")).resolve()
    manifest = load_json(manifest_path)

    context = {
        "workspace": str(workspace),
        "python": args.python,
    }
    only = split_csv(args.only)
    skip = split_csv(args.skip)
    steps = list(manifest.get("steps", []))
    known_ids = {str(step["id"]) for step in steps}
    unknown_only = sorted(only - known_ids)
    unknown_skip = sorted(skip - known_ids)

    if args.list:
        for step in steps:
            print(f"{step['id']}\t{step.get('dataset', '')}\t{step.get('description', '')}")
        return 0

    report_path = (
        args.report
        or (
            workspace
            / "metadata"
            / ("conversion_run_report.json" if args.run else "conversion_dry_run_report.json")
        )
    ).resolve()

    report: dict[str, Any] = {
        "format": "cloc_conversion_run_report_v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "workspace": str(workspace),
        "manifest": str(manifest_path),
        "mode": "run" if args.run else "dry_run",
        "overwrite": bool(args.overwrite),
        "only": sorted(only),
        "skip": sorted(skip),
        "unknown_only": unknown_only,
        "unknown_skip": unknown_skip,
        "steps": [],
        "summary": {},
    }

    if unknown_only or unknown_skip:
        report["summary"] = {"status": "invalid_selection"}
        write_json(report_path, report)
        print(f"[convert_all] invalid selection; report written to {report_path}")
        if unknown_only:
            print(f"[convert_all] unknown --only ids: {', '.join(unknown_only)}")
        if unknown_skip:
            print(f"[convert_all] unknown --skip ids: {', '.join(unknown_skip)}")
        return 2

    failed = False
    selected_count = 0

    for step in steps:
        step_id = str(step["id"])
        if only and step_id not in only:
            continue
        if step_id in skip:
            continue

        selected_count += 1
        script_rel = str(step["script"])
        script_check = check_paths(workspace, [script_rel])[0]
        required_checks = check_paths(workspace, list(step.get("required_inputs", [])))
        output_checks = check_paths(workspace, list(step.get("expected_outputs", [])))
        command = render_command(step, context, overwrite=args.overwrite)

        step_report: dict[str, Any] = {
            "id": step_id,
            "dataset": step.get("dataset", ""),
            "description": step.get("description", ""),
            "script": script_check,
            "required_inputs": required_checks,
            "expected_outputs": output_checks,
            "command": command,
            "command_text": format_command(command),
        }

        missing_required = missing_paths(required_checks)
        if not script_check["exists"]:
            step_report["status"] = "missing_script"
            failed = True
        elif missing_required:
            step_report["status"] = "missing_required_inputs"
            step_report["missing_required_inputs"] = missing_required
            failed = True
        elif not args.run:
            step_report["status"] = "dry_run_ready"
            print(f"[dry-run] {step_id}: {format_command(command)}")
        else:
            print(f"\n[run] {step_id}: {format_command(command)}", flush=True)
            proc = subprocess.run(command, text=True)
            step_report["returncode"] = proc.returncode
            if proc.returncode == 0:
                step_report["status"] = "completed"
            else:
                step_report["status"] = "failed"
                failed = True

        report["steps"].append(step_report)

        if failed and not args.continue_on_error:
            break

    status_counts: dict[str, int] = {}
    for item in report["steps"]:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    report["summary"] = {
        "selected_steps": selected_count,
        "status_counts": status_counts,
        "failed": failed,
    }
    write_json(report_path, report)
    print(f"\n[convert_all] report written to {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
