"""Lightweight local inference API for CountAnything."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import yaml


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or "query"


def _set_if_present(root: Dict, keys: Sequence[str], value) -> None:
    cur = root
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
    if isinstance(cur, dict) and keys[-1] in cur:
        cur[keys[-1]] = value


@dataclass
class CountAnythingResult:
    """Prediction result for one image-query pair."""

    image_path: str
    text_query: str
    count: int
    record: Dict
    run_dir: str

    @staticmethod
    def _coerce_points(value) -> List[List[float]]:
        points = value or []
        return [[float(point[0]), float(point[1])] for point in points]

    @staticmethod
    def _coerce_boxes(value) -> List[List[float]]:
        boxes = value or []
        return [
            [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            for box in boxes
        ]

    @property
    def pred_points(self) -> List[List[float]]:
        return self._coerce_points(
            self.record.get("pred_points", self.record.get("pred_count_points", []))
        )

    @property
    def pred_scores(self) -> List[float]:
        return [
            float(score)
            for score in self.record.get(
                "pred_scores", self.record.get("pred_count_scores", [])
            )
        ]

    @property
    def pred_sources(self) -> List[int]:
        return [
            int(source)
            for source in self.record.get(
                "pred_sources", self.record.get("pred_count_sources", [])
            )
        ]

    @property
    def rsc_boxes(self) -> List[List[float]]:
        return self._coerce_boxes(
            self.record.get("rsc_boxes", self.record.get("kept_rsc_boxes", []))
        )

    def _render(self, *, show_boxes: bool = False):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError("Pillow is required for show()/save().") from exc

        image = Image.open(self.image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, _ = image.size
        label = f"{self.text_query}: {self.count}"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, width // 45))
        except Exception:
            font = ImageFont.load_default()
        line_width = max(2, width // 450)
        point_radius = max(3, width // 180)
        source_colors = {
            0: (255, 183, 3, 230),
            1: (0, 166, 255, 230),
        }
        if show_boxes:
            for box in self.rsc_boxes:
                draw.rectangle(box, outline=(255, 183, 3, 230), width=line_width)

        sources = self.pred_sources
        for idx, point in enumerate(self.pred_points):
            x, y = point
            source = sources[idx] if idx < len(sources) else -1
            fill = source_colors.get(source, (51, 214, 159, 230))
            draw.ellipse(
                [
                    x - point_radius,
                    y - point_radius,
                    x + point_radius,
                    y + point_radius,
                ],
                fill=fill,
                outline=(0, 0, 0, 210),
                width=max(1, line_width // 2),
            )

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = 10
        draw.rectangle(
            [0, 0, min(width, text_w + pad * 2), text_h + pad * 2],
            fill=(0, 0, 0, 170),
        )
        draw.text((pad, pad), label, fill=(255, 255, 255, 255), font=font)
        return image

    def show(self, *, show_boxes: bool = False) -> None:
        """Display the image with the predicted count overlaid."""

        self._render(show_boxes=show_boxes).show()

    def save(
        self,
        path: str | os.PathLike | None = None,
        *,
        show_boxes: bool = False,
    ) -> str:
        """Save the visualized prediction and return its path.

        If no path is provided, the image is saved inside this inference run
        directory next to the generated config and prediction JSON.
        """

        if path is None:
            filename = f"{self._default_output_stem()}.jpg"
            output_path = Path(self.run_dir) / filename
        else:
            output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._render(show_boxes=show_boxes).save(output_path)
        return str(output_path)

    def _default_output_stem(self) -> str:
        return f"{Path(self.image_path).stem}__{_slugify(self.text_query)}__prediction"


class CountAnything:
    """Small inference wrapper with a YOLO-like API.

    Example:
        >>> model = CountAnything("checkpoints/count_anything.pt")
        >>> results = model("path/to/image.jpg", "cells")
        >>> print(results[0].count)
        >>> results[0].show()
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike = "checkpoints/count_anything.pt",
        *,
        config: str | os.PathLike = "config/count_anything_test_cloc.yaml",
        output_dir: str | os.PathLike = "exp/count_anything_inference",
        num_gpus: int = 1,
        python_executable: str = sys.executable,
        runner: Optional[Callable[..., None]] = None,
        repo_root: str | os.PathLike | None = None,
        keep_run_files: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
        self.checkpoint = self._resolve_path(checkpoint)
        self.config = self._resolve_path(config)
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.repo_root / self.output_dir
        self.num_gpus = int(num_gpus)
        self.python_executable = python_executable
        self.runner = runner or self._default_runner
        self.keep_run_files = bool(keep_run_files)

    def __call__(
        self,
        image_path: str | os.PathLike,
        text_query: str,
        *,
        output_dir: str | os.PathLike | None = None,
    ) -> List[CountAnythingResult]:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        text_query = str(text_query).strip()
        if not text_query:
            raise ValueError("text_query must be a non-empty string")

        run_dir = self._make_run_dir(image_path, text_query, output_dir)
        _annotation_path, config_path, detail_path = self._prepare_run_files(
            image_path=image_path,
            text_query=text_query,
            run_dir=run_dir,
        )
        command = [
            self.python_executable,
            "-m",
            "count_anything.train.train",
            "-c",
            str(config_path),
            "--use-cluster",
            "0",
            "--num-gpus",
            str(self.num_gpus),
        ]
        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(self.repo_root)
            if not old_pythonpath
            else str(self.repo_root) + os.pathsep + old_pythonpath
        )
        self.runner(command, cwd=str(self.repo_root), env=env)

        payload = self._load_detail_records(detail_path)
        records = payload.get("records") or []
        if not records:
            raise RuntimeError(f"No prediction records were written to {detail_path}")
        record = records[0]
        if not self.keep_run_files:
            self._cleanup_auxiliary_run_files(run_dir, keep_paths=[detail_path])
        return [
            CountAnythingResult(
                image_path=str(image_path),
                text_query=text_query,
                count=int(record["pred_count"]),
                record=record,
                run_dir=str(run_dir),
            )
        ]

    def _resolve_path(self, path: str | os.PathLike) -> Path:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _make_run_dir(
        self,
        image_path: Path,
        text_query: str,
        output_dir: str | os.PathLike | None,
    ) -> Path:
        base_dir = Path(output_dir) if output_dir is not None else self.output_dir
        if not base_dir.is_absolute():
            base_dir = self.repo_root / base_dir
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{image_path.stem}__{_slugify(text_query)}__{stamp}"
        run_dir = base_dir / run_name
        suffix = 2
        while run_dir.exists():
            run_dir = base_dir / f"{run_name}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _prepare_run_files(self, *, image_path: Path, text_query: str, run_dir: Path):
        annotation_path = run_dir / "temporary_inference_annotation.json"
        output_stem = f"{image_path.stem}__{_slugify(text_query)}__prediction"
        detail_path = run_dir / f"{output_stem}.json"
        config_path = run_dir / "temporary_count_anything_inference.yaml"
        record_key = f"0__{_slugify(text_query)}"
        annotation = {
            record_key: {
                "idx": 0,
                "image_path": str(image_path),
                "image_from": "custom",
                "classes": [text_query],
                "annotation": {text_query: {"point": []}},
                "split": "inference",
                "selected_classes": [text_query],
                "selected_annotation": {text_query: {"point": []}},
            }
        }
        annotation_path.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self._configure_for_inference(config, annotation_path, detail_path, run_dir)
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return annotation_path, config_path, detail_path

    def _configure_for_inference(
        self,
        config: Dict,
        annotation_path: Path,
        detail_path: Path,
        run_dir: Path,
    ) -> None:
        checkpoint = str(self.checkpoint)
        experiment_dir = str(run_dir)
        _set_if_present(config, ["paths", "val_annotation_file"], str(annotation_path))
        _set_if_present(config, ["paths", "train_annotation_file"], str(annotation_path))
        _set_if_present(config, ["paths", "experiment_log_dir"], experiment_dir)
        _set_if_present(config, ["paths", "stage1_checkpoint_path"], checkpoint)
        _set_if_present(
            config,
            ["trainer", "data", "val", "dataset", "ann_file"],
            str(annotation_path),
        )
        _set_if_present(
            config,
            ["trainer", "data", "train", "dataset", "ann_file"],
            str(annotation_path),
        )
        _set_if_present(config, ["trainer", "data", "train", "batch_size"], 1)
        _set_if_present(config, ["trainer", "data", "train", "num_workers"], 0)
        _set_if_present(config, ["trainer", "data", "val", "batch_size"], 1)
        _set_if_present(config, ["trainer", "data", "val", "num_workers"], 0)
        _set_if_present(config, ["trainer", "model", "load_from_HF"], False)
        _set_if_present(config, ["trainer", "model", "checkpoint_path"], None)
        _set_if_present(
            config,
            ["trainer", "checkpoint", "model_weight_initializer", "checkpoint_path"],
            checkpoint,
        )
        _set_if_present(config, ["trainer", "checkpoint", "save_dir"], str(run_dir / "checkpoints"))
        _set_if_present(config, ["trainer", "logging", "log_dir"], str(run_dir))
        _set_if_present(
            config,
            ["trainer", "logging", "tensorboard_writer", "log_dir"],
            str(run_dir / "tensorboard"),
        )
        _set_if_present(
            config,
            ["trainer", "meters", "val", "all", "counting", "detail_records_path"],
            str(detail_path),
        )
        _set_if_present(
            config,
            ["trainer", "meters", "val", "all", "counting", "expected_num_records"],
            1,
        )
        _set_if_present(
            config,
            ["trainer", "meters", "val", "all", "counting", "deduplicate_by_image_id"],
            False,
        )
        _set_if_present(config, ["launcher", "gpus_per_node"], self.num_gpus)
        _set_if_present(config, ["launcher", "experiment_log_dir"], experiment_dir)

    @staticmethod
    def _default_runner(command: Sequence[str], *, cwd: str, env: Dict[str, str]) -> None:
        subprocess.run(list(command), cwd=cwd, env=env, check=True)

    @staticmethod
    def _load_detail_records(detail_path: Path) -> Dict:
        if not detail_path.exists():
            raise RuntimeError(f"Prediction detail file was not created: {detail_path}")
        with detail_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _cleanup_auxiliary_run_files(run_dir: Path, keep_paths: Sequence[Path]) -> None:
        keep = {path.resolve() for path in keep_paths}
        for child in run_dir.iterdir():
            if child.resolve() in keep:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


__all__ = ["CountAnything", "CountAnythingResult"]
