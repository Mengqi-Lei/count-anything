from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return value or "query"


@dataclass
class CountAnythingResult:
    image_path: str
    text_query: str
    count: int
    pred_points: List[Dict[str, float]]
    run_dir: str

    def _render(self):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError("Pillow is required for visualization.") from exc

        image = Image.open(self.image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        radius = max(4, round(min(width, height) / 180))
        for point in self.pred_points:
            x = max(0.0, min(float(point["x"]), float(width - 1)))
            y = max(0.0, min(float(point["y"]), float(height - 1)))
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(230, 40, 35, 230),
                outline=(255, 255, 255, 255),
                width=max(1, radius // 2),
            )
        label = f"{self.text_query}: {self.count}"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, width // 45))
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        pad = 10
        draw.rectangle(
            [0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2],
            fill=(0, 0, 0, 170),
        )
        draw.text((pad, pad), label, fill=(255, 255, 255, 255), font=font)
        return image

    def show(self) -> None:
        self._render().show()

    def _json_payload(self) -> Dict:
        return {
            "image_path": self.image_path,
            "text_query": self.text_query,
            "count": self.count,
            "points": self.pred_points,
        }

    def save_json(self, path: str | os.PathLike | None = None) -> str:
        if path is None:
            filename = f"{Path(self.image_path).stem}__{_slugify(self.text_query)}.json"
            path = Path(self.run_dir) / filename
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._json_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def save(self, path: str | os.PathLike | None = None) -> str:
        if path is None:
            filename = f"{Path(self.image_path).stem}__{_slugify(self.text_query)}.jpg"
            path = Path(self.run_dir) / filename
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._render().save(path)
        self.save_json(path.with_suffix(".json"))
        return str(path)


class CountAnything:
    def __init__(
        self,
        checkpoint: str | os.PathLike = "checkpoints/count_anything.pt",
        *,
        config: str | os.PathLike = "config/count_anything_test_cloc.yaml",
        output_dir: str | os.PathLike = "exp/count_anything_inference",
        num_gpus: int = 1,
        python_executable: str = sys.executable,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.checkpoint = self._resolve_path(checkpoint)
        self.config = self._resolve_path(config)
        self.output_dir = self._resolve_path(output_dir)
        self.num_gpus = int(num_gpus)
        self.python_executable = python_executable

    def __call__(self, image_path: str | os.PathLike, text_query: str) -> List[CountAnythingResult]:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        text_query = str(text_query).strip()
        if not text_query:
            raise ValueError("text_query must be a non-empty string")

        run_dir = self._make_run_dir(image_path, text_query)
        annotation_path = run_dir / "temporary_inference_annotation.json"
        config_path = run_dir / "temporary_count_anything_inference.yaml"
        detail_path = run_dir / "temporary_prediction_records.json"

        self._write_annotation(annotation_path, image_path, text_query)
        self._write_config(config_path, annotation_path, detail_path, run_dir)
        self._run(config_path)

        record = self._load_record(detail_path)
        detail_path.unlink(missing_ok=True)
        (run_dir / "log.txt").unlink(missing_ok=True)
        return [
            CountAnythingResult(
                image_path=str(image_path),
                text_query=text_query,
                count=int(record["pred_count"]),
                pred_points=record.get("points", []),
                run_dir=str(run_dir),
            )
        ]

    def _resolve_path(self, path: str | os.PathLike) -> Path:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _make_run_dir(self, image_path: Path, text_query: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{image_path.stem}__{_slugify(text_query)}__{stamp}"
        run_dir = self.output_dir / name
        index = 2
        while run_dir.exists():
            run_dir = self.output_dir / f"{name}_{index}"
            index += 1
        run_dir.mkdir(parents=True)
        return run_dir

    def _write_annotation(self, path: Path, image_path: Path, text_query: str) -> None:
        annotation = {
            f"0__{_slugify(text_query)}": {
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
        path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_config(
        self,
        path: Path,
        annotation_path: Path,
        detail_path: Path,
        run_dir: Path,
    ) -> None:
        with self.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        checkpoint = str(self.checkpoint)
        annotation_file = str(annotation_path)
        config["paths"]["train_annotation_file"] = annotation_file
        config["paths"]["val_annotation_file"] = annotation_file
        config["paths"]["experiment_log_dir"] = str(run_dir)
        config["paths"]["stage1_checkpoint_path"] = checkpoint

        trainer = config["trainer"]
        for split in ("train", "val"):
            trainer["data"][split]["dataset"]["ann_file"] = annotation_file
            trainer["data"][split]["batch_size"] = 1
            trainer["data"][split]["num_workers"] = 0
        trainer["model"]["load_from_HF"] = False
        trainer["model"]["checkpoint_path"] = None
        trainer["checkpoint"]["model_weight_initializer"]["checkpoint_path"] = checkpoint
        trainer["checkpoint"]["save_dir"] = str(run_dir / "checkpoints")
        trainer["logging"]["log_dir"] = str(run_dir)
        trainer["logging"]["tensorboard_writer"]["log_dir"] = str(run_dir / "tensorboard")
        trainer["meters"]["val"]["all"]["counting"]["detail_records_path"] = str(detail_path)
        trainer["meters"]["val"]["all"]["counting"]["include_points"] = True
        trainer["meters"]["val"]["all"]["counting"]["include_scores"] = True
        trainer["meters"]["val"]["all"]["counting"]["expected_num_records"] = 1

        config["launcher"]["gpus_per_node"] = self.num_gpus
        config["launcher"]["experiment_log_dir"] = str(run_dir)

        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _run(self, config_path: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [
                self.python_executable,
                "-m",
                "count_anything.train.train",
                "-c",
                str(config_path),
                "--use-cluster",
                "0",
                "--num-gpus",
                str(self.num_gpus),
            ],
            cwd=str(self.repo_root),
            env=env,
            check=True,
        )

    @staticmethod
    def _load_record(detail_path: Path) -> Dict:
        with detail_path.open("r", encoding="utf-8") as f:
            records = json.load(f).get("predictions") or []
        if not records:
            raise RuntimeError(f"No prediction records were written to {detail_path}")
        return records[0]


__all__ = ["CountAnything", "CountAnythingResult"]
