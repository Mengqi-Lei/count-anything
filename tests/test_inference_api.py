import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - Pillow is optional for non-visual unit runs.
    PILImage = None


class CountAnythingInferenceTest(unittest.TestCase):
    def test_run_directory_uses_timestamp_without_random_id(self):
        from count_anything.inference import CountAnything

        class FixedDatetime:
            @staticmethod
            def now():
                class FixedNow:
                    def strftime(self, fmt):
                        return "20260707_230703"

                return FixedNow()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            image_path.write_bytes(b"not a real image; runner is mocked")
            model = CountAnything(
                checkpoint="checkpoints/count_anything.pt",
                runner=lambda *args, **kwargs: None,
                python_executable=sys.executable,
                output_dir=tmp_path / "out",
            )

            with patch("count_anything.inference.datetime", FixedDatetime):
                first = model._make_run_dir(image_path, "cells", None)
                second = model._make_run_dir(image_path, "cells", None)

            self.assertEqual("sample__cells__20260707_230703", first.name)
            self.assertEqual("sample__cells__20260707_230703_2", second.name)

    def test_single_image_query_uses_empty_points_and_returns_prediction(self):
        from count_anything import CountAnything

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            if PILImage is None:
                image_path.write_bytes(b"not a real image; runner is mocked")
            else:
                PILImage.new("RGB", (100, 100), color="white").save(image_path)
            output_dir = tmp_path / "out"
            captured = {}

            def fake_runner(command, *, cwd, env):
                captured["command"] = list(command)
                captured["cwd"] = Path(cwd)
                config_path = Path(command[command.index("-c") + 1])
                with config_path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                captured["config"] = config

                annotation_path = Path(config["paths"]["val_annotation_file"])
                with annotation_path.open("r", encoding="utf-8") as f:
                    captured["annotation"] = json.load(f)

                detail_path = Path(
                    config["trainer"]["meters"]["val"]["all"]["counting"][
                        "detail_records_path"
                    ]
                )
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                detail_path.write_text(
                    json.dumps(
                        {
                            "metrics": {"num_images": 1},
                            "records": [
                                {
                                    "image_id": 0,
                                    "gt_count": 0,
                                    "pred_count": 7,
                                    "rsc_count": 6,
                                    "pdc_count": 7,
                                    "pred_points": [[10.0, 12.0], [20.0, 22.0]],
                                    "pred_scores": [0.9, 0.7],
                                    "pred_sources": [0, 1],
                                    "rsc_boxes": [[70.0, 70.0, 90.0, 90.0]],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            model = CountAnything(
                checkpoint="checkpoints/count_anything.pt",
                runner=fake_runner,
                python_executable=sys.executable,
            )

            results = model(str(image_path), "cells", output_dir=output_dir)

            self.assertEqual(1, len(results))
            self.assertEqual(7, results[0].count)
            self.assertEqual([[10.0, 12.0], [20.0, 22.0]], results[0].pred_points)
            self.assertEqual([0, 1], results[0].pred_sources)
            self.assertEqual([[70.0, 70.0, 90.0, 90.0]], results[0].rsc_boxes)
            self.assertEqual("cells", results[0].text_query)
            self.assertEqual(str(image_path.resolve()), results[0].image_path)
            self.assertIn("-m", captured["command"])
            self.assertIn("count_anything.train.train", captured["command"])
            config_path = Path(captured["command"][captured["command"].index("-c") + 1])
            self.assertEqual("temporary_count_anything_inference.yaml", config_path.name)
            self.assertEqual(1, captured["config"]["launcher"]["gpus_per_node"])
            self.assertFalse(
                captured["config"]["trainer"]["model"].get("load_from_HF", True)
            )

            record = next(iter(captured["annotation"].values()))
            self.assertEqual(["cells"], record["classes"])
            self.assertEqual([], record["annotation"]["cells"]["point"])
            self.assertEqual(str(image_path.resolve()), record["image_path"])
            annotation_path = Path(captured["config"]["paths"]["val_annotation_file"])
            self.assertEqual(
                str(annotation_path), captured["config"]["paths"]["train_annotation_file"]
            )
            self.assertEqual(
                str(annotation_path),
                captured["config"]["trainer"]["data"]["train"]["dataset"]["ann_file"],
            )
            run_dir = Path(results[0].run_dir)
            detail_path = Path(
                captured["config"]["trainer"]["meters"]["val"]["all"]["counting"][
                    "detail_records_path"
                ]
            )
            self.assertEqual(
                run_dir / "temporary_inference_annotation.json",
                annotation_path,
            )
            default_stem = "sample__cells__prediction"
            self.assertEqual(run_dir / f"{default_stem}.json", detail_path)
            self.assertTrue(detail_path.exists())

            if PILImage is not None:
                vis_path = tmp_path / "vis.jpg"
                self.assertEqual(str(vis_path), results[0].save(vis_path))
                self.assertTrue(vis_path.exists())
                default_no_box_path = tmp_path / "default_no_box.png"
                results[0].save(default_no_box_path)
                self.assertEqual(
                    (255, 255, 255),
                    PILImage.open(default_no_box_path).convert("RGB").getpixel((70, 70)),
                )
                with_box_path = tmp_path / "with_box.png"
                results[0].save(with_box_path, show_boxes=True)
                self.assertNotEqual(
                    (255, 255, 255),
                    PILImage.open(with_box_path).convert("RGB").getpixel((70, 70)),
                )
                default_vis_path = run_dir / f"{default_stem}.jpg"
                self.assertEqual(str(default_vis_path), results[0].save())
                self.assertTrue(default_vis_path.exists())
                self.assertTrue(default_vis_path.is_absolute())

    def test_default_removes_auxiliary_outputs(self):
        from count_anything import CountAnything

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            if PILImage is None:
                image_path.write_bytes(b"not a real image; runner is mocked")
            else:
                PILImage.new("RGB", (32, 32), color="white").save(image_path)
            output_dir = tmp_path / "out"
            captured = {}

            def fake_runner(command, *, cwd, env):
                config_path = Path(command[command.index("-c") + 1])
                with config_path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                detail_path = Path(
                    config["trainer"]["meters"]["val"]["all"]["counting"][
                        "detail_records_path"
                    ]
                )
                run_dir = detail_path.parent
                captured["run_dir"] = run_dir
                captured["config_path"] = config_path
                captured["annotation_path"] = Path(config["paths"]["val_annotation_file"])

                detail_path.write_text(
                    json.dumps(
                        {
                            "metrics": {"num_images": 1},
                            "records": [{"image_id": 0, "gt_count": 0, "pred_count": 3}],
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "config.yaml").write_text("temporary config", encoding="utf-8")
                (run_dir / "config_resolved.yaml").write_text(
                    "temporary config", encoding="utf-8"
                )
                (run_dir / "log.txt").write_text("temporary log", encoding="utf-8")
                (run_dir / "val_stats.json").write_text("{}", encoding="utf-8")
                tensorboard_dir = run_dir / "tensorboard"
                tensorboard_dir.mkdir(parents=True, exist_ok=True)
                (tensorboard_dir / "events.out.tfevents.mock").write_text(
                    "temporary tensorboard", encoding="utf-8"
                )

            model = CountAnything(
                checkpoint="checkpoints/count_anything.pt",
                runner=fake_runner,
                python_executable=sys.executable,
            )

            results = model(str(image_path), "cells", output_dir=output_dir)

            run_dir = captured["run_dir"]
            prediction_json = run_dir / "sample__cells__prediction.json"
            self.assertEqual(3, results[0].count)
            self.assertTrue(prediction_json.exists())
            self.assertFalse(captured["config_path"].exists())
            self.assertFalse(captured["annotation_path"].exists())
            self.assertFalse((run_dir / "config.yaml").exists())
            self.assertFalse((run_dir / "config_resolved.yaml").exists())
            self.assertFalse((run_dir / "log.txt").exists())
            self.assertFalse((run_dir / "val_stats.json").exists())
            self.assertFalse((run_dir / "tensorboard").exists())

    def test_keep_run_files_true_preserves_auxiliary_outputs(self):
        from count_anything import CountAnything

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            image_path.write_bytes(b"not a real image; runner is mocked")
            output_dir = tmp_path / "out"
            captured = {}

            def fake_runner(command, *, cwd, env):
                config_path = Path(command[command.index("-c") + 1])
                with config_path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                detail_path = Path(
                    config["trainer"]["meters"]["val"]["all"]["counting"][
                        "detail_records_path"
                    ]
                )
                run_dir = detail_path.parent
                captured["run_dir"] = run_dir

                detail_path.write_text(
                    json.dumps(
                        {
                            "metrics": {"num_images": 1},
                            "records": [{"image_id": 0, "gt_count": 0, "pred_count": 3}],
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "config.yaml").write_text("temporary config", encoding="utf-8")
                (run_dir / "log.txt").write_text("temporary log", encoding="utf-8")
                (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

            model = CountAnything(
                checkpoint="checkpoints/count_anything.pt",
                runner=fake_runner,
                python_executable=sys.executable,
                keep_run_files=True,
            )

            results = model(str(image_path), "cells", output_dir=output_dir)

            run_dir = captured["run_dir"]
            self.assertEqual(3, results[0].count)
            self.assertTrue((run_dir / "sample__cells__prediction.json").exists())
            self.assertTrue((run_dir / "temporary_count_anything_inference.yaml").exists())
            self.assertTrue((run_dir / "temporary_inference_annotation.json").exists())
            self.assertTrue((run_dir / "config.yaml").exists())
            self.assertTrue((run_dir / "log.txt").exists())
            self.assertTrue((run_dir / "tensorboard").exists())


if __name__ == "__main__":
    unittest.main()
