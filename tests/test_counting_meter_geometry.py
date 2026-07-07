import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - torch is required by the model environment.
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class CountingMeterGeometryTest(unittest.TestCase):
    def test_update_serializes_prediction_geometry(self):
        from sam3.eval.counting_meter import CountingMeter

        class FakePostProcessor:
            use_original_ids = False
            eval_thresholds = (0.5,)

            def __call__(self, **kwargs):
                return [
                    {
                        "pred_count": 2,
                        "rsc_count": 1,
                        "pdc_count": 1,
                        "rsc_counts_by_threshold": {"0.5": 1},
                        "pdc_counts_by_threshold": {"0.5": 1},
                        "pdc_score_max": 0.9,
                        "pdc_score_min": 0.1,
                        "pdc_score_mean": 0.5,
                        "pdc_score_median": 0.5,
                        "pdc_top60_scores": [0.0] * 60,
                        "pred_count_points": torch.tensor([[10.0, 12.0], [20.0, 22.0]]),
                        "pred_count_scores": torch.tensor([0.9, 0.7]),
                        "pred_count_sources": torch.tensor([0, 1]),
                        "kept_rsc_boxes": torch.tensor([[5.0, 6.0, 15.0, 16.0]]),
                        "kept_rsc_points": torch.tensor([[10.0, 12.0]]),
                        "kept_pdc_points": torch.tensor([[20.0, 22.0]]),
                    }
                ]

        meter = CountingMeter(FakePostProcessor(), deduplicate_by_image_id=False)
        stage_meta = SimpleNamespace(
            coco_image_id=torch.tensor([123]),
            original_image_id=torch.tensor([456]),
            original_size=torch.tensor([[48, 64]]),
            processed_scale_xy=None,
            processed_offset_xy=None,
        )
        stage_target = SimpleNamespace(num_boxes=torch.tensor([0]))
        batch = SimpleNamespace(find_targets=[stage_target])

        meter.update(find_stages=[{}], find_metadatas=[stage_meta], batch=batch)

        record = meter.records[0]
        self.assertEqual([[10.0, 12.0], [20.0, 22.0]], record["pred_points"])
        self.assertEqual([0, 1], record["pred_sources"])
        self.assertEqual([[5.0, 6.0, 15.0, 16.0]], record["rsc_boxes"])
        self.assertEqual([[20.0, 22.0]], record["pdc_points"])


if __name__ == "__main__":
    unittest.main()
