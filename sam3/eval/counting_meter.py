# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import json
import math
import os
from typing import Dict, List

import numpy as np

from sam3.train.utils.distributed import all_gather
from sam3.train.utils.distributed import is_main_process


def _threshold_label_to_key(label: str) -> str:
    return label.replace(".", "_")


class CountingMeter:
    def __init__(
        self,
        postprocessor,
        gather_pred_via_filesys: bool = False,
        detail_records_path: str = None,
        deduplicate_by_image_id: bool = True,
        expected_num_records: int = None,
    ) -> None:
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = bool(gather_pred_via_filesys)
        self.detail_records_path = detail_records_path
        self.deduplicate_by_image_id = bool(deduplicate_by_image_id)
        self.expected_num_records = expected_num_records
        self.is_better = lambda cur, best: cur < best
        self.reset()

    def reset(self):
        self.records = {} if self.deduplicate_by_image_id else []

    def update(self, find_stages, find_metadatas, batch, **kwargs):
        # Reuse the CountAnything postprocessor, then aggregate one record per
        # image-class sample for distributed MAE/RMSE computation.
        predictions = None
        if self.deduplicate_by_image_id:
            predictions = self.postprocessor.process_results(
                find_stages=find_stages,
                find_metadatas=find_metadatas,
            )

        assert len(batch.find_targets) == len(find_metadatas)
        for stage_outputs, stage_target, stage_meta in zip(
            find_stages, batch.find_targets, find_metadatas
        ):
            if self.deduplicate_by_image_id:
                stage_predictions = None
            else:
                stage_predictions = self.postprocessor(
                    outputs=stage_outputs,
                    original_sizes=stage_meta.original_size,
                    processed_scales_xy=getattr(
                        stage_meta, "processed_scale_xy", None
                    ),
                    processed_offsets_xy=getattr(
                        stage_meta, "processed_offset_xy", None
                    ),
                )
            image_ids = (
                stage_meta.original_image_id
                if self.postprocessor.use_original_ids
                else stage_meta.coco_image_id
            )
            gt_counts = stage_target.num_boxes.detach().cpu().tolist()

            for sample_idx, (image_id, gt_count) in enumerate(
                zip(image_ids.detach().cpu().tolist(), gt_counts)
            ):
                pred = (
                    predictions[int(image_id)]
                    if self.deduplicate_by_image_id
                    else stage_predictions[sample_idx]
                )
                record = {
                    "image_id": int(image_id),
                    "gt_count": int(gt_count),
                    "pred_count": int(pred["pred_count"]),
                    "rsc_count": int(pred["rsc_count"]),
                    "pdc_count": int(pred["pdc_count"]),
                    "rsc_counts_by_threshold": dict(pred["rsc_counts_by_threshold"]),
                    "pdc_counts_by_threshold": dict(pred["pdc_counts_by_threshold"]),
                    "pdc_score_max": float(pred["pdc_score_max"]),
                    "pdc_score_min": float(pred["pdc_score_min"]),
                    "pdc_score_mean": float(pred["pdc_score_mean"]),
                    "pdc_score_median": float(pred["pdc_score_median"]),
                    "pdc_top60_scores": list(pred["pdc_top60_scores"]),
                }
                if self.deduplicate_by_image_id:
                    self.records[int(image_id)] = record
                else:
                    self.records.append(record)

    def _compute_metrics_from_records(self, records: List[Dict]) -> Dict[str, float]:
        threshold_labels = [
            f"{float(threshold):.2f}".rstrip("0").rstrip(".")
            for threshold in self.postprocessor.eval_thresholds
        ]
        if len(records) == 0:
            out = {
                "mae": 0.0,
                "mse": 0.0,
                "rmse": 0.0,
                "gt_count_avg": 0.0,
                "fused_count_avg": 0.0,
                "rsc_count_avg": 0.0,
                "pdc_count_avg": 0.0,
                "pdc_score_max": 0.0,
                "pdc_score_min": 0.0,
                "pdc_score_mean": 0.0,
                "pdc_score_median": 0.0,
                "fused_count_mae": 0.0,
                "rsc_count_mae": 0.0,
                "pdc_count_mae": 0.0,
                "fused_count_mse": 0.0,
                "rsc_count_mse": 0.0,
                "pdc_count_mse": 0.0,
                "fused_count_rmse": 0.0,
                "rsc_count_rmse": 0.0,
                "pdc_count_rmse": 0.0,
                "num_images": 0.0,
            }
            for topk_idx in range(1, 61):
                out[f"pdc_top60_score_{topk_idx:02d}"] = 0.0
            for threshold_label in threshold_labels:
                key_suffix = _threshold_label_to_key(threshold_label)
                out[f"rsc_count_mae_thr_{key_suffix}"] = 0.0
                out[f"rsc_count_mse_thr_{key_suffix}"] = 0.0
                out[f"pdc_count_mae_thr_{key_suffix}"] = 0.0
                out[f"pdc_count_mse_thr_{key_suffix}"] = 0.0
            return out

        gt = np.asarray([record["gt_count"] for record in records], dtype=np.float64)
        fused = np.asarray(
            [record["pred_count"] for record in records], dtype=np.float64
        )
        rsc = np.asarray([record["rsc_count"] for record in records], dtype=np.float64)
        pdc = np.asarray([record["pdc_count"] for record in records], dtype=np.float64)
        pdc_score_max = np.asarray(
            [record["pdc_score_max"] for record in records], dtype=np.float64
        )
        pdc_score_min = np.asarray(
            [record["pdc_score_min"] for record in records], dtype=np.float64
        )
        pdc_score_mean = np.asarray(
            [record["pdc_score_mean"] for record in records], dtype=np.float64
        )
        pdc_score_median = np.asarray(
            [record["pdc_score_median"] for record in records], dtype=np.float64
        )
        pdc_top60_scores = np.asarray(
            [record["pdc_top60_scores"] for record in records], dtype=np.float64
        )

        fused_abs = np.abs(fused - gt)
        rsc_abs = np.abs(rsc - gt)
        pdc_abs = np.abs(pdc - gt)

        fused_sq = np.square(fused - gt)
        rsc_sq = np.square(rsc - gt)
        pdc_sq = np.square(pdc - gt)

        out = {
            "mae": float(fused_abs.mean()),
            "mse": float(fused_sq.mean()),
            "rmse": float(math.sqrt(fused_sq.mean())),
            "gt_count_avg": float(gt.mean()),
            "fused_count_avg": float(fused.mean()),
            "rsc_count_avg": float(rsc.mean()),
            "pdc_count_avg": float(pdc.mean()),
            "pdc_score_max": float(pdc_score_max.max()),
            "pdc_score_min": float(pdc_score_min.min()),
            "pdc_score_mean": float(pdc_score_mean.mean()),
            "pdc_score_median": float(np.median(pdc_score_median)),
            "fused_count_mae": float(fused_abs.mean()),
            "rsc_count_mae": float(rsc_abs.mean()),
            "pdc_count_mae": float(pdc_abs.mean()),
            "fused_count_mse": float(fused_sq.mean()),
            "rsc_count_mse": float(rsc_sq.mean()),
            "pdc_count_mse": float(pdc_sq.mean()),
            "fused_count_rmse": float(math.sqrt(fused_sq.mean())),
            "rsc_count_rmse": float(math.sqrt(rsc_sq.mean())),
            "pdc_count_rmse": float(math.sqrt(pdc_sq.mean())),
            "num_images": float(len(records)),
        }

        for topk_idx in range(1, 61):
            top_score = float(pdc_top60_scores[:, topk_idx - 1].mean())
            out[f"pdc_top60_score_{topk_idx:02d}"] = top_score

        for threshold_label in threshold_labels:
            key_suffix = _threshold_label_to_key(threshold_label)
            rsc_threshold_counts = np.asarray(
                [
                    record["rsc_counts_by_threshold"][threshold_label]
                    for record in records
                ],
                dtype=np.float64,
            )
            pdc_threshold_counts = np.asarray(
                [
                    record["pdc_counts_by_threshold"][threshold_label]
                    for record in records
                ],
                dtype=np.float64,
            )
            rsc_sq_thr = np.square(rsc_threshold_counts - gt)
            pdc_sq_thr = np.square(pdc_threshold_counts - gt)
            out[f"rsc_count_mae_thr_{key_suffix}"] = float(
                np.abs(rsc_threshold_counts - gt).mean()
            )
            out[f"rsc_count_mse_thr_{key_suffix}"] = float(rsc_sq_thr.mean())
            out[f"pdc_count_mae_thr_{key_suffix}"] = float(
                np.abs(pdc_threshold_counts - gt).mean()
            )
            out[f"pdc_count_mse_thr_{key_suffix}"] = float(pdc_sq_thr.mean())

        return out

    def compute(self):
        records = (
            list(self.records.values())
            if self.deduplicate_by_image_id
            else list(self.records)
        )
        return self._compute_metrics_from_records(records)

    def compute_synced(self):
        ## 新添代码
        # 步骤一：把每张图的局部 count 记录做跨进程聚合。
        # 步骤二：按 image_id 去重，避免分布式或重复采样时同一张图被重复统计。
        # 步骤三：最终只输出 counting 任务真正需要的 MAE / RMSE 指标，不再依赖 COCO evaluator。
        local_records = (
            list(self.records.values())
            if self.deduplicate_by_image_id
            else list(self.records)
        )
        gathered_records = all_gather(
            local_records,
            force_cpu=True,
            force_filesys=self.gather_pred_via_filesys,
        )

        if self.deduplicate_by_image_id:
            merged_records = {}
            for rank_records in gathered_records:
                for record in rank_records:
                    merged_records[int(record["image_id"])] = record
            merged_record_list = list(merged_records.values())
        else:
            merged_record_list = []
            for rank_records in gathered_records:
                merged_record_list.extend(rank_records)
            if (
                self.expected_num_records is not None
                and len(merged_record_list) > int(self.expected_num_records)
            ):
                merged_record_list = merged_record_list[: int(self.expected_num_records)]

        out = self._compute_metrics_from_records(merged_record_list)
        if self.detail_records_path and is_main_process():
            os.makedirs(os.path.dirname(self.detail_records_path), exist_ok=True)
            sort_key = (
                (lambda record: record["image_id"])
                if self.deduplicate_by_image_id
                else None
            )
            records_to_dump = (
                sorted(merged_record_list, key=sort_key)
                if sort_key is not None
                else merged_record_list
            )
            with open(self.detail_records_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metrics": out,
                        "records": records_to_dump,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        self.reset()
        return out
        ## 新添代码
