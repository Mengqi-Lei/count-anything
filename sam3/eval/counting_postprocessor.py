# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch
from sam3.model import box_ops
from sam3.model.data_misc import BatchedInferenceMetadata
from torch import nn


SOURCE_RSC = 0
SOURCE_PDC = 1


def _format_threshold_label(threshold: float) -> str:
    return f"{float(threshold):.2f}".rstrip("0").rstrip(".")


def _score_filter(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    return scores > float(threshold)


def _points_in_boxes(points: torch.Tensor, boxes_xyxy: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0 or boxes_xyxy.numel() == 0:
        return points.new_zeros((boxes_xyxy.shape[0], points.shape[0]), dtype=torch.bool)
    px = points[:, 0][None, :]
    py = points[:, 1][None, :]
    x1 = boxes_xyxy[:, 0:1]
    y1 = boxes_xyxy[:, 1:2]
    x2 = boxes_xyxy[:, 2:3]
    y2 = boxes_xyxy[:, 3:4]
    return (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)


def _box_areas_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((boxes_xyxy.shape[0],))
    wh = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).clamp_min(0.0)
    return wh[:, 0] * wh[:, 1]


def _box_centers_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((boxes_xyxy.shape[0], 2))
    return 0.5 * (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:])


def _box_diagonals_xyxy(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.new_zeros((boxes_xyxy.shape[0],))
    wh = (boxes_xyxy[:, 2:] - boxes_xyxy[:, :2]).clamp_min(0.0)
    return torch.norm(wh, dim=1)


def _pairwise_box_iom(
    boxes1_xyxy: torch.Tensor,
    boxes2_xyxy: torch.Tensor,
) -> torch.Tensor:
    if boxes1_xyxy.numel() == 0 or boxes2_xyxy.numel() == 0:
        return boxes1_xyxy.new_zeros((boxes1_xyxy.shape[0], boxes2_xyxy.shape[0]))
    lt = torch.maximum(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.minimum(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = _box_areas_xyxy(boxes1_xyxy)
    area2 = _box_areas_xyxy(boxes2_xyxy)
    min_area = torch.minimum(area1[:, None], area2[None, :]).clamp_min(1e-6)
    return inter / min_area


def _pairwise_box_full_containment(
    boxes1_xyxy: torch.Tensor,
    boxes2_xyxy: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    if boxes1_xyxy.numel() == 0 or boxes2_xyxy.numel() == 0:
        return boxes1_xyxy.new_zeros(
            (boxes1_xyxy.shape[0], boxes2_xyxy.shape[0]), dtype=torch.bool
        )

    boxes1_in_boxes2 = (
        (boxes1_xyxy[:, None, 0] >= (boxes2_xyxy[None, :, 0] - eps))
        & (boxes1_xyxy[:, None, 1] >= (boxes2_xyxy[None, :, 1] - eps))
        & (boxes1_xyxy[:, None, 2] <= (boxes2_xyxy[None, :, 2] + eps))
        & (boxes1_xyxy[:, None, 3] <= (boxes2_xyxy[None, :, 3] + eps))
    )
    boxes2_in_boxes1 = (
        (boxes2_xyxy[None, :, 0] >= (boxes1_xyxy[:, None, 0] - eps))
        & (boxes2_xyxy[None, :, 1] >= (boxes1_xyxy[:, None, 1] - eps))
        & (boxes2_xyxy[None, :, 2] <= (boxes1_xyxy[:, None, 2] + eps))
        & (boxes2_xyxy[None, :, 3] <= (boxes1_xyxy[:, None, 3] + eps))
    )
    return boxes1_in_boxes2 | boxes2_in_boxes1


def _filter_rsc_detections(
    rsc_points: torch.Tensor,
    rsc_boxes_xyxy: torch.Tensor,
    rsc_scores: torch.Tensor,
    *,
    score_threshold: float,
    iom_filter_enabled: bool,
    iom_threshold: float,
    center_distance_ratio: float,
):
    rsc_keep = _score_filter(rsc_scores, score_threshold)
    rsc_points = rsc_points[rsc_keep]
    rsc_boxes_xyxy = rsc_boxes_xyxy[rsc_keep]
    rsc_scores = rsc_scores[rsc_keep]

    if (
        not iom_filter_enabled
        or rsc_boxes_xyxy.shape[0] <= 1
        or center_distance_ratio <= 0.0
    ):
        return rsc_points, rsc_boxes_xyxy, rsc_scores

    order = torch.argsort(rsc_scores, descending=True)
    kept_indices = []

    for idx in order.tolist():
        if not kept_indices:
            kept_indices.append(idx)
            continue

        kept_idx_tensor = torch.tensor(
            kept_indices,
            device=rsc_boxes_xyxy.device,
            dtype=torch.long,
        )
        iom_vals = _pairwise_box_iom(
            rsc_boxes_xyxy[idx : idx + 1], rsc_boxes_xyxy[kept_idx_tensor]
        ).squeeze(0)
        full_containment = _pairwise_box_full_containment(
            rsc_boxes_xyxy[idx : idx + 1], rsc_boxes_xyxy[kept_idx_tensor]
        ).squeeze(0)
        is_duplicate = torch.any(
            full_containment | (iom_vals > float(iom_threshold))
        )
        if not bool(is_duplicate.item()):
            kept_indices.append(idx)

    kept_idx_tensor = torch.tensor(
        kept_indices,
        device=rsc_boxes_xyxy.device,
        dtype=torch.long,
    )
    return (
        rsc_points[kept_idx_tensor],
        rsc_boxes_xyxy[kept_idx_tensor],
        rsc_scores[kept_idx_tensor],
    )


def _filter_sam_detections(*args, **kwargs):
    """Legacy wrapper for old internal callers."""
    return _filter_rsc_detections(*args, **kwargs)


@torch.no_grad()
def complementary_count_fusion(
    rsc_points: torch.Tensor | None = None,
    rsc_boxes_xyxy: torch.Tensor | None = None,
    rsc_scores: torch.Tensor | None = None,
    pdc_points: torch.Tensor | None = None,
    pdc_scores: torch.Tensor | None = None,
    score_threshold: float | None = None,
    rsc_score_threshold: float = 0.5,
    pdc_score_threshold: float = 0.55,
    rsc_iom_filter_enabled: bool = True,
    rsc_iom_threshold: float = 0.7,
    rsc_center_distance_ratio: float = 0.5,
    fusion_strategy: str = "ccf",
    ccf_max_removed_pdc_per_rsc: int = 1,
):
    if (
        rsc_points is None
        or rsc_boxes_xyxy is None
        or rsc_scores is None
        or pdc_points is None
        or pdc_scores is None
    ):
        raise ValueError("RSC and PDC points/scores are required for CCF.")

    if score_threshold is not None:
        rsc_score_threshold = float(score_threshold)
        pdc_score_threshold = float(score_threshold)
    fusion_strategy = str(fusion_strategy).lower().replace("_", "-")
    if fusion_strategy in {"complementary-count-fusion", "fusion"}:
        fusion_strategy = "ccf"
    elif fusion_strategy in {"remove-all", "remove-all-inside-rsc"}:
        fusion_strategy = "remove-all-inside"
    if fusion_strategy not in {"ccf", "concat", "remove-all-inside"}:
        raise ValueError(
            "fusion_strategy must be one of: ccf, concat, remove-all-inside; "
            f"got {fusion_strategy!r}"
        )

    pdc_keep = _score_filter(pdc_scores, pdc_score_threshold)

    rsc_points, rsc_boxes_xyxy, rsc_scores = _filter_rsc_detections(
        rsc_points,
        rsc_boxes_xyxy,
        rsc_scores,
        score_threshold=rsc_score_threshold,
        iom_filter_enabled=rsc_iom_filter_enabled,
        iom_threshold=rsc_iom_threshold,
        center_distance_ratio=rsc_center_distance_ratio,
    )
    pdc_points = pdc_points[pdc_keep]
    pdc_scores = pdc_scores[pdc_keep]

    if rsc_points.numel() == 0:
        removed_pdc_points = pdc_points.new_zeros((0, 2))
        removed_pdc_scores = pdc_scores.new_zeros((0,))
        return {
            "pred_count_points": pdc_points,
            "pred_count_scores": pdc_scores,
            "pred_count_sources": torch.full(
                (pdc_points.shape[0],),
                SOURCE_PDC,
                dtype=torch.long,
                device=pdc_points.device,
            ),
            "kept_pdc_points": pdc_points,
            "removed_pdc_points": removed_pdc_points,
            "kept_pdc_scores": pdc_scores,
            "removed_pdc_scores": removed_pdc_scores,
            "kept_rsc_points": rsc_points,
            "kept_rsc_boxes": rsc_boxes_xyxy,
            "kept_rsc_scores": rsc_scores,
        }

    alive = torch.ones(pdc_points.shape[0], dtype=torch.bool, device=pdc_points.device)
    inside = _points_in_boxes(pdc_points, rsc_boxes_xyxy)

    if fusion_strategy == "concat":
        pass
    elif fusion_strategy == "remove-all-inside":
        alive = ~inside.any(dim=0)
    else:
        max_removed_per_rsc = max(1, int(ccf_max_removed_pdc_per_rsc))
        for rsc_idx in range(rsc_boxes_xyxy.shape[0]):
            current_candidates = inside[rsc_idx] & alive
            if current_candidates.any():
                local_points = pdc_points[current_candidates]
                local_distance = torch.norm(
                    local_points - rsc_points[rsc_idx : rsc_idx + 1], dim=1
                )
                local_indices = torch.nonzero(
                    current_candidates, as_tuple=False
                ).flatten()
                num_to_remove = min(max_removed_per_rsc, int(local_indices.numel()))
                remove_order = torch.argsort(local_distance)[:num_to_remove]
                alive[local_indices[remove_order]] = False

    kept_pdc_points = pdc_points[alive]
    removed_pdc_points = pdc_points[~alive]
    fused_points = torch.cat([rsc_points, kept_pdc_points], dim=0)
    fused_scores = torch.cat([rsc_scores, pdc_scores[alive]], dim=0)
    fused_sources = torch.cat(
        [
            torch.full(
                (rsc_points.shape[0],),
                SOURCE_RSC,
                dtype=torch.long,
                device=fused_points.device,
            ),
            torch.full(
                (int(alive.sum().item()),),
                SOURCE_PDC,
                dtype=torch.long,
                device=fused_points.device,
            ),
        ],
        dim=0,
    )
    result = {
        "pred_count_points": fused_points,
        "pred_count_scores": fused_scores,
        "pred_count_sources": fused_sources,
        "kept_pdc_points": kept_pdc_points,
        "removed_pdc_points": removed_pdc_points,
        "kept_pdc_scores": pdc_scores[alive],
        "removed_pdc_scores": pdc_scores[~alive],
        "kept_rsc_points": rsc_points,
        "kept_rsc_boxes": rsc_boxes_xyxy,
        "kept_rsc_scores": rsc_scores,
    }
    return result


@torch.no_grad()
def fuse_simple_bbox(
    sam_points: torch.Tensor,
    sam_boxes_xyxy: torch.Tensor,
    sam_scores: torch.Tensor,
    p2p_points: torch.Tensor,
    p2p_scores: torch.Tensor,
    score_threshold: float | None = None,
    sam_score_threshold: float = 0.5,
    p2p_score_threshold: float = 0.55,
    sam_iom_filter_enabled: bool = True,
    sam_iom_threshold: float = 0.7,
    sam_center_distance_ratio: float = 0.5,
):
    """Legacy wrapper for Complementary Count Fusion."""
    return complementary_count_fusion(
        rsc_points=sam_points,
        rsc_boxes_xyxy=sam_boxes_xyxy,
        rsc_scores=sam_scores,
        pdc_points=p2p_points,
        pdc_scores=p2p_scores,
        score_threshold=score_threshold,
        rsc_score_threshold=sam_score_threshold,
        pdc_score_threshold=p2p_score_threshold,
        rsc_iom_filter_enabled=sam_iom_filter_enabled,
        rsc_iom_threshold=sam_iom_threshold,
        rsc_center_distance_ratio=sam_center_distance_ratio,
    )


class CountAnythingPostProcessor(nn.Module):
    def __init__(
        self,
        use_original_ids: bool = True,
        score_threshold: float | None = None,
        pdc_score_threshold: float = 0.55,
        rsc_score_threshold: float = 0.5,
        eval_thresholds=None,
        use_presence: bool = True,
        to_cpu: bool = True,
        rsc_iom_filter_enabled: bool = True,
        rsc_iom_threshold: float = 0.7,
        rsc_center_distance_ratio: float = 0.5,
        fusion_strategy: str = "ccf",
        ccf_max_removed_pdc_per_rsc: int = 1,
    ) -> None:
        super().__init__()
        self.use_original_ids = use_original_ids
        if score_threshold is not None:
            pdc_score_threshold = float(score_threshold)
            rsc_score_threshold = float(score_threshold)
        self.pdc_score_threshold = float(pdc_score_threshold)
        self.rsc_score_threshold = float(rsc_score_threshold)
        if eval_thresholds is None:
            eval_thresholds = [0.05 * i for i in range(1, 11)] + [
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ]
        self.eval_thresholds = tuple(float(thr) for thr in eval_thresholds)
        self.use_presence = bool(use_presence)
        self.to_cpu = bool(to_cpu)
        self.rsc_iom_filter_enabled = bool(rsc_iom_filter_enabled)
        self.rsc_iom_threshold = float(rsc_iom_threshold)
        self.rsc_center_distance_ratio = float(rsc_center_distance_ratio)
        self.fusion_strategy = str(fusion_strategy)
        self.ccf_max_removed_pdc_per_rsc = int(ccf_max_removed_pdc_per_rsc)

    def _get_input_sizes(self, outputs: dict, ref: torch.Tensor) -> torch.Tensor:
        if "input_image_size" in outputs:
            return outputs["input_image_size"].to(ref)
        raise KeyError("Expected input_image_size in model outputs.")

    def _get_rsc_scores(self, outputs: dict) -> torch.Tensor:
        if "rsc_logits" in outputs:
            pred_logits = outputs["rsc_logits"]
        else:
            pred_logits = outputs["pred_logits"]
        if pred_logits.shape[-1] == 1:
            rsc_scores = pred_logits.squeeze(-1).sigmoid()
        else:
            rsc_scores = pred_logits.sigmoid().amax(dim=-1)

        if self.use_presence and "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"]
            if presence_score.ndim == 1:
                presence_score = presence_score[:, None]
            rsc_scores = rsc_scores * presence_score.sigmoid()
        return rsc_scores

    def _get_rsc_geometry(self, outputs: dict):
        if "rsc_boxes" in outputs:
            pred_boxes = outputs["rsc_boxes"]
        else:
            pred_boxes = outputs["pred_boxes"]
        input_sizes = self._get_input_sizes(outputs, pred_boxes)
        input_h = input_sizes[:, 0]
        input_w = input_sizes[:, 1]
        point_scale = torch.stack([input_w, input_h], dim=-1)
        box_scale = torch.stack([input_w, input_h, input_w, input_h], dim=-1)

        rsc_points = pred_boxes[..., :2] * point_scale[:, None, :]
        if "rsc_boxes_xyxy" in outputs:
            rsc_boxes_xyxy = outputs["rsc_boxes_xyxy"] * box_scale[:, None, :]
        elif "pred_boxes_xyxy" in outputs:
            rsc_boxes_xyxy = outputs["pred_boxes_xyxy"] * box_scale[:, None, :]
        else:
            rsc_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes)
            rsc_boxes_xyxy = rsc_boxes_xyxy * box_scale[:, None, :]
        return rsc_points, rsc_boxes_xyxy

    def _get_pdc_geometry(self, outputs: dict):
        if "pdc_points" in outputs and "pdc_logits" in outputs:
            pdc_points = outputs["pdc_points"]
            pdc_scores = outputs["pdc_logits"].softmax(-1)[..., 1]
            return list(pdc_points), list(pdc_scores)

        pred_boxes = outputs.get("rsc_boxes", outputs["pred_boxes"])
        device = pred_boxes.device
        batch_size = pred_boxes.shape[0]
        empty_points = [
            torch.empty((0, 2), dtype=torch.float32, device=device)
            for _ in range(batch_size)
        ]
        empty_scores = [
            torch.empty((0,), dtype=torch.float32, device=device)
            for _ in range(batch_size)
        ]
        return empty_points, empty_scores

    def _compute_score_stats(self, scores: torch.Tensor) -> dict:
        if scores.numel() == 0:
            return {
                "pdc_score_max": 0.0,
                "pdc_score_min": 0.0,
                "pdc_score_mean": 0.0,
                "pdc_score_median": 0.0,
                "pdc_top60_scores": [0.0] * 60,
            }
        topk = min(60, int(scores.numel()))
        top_scores = torch.topk(scores, k=topk, largest=True).values
        if topk < 60:
            pad = scores.new_zeros(60 - topk)
            top_scores = torch.cat([top_scores, pad], dim=0)
        return {
            "pdc_score_max": float(scores.max().item()),
            "pdc_score_min": float(scores.min().item()),
            "pdc_score_mean": float(scores.mean().item()),
            "pdc_score_median": float(scores.median().item()),
            "pdc_top60_scores": [float(v.item()) for v in top_scores],
        }

    def _compute_threshold_counts(self, scores: torch.Tensor) -> dict:
        threshold_counts = {}
        for threshold in self.eval_thresholds:
            threshold_counts[_format_threshold_label(threshold)] = int(
                (scores > threshold).sum().item()
            )
        return threshold_counts

    def _compute_rsc_threshold_counts(
        self,
        rsc_points: torch.Tensor,
        rsc_boxes_xyxy: torch.Tensor,
        rsc_scores: torch.Tensor,
    ) -> dict:
        threshold_counts = {}
        for threshold in self.eval_thresholds:
            _, boxes_thr, _ = _filter_rsc_detections(
                rsc_points,
                rsc_boxes_xyxy,
                rsc_scores,
                score_threshold=threshold,
                iom_filter_enabled=self.rsc_iom_filter_enabled,
                iom_threshold=self.rsc_iom_threshold,
                center_distance_ratio=self.rsc_center_distance_ratio,
            )
            threshold_counts[_format_threshold_label(threshold)] = int(
                boxes_thr.shape[0]
            )
        return threshold_counts

    def _scale_points_to_original_size(
        self,
        points_input_space: torch.Tensor,
        input_hw: torch.Tensor,
        original_hw: torch.Tensor,
        processed_scale_xy: torch.Tensor | None = None,
        processed_offset_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points_input_space.numel() == 0:
            return points_input_space
        if processed_scale_xy is not None and processed_offset_xy is not None:
            scale = processed_scale_xy.to(points_input_space).view(2)
            offset = processed_offset_xy.to(points_input_space).view(2)
            safe_scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
            points_original = (points_input_space - offset) / safe_scale
            max_x = max(float(original_hw[1]) - 1.0, 0.0)
            max_y = max(float(original_hw[0]) - 1.0, 0.0)
            points_original[..., 0].clamp_(min=0.0, max=max_x)
            points_original[..., 1].clamp_(min=0.0, max=max_y)
            return points_original
        scale = points_input_space.new_tensor(
            (
                float(original_hw[1]) / max(float(input_hw[1]), 1.0),
                float(original_hw[0]) / max(float(input_hw[0]), 1.0),
            )
        )
        return points_input_space * scale

    def _scale_boxes_to_original_size(
        self,
        boxes_xyxy_input_space: torch.Tensor,
        input_hw: torch.Tensor,
        original_hw: torch.Tensor,
        processed_scale_xy: torch.Tensor | None = None,
        processed_offset_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boxes_xyxy_input_space.numel() == 0:
            return boxes_xyxy_input_space
        if processed_scale_xy is not None and processed_offset_xy is not None:
            scale = processed_scale_xy.to(boxes_xyxy_input_space).view(2)
            offset = processed_offset_xy.to(boxes_xyxy_input_space).view(2)
            safe_scale = torch.where(scale.abs() < 1e-6, torch.ones_like(scale), scale)
            boxes = boxes_xyxy_input_space.clone()
            boxes[:, 0] = (boxes[:, 0] - offset[0]) / safe_scale[0]
            boxes[:, 2] = (boxes[:, 2] - offset[0]) / safe_scale[0]
            boxes[:, 1] = (boxes[:, 1] - offset[1]) / safe_scale[1]
            boxes[:, 3] = (boxes[:, 3] - offset[1]) / safe_scale[1]
            x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
            x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
            y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
            y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            max_x = max(float(original_hw[1]), 0.0)
            max_y = max(float(original_hw[0]), 0.0)
            boxes[:, [0, 2]].clamp_(min=0.0, max=max_x)
            boxes[:, [1, 3]].clamp_(min=0.0, max=max_y)
            return boxes
        scale = boxes_xyxy_input_space.new_tensor(
            (
                float(original_hw[1]) / max(float(input_hw[1]), 1.0),
                float(original_hw[0]) / max(float(input_hw[0]), 1.0),
                float(original_hw[1]) / max(float(input_hw[1]), 1.0),
                float(original_hw[0]) / max(float(input_hw[0]), 1.0),
            )
        )
        return boxes_xyxy_input_space * scale

    @torch.no_grad()
    def forward(
        self,
        outputs,
        original_sizes,
        processed_scales_xy: torch.Tensor | None = None,
        processed_offsets_xy: torch.Tensor | None = None,
    ):
        rsc_scores = self._get_rsc_scores(outputs)
        rsc_points, rsc_boxes_xyxy = self._get_rsc_geometry(outputs)
        pdc_points_list, pdc_scores_list = self._get_pdc_geometry(outputs)
        input_sizes = self._get_input_sizes(outputs, rsc_boxes_xyxy)

        results = []
        for batch_idx in range(rsc_scores.shape[0]):
            processed_scale_xy = (
                processed_scales_xy[batch_idx]
                if processed_scales_xy is not None
                else None
            )
            processed_offset_xy = (
                processed_offsets_xy[batch_idx]
                if processed_offsets_xy is not None
                else None
            )
            rsc_points_filtered, rsc_boxes_filtered, rsc_scores_filtered = _filter_rsc_detections(
                rsc_points[batch_idx],
                rsc_boxes_xyxy[batch_idx],
                rsc_scores[batch_idx],
                score_threshold=self.rsc_score_threshold,
                iom_filter_enabled=self.rsc_iom_filter_enabled,
                iom_threshold=self.rsc_iom_threshold,
                center_distance_ratio=self.rsc_center_distance_ratio,
            )
            pdc_keep = _score_filter(
                pdc_scores_list[batch_idx], self.pdc_score_threshold
            )
            raw_pdc_points = self._scale_points_to_original_size(
                pdc_points_list[batch_idx][pdc_keep],
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            raw_rsc_points = self._scale_points_to_original_size(
                rsc_points_filtered,
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            raw_rsc_boxes = self._scale_boxes_to_original_size(
                rsc_boxes_filtered,
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            raw_rsc_scores = rsc_scores_filtered

            fused = complementary_count_fusion(
                rsc_points=rsc_points[batch_idx],
                rsc_boxes_xyxy=rsc_boxes_xyxy[batch_idx],
                rsc_scores=rsc_scores[batch_idx],
                pdc_points=pdc_points_list[batch_idx],
                pdc_scores=pdc_scores_list[batch_idx],
                rsc_score_threshold=self.rsc_score_threshold,
                pdc_score_threshold=self.pdc_score_threshold,
                rsc_iom_filter_enabled=self.rsc_iom_filter_enabled,
                rsc_iom_threshold=self.rsc_iom_threshold,
                rsc_center_distance_ratio=self.rsc_center_distance_ratio,
                fusion_strategy=self.fusion_strategy,
                ccf_max_removed_pdc_per_rsc=self.ccf_max_removed_pdc_per_rsc,
            )
            pred_count_points = self._scale_points_to_original_size(
                fused["pred_count_points"],
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            pred_count_sources = fused["pred_count_sources"]
            pred_count_scores = fused["pred_count_scores"]
            kept_pdc_points = self._scale_points_to_original_size(
                fused["kept_pdc_points"],
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            removed_pdc_points = self._scale_points_to_original_size(
                fused["removed_pdc_points"],
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            kept_rsc_points = self._scale_points_to_original_size(
                fused["kept_rsc_points"],
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )
            kept_rsc_boxes = self._scale_boxes_to_original_size(
                fused["kept_rsc_boxes"],
                input_hw=input_sizes[batch_idx],
                original_hw=original_sizes[batch_idx],
                processed_scale_xy=processed_scale_xy,
                processed_offset_xy=processed_offset_xy,
            )

            result = {
                "pred_count_points": pred_count_points,
                "pred_count_scores": pred_count_scores,
                "pred_count_sources": pred_count_sources,
                "pred_count": int(pred_count_points.shape[0]),
                "rsc_count": int(raw_rsc_boxes.shape[0]),
                "pdc_count": int(raw_pdc_points.shape[0]),
                "raw_pdc_points": raw_pdc_points,
                "raw_rsc_points": raw_rsc_points,
                "raw_rsc_boxes": raw_rsc_boxes,
                "raw_rsc_scores": raw_rsc_scores,
                "kept_pdc_points": kept_pdc_points,
                "removed_pdc_points": removed_pdc_points,
                "kept_pdc_scores": fused["kept_pdc_scores"],
                "removed_pdc_scores": fused["removed_pdc_scores"],
                "kept_rsc_points": kept_rsc_points,
                "kept_rsc_boxes": kept_rsc_boxes,
                "kept_rsc_scores": fused["kept_rsc_scores"],
            }
            result.update(self._compute_score_stats(pdc_scores_list[batch_idx]))
            result["rsc_counts_by_threshold"] = self._compute_rsc_threshold_counts(
                rsc_points[batch_idx],
                rsc_boxes_xyxy[batch_idx],
                rsc_scores[batch_idx],
            )
            result["pdc_counts_by_threshold"] = self._compute_threshold_counts(
                pdc_scores_list[batch_idx]
            )

            if self.to_cpu:
                for key, value in list(result.items()):
                    if torch.is_tensor(value):
                        result[key] = value.cpu()
            results.append(result)

        return results

    @torch.no_grad()
    def process_results(
        self,
        find_stages,
        find_metadatas: list[BatchedInferenceMetadata],
        **kwargs,
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            stage_results = self(
                outputs=outputs,
                original_sizes=meta.original_size,
                processed_scales_xy=getattr(meta, "processed_scale_xy", None),
                processed_offsets_xy=getattr(meta, "processed_offset_xy", None),
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            assert len(stage_results) == len(ids)
            for image_id, stage_result in zip(ids, stage_results):
                results[int(image_id.item())] = stage_result
        return results


CountingPostProcessor = CountAnythingPostProcessor
