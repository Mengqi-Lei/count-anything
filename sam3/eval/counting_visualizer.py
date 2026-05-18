from __future__ import annotations

import os
from typing import Iterable, Optional

import torch
from PIL import Image, ImageDraw


def _to_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image.copy()


def _to_point_tuples(points: Optional[torch.Tensor]) -> list[tuple[float, float]]:
    if points is None or points.numel() == 0:
        return []
    points_cpu = points.detach().cpu().to(torch.float32)
    return [(float(x), float(y)) for x, y in points_cpu.tolist()]


def _to_box_tuples(boxes_xyxy: Optional[torch.Tensor]) -> list[tuple[float, float, float, float]]:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return []
    boxes_cpu = boxes_xyxy.detach().cpu().to(torch.float32)
    return [tuple(float(v) for v in box) for box in boxes_cpu.tolist()]


def _box_centers_from_xyxy(
    boxes_xyxy: Optional[torch.Tensor],
) -> list[tuple[float, float]]:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return []
    boxes_cpu = boxes_xyxy.detach().cpu().to(torch.float32)
    centers = []
    for x1, y1, x2, y2 in boxes_cpu.tolist():
        centers.append((float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)))
    return centers


def _to_score_list(scores: Optional[torch.Tensor]) -> list[float]:
    if scores is None or scores.numel() == 0:
        return []
    return [float(v) for v in scores.detach().cpu().to(torch.float32).tolist()]


def _draw_points(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[float, float]],
    *,
    color: tuple[int, int, int],
    radius: int = 2,
) -> None:
    for x, y in points:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=color,
        )


def _score_to_coolwarm_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    if score <= 0.5:
        t = score / 0.5
        c0 = (49, 130, 189)
        c1 = (171, 217, 233)
    else:
        t = (score - 0.5) / 0.5
        c0 = (253, 174, 97)
        c1 = (215, 48, 39)
    return tuple(int(round((1.0 - t) * a + t * b)) for a, b in zip(c0, c1))


def _draw_score_colored_points(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[float, float]],
    scores: Iterable[float],
    *,
    radius: int = 3,
) -> None:
    for (x, y), score in zip(points, scores):
        color = _score_to_coolwarm_color(score)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=color,
        )


def _draw_crosses(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[float, float]],
    *,
    color: tuple[int, int, int],
    size: int = 5,
    width: int = 2,
) -> None:
    for x, y in points:
        draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
        draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes_xyxy: Iterable[tuple[float, float, float, float]],
    *,
    color: tuple[int, int, int],
    scores: Optional[Iterable[float]] = None,
    width: int = 3,
) -> None:
    score_list = list(scores) if scores is not None else []
    for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if idx < len(score_list):
            label = f"{score_list[idx]:.3f}"
            text_x = max(0.0, x1)
            text_y = max(0.0, y1 - 14.0)
            draw.text((text_x, text_y), label, fill=color)


def _expand_boxes_for_visualization(
    boxes_xyxy: Optional[torch.Tensor],
    *,
    min_side_px: float = 8.0,
) -> Optional[torch.Tensor]:
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return boxes_xyxy

    boxes = boxes_xyxy.detach().clone().to(torch.float32)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    half_w = torch.clamp(widths * 0.5, min=min_side_px * 0.5)
    half_h = torch.clamp(heights * 0.5, min=min_side_px * 0.5)
    boxes[:, 0] = cx - half_w
    boxes[:, 2] = cx + half_w
    boxes[:, 1] = cy - half_h
    boxes[:, 3] = cy + half_h
    return boxes


def _save_image(image: Image.Image, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def _draw_info_panel(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    lines = []
    if gt_count is not None:
        lines.append(f"GT: {int(gt_count)}")
    if pred_count is not None:
        lines.append(f"pred: {int(pred_count)}")
    if class_name is not None and str(class_name).strip():
        lines.append(f"class: {str(class_name).strip()}")

    if not lines:
        return

    padding = 8
    line_gap = 4
    text_bboxes = [draw.textbbox((0, 0), line) for line in lines]
    text_width = max((bbox[2] - bbox[0]) for bbox in text_bboxes)
    text_height = sum((bbox[3] - bbox[1]) for bbox in text_bboxes)
    text_height += line_gap * max(0, len(lines) - 1)

    panel_width = text_width + padding * 2
    panel_height = text_height + padding * 2
    left = max(0, image.width - panel_width - 10)
    top = 10
    right = min(image.width, left + panel_width)
    bottom = min(image.height, top + panel_height)

    draw.rectangle(
        (left, top, right, bottom),
        fill=(255, 255, 255),
        outline=(40, 40, 40),
        width=1,
    )

    y = top + padding
    for line, bbox in zip(lines, text_bboxes):
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        x = right - padding - line_width
        draw.text((x, y), line, fill=(20, 20, 20))
        y += line_height + line_gap


def save_p2p_visualization(
    image: Image.Image,
    p2p_points: Optional[torch.Tensor],
    output_path: str,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    canvas = _to_rgb_image(image)
    draw = ImageDraw.Draw(canvas)
    _draw_points(draw, _to_point_tuples(p2p_points), color=(220, 32, 32), radius=2)
    _draw_info_panel(
        draw,
        canvas,
        gt_count=gt_count,
        pred_count=pred_count,
        class_name=class_name,
    )
    _save_image(canvas, output_path)


def save_sam3_visualization(
    image: Image.Image,
    sam_boxes_xyxy: Optional[torch.Tensor],
    sam_scores: Optional[torch.Tensor],
    output_path: str,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    canvas = _to_rgb_image(image)
    draw = ImageDraw.Draw(canvas)
    _draw_boxes(
        draw,
        _to_box_tuples(sam_boxes_xyxy),
        color=(34, 110, 73),
        scores=_to_score_list(sam_scores),
        width=3,
    )
    _draw_info_panel(
        draw,
        canvas,
        gt_count=gt_count,
        pred_count=pred_count,
        class_name=class_name,
    )
    _save_image(canvas, output_path)


def save_sam3_point_visualization(
    image: Image.Image,
    sam_boxes_xyxy: Optional[torch.Tensor],
    output_path: str,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    canvas = _to_rgb_image(image)
    draw = ImageDraw.Draw(canvas)
    _draw_points(
        draw,
        _box_centers_from_xyxy(sam_boxes_xyxy),
        color=(220, 32, 32),
        radius=2,
    )
    _draw_info_panel(
        draw,
        canvas,
        gt_count=gt_count,
        pred_count=pred_count,
        class_name=class_name,
    )
    _save_image(canvas, output_path)


def save_sam3_point_score_visualization(
    image: Image.Image,
    sam_boxes_xyxy: Optional[torch.Tensor],
    sam_scores: Optional[torch.Tensor],
    output_path: str,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    canvas = _to_rgb_image(image)
    draw = ImageDraw.Draw(canvas)
    _draw_score_colored_points(
        draw,
        _box_centers_from_xyxy(sam_boxes_xyxy),
        _to_score_list(sam_scores),
        radius=3,
    )
    _draw_info_panel(
        draw,
        canvas,
        gt_count=gt_count,
        pred_count=pred_count,
        class_name=class_name,
    )
    _save_image(canvas, output_path)


def save_pseudo_gt_box_visualization(
    image: Image.Image,
    pseudo_boxes_xyxy: Optional[torch.Tensor],
    output_path: str,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    canvas = _to_rgb_image(image)
    draw = ImageDraw.Draw(canvas)
    vis_boxes_xyxy = _expand_boxes_for_visualization(pseudo_boxes_xyxy, min_side_px=8.0)
    _draw_boxes(
        draw,
        _to_box_tuples(vis_boxes_xyxy),
        color=(45, 94, 211),
        scores=None,
        width=3,
    )
    _draw_points(
        draw,
        _box_centers_from_xyxy(pseudo_boxes_xyxy),
        color=(0, 196, 255),
        radius=2,
    )
    _draw_info_panel(
        draw,
        canvas,
        gt_count=gt_count,
        pred_count=pred_count,
        class_name=class_name,
    )
    _save_image(canvas, output_path)


def save_total_visualization(
    image: Image.Image,
    sam_boxes_xyxy: Optional[torch.Tensor],
    sam_scores: Optional[torch.Tensor],
    kept_p2p_points: Optional[torch.Tensor],
    removed_p2p_points: Optional[torch.Tensor],
    output_path: str,
    *,
    gt_count: Optional[int] = None,
    pred_count: Optional[int] = None,
    class_name: Optional[str] = None,
) -> None:
    canvas = _to_rgb_image(image)
    draw = ImageDraw.Draw(canvas)
    _draw_boxes(
        draw,
        _to_box_tuples(sam_boxes_xyxy),
        color=(34, 110, 73),
        scores=_to_score_list(sam_scores),
        width=3,
    )
    _draw_points(
        draw,
        _to_point_tuples(kept_p2p_points),
        color=(220, 32, 32),
        radius=2,
    )
    _draw_crosses(
        draw,
        _to_point_tuples(removed_p2p_points),
        color=(165, 62, 54),
        size=5,
        width=2,
    )
    _draw_info_panel(
        draw,
        canvas,
        gt_count=gt_count,
        pred_count=pred_count,
        class_name=class_name,
    )
    _save_image(canvas, output_path)


# Public CountAnything aliases. Keep the legacy drawing functions above so old
# scripts and configs remain valid while new code can use paper terminology.
save_pdc_visualization = save_p2p_visualization
save_rsc_visualization = save_sam3_visualization
save_rsc_point_visualization = save_sam3_point_visualization
save_rsc_point_score_visualization = save_sam3_point_score_visualization
save_ccf_visualization = save_total_visualization
