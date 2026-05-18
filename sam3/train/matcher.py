# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import logging

import numpy as np
import torch

from sam3.model.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch import nn


def _do_matching(cost, repeats=1, return_tgt_indices=False, do_filtering=False):
    if repeats > 1:
        cost = np.tile(cost, (1, repeats))

    i, j = linear_sum_assignment(cost)
    if do_filtering:
        # filter out invalid entries (i.e. those with cost > 1e8)
        valid_thresh = 1e8
        valid_ijs = [(ii, jj) for ii, jj in zip(i, j) if cost[ii, jj] < valid_thresh]
        i, j = zip(*valid_ijs) if len(valid_ijs) > 0 else ([], [])
        i, j = np.array(i, dtype=np.int64), np.array(j, dtype=np.int64)
    if return_tgt_indices:
        return i, j
    order = np.argsort(j)
    return i[order]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid() if focal_loss else nn.Softmax(-1)
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, batched_targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.norm(
            outputs["pred_logits"].flatten(0, 1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = batched_targets["boxes"]

        if "positive_map" in batched_targets:
            # In this case we have a multi-hot target
            positive_map = batched_targets["positive_map"]
            assert len(tgt_bbox) == len(positive_map)

            if self.focal_loss:
                positive_map = positive_map > 1e-4
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg_cost_class = (
                    (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = (
                    (pos_cost_class - neg_cost_class).unsqueeze(1)
                    * positive_map.unsqueeze(0)
                ).sum(-1)
            else:
                # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
                cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(
                    -1
                )
        else:
            # In this case we are doing a "standard" cross entropy
            tgt_ids = batched_targets["labels"]
            assert len(tgt_bbox) == len(tgt_ids)

            if self.focal_loss:
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg_cost_class = (
                    (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be omitted.
                cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        assert cost_class.shape == cost_bbox.shape

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        indices = [_do_matching(c) for c in costs]
        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx


class BinaryHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, batched_targets, repeats=0, repeat_batch=1):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.norm(outputs["pred_logits"].flatten(0, 1)).squeeze(
            -1
        )  # [batch_size * num_queries]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = batched_targets["boxes"]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        return_tgt_indices = False
        for c in costs:
            n_targ = c.shape[1]
            if repeats > 1:
                n_targ *= repeats
            if c.shape[0] < n_targ:
                return_tgt_indices = True
                break
        if return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c, repeats=repeats, return_tgt_indices=return_tgt_indices
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long()
        else:
            indices = [_do_matching(c, repeats=repeats) for c in costs]
            tgt_idx = None

        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx, tgt_idx


class DensePointHungarianMatcher(nn.Module):
    """
    使用匈牙利算法（Hungarian matching）对 SAM3 分支的预测与 GT 点进行一对一匹配。

    这个版本的 matcher 适用于“计数 / 点监督”场景：
    - 模型输出的是一组 query 预测（通常来自检测分支）
    - 每个 query 有一个前景置信度 pred_logits
    - 每个 query 还有一个 box 预测 pred_boxes
    - 这里不直接把 box 当监督目标，而是取 pred_boxes[..., :2] 作为“点坐标”
      来与 GT 点进行匹配

    与原始 P2PNet 的 crowd matcher 相比，这个版本更偏工程化，主要体现在：
    1. 支持 GT padding
    2. 支持 query 有效位过滤（out_is_valid）
    3. 支持 target 有效位过滤（target_is_valid_padded）
    4. 支持 batch 内某些样本 GT 数为 0
    5. 输出是 batched flatten 风格索引，更方便后续统一 loss gather

    参数:
        cost_class:
            分类代价的权重。分类代价这里定义为 -sigmoid(logit)。
            值越大，matcher 越偏向选择“前景分数高”的 query。
            若设为 0，则匹配仅由点距离决定。

        cost_point:
            点坐标距离代价的权重。通常是最主要的项。
            值越大，matcher 越偏向选择几何位置更接近 GT 点的 query。

        remove_samples_with_0_gt:
            是否在匹配前移除 GT 数为 0 的样本。
            True 时，这些样本不会参与 Hungarian matching。
            这样做可以避免空代价矩阵带来的额外处理复杂度。
    """

    def __init__(
        self,
        cost_class: float = 0.0,
        cost_point: float = 1.0,
        remove_samples_with_0_gt: bool = True,
    ):
        super().__init__()

        # 分类代价权重
        self.cost_class = cost_class

        # 点距离代价权重
        self.cost_point = cost_point

        # 将单通道 logit 转成 [0, 1] 前景概率
        # 这里默认 pred_logits 是单通道“前景置信度”
        self.norm = nn.Sigmoid()

        # 是否去掉 GT 数为 0 的样本
        self.remove_samples_with_0_gt = remove_samples_with_0_gt

        # 至少要有一种代价参与匹配，否则矩阵没有意义
        assert (
            cost_class != 0 or cost_point != 0
        ), "all costs cant be 0"

        self._debug_context = {"epoch": None, "phase": None, "step": None}
        self.reset_debug_stats()

    def set_debug_context(self, epoch=None, phase=None, step=None):
        self._debug_context = {"epoch": epoch, "phase": phase, "step": step}

    def reset_debug_stats(self):
        self._debug_stats = {
            "calls": 0,
            "nonfinite_calls": 0,
            "last_nonfinite": None,
        }

    def get_debug_stats(self):
        return {
            "calls": self._debug_stats["calls"],
            "nonfinite_calls": self._debug_stats["nonfinite_calls"],
            "last_nonfinite": self._debug_stats["last_nonfinite"],
        }

    @staticmethod
    def _summarize_tensor(tensor):
        finite_mask = torch.isfinite(tensor)
        bad_count = int((~finite_mask).sum().item())
        if int(finite_mask.sum().item()) == 0:
            return {
                "shape": tuple(tensor.shape),
                "bad": bad_count,
                "min": None,
                "max": None,
                "mean": None,
            }
        finite_tensor = tensor[finite_mask]
        return {
            "shape": tuple(tensor.shape),
            "bad": bad_count,
            "min": float(finite_tensor.min().item()),
            "max": float(finite_tensor.max().item()),
            "mean": float(finite_tensor.mean().item()),
        }

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        """
        执行 batch 级别的 Hungarian matching。

        输入:
            outputs:
                模型输出字典，至少包含：
                - outputs["pred_logits"]: shape = (B, Q, 1)
                    每个 query 的前景 logit
                - outputs["pred_boxes"]: shape = (B, Q, 4)
                    每个 query 的 box 预测
                    这里默认前两维 [:, :, :2] 表示点坐标（例如 box center）

            batched_targets:
                GT 字典，至少包含：
                - batched_targets["points_padded"]: shape = (B, T_max, 2)
                    padding 后的 GT 点坐标
                - batched_targets["num_boxes"]: shape = (B,)
                    每张图真实 GT 点数
                    虽然名字叫 num_boxes，但在这里实际上代表 GT 点数

            repeats:
                兼容外部接口保留的参数。
                当前实现只支持 repeats=1。

            repeat_batch:
                若 >1，则把 target 端和必要的 batch 维做重复。
                这通常用于某些“同一批目标配合重复输出”的训练结构。

            out_is_valid:
                shape = (B, Q) 的 bool mask。
                True 表示该 query 可参与匹配，False 表示该 query 无效。
                无效 query 在代价矩阵中会被赋极大值，从而不会被选中。

            target_is_valid_padded:
                shape = (B, T_max) 的 bool mask。
                True 表示该 GT 位置有效，False 表示这是 padding 位置。
                padding 位置在代价矩阵中会被赋极大值。

        返回:
            batch_idx:
                shape = (N_match,)
                每个匹配对对应的 batch 下标

            src_idx:
                shape = (N_match,)
                每个匹配对对应的预测 query 下标

            tgt_idx:
                shape = (N_match,) 或 None
                每个匹配对对应的“flatten 后 GT 索引”
                当不需要显式返回 target 索引时，可能为 None

        说明:
            返回形式不是传统的 [(src_i, tgt_i), ...] 按图列表，
            而是更方便后续 batched gather 的扁平索引形式。
        """
        # 当前实现不支持 repeats > 1
        # 这里直接 assert，避免外部误用
        assert repeats == 1, "DensePointHungarianMatcher does not support repeats > 1"
        self._debug_stats["calls"] += 1

        # outputs["pred_logits"] 的形状是 (B, Q, 1)
        # 这里只取前两个维度，B = batch size, Q = 每张图 query 数
        _, num_queries = outputs["pred_logits"].shape[:2]

        # 将 pred_logits 从 (B, Q, 1) 压成 (B, Q)
        # 每个元素是该 query 的前景 logit
        out_score = outputs["pred_logits"].squeeze(-1)  # (B, Q)

        # 经过 sigmoid 得到前景概率，shape 仍然是 (B, Q)
        out_prob = self.norm(out_score)

        # 从预测框中取前两维，作为点坐标。
        # 当前 counting 训练里 pred_boxes 和 GT points 都是归一化到 [0, 1] 的。
        # 为了让 cost_point 保持更接近像素空间的语义，这里优先把双方都
        # 反归一化到 resize 后输入图上的像素坐标，再计算欧氏距离。
        # shape = (B, Q, 2)
        out_points = outputs["pred_boxes"][..., :2]

        # 记录 device，方便最后把 numpy / cpu 上的结果转回原设备
        device = out_score.device

        # 每张图 GT 数量，shape = (B,)
        # 转到 cpu 是因为后面很多地方（例如 list / numpy）更方便处理
        num_boxes = batched_targets["num_boxes"].cpu()

        # padding 后的 GT 点，shape = (B, T_max, 2)
        tgt_points = batched_targets["points_padded"]

        # ------------------------------------------------------------
        # Step 0. 若可获得 resize 后输入尺寸，则先把归一化点坐标还原到像素空间
        # ------------------------------------------------------------
        image_size = outputs.get("sam_image_size", None)
        if image_size is None:
            image_size = outputs.get("p2p_image_size", None)
        if image_size is not None:
            if image_size.dim() == 1:
                image_size = image_size.unsqueeze(0).expand(out_points.shape[0], -1)
            # image_size stores (H, W), while point coordinates are (x, y)
            point_scale = image_size[:, [1, 0]].to(
                device=out_points.device, dtype=out_points.dtype
            )
            out_points = out_points * point_scale[:, None, :]
            tgt_points = tgt_points * point_scale[:, None, :]

        # ------------------------------------------------------------
        # Step 1. 如果需要，先移除 GT 数为 0 的样本
        # ------------------------------------------------------------
        if self.remove_samples_with_0_gt:
            # batch_keep: shape = (B,)
            # True 表示该样本至少有一个 GT，可以参与匹配
            batch_keep = num_boxes > 0

            # 只保留有 GT 的样本
            num_boxes = num_boxes[batch_keep]
            tgt_points = tgt_points[batch_keep]

            # 若提供了 target valid mask，也同步裁剪
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded[batch_keep]

        # ------------------------------------------------------------
        # Step 2. 若 repeat_batch > 1，则把 target 端重复
        # ------------------------------------------------------------
        if repeat_batch > 1:
            # 注意：这里是沿 batch 维重复
            num_boxes = num_boxes.repeat(repeat_batch)
            tgt_points = tgt_points.repeat(repeat_batch, 1, 1)
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded.repeat(repeat_batch, 1)

        # ------------------------------------------------------------
        # Step 3. 与 target 对齐地处理 output 端
        # ------------------------------------------------------------
        if self.remove_samples_with_0_gt:
            # 若 target 端做了 repeat，那么 batch_keep 也要同步 repeat
            if repeat_batch > 1:
                batch_keep = batch_keep.repeat(repeat_batch)

            # 只保留对应“有 GT 样本”的输出
            out_prob = out_prob[batch_keep]
            out_points = out_points[batch_keep]

            # 输出有效位也同步裁剪
            if out_is_valid is not None:
                out_is_valid = out_is_valid[batch_keep]

        # 到这里，output 和 target 的 batch 维必须对齐
        assert out_points.shape[0] == tgt_points.shape[0]
        assert out_points.shape[0] == num_boxes.shape[0]

        # ------------------------------------------------------------
        # Step 4. 构造代价矩阵 cost matrix
        # ------------------------------------------------------------
        # cost_point:
        # 对每张图，计算所有 query 点 与 所有 GT 点之间的欧氏距离
        # shape = (B_eff, Q, T_max)
        # 其中 B_eff 是过滤 / repeat 后的有效 batch 大小
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        # cost_class:
        # 将每个 query 的前景概率复制到所有 target 上
        # 这样每个 query 对任何 target 的分类代价都一样
        #
        # 为什么这样定义？
        # 因为这里没有多类别 label 的区别，只有“这个 query 是否像前景目标”这一件事。
        # 所以分类项只是在鼓励 matcher 选前景分数更高的 query。
        #
        # 代价取负号：概率越高，代价越小，越容易被匹配
        # shape = (B_eff, Q, T_max)
        cost_class = -out_prob.unsqueeze(-1).expand_as(cost_point)

        # 总代价 = 点距离代价 + 分类代价
        # 若 cost_class = 0，则退化为纯几何 Hungarian matching
        C = self.cost_point * cost_point + self.cost_class * cost_class

        score_bad = int((~torch.isfinite(out_score)).sum().item())
        prob_bad = int((~torch.isfinite(out_prob)).sum().item())
        point_bad = int((~torch.isfinite(out_points)).sum().item())
        pred_boxes_bad = int((~torch.isfinite(outputs["pred_boxes"])).sum().item())
        cost_point_bad = int((~torch.isfinite(cost_point)).sum().item())
        cost_class_bad = int((~torch.isfinite(cost_class)).sum().item())
        cost_bad = int((~torch.isfinite(C)).sum().item())
        has_nonfinite = (
            score_bad
            or prob_bad
            or point_bad
            or pred_boxes_bad
            or cost_point_bad
            or cost_class_bad
            or cost_bad
        )
        if has_nonfinite:
            self._debug_stats["nonfinite_calls"] += 1
            first_bad_indices = {
                "score": (~torch.isfinite(out_score)).nonzero(as_tuple=False)[:8].tolist(),
                "prob": (~torch.isfinite(out_prob)).nonzero(as_tuple=False)[:8].tolist(),
                "point": (~torch.isfinite(out_points)).nonzero(as_tuple=False)[:8].tolist(),
                "pred_boxes": (~torch.isfinite(outputs["pred_boxes"]))
                .nonzero(as_tuple=False)[:8]
                .tolist(),
                "cost": (~torch.isfinite(C)).nonzero(as_tuple=False)[:8].tolist(),
            }
            payload = {
                "epoch": self._debug_context["epoch"],
                "phase": self._debug_context["phase"],
                "step": self._debug_context["step"],
                "score_bad": score_bad,
                "prob_bad": prob_bad,
                "point_bad": point_bad,
                "pred_boxes_bad": pred_boxes_bad,
                "cost_point_bad": cost_point_bad,
                "cost_class_bad": cost_class_bad,
                "cost_bad": cost_bad,
                "score_stats": self._summarize_tensor(out_score),
                "prob_stats": self._summarize_tensor(out_prob),
                "point_stats": self._summarize_tensor(out_points),
                "pred_boxes_stats": self._summarize_tensor(outputs["pred_boxes"]),
                "cost_point_stats": self._summarize_tensor(cost_point),
                "cost_class_stats": self._summarize_tensor(cost_class),
                "cost_stats": self._summarize_tensor(C),
                "tgt_points_stats": self._summarize_tensor(tgt_points),
                "first_bad_indices": first_bad_indices,
            }
            self._debug_stats["last_nonfinite"] = payload
            logging.error(
                "DensePointHungarianMatcher non-finite detected | "
                "epoch=%s phase=%s step=%s score_bad=%d prob_bad=%d point_bad=%d "
                "pred_boxes_bad=%d cost_point_bad=%d cost_class_bad=%d cost_bad=%d",
                payload["epoch"],
                payload["phase"],
                payload["step"],
                score_bad,
                prob_bad,
                point_bad,
                pred_boxes_bad,
                cost_point_bad,
                cost_class_bad,
                cost_bad,
            )
            logging.error(
                "DensePointHungarianMatcher stats | "
                "score=%s prob=%s point=%s pred_boxes=%s cost_point=%s cost_class=%s cost=%s tgt_points=%s",
                payload["score_stats"],
                payload["prob_stats"],
                payload["point_stats"],
                payload["pred_boxes_stats"],
                payload["cost_point_stats"],
                payload["cost_class_stats"],
                payload["cost_stats"],
                payload["tgt_points_stats"],
            )
            logging.error(
                "DensePointHungarianMatcher first bad indices | %s",
                payload["first_bad_indices"],
            )

        # ------------------------------------------------------------
        # Step 5. 根据有效位 mask 对代价矩阵做屏蔽
        # ------------------------------------------------------------
        # do_filtering 表示本次匹配是否涉及有效位过滤
        do_filtering = out_is_valid is not None or target_is_valid_padded is not None

        # 若某个 query 无效，则它对所有 target 的代价都置为极大值
        # shape broadcasting:
        # out_is_valid[:, :, None] : (B_eff, Q, 1)
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, 1e9)

        # 若某个 target 是 padding / 无效位置，则所有 query 到它的代价都置为极大值
        # shape broadcasting:
        # target_is_valid_padded[:, None, :] : (B_eff, 1, T_max)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, 1e9)

        # Hungarian 的具体实现通常在 CPU / numpy 上更方便
        C = C.cpu().numpy()

        # ------------------------------------------------------------
        # Step 6. 对每张图裁掉 padding 后的多余 target 列
        # ------------------------------------------------------------
        # 虽然 tgt_points 是 padding 到 T_max 的，
        # 但真实 GT 数由 num_boxes 指定，所以这里只保留每张图前 s 个有效 GT 列
        #
        # 最终 costs 是长度为 B_eff 的 list，
        # 其中每个元素 shape = (Q, N_gt_i)
        costs = [C[i, :, :s] for i, s in enumerate(num_boxes.tolist())]

        # ------------------------------------------------------------
        # Step 7. 决定是否需要显式返回 tgt_idx
        # ------------------------------------------------------------
        # 一般来说，当满足以下情况时，显式知道“匹配到了哪些 GT”会更重要：
        # 1. 做了 filtering，可能不是所有 GT 都可匹配
        # 2. num_queries < num_boxes，query 数比 GT 少，无法 full cover GT
        #
        # 注意：
        # 这里 torch.any(num_queries < num_boxes).item() 的语义是：
        # 只要 batch 内有任意样本出现 Q < N_gt，就返回 True
        return_tgt_indices = (
            do_filtering or torch.any(num_queries < num_boxes).item()
        )

        # ------------------------------------------------------------
        # Step 8. 对每张图独立做 Hungarian matching
        # ------------------------------------------------------------
        if len(costs) == 0:
            # 整个 batch 都没有可匹配样本
            indices = []

            # 如果逻辑上需要返回 tgt_idx，则返回空 tensor
            tgt_idx = torch.zeros(0).long().to(device) if return_tgt_indices else None
        elif return_tgt_indices:
            # _do_matching(...) 预期返回:
            #   src_indices, tgt_indices
            # 其中二者都是 numpy 数组 / 可拼接对象
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c,
                        repeats=repeats,
                        return_tgt_indices=True,
                        do_filtering=do_filtering,
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)

            # 下面这一步很关键：
            # 每张图内部的 tgt_indices 都是“局部索引” [0, N_gt_i)
            # 这里把它们改成“flatten 后的全局索引”
            #
            # 例如：
            #   第 0 张图有 3 个 GT
            #   第 1 张图有 5 个 GT
            # 那么第 1 张图局部索引 0..4 需要整体偏移 +3
            sizes = torch.cumsum(num_boxes, -1)[:-1]
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()

            # 拼成一个总的一维 tensor
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long().to(device)
        else:
            # 不需要显式 target 索引时，只取每张图匹配到的 src indices
            indices = [
                _do_matching(c, repeats=repeats, do_filtering=do_filtering)
                for c in costs
            ]
            tgt_idx = None

        # ------------------------------------------------------------
        # Step 9. 构造 batch_idx
        # ------------------------------------------------------------
        # batch_idx 的作用是：
        # 配合 src_idx 一起指出“某个匹配来自哪张图、哪个 query”
        #
        # 例如 batch_idx = [0,0,0,1,1], src_idx = [5,7,9,2,8]
        # 表示：
        #   第0张图的 query 5,7,9 被匹配
        #   第1张图的 query 2,8 被匹配
        if self.remove_samples_with_0_gt:
            # kept_inds 是“原始 batch 下标”中哪些图被保留下来了
            kept_inds = batch_keep.nonzero().squeeze(1)

            # 对于每张保留图 i，如果它有 len(src) 个匹配，
            # 就把对应原始 batch 下标重复 len(src) 次
            batch_idx = torch.as_tensor(
                sum([[kept_inds[i]] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )
        else:
            # 没做 0-GT 过滤时，直接按当前 batch 顺序构造
            batch_idx = torch.as_tensor(
                sum([[i] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )

        if len(indices) > 0:
            # 将每张图的 src indices 拼接成一个总的一维 tensor
            src_idx = torch.from_numpy(np.concatenate(indices)).long().to(device)
        else:
            # 没有任何匹配时返回空 tensor
            src_idx = torch.empty(0, dtype=torch.long, device=device)

        # 最终返回 batched flatten 风格索引
        return batch_idx, src_idx, tgt_idx


class BinaryFocalHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.stable = stable
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, batched_targets, repeats=1, repeat_batch=1):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_score = outputs["pred_logits"].flatten(0, 1).squeeze(-1)
        out_prob = self.norm(out_score)  # [batch_size * num_queries]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = batched_targets["boxes"]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)
        if self.stable:
            rescaled_giou = (-cost_giou + 1) / 2
            out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
            cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                out_prob
            ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
        else:
            # directly computing log sigmoid (more numerically stable)
            log_out_prob = torch.nn.functional.logsigmoid(out_score)
            log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
            cost_class = (
                -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
            )
        if not self.stable:
            cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        return_tgt_indices = False
        for c in costs:
            n_targ = c.shape[1]
            if repeats > 1:
                n_targ *= repeats
            if c.shape[0] < n_targ:
                return_tgt_indices = True
                break
        if return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c, repeats=repeats, return_tgt_indices=return_tgt_indices
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long()
        else:
            indices = [_do_matching(c, repeats=repeats) for c in costs]
            tgt_idx = None

        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx, tgt_idx


class BinaryHungarianMatcherV2(nn.Module):
    """
    This class computes an assignment between the targets and the predictions
    of the network

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    This is a more efficient implementation of BinaryHungarianMatcher.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
        remove_samples_with_0_gt: bool = True,
    ):
        """
        Creates the matcher

        Params:
        - cost_class: Relative weight of the classification error in the
          matching cost
        - cost_bbox: Relative weight of the L1 error of the bounding box
          coordinates in the matching cost
        - cost_giou: This is the relative weight of the giou loss of the
          bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"
        self.focal = focal
        if focal:
            self.alpha = alpha
            self.gamma = gamma
            self.stable = stable
        self.remove_samples_with_0_gt = remove_samples_with_0_gt

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        """
        Performs the matching. The inputs and outputs are the same as
        BinaryHungarianMatcher.forward, except for the optional cached_padded
        flag and the optional "_boxes_padded" entry of batched_targets.

        Inputs:
        - outputs: A dict with the following keys:
            - "pred_logits": Tensor of shape (batch_size, num_queries, 1) with
               classification logits
            - "pred_boxes": Tensor of shape (batch_size, num_queries, 4) with
               predicted box coordinates in cxcywh format.
        - batched_targets: A dict of targets. There may be a variable number of
          targets per batch entry; suppose that there are T_b targets for batch
          entry 0 <= b < batch_size. It should have the following keys:
          - "boxes": Tensor of shape (sum_b T_b, 4) giving ground-truth boxes
             in cxcywh format for all batch entries packed into a single tensor
          - "num_boxes": int64 Tensor of shape (batch_size,) giving the number
             of ground-truth boxes per batch entry: num_boxes[b] = T_b
          - "_boxes_padded": Tensor of shape (batch_size, max_b T_b, 4) giving
            a padded version of ground-truth boxes. If this is not present then
            it will be computed from batched_targets["boxes"] instead, but
            caching it here can improve performance for repeated calls with the
            same targets.
        - out_is_valid: If not None, it should be a boolean tensor of shape
          (batch_size, num_queries) indicating which predictions are valid.
          Invalid predictions are ignored during matching and won't appear in
          the output indices.
        - target_is_valid_padded: If not None, it should be a boolean tensor of
          shape (batch_size, max_num_gt_boxes) in padded format indicating
          which GT boxes are valid. Invalid GT boxes are ignored during matching
          and won't appear in the output indices.

        Returns:
            A list of size batch_size, containing tuples of (idx_i, idx_j):
                - idx_i is the indices of the selected predictions (in order)
                - idx_j is the indices of the corresponding selected targets
                  (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j)
                             = min(num_queries, num_target_boxes)
        """
        _, num_queries = outputs["pred_logits"].shape[:2]

        out_score = outputs["pred_logits"].squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))

        device = out_score.device

        num_boxes = batched_targets["num_boxes"].cpu()
        # Get a padded version of target boxes (as precomputed in the collator).
        # It should work for both repeat==1 (o2o) and repeat>1 (o2m) matching.
        tgt_bbox = batched_targets["boxes_padded"]
        if self.remove_samples_with_0_gt:
            # keep only samples w/ at least 1 GT box in targets (num_boxes and tgt_bbox)
            batch_keep = num_boxes > 0
            num_boxes = num_boxes[batch_keep]
            tgt_bbox = tgt_bbox[batch_keep]
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded[batch_keep]
        # Repeat the targets (for the case of batched aux outputs in the matcher)
        if repeat_batch > 1:
            # In this case, out_prob and out_bbox will be a concatenation of
            # both final and auxiliary outputs, so we also repeat the targets
            num_boxes = num_boxes.repeat(repeat_batch)
            tgt_bbox = tgt_bbox.repeat(repeat_batch, 1, 1)
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded.repeat(repeat_batch, 1)

        # keep only samples w/ at least 1 GT box in outputs
        if self.remove_samples_with_0_gt:
            if repeat_batch > 1:
                batch_keep = batch_keep.repeat(repeat_batch)
            out_score = out_score[batch_keep]
            out_bbox = out_bbox[batch_keep]
            if out_is_valid is not None:
                out_is_valid = out_is_valid[batch_keep]
        assert out_bbox.shape[0] == tgt_bbox.shape[0]
        assert out_bbox.shape[0] == num_boxes.shape[0]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        out_prob = self.norm(out_score)
        if not self.focal:
            cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)
        else:
            if self.stable:
                rescaled_giou = (-cost_giou + 1) / 2
                out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
                cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                    out_prob
                ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
            else:
                # directly computing log sigmoid (more numerically stable)
                log_out_prob = torch.nn.functional.logsigmoid(out_score)
                log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
                cost_class = (
                    -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                    + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
                )
            if not self.stable:
                cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        # assign a very high cost (1e9) to invalid outputs and targets, so that we can
        # filter them out (in `_do_matching`) from bipartite matching results
        do_filtering = out_is_valid is not None or target_is_valid_padded is not None
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, 1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, 1e9)
        C = C.cpu().numpy()
        costs = [C[i, :, :s] for i, s in enumerate(num_boxes.tolist())]
        return_tgt_indices = (
            do_filtering or torch.any(num_queries < num_boxes * max(repeats, 1)).item()
        )
        if len(costs) == 0:
            # We have size 0 in the batch dimension, so we return empty matching indices
            # (note that this can happen due to `remove_samples_with_0_gt=True` even if
            # the original input batch size is not 0, when all queries have empty GTs).
            indices = []
            tgt_idx = torch.zeros(0).long().to(device) if return_tgt_indices else None
        elif return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c,
                        repeats=repeats,
                        return_tgt_indices=return_tgt_indices,
                        do_filtering=do_filtering,
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            sizes = torch.cumsum(num_boxes, -1)[:-1]
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long().to(device)
        else:
            indices = [
                _do_matching(c, repeats=repeats, do_filtering=do_filtering)
                for c in costs
            ]
            tgt_idx = None

        if self.remove_samples_with_0_gt:
            kept_inds = batch_keep.nonzero().squeeze(1)
            batch_idx = torch.as_tensor(
                sum([[kept_inds[i]] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )
        else:
            batch_idx = torch.as_tensor(
                sum([[i] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )

        # indices could be an empty list (since we remove samples w/ 0 GT boxes)
        if len(indices) > 0:
            src_idx = torch.from_numpy(np.concatenate(indices)).long().to(device)
        else:
            src_idx = torch.empty(0, dtype=torch.long, device=device)
        return batch_idx, src_idx, tgt_idx


class BinaryOneToManyMatcher(nn.Module):
    """
    This class computes a greedy assignment between the targets and the predictions of the network.
    In this formulation, several predictions can be assigned to each target, but each prediction can be assigned to
    at most one target.

    See DAC-Detr for details
    """

    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 0.4,
        topk: int = 6,
    ):
        """
        Creates the matcher

        Params:
                alpha: relative balancing between classification and localization
                threshold: threshold used to select positive predictions
                topk: number of top scoring predictions to consider
        """
        super().__init__()
        self.norm = nn.Sigmoid()
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        """
        Performs the matching. The inputs and outputs are the same as
        BinaryHungarianMatcher.forward

        Inputs:
        - outputs: A dict with the following keys:
            - "pred_logits": Tensor of shape (batch_size, num_queries, 1) with
               classification logits
            - "pred_boxes": Tensor of shape (batch_size, num_queries, 4) with
               predicted box coordinates in cxcywh format.
        - batched_targets: A dict of targets. There may be a variable number of
          targets per batch entry; suppose that there are T_b targets for batch
          entry 0 <= b < batch_size. It should have the following keys:
          - "num_boxes": int64 Tensor of shape (batch_size,) giving the number
             of ground-truth boxes per batch entry: num_boxes[b] = T_b
          - "_boxes_padded": Tensor of shape (batch_size, max_b T_b, 4) giving
            a padded version of ground-truth boxes. If this is not present then
            it will be computed from batched_targets["boxes"] instead, but
            caching it here can improve performance for repeated calls with the
            same targets.
        - out_is_valid: If not None, it should be a boolean tensor of shape
          (batch_size, num_queries) indicating which predictions are valid.
          Invalid predictions are ignored during matching and won't appear in
          the output indices.
        - target_is_valid_padded: If not None, it should be a boolean tensor of
          shape (batch_size, max_num_gt_boxes) in padded format indicating
          which GT boxes are valid. Invalid GT boxes are ignored during matching
          and won't appear in the output indices.
        Returns:
            A list of size batch_size, containing tuples of (idx_i, idx_j):
                - idx_i is the indices of the selected predictions (in order)
                - idx_j is the indices of the corresponding selected targets
                  (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j)
                             = min(num_queries, num_target_boxes)
        """
        assert repeats <= 1 and repeat_batch <= 1
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = self.norm(outputs["pred_logits"]).squeeze(-1)  # (B, Q)
        out_bbox = outputs["pred_boxes"]  # (B, Q, 4))

        num_boxes = batched_targets["num_boxes"]

        # Get a padded version of target boxes (as precomputed in the collator).
        tgt_bbox = batched_targets["boxes_padded"]
        assert len(tgt_bbox) == bs
        num_targets = tgt_bbox.shape[1]
        if num_targets == 0:
            return (
                torch.empty(0, dtype=torch.long, device=out_prob.device),
                torch.empty(0, dtype=torch.long, device=out_prob.device),
                torch.empty(0, dtype=torch.long, device=out_prob.device),
            )

        iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        assert iou.shape == (bs, num_queries, num_targets)

        # Final cost matrix (higher is better in `C`; this is unlike the case
        # of BinaryHungarianMatcherV2 above where lower is better in its `C`)
        C = self.alpha * out_prob.unsqueeze(-1) + (1 - self.alpha) * iou
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, -1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, -1e9)

        # Selecting topk predictions
        matches = C > torch.quantile(
            C, 1 - self.topk / num_queries, dim=1, keepdim=True
        )

        # Selecting predictions above threshold
        matches = matches & (C > self.threshold)
        if out_is_valid is not None:
            matches = matches & out_is_valid[:, :, None]
        if target_is_valid_padded is not None:
            matches = matches & target_is_valid_padded[:, None, :]

        # Removing padding
        matches = matches & (
            torch.arange(0, num_targets, device=num_boxes.device)[None]
            < num_boxes[:, None]
        ).unsqueeze(1)

        batch_idx, src_idx, tgt_idx = torch.nonzero(matches, as_tuple=True)

        cum_num_boxes = torch.cat(
            [
                torch.zeros(1, dtype=num_boxes.dtype, device=num_boxes.device),
                num_boxes.cumsum(-1)[:-1],
            ]
        )
        tgt_idx += cum_num_boxes[batch_idx]

        return batch_idx, src_idx, tgt_idx


# Public CountAnything matcher aliases. The underlying implementation is kept
# under its legacy name for checkpoint/config compatibility during migration.
RSCMatcher = DensePointHungarianMatcher
RegionSparseMatcher = DensePointHungarianMatcher
