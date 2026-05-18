"""Pixel-level dense counter loss and matchers."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from sam3.train.loss.loss_fns import LossWithWeights


def _get_pdc_logits(outputs):
    if "pdc_logits" in outputs:
        return outputs["pdc_logits"]
    raise KeyError("Pixel-level dense counter output missing required key: pdc_logits")


def _get_pdc_points(outputs):
    if "pdc_points" in outputs:
        return outputs["pdc_points"]
    raise KeyError("Pixel-level dense counter output missing required key: pdc_points")


def _get_pdc_image_size(outputs):
    if "input_image_size" in outputs:
        return outputs["input_image_size"]
    raise KeyError(
        "Pixel-level dense counter output missing required key: input_image_size"
    )


class PDCSparseMatcher(torch.nn.Module):
    """Sparse matcher for the pixel-level dense counter."""

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_point: float = 1.0,
        top_s: int = 5,
        nonzero_floor: float = 1e-3,
        repair_k: int = 50,
        print_stats: bool = False,
    ):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_point = float(cost_point)
        self.top_s = int(top_s)
        self.nonzero_floor = float(nonzero_floor)
        self.repair_k = int(repair_k)
        self.print_stats = bool(print_stats)
        self.last_unmatched_gt_before_repair = []
        self.last_unmatched_gt_before_repair_avg = 0.0
        assert self.cost_class != 0 or self.cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits_all = _get_pdc_logits(outputs)
        pred_points_all = _get_pdc_points(outputs)
        bs, num_queries = pred_logits_all.shape[:2]
        indices = []
        unmatched_before_repair_per_batch = []

        for b in range(bs):
            pred_logits = pred_logits_all[b]
            pred_points = pred_points_all[b]
            tgt_labels = targets[b]["labels"]
            tgt_points = targets[b]["point"]

            num_pred = int(num_queries)
            num_gt = int(tgt_points.shape[0])

            if num_pred == 0 or num_gt == 0:
                unmatched_before_repair = num_gt
                if self.print_stats:
                    print(
                        f"[PDCSparseMatcher] batch={b} pred={num_pred} gt={num_gt} "
                        f"unmatched_gt_before_repair={unmatched_before_repair}"
                    )
                unmatched_before_repair_per_batch.append(float(unmatched_before_repair))
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            if num_pred <= num_gt:
                out_prob = pred_logits.softmax(-1)
                cost_class = -out_prob[:, tgt_labels]
                cost_point = torch.cdist(pred_points, tgt_points, p=2)
                cost = (
                    self.cost_point * cost_point + self.cost_class * cost_class
                ).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost)
                unmatched_before_repair = num_gt - len(col_ind)
                if self.print_stats:
                    print(
                        f"[PDCSparseMatcher] batch={b} pred={num_pred} gt={num_gt} "
                        f"unmatched_gt_before_repair={unmatched_before_repair}"
                    )
                unmatched_before_repair_per_batch.append(float(unmatched_before_repair))
                indices.append(
                    (
                        torch.as_tensor(row_ind, dtype=torch.int64),
                        torch.as_tensor(col_ind, dtype=torch.int64),
                    )
                )
                continue

            out_prob_np = pred_logits.softmax(-1).cpu().numpy()
            pred_points_np = pred_points.cpu().numpy()
            tgt_points_np = tgt_points.cpu().numpy()
            tgt_labels_np = tgt_labels.cpu().numpy()

            tree_pred = cKDTree(pred_points_np)
            k0 = min(self.top_s, num_pred)
            d_pred, idx_pred = tree_pred.query(tgt_points_np, k=k0, workers=-1)
            if k0 == 1:
                d_pred = d_pred[:, None]
                idx_pred = idx_pred[:, None]

            rows, cols, data = [], [], []
            for j in range(num_gt):
                sel_i = np.asarray(idx_pred[j], dtype=np.int32)
                sel_d = np.asarray(d_pred[j], dtype=np.float64)
                if sel_i.size == 0:
                    continue
                y = int(tgt_labels_np[j])
                p = out_prob_np[sel_i, y]
                cost = self.cost_point * sel_d + self.cost_class * (-p)
                rows.extend([j] * int(sel_i.size))
                cols.extend(sel_i.tolist())
                data.extend(cost.tolist())

            if len(data) > 0:
                data = np.asarray(data, dtype=np.float64)
                min_cost = float(data.min())
                shift = (
                    (self.nonzero_floor - min_cost)
                    if min_cost <= self.nonzero_floor
                    else 0.0
                )
                data = data + shift
                c_max = float(data.max())
            else:
                data = np.asarray([], dtype=np.float64)
                c_max = 0.0

            c_dummy = c_max + 1.0
            rows = np.asarray(rows, dtype=np.int32)
            cols = np.asarray(cols, dtype=np.int32)
            rows_d = np.arange(num_gt, dtype=np.int32)
            cols_d = (num_pred + rows_d).astype(np.int32)
            data_d = np.full((num_gt,), c_dummy, dtype=np.float64)

            all_rows = np.concatenate([rows, rows_d], axis=0)
            all_cols = np.concatenate([cols, cols_d], axis=0)
            all_data = np.concatenate([data, data_d], axis=0)

            sparse_cost = coo_matrix(
                (all_data, (all_rows, all_cols)), shape=(num_gt, num_pred + num_gt)
            ).tocsr()
            row_ind, col_ind = min_weight_full_bipartite_matching(sparse_cost)
            row_ind = np.asarray(row_ind, dtype=np.int32)
            col_ind = np.asarray(col_ind, dtype=np.int32)
            real_mask = col_ind < num_pred
            tgt_idx = row_ind[real_mask]
            src_idx = col_ind[real_mask]
            unmatched_before_repair = num_gt - real_mask.sum()

            if src_idx.size < num_gt and self.repair_k > 0:
                matched_gt = np.zeros((num_gt,), dtype=bool)
                matched_gt[tgt_idx] = True
                matched_src = np.zeros((num_pred,), dtype=bool)
                matched_src[src_idx] = True
                unmatched_gt = np.where(~matched_gt)[0]
                unmatched_src = np.where(~matched_src)[0]
                if unmatched_gt.size > 0 and unmatched_src.size > 0:
                    repair_src = unmatched_src[: min(self.repair_k, unmatched_src.size)]
                    repair_prob = out_prob_np[repair_src]
                    repair_points = pred_points_np[repair_src]
                    repair_tgt_points = tgt_points_np[unmatched_gt]
                    repair_tgt_labels = tgt_labels_np[unmatched_gt]
                    repair_cost_class = -repair_prob[:, repair_tgt_labels]
                    repair_cost_point = np.linalg.norm(
                        repair_points[:, None, :] - repair_tgt_points[None, :, :],
                        axis=-1,
                    )
                    repair_cost = (
                        self.cost_point * repair_cost_point
                        + self.cost_class * repair_cost_class
                    )
                    repair_row_ind, repair_col_ind = linear_sum_assignment(repair_cost)
                    src_idx = np.concatenate([src_idx, repair_src[repair_row_ind]], axis=0)
                    tgt_idx = np.concatenate(
                        [tgt_idx, unmatched_gt[repair_col_ind]], axis=0
                    )

            if self.print_stats:
                print(
                    f"[PDCSparseMatcher] batch={b} pred={num_pred} gt={num_gt} "
                    f"unmatched_gt_before_repair={unmatched_before_repair}"
                )
            unmatched_before_repair_per_batch.append(float(unmatched_before_repair))

            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64),
                    torch.as_tensor(tgt_idx, dtype=torch.int64),
                )
            )

        self.last_unmatched_gt_before_repair = unmatched_before_repair_per_batch
        self.last_unmatched_gt_before_repair_avg = (
            float(np.mean(unmatched_before_repair_per_batch))
            if unmatched_before_repair_per_batch
            else 0.0
        )
        return indices


class PDCHungarianMatcher(torch.nn.Module):
    """Dense Hungarian matcher for the pixel-level dense counter."""

    def __init__(self, cost_class: float = 1.0, cost_point: float = 1.0):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_point = float(cost_point)
        self.last_unmatched_gt_before_repair = []
        self.last_unmatched_gt_before_repair_avg = 0.0
        assert self.cost_class != 0 or self.cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits_all = _get_pdc_logits(outputs)
        pred_points_all = _get_pdc_points(outputs)
        bs, num_queries = pred_logits_all.shape[:2]
        indices = []
        unmatched_before_repair_per_batch = []

        for b in range(bs):
            pred_logits = pred_logits_all[b]
            pred_points = pred_points_all[b]
            tgt_labels = targets[b]["labels"]
            tgt_points = targets[b]["point"]

            num_pred = int(num_queries)
            num_gt = int(tgt_points.shape[0])

            if num_pred == 0 or num_gt == 0:
                unmatched_before_repair_per_batch.append(float(num_gt))
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            out_prob = pred_logits.softmax(-1)
            cost_class = -out_prob[:, tgt_labels]
            cost_point = torch.cdist(pred_points, tgt_points, p=2)
            cost = self.cost_point * cost_point + self.cost_class * cost_class
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            unmatched_before_repair_per_batch.append(float(num_gt - len(col_ind)))
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )

        self.last_unmatched_gt_before_repair = unmatched_before_repair_per_batch
        self.last_unmatched_gt_before_repair_avg = (
            float(np.mean(unmatched_before_repair_per_batch))
            if unmatched_before_repair_per_batch
            else 0.0
        )
        return indices


class PDCPointSetLoss(LossWithWeights):
    def __init__(
        self,
        weight_dict=None,
        compute_aux=False,
        eos_coef=0.5,
        point_loss_coef=0.0002,
        matcher_cost_class=1.0,
        matcher_cost_point=0.05,
        matcher_top_s=5,
        matcher_nonzero_floor=1e-3,
        matcher_repair_k=50,
    ):
        super().__init__(weight_dict, compute_aux, supports_o2m_loss=False)
        self.eos_coef = eos_coef
        self.point_loss_coef = point_loss_coef
        self.matcher = PDCSparseMatcher(
            cost_class=matcher_cost_class,
            cost_point=matcher_cost_point,
            top_s=matcher_top_s,
            nonzero_floor=matcher_nonzero_floor,
            repair_k=matcher_repair_k,
            print_stats=False,
        )
        empty_weight = torch.ones(2)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.target_keys.append("points")

    def _build_pdc_targets(self, outputs, targets):
        target_list = []
        input_image_size = _get_pdc_image_size(outputs)
        for batch_idx, num_gt in enumerate(targets["num_boxes"].tolist()):
            gt_points = targets["points_padded"][batch_idx, :num_gt]
            image_h, image_w = input_image_size[batch_idx]
            scale = gt_points.new_tensor((image_w, image_h))
            gt_points = gt_points * scale
            target_list.append(
                {
                    "labels": torch.ones(
                        num_gt, dtype=torch.long, device=gt_points.device
                    ),
                    "point": gt_points,
                }
            )
        return target_list

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _loss_labels(self, outputs, pdc_targets, indices):
        src_logits = _get_pdc_logits(outputs)
        if sum(len(src) for src, _ in indices) == 0:
            target_classes = torch.zeros(
                src_logits.shape[:2], dtype=torch.int64, device=src_logits.device
            )
        else:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat(
                [t["labels"][j] for t, (_, j) in zip(pdc_targets, indices)]
            )
            target_classes = torch.full(
                src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight.to(src_logits.device),
        )

        log_probs = F.log_softmax(src_logits, dim=-1)
        pos_mask = target_classes == 1
        neg_mask = target_classes == 0

        loss_ce_pos = (
            (-log_probs[..., 1][pos_mask]).sum()
            if pos_mask.any()
            else src_logits.sum() * 0.0
        )
        loss_ce_neg = (
            (-log_probs[..., 0][neg_mask]).sum()
            if neg_mask.any()
            else src_logits.sum() * 0.0
        )
        return loss_ce, loss_ce_pos, loss_ce_neg

    def _loss_points(self, outputs, pdc_targets, indices, num_boxes):
        pred_points = _get_pdc_points(outputs)
        if sum(len(src) for src, _ in indices) == 0:
            return pred_points.sum() * 0.0
        idx = self._get_src_permutation_idx(indices)
        src_points = pred_points[idx]
        target_points = torch.cat(
            [t["point"][j] for t, (_, j) in zip(pdc_targets, indices)], dim=0
        )
        loss_point = F.mse_loss(src_points, target_points, reduction="none")
        return loss_point.sum() / num_boxes

    def get_loss(self, outputs, targets, indices, num_boxes):
        _ = indices
        assert (
            "pdc_logits" in outputs
            and "pdc_points" in outputs
            and "input_image_size" in outputs
        )
        pdc_targets = self._build_pdc_targets(outputs, targets)
        pdc_indices = self.matcher(outputs, pdc_targets)
        unmatched_before_repair_avg = _get_pdc_logits(outputs).new_tensor(
            float(getattr(self.matcher, "last_unmatched_gt_before_repair_avg", 0.0))
        )
        loss_ce, loss_ce_pos, loss_ce_neg = self._loss_labels(
            outputs, pdc_targets, pdc_indices
        )
        loss_point = self._loss_points(outputs, pdc_targets, pdc_indices, num_boxes)
        return {
            "loss_pdc_cls": loss_ce,
            "loss_pdc_point": loss_point,
            "pdc_cls_pos": loss_ce_pos,
            "pdc_cls_neg": loss_ce_neg,
            "pdc_unmatched_gt_before_repair": unmatched_before_repair_avg,
        }


SparsePointMatcher = PDCSparseMatcher


__all__ = [
    "PDCSparseMatcher",
    "PDCHungarianMatcher",
    "PDCPointSetLoss",
    "SparsePointMatcher",
]
