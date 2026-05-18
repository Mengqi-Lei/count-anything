# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

from sam3.model.model_misc import SAM3Output

from sam3.train.utils.distributed import get_world_size

from .loss_fns import CORE_LOSS_KEY, Det2TrkAssoc, Masks


class DummyLoss(torch.nn.Module):
    """A dummy loss that always returns 0 (as a placeholder for eval)"""

    def __init__(
        self,
        core_loss_key: str = CORE_LOSS_KEY,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.core_loss_key = core_loss_key
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        return {self.core_loss_key: torch.tensor(0.0, device=self.device)}

    def accumulate(self, out_dict):
        """
        Called by iterative losses.
        """
        if self.core_loss_key not in out_dict:
            out_dict[self.core_loss_key] = torch.tensor(0.0, device=self.device)
        return out_dict


class Sam3LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_fns_find,
        normalization="global",
        matcher=None,
        o2m_matcher=None,
        o2m_weight=1.0,
        use_o2m_matcher_on_o2m_aux=True,
        loss_fn_semantic_seg=None,
        normalize_by_valid_object_num=False,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
    ):
        super().__init__()
        self.loss_fns_find = loss_fns_find
        assert normalization in ["global", "local", "none"]
        self.normalization = normalization
        self.normalize_by_valid_object_num = normalize_by_valid_object_num
        self.normalize_by_stage_num = normalize_by_stage_num
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        # whether to use the o2m matcher on the o2m queries in auxiliary outputs
        self.use_o2m_matcher_on_o2m_aux = use_o2m_matcher_on_o2m_aux
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.scale_by_find_batch_size = scale_by_find_batch_size

    def _get_num_boxes(self, targets):
        # the average number of target boxes for loss normalization
        if self.normalize_by_valid_object_num:
            # valid boxes are those with non-zero height and width
            # (while padded invisible boxes are )
            boxes_hw = targets["boxes"].view(-1, 4)  # cx, cy, w, h
            num_boxes = (boxes_hw[:, 2:] > 0).all(dim=-1).sum().float()
        else:
            num_boxes = targets["num_boxes"].sum().float()
        if self.normalization == "global":
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
        elif self.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        elif self.normalization == "none":
            num_boxes = 1
        return num_boxes

    def compute_loss(self, nested_out, targets):
        num_boxes = self._get_num_boxes(targets)
        o2m_out_is_valid = nested_out.get("o2m_out_is_valid", None)
        o2m_target_is_valid_padded = nested_out.get("o2m_target_is_valid_padded", None)

        def _as_detached_scalar_tensor(value):
            # 某些分支（例如 compute_aux=False 的占位返回）会给出 Python float 0.0。
            # 这里统一转成 tensor，便于后面做日志拆分统计。
            if torch.is_tensor(value):
                return value.detach()
            return num_boxes.new_tensor(float(value))

        # Get a list of outputs, including auxiliary and first stage outputs
        output_list = [(nested_out, "", False)]  # (out, suffix, is_aux)
        if "aux_outputs" in nested_out:
            output_list.extend(
                (aux_out, f"_aux_{i}", True)
                for i, aux_out in enumerate(nested_out["aux_outputs"])
            )
        if "first_stage" in nested_out:
            output_list.append((nested_out["first_stage"], "_fs", True))

        # Compute all the requested losses
        losses = {}
        total_core_loss = 0.0
        # 额外记录：把最终总损失拆成“主输出加权损失”和“aux 加权损失”两部分，
        # 方便在日志里解释 all_loss 的来源。
        total_main_weighted_loss = 0.0
        total_aux_weighted_loss = 0.0
        for out, suffix, is_aux in output_list:
            # o2o matcher indices need to be computed by the model (as the video model requires
            # a specific way of matching free and locked indices beyond just calling the matcher)
            indices = out["indices"]
            has_o2m_out = "pred_logits_o2m" in out
            o2m_indices = None
            if has_o2m_out:
                o2m_out = {
                    k[: -len("_o2m")]: v for k, v in out.items() if k.endswith("_o2m")
                }
                # o2m targets are the same as the o2o targets (assuming repeat=1)
                o2m_targets = targets
                ## 新添代码
                # 只有当当前 output list 里至少有一个 loss 真的支持 o2m 时，才去构建 o2m matching。
                # 这样 separate supervision 下的独立分支（例如 Stage 6 的 P2P loss）就不会被迫依赖 o2m matcher。
                needs_o2m_matching = any(
                    getattr(loss_fn, "supports_o2m_loss", True)
                    for loss_fn in self.loss_fns_find
                )
                if needs_o2m_matching:
                    if self.use_o2m_matcher_on_o2m_aux or not is_aux:
                        o2m_indices = self.o2m_matcher(
                            o2m_out,
                            o2m_targets,
                            out_is_valid=o2m_out_is_valid,
                            target_is_valid_padded=o2m_target_is_valid_padded,
                        )
                    else:
                        o2m_indices = self.matcher(
                            o2m_out,
                            o2m_targets,
                            out_is_valid=o2m_out_is_valid,
                            target_is_valid_padded=o2m_target_is_valid_padded,
                        )
                ## 新添代码

            for loss_fn in self.loss_fns_find:
                ## 新添代码
                # 当前 counting 版本已经移除了 pred_masks 输出。
                # 如果旧配置里仍然保留了 mask loss，这里直接跳过，避免继续依赖 mask 数据流。
                if isinstance(loss_fn, Masks) and "pred_masks" not in out:
                    continue
                ## 新添代码
                l_dict = loss_fn(
                    outputs=out,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=is_aux,
                )
                cur_core_loss = l_dict.pop(CORE_LOSS_KEY)
                total_core_loss += cur_core_loss
                if is_aux:
                    total_aux_weighted_loss += _as_detached_scalar_tensor(cur_core_loss)
                else:
                    total_main_weighted_loss += _as_detached_scalar_tensor(cur_core_loss)
                losses.update({f"{k}{suffix}": v for k, v in l_dict.items()})

                compute_o2m_loss = has_o2m_out
                ## 新添代码
                # separate supervision 下的某些 loss（例如 Stage 6 的 P2P loss）
                # 只作用在主输出，不应被误应用到 o2m 分支。
                if not getattr(loss_fn, "supports_o2m_loss", True):
                    compute_o2m_loss = False
                ## 新添代码
                # a special handling to allow turning off mask loss in o2m
                # (to be compatible with the original implementation)
                if isinstance(loss_fn, Masks):
                    compute_o2m_loss = compute_o2m_loss and "pred_masks" in o2m_out
                if isinstance(loss_fn, Det2TrkAssoc):
                    compute_o2m_loss = False  # Det2TrkAssoc does not support o2m
                if compute_o2m_loss:
                    l_dict = loss_fn(
                        outputs=o2m_out,
                        targets=o2m_targets,
                        indices=o2m_indices,
                        num_boxes=num_boxes,
                        is_aux=is_aux,
                    )
                    for k in l_dict:
                        l_dict[k] *= self.o2m_weight
                    cur_o2m_core_loss = l_dict.pop(CORE_LOSS_KEY)
                    total_core_loss += cur_o2m_core_loss
                    if is_aux:
                        total_aux_weighted_loss += _as_detached_scalar_tensor(
                            cur_o2m_core_loss
                        )
                    else:
                        total_main_weighted_loss += _as_detached_scalar_tensor(
                            cur_o2m_core_loss
                        )
                    losses.update({f"{k}{suffix}_o2m": v for k, v in l_dict.items()})

        losses[CORE_LOSS_KEY] = total_core_loss
        losses["main_weighted_loss"] = total_main_weighted_loss
        losses["aux_weighted_loss"] = total_aux_weighted_loss
        return losses

    def forward(self, find_stages: SAM3Output, find_targets):
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as find_stages:
            assert len(find_stages) == len(find_targets)
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                # If there are multiple steps within a stage, compute the loss for all of them (e.g. interactivity)
                for outputs, targets in zip(stage_outputs, stage_targets):
                    cur_losses = self.compute_loss(outputs, targets)

                    if self.loss_fn_semantic_seg is not None and "semantic_seg" in outputs:
                        cur_losses_semantic = self.loss_fn_semantic_seg(
                            outputs, targets
                        )
                        cur_losses[CORE_LOSS_KEY] += cur_losses_semantic.pop(
                            CORE_LOSS_KEY
                        )
                        # make sure the semantic losses don't overlap with the find losses
                        assert set(cur_losses).isdisjoint(set(cur_losses_semantic))
                        cur_losses.update(cur_losses_semantic)

                    # Optionally, normalize the loss by the number of find stages (training video frames) so that
                    # image batches and video batches have similar loss scales. (Otherwise video batches would
                    # have a much higher loss scale due to summing the losses over all the find stages.)
                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] /= len(find_stages)
                        if "main_weighted_loss" in cur_losses:
                            cur_losses["main_weighted_loss"] /= len(find_stages)
                        if "aux_weighted_loss" in cur_losses:
                            cur_losses["aux_weighted_loss"] /= len(find_stages)

                    if self.scale_by_find_batch_size:
                        bs = targets["num_boxes"].shape[0]
                        # sqrt scaling based on the "effective" batch size
                        batch_scale = bs**0.5
                        cur_losses[CORE_LOSS_KEY] *= batch_scale
                        cur_losses["batch_scale"] = cur_losses[CORE_LOSS_KEY].new_tensor(
                            batch_scale
                        )
                    else:
                        cur_losses["batch_scale"] = cur_losses[CORE_LOSS_KEY].new_tensor(
                            1.0
                        )

                    for k, v in cur_losses.items():
                        # batch_scale 对当前 batch 是一个全局描述项，不应在多 stage 上重复累加。
                        if k == "batch_scale":
                            total_losses[k] = v
                        elif k not in total_losses:
                            total_losses[k] = v
                        else:
                            total_losses[k] += v

        return total_losses
