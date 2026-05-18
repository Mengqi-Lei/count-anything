# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import contextlib
from copy import deepcopy
import fnmatch
import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from sam3.eval.counting_visualizer import (
    save_ccf_visualization,
    save_pdc_visualization,
    save_rsc_visualization,
)
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.optim.optimizer import construct_optimizer
from sam3.train.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)

from sam3.train.utils.distributed import all_reduce_max, barrier, get_rank

from sam3.train.utils.logger import Logger, setup_logging
from sam3.train.utils.logger import emit_plain_log_line
from sam3.train.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"
DEFAULT_SINGLE_STAGE_TRAINING_STAGE = "joint_counter_adaptation"
SAM3_BRANCH_BASE_LR = 1.0e-05
P2P_BRANCH_BASE_LR = 1.0e-04
LORA_BRANCH_BASE_LR = 1.0e-03
SAM3_BRANCH_DROP_EPOCHS = [25]
P2P_BRANCH_DROP_EPOCHS = [25]
TRAINING_STAGE_ALIASES = {
    "stage1_p2p_head_only": "stage1_pdc_head_only",
    "p2p_branch_only": "pdc_branch_only",
    "p2p_only": "pdc_branch_only",
    "pdc_only": "pdc_branch_only",
    "stage2_joint_adaptation": "joint_counter_adaptation",
    "sam_det_only": "rsc_only",
    "sam_only": "rsc_only",
    "encoder_p2p_only": "encoder_pdc_only",
}
TRAINING_GROUP_ALIASES = {
    "sam_detection_branch": "rsc_branch",
    "p2p_feature_input": "pdc_feature_path",
    "p2p_head": "pdc_head",
}
VISUAL_OUTPUT_ALIASES = {
    "p2p": "pdc",
    "pdc": "pdc",
    "pixel_dense_counter": "pdc",
    "pixel-level_dense_counter": "pdc",
    "sam3": "rsc",
    "sam": "rsc",
    "rsc": "rsc",
    "region_sparse_counter": "rsc",
    "region-level_sparse_counter": "rsc",
    "total": "ccf",
    "fusion": "ccf",
    "ccf": "ccf",
    "complementary_count_fusion": "ccf",
}
VISUAL_OUTPUT_LEGACY_NAMES = {
    "pdc": "p2p",
    "rsc": "sam3",
    "ccf": "total",
}


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _get_debug_matcher(model):
    return getattr(unwrap_ddp_if_wrapped(model), "matcher", None)


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = False  # PyTorch DDP default is False
    static_graph: bool = False  # PyTorch DDP default is False


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    wandb_writer: Optional[Any] = None
    show_epoch_progress: bool = False
    heartbeat_freq: Optional[int] = None


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8
    BEST_MAE_TRACKED_KEY = "val_all/counting/mae"
    BEST_MAE_CHECKPOINT_NAME = "best_mae"
    VALID_TRAINING_STAGES = {
        None,
        "stage1_pdc_head_only",
        "pdc_branch_only",
        "joint_counter_adaptation",
        "rsc_only",
        "encoder_pdc_only",
    }

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        skip_first_val: bool = False,
        skip_saving_ckpts: bool = False,
        empty_gpu_mem_cache_after_eval: bool = True,
        gradient_accumulation_steps: int = 1,
        training_stage: Optional[str] = None,
        visualize_val_every_n_epochs: int = 0,
        visualize_val_outputs: Optional[List[str]] = None,
        validate_before_train: bool = False,
        skip_resume_previous_val: bool = False,
        val_epoch_override: Optional[int] = None,
        train_foundation_encoder_lora: bool = False,
        lora_branch_base_lr: float = LORA_BRANCH_BASE_LR,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else OptimConf()
        self.meters_conf = meters
        self.loss_conf = loss
        self.gradient_accumulation_steps = gradient_accumulation_steps
        raw_training_stage = (
            training_stage
            if training_stage is not None
            else (
                DEFAULT_SINGLE_STAGE_TRAINING_STAGE
                if str(mode).lower() == "train"
                else None
            )
        )
        self.requested_training_stage = raw_training_stage
        self.training_stage = self._canonicalize_training_stage_name(
            raw_training_stage
        )
        self.visualize_val_every_n_epochs = int(visualize_val_every_n_epochs)
        if visualize_val_outputs is None:
            visualize_val_outputs = ["p2p", "sam3", "total"]
        elif isinstance(visualize_val_outputs, str):
            visualize_val_outputs = [visualize_val_outputs]
        self.requested_visualize_val_outputs = {
            str(output_name).lower() for output_name in visualize_val_outputs
        }
        self.visualize_val_outputs = self._canonicalize_visualize_val_outputs(
            self.requested_visualize_val_outputs
        )
        self.validate_before_train = bool(validate_before_train)
        self.skip_resume_previous_val = bool(skip_resume_previous_val)
        self.val_epoch_override = val_epoch_override
        self.train_foundation_encoder_lora = bool(train_foundation_encoder_lora)
        self.lora_branch_base_lr = float(lora_branch_base_lr)
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self.skip_first_val = skip_first_val
        self.skip_saving_ckpts = skip_saving_ckpts
        self.empty_gpu_mem_cache_after_eval = empty_gpu_mem_cache_after_eval
        self._active_val_visual_image_ids = set()

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            if self.distributed_rank == 0 and get_resume_checkpoint(
                self.checkpoint_conf.save_dir
            ) is None:
                # Copy the source checkpoint into the run directory using an epoch-based
                # name so the checkpoint folder layout stays consistent.
                makedir(self.checkpoint_conf.save_dir)
                resume_epoch = 0
                try:
                    with g_pathmgr.open(self.checkpoint_conf.resume_from, "rb") as f:
                        resume_ckpt = torch.load(
                            f, map_location="cpu", weights_only=False
                        )
                    resume_epoch = int(resume_ckpt.get("epoch", 0))
                except Exception as exc:
                    logging.warning(
                        "Failed to read epoch from resume checkpoint %s, defaulting to epoch0 copy: %s",
                        self.checkpoint_conf.resume_from,
                        exc,
                    )
                dst = os.path.join(
                    self.checkpoint_conf.save_dir, f"epoch{resume_epoch}.pt"
                )
                if not os.path.exists(dst):
                    g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _rgetattr(self, obj, dotted_path, default=None):
        if dotted_path is None:
            return obj
        cur = obj
        for attr in dotted_path.split("."):
            if not hasattr(cur, attr):
                return default
            cur = getattr(cur, attr)
        return cur

    @staticmethod
    def _canonicalize_training_stage_name(training_stage: Optional[str]) -> Optional[str]:
        if training_stage is None:
            return None
        training_stage = str(training_stage)
        return TRAINING_STAGE_ALIASES.get(training_stage, training_stage)

    @staticmethod
    def _canonicalize_training_group_name(group_name: str) -> str:
        return TRAINING_GROUP_ALIASES.get(group_name, group_name)

    def _validate_training_stage(self):
        if self.training_stage not in self.VALID_TRAINING_STAGES:
            accepted_stages = sorted(
                {
                    stage
                    for stage in self.VALID_TRAINING_STAGES
                    if stage is not None
                }
                | set(TRAINING_STAGE_ALIASES.keys())
            )
            raise ValueError(
                f"Unsupported training_stage={self.training_stage}. "
                f"Expected one of {accepted_stages} "
                "or None."
            )

    def _get_stage_module_paths(self) -> Dict[str, List[str]]:
        # 这里把“冻结/激活的最小模块”显式写死，避免后续实现者再做二次猜测。
        return {
            "foundation": [
                "backbone",
                "geometry_encoder",
                "transformer.encoder",
            ],
            "rsc_branch": [
                "transformer.decoder",
                "dot_prod_scoring",
                "instance_dot_prod_scoring",
                "class_embed",
                "instance_class_embed",
            ],
            "pdc_feature_path": ["segmentation_head"],
            "pdc_head": ["pdc_branch"],
        }

    def _get_training_stage_module_states(self) -> Dict[str, bool]:
        if self.training_stage is None:
            return {
                "foundation": True,
                "rsc_branch": True,
                "pdc_feature_path": True,
                "pdc_head": True,
            }
        if self.training_stage == "stage1_pdc_head_only":
            return {
                "foundation": False,
                "rsc_branch": False,
                "pdc_feature_path": False,
                "pdc_head": True,
            }
        if self.training_stage == "pdc_branch_only":
            return {
                "foundation": False,
                "rsc_branch": False,
                "pdc_feature_path": True,
                "pdc_head": True,
            }
        if self.training_stage == "joint_counter_adaptation":
            return {
                "foundation": False,
                "rsc_branch": True,
                "pdc_feature_path": True,
                "pdc_head": True,
            }
        if self.training_stage == "rsc_only":
            return {
                "foundation": False,
                "rsc_branch": True,
                "pdc_feature_path": False,
                "pdc_head": False,
            }
        if self.training_stage == "encoder_pdc_only":
            return {
                "foundation": True,
                "rsc_branch": False,
                "pdc_feature_path": True,
                "pdc_head": True,
            }
        raise AssertionError(f"Unhandled training stage: {self.training_stage}")

    def _get_stage_kept_loss_names(self) -> Optional[Set[str]]:
        if self.training_stage in {"stage1_pdc_head_only", "pdc_branch_only"}:
            return {"PDCPointSetLoss"}
        if self.training_stage == "rsc_only":
            return {"Boxes", "Points", "IABCEMdetr"}
        return None

    def _filter_stage_specific_losses(self) -> None:
        kept_loss_names = self._get_stage_kept_loss_names()
        if self.loss is None or kept_loss_names is None:
            return

        removed_summary = []

        for wrapped_loss_name, wrapped_loss in self.loss.items():
            loss_fns = getattr(wrapped_loss, "loss_fns_find", None)
            if loss_fns is None:
                continue

            original_loss_fns = list(loss_fns)
            filtered_loss_fns = [
                loss_fn
                for loss_fn in original_loss_fns
                if loss_fn.__class__.__name__ in kept_loss_names
            ]

            if not filtered_loss_fns:
                raise RuntimeError(
                    f"training_stage={self.training_stage} requires one of "
                    f"{sorted(kept_loss_names)}, "
                    f"but wrapped loss '{wrapped_loss_name}' does not contain it."
                )

            removed_names = [
                loss_fn.__class__.__name__
                for loss_fn in original_loss_fns
                if loss_fn.__class__.__name__ not in kept_loss_names
            ]
            if removed_names:
                removed_summary.append(
                    f"{wrapped_loss_name}: removed={removed_names}"
                )

            if isinstance(loss_fns, nn.ModuleList):
                wrapped_loss.loss_fns_find = nn.ModuleList(filtered_loss_fns)
            elif isinstance(loss_fns, tuple):
                wrapped_loss.loss_fns_find = tuple(filtered_loss_fns)
            else:
                wrapped_loss.loss_fns_find = filtered_loss_fns

        if removed_summary:
            logging.info(
                "STAGE_LOSS_FILTER training_stage=%s kept=%s %s",
                self.training_stage,
                sorted(kept_loss_names),
                " | ".join(removed_summary),
            )

    def _collect_group_parameter_names(
        self, model: nn.Module, group_name: str
    ) -> Set[str]:
        group_name = self._canonicalize_training_group_name(group_name)
        names = set()
        for module_path in self._get_stage_module_paths()[group_name]:
            module = self._rgetattr(model, module_path, default=None)
            if module is None:
                continue
            for param_name, _ in module.named_parameters(recurse=True):
                full_name = (
                    module_path if param_name == "" else f"{module_path}.{param_name}"
                )
                names.add(full_name)
        return names

    def _get_foundation_encoder_lora_param_names(self, model: nn.Module) -> Set[str]:
        return {
            name
            for name, _ in model.named_parameters()
            if name.startswith("transformer.encoder.")
            and ".cross_attn_image." in name
            and ".lora_" in name
        }

    def _set_foundation_encoder_lora_trainable(self, model: nn.Module) -> Set[str]:
        lora_names = self._get_foundation_encoder_lora_param_names(model)
        if not lora_names:
            raise ValueError(
                "train_foundation_encoder_lora=True but no encoder cross-attn "
                "LoRA params were found. Set model.enable_encoder_cross_attn_lora=True."
            )

        trainable_tensors = 0
        trainable_params = 0
        for name, parameter in model.named_parameters():
            if name in lora_names:
                parameter.requires_grad_(True)
                trainable_tensors += 1
                trainable_params += parameter.numel()

        for module_name, module in model.named_modules():
            if (
                module_name.startswith("transformer.encoder.")
                and "cross_attn_image" in module_name
                and hasattr(module, "lora_parameters")
            ):
                module.train()

        logging.info(
            "FOUNDATION_ENCODER_LORA trainable_params=%s trainable_tensors=%s",
            trainable_params,
            trainable_tensors,
        )
        return lora_names

    def _iter_effective_loss_fns(self):
        if getattr(self, "loss", None) is None:
            return
        for wrapped_loss in self.loss.values():
            loss_fns = getattr(wrapped_loss, "loss_fns_find", None)
            if loss_fns is None:
                continue
            for loss_fn in loss_fns:
                yield loss_fn

    def _should_freeze_presence_only_parameters(self) -> bool:
        iabce_losses = [
            loss_fn
            for loss_fn in self._iter_effective_loss_fns() or []
            if loss_fn.__class__.__name__ == "IABCEMdetr"
        ]
        if not iabce_losses:
            return False

        for loss_fn in iabce_losses:
            use_presence = bool(getattr(loss_fn, "use_presence", True))
            presence_loss = getattr(loss_fn, "presence_loss", None)
            if isinstance(presence_loss, torch.Tensor):
                presence_loss = presence_loss.detach().item()
            presence_loss = 0.0 if presence_loss is None else float(presence_loss)
            if use_presence or presence_loss > 0.0:
                return False
        return True

    def _freeze_presence_only_parameters_if_needed(self) -> Set[str]:
        self._presence_frozen_param_names = set()
        if not self._should_freeze_presence_only_parameters():
            return set()

        model = unwrap_ddp_if_wrapped(self.model)
        decoder = self._rgetattr(model, "transformer.decoder", default=None)
        if decoder is None:
            return set()

        frozen_names = set()
        presence_module_paths = [
            ("transformer.decoder.presence_token", getattr(decoder, "presence_token", None)),
            (
                "transformer.decoder.presence_token_head",
                getattr(decoder, "presence_token_head", None),
            ),
            (
                "transformer.decoder.presence_token_out_norm",
                getattr(decoder, "presence_token_out_norm", None),
            ),
        ]

        for module_path, module in presence_module_paths:
            if module is None:
                continue
            for param_name, parameter in module.named_parameters(recurse=True):
                full_name = (
                    module_path if param_name == "" else f"{module_path}.{param_name}"
                )
                if parameter.requires_grad:
                    parameter.requires_grad_(False)
                    frozen_names.add(full_name)
            module.eval()

        if frozen_names:
            logging.info(
                "PRESENCE_PARAM_FREEZE training_stage=%s frozen_params=%s",
                self.training_stage,
                sorted(frozen_names),
            )
        self._presence_frozen_param_names = frozen_names
        return frozen_names

    def _get_gradient_clip_parameter_groups(self) -> Optional[Dict[str, List[nn.Parameter]]]:
        model = unwrap_ddp_if_wrapped(self.model)
        trainable_named_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not trainable_named_parameters:
            return None

        active_param_names = self._get_optimizer_param_allowlist()
        if active_param_names is None:
            active_param_names = set(trainable_named_parameters.keys())
        else:
            active_param_names = set(active_param_names) & set(trainable_named_parameters.keys())

        if not active_param_names:
            return None

        def _names_for_groups(*group_names: str) -> Set[str]:
            collected = set()
            for group_name in group_names:
                collected.update(self._collect_group_parameter_names(model, group_name))
            return collected & active_param_names

        grouped_names = OrderedDict()
        grouped_names["pdc"] = _names_for_groups("pdc_feature_path", "pdc_head")
        grouped_names["rsc"] = _names_for_groups("rsc_branch")
        encoder_lora_names = (
            self._get_foundation_encoder_lora_param_names(model) & active_param_names
        )
        grouped_names["encoder_lora"] = encoder_lora_names
        grouped_names["foundation"] = _names_for_groups("foundation") - encoder_lora_names

        used_names = set().union(*grouped_names.values()) if grouped_names else set()
        other_names = active_param_names - used_names
        if other_names:
            grouped_names["other"] = other_names

        parameter_groups = OrderedDict()
        for group_name, param_names in grouped_names.items():
            if not param_names:
                continue
            params = [trainable_named_parameters[name] for name in sorted(param_names)]
            if params:
                parameter_groups[group_name] = params

        return parameter_groups or None

    def _get_branch_numeric_debug_named_parameters(
        self,
    ) -> Optional[Dict[str, List[tuple[str, nn.Parameter]]]]:
        model = unwrap_ddp_if_wrapped(self.model)
        trainable_named_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not trainable_named_parameters:
            return None

        active_param_names = self._get_optimizer_param_allowlist()
        if active_param_names is None:
            active_param_names = set(trainable_named_parameters.keys())
        else:
            active_param_names = set(active_param_names) & set(
                trainable_named_parameters.keys()
            )

        if not active_param_names:
            return None

        def _names_for_groups(*group_names: str) -> Set[str]:
            collected = set()
            for group_name in group_names:
                collected.update(
                    self._collect_group_parameter_names(
                        model=model, group_name=group_name
                    )
                )
            return collected & active_param_names

        grouped_names = OrderedDict()
        grouped_names["rsc_branch"] = _names_for_groups("rsc_branch")
        grouped_names["pdc_feature_path"] = _names_for_groups("pdc_feature_path")
        grouped_names["pdc_head"] = _names_for_groups("pdc_head")
        grouped_names["encoder_lora"] = (
            self._get_foundation_encoder_lora_param_names(model) & active_param_names
        )

        grouped_named_parameters = OrderedDict()
        for group_name, param_names in grouped_names.items():
            if not param_names:
                continue
            grouped_named_parameters[group_name] = [
                (name, trainable_named_parameters[name]) for name in sorted(param_names)
            ]

        return grouped_named_parameters or None

    def _reset_branch_numeric_debug(self):
        self._branch_numeric_debug_named_parameters = (
            self._get_branch_numeric_debug_named_parameters()
        )
        self._branch_numeric_debug_stats = OrderedDict()
        self._last_step_loss_snapshot = None
        self._sam_det_forward_nonfinite_logged = False
        self._sam_det_backward_nonfinite_logged = set()
        if not self._branch_numeric_debug_named_parameters:
            return

        for group_name, named_parameters in self._branch_numeric_debug_named_parameters.items():
            self._branch_numeric_debug_stats[group_name] = {
                "checks": 0,
                "param_count": len(named_parameters),
                "nonfinite_grad_events": 0,
                "nonfinite_param_events": 0,
                "first_nonfinite_grad_step": None,
                "first_nonfinite_param_step": None,
                "max_grad_norm": 0.0,
                "last_grad_norm": None,
                "last_grad_bad_count": 0,
                "last_param_bad_count": 0,
                "last_grad_present": 0,
                "last_grad_missing": len(named_parameters),
                "last_grad_all_finite": True,
                "last_param_all_finite": True,
            }

    def _capture_step_loss_snapshot(
        self,
        phase: str,
        loss_key: str,
        loss: torch.Tensor,
        extra_losses: Dict[str, torch.Tensor],
    ) -> None:
        if phase != Phase.TRAIN:
            return

        snapshot = OrderedDict()
        snapshot["loss_key"] = loss_key
        snapshot["loss"] = self._as_python_float(loss)
        snapshot["scaler_scale"] = (
            float(self.scaler.get_scale())
            if self.optim_conf.amp.enabled
            else None
        )
        snapshot["lrs"] = [
            float(param_group.get("lr", 0.0))
            for param_group in self.optim.optimizer.param_groups
        ]
        snapshot["extra_losses"] = OrderedDict(
            (extra_loss_key, self._as_python_float(extra_loss))
            for extra_loss_key, extra_loss in sorted(extra_losses.items())
        )
        self._last_step_loss_snapshot = snapshot

    def _format_step_loss_snapshot(self) -> str:
        snapshot = getattr(self, "_last_step_loss_snapshot", None)
        if not snapshot:
            return "loss_snapshot=None"

        extra_losses = snapshot.get("extra_losses", {})
        extra_loss_str = ",".join(
            f"{key}={value:.6e}" for key, value in extra_losses.items()
        )
        lrs = snapshot.get("lrs", [])
        lr_str = ",".join(f"{lr:.3e}" for lr in lrs)
        scaler_scale = snapshot.get("scaler_scale")
        scaler_str = "None" if scaler_scale is None else f"{scaler_scale:.6e}"
        return (
            f"loss_key={snapshot.get('loss_key')} "
            f"loss={snapshot.get('loss', 0.0):.6e} "
            f"scaler_scale={scaler_str} "
            f"lrs=[{lr_str}] "
            f"extra_losses=[{extra_loss_str}]"
        )

    def _tensor_finite_summary(
        self, tensor: Optional[torch.Tensor], with_layer_breakdown: bool = False
    ) -> Dict[str, Any]:
        if tensor is None:
            return {"shape": None, "bad": 0, "first_bad_indices": [], "layer_bad": None}

        data = tensor.detach()
        finite_mask = torch.isfinite(data)
        bad_mask = ~finite_mask
        bad_count = int(bad_mask.sum().item())
        first_bad_indices = bad_mask.nonzero(as_tuple=False)[:8].tolist()
        finite_values = data[finite_mask]
        if finite_values.numel() > 0:
            finite_values = finite_values.double()
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            mean_value = float(finite_values.mean().item())
        else:
            min_value = None
            max_value = None
            mean_value = None

        layer_bad = None
        if with_layer_breakdown and data.dim() >= 1:
            layer_bad = bad_mask.reshape(data.shape[0], -1).sum(dim=1).tolist()

        return {
            "shape": tuple(data.shape),
            "bad": bad_count,
            "first_bad_indices": first_bad_indices,
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "layer_bad": layer_bad,
        }

    def _iter_stage_output_dicts(self, find_stages: SAM3Output):
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as iter_find_stages:
            for stage_idx, stage_outputs in enumerate(iter_find_stages):
                for stage_step_idx, outputs in enumerate(stage_outputs):
                    yield stage_idx, stage_step_idx, "main", outputs
                    for aux_idx, aux_out in enumerate(outputs.get("aux_outputs", [])):
                        yield stage_idx, stage_step_idx, f"aux_{aux_idx}", aux_out

    def _log_sam_det_forward_finite_probe(
        self, find_stages: SAM3Output, phase: str
    ) -> None:
        if phase != Phase.TRAIN or self._sam_det_forward_nonfinite_logged:
            return

        tensor_specs = (
            ("pred_logits", "pred_logits", False),
            ("pred_boxes", "pred_boxes", False),
            ("presence_logit_dec", "presence_logit_dec", False),
            ("_debug_dec_presence_out", "decoder_presence", False),
            ("_debug_reference_boxes", "decoder_reference_boxes", True),
            ("_debug_decoder_hs", "decoder_hs", True),
        )

        for stage_idx, stage_step_idx, output_name, outputs in self._iter_stage_output_dicts(
            find_stages
        ):
            for key, label, with_layer_breakdown in tensor_specs:
                tensor = outputs.get(key)
                if tensor is None:
                    continue
                summary = self._tensor_finite_summary(
                    tensor, with_layer_breakdown=with_layer_breakdown
                )
                if summary["bad"] <= 0:
                    continue
                self._sam_det_forward_nonfinite_logged = True
                logging.error(
                    "SAM_DET_FORWARD_NONFINITE | epoch=%s phase=%s step=%s stage_idx=%s "
                    "stage_step=%s output=%s tensor=%s shape=%s bad=%s first_bad=%s "
                    "layer_bad=%s min=%s max=%s mean=%s %s",
                    self.epoch,
                    phase,
                    self.steps[phase],
                    stage_idx,
                    stage_step_idx,
                    output_name,
                    label,
                    summary["shape"],
                    summary["bad"],
                    summary["first_bad_indices"],
                    summary["layer_bad"],
                    summary["min"],
                    summary["max"],
                    summary["mean"],
                    self._format_step_loss_snapshot(),
                )
                return

    def _register_sam_det_backward_finite_hooks(
        self, find_stages: SAM3Output, phase: str
    ) -> None:
        if phase != Phase.TRAIN:
            return

        tensor_specs = (
            ("pred_logits", "pred_logits", False),
            ("pred_boxes", "pred_boxes", False),
            ("presence_logit_dec", "presence_logit_dec", False),
            ("_debug_dec_presence_out", "decoder_presence", False),
            ("_debug_reference_boxes", "decoder_reference_boxes", True),
            ("_debug_decoder_hs", "decoder_hs", True),
        )

        def _make_hook(
            *,
            stage_idx: int,
            stage_step_idx: int,
            output_name: str,
            tensor_label: str,
            with_layer_breakdown: bool,
        ):
            def _hook(grad: torch.Tensor):
                summary = self._tensor_finite_summary(
                    grad, with_layer_breakdown=with_layer_breakdown
                )
                if summary["bad"] <= 0:
                    return grad

                hook_key = (
                    stage_idx,
                    stage_step_idx,
                    output_name,
                    tensor_label,
                )
                if hook_key in self._sam_det_backward_nonfinite_logged:
                    return grad

                self._sam_det_backward_nonfinite_logged.add(hook_key)
                logging.error(
                    "SAM_DET_BACKWARD_NONFINITE | epoch=%s phase=%s step=%s stage_idx=%s "
                    "stage_step=%s output=%s tensor=%s shape=%s bad=%s first_bad=%s "
                    "layer_bad=%s min=%s max=%s mean=%s %s",
                    self.epoch,
                    phase,
                    max(int(self.steps[phase]) - 1, 0),
                    stage_idx,
                    stage_step_idx,
                    output_name,
                    tensor_label,
                    summary["shape"],
                    summary["bad"],
                    summary["first_bad_indices"],
                    summary["layer_bad"],
                    summary["min"],
                    summary["max"],
                    summary["mean"],
                    self._format_step_loss_snapshot(),
                )
                return grad

            return _hook

        for stage_idx, stage_step_idx, output_name, outputs in self._iter_stage_output_dicts(
            find_stages
        ):
            for key, label, with_layer_breakdown in tensor_specs:
                tensor = outputs.get(key)
                if tensor is None or not torch.is_tensor(tensor) or not tensor.requires_grad:
                    continue
                tensor.register_hook(
                    _make_hook(
                        stage_idx=stage_idx,
                        stage_step_idx=stage_step_idx,
                        output_name=output_name,
                        tensor_label=label,
                        with_layer_breakdown=with_layer_breakdown,
                    )
                )

    def _collect_parameter_group_numeric_stats(
        self, named_parameters: List[tuple[str, nn.Parameter]]
    ) -> Dict[str, Any]:
        param_bad_count = 0
        grad_bad_count = 0
        grad_present = 0
        grad_missing = 0
        grad_sq_sum = 0.0
        bad_grad_params = []
        bad_param_params = []
        top_grad_candidates = []

        for name, parameter in named_parameters:
            param_data = parameter.detach()
            param_finite_mask = torch.isfinite(param_data)
            cur_param_bad_count = int((~param_finite_mask).sum().item())
            param_bad_count += cur_param_bad_count
            if cur_param_bad_count > 0 and len(bad_param_params) < 8:
                bad_param_params.append(f"{name}:bad={cur_param_bad_count}")

            grad = parameter.grad
            if grad is None:
                grad_missing += 1
                continue

            grad_present += 1
            grad_data = grad.detach()
            grad_finite_mask = torch.isfinite(grad_data)
            cur_grad_bad_count = int((~grad_finite_mask).sum().item())
            grad_bad_count += cur_grad_bad_count
            if grad_finite_mask.any():
                finite_grad = grad_data[grad_finite_mask].double()
                cur_grad_sq_sum = float(torch.sum(finite_grad * finite_grad).item())
                grad_sq_sum += cur_grad_sq_sum
                cur_grad_norm = math.sqrt(cur_grad_sq_sum) if cur_grad_sq_sum > 0.0 else 0.0
                cur_grad_abs_max = float(torch.max(torch.abs(finite_grad)).item())
            else:
                cur_grad_norm = 0.0
                cur_grad_abs_max = None

            top_grad_candidates.append(
                (
                    cur_grad_norm,
                    name,
                    cur_grad_bad_count,
                    cur_grad_abs_max,
                )
            )
            if cur_grad_bad_count > 0 and len(bad_grad_params) < 8:
                abs_max_str = (
                    "None" if cur_grad_abs_max is None else f"{cur_grad_abs_max:.6e}"
                )
                bad_grad_params.append(
                    f"{name}:bad={cur_grad_bad_count}:finite_norm={cur_grad_norm:.6e}:finite_absmax={abs_max_str}"
                )

        top_grad_candidates.sort(key=lambda item: item[0], reverse=True)
        top_grad_params = []
        for grad_norm, name, grad_bad, grad_abs_max in top_grad_candidates[:5]:
            abs_max_str = "None" if grad_abs_max is None else f"{grad_abs_max:.6e}"
            top_grad_params.append(
                f"{name}:finite_norm={grad_norm:.6e}:grad_bad={grad_bad}:finite_absmax={abs_max_str}"
            )

        return {
            "param_all_finite": param_bad_count == 0,
            "param_bad_count": param_bad_count,
            "grad_all_finite": grad_bad_count == 0,
            "grad_bad_count": grad_bad_count,
            "grad_present": grad_present,
            "grad_missing": grad_missing,
            "grad_norm": math.sqrt(grad_sq_sum) if grad_sq_sum > 0.0 else 0.0,
            "bad_grad_params": bad_grad_params,
            "bad_param_params": bad_param_params,
            "top_grad_params": top_grad_params,
        }

    def _log_branch_numeric_debug(self, phase: str):
        if phase != "train":
            return
        if not getattr(self, "_branch_numeric_debug_named_parameters", None):
            return

        current_step = max(int(self.steps[phase]) - 1, 0)
        for group_name, named_parameters in self._branch_numeric_debug_named_parameters.items():
            current_stats = self._collect_parameter_group_numeric_stats(named_parameters)
            summary = self._branch_numeric_debug_stats[group_name]
            summary["checks"] += 1
            summary["last_grad_norm"] = current_stats["grad_norm"]
            summary["last_grad_bad_count"] = current_stats["grad_bad_count"]
            summary["last_param_bad_count"] = current_stats["param_bad_count"]
            summary["last_grad_present"] = current_stats["grad_present"]
            summary["last_grad_missing"] = current_stats["grad_missing"]
            summary["last_grad_all_finite"] = current_stats["grad_all_finite"]
            summary["last_param_all_finite"] = current_stats["param_all_finite"]
            summary["max_grad_norm"] = max(
                float(summary["max_grad_norm"]), float(current_stats["grad_norm"])
            )

            if not current_stats["grad_all_finite"]:
                summary["nonfinite_grad_events"] += 1
                if summary["first_nonfinite_grad_step"] is None:
                    summary["first_nonfinite_grad_step"] = current_step
                    logging.error(
                        "BRANCH_NUMERICS_ALERT | epoch=%s phase=%s step=%s group=%s "
                        "grad_all_finite=%s grad_bad_count=%s grad_norm=%.6e "
                        "param_all_finite=%s param_bad_count=%s grad_present=%s grad_missing=%s",
                        self.epoch,
                        phase,
                        current_step,
                        group_name,
                        current_stats["grad_all_finite"],
                        current_stats["grad_bad_count"],
                        float(current_stats["grad_norm"]),
                        current_stats["param_all_finite"],
                        current_stats["param_bad_count"],
                        current_stats["grad_present"],
                        current_stats["grad_missing"],
                    )
                    logging.error(
                        "BRANCH_NUMERICS_DETAIL | epoch=%s phase=%s step=%s group=%s "
                        "bad_grad_params=%s top_grad_params=%s %s",
                        self.epoch,
                        phase,
                        current_step,
                        group_name,
                        current_stats["bad_grad_params"],
                        current_stats["top_grad_params"],
                        self._format_step_loss_snapshot(),
                    )

            if not current_stats["param_all_finite"]:
                summary["nonfinite_param_events"] += 1
                if summary["first_nonfinite_param_step"] is None:
                    summary["first_nonfinite_param_step"] = current_step
                    logging.error(
                        "BRANCH_PARAM_ALERT | epoch=%s phase=%s step=%s group=%s "
                        "param_all_finite=%s param_bad_count=%s grad_all_finite=%s grad_bad_count=%s",
                        self.epoch,
                        phase,
                        current_step,
                        group_name,
                        current_stats["param_all_finite"],
                        current_stats["param_bad_count"],
                        current_stats["grad_all_finite"],
                        current_stats["grad_bad_count"],
                    )
                    logging.error(
                        "BRANCH_PARAM_DETAIL | epoch=%s phase=%s step=%s group=%s "
                        "bad_param_params=%s top_grad_params=%s %s",
                        self.epoch,
                        phase,
                        current_step,
                        group_name,
                        current_stats["bad_param_params"],
                        current_stats["top_grad_params"],
                        self._format_step_loss_snapshot(),
                    )

    def _log_branch_numeric_debug_summary(self, phase: str):
        if phase != "train":
            return
        stats = getattr(self, "_branch_numeric_debug_stats", None)
        if not stats:
            return

        for group_name, summary in stats.items():
            logging.info(
                "BRANCH_NUMERICS_SUMMARY | epoch=%s phase=%s group=%s checks=%s "
                "param_count=%s grad_present=%s grad_missing=%s grad_all_finite=%s "
                "param_all_finite=%s nonfinite_grad_events=%s first_nonfinite_grad_step=%s "
                "nonfinite_param_events=%s first_nonfinite_param_step=%s "
                "max_grad_norm=%.6e last_grad_norm=%.6e last_grad_bad_count=%s last_param_bad_count=%s",
                self.epoch,
                phase,
                group_name,
                summary["checks"],
                summary["param_count"],
                summary["last_grad_present"],
                summary["last_grad_missing"],
                summary["last_grad_all_finite"],
                summary["last_param_all_finite"],
                summary["nonfinite_grad_events"],
                summary["first_nonfinite_grad_step"],
                summary["nonfinite_param_events"],
                summary["first_nonfinite_param_step"],
                float(summary["max_grad_norm"]),
                float(summary["last_grad_norm"] or 0.0),
                summary["last_grad_bad_count"],
                summary["last_param_bad_count"],
            )

    def _get_branch_lr_options(self):
        options_conf = deepcopy(self.optim_conf.options)
        if not options_conf or "lr" not in options_conf or not options_conf["lr"]:
            return options_conf

        model = unwrap_ddp_if_wrapped(self.model)
        active_param_names = self._get_optimizer_param_allowlist()
        if active_param_names is None:
            active_param_names = {name for name, _ in model.named_parameters()}

        pdc_param_names = set()
        for group_name in ("pdc_feature_path", "pdc_head"):
            pdc_param_names.update(self._collect_group_parameter_names(model, group_name))
        pdc_param_names &= active_param_names
        lora_param_names = (
            self._get_foundation_encoder_lora_param_names(model) & active_param_names
        )

        lr_cfg_template = deepcopy(options_conf["lr"][0])
        lr_cfg_template.pop("param_names", None)
        lr_cfg_template["scheduler"]["base_lr"] = SAM3_BRANCH_BASE_LR
        if "drop_epochs" in lr_cfg_template["scheduler"]:
            lr_cfg_template["scheduler"]["drop_epochs"] = list(SAM3_BRANCH_DROP_EPOCHS)
        options_conf["lr"] = [lr_cfg_template]

        if pdc_param_names:
            pdc_lr_cfg = deepcopy(lr_cfg_template)
            pdc_lr_cfg["scheduler"]["base_lr"] = P2P_BRANCH_BASE_LR
            if "drop_epochs" in pdc_lr_cfg["scheduler"]:
                pdc_lr_cfg["scheduler"]["drop_epochs"] = list(P2P_BRANCH_DROP_EPOCHS)
            pdc_lr_cfg["param_names"] = sorted(pdc_param_names)
            options_conf["lr"].append(pdc_lr_cfg)

        if lora_param_names:
            lora_lr_cfg = deepcopy(lr_cfg_template)
            lora_lr_cfg["scheduler"]["base_lr"] = self.lora_branch_base_lr
            if "drop_epochs" in lora_lr_cfg["scheduler"]:
                lora_lr_cfg["scheduler"]["drop_epochs"] = list(SAM3_BRANCH_DROP_EPOCHS)
            lora_lr_cfg["param_names"] = sorted(lora_param_names)
            options_conf["lr"].append(lora_lr_cfg)

        rsc_or_encoder_param_count = len(
            active_param_names - pdc_param_names - lora_param_names
        )
        logging.info(
            "LR_GROUPS rsc_lr=%.1e rsc_drop_epochs=%s rsc_or_encoder_params=%d "
            "pdc_lr=%.1e pdc_drop_epochs=%s pdc_params=%d "
            "encoder_lora_lr=%.1e encoder_lora_params=%d",
            SAM3_BRANCH_BASE_LR,
            SAM3_BRANCH_DROP_EPOCHS,
            rsc_or_encoder_param_count,
            P2P_BRANCH_BASE_LR,
            P2P_BRANCH_DROP_EPOCHS,
            len(pdc_param_names),
            self.lora_branch_base_lr,
            len(lora_param_names),
        )
        return options_conf

    def _set_module_trainability(self, module: nn.Module, is_active: bool):
        for parameter in module.parameters(recurse=True):
            parameter.requires_grad = is_active
        if is_active:
            module.train()
        else:
            module.eval()

    def _apply_training_stage(self):
        self._validate_training_stage()
        if self.training_stage is None:
            return

        model = unwrap_ddp_if_wrapped(self.model)
        module_states = self._get_training_stage_module_states()
        module_paths = self._get_stage_module_paths()

        for group_name, is_active in module_states.items():
            for module_path in module_paths[group_name]:
                module = self._rgetattr(model, module_path, default=None)
                if module is None:
                    continue
                self._set_module_trainability(module, is_active)

        # Stage 配置里被要求激活的模块必须真实存在且拥有参数，
        # 否则说明模型构建或配置本身有问题，应该尽早失败。
        for group_name, is_active in module_states.items():
            if not is_active:
                continue
            group_param_names = self._collect_group_parameter_names(model, group_name)
            if len(group_param_names) == 0:
                raise ValueError(
                    f"Training stage {self.training_stage} requires active group "
                    f"'{group_name}', but no parameters were found."
                )

        if self.train_foundation_encoder_lora:
            self._set_foundation_encoder_lora_trainable(model)

        self._freeze_presence_only_parameters_if_needed()

    def _get_optimizer_param_allowlist(self) -> Optional[Set[str]]:
        if self.training_stage is None:
            return None
        model = unwrap_ddp_if_wrapped(self.model)
        module_states = self._get_training_stage_module_states()
        allowlist = set()
        for group_name, is_active in module_states.items():
            if not is_active:
                continue
            allowlist.update(self._collect_group_parameter_names(model, group_name))
        if self.train_foundation_encoder_lora:
            allowlist.update(self._get_foundation_encoder_lora_param_names(model))
        allowlist -= getattr(self, "_presence_frozen_param_names", set())
        return allowlist

    def _get_training_stage_log_strings(self):
        if self.training_stage is None:
            return None, None

        model = unwrap_ddp_if_wrapped(self.model)
        module_states = self._get_training_stage_module_states()
        group_alias = {
            "foundation": ("foundation",),
            "rsc_branch": ("rsc",),
            "pdc_feature_path": ("pdc_features",),
            "pdc_head": ("pdc_head",),
        }
        state_items = []
        count_items = []
        encoder_lora_param_names = self._get_foundation_encoder_lora_param_names(model)
        for group_name in [
            "foundation",
            "rsc_branch",
            "pdc_feature_path",
            "pdc_head",
        ]:
            is_active = module_states[group_name]
            state = "ACTIVE" if is_active else "FROZEN"
            group_param_names = self._collect_group_parameter_names(model, group_name)
            if group_name == "foundation":
                group_param_names = group_param_names - encoder_lora_param_names
            trainable_count = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in group_param_names and parameter.requires_grad
            )
            for alias in group_alias[group_name]:
                state_items.append(f"{alias}={state}")
                count_items.append(f"{alias}={trainable_count}")
        encoder_lora_trainable_count = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in encoder_lora_param_names and parameter.requires_grad
        )
        state_items.append(
            f"encoder_lora={'ACTIVE' if encoder_lora_trainable_count > 0 else 'FROZEN'}"
        )
        count_items.append(f"encoder_lora={encoder_lora_trainable_count}")
        return (
            "MODULE_STATE " + " ".join(state_items),
            "TRAINABLE_PARAMS " + " ".join(count_items),
        )

    def _log_training_stage_state(self):
        if (
            getattr(self, "requested_training_stage", self.training_stage)
            != self.training_stage
        ):
            logging.info(
                "TRAINING_STAGE_ALIAS requested=%s canonical=%s",
                self.requested_training_stage,
                self.training_stage,
            )
        state_line, count_line = self._get_training_stage_log_strings()
        if state_line is not None:
            logging.info(state_line)
            logging.info(count_line)
        matcher_line = self._get_pdc_matcher_log_string()
        if matcher_line is not None:
            logging.info(matcher_line)

    def _get_pdc_matcher_log_string(self) -> Optional[str]:
        loss_obj = getattr(self, "loss", None)
        if loss_obj is None or "all" not in loss_obj:
            return None
        wrapped_loss = loss_obj["all"]
        loss_fns = getattr(wrapped_loss, "loss_fns_find", None)
        if loss_fns is None:
            return None
        for loss_fn in loss_fns:
            if loss_fn.__class__.__name__ != "PDCPointSetLoss":
                continue
            matcher = getattr(loss_fn, "matcher", None)
            eos_coef = getattr(loss_fn, "eos_coef", None)
            cost_point = getattr(matcher, "cost_point", None) if matcher is not None else None
            cost_class = getattr(matcher, "cost_class", None) if matcher is not None else None
            if cost_point is None or cost_class is None:
                continue
            parts = [
                "PDC_MATCHER_WEIGHTS",
                f"point={self._format_log_scalar(cost_point)}",
                f"score={self._format_log_scalar(cost_class)}",
            ]
            if eos_coef is not None:
                parts.append(f"eos={self._format_log_scalar(eos_coef)}")
            return " ".join(parts)
        return None

    def _emit_log_blank_line(self):
        # 直接向 handler stream 写空行，避免出现带 formatter 前缀的“伪空行”。
        emit_plain_log_line("")

    def _format_log_scalar(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"

    def _format_log_lr(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        return f"{float(value):.1e}"

    def _get_optimizer_lr_summary(self) -> str:
        lr_values = [self._format_log_lr(pg["lr"]) for pg in self.optim.optimizer.param_groups]
        if len(lr_values) == 1:
            return lr_values[0]
        return ",".join(f"{idx}:{lr}" for idx, lr in enumerate(lr_values))
        return str(value)

    def _as_python_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.item())
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        raise TypeError(f"Unsupported scalar value type: {type(value)}")

    def _get_loss_weight(self, loss_key: str) -> Optional[float]:
        if self.loss is None or "all" not in self.loss:
            return None
        wrapped_loss = self.loss["all"]
        loss_fns = getattr(wrapped_loss, "loss_fns_find", None)
        if loss_fns is None:
            return None
        for loss_fn in loss_fns:
            weight_dict = getattr(loss_fn, "weight_dict", None)
            if weight_dict is not None and loss_key in weight_dict:
                return float(weight_dict[loss_key])
        return None

    def _format_weight_multiplier(self, multiplier: float) -> str:
        if float(multiplier).is_integer():
            return f"{int(multiplier)}x"
        return f"{multiplier:g}x"

    def _format_weighted_loss_for_summary(
        self,
        out_dict: Mapping[str, Any],
        dict_key: str,
        loss_key: str,
    ) -> Optional[str]:
        if dict_key not in out_dict:
            return None
        weight = self._get_loss_weight(loss_key)
        if weight is None:
            return self._format_log_scalar(out_dict[dict_key])
        weighted_value = self._as_python_float(out_dict[dict_key]) * weight
        return (
            f"{self._format_log_scalar(weighted_value)}"
            f"({self._format_weight_multiplier(weight)})"
        )

    def _build_compact_summary_line(
        self, prefix: str, out_dict: Mapping[str, Any], summary_keys: List[tuple]
    ) -> str:
        # 训练日志只保留核心指标，避免整份原始字典直接刷屏。
        summary_parts = [prefix]
        for summary_key in summary_keys:
            if len(summary_key) == 2:
                log_label, dict_key = summary_key
                value_str = (
                    self._format_log_scalar(out_dict[dict_key])
                    if dict_key in out_dict
                    else None
                )
            elif len(summary_key) == 3:
                log_label, dict_key, loss_key = summary_key
                value_str = self._format_weighted_loss_for_summary(
                    out_dict, dict_key, loss_key
                )
            else:
                raise ValueError(f"Unsupported summary key spec: {summary_key}")
            if value_str is None:
                continue
            summary_parts.append(f"{log_label}={value_str}")
        return " | ".join(summary_parts)

    def _get_train_summary_line(self, out_dict: Mapping[str, Any]) -> str:
        return self._build_compact_summary_line(
            "TRAIN_SUMMARY",
            out_dict,
            [
                ("epoch", "Trainer/epoch"),
                ("step", "Trainer/steps_train"),
                ("all_loss", "Losses/train_all_loss"),
                ("main_weighted_loss", "Losses/train_all_main_weighted_loss"),
                ("aux_weighted_loss", "Losses/train_all_aux_weighted_loss"),
                ("batch_scale", "Losses/train_all_batch_scale"),
                ("sam_is", "Meters_train/train_all/sam_matching_is/is_mean"),
                ("sam_is_over_gt", "Meters_train/train_all/sam_matching_is/is_over_gt_mean"),
                ("sam_is_images", "Meters_train/train_all/sam_matching_is/num_images_compared"),
                ("pdc_cls", "Losses/train_all_loss_pdc_cls", "loss_pdc_cls"),
                ("pdc_point", "Losses/train_all_loss_pdc_point", "loss_pdc_point"),
                ("pdc_cls_pos", "Losses/train_all_pdc_cls_pos"),
                ("pdc_cls_neg", "Losses/train_all_pdc_cls_neg"),
                (
                    "pdc_unmatched_gt_before_repair",
                    "Losses/train_all_pdc_unmatched_gt_before_repair",
                ),
                ("rsc_cls", "Losses/train_all_loss_rsc_cls", "loss_rsc_cls"),
                ("rsc_point", "Losses/train_all_loss_rsc_point", "loss_rsc_point"),
                ("rsc_box", "Losses/train_all_loss_rsc_box", "loss_rsc_box"),
                ("rsc_giou", "Losses/train_all_loss_rsc_giou", "loss_rsc_giou"),
                ("ce", "Losses/train_all_loss_ce", "loss_ce"),
                ("point", "Losses/train_all_loss_point", "loss_point"),
                ("bbox", "Losses/train_all_loss_bbox", "loss_bbox"),
                ("giou", "Losses/train_all_loss_giou", "loss_giou"),
                ("presence", "Losses/train_all_presence_loss", "presence_loss"),
            ],
        )

    def _get_val_summary_line(self, out_dict: Mapping[str, Any]) -> str:
        return self._build_compact_summary_line(
            "VAL_SUMMARY",
            out_dict,
            [
                ("epoch", "Trainer/epoch"),
                ("step", "Trainer/steps_val"),
                ("all_loss", "Losses/val_all_loss"),
                ("main_weighted_loss", "Losses/val_all_main_weighted_loss"),
                ("aux_weighted_loss", "Losses/val_all_aux_weighted_loss"),
                ("batch_scale", "Losses/val_all_batch_scale"),
                ("sam_is", "Meters_train/val_all/sam_matching_is/is_mean"),
                ("sam_is_over_gt", "Meters_train/val_all/sam_matching_is/is_over_gt_mean"),
                ("sam_is_images", "Meters_train/val_all/sam_matching_is/num_images_compared"),
                ("pdc_cls", "Losses/val_all_loss_pdc_cls", "loss_pdc_cls"),
                ("pdc_point", "Losses/val_all_loss_pdc_point", "loss_pdc_point"),
                ("pdc_cls_pos", "Losses/val_all_pdc_cls_pos"),
                ("pdc_cls_neg", "Losses/val_all_pdc_cls_neg"),
                (
                    "pdc_unmatched_gt_before_repair",
                    "Losses/val_all_pdc_unmatched_gt_before_repair",
                ),
                ("rsc_cls", "Losses/val_all_loss_rsc_cls", "loss_rsc_cls"),
                ("rsc_point", "Losses/val_all_loss_rsc_point", "loss_rsc_point"),
                ("rsc_box", "Losses/val_all_loss_rsc_box", "loss_rsc_box"),
                ("rsc_giou", "Losses/val_all_loss_rsc_giou", "loss_rsc_giou"),
                ("ce", "Losses/val_all_loss_ce", "loss_ce"),
                ("point", "Losses/val_all_loss_point", "loss_point"),
                ("bbox", "Losses/val_all_loss_bbox", "loss_bbox"),
                ("giou", "Losses/val_all_loss_giou", "loss_giou"),
                ("presence", "Losses/val_all_presence_loss", "presence_loss"),
                ("mae", "Meters_train/val_all/counting/mae"),
                ("mse", "Meters_train/val_all/counting/mse"),
                ("gt_avg", "Meters_train/val_all/counting/gt_count_avg"),
                ("rsc_avg", "Meters_train/val_all/counting/rsc_count_avg"),
                ("pdc_pred_avg", "Meters_train/val_all/counting/pdc_count_avg"),
                ("pdc_score_max", "Meters_train/val_all/counting/pdc_score_max"),
                ("pdc_score_min", "Meters_train/val_all/counting/pdc_score_min"),
                ("pdc_score_mean", "Meters_train/val_all/counting/pdc_score_mean"),
                ("pdc_score_median", "Meters_train/val_all/counting/pdc_score_median"),
                ("rsc_mae", "Meters_train/val_all/counting/rsc_count_mae"),
                ("rsc_mse", "Meters_train/val_all/counting/rsc_count_mse"),
                ("pdc_mae", "Meters_train/val_all/counting/pdc_count_mae"),
                ("pdc_mse", "Meters_train/val_all/counting/pdc_count_mse"),
            ],
        )

    def _get_threshold_sweep_summary_line(
        self, out_dict: Mapping[str, Any], *, branch_name: str, metric_prefix: str
    ) -> str:
        thresholds = [0.05 * i for i in range(1, 11)] + [
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
        summary_parts = [branch_name, f"epoch={self.epoch}"]
        for threshold in thresholds:
            threshold_label = f"{threshold:.2f}".rstrip("0").rstrip(".")
            threshold_key = threshold_label.replace(".", "_")
            mae_key = (
                f"Meters_train/val_all/counting/{metric_prefix}_mae_thr_{threshold_key}"
            )
            mse_key = (
                f"Meters_train/val_all/counting/{metric_prefix}_mse_thr_{threshold_key}"
            )
            if mae_key not in out_dict or mse_key not in out_dict:
                continue
            summary_parts.append(
                f"{threshold_label}:mae={self._format_log_scalar(out_dict[mae_key])},"
                f"mse={self._format_log_scalar(out_dict[mse_key])}"
            )
        return " | ".join(summary_parts)

    def _get_point_soft_ce_r_topk_line(
        self, out_dict: Mapping[str, Any], *, phase: str, topk: int = 200
    ) -> Optional[str]:
        prefix = f"Losses/{phase}_all_point_soft_ce_r_rank_"
        values = []
        for idx in range(1, topk + 1):
            key = f"{prefix}{idx:02d}"
            if key not in out_dict:
                continue
            values.append(self._format_log_scalar(out_dict[key]))
        if not values:
            return None
        branch = "VAL_POINT_SOFT_CE_R_TOP200" if phase == "val" else "TRAIN_POINT_SOFT_CE_R_TOP200"
        return " | ".join([branch, f"epoch={self.epoch}", ",".join(values)])

    def _get_train_progress_line(
        self,
        *,
        data_iter: int,
        iters_per_epoch: int,
        loss_mts: Mapping[str, AverageMeter],
    ) -> str:
        # 轻量 heartbeat：长训时只保留当前 epoch / step 位置和一个短的 loss 摘要。
        all_loss_meter = loss_mts.get("Losses/train_all_loss")
        loss_value = all_loss_meter.avg if all_loss_meter is not None else float("nan")
        return " | ".join(
            [
                "TRAIN_PROGRESS",
                f"epoch={self.epoch}",
                f"step={data_iter + 1}/{iters_per_epoch}",
                f"all_loss={self._format_log_scalar(loss_value)}",
                f"lr={self._get_optimizer_lr_summary()}",
            ]
        )

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)
        self._active_val_visual_image_ids = set()

    def _get_stage_output_dir(self) -> str:
        return os.path.dirname(os.path.normpath(self.logging_conf.log_dir))

    def _should_visualize_val_epoch(self) -> bool:
        return (
            self.visualize_val_every_n_epochs > 0
            and self.epoch >= 0
            and (self.epoch + 1) % self.visualize_val_every_n_epochs == 0
        )

    def _canonicalize_visualize_val_outputs(self, output_names) -> set[str]:
        canonical_outputs = set()
        invalid_outputs = []
        for output_name in output_names:
            output_name = str(output_name).lower()
            canonical = VISUAL_OUTPUT_ALIASES.get(output_name)
            if canonical is None:
                invalid_outputs.append(output_name)
                continue
            canonical_outputs.add(canonical)
        if invalid_outputs:
            valid = ", ".join(sorted(VISUAL_OUTPUT_ALIASES))
            invalid = ", ".join(sorted(invalid_outputs))
            raise ValueError(
                f"Unknown visualize_val_outputs value(s): {invalid}. "
                f"Valid values are: {valid}."
            )
        return canonical_outputs

    def _get_visualization_dir_name(self, canonical_output: str) -> str:
        legacy_name = VISUAL_OUTPUT_LEGACY_NAMES[canonical_output]
        if canonical_output in self.requested_visualize_val_outputs:
            return f"vis_{canonical_output}"
        if legacy_name in self.requested_visualize_val_outputs:
            return f"vis_{legacy_name}"
        return f"vis_{canonical_output}"

    def _get_val_visualization_dirs(self) -> Dict[str, str]:
        epoch_label = f"epoch_{self.epoch + 1:04d}"
        stage_dir = self._get_stage_output_dir()
        vis_dirs = {}
        if "pdc" in self.visualize_val_outputs:
            vis_dirs["pdc"] = os.path.join(
                stage_dir, self._get_visualization_dir_name("pdc"), epoch_label
            )
        if "rsc" in self.visualize_val_outputs:
            vis_dirs["rsc"] = os.path.join(
                stage_dir, self._get_visualization_dir_name("rsc"), epoch_label
            )
        if "ccf" in self.visualize_val_outputs:
            vis_dirs["ccf"] = os.path.join(
                stage_dir, self._get_visualization_dir_name("ccf"), epoch_label
            )
        return vis_dirs

    def _prepare_val_visualization_dirs(self) -> None:
        self._active_val_visual_image_ids = set()
        for vis_dir in self._get_val_visualization_dirs().values():
            makedir(vis_dir)

    def _get_counting_postprocessor(self):
        if self.meters is None:
            return None
        for _, key_meters in self.meters.get(Phase.VAL, {}).items():
            if key_meters is None:
                continue
            for _, meter in key_meters.items():
                postprocessor = getattr(meter, "postprocessor", None)
                if postprocessor is not None:
                    return postprocessor
        return None

    def _visualize_validation_outputs(
        self,
        *,
        batch: BatchedDatapoint,
        find_stages,
    ) -> None:
        if not self._should_visualize_val_epoch():
            return
        if batch.raw_images is None or len(batch.raw_images) == 0:
            return

        postprocessor = self._get_counting_postprocessor()
        if postprocessor is None:
            return

        vis_dirs = self._get_val_visualization_dirs()
        for stage_outputs, stage_meta, stage_inputs, stage_targets in zip(
            find_stages, batch.find_metadatas, batch.find_inputs, batch.find_targets
        ):
            stage_results = postprocessor(
                outputs=stage_outputs,
                original_sizes=stage_meta.original_size,
                processed_scales_xy=getattr(stage_meta, "processed_scale_xy", None),
                processed_offsets_xy=getattr(stage_meta, "processed_offset_xy", None),
            )
            image_ids = stage_meta.original_image_id.detach().cpu().tolist()
            img_indices = stage_inputs.img_ids.detach().cpu().tolist()
            text_ids = stage_inputs.text_ids.detach().cpu().tolist()
            gt_counts = stage_targets.num_boxes.detach().cpu().tolist()

            for image_id, img_idx, text_id, gt_count, stage_result in zip(
                image_ids, img_indices, text_ids, gt_counts, stage_results
            ):
                image_id = int(image_id)
                img_idx = int(img_idx)
                text_id = int(text_id)
                gt_count = int(gt_count)
                if image_id in self._active_val_visual_image_ids:
                    continue
                self._active_val_visual_image_ids.add(image_id)

                file_name = f"{image_id:06d}.png"
                raw_image = batch.raw_images[img_idx]
                class_name = (
                    batch.find_text_batch[text_id]
                    if 0 <= text_id < len(batch.find_text_batch)
                    else ""
                )
                if "pdc" in vis_dirs:
                    save_pdc_visualization(
                        raw_image,
                        stage_result["raw_pdc_points"],
                        os.path.join(vis_dirs["pdc"], file_name),
                        gt_count=gt_count,
                        pred_count=int(stage_result["pdc_count"]),
                        class_name=class_name,
                    )
                if "rsc" in vis_dirs:
                    save_rsc_visualization(
                        raw_image,
                        stage_result["raw_rsc_boxes"],
                        stage_result["raw_rsc_scores"],
                        os.path.join(vis_dirs["rsc"], file_name),
                        gt_count=gt_count,
                        pred_count=int(stage_result["rsc_count"]),
                        class_name=class_name,
                    )
                if "ccf" in vis_dirs:
                    save_ccf_visualization(
                        raw_image,
                        stage_result["kept_rsc_boxes"],
                        stage_result["kept_rsc_scores"],
                        stage_result["kept_pdc_points"],
                        stage_result["removed_pdc_points"],
                        os.path.join(vis_dirs["ccf"], file_name),
                        gt_count=gt_count,
                        pred_count=int(stage_result["pred_count"]),
                        class_name=class_name,
                    )
    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            static_graph=distributed_conf.static_graph,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        if self.skip_saving_ckpts:
            logging.info(
                "skip_saving_ckpts is set to True. So, no checkpoints have been saved."
            )
            return
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            # Regular training checkpoints are stored epoch-by-epoch only.
            checkpoint_names = [f"epoch{int(epoch)}"]

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            # 这里恢复的是完整训练 checkpoint，需要显式关闭 weights_only 模式。
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        skip_epoch = self.skip_first_val and epoch == 0
        return (
            epoch % self.val_epoch_freq == 0
            and epoch < self.max_epochs - 1
            and not skip_epoch
        )

    def _find_loss(self, key: str):
        if key in self.loss:
            return self.loss[key]

        assert key != "all", "Loss must be specified for key='all'"
        assert (
            "default" in self.loss
        ), f"Key {key} not found in losss, and no default provided"
        return self.loss["default"]

    def _find_meter(self, phase: str, key: str):
        if key in self.meters[phase]:
            return self.meters[phase][key]

        for cand_key, meter in self.meters[phase].items():
            if fnmatch.fnmatch(key, cand_key):
                return meter
        return None

    def _step(
        self,
        batch: BatchedDatapoint,
        model: nn.Module,
        phase: str,
    ):
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)
        matcher = _get_debug_matcher(model)
        if matcher is not None and hasattr(matcher, "set_debug_context"):
            matcher.set_debug_context(
                epoch=int(self.epoch),
                phase=phase,
                step=int(self.steps[phase]),
            )

        find_stages = model(batch)
        self._log_sam_det_forward_finite_probe(find_stages=find_stages, phase=phase)
        self._register_sam_det_backward_finite_hooks(
            find_stages=find_stages, phase=phase
        )
        find_targets = [
            unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets
        ]
        batch_size = len(batch.img_batch)
        loss = self._find_loss(key)(find_stages, find_targets)

        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase not in self.meters:
            return ret_tuple

        meters_dict = self._find_meter(phase, key)
        if meters_dict is None:
            return ret_tuple
        if meters_dict is not None:
            for _, meter in meters_dict.items():
                meter.update(
                    find_stages=find_stages,
                    find_metadatas=batch.find_metadatas,
                    model=model,
                    batch=batch,
                    key=key,
                )
            if phase == Phase.VAL:
                self._visualize_validation_outputs(
                    batch=batch,
                    find_stages=find_stages,
                )
            # Cleanup memory
            if isinstance(find_stages, SAM3Output):
                for fs in find_stages:
                    for k in list(fs.keys()):
                        del fs[k]

        return ret_tuple

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if (
                    not self.skip_resume_previous_val
                    and self.is_intermediate_val_epoch(self.epoch - 1)
                ):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
                elif self.skip_resume_previous_val:
                    logging.info("Skipping previous val epoch on resume")
            elif self.validate_before_train:
                logging.info(
                    "Running validation before the first optimizer step with epoch=-1"
                )
                original_epoch = self.epoch
                self.epoch = -1
                self.run_val(dataloader_epoch=0)
                self.epoch = original_epoch
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            if self.val_epoch_override is None:
                self.run_val()
            else:
                original_epoch = self.epoch
                self.epoch = int(self.val_epoch_override)
                dataloader_epoch = 0 if self.epoch < 0 else self.epoch
                self.run_val(dataloader_epoch=dataloader_epoch)
                self.epoch = original_epoch
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))
            self._log_dataset_transforms("val", self.val_dataset)

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)
            self._log_dataset_transforms("train", self.train_dataset)

    def _format_transform_log_value(self, value):
        simple_types = (int, float, bool, str, type(None))
        if isinstance(value, simple_types):
            return repr(value)
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, simple_types) for item in value
        ):
            return repr(list(value))
        return f"<{type(value).__name__}>"

    def _format_transform_for_log(self, transform, indent=0):
        prefix = "  " * indent
        attrs = []
        child_transforms = None
        for key, value in sorted(vars(transform).items()):
            if key in {"transforms", "_transforms"} and isinstance(value, (list, tuple)):
                child_transforms = value
                continue
            if key.startswith("_"):
                continue
            attrs.append(f"{key}={self._format_transform_log_value(value)}")

        header = f"{prefix}{transform.__class__.__name__}"
        if attrs:
            header += f"({', '.join(attrs)})"

        if not child_transforms:
            return header

        lines = [header]
        for child_transform in child_transforms:
            lines.append(self._format_transform_for_log(child_transform, indent + 1))
        return "\n".join(lines)

    def _log_dataset_transforms(self, split_name, torch_dataset):
        dataset = getattr(torch_dataset, "dataset", None)
        transforms = getattr(dataset, "_transforms", None)
        if transforms is None:
            transforms = getattr(dataset, "transforms", None)

        if transforms is None:
            logging.info(
                "%s_DATASET_TRANSFORMS unavailable", str(split_name).upper()
            )
            return

        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]

        formatted = "\n".join(
            self._format_transform_for_log(transform, indent=1)
            for transform in transforms
        )
        logging.info("%s_DATASET_TRANSFORMS\n%s", str(split_name).upper(), formatted)

    def run_train(self):
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()
                if torch.cuda.is_available() and self.empty_gpu_mem_cache_after_eval:
                    # release memory buffers held by the model during eval (which typically
                    # involves a lot more frames in video grounding that during training)
                    torch.cuda.empty_cache()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self, dataloader_epoch: Optional[int] = None):
        if not self.val_dataset:
            return

        loader_epoch = int(self.epoch if dataloader_epoch is None else dataloader_epoch)
        dataloader = self.val_dataset.get_loader(epoch=loader_epoch)
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()
            matcher = _get_debug_matcher(model)
            if matcher is not None and hasattr(matcher, "reset_debug_stats"):
                matcher.reset_debug_stats()
                if hasattr(matcher, "set_debug_context"):
                    matcher.set_debug_context(
                        epoch=int(self.epoch),
                        phase=phase,
                        step=int(self.steps[phase]),
                    )

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self._emit_log_blank_line()
        if self._should_visualize_val_epoch():
            self._prepare_val_visualization_dirs()

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # batch = batch.to(self.device, non_blocking=True)

            # compute output
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        if loss_key in loss_mts:
                            loss_mts[loss_key].update(
                                self._as_python_float(loss), batch_size
                            )

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(
                                self._as_python_float(v), batch_size
                            )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # Log progress meters.
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            if data_iter % 10 == 0:
                dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()
            self._log_matcher_debug_summary(model=model, phase=phase)

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        self._emit_log_blank_line()
        logging.info(self._get_val_summary_line(out_dict))
        logging.info(
            self._get_threshold_sweep_summary_line(
                out_dict,
                branch_name="VAL_RSC_THRESH",
                metric_prefix="rsc_count",
            )
        )
        logging.info(
            self._get_threshold_sweep_summary_line(
                out_dict,
                branch_name="VAL_PDC_THRESH",
                metric_prefix="pdc_count",
            )
        )
        r_topk_line = self._get_point_soft_ce_r_topk_line(out_dict, phase="val")
        if r_topk_line is not None:
            logging.info(r_topk_line)
        if self._should_visualize_val_epoch():
            dist.barrier()
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        self._apply_training_stage()
        self._log_training_stage_state()
        self._reset_branch_numeric_debug()
        matcher = _get_debug_matcher(self.model)
        if matcher is not None and hasattr(matcher, "reset_debug_stats"):
            matcher.reset_debug_stats()
            if hasattr(matcher, "set_debug_context"):
                matcher.set_debug_context(
                    epoch=int(self.epoch),
                    phase=phase,
                    step=int(self.steps[phase]),
                )
        self._emit_log_blank_line()
        end = time.time()
        heartbeat_freq = (
            self.logging_conf.heartbeat_freq
            if self.logging_conf.heartbeat_freq is not None
            else self.logging_conf.log_freq
        )

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            # batch = batch.to(
            #     self.device, non_blocking=True
            # )  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self._log_branch_numeric_debug(phase=phase)
                    parameter_groups = self._get_gradient_clip_parameter_groups()
                    if parameter_groups is not None:
                        self.gradient_clipper(parameter_groups=parameter_groups)
                    else:
                        self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

                should_log_heartbeat = (
                    heartbeat_freq > 0
                    and (
                        (data_iter + 1) % heartbeat_freq == 0
                        or (data_iter + 1) == iters_per_epoch
                    )
                )
                if should_log_heartbeat:
                    logging.info(
                        self._get_train_progress_line(
                            data_iter=data_iter,
                            iters_per_epoch=iters_per_epoch,
                            loss_mts=loss_mts,
                        )
                    )

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        self._emit_log_blank_line()
        self._log_matcher_debug_summary(model=self.model, phase=phase)
        self._log_branch_numeric_debug_summary(phase=phase)
        logging.info(self._get_train_summary_line(out_dict))
        self._reset_meters([phase])
        return out_dict

    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _log_matcher_debug_summary(self, model, phase):
        matcher = _get_debug_matcher(model)
        logging.info(
            "AMP_SCALER | epoch=%s phase=%s scale=%.6e",
            self.epoch,
            phase,
            float(self.scaler.get_scale()),
        )
        if matcher is None or not hasattr(matcher, "get_debug_stats"):
            return
        stats = matcher.get_debug_stats()
        logging.info(
            "DensePointHungarianMatcher summary | epoch=%s phase=%s calls=%s nonfinite_calls=%s last_nonfinite=%s",
            self.epoch,
            phase,
            stats.get("calls"),
            stats.get("nonfinite_calls"),
            stats.get("last_nonfinite"),
        )

    def _run_step(
        self,
        batch: BatchedDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)

        if self.gradient_accumulation_steps > 1:
            assert isinstance(
                batch, list
            ), f"Expected a list of batches, got {type(batch)}"
            assert (
                len(batch) == self.gradient_accumulation_steps
            ), f"Expected {self.gradient_accumulation_steps} batches, got {len(batch)}"
            accum_steps = len(batch)
        else:
            accum_steps = 1
            batch = [batch]

        for i, chunked_batch in enumerate(batch):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )
            with ddp_context:
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    loss_dict, batch_size, extra_losses = self._step(
                        chunked_batch,
                        self.model,
                        phase,
                    )

                assert len(loss_dict) == 1
                loss_key, loss = loss_dict.popitem()

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    if raise_on_error:
                        raise FloatingPointError(error_msg)
                    else:
                        return

                self._capture_step_loss_snapshot(
                    phase=phase,
                    loss_key=loss_key,
                    loss=loss,
                    extra_losses=extra_losses,
                )
                self.scaler.scale(loss).backward()
                loss_mts[loss_key].update(self._as_python_float(loss), batch_size)
                for extra_loss_key, extra_loss in extra_losses.items():
                    if extra_loss_key not in extra_loss_mts:
                        extra_loss_mts[extra_loss_key] = AverageMeter(
                            extra_loss_key, self.device, ":.2e"
                        )
                    extra_loss_mts[extra_loss_key].update(
                        self._as_python_float(extra_loss), batch_size
                    )

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        out_dict = {}
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

        return out_dict

    def _log_timers(self, phase):
        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                if "default" not in loss_keys:
                    for k in val_keys:
                        assert (
                            k in loss_keys
                        ), f"Error: key {k} is not defined in the losses, and no default is set"

    def _setup_components(self):
        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)
            self._filter_stage_specific_losses()

        self._apply_training_stage()
        self._log_training_stage_state()
        print_model_summary(self.model)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self._get_branch_lr_options(),
            self.optim_conf.param_group_modifiers,
            param_allowlist=self._get_optimizer_param_allowlist(),
        )

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info(f"MODEL_TYPE {type(model)}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
