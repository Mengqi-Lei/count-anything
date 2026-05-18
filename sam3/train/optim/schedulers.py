# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math


class InverseSquareRootParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: int,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.timescale = timescale

    def __call__(self, step: int, where: float):
        lr = self.base_lr

        if where > 0:
            total_steps = step / where
            progress = (step - self.warmup_steps) / float(
                total_steps - self.warmup_steps
            )
            progress = max(min(progress, 1), 0)
        else:
            progress = 0
            total_steps = 1

        shift = self.timescale - self.warmup_steps
        if self.warmup_steps < step:
            lr = lr / math.sqrt((step + shift) / self.timescale)

        if self.warmup_steps:
            lr = lr * min(1.0, step / self.warmup_steps)
        if self.cooldown_steps:
            lr = lr * min(1.0, (total_steps - step) / self.cooldown_steps)

        return lr


class EpochStepParamScheduler:
    def __init__(
        self,
        base_lr: float,
        max_epochs: int,
        drop_epochs,
        gamma: float = 0.1,
    ):
        # 这个 scheduler 的目标不是“按 iteration 数衰减”，
        # 而是把 StepLR 的“按 epoch 边界跳变”语义映射到当前 trainer 的
        # `where/step` 接口里，避免概念和实现不一致。
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.drop_epochs = sorted(drop_epochs or [])
        self.gamma = gamma

    def __call__(self, step: int, where: float):
        del step
        if self.max_epochs <= 0:
            return self.base_lr

        current_epoch = max(0.0, min(where * self.max_epochs, float(self.max_epochs)))
        num_drops = sum(drop_epoch <= current_epoch for drop_epoch in self.drop_epochs)
        return self.base_lr * (self.gamma ** num_drops)


class EpochCosineParamScheduler:
    def __init__(
        self,
        base_lr: float,
        max_epochs: int,
        min_lr: float = 0.0,
        min_lr_ratio: float | None = None,
        start_epoch: float = 0.0,
        end_epoch: float | None = None,
    ):
        self.base_lr = base_lr
        self.max_epochs = max_epochs
        self.start_epoch = float(start_epoch)
        self.end_epoch = float(max_epochs if end_epoch is None else end_epoch)
        if min_lr_ratio is not None:
            self.min_lr = base_lr * float(min_lr_ratio)
        else:
            self.min_lr = float(min_lr)

    def __call__(self, step: int, where: float):
        del step
        if self.max_epochs <= 0:
            return self.base_lr

        current_epoch = max(0.0, min(where * self.max_epochs, float(self.max_epochs)))
        if current_epoch <= self.start_epoch:
            return self.base_lr
        if self.end_epoch <= self.start_epoch:
            return self.min_lr

        progress = (current_epoch - self.start_epoch) / (
            self.end_epoch - self.start_epoch
        )
        progress = max(0.0, min(progress, 1.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine
