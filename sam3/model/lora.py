# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""LoRA adapters for attention modules.

This file intentionally keeps the wrapped MultiheadAttention state-dict keys
unchanged, so pretrained SAM3 checkpoints can still load the base attention
weights with their original names.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRAMultiheadAttention(nn.MultiheadAttention):
    """A key-compatible LoRA wrapper for ``nn.MultiheadAttention``.

    The module exposes the same base parameters as ``nn.MultiheadAttention``
    (``in_proj_weight``, ``in_proj_bias``, ``out_proj.weight``, and
    ``out_proj.bias``) and adds low-rank adapters for Q/K/V/O projections.
    """

    def __init__(
        self,
        base_mha: nn.MultiheadAttention,
        *,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
    ) -> None:
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}.")
        if not getattr(base_mha, "_qkv_same_embed_dim", False):
            raise ValueError("LoRAMultiheadAttention only supports packed QKV weights.")

        factory_kwargs = {
            "device": base_mha.in_proj_weight.device,
            "dtype": base_mha.in_proj_weight.dtype,
        }
        super().__init__(
            embed_dim=base_mha.embed_dim,
            num_heads=base_mha.num_heads,
            dropout=base_mha.dropout,
            bias=base_mha.in_proj_bias is not None,
            add_bias_kv=base_mha.bias_k is not None,
            add_zero_attn=base_mha.add_zero_attn,
            kdim=base_mha.kdim,
            vdim=base_mha.vdim,
            batch_first=base_mha.batch_first,
            **factory_kwargs,
        )

        with torch.no_grad():
            self.in_proj_weight.copy_(base_mha.in_proj_weight)
            if self.in_proj_bias is not None and base_mha.in_proj_bias is not None:
                self.in_proj_bias.copy_(base_mha.in_proj_bias)
            self.out_proj.weight.copy_(base_mha.out_proj.weight)
            if self.out_proj.bias is not None and base_mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(base_mha.out_proj.bias)
            if self.bias_k is not None and base_mha.bias_k is not None:
                self.bias_k.copy_(base_mha.bias_k)
            if self.bias_v is not None and base_mha.bias_v is not None:
                self.bias_v.copy_(base_mha.bias_v)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(float(dropout))

        embed_dim = int(base_mha.embed_dim)
        self.lora_A_q = nn.Parameter(torch.empty(self.r, embed_dim, **factory_kwargs))
        self.lora_B_q = nn.Parameter(torch.empty(embed_dim, self.r, **factory_kwargs))
        self.lora_A_k = nn.Parameter(torch.empty(self.r, embed_dim, **factory_kwargs))
        self.lora_B_k = nn.Parameter(torch.empty(embed_dim, self.r, **factory_kwargs))
        self.lora_A_v = nn.Parameter(torch.empty(self.r, embed_dim, **factory_kwargs))
        self.lora_B_v = nn.Parameter(torch.empty(embed_dim, self.r, **factory_kwargs))
        self.lora_A_o = nn.Parameter(torch.empty(self.r, embed_dim, **factory_kwargs))
        self.lora_B_o = nn.Parameter(torch.empty(embed_dim, self.r, **factory_kwargs))

        self.reset_lora_parameters()
        self.freeze_base_parameters()

    @property
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        return (
            self.lora_A_q,
            self.lora_B_q,
            self.lora_A_k,
            self.lora_B_k,
            self.lora_A_v,
            self.lora_B_v,
            self.lora_A_o,
            self.lora_B_o,
        )

    def freeze_base_parameters(self) -> None:
        self.in_proj_weight.requires_grad_(False)
        if self.in_proj_bias is not None:
            self.in_proj_bias.requires_grad_(False)
        self.out_proj.weight.requires_grad_(False)
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad_(False)
        if self.bias_k is not None:
            self.bias_k.requires_grad_(False)
        if self.bias_v is not None:
            self.bias_v.requires_grad_(False)

    def reset_lora_parameters(self) -> None:
        # B starts at zero so enabling LoRA is initially behavior-preserving.
        for param in (self.lora_A_q, self.lora_A_k, self.lora_A_v, self.lora_A_o):
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in (self.lora_B_q, self.lora_B_k, self.lora_B_v, self.lora_B_o):
            nn.init.zeros_(param)

    def _lora_delta(self, lora_b: Tensor, lora_a: Tensor) -> Tensor:
        return self.scaling * (lora_b @ lora_a)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        # Match MultiheadAttentionWrapper: avoid materializing attention weights
        # unless a caller explicitly asks for them.
        need_weights = bool(need_weights)

        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        embed_dim = self.embed_dim
        w_q = self.in_proj_weight[:embed_dim] + self._lora_delta(
            self.lora_B_q, self.lora_A_q
        )
        w_k = self.in_proj_weight[embed_dim : 2 * embed_dim] + self._lora_delta(
            self.lora_B_k, self.lora_A_k
        )
        w_v = self.in_proj_weight[2 * embed_dim :] + self._lora_delta(
            self.lora_B_v, self.lora_A_v
        )
        in_proj_weight = torch.cat((w_q, w_k, w_v), dim=0)
        out_proj_weight = self.out_proj.weight + self._lora_delta(
            self.lora_B_o, self.lora_A_o
        )

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_output_weights


def wrap_mha_for_lora(
    base_mha: nn.MultiheadAttention,
    *,
    r: int = 8,
    alpha: int = 8,
    dropout: float = 0.0,
) -> LoRAMultiheadAttention:
    if isinstance(base_mha, LoRAMultiheadAttention):
        return base_mha
    return LoRAMultiheadAttention(base_mha, r=r, alpha=alpha, dropout=dropout)
