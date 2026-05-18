"""Public CountAnything model builder."""

import torch
from sam3.model_builder import build_sam3_image_model


def build_count_anything_model(
    *,
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_pdc_branch=True,
    pdc_row=2,
    pdc_line=2,
    pdc_use_feature_adapter=True,
    pdc_feature_adapter_num_blocks=3,
    pdc_feature_adapter_use_coordconv=False,
    pdc_feature_adapter_gn_groups=32,
    compile=False,
    enable_encoder_cross_attn_lora=False,
    encoder_cross_attn_lora_r=8,
    encoder_cross_attn_lora_alpha=8,
    encoder_cross_attn_lora_dropout=0.0,
):
    """Build CountAnything using public RSC/PDC naming.

    The underlying implementation still delegates to the SAM3 image-model
    builder so upstream `sam3.pt` checkpoint keys stay valid.
    """

    return build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=eval_mode,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_pdc_branch=enable_pdc_branch,
        pdc_row=pdc_row,
        pdc_line=pdc_line,
        pdc_use_feature_adapter=pdc_use_feature_adapter,
        pdc_feature_adapter_num_blocks=pdc_feature_adapter_num_blocks,
        pdc_feature_adapter_use_coordconv=pdc_feature_adapter_use_coordconv,
        pdc_feature_adapter_gn_groups=pdc_feature_adapter_gn_groups,
        compile=compile,
        enable_encoder_cross_attn_lora=enable_encoder_cross_attn_lora,
        encoder_cross_attn_lora_r=encoder_cross_attn_lora_r,
        encoder_cross_attn_lora_alpha=encoder_cross_attn_lora_alpha,
        encoder_cross_attn_lora_dropout=encoder_cross_attn_lora_dropout,
    )


__all__ = ["build_count_anything_model"]
