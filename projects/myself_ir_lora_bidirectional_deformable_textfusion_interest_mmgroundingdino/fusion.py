"""Bidirectional deformable enhancement followed by text-gated fusion."""

from typing import Optional

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder, )
from mmdet.registry import MODELS
from projects.myself_ir_lora_textfusion_interest_mmgroundingdino.fusion import (
    TextFusionInterestFusion, )


@MODELS.register_module()
class BidirectionalDeformableTextFusionInterestFusion(
        TextFusionInterestFusion):
    """Enhance both modalities, then fuse them with a text dynamic gate."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 gate_hidden_dims: int = 256,
                 gate_init_bias: float = 2.0) -> None:
        super().__init__(
            fusion_mode='dynamic_gate',
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            num_points=num_points,
            gate_hidden_dims=gate_hidden_dims,
            gate_init_bias=gate_init_bias)
        attention_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        # ``vis_from_ir`` is created by the validated parent implementation.
        # The reverse direction intentionally owns independent parameters.
        self.ir_from_vis = MultiScaleDeformableAttention(**attention_cfg)

    def init_weights(self) -> None:
        self.vis_from_ir.init_weights()
        self.ir_from_vis.init_weights()
        nn.init.eye_(self.vis_proj.weight)
        nn.init.eye_(self.ir_proj.weight)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes)
        if text_token_mask is None:
            raise ValueError('text_token_mask is required.')

        reference_points = \
            DeformableDetrTransformerEncoder.get_encoder_reference_points(
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                device=vis_feat.device)
        vis_norm = self.vis_norm(vis_feat)
        ir_norm = self.ir_norm(ir_feat)
        vis_enhanced = self.vis_from_ir(
            query=vis_norm,
            value=ir_norm,
            identity=vis_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        ir_enhanced = self.ir_from_vis(
            query=ir_norm,
            value=vis_norm,
            identity=ir_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        text_global = self._global_text(text_feat, text_token_mask)
        text_spatial = text_global[:, None, :].expand_as(ir_enhanced)
        gate_input = torch.cat(
            [vis_enhanced, ir_enhanced, text_spatial], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        alpha = self.fusion_alpha.to(ir_enhanced.dtype)
        fused = (
            self.vis_proj(vis_enhanced) +
            alpha * gate * self.ir_proj(ir_enhanced))

        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].bool(), 0)
        if self.training:
            # Keep the same raw-modality targets as Fusion-only so adding the
            # bidirectional block is the sole ablation variable.
            self._interest_cache = dict(
                vis=vis_feat,
                ir=ir_feat,
                fused=fused,
                spatial_shapes=spatial_shapes,
                feat_mask=feat_mask,
                gate=gate)
        return fused
