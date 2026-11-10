"""Text-conditioned dynamic fusion without IR-to-VIS deformable attention."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_ir_lora_textfusion_interest_mmgroundingdino.fusion import (
    TextFusionInterestFusion, )


@MODELS.register_module()
class TextFusionInterestFusionOnly(TextFusionInterestFusion):
    """Fuse raw VIS/IR tokens using the text-conditioned dynamic gate.

    This ablation intentionally removes the preceding IR-to-VIS deformable
    enhancement.  It keeps the projection, gate, schedule, diagnostics and
    interest-region loss identical to the full dynamic-gate experiment.
    """

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
        # The parent owns the validated interest-loss implementation.  Remove
        # its deformable block entirely so this ablation has no unused or
        # accidentally trainable IR-to-VIS attention parameters.
        del self.vis_from_ir
        del self.vis_norm
        del self.ir_norm

    def init_weights(self) -> None:
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
        del level_start_index, valid_ratios, feat_pos
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes)
        if text_token_mask is None:
            raise ValueError('text_token_mask is required.')

        text_global = self._global_text(text_feat, text_token_mask)
        text_spatial = text_global[:, None, :].expand_as(ir_feat)
        gate_input = torch.cat([vis_feat, ir_feat, text_spatial], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        alpha = self.fusion_alpha.to(ir_feat.dtype)
        fused = (
            self.vis_proj(vis_feat) +
            alpha * gate * self.ir_proj(ir_feat))

        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].bool(), 0)
        if self.training:
            self._interest_cache = dict(
                vis=vis_feat,
                ir=ir_feat,
                fused=fused,
                spatial_shapes=spatial_shapes,
                feat_mask=feat_mask,
                gate=gate)
        return fused
