"""One-way IR-to-VIS deformable enhancement plus original IR addition."""

from typing import Optional

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder, )
from mmdet.registry import MODELS


@MODELS.register_module()
class IRToVISDeformableAddIRFusion(nn.Module):
    """Return IR-enhanced VIS plus unchanged post-neck IR tokens."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError('num_feature_levels must be positive.')
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.vis_norm = nn.LayerNorm(embed_dims)
        self.ir_norm = nn.LayerNorm(embed_dims)
        self.vis_from_ir = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)

    def init_weights(self) -> None:
        self.vis_from_ir.init_weights()

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Optional[Tensor],
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        del text_feat, text_token_mask
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes)
        reference_points = \
            DeformableDetrTransformerEncoder.get_encoder_reference_points(
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                device=vis_feat.device)
        vis_enhanced = self.vis_from_ir(
            query=self.vis_norm(vis_feat),
            value=self.ir_norm(ir_feat),
            identity=vis_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        fused = vis_enhanced + ir_feat
        if feat_mask is not None:
            fused = fused.masked_fill(
                feat_mask[..., None].to(torch.bool), 0)
        return fused

    def _validate_inputs(self, vis_feat: Tensor, ir_feat: Tensor,
                         spatial_shapes: Tensor) -> None:
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS/IR shapes differ: {vis_feat.shape} vs {ir_feat.shape}.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError('Features must have shape [B, N, embed_dims].')
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} feature levels, got '
                f'{spatial_shapes.size(0)}.')
        token_count = int(spatial_shapes.prod(dim=1).sum().item())
        if token_count != vis_feat.size(1):
            raise ValueError(
                f'Expected {token_count} flattened tokens, got '
                f'{vis_feat.size(1)}.')
