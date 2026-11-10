"""Bidirectional deformable VIS/IR attention followed by direct addition."""

from typing import Optional

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder,
)
from mmdet.registry import MODELS


@MODELS.register_module()
class BidirectionalDeformableSumFusion(nn.Module):
    """Enhance both modalities with the other modality, then add them.

    This is the bidirectional deformable cross-modal mechanism from the v2
    no-expectation experiment, isolated from its direct text-attention stage:

    ``VIS' = VIS + DeformAttn(query=VIS, value=IR)``
    ``IR'  = IR  + DeformAttn(query=IR,  value=VIS)``
    ``fused = VIS' + IR'``
    """

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
        attention_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        # The two directions intentionally have independent offsets, weights,
        # value projections and output projections.
        self.vis_from_ir = MultiScaleDeformableAttention(**attention_cfg)
        self.ir_from_vis = MultiScaleDeformableAttention(**attention_cfg)

    def init_weights(self) -> None:
        """Use MMCV's standard initialization in both directions."""
        self.vis_from_ir.init_weights()
        self.ir_from_vis.init_weights()

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
        # This ablation migrates only image-to-image cross-attention. Text is
        # consumed later by the unchanged Grounding DINO encoder.
        del text_feat, text_token_mask
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes, feat_mask)

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

        fused = vis_enhanced + ir_enhanced
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].bool(), 0)
        return fused

    def _validate_inputs(self, vis_feat: Tensor, ir_feat: Tensor,
                         spatial_shapes: Tensor,
                         feat_mask: Optional[Tensor]) -> None:
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS/IR shapes differ: {vis_feat.shape} vs {ir_feat.shape}.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError('Features must have shape [B, N, embed_dims].')
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} levels, got '
                f'{spatial_shapes.size(0)}.')
        token_count = int(spatial_shapes.prod(dim=1).sum().item())
        if token_count != vis_feat.size(1):
            raise ValueError(
                f'Expected {token_count} flattened tokens, got '
                f'{vis_feat.size(1)}.')
        if feat_mask is not None and feat_mask.shape != vis_feat.shape[:2]:
            raise ValueError('feat_mask must have shape [B, N].')
