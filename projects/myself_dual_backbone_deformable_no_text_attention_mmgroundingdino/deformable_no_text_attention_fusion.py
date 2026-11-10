"""Bidirectional deformable VIS/IR fusion without text attention."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_mmgroundingdino.deformable_multimodal_fusion import (  # noqa: E501
    BidirectionalDeformableCrossModalAttention, )


@MODELS.register_module()
class DeformableNoTextAttentionFusionBlock(nn.Module):
    """Fuse VIS and IR using only bidirectional deformable attention.

    ``text_feat`` and ``text_token_mask`` remain in the forward signature so
    this ablation can reuse the detector pipeline unchanged.  They are
    intentionally not consumed here; Grounding DINO's native encoder and
    decoder still perform their normal vision-language interaction after this
    visual fusion stage.
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
        self.cross_modal_attention = BidirectionalDeformableCrossModalAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            num_points=num_points)

    def init_weights(self) -> None:
        """Initialize both deformable cross-attention directions."""
        self.cross_modal_attention.init_weights()

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
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS and IR feature shapes must match, got {vis_feat.shape} '
                f'and {ir_feat.shape}.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError(
                'VIS and IR features must have shape [B, N, embed_dims].')
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError(
                'spatial_shapes must have shape [num_levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} feature levels, but got '
                f'{spatial_shapes.size(0)}.')
        flattened_length = int(spatial_shapes.prod(dim=1).sum().item())
        if flattened_length != vis_feat.size(1):
            raise ValueError(
                'The sum of flattened spatial shapes must equal the visual '
                f'token length, but got {flattened_length} and '
                f'{vis_feat.size(1)}.')

        vis_enhanced, ir_enhanced = self.cross_modal_attention(
            vis=vis_feat,
            ir=ir_feat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask)
        fused = vis_enhanced + ir_enhanced
        if feat_mask is not None:
            fused = fused.masked_fill(
                feat_mask[..., None].to(torch.bool), 0)
        return fused
