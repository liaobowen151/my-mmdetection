"""IR-to-VIS enhancement, IR addition, and VIS-guided refinement."""

from typing import Optional

import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder, )
from mmdet.registry import MODELS


@MODELS.register_module()
class IRToVISDeformableAddIRVISRefineFusion(nn.Module):
    """Refine IR-added features by querying the original VIS features.

    The fusion is performed in two deformable-attention stages:

    1. Original VIS queries IR, with original VIS as the residual identity.
    2. The enhanced VIS is added to the original IR.
    3. The resulting fused feature queries the *original* VIS feature, with
       the fused feature as the residual identity.

    Text features are intentionally unused in this first ablation version.
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

        self.vis_query_norm = nn.LayerNorm(embed_dims)
        self.ir_value_norm = nn.LayerNorm(embed_dims)
        self.vis_from_ir = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)

        self.fused_query_norm = nn.LayerNorm(embed_dims)
        self.vis_value_norm = nn.LayerNorm(embed_dims)
        self.fused_from_vis = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)

    def init_weights(self) -> None:
        self.vis_from_ir.init_weights()
        self.fused_from_vis.init_weights()

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

        # Stage 1: selectively inject IR information into the VIS stream.
        vis_enhanced = self.vis_from_ir(
            query=self.vis_query_norm(vis_feat),
            value=self.ir_value_norm(ir_feat),
            identity=vis_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        # Keep the baseline's direct addition of the unchanged IR features.
        fused = vis_enhanced + ir_feat

        # Stage 2: use the IR-added feature as query and recover useful
        # information from the untouched original VIS feature bank.
        fused_refined = self.fused_from_vis(
            query=self.fused_query_norm(fused),
            value=self.vis_value_norm(vis_feat),
            identity=fused,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        if feat_mask is not None:
            fused_refined = fused_refined.masked_fill(
                feat_mask[..., None].to(bool), 0)
        return fused_refined

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
