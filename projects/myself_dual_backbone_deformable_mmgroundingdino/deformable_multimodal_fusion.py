"""Multimodal fusion with bidirectional deformable cross-attention."""

from typing import Optional

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder, )
from mmdet.registry import MODELS
from projects.myself_multimodal_mmgroundingdino.models.multimodal_fusion import (  # noqa: E501
    TextGuidedVisualBackProjection, )


class BidirectionalDeformableCrossModalAttention(nn.Module):
    """Perform multi-scale IR-to-VIS and VIS-to-IR deformable attention."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4) -> None:
        super().__init__()
        self.vis_norm = nn.LayerNorm(embed_dims)
        self.ir_norm = nn.LayerNorm(embed_dims)
        attention_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        self.vis_from_ir = MultiScaleDeformableAttention(**attention_cfg)
        self.ir_from_vis = MultiScaleDeformableAttention(**attention_cfg)

    def init_weights(self) -> None:
        """Use MMCV's standard deformable-attention initialization."""
        self.vis_from_ir.init_weights()
        self.ir_from_vis.init_weights()

    def forward(self,
                vis: Tensor,
                ir: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None) -> tuple:
        if vis.shape != ir.shape:
            raise ValueError(
                f'VIS and IR feature shapes must match, got {vis.shape} and '
                f'{ir.shape}.')

        reference_points = \
            DeformableDetrTransformerEncoder.get_encoder_reference_points(
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                device=vis.device)
        vis_norm = self.vis_norm(vis)
        ir_norm = self.ir_norm(ir)

        vis_enhanced = self.vis_from_ir(
            query=vis_norm,
            value=ir_norm,
            identity=vis,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        ir_enhanced = self.ir_from_vis(
            query=ir_norm,
            value=vis_norm,
            identity=ir,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        if feat_mask is not None:
            padding = feat_mask[..., None]
            vis_enhanced = vis_enhanced.masked_fill(padding, 0)
            ir_enhanced = ir_enhanced.masked_fill(padding, 0)
        return vis_enhanced, ir_enhanced


@MODELS.register_module()
class DeformableMultimodalFusionBlock(nn.Module):
    """Fuse VIS, IR and text using deformable image cross-attention."""

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
        self.vis_text_guided_layers = nn.ModuleList([
            TextGuidedVisualBackProjection(
                embed_dims=embed_dims, num_heads=num_heads, dropout=dropout)
            for _ in range(num_feature_levels)
        ])
        self.ir_text_guided_layers = nn.ModuleList([
            TextGuidedVisualBackProjection(
                embed_dims=embed_dims, num_heads=num_heads, dropout=dropout)
            for _ in range(num_feature_levels)
        ])
        self.cross_modal_attention = \
            BidirectionalDeformableCrossModalAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                num_feature_levels=num_feature_levels,
                num_points=num_points)
        self.modal_projection = nn.Linear(2 * embed_dims, embed_dims)

    def init_weights(self) -> None:
        self.cross_modal_attention.init_weights()

    @staticmethod
    def _split_levels(tensor: Optional[Tensor], level_lengths: tuple) -> tuple:
        if tensor is None:
            return (None, ) * len(level_lengths)
        return torch.split(tensor, level_lengths, dim=1)

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
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [num_levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} feature levels, but got '
                f'{spatial_shapes.size(0)}.')
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS and IR feature shapes must match, got {vis_feat.shape} '
                f'and {ir_feat.shape}.')

        level_lengths = tuple(
            int(length) for length in spatial_shapes.prod(dim=1).tolist())
        if sum(level_lengths) != vis_feat.size(1):
            raise ValueError(
                'The sum of flattened spatial shapes must equal the visual '
                f'token length, but got {sum(level_lengths)} and '
                f'{vis_feat.size(1)}.')

        vis_levels = self._split_levels(vis_feat, level_lengths)
        ir_levels = self._split_levels(ir_feat, level_lengths)
        pos_levels = self._split_levels(feat_pos, level_lengths)
        mask_levels = self._split_levels(feat_mask, level_lengths)

        vis_text_levels = []
        ir_text_levels = []
        for level_idx in range(self.num_feature_levels):
            vis_text_levels.append(self.vis_text_guided_layers[level_idx](
                visual=vis_levels[level_idx],
                text=text_feat,
                visual_pos=pos_levels[level_idx],
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask))
            ir_text_levels.append(self.ir_text_guided_layers[level_idx](
                visual=ir_levels[level_idx],
                text=text_feat,
                visual_pos=pos_levels[level_idx],
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask))

        vis_text = torch.cat(vis_text_levels, dim=1)
        ir_text = torch.cat(ir_text_levels, dim=1)
        vis_enhanced, ir_enhanced = self.cross_modal_attention(
            vis=vis_text,
            ir=ir_text,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask)

        fused = self.modal_projection(
            torch.cat([vis_enhanced, ir_enhanced], dim=-1))
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None], 0)
        return fused
