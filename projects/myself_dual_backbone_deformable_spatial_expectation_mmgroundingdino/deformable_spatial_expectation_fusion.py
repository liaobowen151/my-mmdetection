"""Deformable cross-modal enhancement followed by E1-Z fusion."""

from typing import Optional

import torch
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_no_expectation_mmgroundingdino.direct_text_deformable_fusion import (  # noqa: E501
    DirectTextDeformableNoExpectationFusionBlock, )
from projects.myself_dual_backbone_deformable_no_text_attention_mmgroundingdino.deformable_no_text_attention_fusion import (  # noqa: E501
    DeformableNoTextAttentionFusionBlock, )
from projects.myself_dual_backbone_spatial_channel_expectation_mmgroundingdino.spatial_channel_expectation_fusion import (  # noqa: E501
    SpatialChannelExpectationFusion, )


class _SpatialExpectationStatisticsMixin:
    """Attach E1-Z fusion and expose its diagnostics to the existing hook."""

    def _init_spatial_expectation(
            self,
            embed_dims: int,
            token_temperature: float,
            gate_temperature: float,
            token_fpn_mode: str,
            with_channel_gate: bool,
            channel_reduction: int,
            center_alignment: bool,
            alignment_normalization: str,
            alignment_normalization_eps: float,
            spatial_gate_mode: str,
            high_response_threshold: float,
            low_response_threshold: float,
            token_temperature_mode: str,
            token_temperature_reference_length: float,
            token_temperature_length_power: float,
            token_temperature_min: float,
            token_temperature_max: float) -> None:
        self.spatial_expectation_fusion = SpatialChannelExpectationFusion(
            embed_dims=embed_dims,
            token_temperature=token_temperature,
            gate_temperature=gate_temperature,
            token_fpn_mode=token_fpn_mode,
            with_channel_gate=with_channel_gate,
            channel_reduction=channel_reduction,
            center_alignment=center_alignment,
            alignment_normalization=alignment_normalization,
            alignment_normalization_eps=alignment_normalization_eps,
            spatial_gate_mode=spatial_gate_mode,
            high_response_threshold=high_response_threshold,
            low_response_threshold=low_response_threshold,
            token_temperature_mode=token_temperature_mode,
            token_temperature_reference_length=token_temperature_reference_length,
            token_temperature_length_power=token_temperature_length_power,
            token_temperature_min=token_temperature_min,
            token_temperature_max=token_temperature_max)

    def _spatial_expectation_fuse(
            self, vis_feat: Tensor, ir_feat: Tensor, text_feat: Tensor,
            spatial_shapes: Tensor, level_start_index: Tensor,
            feat_mask: Optional[Tensor],
            text_token_mask: Optional[Tensor]) -> Tensor:
        return self.spatial_expectation_fusion(
            vis_feat=vis_feat,
            ir_feat=ir_feat,
            text_feat=text_feat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            feat_mask=feat_mask,
            text_token_mask=text_token_mask)

    def start_statistics(self) -> None:
        self.spatial_expectation_fusion.start_statistics()

    def stop_statistics(self) -> None:
        self.spatial_expectation_fusion.stop_statistics()

    def get_statistics(self):
        return self.spatial_expectation_fusion.get_statistics()


class _SpatialExpectationConfigMixin:
    """Common E1-Z constructor options for the two deformable variants."""

    @staticmethod
    def expectation_defaults():
        return dict(
            token_temperature=1.0,
            gate_temperature=1.0,
            token_fpn_mode='level_balanced_shared',
            with_channel_gate=False,
            channel_reduction=4,
            center_alignment=False,
            alignment_normalization='zscore',
            alignment_normalization_eps=1e-6,
            spatial_gate_mode='dense',
            high_response_threshold=1.5,
            low_response_threshold=-0.5,
            token_temperature_mode='fixed',
            token_temperature_reference_length=16.0,
            token_temperature_length_power=0.5,
            token_temperature_min=0.5,
            token_temperature_max=4.0)


@MODELS.register_module()
class DirectTextDeformableSpatialExpectationFusionBlock(
        _SpatialExpectationStatisticsMixin, _SpatialExpectationConfigMixin,
        DirectTextDeformableNoExpectationFusionBlock):
    """Direct text, deformable VIS/IR attention, then E1-Z fusion."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 text_attention_dims: int = 1024,
                 text_attention_heads: int = 4,
                 text_attention_init_values: float = 1e-4,
                 **expectation_kwargs) -> None:
        DirectTextDeformableNoExpectationFusionBlock.__init__(
            self,
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            num_points=num_points,
            text_attention_dims=text_attention_dims,
            text_attention_heads=text_attention_heads,
            text_attention_init_values=text_attention_init_values)
        options = self.expectation_defaults()
        options.update(expectation_kwargs)
        self._init_spatial_expectation(embed_dims=embed_dims, **options)

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
        level_lengths = tuple(
            int(length) for length in spatial_shapes.prod(dim=1).tolist())
        if sum(level_lengths) != vis_feat.size(1):
            raise ValueError('Flattened spatial shapes do not match token length.')
        vis_levels = self._split_levels(vis_feat, level_lengths)
        ir_levels = self._split_levels(ir_feat, level_lengths)
        mask_levels = self._split_levels(feat_mask, level_lengths)
        vis_text = torch.cat([
            self.vis_text_attention(
                visual=vis_levels[level_idx], text=text_feat,
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask)
            for level_idx in range(self.num_feature_levels)
        ], dim=1)
        ir_text = torch.cat([
            self.ir_text_attention(
                visual=ir_levels[level_idx], text=text_feat,
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask)
            for level_idx in range(self.num_feature_levels)
        ], dim=1)
        vis_enhanced, ir_enhanced = self.cross_modal_attention(
            vis=vis_text,
            ir=ir_text,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask)
        return self._spatial_expectation_fuse(
            vis_enhanced, ir_enhanced, text_feat, spatial_shapes,
            level_start_index, feat_mask, text_token_mask)


@MODELS.register_module()
class NoDirectTextDeformableSpatialExpectationFusionBlock(
        _SpatialExpectationStatisticsMixin, _SpatialExpectationConfigMixin,
        DeformableNoTextAttentionFusionBlock):
    """Deformable VIS/IR attention, then E1-Z fusion without direct text."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 **expectation_kwargs) -> None:
        DeformableNoTextAttentionFusionBlock.__init__(
            self,
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            num_points=num_points)
        options = self.expectation_defaults()
        options.update(expectation_kwargs)
        self._init_spatial_expectation(embed_dims=embed_dims, **options)

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
        if vis_feat.shape != ir_feat.shape:
            raise ValueError('VIS and IR feature shapes must match.')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError('Unexpected number of feature levels.')
        if int(spatial_shapes.prod(dim=1).sum().item()) != vis_feat.size(1):
            raise ValueError('Flattened spatial shapes do not match token length.')
        vis_enhanced, ir_enhanced = self.cross_modal_attention(
            vis=vis_feat,
            ir=ir_feat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask)
        return self._spatial_expectation_fuse(
            vis_enhanced, ir_enhanced, text_feat, spatial_shapes,
            level_start_index, feat_mask, text_token_mask)
