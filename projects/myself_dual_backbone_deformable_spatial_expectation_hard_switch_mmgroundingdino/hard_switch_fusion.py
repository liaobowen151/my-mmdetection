"""No-direct-text deformable fusion with a hard E1-Z activation switch."""

from typing import Optional

import torch
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_spatial_expectation_mmgroundingdino.deformable_spatial_expectation_fusion import (  # noqa: E501
    NoDirectTextDeformableSpatialExpectationFusionBlock, )


@MODELS.register_module()
class NoDirectTextDeformableSpatialExpectationHardSwitchFusionBlock(
        NoDirectTextDeformableSpatialExpectationFusionBlock):
    """Use deformable sum fusion until E1-Z is enabled by a schedule hook."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Persistent state makes a resumed run use the correct fusion stage.
        self.register_buffer(
            'spatial_expectation_enabled', torch.tensor(False),
            persistent=True)

    def set_spatial_expectation_enabled(self, enabled: bool) -> None:
        self.spatial_expectation_enabled.fill_(bool(enabled))

    def is_spatial_expectation_enabled(self) -> bool:
        return bool(self.spatial_expectation_enabled.item())

    def get_statistics(self):
        statistics = super().get_statistics()
        statistics['stage/spatial_expectation_enabled'] = torch.tensor(
            [float(self.is_spatial_expectation_enabled()), 1.0],
            dtype=torch.float64,
            device=self.spatial_expectation_enabled.device)
        return statistics

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
        if self.is_spatial_expectation_enabled():
            return self._spatial_expectation_fuse(
                vis_enhanced, ir_enhanced, text_feat, spatial_shapes,
                level_start_index, feat_mask, text_token_mask)

        fused = vis_enhanced + ir_enhanced
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].to(torch.bool), 0)
        return fused
