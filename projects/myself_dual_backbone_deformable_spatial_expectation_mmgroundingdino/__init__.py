"""Deformable VIS/IR fusion followed by E1-Z spatial expectation fusion."""

from projects.myself_dual_backbone_deformable_no_expectation_mmgroundingdino import (  # noqa: E501,F401
    DirectTextDeformableNoExpectationFusionBlock,
    DualBackboneDeformableNoExpectationGroundingDINO,
)
from projects.myself_dual_backbone_deformable_no_text_attention_mmgroundingdino import (  # noqa: E501,F401
    DeformableNoTextAttentionFusionBlock,
    DualBackboneDeformableNoTextAttentionGroundingDINO,
)
from projects.myself_dual_backbone_spatial_channel_expectation_mmgroundingdino import (  # noqa: E501,F401
    SpatialChannelExpectationStatisticsHook,
)

from .deformable_spatial_expectation_fusion import (
    DirectTextDeformableSpatialExpectationFusionBlock,
    NoDirectTextDeformableSpatialExpectationFusionBlock,
)
from .dual_backbone_deformable_spatial_expectation_grounding_dino import (
    DualBackboneDirectTextDeformableSpatialExpectationGroundingDINO,
    DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO,
)

__all__ = [
    'DirectTextDeformableSpatialExpectationFusionBlock',
    'NoDirectTextDeformableSpatialExpectationFusionBlock',
    'DualBackboneDirectTextDeformableSpatialExpectationGroundingDINO',
    'DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO',
    'SpatialChannelExpectationStatisticsHook',
]
