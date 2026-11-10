"""Deformable cross-modal fusion with projected-cosine E1-Z."""

from projects.myself_dual_backbone_deformable_spatial_expectation_mmgroundingdino import (  # noqa: E501,F401
    DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO,
    SpatialChannelExpectationStatisticsHook,
)

from .projected_cosine_deformable_fusion import (
    NoDirectTextDeformableProjectedCosineSpatialExpectationFusionBlock,
)

__all__ = [
    'NoDirectTextDeformableProjectedCosineSpatialExpectationFusionBlock',
    'DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO',
    'SpatialChannelExpectationStatisticsHook',
]
