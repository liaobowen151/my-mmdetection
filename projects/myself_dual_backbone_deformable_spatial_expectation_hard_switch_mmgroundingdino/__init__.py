"""Hard-switch training for deformable attention plus E1-Z fusion."""

from projects.myself_dual_backbone_deformable_spatial_expectation_mmgroundingdino import (  # noqa: E501,F401
    SpatialChannelExpectationStatisticsHook,
)

from .hard_switch_fusion import (
    NoDirectTextDeformableSpatialExpectationHardSwitchFusionBlock, )
from .hard_switch_hook import HardSwitchFusionScheduleHook
from .hard_switch_grounding_dino import (
    DualBackboneNoDirectTextDeformableSpatialExpectationHardSwitchGroundingDINO,
)

__all__ = [
    'NoDirectTextDeformableSpatialExpectationHardSwitchFusionBlock',
    'HardSwitchFusionScheduleHook',
    'SpatialChannelExpectationStatisticsHook',
    'DualBackboneNoDirectTextDeformableSpatialExpectationHardSwitchGroundingDINO',
]
