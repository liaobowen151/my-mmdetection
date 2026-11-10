"""Detector registration for hard-switch E1-Z experiments."""

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_spatial_expectation_mmgroundingdino.dual_backbone_deformable_spatial_expectation_grounding_dino import (  # noqa: E501
    DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO, )


@MODELS.register_module()
class DualBackboneNoDirectTextDeformableSpatialExpectationHardSwitchGroundingDINO(  # noqa: E501
        DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO):
    """No-direct-text deformable model with epoch-scheduled E1-Z fusion."""

    pass
