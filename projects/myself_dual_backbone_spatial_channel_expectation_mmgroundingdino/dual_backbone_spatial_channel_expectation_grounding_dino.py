"""Detector registration for spatial/channel expectation experiments."""

from mmdet.registry import MODELS
from projects.myself_dual_backbone_spatial_expectation_mmgroundingdino.dual_backbone_spatial_expectation_grounding_dino import (  # noqa: E501
    DualBackboneSpatialExpectationGroundingDINO, )


@MODELS.register_module()
class DualBackboneSpatialChannelExpectationGroundingDINO(
        DualBackboneSpatialExpectationGroundingDINO):
    """An isolated detector name for the E1--E4 fusion ablations."""

    pass
