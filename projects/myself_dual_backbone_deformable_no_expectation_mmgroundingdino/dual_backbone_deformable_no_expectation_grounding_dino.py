"""Detector registration for the no-expectation ablation."""

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_mmgroundingdino.dual_backbone_deformable_multimodal_grounding_dino import (  # noqa: E501
    DualBackboneDeformableMultimodalGroundingDINO, )


@MODELS.register_module()
class DualBackboneDeformableNoExpectationGroundingDINO(
        DualBackboneDeformableMultimodalGroundingDINO):
    """Independent experiment registration; pipeline is unchanged from v2."""

    pass
