"""Detector registrations for deformable plus E1-Z fusion experiments."""

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_no_expectation_mmgroundingdino.dual_backbone_deformable_no_expectation_grounding_dino import (  # noqa: E501
    DualBackboneDeformableNoExpectationGroundingDINO, )
from projects.myself_dual_backbone_deformable_no_text_attention_mmgroundingdino.dual_backbone_deformable_no_text_attention_grounding_dino import (  # noqa: E501
    DualBackboneDeformableNoTextAttentionGroundingDINO, )


@MODELS.register_module()
class DualBackboneDirectTextDeformableSpatialExpectationGroundingDINO(
        DualBackboneDeformableNoExpectationGroundingDINO):
    """Direct-text/deformable model with E1-Z post-cross fusion."""

    pass


@MODELS.register_module()
class DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO(
        DualBackboneDeformableNoTextAttentionGroundingDINO):
    """No-direct-text/deformable model with E1-Z post-cross fusion."""

    pass
