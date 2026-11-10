"""Detector registration for the no per-modality text-attention ablation."""

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_mmgroundingdino.dual_backbone_deformable_multimodal_grounding_dino import (  # noqa: E501
    DualBackboneDeformableMultimodalGroundingDINO, )


@MODELS.register_module()
class DualBackboneDeformableNoTextAttentionGroundingDINO(
        DualBackboneDeformableMultimodalGroundingDINO):
    """Grounding DINO with visual-only VIS/IR pre-encoder fusion."""

    pass
