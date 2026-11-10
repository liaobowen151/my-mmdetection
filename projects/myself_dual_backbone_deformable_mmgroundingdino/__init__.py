"""Independent dual-backbone model with deformable cross-modal attention."""

# Reuse only the unchanged paired-image data components and base registrations.
from projects.myself_multimodal_mmgroundingdino import *  # noqa: F401,F403

from .deformable_multimodal_fusion import DeformableMultimodalFusionBlock
from .dual_backbone_deformable_multimodal_grounding_dino import (
    DualBackboneDeformableMultimodalGroundingDINO, )

__all__ = [
    'DeformableMultimodalFusionBlock',
    'DualBackboneDeformableMultimodalGroundingDINO'
]
