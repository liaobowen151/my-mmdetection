"""Shared-encoder IR-guided Grounding DINO with separate assignments."""

from projects.myself_ir_guided_vis_mmgroundingdino import (  # noqa: F401
    IRGuidedVISGroundingDINO, IRGuidedVISGroundingDINOHead)

from .shared_encoder_independent_hungarian_grounding_dino import (
    SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO, )
from .shared_encoder_independent_hungarian_grounding_dino_head import (
    SharedEncoderIndependentHungarianIRGuidedVISGroundingDINOHead, )

__all__ = [
    'SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO',
    'SharedEncoderIndependentHungarianIRGuidedVISGroundingDINOHead'
]
