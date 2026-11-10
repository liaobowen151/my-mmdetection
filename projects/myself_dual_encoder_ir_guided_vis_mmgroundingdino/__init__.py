"""Dual-encoder IR-guided, VIS-decoded Grounding DINO."""

from projects.myself_ir_guided_vis_mmgroundingdino import (  # noqa: F401
    IRGuidedVISGroundingDINO, IRGuidedVISGroundingDINOHead)

from .dual_encoder_ir_guided_vis_grounding_dino import (
    DualEncoderIRGuidedVISGroundingDINO, )
from .dual_encoder_ir_guided_vis_grounding_dino_head import (
    DualEncoderIRGuidedVISGroundingDINOHead, )

__all__ = [
    'DualEncoderIRGuidedVISGroundingDINO',
    'DualEncoderIRGuidedVISGroundingDINOHead'
]
