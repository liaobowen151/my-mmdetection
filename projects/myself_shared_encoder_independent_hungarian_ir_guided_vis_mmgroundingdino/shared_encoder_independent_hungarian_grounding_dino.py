"""Shared-encoder IR-guided Grounding DINO detector."""

from mmdet.registry import MODELS
from projects.myself_ir_guided_vis_mmgroundingdino import (
    IRGuidedVISGroundingDINO, )


@MODELS.register_module()
class SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO(
        IRGuidedVISGroundingDINO):
    """Share the multimodal encoder and use IR-selected decoder proposals.

    The inherited detector executes the same encoder parameters once for VIS
    and once for IR. IR selects the Top-K spatial indices and supplies decoder
    reference boxes; VIS supplies decoder visual/text memory. Independent IR
    and VIS Hungarian assignments are implemented by the paired head.
    """

    pass
