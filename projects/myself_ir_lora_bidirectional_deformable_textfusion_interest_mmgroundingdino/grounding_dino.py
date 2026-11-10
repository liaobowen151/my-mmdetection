"""Grounding DINO for bidirectional deformable text-gated fusion."""

from mmdet.registry import MODELS
from projects.myself_ir_lora_textfusion_interest_mmgroundingdino import (
    IRLoRATextFusionInterestGroundingDINO, )


@MODELS.register_module()
class IRLoRABidirectionalDeformableTextFusionInterestGroundingDINO(
        IRLoRATextFusionInterestGroundingDINO):
    """Use the common dual-stream detector with the bidirectional fusion."""

    pass
