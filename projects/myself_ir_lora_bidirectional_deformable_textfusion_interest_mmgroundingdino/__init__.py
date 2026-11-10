"""Bidirectional deformable text-gated fusion with interest supervision."""

from projects.myself_ir_lora_textfusion_interest_mmgroundingdino.dataset import (
    InterestMaskODVGDataset, M3FDInterestMaskODVGDataset)
from projects.myself_ir_lora_textfusion_interest_mmgroundingdino.schedule_hook import (
    TextFusionInterestScheduleHook, )

from .fusion import BidirectionalDeformableTextFusionInterestFusion
from .grounding_dino import (
    IRLoRABidirectionalDeformableTextFusionInterestGroundingDINO, )

__all__ = [
    'InterestMaskODVGDataset', 'M3FDInterestMaskODVGDataset',
    'TextFusionInterestScheduleHook',
    'BidirectionalDeformableTextFusionInterestFusion',
    'IRLoRABidirectionalDeformableTextFusionInterestGroundingDINO'
]
