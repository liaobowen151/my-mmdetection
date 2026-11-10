"""Fusion-only text-conditioned IR fusion with interest supervision."""

from projects.myself_ir_lora_textfusion_interest_mmgroundingdino.dataset import (
    InterestMaskODVGDataset, M3FDInterestMaskODVGDataset)
from projects.myself_ir_lora_textfusion_interest_mmgroundingdino.schedule_hook import (
    TextFusionInterestScheduleHook, )

from .fusion_only import TextFusionInterestFusionOnly
from .grounding_dino import (
    IRLoRATextFusionInterestFusionOnlyGroundingDINO, )

__all__ = [
    'InterestMaskODVGDataset', 'M3FDInterestMaskODVGDataset',
    'TextFusionInterestScheduleHook', 'TextFusionInterestFusionOnly',
    'IRLoRATextFusionInterestFusionOnlyGroundingDINO'
]
