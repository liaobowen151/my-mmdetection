"""Text-conditioned IR fusion with D3 interest-mask supervision."""

from projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino import *  # noqa: F401,F403

from .dataset import InterestMaskODVGDataset, M3FDInterestMaskODVGDataset
from .fusion import TextFusionInterestFusion
from .grounding_dino import IRLoRATextFusionInterestGroundingDINO
from .schedule_hook import TextFusionInterestScheduleHook

__all__ = [
    'InterestMaskODVGDataset', 'M3FDInterestMaskODVGDataset',
    'TextFusionInterestFusion',
    'IRLoRATextFusionInterestGroundingDINO', 'TextFusionInterestScheduleHook'
]
