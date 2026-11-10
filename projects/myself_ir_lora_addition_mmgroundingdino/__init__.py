"""IR-only LoRA backbone with post-neck VIS/IR addition."""

# Register the paired-image loader and data preprocessor.
from projects.myself_multimodal_mmgroundingdino import *  # noqa: F401,F403

from .ir_lora_addition_grounding_dino import IRLoRAAdditionGroundingDINO
from .ir_lora_swin import IRLoRALinear, IRLoRASwinTransformer

__all__ = [
    'IRLoRALinear', 'IRLoRASwinTransformer',
    'IRLoRAAdditionGroundingDINO'
]
