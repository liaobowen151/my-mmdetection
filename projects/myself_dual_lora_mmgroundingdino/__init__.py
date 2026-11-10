"""Independent dual-adapter LoRA extension for multimodal Grounding DINO."""

# Importing the existing package registers the paired loader, preprocessor and
# multimodal fusion block without changing their implementation.
from projects.myself_multimodal_mmgroundingdino import *  # noqa: F401,F403

from .dual_lora_multimodal_grounding_dino import (
    DualLoRAMultimodalGroundingDINO,
)
from .dual_lora_swin import (DualAdapterLoRALinear,
                             DualAdapterLoRASwinTransformer)

__all__ = [
    'DualAdapterLoRALinear', 'DualAdapterLoRASwinTransformer',
    'DualLoRAMultimodalGroundingDINO'
]
