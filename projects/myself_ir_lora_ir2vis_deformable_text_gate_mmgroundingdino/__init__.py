"""Text-gated one-way IR-to-VIS deformable fusion."""

# Importing the original package registers the shared-backbone detector and
# IR-LoRA backbone reused by these experiments.
from projects.myself_ir_lora_ir2vis_deformable_mmgroundingdino import *  # noqa: F401,F403

from .text_gated_deformable_fusion import (  # noqa: F401
    IRToVISDeformableDirectTextGateFusion,
    IRToVISDeformableLSETextGateFusion,
)

