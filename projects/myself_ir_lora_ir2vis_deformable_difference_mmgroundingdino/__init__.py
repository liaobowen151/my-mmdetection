"""IR-LoRA shared-sampling deformable difference fusion."""

from projects.myself_ir_lora_ir2vis_deformable_mmgroundingdino import (  # noqa: F401
    IRLoRAIRToVISDeformableGroundingDINO,
)

from .shared_difference_fusion import (
    IRToVISDeformableDifferenceConcatFusion,
    IRToVISDeformableDifferenceFusion,
)

__all__ = [
    'IRLoRAIRToVISDeformableGroundingDINO',
    'IRToVISDeformableDifferenceFusion',
    'IRToVISDeformableDifferenceConcatFusion',
]
