"""Grounding DINO wrapper that exposes the fusion auxiliary loss."""

from typing import Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino import (
    IRLoRAIRToVISDeformableAddIRGroundingDINO, )


@MODELS.register_module()
class IRLoRAIRToVISDeformableLanguageGatedGroundingDINO(
        IRLoRAIRToVISDeformableAddIRGroundingDINO):
    """Add the deep text-preservation loss to detection losses."""

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        losses = super().loss(batch_inputs, batch_data_samples)
        preserve_loss = self.multimodal_fusion.text_preserve_loss
        if preserve_loss is not None:
            losses['loss_text_preserve'] = (
                self.multimodal_fusion.text_preserve_loss_weight *
                preserve_loss)
        return losses
