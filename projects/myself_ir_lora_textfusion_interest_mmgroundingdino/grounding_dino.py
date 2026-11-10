"""Grounding DINO with TextFusion-inspired interest supervision."""

from typing import Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino import (
    IRLoRAIRToVISDeformableAddIRGroundingDINO, )


@MODELS.register_module()
class IRLoRATextFusionInterestGroundingDINO(
        IRLoRAIRToVISDeformableAddIRGroundingDINO):
    """Add the target-specific D3 interest loss to detection losses."""

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # Reuse Grounding DINO's complete text/positive-map/detection path.
        losses = super().loss(batch_inputs, batch_data_samples)
        interest_masks = []
        for sample in batch_data_samples:
            if 'gt_sem_seg' not in sample:
                raise RuntimeError(
                    'Training sample is missing its D3 interest mask.')
            interest_masks.append(sample.gt_sem_seg.sem_seg.squeeze(0))
        losses['loss_interest'] = self.multimodal_fusion.interest_loss(
            interest_masks)
        gate_statistics = self.multimodal_fusion.last_gate_statistics
        if gate_statistics is not None:
            losses.update(gate_statistics)
        return losses
