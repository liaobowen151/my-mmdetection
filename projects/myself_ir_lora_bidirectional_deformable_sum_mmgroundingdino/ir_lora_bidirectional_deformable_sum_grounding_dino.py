"""Grounding DINO with shared Swin, IR LoRA and bidirectional fusion."""

import torch

from mmdet.registry import MODELS
from projects.myself_ir_lora_ir2vis_deformable_mmgroundingdino.ir_lora_ir2vis_deformable_grounding_dino import (  # noqa: E501
    IRLoRAIRToVISDeformableGroundingDINO,
)


@MODELS.register_module()
class IRLoRABidirectionalDeformableSumGroundingDINO(
        IRLoRAIRToVISDeformableGroundingDINO):
    """Run shared RGB/IR-LoRA features through bidirectional sum fusion."""

    @staticmethod
    def _validate_encoder_layout(vis_inputs, ir_inputs) -> None:
        for key in ('spatial_shapes', 'level_start_index', 'valid_ratios'):
            if not torch.equal(vis_inputs[key], ir_inputs[key]):
                raise RuntimeError(f'VIS and IR {key} must be identical.')
        vis_mask = vis_inputs['feat_mask']
        ir_mask = ir_inputs['feat_mask']
        if (vis_mask is None) != (ir_mask is None):
            raise RuntimeError('VIS and IR feature masks must both be present '
                               'or both be absent.')
        if vis_mask is not None and not torch.equal(vis_mask, ir_mask):
            raise RuntimeError('VIS and IR feature masks must be identical.')

