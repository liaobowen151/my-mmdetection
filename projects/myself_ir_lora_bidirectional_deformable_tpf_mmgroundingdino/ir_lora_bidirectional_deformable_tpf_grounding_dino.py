"""Grounding DINO with bidirectional deformable tri-prior fusion."""

from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_ir_lora_bidirectional_deformable_sum_mmgroundingdino import (  # noqa: E501
    IRLoRABidirectionalDeformableSumGroundingDINO, )


@MODELS.register_module()
class IRLoRABidirectionalDeformableTPFGroundingDINO(
        IRLoRABidirectionalDeformableSumGroundingDINO):
    """Pass normalized RGB input to TPF while retaining the base pipeline."""

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        img_feats = super().extract_feat(batch_inputs)
        img_feats['rgb_image'] = batch_inputs[:, :3]
        return img_feats

    def forward_transformer(self,
                            img_feats,
                            text_dict,
                            batch_data_samples=None) -> Dict:
        vis_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_inputs, _ = self.pre_transformer(img_feats['ir_feats'],
                                            batch_data_samples)
        self._validate_encoder_layout(vis_inputs, ir_inputs)
        vis_inputs['feat'] = self.multimodal_fusion(
            vis_feat=vis_inputs['feat'],
            ir_feat=ir_inputs['feat'],
            text_feat=text_dict['embedded'],
            spatial_shapes=vis_inputs['spatial_shapes'],
            level_start_index=vis_inputs['level_start_index'],
            valid_ratios=vis_inputs['valid_ratios'],
            rgb_image=img_feats['rgb_image'],
            feat_pos=vis_inputs['feat_pos'],
            feat_mask=vis_inputs['feat_mask'],
            text_token_mask=text_dict['text_token_mask'])
        encoder_outputs = self.forward_encoder(
            **vis_inputs, text_dict=text_dict)
        tmp_decoder_inputs, head_inputs = self.pre_decoder(
            **encoder_outputs, batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)
        decoder_outputs = self.forward_decoder(**decoder_inputs)
        head_inputs.update(decoder_outputs)
        return head_inputs
