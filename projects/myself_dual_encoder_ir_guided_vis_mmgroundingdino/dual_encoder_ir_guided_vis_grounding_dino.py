"""Grounding DINO with separate VIS and IR multimodal encoders."""

import copy
from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_ir_guided_vis_mmgroundingdino import (
    IRGuidedVISGroundingDINO, )


@MODELS.register_module()
class DualEncoderIRGuidedVISGroundingDINO(IRGuidedVISGroundingDINO):
    """Use separate encoders while retaining IR-selected decoder proposals.

    ``self.encoder`` remains the VIS encoder so pretrained Grounding DINO
    checkpoints keep their original key names. ``self.ir_encoder`` is an
    independent copy. Old checkpoints without ``ir_encoder.*`` initialize it
    from ``encoder.*``; checkpoints from this class restore both branches.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ir_encoder = copy.deepcopy(self.encoder)

    def init_weights(self) -> None:
        super().init_weights()
        # This also gives identical initialization when no checkpoint is used.
        self.ir_encoder.load_state_dict(self.encoder.state_dict())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs) -> None:
        vis_prefix = prefix + 'encoder.'
        ir_prefix = prefix + 'ir_encoder.'
        has_ir_encoder = any(key.startswith(ir_prefix) for key in state_dict)
        if not has_ir_encoder:
            copied = {
                ir_prefix + key[len(vis_prefix):]: value
                for key, value in list(state_dict.items())
                if key.startswith(vis_prefix)
            }
            state_dict.update(copied)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    @staticmethod
    def _forward_encoder_branch(encoder, feat: Tensor, feat_mask: Tensor,
                                feat_pos: Tensor, spatial_shapes: Tensor,
                                level_start_index: Tensor,
                                valid_ratios: Tensor,
                                text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        return dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)

    def forward_transformer(self,
                            img_feats: Dict,
                            text_dict: Dict,
                            batch_data_samples: OptSampleList = None) -> Dict:
        vis_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_inputs, _ = self.pre_transformer(
            img_feats['ir_feats'], batch_data_samples)
        self._validate_encoder_layout(vis_inputs, ir_inputs)

        vis_outputs = self._forward_encoder_branch(
            self.encoder, **vis_inputs, text_dict=dict(text_dict))
        ir_outputs = self._forward_encoder_branch(
            self.ir_encoder, **ir_inputs, text_dict=dict(text_dict))

        tmp_decoder_inputs, head_inputs = self.pre_decoder_ir_guided_vis(
            vis_outputs=vis_outputs,
            ir_outputs=ir_outputs,
            batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)
        decoder_outputs = self.forward_decoder(**decoder_inputs)
        head_inputs.update(decoder_outputs)
        return head_inputs
