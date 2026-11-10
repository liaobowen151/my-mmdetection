"""Grounding DINO encoders with private early and shared late layers."""

import copy
from typing import Dict

import torch
from torch import Tensor, nn

from mmdet.models.layers.transformer.grounding_dino_layers import (
    get_text_sine_pos_embed, )
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_shared_encoder_independent_hungarian_ir_guided_vis_mmgroundingdino import (  # noqa: E501
    SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO, )


class _PartialSharedEncoderMixin:
    """Split the first encoder layers by modality and share the remainder."""

    private_num_layers = 3
    private_text_layers = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.encoder.num_layers != 6:
            raise ValueError(
                'The partial-sharing variants require a six-layer encoder, '
                f'but got {self.encoder.num_layers}.')
        if not 0 < self.private_num_layers < self.encoder.num_layers:
            raise ValueError('private_num_layers must split the encoder.')

        split = self.private_num_layers
        self.encoder.ir_private_layers = nn.ModuleList([
            copy.deepcopy(self.encoder.layers[i]) for i in range(split)
        ])
        self.encoder.ir_private_fusion_layers = nn.ModuleList([
            copy.deepcopy(self.encoder.fusion_layers[i])
            for i in range(split)
        ])
        if self.private_text_layers:
            self.encoder.ir_private_text_layers = nn.ModuleList([
                copy.deepcopy(self.encoder.text_layers[i])
                for i in range(split)
            ])

    def init_weights(self) -> None:
        super().init_weights()
        self._copy_vis_private_weights_to_ir()

    def _copy_vis_private_weights_to_ir(self) -> None:
        split = self.private_num_layers
        for i in range(split):
            self.encoder.ir_private_layers[i].load_state_dict(
                self.encoder.layers[i].state_dict())
            self.encoder.ir_private_fusion_layers[i].load_state_dict(
                self.encoder.fusion_layers[i].state_dict())
            if self.private_text_layers:
                self.encoder.ir_private_text_layers[i].load_state_dict(
                    self.encoder.text_layers[i].state_dict())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs) -> None:
        """Initialize new private IR keys from old standard checkpoints."""
        ir_prefix = prefix + 'encoder.ir_private_layers.'
        has_private_ir = any(
            key.startswith(ir_prefix) for key in state_dict)
        if not has_private_ir:
            mappings = [
                ('encoder.layers.', 'encoder.ir_private_layers.'),
                ('encoder.fusion_layers.',
                 'encoder.ir_private_fusion_layers.'),
            ]
            if self.private_text_layers:
                mappings.append(
                    ('encoder.text_layers.',
                     'encoder.ir_private_text_layers.'))
            for source, target in mappings:
                source = prefix + source
                target = prefix + target
                for key, value in list(state_dict.items()):
                    if not key.startswith(source):
                        continue
                    suffix = key[len(source):]
                    layer_id = int(suffix.split('.', 1)[0])
                    if layer_id < self.private_num_layers:
                        state_dict[target + suffix] = value
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def _forward_partial_encoder(
            self, modality: str, feat: Tensor, feat_mask: Tensor,
            feat_pos: Tensor, spatial_shapes: Tensor,
            level_start_index: Tensor, valid_ratios: Tensor,
            text_dict: Dict) -> Dict:
        if modality not in ('vis', 'ir'):
            raise ValueError(f'Unsupported modality: {modality}')

        encoder = self.encoder
        output = feat
        memory_text = text_dict['embedded']
        text_token_mask = text_dict['text_token_mask']
        text_attention_mask = ~text_token_mask
        reference_points = encoder.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        bs, n_text, _ = memory_text.shape
        position_ids = text_dict.get('position_ids')
        if position_ids is not None:
            pos_text = get_text_sine_pos_embed(
                position_ids[..., None], num_pos_feats=256,
                exchange_xy=False)
        else:
            pos_text = torch.arange(
                n_text, device=memory_text.device).float()
            pos_text = pos_text[None, :, None].repeat(bs, 1, 1)
            pos_text = get_text_sine_pos_embed(
                pos_text, num_pos_feats=256, exchange_xy=False)

        split = self.private_num_layers
        for layer_id in range(encoder.num_layers):
            use_ir_private = modality == 'ir' and layer_id < split
            if use_ir_private:
                fusion_layer = encoder.ir_private_fusion_layers[layer_id]
                visual_layer = encoder.ir_private_layers[layer_id]
                if self.private_text_layers:
                    text_layer = encoder.ir_private_text_layers[layer_id]
                else:
                    text_layer = encoder.text_layers[layer_id]
            else:
                fusion_layer = encoder.fusion_layers[layer_id]
                text_layer = encoder.text_layers[layer_id]
                visual_layer = encoder.layers[layer_id]

            output, memory_text = fusion_layer(
                visual_feature=output,
                lang_feature=memory_text,
                attention_mask_v=feat_mask,
                attention_mask_l=text_attention_mask)
            text_num_heads = text_layer.self_attn_cfg.num_heads
            memory_text = text_layer(
                query=memory_text,
                query_pos=pos_text,
                attn_mask=~text_dict['masks'].repeat(
                    text_num_heads, 1, 1),
                key_padding_mask=None)
            output = visual_layer(
                query=output,
                query_pos=feat_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=feat_mask)

        return dict(
            memory=output,
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

        vis_outputs = self._forward_partial_encoder(
            'vis', **vis_inputs, text_dict=dict(text_dict))
        ir_outputs = self._forward_partial_encoder(
            'ir', **ir_inputs, text_dict=dict(text_dict))

        tmp_decoder_inputs, head_inputs = self.pre_decoder_ir_guided_vis(
            vis_outputs=vis_outputs,
            ir_outputs=ir_outputs,
            batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)
        decoder_outputs = self.forward_decoder(**decoder_inputs)
        head_inputs.update(decoder_outputs)
        return head_inputs


@MODELS.register_module()
class PartialSharedEncoderFullPrivateIRGuidedVISGroundingDINO(
        _PartialSharedEncoderMixin,
        SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO):
    """方案 A: private visual, text and fusion layers 0--2."""

    private_text_layers = True


@MODELS.register_module()
class PartialSharedEncoderSharedTextIRGuidedVISGroundingDINO(
        _PartialSharedEncoderMixin,
        SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO):
    """方案 B: private visual/fusion layers 0--2 and shared text layers."""

    private_text_layers = False
