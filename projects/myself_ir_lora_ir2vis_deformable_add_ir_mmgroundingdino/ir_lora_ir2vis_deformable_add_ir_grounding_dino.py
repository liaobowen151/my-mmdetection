"""IR-LoRA Grounding DINO using enhanced VIS plus original IR."""

from typing import Dict

from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from projects.myself_ir_lora_addition_mmgroundingdino import (
    IRLoRASwinTransformer, )


@MODELS.register_module()
class IRLoRAIRToVISDeformableAddIRGroundingDINO(GroundingDINO):
    """Feed IR-enhanced VIS plus original IR into the original encoder."""

    def __init__(self,
                 *args,
                 multimodal_fusion: dict,
                 freeze_visual_backbone: bool = True,
                 freeze_language_model: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multimodal_fusion = MODELS.build(multimodal_fusion)
        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model
        if not isinstance(self.backbone, IRLoRASwinTransformer):
            raise TypeError('This detector requires IRLoRASwinTransformer.')
        if not freeze_visual_backbone:
            raise ValueError('The shared visual backbone must stay frozen.')
        if self.backbone.uses_activation_checkpointing():
            raise ValueError('Adapter switching requires with_cp=False.')
        self.backbone.freeze_base_and_enable_lora()
        self.backbone.eval()
        if freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
        self.text_feat_map.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.backbone.set_lora_dropout_mode(mode)
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                f'Expected [B, 6, H, W], got {tuple(batch_inputs.shape)}.')
        self.backbone.set_ir_lora_enabled(False)
        vis_feats = self.backbone(batch_inputs[:, :3])
        self.backbone.set_ir_lora_enabled(True)
        ir_feats = self.backbone(batch_inputs[:, 3:])
        self.backbone.set_ir_lora_enabled(False)
        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)
        self._validate_feature_levels(vis_feats, ir_feats)
        return dict(vis_feats=tuple(vis_feats), ir_feats=tuple(ir_feats))

    @staticmethod
    def _validate_feature_levels(vis_feats, ir_feats) -> None:
        if len(vis_feats) != len(ir_feats):
            raise RuntimeError('VIS and IR returned different feature counts.')
        for level, (vis_feat, ir_feat) in enumerate(zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'Feature mismatch at level {level}: '
                    f'{vis_feat.shape} vs {ir_feat.shape}.')

    def forward_transformer(self, img_feats, text_dict,
                            batch_data_samples=None) -> Dict:
        vis_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_inputs, _ = self.pre_transformer(
            img_feats['ir_feats'], batch_data_samples)
        self._validate_encoder_layout(vis_inputs, ir_inputs)
        vis_inputs['feat'] = self.multimodal_fusion(
            vis_feat=vis_inputs['feat'],
            ir_feat=ir_inputs['feat'],
            text_feat=text_dict['embedded'],
            spatial_shapes=vis_inputs['spatial_shapes'],
            level_start_index=vis_inputs['level_start_index'],
            valid_ratios=vis_inputs['valid_ratios'],
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

    @staticmethod
    def _validate_encoder_layout(vis_inputs, ir_inputs) -> None:
        for key in ('spatial_shapes', 'level_start_index'):
            if not vis_inputs[key].equal(ir_inputs[key]):
                raise RuntimeError(f'VIS and IR {key} must be identical.')
