"""Dual-backbone Grounding DINO for configured pre-encoder fusion."""

import copy
from typing import Dict

from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS


@MODELS.register_module()
class DualBackboneSpatialExpectationGroundingDINO(GroundingDINO):
    """Keep VIS/IR features separate until the configured fusion module."""

    def __init__(self,
                 *args,
                 freeze_visual_backbone: bool = False,
                 freeze_language_model: bool = True,
                 multimodal_fusion: dict = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if multimodal_fusion is None:
            raise ValueError('multimodal_fusion must be configured.')
        self.multimodal_fusion = MODELS.build(multimodal_fusion)
        self.ir_backbone = copy.deepcopy(self.backbone)
        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model

        if freeze_visual_backbone:
            self.backbone.requires_grad_(False)
            self.ir_backbone.requires_grad_(False)
            self.backbone.eval()
            self.ir_backbone.eval()
        else:
            self.backbone.requires_grad_(True)
            self.ir_backbone.requires_grad_(True)
        if freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
        else:
            self.language_model.requires_grad_(True)
        self.text_feat_map.requires_grad_(True)
        self._register_load_state_dict_pre_hook(
            self._mirror_vis_checkpoint_to_ir, with_module=True)

    def _mirror_vis_checkpoint_to_ir(self, module, state_dict, prefix,
                                     local_metadata, strict, missing_keys,
                                     unexpected_keys, error_msgs) -> None:
        if module is not self:
            raise RuntimeError('Checkpoint hook registered on wrong model.')
        vis_prefix = prefix + 'backbone.'
        ir_prefix = prefix + 'ir_backbone.'
        if any(key.startswith(ir_prefix) for key in state_dict):
            return
        state_dict.update({
            ir_prefix + key[len(vis_prefix):]: value.clone()
            for key, value in list(state_dict.items())
            if key.startswith(vis_prefix)
        })

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual_backbone:
            self.backbone.eval()
            self.ir_backbone.eval()
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError('Expected paired inputs with shape [B, 6, H, W], '
                             f'got {tuple(batch_inputs.shape)}.')
        vis_feats = self.backbone(batch_inputs[:, :3])
        ir_feats = self.ir_backbone(batch_inputs[:, 3:])
        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)
        if len(vis_feats) != len(ir_feats):
            raise RuntimeError('VIS and IR branches returned different levels.')
        for vis_feat, ir_feat in zip(vis_feats, ir_feats):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError('VIS and IR feature shapes must match.')
        return dict(vis_feats=tuple(vis_feats), ir_feats=tuple(ir_feats))

    def forward_transformer(self, img_feats, text_dict,
                            batch_data_samples=None) -> Dict:
        vis_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_inputs, _ = self.pre_transformer(
            img_feats['ir_feats'], batch_data_samples)
        if not vis_inputs['spatial_shapes'].equal(ir_inputs['spatial_shapes']):
            raise RuntimeError('VIS and IR spatial_shapes must match.')
        if not vis_inputs['level_start_index'].equal(
                ir_inputs['level_start_index']):
            raise RuntimeError('VIS and IR level_start_index must match.')
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
        encoder_outputs = self.forward_encoder(**vis_inputs, text_dict=text_dict)
        tmp_decoder_inputs, head_inputs = self.pre_decoder(
            **encoder_outputs, batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)
        head_inputs.update(self.forward_decoder(**decoder_inputs))
        return head_inputs
