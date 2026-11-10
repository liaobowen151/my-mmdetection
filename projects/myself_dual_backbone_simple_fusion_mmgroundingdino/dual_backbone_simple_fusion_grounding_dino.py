"""Dual-backbone Grounding DINO with a minimal configurable fusion rule."""

from typing import Dict

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_dual_backbone_base_mmgroundingdino.dual_backbone_base_grounding_dino import (  # noqa: E501
    DualBackboneBaseGroundingDINO, )


@MODELS.register_module()
class DualBackboneSimpleFusionGroundingDINO(DualBackboneBaseGroundingDINO):
    """Replace direct post-neck addition with one isolated fusion module."""

    def __init__(self, *args, multimodal_fusion: dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multimodal_fusion = MODELS.build(multimodal_fusion)

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'DualBackboneSimpleFusionGroundingDINO expects batch_inputs '
                f'with shape [B, 6, H, W], got {tuple(batch_inputs.shape)}.')

        vis_feats = self.backbone(batch_inputs[:, :3])
        ir_feats = self.ir_backbone(batch_inputs[:, 3:])
        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)

        if len(vis_feats) != len(ir_feats):
            raise RuntimeError(
                'VIS and IR branches returned different feature counts: '
                f'{len(vis_feats)} versus {len(ir_feats)}.')
        for level, (vis_feat, ir_feat) in enumerate(zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'VIS/IR feature shape mismatch at level {level}: '
                    f'{tuple(vis_feat.shape)} versus {tuple(ir_feat.shape)}.')
        return dict(vis_feats=tuple(vis_feats), ir_feats=tuple(ir_feats))

    def forward_transformer(
        self,
        img_feats: Dict,
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        vis_encoder_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_encoder_inputs, _ = self.pre_transformer(
            img_feats['ir_feats'], batch_data_samples)

        for key in ('spatial_shapes', 'level_start_index'):
            if not torch.equal(vis_encoder_inputs[key], ir_encoder_inputs[key]):
                raise RuntimeError(f'VIS and IR {key} must be identical.')

        vis_encoder_inputs['feat'] = self.multimodal_fusion(
            vis_feat=vis_encoder_inputs['feat'],
            ir_feat=ir_encoder_inputs['feat'],
            text_feat=text_dict['embedded'],
            feat_mask=vis_encoder_inputs['feat_mask'],
            text_token_mask=text_dict['text_token_mask'])

        encoder_outputs = self.forward_encoder(
            **vis_encoder_inputs, text_dict=text_dict)
        tmp_decoder_inputs, head_inputs = self.pre_decoder(
            **encoder_outputs, batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)
        decoder_outputs = self.forward_decoder(**decoder_inputs)
        head_inputs.update(decoder_outputs)
        return head_inputs
