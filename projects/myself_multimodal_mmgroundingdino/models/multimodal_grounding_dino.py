from typing import Dict

import torch
from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList


@MODELS.register_module()
class MultimodalGroundingDINO(GroundingDINO):
    """Grounding DINO with shared-backbone VIS/IR feature fusion."""

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
        self._freeze_pretrained_backbones()

    def _freeze_pretrained_backbones(self) -> None:
        if self.freeze_visual_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # ``model.train()`` is called by the runner every training stage. Keep
        # stochastic layers in frozen Swin-T and BERT disabled.
        if self.freeze_visual_backbone:
            self.backbone.eval()
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        """Extract paired features with one shared Swin-T and neck call."""
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'MultimodalGroundingDINO expects batch_inputs with shape '
                f'[B, 6, H, W], but got {tuple(batch_inputs.shape)}.')

        batch_size = batch_inputs.size(0)
        vis_inputs = batch_inputs[:, :3]
        ir_inputs = batch_inputs[:, 3:]
        paired_inputs = torch.cat([vis_inputs, ir_inputs], dim=0)

        paired_feats = self.backbone(paired_inputs)
        if self.with_neck:
            paired_feats = self.neck(paired_feats)

        vis_feats = []
        ir_feats = []
        for feat in paired_feats:
            if feat.size(0) != 2 * batch_size:
                raise RuntimeError(
                    'Shared feature batch dimension changed unexpectedly: '
                    f'expected {2 * batch_size}, got {feat.size(0)}.')
            vis_feats.append(feat[:batch_size])
            ir_feats.append(feat[batch_size:])

        return dict(
            vis_feats=tuple(vis_feats),
            ir_feats=tuple(ir_feats))

    def forward_transformer(
        self,
        img_feats: Dict,
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Fuse VIS/IR/text before the official Grounding DINO encoder."""
        vis_encoder_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_encoder_inputs, _ = self.pre_transformer(
            img_feats['ir_feats'], batch_data_samples)

        if not torch.equal(vis_encoder_inputs['spatial_shapes'],
                           ir_encoder_inputs['spatial_shapes']):
            raise RuntimeError('VIS and IR spatial_shapes must be identical.')
        if not torch.equal(vis_encoder_inputs['level_start_index'],
                           ir_encoder_inputs['level_start_index']):
            raise RuntimeError(
                'VIS and IR level_start_index must be identical.')

        fused_feat = self.multimodal_fusion(
            vis_feat=vis_encoder_inputs['feat'],
            ir_feat=ir_encoder_inputs['feat'],
            text_feat=text_dict['embedded'],
            spatial_shapes=vis_encoder_inputs['spatial_shapes'],
            feat_pos=vis_encoder_inputs['feat_pos'],
            feat_mask=vis_encoder_inputs['feat_mask'],
            text_token_mask=text_dict['text_token_mask'])
        vis_encoder_inputs['feat'] = fused_feat

        encoder_outputs = self.forward_encoder(
            **vis_encoder_inputs, text_dict=text_dict)
        tmp_decoder_inputs, head_inputs = self.pre_decoder(
            **encoder_outputs, batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)

        decoder_outputs = self.forward_decoder(**decoder_inputs)
        head_inputs.update(decoder_outputs)
        return head_inputs
