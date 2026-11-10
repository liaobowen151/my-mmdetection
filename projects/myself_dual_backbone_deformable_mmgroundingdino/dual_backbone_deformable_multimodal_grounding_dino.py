"""Dual-backbone Grounding DINO with deformable cross-modal attention."""

import copy
from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_multimodal_mmgroundingdino.models.multimodal_grounding_dino import (  # noqa: E501
    MultimodalGroundingDINO, )


@MODELS.register_module()
class DualBackboneDeformableMultimodalGroundingDINO(MultimodalGroundingDINO):
    """Use independent VIS/IR backbones and deformable cross-modal fusion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ir_backbone = copy.deepcopy(self.backbone)

        if not self.freeze_visual_backbone:
            self.backbone.requires_grad_(True)
            self.ir_backbone.requires_grad_(True)
        else:
            self.ir_backbone.requires_grad_(False)
            self.ir_backbone.eval()

        self._register_load_state_dict_pre_hook(
            self._mirror_vis_checkpoint_to_ir, with_module=True)

    def _mirror_vis_checkpoint_to_ir(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        if module is not self:
            raise RuntimeError(
                'Checkpoint pre-hook registered on wrong model.')

        vis_prefix = prefix + 'backbone.'
        ir_prefix = prefix + 'ir_backbone.'
        if any(key.startswith(ir_prefix) for key in state_dict):
            return

        mirrored = {}
        for key, value in state_dict.items():
            if key.startswith(vis_prefix):
                suffix = key[len(vis_prefix):]
                mirrored[ir_prefix + suffix] = value.clone()
        state_dict.update(mirrored)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual_backbone:
            self.ir_backbone.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'DualBackboneDeformableMultimodalGroundingDINO expects '
                'batch_inputs with shape [B, 6, H, W], got '
                f'{tuple(batch_inputs.shape)}.')

        vis_feats = self.backbone(batch_inputs[:, :3])
        ir_feats = self.ir_backbone(batch_inputs[:, 3:])
        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)

        if len(vis_feats) != len(ir_feats):
            raise RuntimeError(
                'VIS and IR backbones returned different feature counts: '
                f'{len(vis_feats)} versus {len(ir_feats)}.')
        for level, (vis_feat, ir_feat) in enumerate(zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'VIS/IR feature shape mismatch at level {level}: '
                    f'{tuple(vis_feat.shape)} versus {tuple(ir_feat.shape)}.')
        return dict(vis_feats=tuple(vis_feats), ir_feats=tuple(ir_feats))

    def forward_transformer(self,
                            img_feats,
                            text_dict,
                            batch_data_samples=None) -> Dict:
        vis_encoder_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_encoder_inputs, _ = self.pre_transformer(img_feats['ir_feats'],
                                                    batch_data_samples)

        if not vis_encoder_inputs['spatial_shapes'].equal(
                ir_encoder_inputs['spatial_shapes']):
            raise RuntimeError('VIS and IR spatial_shapes must be identical.')
        if not vis_encoder_inputs['level_start_index'].equal(
                ir_encoder_inputs['level_start_index']):
            raise RuntimeError(
                'VIS and IR level_start_index must be identical.')

        vis_encoder_inputs['feat'] = self.multimodal_fusion(
            vis_feat=vis_encoder_inputs['feat'],
            ir_feat=ir_encoder_inputs['feat'],
            text_feat=text_dict['embedded'],
            spatial_shapes=vis_encoder_inputs['spatial_shapes'],
            level_start_index=vis_encoder_inputs['level_start_index'],
            valid_ratios=vis_encoder_inputs['valid_ratios'],
            feat_pos=vis_encoder_inputs['feat_pos'],
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
