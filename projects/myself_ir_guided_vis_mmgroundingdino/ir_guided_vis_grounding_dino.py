"""Grounding DINO with IR proposal selection and VIS decoding."""

from typing import Dict, Tuple

import torch
from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_ir_lora_addition_mmgroundingdino import (
    IRLoRASwinTransformer, )


@MODELS.register_module()
class IRGuidedVISGroundingDINO(GroundingDINO):
    """Use IR to select proposals and VIS memory for the decoder.

    The pretrained Grounding DINO encoder is shared and executed once per
    modality. IR encoder scores select the top-k spatial indices and IR boxes
    initialize decoder reference points. The decoder query content remains the
    original learned embedding and attends only to VIS visual/text memory.
    """

    def __init__(self,
                 *args,
                 freeze_visual_backbone: bool = True,
                 freeze_language_model: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model

        if not isinstance(self.backbone, IRLoRASwinTransformer):
            raise TypeError('IRGuidedVISGroundingDINO requires an '
                            'IRLoRASwinTransformer backbone.')
        if not self.freeze_visual_backbone:
            raise ValueError('The shared base backbone must remain frozen.')
        if self.backbone.uses_activation_checkpointing():
            raise ValueError('Adapter switching requires backbone with_cp=False.')

        self.backbone.freeze_base_and_enable_lora()
        self.backbone.eval()
        if self.freeze_language_model:
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

    def extract_feat(self, batch_inputs: Tensor) -> Dict[str, Tuple[Tensor]]:
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'IRGuidedVISGroundingDINO expects [B, 6, H, W], got '
                f'{tuple(batch_inputs.shape)}.')

        self.backbone.set_ir_lora_enabled(False)
        vis_feats = self.backbone(batch_inputs[:, :3])

        self.backbone.set_ir_lora_enabled(True)
        ir_feats = self.backbone(batch_inputs[:, 3:])

        # Leave the shared backbone in its safe RGB/base state.
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
                    f'{tuple(vis_feat.shape)} vs {tuple(ir_feat.shape)}.')

    @staticmethod
    def _validate_encoder_layout(vis_inputs: Dict, ir_inputs: Dict) -> None:
        for key in ('spatial_shapes', 'level_start_index', 'valid_ratios'):
            if not torch.equal(vis_inputs[key], ir_inputs[key]):
                raise RuntimeError(f'VIS and IR {key} must be identical.')
        vis_mask = vis_inputs['feat_mask']
        ir_mask = ir_inputs['feat_mask']
        if (vis_mask is None) != (ir_mask is None):
            raise RuntimeError('VIS and IR feature masks must be identical.')
        if vis_mask is not None and not torch.equal(vis_mask, ir_mask):
            raise RuntimeError('VIS and IR feature masks must be identical.')

    def forward_transformer(self,
                            img_feats: Dict[str, Tuple[Tensor]],
                            text_dict: Dict,
                            batch_data_samples: OptSampleList = None) -> Dict:
        vis_inputs, decoder_inputs = self.pre_transformer(
            img_feats['vis_feats'], batch_data_samples)
        ir_inputs, _ = self.pre_transformer(
            img_feats['ir_feats'], batch_data_samples)
        self._validate_encoder_layout(vis_inputs, ir_inputs)

        # The same pretrained encoder parameters are used in two independent
        # forwards. Each forward updates its own visual and text memories.
        vis_outputs = self.forward_encoder(
            **vis_inputs, text_dict=dict(text_dict))
        ir_outputs = self.forward_encoder(
            **ir_inputs, text_dict=dict(text_dict))

        tmp_decoder_inputs, head_inputs = self.pre_decoder_ir_guided_vis(
            vis_outputs=vis_outputs,
            ir_outputs=ir_outputs,
            batch_data_samples=batch_data_samples)
        decoder_inputs.update(tmp_decoder_inputs)
        decoder_outputs = self.forward_decoder(**decoder_inputs)
        head_inputs.update(decoder_outputs)
        return head_inputs

    def _encoder_predictions(self, outputs: Dict) -> Tuple[Tensor, Tensor]:
        output_memory, output_proposals = \
            self.gen_encoder_output_proposals(
                outputs['memory'], outputs['memory_mask'],
                outputs['spatial_shapes'])
        cls_scores = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory, outputs['memory_text'],
                outputs['text_token_mask'])
        bbox_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        return cls_scores, bbox_unact

    @staticmethod
    def _gather_queries(values: Tensor, indices: Tensor) -> Tensor:
        gather_index = indices.unsqueeze(-1).expand(
            -1, -1, values.size(-1))
        return torch.gather(values, 1, gather_index)

    def pre_decoder_ir_guided_vis(
            self,
            vis_outputs: Dict,
            ir_outputs: Dict,
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        bs = vis_outputs['memory'].size(0)
        if vis_outputs['memory'].shape != ir_outputs['memory'].shape:
            raise RuntimeError('VIS and IR encoder memories must align.')

        ir_cls_all, ir_bbox_unact_all = self._encoder_predictions(ir_outputs)
        vis_cls_all, vis_bbox_unact_all = self._encoder_predictions(vis_outputs)

        # IR alone decides which spatial positions initialize matching queries.
        ir_topk_indices = torch.topk(
            ir_cls_all.max(-1)[0], k=self.num_queries, dim=1)[1]
        ir_topk_cls = self._gather_queries(ir_cls_all, ir_topk_indices)
        ir_topk_bbox_unact = self._gather_queries(
            ir_bbox_unact_all, ir_topk_indices)
        ir_topk_bbox = ir_topk_bbox_unact.sigmoid()

        # Supervise VIS predictions at exactly the spatial indices selected by IR.
        vis_selected_cls = self._gather_queries(vis_cls_all, ir_topk_indices)
        vis_selected_bbox = self._gather_queries(
            vis_bbox_unact_all, ir_topk_indices).sigmoid()

        # Keep the original learned DINO matching-query content. Only its
        # reference boxes are supplied by IR.
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        matching_refs_unact = ir_topk_bbox_unact.detach()
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, matching_refs_unact], dim=1)
        else:
            reference_points = matching_refs_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs = dict(
            query=query,
            memory=vis_outputs['memory'],
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=vis_outputs['memory_text'],
            text_attention_mask=~vis_outputs['text_token_mask'])

        head_inputs = dict()
        if self.training:
            head_inputs.update(
                ir_enc_outputs_class=ir_topk_cls,
                ir_enc_outputs_coord=ir_topk_bbox,
                vis_enc_outputs_class=vis_selected_cls,
                vis_enc_outputs_coord=vis_selected_bbox,
                dn_meta=dn_meta)
        head_inputs['memory_text'] = vis_outputs['memory_text']
        head_inputs['text_token_mask'] = vis_outputs['text_token_mask']
        return decoder_inputs, head_inputs
