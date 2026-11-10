"""Combine pre-encoder deformable fusion and post-encoder EAH objectness."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_dual_backbone_deformable_no_expectation_mmgroundingdino import (  # noqa: E501
    DualBackboneDeformableNoExpectationGroundingDINO, )
from projects.myself_dual_backbone_post_encoder_eah_mmgroundingdino import (
    DualBackbonePostEncoderEAHGroundingDINO, )


@MODELS.register_module()
class DualBackboneDeformableNoExpectationPostEncoderEAHObjectnessGroundingDINO(  # noqa: E501
        DualBackboneDeformableNoExpectationGroundingDINO):
    """Use deformable VIS/IR fusion before encoder and EAH Top-K after it.

    The pre-encoder path is inherited unchanged from the no-expectation v2
    model. EAH only replaces the scalar used to rank encoder proposals. The
    selected original token logits, boxes, learned query content and decoder
    remain unchanged. During training, selected EAH logits are exposed to the
    objectness head for its auxiliary focal loss.
    """

    def __init__(self,
                 *args,
                 eah_exclude_cls_token: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eah_exclude_cls_token = bool(eah_exclude_cls_token)

    # Reuse the parameter-free, per-level EAH score computation verbatim. It
    # only depends on eah_exclude_cls_token and is independent of Base fusion.
    _eah_proposal_scores = \
        DualBackbonePostEncoderEAHGroundingDINO._eah_proposal_scores

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        batch_size, num_candidates, _ = memory.shape
        if num_candidates < self.num_queries:
            raise ValueError(
                f'Cannot select {self.num_queries} queries from only '
                f'{num_candidates} encoder positions.')

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        eah_score = self._eah_proposal_scores(
            enc_outputs_class, text_token_mask, memory_mask, spatial_shapes)
        topk_indices = torch.topk(
            eah_score, k=self.num_queries, dim=1)[1]

        cls_out_features = enc_outputs_class.size(-1)
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        topk_eam_logits = torch.gather(eah_score, 1, topk_indices)

        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        decoder_inputs = dict(
            query=query,
            memory=memory,
            reference_points=reference_points.sigmoid(),
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask)
        head_inputs = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            eam_logits=topk_eam_logits,
            dn_meta=dn_meta) if self.training else dict()
        head_inputs['memory_text'] = memory_text
        head_inputs['text_token_mask'] = text_token_mask
        return decoder_inputs, head_inputs
