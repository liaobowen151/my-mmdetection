"""Post-encoder EAH Top-K with proposal-level EAM supervision."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_dual_backbone_post_encoder_eah_mmgroundingdino import (
    DualBackbonePostEncoderEAHGroundingDINO, )


@MODELS.register_module()
class DualBackbonePostEncoderEAHObjectnessGroundingDINO(
        DualBackbonePostEncoderEAHGroundingDINO):
    """Supervise the EAH scores of the selected encoder proposals.

    EAH continues to select the matching-query reference boxes. During
    training, the selected scalar EAM logits are additionally forwarded to
    the bbox head for phrase/objectness supervision. Visual memory, original
    token logits, learned query content, and the decoder remain unchanged.
    """

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
