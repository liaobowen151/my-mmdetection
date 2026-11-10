"""Grounding DINO with expectation-guided encoder proposal ranking."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_dual_backbone_base_mmgroundingdino import (
    DualBackboneBaseGroundingDINO, )


@MODELS.register_module()
class DualBackbonePostEncoderEAHGroundingDINO(DualBackboneBaseGroundingDINO):
    """Use EAH scores only to rank two-stage encoder proposals.

    The detector keeps the direct-addition RGB/IR baseline unchanged.  After
    the Grounding DINO encoder, the original token logits are reused to form
    one expectation-guided score per spatial position.  Only the score passed
    to ``topk`` is replaced; the selected original token logits are still
    passed to the unmodified Grounding DINO head and losses.

    Args:
        eah_exclude_cls_token: Exclude the first BERT token (``[CLS]``) when
            estimating token importance.  The token is excluded only from EAH
            ranking; the original classification logits remain untouched.
    """

    def __init__(self,
                 *args,
                 eah_exclude_cls_token: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eah_exclude_cls_token = bool(eah_exclude_cls_token)

    def _eah_proposal_scores(
        self,
        token_logits: Tensor,
        text_token_mask: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
    ) -> Tensor:
        """Compute a per-level expectation score without adding parameters.

        Args:
            token_logits: Original encoder token logits padded to
                ``max_text_len``, with shape ``[B, N, max_text_len]``.
            text_token_mask: Valid-token mask with shape ``[B, L]``.
            memory_mask: Padded-position mask with shape ``[B, N]``.
            spatial_shapes: Feature shapes ``[num_levels, 2]`` in the same
                flattened order as the encoder memory.

        Returns:
            Tensor: EAH proposal ranking scores with shape ``[B, N]``.
        """
        if token_logits.ndim != 3:
            raise ValueError('token_logits must have shape [B, N, T].')
        if text_token_mask.ndim != 2:
            raise ValueError('text_token_mask must have shape [B, L].')
        if token_logits.size(0) != text_token_mask.size(0):
            raise ValueError('Token-logit and text-mask batches must match.')

        num_text_tokens = text_token_mask.size(1)
        if num_text_tokens > token_logits.size(2):
            raise ValueError('The text mask is longer than the token logits.')

        # Work only on the real BERT sequence. ContrastiveEmbed pads the rest
        # of its output to max_text_len with -inf values.
        similarity = token_logits[..., :num_text_tokens]
        valid_tokens = text_token_mask.to(torch.bool).clone()
        if self.eah_exclude_cls_token:
            valid_tokens[:, 0] = False
        if not valid_tokens.any(dim=1).all():
            raise ValueError('EAH requires at least one valid non-CLS token.')

        batch_size, num_positions, _ = similarity.shape
        if memory_mask is None:
            valid_positions = torch.ones((batch_size, num_positions),
                                         dtype=torch.bool,
                                         device=similarity.device)
        else:
            if memory_mask.shape != (batch_size, num_positions):
                raise ValueError('memory_mask must have shape [B, N].')
            valid_positions = ~memory_mask.to(torch.bool)

        level_sizes = [
            int(height) * int(width)
            for height, width in spatial_shapes.tolist()
        ]
        if sum(level_sizes) != num_positions:
            raise ValueError(
                'spatial_shapes do not cover all encoder positions: '
                f'{sum(level_sizes)} versus {num_positions}.')

        level_scores = []
        level_start = 0
        token_mask_3d = valid_tokens[:, None, :]
        for level_size in level_sizes:
            level_end = level_start + level_size
            level_similarity = similarity[:, level_start:level_end]
            level_valid_positions = valid_positions[:, level_start:level_end]

            #将无效token设为0
            safe_similarity = level_similarity.masked_fill(~token_mask_3d, 0.0)
            position_weight = level_valid_positions[:, :, None].to(
                safe_similarity.dtype)
            position_count = position_weight.sum(dim=1).clamp_min(1.0)
            token_evidence = (safe_similarity * position_weight).sum(dim=1)
            token_evidence = token_evidence / position_count
            token_evidence = token_evidence.masked_fill(
                ~valid_tokens, float('-inf'))

            # ExpAlign-style token expectation with temperature fixed at 1.
            token_weight = torch.softmax(
                token_evidence.float(), dim=-1).to(similarity.dtype)
            score = (safe_similarity * token_weight[:, None, :]).sum(dim=-1)
            score = score.masked_fill(~level_valid_positions, float('-inf'))
            level_scores.append(score)
            level_start = level_end

        return torch.cat(level_scores, dim=1)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """Select proposals with EAH and keep all downstream logits intact."""
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

        # Scheme A: EAH replaces only the scalar used by the discrete Top-K.
        # The full, original enc_outputs_class tensor is gathered below.
        eah_score = self._eah_proposal_scores(enc_outputs_class,
                                              text_token_mask, memory_mask,
                                              spatial_shapes)
        topk_indices = torch.topk(eah_score, k=self.num_queries, dim=1)[1]

        cls_out_features = enc_outputs_class.size(-1)
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # Keep the original Grounding DINO learned query content and decoder.
        query = self.query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
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
            dn_meta=dn_meta) if self.training else dict()
        head_inputs['memory_text'] = memory_text
        head_inputs['text_token_mask'] = text_token_mask
        return decoder_inputs, head_inputs
