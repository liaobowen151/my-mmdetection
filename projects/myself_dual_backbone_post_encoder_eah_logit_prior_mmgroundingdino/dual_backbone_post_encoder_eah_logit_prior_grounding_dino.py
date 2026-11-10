"""Grounding DINO with an EAH prior on encoder token logits."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from projects.myself_dual_backbone_post_encoder_eah_mmgroundingdino import (
    DualBackbonePostEncoderEAHGroundingDINO, )


@MODELS.register_module()
class DualBackbonePostEncoderEAHLogitPriorGroundingDINO(
        DualBackbonePostEncoderEAHGroundingDINO):
    """Add a bounded per-level EAH prior to encoder token logits.

    Proposal ranking is identical to ``DualBackbonePostEncoderEAHGroundingDINO``:
    the original encoder token logits are aggregated with per-level EAH token
    weights.  The additive prior is applied only to the selected encoder token
    logits passed to the detection head, so it affects the encoder Hungarian
    classification cost and ``enc_loss_cls`` without changing memory, boxes,
    decoder computation, or inference classification outputs.

    Args:
        eah_prior_scale: Scale lambda applied to the bounded log prior.
        eah_prior_max: Symmetric bound for the log prior before scaling.
        eah_prior_eps: Numerical stabilizer inside the logarithm.
        eah_prior_detach: Whether to stop gradients through the EAH prior.
    """

    def __init__(self,
                 *args,
                 eah_prior_scale: float = 0.2,
                 eah_prior_max: float = 2.0,
                 eah_prior_eps: float = 1e-6,
                 eah_prior_detach: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if eah_prior_scale < 0:
            raise ValueError('eah_prior_scale must be non-negative.')
        if eah_prior_max <= 0:
            raise ValueError('eah_prior_max must be positive.')
        if eah_prior_eps <= 0:
            raise ValueError('eah_prior_eps must be positive.')
        self.eah_prior_scale = float(eah_prior_scale)
        self.eah_prior_max = float(eah_prior_max)
        self.eah_prior_eps = float(eah_prior_eps)
        self.eah_prior_detach = bool(eah_prior_detach)

    def _eah_scores_and_prior_logits(
        self,
        token_logits: Tensor,
        text_token_mask: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Return unchanged EAH scores and prior-adjusted token logits."""
        if token_logits.ndim != 3:
            raise ValueError('token_logits must have shape [B, N, T].')
        if text_token_mask.ndim != 2:
            raise ValueError('text_token_mask must have shape [B, L].')
        if token_logits.size(0) != text_token_mask.size(0):
            raise ValueError('Token-logit and text-mask batches must match.')

        num_text_tokens = text_token_mask.size(1)
        if num_text_tokens > token_logits.size(2):
            raise ValueError('The text mask is longer than the token logits.')

        similarity = token_logits[..., :num_text_tokens]
        valid_tokens = text_token_mask.to(torch.bool).clone()
        if self.eah_exclude_cls_token:
            valid_tokens[:, 0] = False
        if not valid_tokens.any(dim=1).all():
            raise ValueError('EAH requires at least one valid non-CLS token.')

        batch_size, num_positions, _ = similarity.shape
        if memory_mask is None:
            valid_positions = torch.ones(
                (batch_size, num_positions),
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
        level_prior_logits = []
        level_start = 0
        token_mask_3d = valid_tokens[:, None, :]
        valid_token_count = valid_tokens.sum(dim=-1, keepdim=True).float()

        for level_size in level_sizes:
            level_end = level_start + level_size
            level_similarity = similarity[:, level_start:level_end]
            level_valid_positions = valid_positions[:, level_start:level_end]

            safe_similarity = level_similarity.masked_fill(
                ~token_mask_3d, 0.0)
            position_weight = level_valid_positions[:, :, None].to(
                safe_similarity.dtype)
            position_count = position_weight.sum(dim=1).clamp_min(1.0)
            token_evidence = (safe_similarity * position_weight).sum(dim=1)
            token_evidence = token_evidence / position_count
            token_evidence = token_evidence.masked_fill(
                ~valid_tokens, float('-inf'))

            # Keep the base EAH proposal score exactly unchanged.
            token_weight = torch.softmax(token_evidence.float(), dim=-1)
            score = (safe_similarity * token_weight[:, None, :].to(
                similarity.dtype)).sum(dim=-1)
            score = score.masked_fill(
                ~level_valid_positions, float('-inf'))
            level_scores.append(score)

            # log(T_valid * w_t) is zero for uniform valid-token weights.
            token_prior = torch.log(
                valid_token_count * token_weight + self.eah_prior_eps)
            token_prior = token_prior.clamp(
                min=-self.eah_prior_max, max=self.eah_prior_max)
            # CLS and invalid/padded text positions retain their original logit.
            token_prior = token_prior.masked_fill(~valid_tokens, 0.0)
            if self.eah_prior_detach:
                token_prior = token_prior.detach()
            prior_logits = level_similarity + self.eah_prior_scale * \
                token_prior[:, None, :].to(level_similarity.dtype)
            level_prior_logits.append(prior_logits)
            level_start = level_end

        eah_scores = torch.cat(level_scores, dim=1)
        prior_similarity = torch.cat(level_prior_logits, dim=1)
        # ContrastiveEmbed pads beyond the real BERT sequence with -inf.
        # Preserve that tail byte-for-byte instead of applying a prior to it.
        prior_token_logits = torch.cat(
            [prior_similarity, token_logits[..., num_text_tokens:]], dim=-1)
        return eah_scores, prior_token_logits

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """Use prior logits only for encoder matching and classification."""
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

        eah_score, prior_enc_outputs_class = \
            self._eah_scores_and_prior_logits(
                enc_outputs_class, text_token_mask, memory_mask,
                spatial_shapes)
        topk_indices = torch.topk(
            eah_score, k=self.num_queries, dim=1)[1]

        cls_out_features = prior_enc_outputs_class.size(-1)
        topk_score = torch.gather(
            prior_enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4))
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
            dn_meta=dn_meta) if self.training else dict()
        head_inputs['memory_text'] = memory_text
        head_inputs['text_token_mask'] = text_token_mask
        return decoder_inputs, head_inputs
