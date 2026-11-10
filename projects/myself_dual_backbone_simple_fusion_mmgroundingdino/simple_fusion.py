"""Minimal replacements for direct VIS/IR feature addition."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class ConcatLinearFusion(nn.Module):
    """Concatenate flattened VIS/IR tokens and project 2C back to C."""

    def __init__(self, embed_dims: int = 256) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.modal_projection = nn.Linear(2 * embed_dims, embed_dims)

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        del text_feat, text_token_mask
        if vis_feat.shape != ir_feat.shape:
            raise ValueError('VIS and IR feature shapes must match, but got '
                             f'{tuple(vis_feat.shape)} and '
                             f'{tuple(ir_feat.shape)}.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError(
                f'Expected VIS/IR features shaped [B, N, {self.embed_dims}], '
                f'but got {tuple(vis_feat.shape)}.')

        fused = self.modal_projection(torch.cat([vis_feat, ir_feat], dim=-1))
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].to(torch.bool), 0)
        return fused


@MODELS.register_module()
class ExpectationFusion(nn.Module):
    """Fuse raw VIS/IR tokens with text-conditioned soft expectations."""

    def __init__(self,
                 embed_dims: int = 256,
                 token_temperature: float = 1.0,
                 modality_temperature: float = 1.0,
                 preserve_sum_scale: bool = True) -> None:
        super().__init__()
        if token_temperature <= 0:
            raise ValueError('token_temperature must be positive, but got '
                             f'{token_temperature}.')
        if modality_temperature <= 0:
            raise ValueError('modality_temperature must be positive, but got '
                             f'{modality_temperature}.')

        self.embed_dims = embed_dims
        self.vis_similarity_bn = nn.BatchNorm1d(embed_dims)
        self.ir_similarity_bn = nn.BatchNorm1d(embed_dims)
        self.token_temperature = float(token_temperature)
        self.modality_temperature = float(modality_temperature)
        self.preserve_sum_scale = bool(preserve_sum_scale)

    @staticmethod
    def _normalize_for_similarity(feature: Tensor,
                                  normalization: nn.BatchNorm1d) -> Tensor:
        return normalization(feature.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def _valid_masks(
            visual: Tensor, text: Tensor, feat_mask: Optional[Tensor],
            text_token_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        batch_size, num_positions, _ = visual.shape
        if feat_mask is None:
            valid_positions = torch.ones(
                (batch_size, num_positions),
                dtype=torch.bool,
                device=visual.device)
        else:
            if feat_mask.shape != (batch_size, num_positions):
                raise ValueError('feat_mask must have shape [B, N], but got '
                                 f'{tuple(feat_mask.shape)}.')
            valid_positions = ~feat_mask.to(torch.bool)

        if text_token_mask is None:
            valid_tokens = torch.ones(
                text.shape[:2], dtype=torch.bool, device=text.device)
        else:
            if text_token_mask.shape != text.shape[:2]:
                raise ValueError(
                    'text_token_mask must have shape [B, L], but got '
                    f'{tuple(text_token_mask.shape)}.')
            valid_tokens = text_token_mask.to(torch.bool)

        if not valid_positions.any(dim=1).all():
            raise ValueError('Every image must contain a valid visual position.')
        if not valid_tokens.any(dim=1).all():
            raise ValueError('Every text sequence must contain a valid token.')
        return valid_positions, valid_tokens

    def _token_expectation(
            self, visual: Tensor, text: Tensor, normalization: nn.BatchNorm1d,
            valid_positions: Tensor,
            valid_tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        visual_for_similarity = self._normalize_for_similarity(
            visual, normalization)
        similarity = torch.einsum('bnc,blc->bnl', visual_for_similarity, text)

        position_weight = valid_positions[:, :, None].to(similarity.dtype)
        position_count = position_weight.sum(dim=1).clamp_min(1.0)
        token_evidence = (
            similarity * position_weight).sum(dim=1) / position_count
        token_evidence = token_evidence.masked_fill(
            ~valid_tokens, float('-inf'))
        token_weight = torch.softmax(
            token_evidence.float() / self.token_temperature,
            dim=-1).to(similarity.dtype)
        alignment = (similarity * token_weight[:, None, :]).sum(dim=-1)
        return similarity, token_weight, alignment

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Tensor,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None,
                return_weights: bool = False):
        if vis_feat.shape != ir_feat.shape:
            raise ValueError('VIS and IR feature shapes must match, but got '
                             f'{tuple(vis_feat.shape)} and '
                             f'{tuple(ir_feat.shape)}.')
        if vis_feat.ndim != 3 or text_feat.ndim != 3:
            raise ValueError('Visual and text features must both be 3D.')
        if vis_feat.size(0) != text_feat.size(0):
            raise ValueError('Visual and text batch sizes must match.')
        if vis_feat.size(-1) != self.embed_dims or \
                text_feat.size(-1) != self.embed_dims:
            raise ValueError(
                f'Expected feature dimension {self.embed_dims}, but got '
                f'visual={vis_feat.size(-1)} and '
                f'text={text_feat.size(-1)}.')

        valid_positions, valid_tokens = self._valid_masks(
            vis_feat, text_feat, feat_mask, text_token_mask)
        _, vis_token_weight, vis_alignment = self._token_expectation(
            vis_feat, text_feat, self.vis_similarity_bn, valid_positions,
            valid_tokens)
        _, ir_token_weight, ir_alignment = self._token_expectation(
            ir_feat, text_feat, self.ir_similarity_bn, valid_positions,
            valid_tokens)

        modality_logits = torch.stack(
            [vis_alignment, ir_alignment], dim=-1)
        modality_weight = torch.softmax(
            modality_logits.float() / self.modality_temperature,
            dim=-1).to(vis_feat.dtype)
        fused = (modality_weight[..., 0:1] * vis_feat +
                 modality_weight[..., 1:2] * ir_feat)
        if self.preserve_sum_scale:
            fused = fused * 2.0
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].to(torch.bool), 0)

        if return_weights:
            return (fused, vis_token_weight, ir_token_weight,
                    modality_weight)
        return fused
