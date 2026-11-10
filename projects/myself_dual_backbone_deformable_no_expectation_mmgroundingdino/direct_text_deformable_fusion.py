"""Direct text and deformable VIS/IR enhancement with fixed sum fusion."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_mmgroundingdino.deformable_multimodal_fusion import (  # noqa: E501
    DeformableMultimodalFusionBlock, )


class SharedDirectTextToVisualAttention(nn.Module):
    """Independent copy of v2's one-sided ``V + Attn(V, T, T)`` update."""

    def __init__(self, embed_dims=256, attention_dims=1024, num_heads=4,
                 dropout=0.0, init_values=1e-4) -> None:
        super().__init__()
        if attention_dims % num_heads != 0:
            raise ValueError('attention_dims must be divisible by num_heads.')
        self.embed_dims = embed_dims
        self.attention_dims = attention_dims
        self.num_heads = num_heads
        self.head_dims = attention_dims // num_heads
        self.scale = self.head_dims**-0.5
        self.visual_norm = nn.LayerNorm(embed_dims)
        self.text_norm = nn.LayerNorm(embed_dims)
        self.q_proj = nn.Linear(embed_dims, attention_dims)
        self.k_proj = nn.Linear(embed_dims, attention_dims)
        self.v_proj = nn.Linear(embed_dims, attention_dims)
        self.out_proj = nn.Linear(attention_dims, embed_dims)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.full((embed_dims,), init_values))

    def _split_heads(self, feature: Tensor) -> Tensor:
        batch_size, length, _ = feature.shape
        return feature.reshape(batch_size, length, self.num_heads,
                               self.head_dims).transpose(1, 2)

    def _merge_heads(self, feature: Tensor) -> Tensor:
        batch_size, _, length, _ = feature.shape
        return feature.transpose(1, 2).contiguous().reshape(
            batch_size, length, self.attention_dims)

    def forward(self, visual: Tensor, text: Tensor,
                visual_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        if visual.ndim != 3 or text.ndim != 3:
            raise ValueError('visual and text must have shape [B, N/L, C].')
        if visual.size(0) != text.size(0) or \
                visual.size(-1) != self.embed_dims or \
                text.size(-1) != self.embed_dims:
            raise ValueError('Visual/text batch and embedding dimensions must match.')
        query = self._split_heads(self.q_proj(self.visual_norm(visual)))
        text = self.text_norm(text)
        key = self._split_heads(self.k_proj(text))
        value = self._split_heads(self.v_proj(text))
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if text_token_mask is not None:
            if text_token_mask.shape != text.shape[:2]:
                raise ValueError('text_token_mask must have shape [B, L].')
            valid_tokens = text_token_mask.to(torch.bool)
            if not valid_tokens.any(dim=1).all():
                raise ValueError('Every text sequence needs a valid token.')
            logits = logits.masked_fill(~valid_tokens[:, None, None, :],
                                        torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        update = self._merge_heads(
            torch.matmul(self.attn_dropout(attention), value))
        output = visual + self.gamma * self.out_dropout(self.out_proj(update))
        if visual_mask is not None:
            if visual_mask.shape != visual.shape[:2]:
                raise ValueError('visual_mask must have shape [B, N].')
            output = output.masked_fill(visual_mask[..., None].to(torch.bool), 0)
        return output


@MODELS.register_module()
class DirectTextDeformableNoExpectationFusionBlock(
        DeformableMultimodalFusionBlock):
    """v2 enhancement stages followed by parameter-free ``VIS + IR``.

    This module deliberately retains the v2 direct text-to-visual attention
    and bidirectional deformable VIS/IR attention.  It removes all expectation
    fusion operations: no similarity BatchNorm, ``einsum``, token softmax or
    position-wise modality softmax is constructed.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 text_attention_dims: int = 1024,
                 text_attention_heads: int = 4,
                 text_attention_init_values: float = 1e-4) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            num_points=num_points)
        # Remove v1 text back-projection and concat-linear fusion.  No
        # expectation-specific parameters are added in their place.
        del self.vis_text_guided_layers
        del self.ir_text_guided_layers
        del self.modal_projection
        self.vis_text_attention = SharedDirectTextToVisualAttention(
            embed_dims=embed_dims,
            attention_dims=text_attention_dims,
            num_heads=text_attention_heads,
            dropout=dropout,
            init_values=text_attention_init_values)
        self.ir_text_attention = SharedDirectTextToVisualAttention(
            embed_dims=embed_dims,
            attention_dims=text_attention_dims,
            num_heads=text_attention_heads,
            dropout=dropout,
            init_values=text_attention_init_values)

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        if vis_feat.shape != ir_feat.shape:
            raise ValueError('VIS and IR feature shapes must match.')
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [num_levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError('Unexpected number of feature levels.')
        level_lengths = tuple(int(length)
                              for length in spatial_shapes.prod(dim=1).tolist())
        if sum(level_lengths) != vis_feat.size(1):
            raise ValueError('Flattened spatial shapes do not match token length.')

        vis_levels = self._split_levels(vis_feat, level_lengths)
        ir_levels = self._split_levels(ir_feat, level_lengths)
        mask_levels = self._split_levels(feat_mask, level_lengths)
        vis_text = torch.cat([
            self.vis_text_attention(
                visual=vis_levels[level_idx], text=text_feat,
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask)
            for level_idx in range(self.num_feature_levels)
        ], dim=1)
        ir_text = torch.cat([
            self.ir_text_attention(
                visual=ir_levels[level_idx], text=text_feat,
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask)
            for level_idx in range(self.num_feature_levels)
        ], dim=1)
        vis_enhanced, ir_enhanced = self.cross_modal_attention(
            vis=vis_text,
            ir=ir_text,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask)
        fused = vis_enhanced + ir_enhanced
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].to(torch.bool), 0)
        return fused
