from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


class TextGuidedVisualBackProjection(nn.Module):
    """Use text queries to aggregate visual tokens and project them back.

    This module implements the agreed operation

    ``visual + A.transpose(-2, -1) @ (A @ V)``

    independently in every attention head. ``A`` is normalized only in the
    text-to-visual direction. Its transpose is used directly without a second
    softmax or other normalization.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads, but '
                             f'got {embed_dims} and {num_heads}.')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5

        self.visual_norm = nn.LayerNorm(embed_dims)
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dims)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, seq_len, self.embed_dims)

    def forward(self,
                visual: Tensor,
                text: Tensor,
                visual_pos: Optional[Tensor] = None,
                visual_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        if visual.ndim != 3 or text.ndim != 3:
            raise ValueError('visual and text must both be [B, length, D].')
        if visual.size(0) != text.size(0):
            raise ValueError('visual and text batch sizes must match.')
        if visual.size(-1) != self.embed_dims or \
                text.size(-1) != self.embed_dims:
            raise ValueError(
                f'Expected feature dimension {self.embed_dims}, but got '
                f'visual={visual.size(-1)} and text={text.size(-1)}.')

        visual_norm = self.visual_norm(visual)

        visual_for_key = visual_norm
        if visual_pos is not None:
            visual_for_key = visual_for_key + visual_pos

        query = self._split_heads(self.q_proj(text))
        key = self._split_heads(self.k_proj(visual_for_key))
        value = self._split_heads(self.v_proj(visual_norm))

        attention_logits = torch.matmul(
            query, key.transpose(-2, -1)) * self.scale
        if visual_mask is not None:
            visual_mask = visual_mask.to(torch.bool)
            attention_logits = attention_logits.masked_fill(
                visual_mask[:, None, None, :],
                torch.finfo(attention_logits.dtype).min)

        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.attn_dropout(attention)

        text_visual = torch.matmul(attention, value)
        if text_token_mask is not None:
            valid_text = text_token_mask[:, None, :, None].to(
                dtype=text_visual.dtype)
            text_visual = text_visual * valid_text

        back_projected = torch.matmul(
            attention.transpose(-2, -1), text_visual)
        back_projected = self._merge_heads(back_projected)
        back_projected = self.out_dropout(self.out_proj(back_projected))

        output = visual + back_projected
        if visual_mask is not None:
            output = output.masked_fill(visual_mask[..., None], 0)
        return output


class BidirectionalCrossModalAttention(nn.Module):
    """Perform VIS-to-IR and IR-to-VIS cross attention in parallel."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.vis_norm = nn.LayerNorm(embed_dims) 
        self.ir_norm = nn.LayerNorm(embed_dims) 
        self.vis_from_ir = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        self.ir_from_vis = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                vis: Tensor,
                ir: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if vis.shape != ir.shape:
            raise ValueError(
                f'VIS and IR feature shapes must match, got {vis.shape} and '
                f'{ir.shape}.')

        vis_norm = self.vis_norm(vis)
        ir_norm = self.ir_norm(ir)

        vis_query = vis_norm if feat_pos is None else vis_norm + feat_pos
        ir_query = ir_norm if feat_pos is None else ir_norm + feat_pos

        vis_delta, _ = self.vis_from_ir(
            query=vis_query,
            key=ir_query,
            value=ir_norm,
            key_padding_mask=feat_mask,
            need_weights=False)
        ir_delta, _ = self.ir_from_vis(
            query=ir_query,
            key=vis_query,
            value=vis_norm,
            key_padding_mask=feat_mask,
            need_weights=False)

        vis_enhanced = vis + self.dropout(vis_delta)
        ir_enhanced = ir + self.dropout(ir_delta)
        if feat_mask is not None:
            padding = feat_mask[..., None]
            vis_enhanced = vis_enhanced.masked_fill(padding, 0)
            ir_enhanced = ir_enhanced.masked_fill(padding, 0)
        return vis_enhanced, ir_enhanced


@MODELS.register_module()
class MultimodalFusionBlock(nn.Module):
    """Fuse flattened multi-scale VIS, IR and text features."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError('num_feature_levels must be positive.')

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.vis_text_guided_layers = nn.ModuleList([
            TextGuidedVisualBackProjection(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout) for _ in range(num_feature_levels)
        ])
        self.ir_text_guided_layers = nn.ModuleList([
            TextGuidedVisualBackProjection(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout) for _ in range(num_feature_levels)
        ])
        self.cross_modal_layers = nn.ModuleList([
            BidirectionalCrossModalAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout) for _ in range(num_feature_levels)
        ])
        self.modal_projection = nn.Linear(2 * embed_dims, embed_dims)

    def _split_levels(
            self, tensor: Optional[Tensor],
            level_lengths: Tuple[int, ...]) -> Tuple[Optional[Tensor], ...]:
        if tensor is None:
            return (None, ) * len(level_lengths)
        return torch.split(tensor, level_lengths, dim=1)

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Tensor,
                spatial_shapes: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [num_levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} feature levels, but got '
                f'{spatial_shapes.size(0)}.')
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS and IR feature shapes must match, got {vis_feat.shape} '
                f'and {ir_feat.shape}.')

        level_lengths = tuple(
            int(length) for length in spatial_shapes.prod(dim=1).tolist())
        if sum(level_lengths) != vis_feat.size(1):
            raise ValueError(
                'The sum of flattened spatial shapes must equal the visual '
                f'token length, but got {sum(level_lengths)} and '
                f'{vis_feat.size(1)}.')

        vis_levels = self._split_levels(vis_feat, level_lengths)
        ir_levels = self._split_levels(ir_feat, level_lengths)
        pos_levels = self._split_levels(feat_pos, level_lengths)
        mask_levels = self._split_levels(feat_mask, level_lengths)

        vis_enhanced_levels = []
        ir_enhanced_levels = []
        for level_idx in range(self.num_feature_levels):
            vis_text = self.vis_text_guided_layers[level_idx](
                visual=vis_levels[level_idx],
                text=text_feat,
                visual_pos=pos_levels[level_idx],
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask)
            ir_text = self.ir_text_guided_layers[level_idx](
                visual=ir_levels[level_idx],
                text=text_feat,
                visual_pos=pos_levels[level_idx],
                visual_mask=mask_levels[level_idx],
                text_token_mask=text_token_mask)

            vis_enhanced, ir_enhanced = self.cross_modal_layers[level_idx](
                vis=vis_text,
                ir=ir_text,
                feat_pos=pos_levels[level_idx],
                feat_mask=mask_levels[level_idx])
            vis_enhanced_levels.append(vis_enhanced)
            ir_enhanced_levels.append(ir_enhanced)

        vis_enhanced = torch.cat(vis_enhanced_levels, dim=1)
        ir_enhanced = torch.cat(ir_enhanced_levels, dim=1)

        fused = self.modal_projection(
            torch.cat([vis_enhanced, ir_enhanced], dim=-1))

        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None], 0)
        return fused
