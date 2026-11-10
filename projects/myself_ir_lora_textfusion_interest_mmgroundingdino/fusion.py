"""TextFusion-inspired feature fusion and interest-region supervision."""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder, )
from mmdet.registry import MODELS


@MODELS.register_module()
class TextFusionInterestFusion(nn.Module):
    """IR-to-VIS enhancement followed by direct-text or gated IR fusion."""

    def __init__(self,
                 fusion_mode: str,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 gate_hidden_dims: int = 256,
                 gate_init_bias: float = 2.0) -> None:
        super().__init__()
        if fusion_mode not in ('direct_text', 'direct_text_projected',
                               'dynamic_gate', 'dynamic_gate_identity'):
            raise ValueError(f'Unsupported fusion_mode: {fusion_mode}')
        self.fusion_mode = fusion_mode
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.vis_norm = nn.LayerNorm(embed_dims)
        self.ir_norm = nn.LayerNorm(embed_dims)
        self.vis_from_ir = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)

        if fusion_mode in ('direct_text_projected', 'dynamic_gate'):
            self.vis_proj = nn.Linear(embed_dims, embed_dims, bias=False)
            self.ir_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        if fusion_mode in ('dynamic_gate', 'dynamic_gate_identity'):
            self.gate_mlp = nn.Sequential(
                nn.Linear(embed_dims * 3, gate_hidden_dims),
                nn.ReLU(inplace=True), nn.Linear(gate_hidden_dims, embed_dims))
            self.gate_init_bias = float(gate_init_bias)

        self.register_buffer('fusion_alpha', torch.tensor(0.1))
        self.register_buffer('interest_loss_weight', torch.tensor(0.1))
        self._interest_cache = None
        self.last_gate_statistics = None

    def init_weights(self) -> None:
        self.vis_from_ir.init_weights()
        if self.fusion_mode in ('direct_text_projected', 'dynamic_gate'):
            nn.init.eye_(self.vis_proj.weight)
            nn.init.eye_(self.ir_proj.weight)
        if self.fusion_mode in ('dynamic_gate', 'dynamic_gate_identity'):
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, self.gate_init_bias)

    def set_schedule(self, alpha: float, interest_weight: float) -> None:
        self.fusion_alpha.fill_(float(alpha))
        self.interest_loss_weight.fill_(float(interest_weight))

    @staticmethod
    def _global_text(text_feat: Tensor, text_token_mask: Tensor) -> Tensor:
        """Mean-pool full REC words, excluding CLS, final dot and SEP."""
        valid = text_token_mask.to(torch.bool).clone()
        lengths = valid.sum(dim=1)
        positions = torch.arange(valid.size(1), device=valid.device)[None]
        valid &= positions > 0
        valid &= positions < (lengths[:, None] - 2)
        # Fall back to all non-padding tokens except CLS/SEP for unusual text.
        empty = valid.sum(dim=1) == 0
        if empty.any():
            fallback = text_token_mask.to(torch.bool).clone()
            fallback &= positions > 0
            fallback &= positions < (lengths[:, None] - 1)
            valid[empty] = fallback[empty]
        weights = valid.to(text_feat.dtype)
        pooled = (text_feat * weights[..., None]).sum(dim=1)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled.detach()

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
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes)
        if text_token_mask is None:
            raise ValueError('text_token_mask is required.')
        reference_points = \
            DeformableDetrTransformerEncoder.get_encoder_reference_points(
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                device=vis_feat.device)
        vis_enhanced = self.vis_from_ir(
            query=self.vis_norm(vis_feat),
            value=self.ir_norm(ir_feat),
            identity=vis_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        text_global = self._global_text(text_feat, text_token_mask)
        text_spatial = text_global[:, None, :].expand_as(ir_feat)
        alpha = self.fusion_alpha.to(ir_feat.dtype)
        gate = None
        if self.fusion_mode == 'direct_text':
            fused = vis_enhanced + alpha * text_spatial * ir_feat
        elif self.fusion_mode == 'direct_text_projected':
            fused = (
                self.vis_proj(vis_enhanced) +
                alpha * text_spatial * self.ir_proj(ir_feat))
        elif self.fusion_mode == 'dynamic_gate':
            gate_input = torch.cat([vis_enhanced, ir_feat, text_spatial],
                                   dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input))
            fused = (
                self.vis_proj(vis_enhanced) +
                alpha * gate * self.ir_proj(ir_feat))
        else:
            gate_input = torch.cat([vis_enhanced, ir_feat, text_spatial],
                                   dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input))
            fused = vis_enhanced + alpha * gate * ir_feat

        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].bool(), 0)
        if self.training:
            self._interest_cache = dict(
                vis=vis_feat,
                ir=ir_feat,
                fused=fused,
                spatial_shapes=spatial_shapes,
                feat_mask=feat_mask,
                gate=gate)
        return fused

    def interest_loss(self, interest_masks: Sequence[Tensor]) -> Tensor:
        """Compute L0 activity-weighted loss on all four feature levels."""
        if self._interest_cache is None:
            raise RuntimeError('Fusion forward must run before interest_loss.')
        cache = self._interest_cache
        shapes = [(int(h), int(w)) for h, w in cache['spatial_shapes']]
        counts = [h * w for h, w in shapes]
        vis_levels = cache['vis'].split(counts, dim=1)
        ir_levels = cache['ir'].split(counts, dim=1)
        fused_levels = cache['fused'].split(counts, dim=1)
        mask_levels = (
            cache['feat_mask'].split(counts, dim=1)
            if cache['feat_mask'] is not None else [None] * len(counts))

        # The validated L0 activity map determines modality weights at every
        # level. Detach it so this target cannot be optimized trivially.
        h0, w0 = shapes[0]
        a_vis = vis_levels[0].detach().abs().mean(dim=-1).reshape(-1, h0, w0)
        a_ir = ir_levels[0].detach().abs().mean(dim=-1).reshape(-1, h0, w0)
        weights0 = torch.softmax(torch.stack([a_vis, a_ir], dim=1), dim=1)

        batch_masks = torch.stack([
            mask.to(device=cache['fused'].device, dtype=torch.float32)
            for mask in interest_masks
        ])
        if batch_masks.ndim == 3:
            batch_masks = batch_masks[:, None]
        batch_masks = (batch_masks > 0).to(cache['fused'].dtype)

        total = cache['fused'].new_zeros(())
        gate_interest = []
        gate_noninterest = []
        gate_levels = (
            cache['gate'].split(counts, dim=1)
            if cache['gate'] is not None else [None] * len(counts))
        for (h, w), vis, ir, fused, padding in zip(shapes, vis_levels,
                                                   ir_levels, fused_levels,
                                                   mask_levels):
            weights = F.interpolate(weights0, size=(h, w), mode='area')
            weights = weights / weights.sum(
                dim=1, keepdim=True).clamp_min(1e-6)
            target = F.interpolate(batch_masks, size=(h, w), mode='area')[:, 0]
            if padding is not None:
                target = target * (~padding.reshape(-1, h, w).bool()).to(
                    target.dtype)

            vis_n = F.layer_norm(vis.detach(), (self.embed_dims, ))
            ir_n = F.layer_norm(ir.detach(), (self.embed_dims, ))
            fused_n = F.layer_norm(fused, (self.embed_dims, ))
            d_vis = (fused_n - vis_n).abs().mean(dim=-1).reshape(-1, h, w)
            d_ir = (fused_n - ir_n).abs().mean(dim=-1).reshape(-1, h, w)
            distance = weights[:, 0] * d_vis + weights[:, 1] * d_ir
            numerator = (target * distance).sum(dim=(1, 2))
            denominator = target.sum(dim=(1, 2)).clamp_min(1e-6)
            total = total + (numerator / denominator).mean()

        if cache['gate'] is not None:
            for (h, w), gate, padding in zip(shapes, gate_levels, mask_levels):
                target = F.interpolate(
                    batch_masks, size=(h, w), mode='area')[:, 0]
                valid = torch.ones_like(target)
                if padding is not None:
                    valid = (~padding.reshape(-1, h, w).bool()).to(
                        target.dtype)
                gate_map = gate.detach().mean(dim=-1).reshape(-1, h, w)
                interest = target * valid
                noninterest = (1 - target) * valid
                gate_interest.append((gate_map * interest).sum() /
                                     interest.sum().clamp_min(1e-6))
                gate_noninterest.append((gate_map * noninterest).sum() /
                                        noninterest.sum().clamp_min(1e-6))
            gate = cache['gate'].detach()
            self.last_gate_statistics = dict(
                gate_mean=gate.mean(),
                gate_std=gate.std(),
                gate_min=gate.min(),
                gate_max=gate.max(),
                gate_interest_mean=torch.stack(gate_interest).mean(),
                gate_noninterest_mean=torch.stack(gate_noninterest).mean())
        else:
            self.last_gate_statistics = None

        self._interest_cache = None
        return total / len(shapes) * self.interest_loss_weight.to(total.dtype)

    def _validate_inputs(self, vis_feat: Tensor, ir_feat: Tensor,
                         spatial_shapes: Tensor) -> None:
        if vis_feat.shape != ir_feat.shape:
            raise ValueError('VIS and IR feature shapes must match.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError('Features must have shape [B, N, embed_dims].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError('Unexpected number of feature levels.')
