"""Bidirectional deformable enhancement followed by tri-prior fusion."""

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
class BidirectionalDeformableTPFFusion(nn.Module):
    """Enhance VIS/IR mutually, then fuse them with IAFv3-style TPF.

    The three reliability priors are:

    - illumination: a per-level spatial prior computed from the RGB image;
    - local: a learned per-token gate from enhanced VIS/IR features;
    - global: a learned channel gate from mask-aware pooled features.

    Unlike the ViT implementation, Grounding DINO has no visual CLS token.
    The output therefore contains only the fused multi-scale spatial tokens.
    Text is intentionally consumed by the unchanged Grounding DINO encoder
    after this module, matching the placement of the former sum fusion.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 rgb_mean: Sequence[float] = (123.675, 116.28, 103.53),
                 rgb_std: Sequence[float] = (58.395, 57.12, 57.375),
                 illumination_pool_size: int = 8) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError('num_feature_levels must be positive.')
        if illumination_pool_size <= 0:
            raise ValueError('illumination_pool_size must be positive.')
        if len(rgb_mean) != 3 or len(rgb_std) != 3:
            raise ValueError('rgb_mean and rgb_std must each have 3 values.')

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.illumination_pool_size = illumination_pool_size

        self.vis_norm = nn.LayerNorm(embed_dims)
        self.ir_norm = nn.LayerNorm(embed_dims)
        attention_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        self.vis_from_ir = MultiScaleDeformableAttention(**attention_cfg)
        self.ir_from_vis = MultiScaleDeformableAttention(**attention_cfg)

        self.tpf_token_gate = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
        )
        self.tpf_channel_gate = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.tpf_alpha = nn.Parameter(torch.tensor(1.0))
        self.tpf_beta = nn.Parameter(torch.tensor(0.9))
        self.tpf_gamma = nn.Parameter(torch.tensor(0.6))
        self.tpf_delta = nn.Parameter(torch.tensor(0.2))
        self.tpf_temperature = nn.Parameter(torch.tensor(1.0))

        self.register_buffer(
            'rgb_mean',
            torch.tensor(rgb_mean).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'rgb_std',
            torch.tensor(rgb_std).view(1, 3, 1, 1),
            persistent=False)

    def init_weights(self) -> None:
        """Use MMCV initialization for both deformable attention paths."""
        self.vis_from_ir.init_weights()
        self.ir_from_vis.init_weights()

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Optional[Tensor],
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                rgb_image: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        # TPF itself has no direct text branch. The fused tokens and text are
        # jointly processed by the original Grounding DINO encoder afterwards.
        del text_feat, text_token_mask
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes, feat_mask,
                              rgb_image)

        reference_points = \
            DeformableDetrTransformerEncoder.get_encoder_reference_points(
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                device=vis_feat.device)
        vis_attn_input = self.vis_norm(vis_feat)
        ir_attn_input = self.ir_norm(ir_feat)

        vis_enhanced = self.vis_from_ir(
            query=vis_attn_input,
            value=ir_attn_input,
            identity=vis_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)
        ir_enhanced = self.ir_from_vis(
            query=ir_attn_input,
            value=vis_attn_input,
            identity=ir_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        fused = self._tri_prior_fusion(vis_enhanced, ir_enhanced, rgb_image,
                                       spatial_shapes, level_start_index,
                                       feat_mask)
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].bool(), 0)
        return fused

    def _tri_prior_fusion(self, vis_feat: Tensor, ir_feat: Tensor,
                          rgb_image: Tensor, spatial_shapes: Tensor,
                          level_start_index: Tensor,
                          feat_mask: Optional[Tensor]) -> Tensor:
        illumination = self._illumination_prior(rgb_image, spatial_shapes,
                                                level_start_index, feat_mask)
        illumination_confidence = 1.0 - torch.clamp(
            4.0 * (illumination - 0.5).square(), min=0.0, max=1.0)

        # Match the original IAFv3: normalize features only for estimating
        # priors, while mixing the unnormalized enhanced features.
        vis_normalized = F.layer_norm(vis_feat, (vis_feat.shape[-1], ))
        ir_normalized = F.layer_norm(ir_feat, (ir_feat.shape[-1], ))
        local_input = torch.cat([
            vis_normalized, ir_normalized,
            torch.abs(vis_normalized - ir_normalized)
        ],
                                dim=-1)
        temperature = torch.clamp(self.tpf_temperature, min=0.5, max=2.0)
        local_gate = torch.sigmoid(
            self.tpf_token_gate(local_input) / temperature)

        valid = self._valid_token_weights(vis_feat, feat_mask)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        vis_global = (vis_normalized * valid).sum(dim=1) / denominator
        ir_global = (ir_normalized * valid).sum(dim=1) / denominator
        global_input = torch.cat(
            [vis_global, ir_global,
             torch.abs(vis_global - ir_global)], dim=-1)
        channel_gate = torch.sigmoid(
            self.tpf_channel_gate(global_input)).unsqueeze(1)
        global_bias = channel_gate.mean(dim=-1, keepdim=True)

        rgb_weight = torch.sigmoid(self.tpf_alpha * (illumination - 0.5) *
                                   illumination_confidence + self.tpf_beta *
                                   (local_gate - 0.5) + self.tpf_gamma *
                                   (global_bias - 0.5))

        fused = rgb_weight * vis_feat + (1.0 - rgb_weight) * ir_feat
        fused = fused + (
            torch.tanh(self.tpf_delta) * channel_gate * (vis_feat - ir_feat))
        return fused

    def _illumination_prior(self, rgb_image: Tensor, spatial_shapes: Tensor,
                            level_start_index: Tensor,
                            feat_mask: Optional[Tensor]) -> Tensor:
        # The detector receives normalized tensors. Undo RGB normalization so
        # the fixed luminance coefficients operate on physical RGB intensities.
        mean = self.rgb_mean.to(dtype=rgb_image.dtype)
        std = self.rgb_std.to(dtype=rgb_image.dtype)
        rgb = (rgb_image * std + mean).div(255.0).clamp(0.0, 1.0)
        gray = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] +
                0.114 * rgb[:, 2:3])
        gray = F.avg_pool2d(
            gray,
            kernel_size=self.illumination_pool_size,
            stride=self.illumination_pool_size)

        level_priors = []
        for level in range(spatial_shapes.size(0)):
            height = int(spatial_shapes[level, 0].item())
            width = int(spatial_shapes[level, 1].item())
            level_map = F.interpolate(
                gray,
                size=(height, width),
                mode='bilinear',
                align_corners=False).flatten(2).transpose(1, 2)

            start = int(level_start_index[level].item())
            end = start + height * width
            level_mask = None if feat_mask is None else feat_mask[:, start:end]
            valid = self._valid_token_weights(level_map, level_mask)
            denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_l = (level_map * valid).sum(dim=1, keepdim=True) / denominator
            variance_l = ((level_map - mean_l).square() * valid).sum(
                dim=1, keepdim=True) / denominator
            level_map = (level_map - mean_l) / torch.sqrt(variance_l + 1e-6)
            level_map = torch.sigmoid(level_map)
            if level_mask is not None:
                # Neutral prior at padding positions; they are zeroed later.
                level_map = level_map.masked_fill(level_mask[..., None].bool(),
                                                  0.5)
            level_priors.append(level_map)
        return torch.cat(level_priors, dim=1)

    @staticmethod
    def _valid_token_weights(feat: Tensor,
                             feat_mask: Optional[Tensor]) -> Tensor:
        if feat_mask is None:
            return feat.new_ones((*feat.shape[:2], 1))
        return (~feat_mask.bool()).unsqueeze(-1).to(dtype=feat.dtype)

    def _validate_inputs(self, vis_feat: Tensor, ir_feat: Tensor,
                         spatial_shapes: Tensor, feat_mask: Optional[Tensor],
                         rgb_image: Tensor) -> None:
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS/IR shapes differ: {vis_feat.shape} vs {ir_feat.shape}.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError('Features must have shape [B, N, embed_dims].')
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(f'Expected {self.num_feature_levels} levels, got '
                             f'{spatial_shapes.size(0)}.')
        token_count = int(spatial_shapes.prod(dim=1).sum().item())
        if token_count != vis_feat.size(1):
            raise ValueError(f'Expected {token_count} flattened tokens, got '
                             f'{vis_feat.size(1)}.')
        if feat_mask is not None and feat_mask.shape != vis_feat.shape[:2]:
            raise ValueError('feat_mask must have shape [B, N].')
        if (rgb_image.ndim != 4 or rgb_image.size(0) != vis_feat.size(0)
                or rgb_image.size(1) != 3):
            raise ValueError('rgb_image must have shape [B, 3, H, W].')
