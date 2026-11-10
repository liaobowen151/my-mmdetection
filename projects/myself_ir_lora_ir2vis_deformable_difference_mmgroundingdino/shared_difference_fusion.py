"""Shared-sampling IR/VIS deformable difference fusion modules."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder,
)
from mmdet.registry import MODELS


class _SharedSamplingDifferenceFusion(nn.Module):
    """Sample IR and VIS with one shared deformable sampling strategy."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 concat_inputs: bool = False) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError('num_feature_levels must be positive.')
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.concat_inputs = concat_inputs
        self.vis_norm = nn.LayerNorm(embed_dims)
        self.ir_norm = nn.LayerNorm(embed_dims)
        self.vis_from_ir = MultiScaleDeformableAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_feature_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=True)
        if concat_inputs:
            # [z_ir - z_vis, z_ir, z_vis] has 3 * embed_dims channels.
            self.vis_from_ir.output_proj = nn.Linear(
                3 * embed_dims, embed_dims)
            nn.init.xavier_uniform_(self.vis_from_ir.output_proj.weight)
            nn.init.zeros_(self.vis_from_ir.output_proj.bias)

    def init_weights(self) -> None:
        self.vis_from_ir.init_weights()

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Optional[Tensor],
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        del text_feat, text_token_mask
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes)
        reference_points = \
            DeformableDetrTransformerEncoder.get_encoder_reference_points(
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                device=vis_feat.device)

        z_ir, z_vis = self._shared_sample(
            query=self.vis_norm(vis_feat),
            ir_value=self.ir_norm(ir_feat),
            vis_value=self.vis_norm(vis_feat),
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        z_diff = z_ir - z_vis
        if self.concat_inputs:
            projection_input = torch.cat(
                (z_diff, z_ir, z_vis), dim=-1)
        else:
            projection_input = z_diff
        delta = self.vis_from_ir.output_proj(projection_input)
        output = vis_feat + self.vis_from_ir.dropout(delta)
        if feat_mask is not None:
            output = output.masked_fill(
                feat_mask[..., None].to(torch.bool), 0)
        return output

    def _shared_sample(self,
                       query: Tensor,
                       ir_value: Tensor,
                       vis_value: Tensor,
                       query_pos: Optional[Tensor],
                       key_padding_mask: Optional[Tensor],
                       reference_points: Tensor,
                       spatial_shapes: Tensor,
                       level_start_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Return pre-output-projection IR and VIS sampled features."""
        attention = self.vis_from_ir
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.shape
        _, num_value, _ = ir_value.shape

        ir_value = attention.value_proj(ir_value)
        vis_value = attention.value_proj(vis_value)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None].to(torch.bool)
            ir_value = ir_value.masked_fill(mask, 0.0)
            vis_value = vis_value.masked_fill(mask, 0.0)
        ir_value = ir_value.view(bs, num_value, attention.num_heads, -1)
        vis_value = vis_value.view(bs, num_value, attention.num_heads, -1)

        sampling_offsets = attention.sampling_offsets(query).view(
            bs, num_query, attention.num_heads, attention.num_levels,
            attention.num_points, 2)
        attention_weights = attention.attention_weights(query).view(
            bs, num_query, attention.num_heads,
            attention.num_levels * attention.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, attention.num_heads, attention.num_levels,
            attention.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / attention.num_points
                * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                'The last reference_points dimension must be 2 or 4, '
                f'but got {reference_points.shape[-1]}.')

        z_ir = self._deformable_sample(
            ir_value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights)
        z_vis = self._deformable_sample(
            vis_value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights)
        return z_ir, z_vis

    def _deformable_sample(self,
                           value: Tensor,
                           spatial_shapes: Tensor,
                           level_start_index: Tensor,
                           sampling_locations: Tensor,
                           attention_weights: Tensor) -> Tensor:
        attention = self.vis_from_ir
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            return MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights,
                attention.im2col_step)
        return multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)

    def _validate_inputs(self, vis_feat: Tensor, ir_feat: Tensor,
                         spatial_shapes: Tensor) -> None:
        if vis_feat.shape != ir_feat.shape:
            raise ValueError(
                f'VIS/IR shapes differ: {vis_feat.shape} vs {ir_feat.shape}.')
        if vis_feat.ndim != 3 or vis_feat.size(-1) != self.embed_dims:
            raise ValueError('Features must have shape [B, N, embed_dims].')
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [levels, 2].')
        if spatial_shapes.size(0) != self.num_feature_levels:
            raise ValueError(
                f'Expected {self.num_feature_levels} feature levels, got '
                f'{spatial_shapes.size(0)}.')
        token_count = int(spatial_shapes.prod(dim=1).sum().item())
        if token_count != vis_feat.size(1):
            raise ValueError(
                f'Expected {token_count} flattened tokens, got '
                f'{vis_feat.size(1)}.')


@MODELS.register_module()
class IRToVISDeformableDifferenceFusion(
        _SharedSamplingDifferenceFusion):
    """Project only the shared-sampling difference z_ir - z_vis."""

    def __init__(self, **kwargs) -> None:
        super().__init__(concat_inputs=False, **kwargs)


@MODELS.register_module()
class IRToVISDeformableDifferenceConcatFusion(
        _SharedSamplingDifferenceFusion):
    """Project [z_ir - z_vis, z_ir, z_vis] jointly."""

    def __init__(self, **kwargs) -> None:
        super().__init__(concat_inputs=True, **kwargs)
