"""Deep-level language-gated fusion with text-conditioned preservation."""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerEncoder, )
from mmdet.registry import MODELS
from projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino.ir2vis_deformable_add_ir_fusion import (  # noqa: E501
    IRToVISDeformableAddIRFusion,
)


class PixelWordAttention(nn.Module):
    """Multi-head attention from flattened visual tokens to text tokens."""

    def __init__(self, visual_dims: int, text_dims: int,
                 num_heads: int) -> None:
        super().__init__()
        if text_dims % num_heads != 0:
            raise ValueError('text_dims must be divisible by num_heads.')
        self.num_heads = num_heads
        self.head_dims = text_dims // num_heads
        self.scale = self.head_dims**-0.5
        self.query_proj = nn.Linear(visual_dims, text_dims)
        self.key_proj = nn.Linear(text_dims, text_dims)
        self.value_proj = nn.Linear(text_dims, text_dims)
        self.output_proj = nn.Linear(text_dims, text_dims)

    def forward(self, visual: Tensor, text: Tensor,
                text_token_mask: Tensor) -> Tensor:
        batch_size, num_visual_tokens, _ = visual.shape
        num_text_tokens = text.size(1)
        query = self.query_proj(visual).view(
            batch_size, num_visual_tokens, self.num_heads,
            self.head_dims).transpose(1, 2)
        key = self.key_proj(text).view(
            batch_size, num_text_tokens, self.num_heads,
            self.head_dims).transpose(1, 2)
        value = self.value_proj(text).view(
            batch_size, num_text_tokens, self.num_heads,
            self.head_dims).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = attention.masked_fill(
            ~text_token_mask[:, None, None, :].bool(), -torch.inf)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_visual_tokens, -1)
        return self.output_proj(context)


class LanguageGatedFusionBlock(nn.Module):
    """RIS-FUSION-style gate for one flattened feature level."""

    def __init__(self, embed_dims: int, num_heads: int,
                 hidden_dims: int, gamma_scale: float) -> None:
        super().__init__()
        self.gamma_scale = gamma_scale
        self.pixel_word_attention = PixelWordAttention(
            visual_dims=2 * embed_dims,
            text_dims=embed_dims,
            num_heads=num_heads)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims), nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 1), nn.Sigmoid())
        self.film = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims), nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, 2))

    def forward(self, vis_enhanced: Tensor, ir_feat: Tensor,
                text_feat: Tensor,
                text_token_mask: Tensor) -> Tuple[Tensor, Tensor]:
        visual_query = torch.cat([vis_enhanced, ir_feat], dim=-1)
        context = self.pixel_word_attention(
            visual_query, text_feat, text_token_mask)
        alpha = self.fusion_gate(context)
        blended = (1.0 - alpha) * vis_enhanced + alpha * ir_feat
        gamma, beta = self.film(context).chunk(2, dim=-1)
        output = (1.0 + self.gamma_scale * gamma) * blended + beta
        return output, alpha


@MODELS.register_module()
class IRToVISDeformableLanguageGatedFusion(
        IRToVISDeformableAddIRFusion):
    """Add shallow levels and language-gate deep levels.

    The IR-to-VIS deformable enhancement still operates jointly over all
    levels. Levels in ``language_levels`` use independent language-gated
    blocks; all other levels retain ``vis_enhanced + ir``.

    During training, deep outputs receive a text-conditioned preservation
    loss. At text-relevant positions, it aligns the fused feature with the
    source modality that has the larger text similarity. Source features,
    modality weights and spatial relevance weights are fixed targets.
    """

    def __init__(self,
                 *args,
                 language_num_heads: int = 4,
                 language_hidden_dims: int = 128,
                 language_levels: Sequence[int] = (2, 3),
                 gamma_scale: float = 0.1,
                 text_temperature: float = 1.0,
                 token_pooling_temperature: float = 0.25,
                 modality_temperature: float = 0.25,
                 spatial_temperature: float = 0.25,
                 text_preserve_loss_weight: float = 0.1,
                 exclude_boundary_tokens: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if language_hidden_dims <= 0:
            raise ValueError('language_hidden_dims must be positive.')
        if gamma_scale < 0:
            raise ValueError('gamma_scale cannot be negative.')
        temperatures = dict(
            text_temperature=text_temperature,
            token_pooling_temperature=token_pooling_temperature,
            modality_temperature=modality_temperature,
            spatial_temperature=spatial_temperature)
        for name, value in temperatures.items():
            if value <= 0:
                raise ValueError(f'{name} must be positive, got {value}.')
        if text_preserve_loss_weight < 0:
            raise ValueError('text_preserve_loss_weight cannot be negative.')

        levels = tuple(int(level) for level in language_levels)
        if len(set(levels)) != len(levels):
            raise ValueError('language_levels cannot contain duplicates.')
        if any(level < 0 or level >= self.num_feature_levels
               for level in levels):
            raise ValueError(
                'language_levels must be valid zero-based feature levels.')

        self.language_levels = levels
        self.text_temperature = text_temperature
        self.token_pooling_temperature = token_pooling_temperature
        self.modality_temperature = modality_temperature
        self.spatial_temperature = spatial_temperature
        self.text_preserve_loss_weight = text_preserve_loss_weight
        self.exclude_boundary_tokens = exclude_boundary_tokens
        self.language_blocks = nn.ModuleDict({
            str(level): LanguageGatedFusionBlock(
                embed_dims=self.embed_dims,
                num_heads=language_num_heads,
                hidden_dims=language_hidden_dims,
                gamma_scale=gamma_scale)
            for level in levels
        })
        self.text_preserve_loss: Optional[Tensor] = None
        self.latest_alpha = {}

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
        self._validate_inputs(vis_feat, ir_feat, spatial_shapes)
        self._validate_text(text_feat, text_token_mask, vis_feat)
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

        effective_text_mask = self._effective_text_mask(text_token_mask)
        outputs = []
        preserve_losses = []
        self.latest_alpha = {}
        for level in range(self.num_feature_levels):
            start = int(level_start_index[level].item())
            length = int(spatial_shapes[level].prod().item())
            end = start + length
            vis_level = vis_enhanced[:, start:end]
            ir_level = ir_feat[:, start:end]
            level_mask = None if feat_mask is None else feat_mask[:, start:end]

            if level in self.language_levels:
                fused_level, alpha = self.language_blocks[str(level)](
                    vis_level, ir_level, text_feat, effective_text_mask)
                self.latest_alpha[level] = alpha.detach()
                if self.training and self.text_preserve_loss_weight > 0:
                    preserve_losses.append(self._text_preservation_loss(
                        fused_level, vis_level, ir_level, text_feat,
                        effective_text_mask, level_mask))
            else:
                fused_level = vis_level + ir_level

            if level_mask is not None:
                fused_level = fused_level.masked_fill(
                    level_mask[..., None].bool(), 0)
            outputs.append(fused_level)

        output = torch.cat(outputs, dim=1)
        self.text_preserve_loss = None
        if preserve_losses:
            self.text_preserve_loss = torch.stack(preserve_losses).mean()
        return output

    def _text_preservation_loss(self, fused: Tensor, vis: Tensor, ir: Tensor,
                                text: Tensor, text_mask: Tensor,
                                visual_mask: Optional[Tensor]) -> Tensor:
        # Targets and weights are detached so the loss cannot move its own
        # target or suppress its own relevance signal to obtain a shortcut.
        vis_target = F.normalize(vis.detach(), dim=-1, eps=1e-6)
        ir_target = F.normalize(ir.detach(), dim=-1, eps=1e-6)
        text_target = F.normalize(text.detach(), dim=-1, eps=1e-6)
        fused_norm = F.normalize(fused, dim=-1, eps=1e-6)

        vis_evidence = self._text_evidence(
            vis_target, text_target, text_mask)
        ir_evidence = self._text_evidence(
            ir_target, text_target, text_mask)
        modality_weights = F.softmax(
            torch.stack([vis_evidence, ir_evidence], dim=-1) /
            self.modality_temperature,
            dim=-1).detach()

        relevance_logits = torch.maximum(vis_evidence, ir_evidence)
        if visual_mask is not None:
            relevance_logits = relevance_logits.masked_fill(
                visual_mask.bool(), -torch.inf)
        spatial_weights = F.softmax(
            relevance_logits / self.spatial_temperature, dim=1).detach()
        if visual_mask is not None:
            spatial_weights = spatial_weights.masked_fill(
                visual_mask.bool(), 0)

        distance_vis = 1.0 - (fused_norm * vis_target).sum(dim=-1)
        distance_ir = 1.0 - (fused_norm * ir_target).sum(dim=-1)
        per_token_loss = (
            modality_weights[..., 0] * distance_vis +
            modality_weights[..., 1] * distance_ir)
        return (spatial_weights * per_token_loss).sum(dim=1).mean()

    def _text_evidence(self, visual: Tensor, text: Tensor,
                       text_mask: Tensor) -> Tensor:
        similarity = torch.matmul(visual, text.transpose(-1, -2))
        similarity = similarity / self.text_temperature
        similarity = similarity.masked_fill(
            ~text_mask[:, None, :].bool(), -torch.inf)
        temperature = self.token_pooling_temperature
        return temperature * torch.logsumexp(
            similarity / temperature, dim=-1)

    def _effective_text_mask(self, text_token_mask: Tensor) -> Tensor:
        valid = text_token_mask.bool()
        if not self.exclude_boundary_tokens:
            return valid
        effective = valid.clone()
        effective[:, 0] = False
        last = valid.long().sum(dim=1).sub(1).clamp_min(0)
        effective.scatter_(1, last[:, None], False)
        has_interior = effective.any(dim=1, keepdim=True)
        return torch.where(has_interior, effective, valid)

    def _validate_text(self, text_feat: Optional[Tensor],
                       text_token_mask: Optional[Tensor],
                       visual_feat: Tensor) -> None:
        if text_feat is None or text_token_mask is None:
            raise ValueError(
                'Language-gated fusion requires text features and a mask.')
        if text_feat.ndim != 3 or text_feat.size(-1) != self.embed_dims:
            raise ValueError(
                'text_feat must have shape [B, L, embed_dims], got '
                f'{tuple(text_feat.shape)}.')
        if text_token_mask.shape != text_feat.shape[:2]:
            raise ValueError(
                'text_token_mask must have shape [B, L], got '
                f'{tuple(text_token_mask.shape)}.')
        if text_feat.size(0) != visual_feat.size(0):
            raise ValueError('Visual and text batch sizes must match.')
        if not text_token_mask.bool().any(dim=1).all():
            raise ValueError('Every sample must contain a valid text token.')
