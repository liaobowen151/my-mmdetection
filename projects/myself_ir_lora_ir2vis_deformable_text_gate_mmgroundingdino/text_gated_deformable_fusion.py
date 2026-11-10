"""Text-evidence gates for one-way IR-to-VIS deformable enhancement."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MessageHub
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_ir_lora_ir2vis_deformable_mmgroundingdino.ir2vis_deformable_fusion import (  # noqa: E501
    IRToVISDeformableFusion,
)


class _IRToVISDeformableTextGateFusion(IRToVISDeformableFusion):
    """Gate the original IR deformable residual with text evidence.

    The deformable branch is deliberately unchanged.  It still computes the
    IR content ``delta_ir`` sampled for every VIS query.  This class only
    changes the final update from ``VIS + delta_ir`` to
    ``VIS + gate * delta_ir``.
    """

    def __init__(self,
                 *args,
                 token_pooling: str,
                 text_temperature: float = 1.0,
                 pooling_temperature: float = 0.25,
                 gate_temperature_vis: float = 0.25,
                 gate_temperature_ir: float = 0.25,
                 gate_temperature_advantage: float = 0.25,
                 vis_threshold: float = 0.0,
                 ir_threshold: float = 0.0,
                 advantage_margin: float = 0.0,
                 standardize_evidence: bool = False,
                 exclude_boundary_tokens: bool = True,
                 evidence_eps: float = 1e-6,
                 diagnostics_interval: int = 100,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if token_pooling not in {'max', 'lse'}:
            raise ValueError("token_pooling must be either 'max' or 'lse'.")
        positive_values = dict(
            text_temperature=text_temperature,
            pooling_temperature=pooling_temperature,
            gate_temperature_vis=gate_temperature_vis,
            gate_temperature_ir=gate_temperature_ir,
            gate_temperature_advantage=gate_temperature_advantage,
            evidence_eps=evidence_eps)
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f'{name} must be positive, got {value}.')
        if diagnostics_interval < 0:
            raise ValueError('diagnostics_interval cannot be negative.')

        self.token_pooling = token_pooling
        self.text_temperature = text_temperature
        self.pooling_temperature = pooling_temperature
        self.gate_temperature_vis = gate_temperature_vis
        self.gate_temperature_ir = gate_temperature_ir
        self.gate_temperature_advantage = gate_temperature_advantage
        self.vis_threshold = vis_threshold
        self.ir_threshold = ir_threshold
        self.advantage_margin = advantage_margin
        self.standardize_evidence = standardize_evidence
        self.exclude_boundary_tokens = exclude_boundary_tokens
        self.evidence_eps = evidence_eps
        self.diagnostics_interval = diagnostics_interval

        # VIS and IR intentionally share the same visual projection so their
        # evidence values live in one coordinate system.
        self.visual_evidence_norm = nn.LayerNorm(self.embed_dims)
        self.text_evidence_norm = nn.LayerNorm(self.embed_dims)
        self.visual_evidence_proj = nn.Linear(
            self.embed_dims, self.embed_dims, bias=False)
        self.text_evidence_proj = nn.Linear(
            self.embed_dims, self.embed_dims, bias=False)
        self._diagnostic_step = 0
        self.latest_gate_stats: Dict[str, float] = {}

    def init_weights(self) -> None:
        super().init_weights()
        # Preserve Grounding DINO's existing 256-D visual/text coordinate
        # system at initialization.  Independent random projections would
        # destroy that alignment before either projection receives gradients.
        nn.init.eye_(self.visual_evidence_proj.weight)
        nn.init.eye_(self.text_evidence_proj.weight)

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

        # Calling the original implementation preserves its query, sampling,
        # attention and output-projection behavior exactly.
        unconditional = super().forward(
            vis_feat=vis_feat,
            ir_feat=ir_feat,
            text_feat=text_feat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask,
            text_token_mask=text_token_mask)
        delta_ir = unconditional - vis_feat

        vis_evidence = self._text_evidence(
            vis_feat, text_feat, text_token_mask)
        ir_evidence = self._text_evidence(
            ir_feat, text_feat, text_token_mask)
        if self.standardize_evidence:
            vis_evidence = self._standardize_per_level(
                vis_evidence, spatial_shapes, level_start_index, feat_mask)
            ir_evidence = self._standardize_per_level(
                ir_evidence, spatial_shapes, level_start_index, feat_mask)

        weak_vis = torch.sigmoid(
            (self.vis_threshold - vis_evidence) /
            self.gate_temperature_vis)
        strong_ir = torch.sigmoid(
            (ir_evidence - self.ir_threshold) /
            self.gate_temperature_ir)
        advantage = torch.sigmoid(
            (ir_evidence - vis_evidence - self.advantage_margin) /
            self.gate_temperature_advantage)
        gate = (weak_vis * strong_ir * advantage).unsqueeze(-1)
        if feat_mask is not None:
            gate = gate.masked_fill(feat_mask[..., None].bool(), 0)

        output = vis_feat + gate * delta_ir
        if feat_mask is not None:
            output = output.masked_fill(feat_mask[..., None].bool(), 0)
        self._record_diagnostics(
            gate, vis_evidence, ir_evidence, feat_mask)
        return output

    def _validate_text(self, text_feat: Optional[Tensor],
                       text_token_mask: Optional[Tensor],
                       vis_feat: Tensor) -> None:
        if text_feat is None or text_token_mask is None:
            raise ValueError('Text-gated fusion requires text features and mask.')
        if text_feat.ndim != 3 or text_feat.size(-1) != self.embed_dims:
            raise ValueError(
                'text_feat must have shape [B, L, embed_dims], got '
                f'{tuple(text_feat.shape)}.')
        if text_token_mask.shape != text_feat.shape[:2]:
            raise ValueError('text_token_mask must have shape [B, L].')
        if text_feat.size(0) != vis_feat.size(0):
            raise ValueError('Visual and text batch sizes must match.')
        if not text_token_mask.bool().any(dim=1).all():
            raise ValueError('Every sample must contain a valid text token.')

    def _text_evidence(self, visual: Tensor, text: Tensor,
                       text_token_mask: Tensor) -> Tensor:
        visual = F.normalize(
            self.visual_evidence_proj(self.visual_evidence_norm(visual)),
            dim=-1,
            eps=self.evidence_eps)
        text = F.normalize(
            self.text_evidence_proj(self.text_evidence_norm(text)),
            dim=-1,
            eps=self.evidence_eps)
        similarity = torch.matmul(visual, text.transpose(-1, -2))
        similarity = similarity / self.text_temperature
        valid_tokens = self._effective_text_mask(text_token_mask).unsqueeze(1)
        similarity = similarity.masked_fill(~valid_tokens, -torch.inf)
        if self.token_pooling == 'max':
            return similarity.max(dim=-1).values
        return self.pooling_temperature * torch.logsumexp(
            similarity / self.pooling_temperature, dim=-1)

    def _effective_text_mask(self, text_token_mask: Tensor) -> Tensor:
        """Remove leading/trailing BERT boundary tokens when possible."""
        valid = text_token_mask.bool()
        if not self.exclude_boundary_tokens:
            return valid
        effective = valid.clone()
        effective[:, 0] = False
        last = valid.long().sum(dim=1).sub(1).clamp_min(0)
        effective.scatter_(1, last[:, None], False)
        # Very short/synthetic prompts may contain no interior token.  Falling
        # back is safer than producing an all-negative-infinity reduction.
        has_interior = effective.any(dim=1, keepdim=True)
        return torch.where(has_interior, effective, valid)

    def _standardize_per_level(self, evidence: Tensor,
                               spatial_shapes: Tensor,
                               level_start_index: Tensor,
                               feat_mask: Optional[Tensor]) -> Tensor:
        standardized = torch.empty_like(evidence)
        for level in range(spatial_shapes.size(0)):
            start = int(level_start_index[level].item())
            length = int(spatial_shapes[level].prod().item())
            end = start + length
            values = evidence[:, start:end]
            if feat_mask is None:
                mean = values.mean(dim=1, keepdim=True)
                variance = values.var(dim=1, keepdim=True, unbiased=False)
            else:
                valid = (~feat_mask[:, start:end].bool()).to(values.dtype)
                count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean = (values * valid).sum(dim=1, keepdim=True) / count
                variance = ((values - mean).square() * valid).sum(
                    dim=1, keepdim=True) / count
            standardized[:, start:end] = (
                values - mean) / torch.sqrt(variance + self.evidence_eps)
        return standardized

    def _record_diagnostics(self, gate: Tensor, vis_evidence: Tensor,
                            ir_evidence: Tensor,
                            feat_mask: Optional[Tensor]) -> None:
        self._diagnostic_step += 1
        if self.diagnostics_interval == 0 or \
                self._diagnostic_step % self.diagnostics_interval != 0:
            return
        valid = torch.ones_like(gate[..., 0], dtype=torch.bool) \
            if feat_mask is None else ~feat_mask.bool()
        gate_values = gate[..., 0][valid]
        vis_values = vis_evidence[valid]
        ir_values = ir_evidence[valid]
        stats = dict(
            gate_mean=gate_values.mean(),
            gate_min=gate_values.min(),
            gate_max=gate_values.max(),
            gate_ratio_gt_01=(gate_values > 0.1).float().mean(),
            gate_ratio_gt_05=(gate_values > 0.5).float().mean(),
            evidence_vis_mean=vis_values.mean(),
            evidence_ir_mean=ir_values.mean(),
            evidence_advantage_mean=(ir_values - vis_values).mean())
        self.latest_gate_stats = {
            key: float(value.detach().cpu()) for key, value in stats.items()
        }
        try:
            message_hub = MessageHub.get_current_instance()
        except RuntimeError:
            return
        prefix = f'text_gate_{self.token_pooling}'
        for key, value in stats.items():
            message_hub.update_scalar(f'{prefix}/{key}', value.detach())


@MODELS.register_module()
class IRToVISDeformableDirectTextGateFusion(
        _IRToVISDeformableTextGateFusion):
    """Use the strongest valid text-token similarity as evidence."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, token_pooling='max', **kwargs)


@MODELS.register_module()
class IRToVISDeformableLSETextGateFusion(
        _IRToVISDeformableTextGateFusion):
    """Use smooth log-sum-exp aggregation over valid text tokens."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, token_pooling='lse', **kwargs)
