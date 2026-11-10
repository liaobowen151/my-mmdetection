"""Text-guided spatial residual fusion with optional channel selection."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module()
class SpatialChannelExpectationFusion(nn.Module):
    """Fuse VIS/IR features using token, spatial, and channel gates.

    Two token aggregation modes are supported:

    - ``level_balanced_shared``: average the per-level token evidence with
      equal level weights, then share one token distribution across levels.
    - ``per_level``: form an independent token distribution at every level.

    Visual features are the post-ChannelMapper features and receive no extra
    normalization here.  Similarity follows Grounding DINO's contrastive head
    and is divided by ``sqrt(embed_dims)``.  The original text tensor is read
    without modifying the tensor passed to the rest of Grounding DINO.
    """

    _TOKEN_MODES = ('level_balanced_shared', 'per_level')
    _ALIGNMENT_NORMALIZATIONS = ('none', 'center', 'zscore')
    _SPATIAL_GATE_MODES = ('dense', 'six_state')

    def __init__(self,
                 embed_dims: int = 256,
                 token_temperature: float = 1.0,
                 gate_temperature: float = 1.0,
                 token_fpn_mode: str = 'level_balanced_shared',
                 with_channel_gate: bool = True,
                 channel_reduction: int = 4,
                 center_alignment: bool = False,
                 alignment_normalization: str = 'none',
                 alignment_normalization_eps: float = 1e-6,
                 spatial_gate_mode: str = 'dense',
                 high_response_threshold: float = 1.5,
                 low_response_threshold: float = -0.5,
                 token_temperature_mode: str = 'fixed',
                 token_temperature_reference_length: float = 16.0,
                 token_temperature_length_power: float = 0.5,
                 token_temperature_min: float = 0.5,
                 token_temperature_max: float = 4.0) -> None:
        super().__init__()
        if embed_dims <= 0:
            raise ValueError('embed_dims must be positive.')
        if token_temperature <= 0 or gate_temperature <= 0:
            raise ValueError('Both temperatures must be positive.')
        if token_fpn_mode not in self._TOKEN_MODES:
            raise ValueError(f'Unsupported token_fpn_mode={token_fpn_mode!r}.')
        if token_temperature_mode not in ('fixed', 'text_length'):
            raise ValueError('token_temperature_mode must be fixed or '
                             'text_length.')
        if token_temperature_reference_length <= 0:
            raise ValueError(
                'token_temperature_reference_length must be positive.')
        if token_temperature_length_power < 0:
            raise ValueError(
                'token_temperature_length_power must be non-negative.')
        if token_temperature_min <= 0 or token_temperature_max <= 0 or \
                token_temperature_min > token_temperature_max:
            raise ValueError('Invalid token temperature bounds.')
        if channel_reduction <= 0 or embed_dims // channel_reduction <= 0:
            raise ValueError('channel_reduction is invalid for embed_dims.')
        if alignment_normalization not in self._ALIGNMENT_NORMALIZATIONS:
            raise ValueError(
                'alignment_normalization must be none, center, or zscore.')
        if spatial_gate_mode not in self._SPATIAL_GATE_MODES:
            raise ValueError('spatial_gate_mode must be dense or six_state.')
        if alignment_normalization_eps <= 0:
            raise ValueError('alignment_normalization_eps must be positive.')
        if low_response_threshold >= high_response_threshold:
            raise ValueError('low_response_threshold must be smaller than '
                             'high_response_threshold.')
        if center_alignment and alignment_normalization != 'none':
            raise ValueError('center_alignment and alignment_normalization '
                             'cannot both be enabled.')
        if spatial_gate_mode == 'six_state' and \
                alignment_normalization != 'zscore':
            raise ValueError('six_state gating requires zscore alignment.')

        self.embed_dims = int(embed_dims)
        self.token_temperature = float(token_temperature)
        self.gate_temperature = float(gate_temperature)
        self.token_fpn_mode = token_fpn_mode
        self.with_channel_gate = bool(with_channel_gate)
        self.center_alignment = bool(center_alignment)
        self.alignment_normalization = alignment_normalization
        self.alignment_normalization_eps = float(
            alignment_normalization_eps)
        self.spatial_gate_mode = spatial_gate_mode
        self.high_response_threshold = float(high_response_threshold)
        self.low_response_threshold = float(low_response_threshold)
        self.token_temperature_mode = token_temperature_mode
        self.token_temperature_reference_length = float(
            token_temperature_reference_length)
        self.token_temperature_length_power = float(
            token_temperature_length_power)
        self.token_temperature_min = float(token_temperature_min)
        self.token_temperature_max = float(token_temperature_max)
        self.similarity_scale = self.embed_dims**-0.5

        if self.with_channel_gate:
            hidden_dims = self.embed_dims // channel_reduction
            self.channel_gate = nn.Sequential(
                nn.Linear(3 * self.embed_dims, hidden_dims),
                nn.ReLU(inplace=True), nn.Linear(hidden_dims, self.embed_dims),
                nn.Sigmoid())
            # Begin from a deterministic, channel-neutral value of 0.5.
            nn.init.zeros_(self.channel_gate[2].weight)
            nn.init.zeros_(self.channel_gate[2].bias)
        else:
            self.channel_gate = None

        self._collect_statistics = False
        self._statistics: Dict[str, Tensor] = {}

    @staticmethod
    def _validate_masks(
            visual: Tensor, text: Tensor, feat_mask: Optional[Tensor],
            text_token_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        batch_size, num_positions, _ = visual.shape
        if feat_mask is None:
            valid_positions = torch.ones((batch_size, num_positions),
                                         dtype=torch.bool,
                                         device=visual.device)
        else:
            if feat_mask.shape != (batch_size, num_positions):
                raise ValueError('feat_mask must have shape [B, N].')
            valid_positions = ~feat_mask.to(torch.bool)

        if text_token_mask is None:
            valid_tokens = torch.ones(
                text.shape[:2], dtype=torch.bool, device=text.device)
        else:
            if text_token_mask.shape != text.shape[:2]:
                raise ValueError('text_token_mask must have shape [B, L].')
            valid_tokens = text_token_mask.to(torch.bool)
        if not valid_positions.any(dim=1).all():
            raise ValueError(
                'Every image must contain a valid visual position.')
        if not valid_tokens.any(dim=1).all():
            raise ValueError('Every text sequence must contain a valid token.')
        return valid_positions, valid_tokens

    @staticmethod
    def _level_ranges(spatial_shapes: Tensor, level_start_index: Tensor,
                      num_positions: int) -> List[Tuple[int, int]]:
        if spatial_shapes.ndim != 2 or spatial_shapes.size(-1) != 2:
            raise ValueError('spatial_shapes must have shape [K, 2].')
        if level_start_index.ndim != 1 or \
                level_start_index.numel() != spatial_shapes.size(0):
            raise ValueError('level_start_index must have shape [K].')
        starts = [int(value) for value in level_start_index.detach().cpu()]
        sizes = [
            int(shape[0] * shape[1])
            for shape in spatial_shapes.detach().cpu()
        ]
        ranges = [(start, start + size) for start, size in zip(starts, sizes)]
        if not ranges or ranges[0][0] != 0 or ranges[-1][1] != num_positions:
            raise ValueError(
                'FPN level ranges do not cover the token sequence.')
        return ranges

    def _similarity_and_evidence(
            self, visual: Tensor, text: Tensor,
            valid_positions: Tensor) -> Tuple[Tensor, Tensor]:
        similarity = torch.einsum('bnc,blc->bnl', visual, text)
        similarity = similarity * self.similarity_scale
        position_weight = valid_positions[:, :, None].to(similarity.dtype)
        position_count = position_weight.sum(dim=1).clamp_min(1.0)
        evidence = (similarity * position_weight).sum(dim=1)
        evidence = evidence / position_count
        return similarity, evidence

    def _token_temperature(self, valid_tokens: Tensor) -> Tensor:
        """Return one token-softmax temperature for every sample."""
        if self.token_temperature_mode == 'fixed':
            return valid_tokens.new_full(
                (valid_tokens.size(0), 1),
                self.token_temperature,
                dtype=torch.float32)
        valid_length = valid_tokens.sum(dim=-1, keepdim=True).float()
        length_scale = (
            valid_length / self.token_temperature_reference_length).pow(
                self.token_temperature_length_power)
        return (self.token_temperature * length_scale).clamp(
            min=self.token_temperature_min,
            max=self.token_temperature_max)

    def _token_weight(self, evidence: Tensor,
                      valid_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        evidence = evidence.masked_fill(~valid_tokens, float('-inf'))
        temperature = self._token_temperature(valid_tokens)
        weight = torch.softmax(
            evidence.float() / temperature, dim=-1).to(evidence.dtype)
        return weight, temperature

    @staticmethod
    def _center_spatial_alignment(alignment: Tensor,
                                  valid_positions: Tensor) -> Tensor:
        weight = valid_positions.to(alignment.dtype)
        mean = (alignment * weight).sum(dim=1, keepdim=True)
        mean = mean / weight.sum(dim=1, keepdim=True).clamp_min(1)
        centered = alignment - mean
        return centered.masked_fill(~valid_positions, 0)

    def _normalize_spatial_alignment(
            self, alignment: Tensor,
            valid_positions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Normalize each sample over valid positions in one FPN level."""
        alignment_fp32 = alignment.float()
        weight = valid_positions.to(alignment_fp32.dtype)
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (alignment_fp32 * weight).sum(dim=1, keepdim=True) / count
        centered = alignment_fp32 - mean
        variance = (centered.square() * weight).sum(
            dim=1, keepdim=True) / count
        std = torch.sqrt(variance + self.alignment_normalization_eps)

        if self.alignment_normalization == 'zscore':
            normalized = centered / std
        elif self.alignment_normalization == 'center':
            normalized = centered
        elif self.center_alignment:
            # Backward compatibility for the existing E1-C configuration.
            normalized = centered
        else:
            normalized = alignment_fp32
        normalized = normalized.masked_fill(~valid_positions, 0)
        return normalized, mean, std

    def _six_state_gates(
            self, vis_alignment: Tensor, ir_alignment: Tensor,
            valid_positions: Tensor
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Select six mutually exclusive states and apply tanh selectively."""
        vis_high = vis_alignment >= self.high_response_threshold
        ir_high = ir_alignment >= self.high_response_threshold
        vis_low = vis_alignment <= self.low_response_threshold
        ir_low = ir_alignment <= self.low_response_threshold
        vis_mid = ~(vis_high | vis_low)
        ir_mid = ~(ir_high | ir_low)

        states = {
            'case_1_vis_enhance_ir_suppress': vis_high & ir_low,
            'case_2_ir_enhance_vis_suppress': ir_high & vis_low,
            'case_3_both_enhance': vis_high & ir_high,
            'case_4_vis_enhance_ir_keep': vis_high & ir_mid,
            'case_5_ir_enhance_vis_keep': ir_high & vis_mid,
        }
        assigned = torch.zeros_like(valid_positions)
        for name in states:
            states[name] = states[name] & valid_positions
            assigned |= states[name]
        states['case_6_base'] = valid_positions & ~assigned

        vis_active = (states['case_1_vis_enhance_ir_suppress'] |
                      states['case_2_ir_enhance_vis_suppress'] |
                      states['case_3_both_enhance'] |
                      states['case_4_vis_enhance_ir_keep'])
        ir_active = (states['case_1_vis_enhance_ir_suppress'] |
                     states['case_2_ir_enhance_vis_suppress'] |
                     states['case_3_both_enhance'] |
                     states['case_5_ir_enhance_vis_keep'])
        vis_candidate = torch.tanh(vis_alignment / self.gate_temperature)
        ir_candidate = torch.tanh(ir_alignment / self.gate_temperature)
        vis_gate = torch.where(vis_active, vis_candidate,
                               torch.zeros_like(vis_candidate))
        ir_gate = torch.where(ir_active, ir_candidate,
                              torch.zeros_like(ir_candidate))
        return vis_gate, ir_gate, states

    @staticmethod
    def _masked_mean(feature: Tensor, valid_positions: Tensor) -> Tensor:
        weight = valid_positions[:, :, None].to(feature.dtype)
        return (feature * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)

    def _make_channel_gate(self, vis: Tensor, ir: Tensor,
                           valid_positions: Tensor) -> Tensor:
        if self.channel_gate is None:
            return vis.new_ones((vis.size(0), 1, vis.size(-1)))
        vis_global = self._masked_mean(vis, valid_positions)
        ir_global = self._masked_mean(ir, valid_positions)
        difference_global = self._masked_mean((vis - ir).abs(),
                                              valid_positions)
        descriptor = torch.cat((vis_global, ir_global, difference_global),
                               dim=-1)
        return self.channel_gate(descriptor).unsqueeze(1)

    def forward(self,
                vis_feat: Tensor,
                ir_feat: Tensor,
                text_feat: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Optional[Tensor] = None,
                feat_pos: Optional[Tensor] = None,
                feat_mask: Optional[Tensor] = None,
                text_token_mask: Optional[Tensor] = None) -> Tensor:
        del valid_ratios, feat_pos
        if vis_feat.ndim != 3 or ir_feat.ndim != 3 or text_feat.ndim != 3:
            raise ValueError('VIS, IR and text must be three-dimensional.')
        if vis_feat.shape != ir_feat.shape:
            raise ValueError('VIS and IR feature shapes must match.')
        if vis_feat.size(0) != text_feat.size(0):
            raise ValueError('Visual and text batch sizes must match.')
        if vis_feat.size(-1) != self.embed_dims or \
                text_feat.size(-1) != self.embed_dims:
            raise ValueError(f'Expected feature dimension {self.embed_dims}.')

        valid_positions, valid_tokens = self._validate_masks(
            vis_feat, text_feat, feat_mask, text_token_mask)
        level_ranges = self._level_ranges(spatial_shapes, level_start_index,
                                          vis_feat.size(1))

        vis_levels, ir_levels, valid_levels = [], [], []
        vis_similarities, ir_similarities = [], []
        vis_evidences, ir_evidences = [], []
        for start, end in level_ranges:
            vis_level = vis_feat[:, start:end]
            ir_level = ir_feat[:, start:end]
            valid_level = valid_positions[:, start:end]
            if not valid_level.any(dim=1).all():
                raise ValueError(
                    'Every FPN level must contain a valid position.')
            vis_similarity, vis_evidence = self._similarity_and_evidence(
                vis_level, text_feat, valid_level)
            ir_similarity, ir_evidence = self._similarity_and_evidence(
                ir_level, text_feat, valid_level)
            vis_levels.append(vis_level)
            ir_levels.append(ir_level)
            valid_levels.append(valid_level)
            vis_similarities.append(vis_similarity)
            ir_similarities.append(ir_similarity)
            vis_evidences.append(vis_evidence)
            ir_evidences.append(ir_evidence)

        if self.token_fpn_mode == 'level_balanced_shared':
            vis_weight, vis_token_temperature = self._token_weight(
                torch.stack(vis_evidences, dim=1).mean(dim=1), valid_tokens)
            ir_weight, ir_token_temperature = self._token_weight(
                torch.stack(ir_evidences, dim=1).mean(dim=1), valid_tokens)
            vis_weights = [vis_weight] * len(level_ranges)
            ir_weights = [ir_weight] * len(level_ranges)
            vis_token_temperatures = [vis_token_temperature] * len(
                level_ranges)
            ir_token_temperatures = [ir_token_temperature] * len(level_ranges)
        else:
            vis_weight_temperature = [
                self._token_weight(evidence, valid_tokens)
                for evidence in vis_evidences
            ]
            ir_weight_temperature = [
                self._token_weight(evidence, valid_tokens)
                for evidence in ir_evidences
            ]
            vis_weights, vis_token_temperatures = map(
                list, zip(*vis_weight_temperature))
            ir_weights, ir_token_temperatures = map(
                list, zip(*ir_weight_temperature))

        fused_levels = []
        for level_id, (vis_level, ir_level, valid_level, vis_similarity,
                       ir_similarity, vis_weight, ir_weight) in enumerate(
                           zip(vis_levels, ir_levels, valid_levels,
                               vis_similarities, ir_similarities, vis_weights,
                               ir_weights)):
            vis_alignment = (vis_similarity *
                             vis_weight[:, None, :]).sum(dim=-1)
            ir_alignment = (ir_similarity * ir_weight[:, None, :]).sum(dim=-1)
            vis_gate_alignment, vis_alignment_mean, vis_alignment_std = \
                self._normalize_spatial_alignment(vis_alignment, valid_level)
            ir_gate_alignment, ir_alignment_mean, ir_alignment_std = \
                self._normalize_spatial_alignment(ir_alignment, valid_level)
            if self.spatial_gate_mode == 'six_state':
                vis_gate_fp32, ir_gate_fp32, state_masks = \
                    self._six_state_gates(vis_gate_alignment,
                                          ir_gate_alignment, valid_level)
            else:
                vis_gate_fp32 = torch.tanh(
                    vis_gate_alignment / self.gate_temperature)
                ir_gate_fp32 = torch.tanh(
                    ir_gate_alignment / self.gate_temperature)
                state_masks = None
            vis_gate = vis_gate_fp32.to(vis_feat.dtype)
            ir_gate = ir_gate_fp32.to(ir_feat.dtype)
            channel_gate = self._make_channel_gate(vis_level, ir_level,
                                                   valid_level)

            fused_level = (
                vis_level + ir_level +
                vis_gate[..., None] * channel_gate * vis_level +
                ir_gate[..., None] * channel_gate * ir_level)
            fused_level = fused_level.masked_fill(~valid_level[..., None], 0)
            fused_levels.append(fused_level)

            self._record_level_statistics(
                level_id, vis_alignment, ir_alignment, vis_gate_alignment,
                ir_gate_alignment, vis_gate, ir_gate, vis_weight, ir_weight,
                vis_token_temperatures[level_id],
                ir_token_temperatures[level_id], channel_gate, valid_level,
                valid_tokens, vis_level, ir_level, state_masks,
                vis_alignment_mean, vis_alignment_std, ir_alignment_mean,
                ir_alignment_std)

        fused = torch.cat(fused_levels, dim=1)
        if feat_mask is not None:
            fused = fused.masked_fill(feat_mask[..., None].to(torch.bool), 0)
        return fused

    def start_statistics(self) -> None:
        self._statistics = {}
        self._collect_statistics = True

    def stop_statistics(self) -> None:
        self._collect_statistics = False

    def get_statistics(self) -> Dict[str, Tensor]:
        return {key: value.clone() for key, value in self._statistics.items()}

    def _record(self, key: str, value: Tensor) -> None:
        if not self._collect_statistics:
            return
        scalar = value.detach().float().mean().double()
        if not torch.isfinite(scalar):
            return
        if key not in self._statistics:
            self._statistics[key] = torch.zeros(
                2, dtype=torch.float64, device=scalar.device)
        self._statistics[key][0].add_(scalar)
        self._statistics[key][1].add_(1)

    def _record_token_statistics(self, prefix: str, token_weight: Tensor,
                                 valid_tokens: Tensor) -> None:
        probability = token_weight.float().clamp_min(1e-12)
        entropy = -(probability * probability.log()).sum(dim=-1)
        valid_length = valid_tokens.sum(dim=-1).float()
        entropy_normalized = entropy / valid_length.log().clamp_min(1e-6)
        effective_count = entropy.exp()
        sorted_weight = probability.sort(dim=-1, descending=True).values
        self._record(f'{prefix}/token_entropy_normalized', entropy_normalized)
        self._record(f'{prefix}/token_max_weight', sorted_weight[:, 0])
        self._record(f'{prefix}/token_top3_mass',
                     sorted_weight[:, :min(3, sorted_weight.size(1))].sum(-1))
        self._record(f'{prefix}/token_top5_mass',
                     sorted_weight[:, :min(5, sorted_weight.size(1))].sum(-1))
        self._record(f'{prefix}/effective_token_count', effective_count)
        self._record(f'{prefix}/effective_token_ratio',
                     effective_count / valid_length.clamp_min(1))
        self._record(f'{prefix}/valid_text_length', valid_length)

    def _record_level_statistics(self, level_id: int, vis_alignment: Tensor,
                                 ir_alignment: Tensor,
                                 vis_gate_alignment: Tensor,
                                 ir_gate_alignment: Tensor, vis_gate: Tensor,
                                 ir_gate: Tensor, vis_weight: Tensor,
                                 ir_weight: Tensor,
                                 vis_token_temperature: Tensor,
                                 ir_token_temperature: Tensor,
                                 channel_gate: Tensor,
                                 valid_positions: Tensor, valid_tokens: Tensor,
                                 vis_level: Tensor, ir_level: Tensor,
                                 state_masks: Optional[Dict[str, Tensor]],
                                 vis_alignment_mean: Tensor,
                                 vis_alignment_std: Tensor,
                                 ir_alignment_mean: Tensor,
                                 ir_alignment_std: Tensor) -> None:
        if not self._collect_statistics:
            return
        valid = valid_positions
        for name, alignment, gate_alignment, gate, token_temperature in (
                ('vis', vis_alignment, vis_gate_alignment, vis_gate,
                 vis_token_temperature),
                ('ir', ir_alignment, ir_gate_alignment, ir_gate,
                 ir_token_temperature)):
            values = alignment[valid]
            gate_inputs = gate_alignment[valid]
            gates = gate[valid]
            prefix = f'level_{level_id}/{name}'
            self._record(f'{prefix}_alignment_mean', values.mean())
            self._record(f'{prefix}_alignment_std', values.std(unbiased=False))
            self._record(f'{prefix}_alignment_min', values.min())
            self._record(f'{prefix}_alignment_max', values.max())
            self._record(f'{prefix}_centered_alignment_mean',
                         gate_inputs.mean())
            self._record(f'{prefix}_centered_alignment_std',
                         gate_inputs.std(unbiased=False))
            self._record(f'{prefix}_centered_alignment_min',
                         gate_inputs.min())
            self._record(f'{prefix}_centered_alignment_max',
                         gate_inputs.max())
            self._record(f'{prefix}_token_temperature', token_temperature)
            self._record(f'{prefix}_gate_mean', gates.mean())
            self._record(f'{prefix}_gate_std', gates.std(unbiased=False))
            self._record(f'{prefix}_gate_negative_ratio', (gates < 0).float())
            self._record(f'{prefix}_gate_positive_ratio', (gates > 0).float())
            self._record(f'{prefix}_gate_neutral_ratio', (gates.abs()
                                                          < 0.1).float())
            self._record(f'{prefix}_gate_strong_suppress_ratio',
                         (gates < -0.9).float())
            self._record(f'{prefix}_gate_strong_enhance_ratio',
                         (gates > 0.9).float())
            self._record(f'{prefix}_effective_weight_mean',
                         (1.0 + gates).mean())
            self._record(f'{prefix}_effective_weight_std',
                         (1.0 + gates).std(unbiased=False))

        self._record(f'level_{level_id}/vis_normalization_mean',
                     vis_alignment_mean)
        self._record(f'level_{level_id}/vis_normalization_std',
                     vis_alignment_std)
        self._record(f'level_{level_id}/ir_normalization_mean',
                     ir_alignment_mean)
        self._record(f'level_{level_id}/ir_normalization_std',
                     ir_alignment_std)
        if state_masks is not None:
            valid_count = valid.float().sum().clamp_min(1.0)
            for state_name, state_mask in state_masks.items():
                ratio = state_mask.float().sum() / valid_count
                self._record(f'level_{level_id}/{state_name}_ratio', ratio)
            state_total = sum(mask.int() for mask in state_masks.values())
            self._record(f'level_{level_id}/six_state_partition_error_ratio',
                         ((state_total != 1) & valid).float().sum() /
                         valid_count)

        vis_valid_gate = vis_gate[valid]
        ir_valid_gate = ir_gate[valid]
        self._record(f'level_{level_id}/joint_gate_vis_pos_ir_pos_ratio',
                     ((vis_valid_gate > 0) &
                      (ir_valid_gate > 0)).float())
        self._record(f'level_{level_id}/joint_gate_vis_pos_ir_neg_ratio',
                     ((vis_valid_gate > 0) &
                      (ir_valid_gate < 0)).float())
        self._record(f'level_{level_id}/joint_gate_vis_neg_ir_pos_ratio',
                     ((vis_valid_gate < 0) &
                      (ir_valid_gate > 0)).float())
        self._record(f'level_{level_id}/joint_gate_vis_neg_ir_neg_ratio',
                     ((vis_valid_gate < 0) &
                      (ir_valid_gate < 0)).float())

        self._record_token_statistics(f'level_{level_id}/vis', vis_weight,
                                      valid_tokens)
        self._record_token_statistics(f'level_{level_id}/ir', ir_weight,
                                      valid_tokens)
        prefix = f'level_{level_id}/channel'
        if self.with_channel_gate:
            self._record(f'{prefix}_gate_mean', channel_gate)
            self._record(f'{prefix}_gate_std',
                         channel_gate.std(dim=-1, unbiased=False))
            self._record(f'{prefix}_gate_lt_0.1_ratio',
                         (channel_gate < 0.1).float())
            self._record(f'{prefix}_gate_gt_0.9_ratio',
                         (channel_gate > 0.9).float())
            self._record(f'{prefix}_active_ratio',
                         (channel_gate > 0.5).float())
        difference = (vis_level - ir_level).abs()
        difference = difference[valid]
        self._record(f'{prefix}_difference_mean', difference.mean())
        self._record(f'{prefix}_difference_std',
                     difference.std(unbiased=False))
