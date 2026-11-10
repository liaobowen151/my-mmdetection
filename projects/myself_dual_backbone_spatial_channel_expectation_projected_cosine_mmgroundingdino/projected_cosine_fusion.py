"""Shared projected-cosine similarity for the otherwise unchanged E1-Z."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_dual_backbone_spatial_channel_expectation_mmgroundingdino.spatial_channel_expectation_fusion import (  # noqa: E501
    SpatialChannelExpectationFusion, )


@MODELS.register_module()
class SpatialChannelExpectationProjectedCosineFusion(
        SpatialChannelExpectationFusion):
    """Use one shared learned space for VIS/IR-to-token similarity.

    This class changes only ``_similarity_and_evidence`` from the parent
    E1-Z fusion.  RGB and IR calls pass through the same ``visual_projection``
    and ``visual_norm`` modules.  Both projected visual features and projected
    text tokens are L2-normalized before their dot product, and the resulting
    cosine similarity is divided by one fixed shared temperature.

    All downstream behavior (valid-position averaging, level-balanced token
    evidence, token softmax, spatial z-score, tanh gate, and residual fusion)
    remains inherited without modification.
    """

    def __init__(self,
                 *args,
                 similarity_temperature: float = 1.0,
                 similarity_normalization_eps: float = 1e-6,
                 enable_hard_switch: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if similarity_temperature <= 0:
            raise ValueError('similarity_temperature must be positive.')
        if similarity_normalization_eps <= 0:
            raise ValueError(
                'similarity_normalization_eps must be positive.')

        self.similarity_temperature = float(similarity_temperature)
        self.similarity_normalization_eps = float(
            similarity_normalization_eps)
        self.enable_hard_switch = bool(enable_hard_switch)

        # A single module instance is deliberately used by both modalities so
        # that their similarities remain in the same learned coordinate system.
        self.visual_projection = nn.Linear(
            self.embed_dims, self.embed_dims, bias=False)
        self.visual_norm = nn.LayerNorm(self.embed_dims)
        self.text_projection = nn.Linear(
            self.embed_dims, self.embed_dims, bias=False)
        self._init_alignment_as_identity()

        # Existing projected-cosine experiments remain enabled by default.
        # Only a config that explicitly requests the hard switch starts from
        # the exact base fusion (VIS + IR).  The state is checkpointed so a
        # resumed run preserves its current stage.
        self.register_buffer(
            'projected_cosine_e1z_enabled',
            torch.tensor(not self.enable_hard_switch),
            persistent=True)

    def set_projected_cosine_e1z_enabled(self, enabled: bool) -> None:
        """Select exact base addition or projected-cosine E1-Z fusion."""
        self.projected_cosine_e1z_enabled.fill_(bool(enabled))

    def is_projected_cosine_e1z_enabled(self) -> bool:
        return bool(self.projected_cosine_e1z_enabled.item())

    def _init_alignment_as_identity(self) -> None:
        """Preserve the pretrained feature axes at initialization."""
        with torch.no_grad():
            nn.init.eye_(self.visual_projection.weight)
            nn.init.eye_(self.text_projection.weight)
            nn.init.ones_(self.visual_norm.weight)
            nn.init.zeros_(self.visual_norm.bias)

    def _similarity_and_evidence(
            self, visual: Tensor, text: Tensor,
            valid_positions: Tensor):
        """Return shared projected cosine and unchanged spatial evidence."""
        visual_aligned = self.visual_norm(
            self.visual_projection(visual))
        text_aligned = self.text_projection(text)

        visual_aligned = F.normalize(
            visual_aligned,
            p=2,
            dim=-1,
            eps=self.similarity_normalization_eps)
        text_aligned = F.normalize(
            text_aligned,
            p=2,
            dim=-1,
            eps=self.similarity_normalization_eps)

        similarity = torch.einsum(
            'bnc,blc->bnl', visual_aligned, text_aligned)
        similarity = similarity / self.similarity_temperature

        # This is intentionally identical to the parent implementation.
        position_weight = valid_positions[:, :, None].to(similarity.dtype)
        position_count = position_weight.sum(dim=1).clamp_min(1.0)
        evidence = (similarity * position_weight).sum(dim=1)
        evidence = evidence / position_count
        return similarity, evidence

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
        """Use exact base addition until the schedule enables E1-Z."""
        if not self.is_projected_cosine_e1z_enabled():
            if vis_feat.shape != ir_feat.shape:
                raise ValueError('VIS and IR feature shapes must match.')
            fused = vis_feat + ir_feat
            # Keep every new fusion parameter visible to DDP while preserving
            # the exact numerical result of base addition. This avoids the
            # find_unused_parameters/reentrant-checkpoint incompatibility in
            # Grounding DINO. The schedule hook restores these parameters and
            # clears their optimizer state throughout the base stage.
            parameter_anchor = sum(
                parameter.reshape(-1)[0] * 0.0
                for parameter in self.parameters()
                if parameter.requires_grad)
            fused = fused + parameter_anchor.to(fused.dtype)
            if feat_mask is not None:
                fused = fused.masked_fill(
                    feat_mask[..., None].to(torch.bool), 0)
            return fused

        return super().forward(
            vis_feat=vis_feat,
            ir_feat=ir_feat,
            text_feat=text_feat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            feat_pos=feat_pos,
            feat_mask=feat_mask,
            text_token_mask=text_token_mask)

    def get_statistics(self):
        statistics = super().get_statistics()
        statistics['stage/projected_cosine_e1z_enabled'] = torch.tensor(
            [float(self.is_projected_cosine_e1z_enabled()), 1.0],
            dtype=torch.float64,
            device=self.projected_cosine_e1z_enabled.device)
        return statistics
