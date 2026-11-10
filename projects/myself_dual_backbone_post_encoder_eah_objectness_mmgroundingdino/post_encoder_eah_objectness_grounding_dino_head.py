"""Grounding DINO head with proposal-level EAM Focal Loss."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import reduce_mean


@MODELS.register_module()
class PostEncoderEAHObjectnessGroundingDINOHead(GroundingDINOHead):
    """Add phrase/objectness supervision to selected EAM proposal logits."""

    def __init__(self,
                 *args,
                 loss_eam: dict = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.5),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_eam = MODELS.build(loss_eam)

    def loss(self,
             hidden_states: Tensor,
             references: List[Tensor],
             memory_text: Tensor,
             text_token_mask: Tensor,
             enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int],
             eam_logits: Optional[Tensor] = None) -> dict:
        """Compute unchanged Grounding DINO losses plus Top-K EAM loss."""
        losses = super().loss(
            hidden_states=hidden_states,
            references=references,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            batch_data_samples=batch_data_samples,
            dn_meta=dn_meta)

        if eam_logits is None:
            return losses
        if eam_logits.ndim != 2:
            raise ValueError('eam_logits must have shape [B, num_queries].')
        if eam_logits.shape[:2] != enc_outputs_class.shape[:2]:
            raise ValueError(
                'EAM logits and encoder outputs must select the same queries: '
                f'{tuple(eam_logits.shape)} versus '
                f'{tuple(enc_outputs_class.shape[:2])}.')

        batch_gt_instances = [
            data_sample.gt_instances for data_sample in batch_data_samples
        ]
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        cls_scores_list = [
            enc_outputs_class[i]
            for i in range(enc_outputs_class.size(0))
        ]
        bbox_preds_list = [
            enc_outputs_coord[i]
            for i in range(enc_outputs_coord.size(0))
        ]

        # Re-run the exact encoder Hungarian target construction under no_grad.
        # This keeps all original matching costs unchanged and avoids modifying
        # the MMDetection core head merely to expose its internal assignment.
        with torch.no_grad():
            target_results = self.get_targets(
                cls_scores_list, bbox_preds_list, batch_gt_instances,
                batch_img_metas)
        _, _, _, bbox_weights_list, _, _ = target_results
        bbox_weights = torch.stack(bbox_weights_list, dim=0)

        # Hungarian positives have four unit bbox weights; backgrounds have
        # four zeros. This is exactly equivalent to its internal ``pos_inds``.
        eam_targets = (bbox_weights.sum(dim=-1) > 0).to(eam_logits.dtype)

        # EAH assigns -inf to padded image positions. If an image has fewer
        # valid encoder positions than ``num_queries``, Top-K necessarily
        # includes some of them. They are not real background proposals and
        # must be excluded: BCEWithLogits/FocalLoss(-inf, 0) evaluates to NaN.
        if torch.isnan(eam_logits).any() or torch.isposinf(eam_logits).any():
            raise FloatingPointError(
                'EAM logits contain unexpected NaN or +inf values.')
        valid_eam_mask = torch.isfinite(eam_logits)
        valid_eam_logits = eam_logits[valid_eam_mask]
        valid_eam_targets = eam_targets[valid_eam_mask]

        if valid_eam_logits.numel() == 0:
            losses['loss_eam'] = hidden_states.sum() * 0.0
            return losses

        num_valid_pos = valid_eam_targets.sum()
        synced_num_pos = reduce_mean(
            num_valid_pos.detach().float().reshape(1))
        eam_avg_factor = max(float(synced_num_pos.item()), 1.0)
        losses['loss_eam'] = self.loss_eam(
            valid_eam_logits.reshape(-1, 1),
            valid_eam_targets.reshape(-1, 1),
            avg_factor=eam_avg_factor)
        return losses
