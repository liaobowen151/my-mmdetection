"""Head with independent Hungarian assignments for IR and VIS encoders."""

from typing import Dict, List

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from projects.myself_ir_guided_vis_mmgroundingdino import (
    IRGuidedVISGroundingDINOHead, )


@MODELS.register_module()
class DualEncoderIRGuidedVISGroundingDINOHead(
        IRGuidedVISGroundingDINOHead):
    """Assign IR and VIS positives independently within IR-selected Top-K."""

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             ir_enc_outputs_class: Tensor, ir_enc_outputs_coord: Tensor,
             vis_enc_outputs_class: Tensor, vis_enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        batch_gt_instances = [
            sample.gt_instances for sample in batch_data_samples
        ]
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]

        outs = self(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask
        losses = super(IRGuidedVISGroundingDINOHead, self).loss_by_feat(
            *outs, None, None, batch_gt_instances, batch_img_metas, dn_meta)

        with torch.no_grad():
            ir_targets = self._get_ir_shared_targets(
                ir_enc_outputs_class, ir_enc_outputs_coord,
                batch_gt_instances, batch_img_metas)
            vis_targets = self._get_ir_shared_targets(
                vis_enc_outputs_class, vis_enc_outputs_coord,
                batch_gt_instances, batch_img_metas)

        ir_losses = self._loss_from_shared_targets(
            ir_enc_outputs_class, ir_enc_outputs_coord, ir_targets,
            batch_img_metas)
        vis_losses = self._loss_from_shared_targets(
            vis_enc_outputs_class, vis_enc_outputs_coord, vis_targets,
            batch_img_metas)

        weights = self.encoder_loss_weights
        losses.update(
            ir_enc_loss_cls=weights['ir_cls'] * ir_losses[0],
            ir_enc_loss_bbox=weights['ir_bbox'] * ir_losses[1],
            ir_enc_loss_iou=weights['ir_iou'] * ir_losses[2],
            vis_enc_loss_cls=weights['vis_cls'] * vis_losses[0],
            vis_enc_loss_bbox=weights['vis_bbox'] * vis_losses[1],
            vis_enc_loss_iou=weights['vis_iou'] * vis_losses[2])

        ir_stats = self._ir_diagnostics(
            ir_enc_outputs_coord, batch_gt_instances, batch_img_metas,
            ir_targets)
        ir_stats['ir_matching_iou_gap'] = (
            ir_stats['ir_topk_max_iou'] -
            ir_stats['ir_hungarian_pos_iou'])
        losses.update(ir_stats)
        losses.update(self._assignment_diagnostics(
            ir_targets, vis_targets, ir_enc_outputs_coord,
            vis_enc_outputs_coord, batch_gt_instances, batch_img_metas))
        return losses

    @torch.no_grad()
    def _assignment_diagnostics(
            self, ir_targets: Dict, vis_targets: Dict,
            ir_bbox_preds: Tensor, vis_bbox_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Measure VIS matching quality and IR/VIS positive agreement."""
        vis_stats = self._ir_diagnostics(
            vis_bbox_preds, batch_gt_instances, batch_img_metas, vis_targets)

        agreement_sum = ir_bbox_preds.new_zeros(())
        agreement_count = ir_bbox_preds.new_zeros(())
        for ir_pos, vis_pos in zip(ir_targets['pos_inds'],
                                   vis_targets['pos_inds']):
            if ir_pos.numel() == 1 and vis_pos.numel() == 1:
                agreement_sum += (ir_pos[0] == vis_pos[0]).to(
                    agreement_sum.dtype)
                agreement_count += 1

        # The regular training log already contains per-rank reduced values.
        agreement = agreement_sum / agreement_count.clamp(min=1)
        return dict(
            vis_hungarian_pos_iou=vis_stats['ir_hungarian_pos_iou'],
            vis_selected_max_iou=vis_stats['ir_topk_max_iou'],
            ir_vis_pos_index_agreement=agreement)
