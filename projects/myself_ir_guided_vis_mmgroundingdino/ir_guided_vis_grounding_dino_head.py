"""Grounding DINO head for IR-guided VIS decoding."""

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import InstanceList, reduce_mean


@MODELS.register_module()
class IRGuidedVISGroundingDINOHead(GroundingDINOHead):
    """Use one IR assignment for modality-weighted IR/VIS encoder losses."""

    def __init__(self, encoder_loss_weights: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        required = {'ir_cls', 'ir_bbox', 'ir_iou',
                    'vis_cls', 'vis_bbox', 'vis_iou'}
        missing = required.difference(encoder_loss_weights)
        if missing:
            raise ValueError(f'Missing encoder loss weights: {sorted(missing)}')
        self.encoder_loss_weights = {
            key: float(encoder_loss_weights[key]) for key in required
        }

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

        # Preserve the original decoder matching and denoising losses, while
        # deliberately disabling DINO's single encoder-loss branch.
        losses = super().loss_by_feat(
            *outs,
            None,
            None,
            batch_gt_instances,
            batch_img_metas,
            dn_meta)

        with torch.no_grad():
            shared_targets = self._get_ir_shared_targets(
                ir_enc_outputs_class, ir_enc_outputs_coord,
                batch_gt_instances, batch_img_metas)

        ir_losses = self._loss_from_shared_targets(
            ir_enc_outputs_class, ir_enc_outputs_coord, shared_targets,
            batch_img_metas)
        vis_losses = self._loss_from_shared_targets(
            vis_enc_outputs_class, vis_enc_outputs_coord, shared_targets,
            batch_img_metas)

        weights = self.encoder_loss_weights
        losses.update(
            ir_enc_loss_cls=weights['ir_cls'] * ir_losses[0],
            ir_enc_loss_bbox=weights['ir_bbox'] * ir_losses[1],
            ir_enc_loss_iou=weights['ir_iou'] * ir_losses[2],
            vis_enc_loss_cls=weights['vis_cls'] * vis_losses[0],
            vis_enc_loss_bbox=weights['vis_bbox'] * vis_losses[1],
            vis_enc_loss_iou=weights['vis_iou'] * vis_losses[2])
        losses.update(self._ir_diagnostics(
            ir_enc_outputs_coord, batch_gt_instances, batch_img_metas,
            shared_targets))
        return losses

    def _get_ir_shared_targets(self, cls_scores: Tensor, bbox_preds: Tensor,
                               batch_gt_instances: InstanceList,
                               batch_img_metas: List[dict]) -> Dict:
        targets = [
            self._get_targets_single(cls_scores[i], bbox_preds[i],
                                     batch_gt_instances[i],
                                     batch_img_metas[i])
            for i in range(cls_scores.size(0))
        ]
        labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds = \
            zip(*targets)
        return dict(
            labels=list(labels),
            label_weights=list(label_weights),
            bbox_targets=list(bbox_targets),
            bbox_weights=list(bbox_weights),
            pos_inds=list(pos_inds),
            neg_inds=list(neg_inds),
            num_total_pos=sum(item.numel() for item in pos_inds),
            num_total_neg=sum(item.numel() for item in neg_inds))

    def _loss_from_shared_targets(
            self, cls_scores: Tensor, bbox_preds: Tensor, targets: Dict,
            batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor, Tensor]:
        labels = torch.stack(targets['labels'], 0)
        label_weights = torch.stack(targets['label_weights'], 0)
        bbox_targets = torch.cat(targets['bbox_targets'], 0)
        bbox_weights = torch.cat(targets['bbox_weights'], 0)
        num_total_pos = targets['num_total_pos']
        num_total_neg = targets['num_total_neg']

        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1).repeat(
            1, cls_scores.size(1), 1)
        masked_cls_scores = torch.masked_select(
            cls_scores, text_mask).contiguous()
        masked_labels = torch.masked_select(labels, text_mask)
        expanded_label_weights = label_weights[..., None].repeat(
            1, 1, text_mask.size(-1))
        masked_label_weights = torch.masked_select(
            expanded_label_weights, text_mask)

        cls_avg_factor = (num_total_pos * 1.0 +
                          num_total_neg * self.bg_cls_weight)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                masked_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss is not supported for this head.')
        loss_cls = self.loss_cls(
            masked_cls_scores, masked_labels, masked_label_weights,
            avg_factor=cls_avg_factor)

        pos_avg = loss_cls.new_tensor([num_total_pos])
        pos_avg = torch.clamp(reduce_mean(pos_avg), min=1).item()
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor(
                [img_w, img_h, img_w, img_h]).unsqueeze(0)
            factors.append(factor.repeat(bbox_pred.size(0), 1))
        factors = torch.cat(factors, 0)

        flat_bbox_preds = bbox_preds.reshape(-1, 4)
        pred_xyxy = bbox_cxcywh_to_xyxy(flat_bbox_preds) * factors
        target_xyxy = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_iou = self.loss_iou(
            pred_xyxy, target_xyxy, bbox_weights, avg_factor=pos_avg)
        loss_bbox = self.loss_bbox(
            flat_bbox_preds, bbox_targets, bbox_weights,
            avg_factor=pos_avg)
        return loss_cls, loss_bbox, loss_iou

    @torch.no_grad()
    def _ir_diagnostics(self, ir_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict], targets: Dict) -> Dict:
        device = ir_bbox_preds.device
        dtype = ir_bbox_preds.dtype
        pos_iou_sum = torch.zeros((), device=device, dtype=dtype)
        pos_count = torch.zeros((), device=device, dtype=dtype)
        best_iou_sum = torch.zeros((), device=device, dtype=dtype)
        gt_count = torch.zeros((), device=device, dtype=dtype)
        recall_50_count = torch.zeros((), device=device, dtype=dtype)
        recall_70_count = torch.zeros((), device=device, dtype=dtype)

        for img_id, (bbox_pred, gt_instances, img_meta) in enumerate(zip(
                ir_bbox_preds, batch_gt_instances, batch_img_metas)):
            gt_bboxes = gt_instances.bboxes
            if len(gt_bboxes) == 0:
                continue
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
            pred_xyxy = bbox_cxcywh_to_xyxy(bbox_pred) * factor
            all_ious = bbox_overlaps(pred_xyxy, gt_bboxes, mode='iou')
            gt_best_ious = all_ious.max(dim=0).values
            best_iou_sum += gt_best_ious.sum()
            gt_count += gt_best_ious.numel()
            recall_50_count += (gt_best_ious >= 0.5).sum()
            recall_70_count += (gt_best_ious >= 0.7).sum()

            pos_inds = targets['pos_inds'][img_id]
            if pos_inds.numel() > 0:
                target_xyxy = bbox_cxcywh_to_xyxy(
                    targets['bbox_targets'][img_id][pos_inds]) * factor
                pos_ious = bbox_overlaps(
                    pred_xyxy[pos_inds], target_xyxy,
                    mode='iou', is_aligned=True)
                pos_iou_sum += pos_ious.sum()
                pos_count += pos_ious.numel()

        stats = torch.stack([
            pos_iou_sum, pos_count, best_iou_sum, gt_count,
            recall_50_count, recall_70_count
        ])
        stats = reduce_mean(stats)
        eps = torch.finfo(dtype).eps
        return dict(
            ir_hungarian_pos_iou=stats[0] / stats[1].clamp(min=eps),
            ir_topk_max_iou=stats[2] / stats[3].clamp(min=eps),
            ir_topk_recall_50=stats[4] / stats[3].clamp(min=eps),
            ir_topk_recall_70=stats[5] / stats[3].clamp(min=eps))
