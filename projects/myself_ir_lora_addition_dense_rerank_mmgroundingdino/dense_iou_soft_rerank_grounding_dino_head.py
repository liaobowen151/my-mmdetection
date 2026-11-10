"""Dense IoU-aware soft reranking for Grounding DINO."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from projects.myself_ir_lora_addition_rerank_mmgroundingdino import (
    RerankGroundingDINOHead,
)


@MODELS.register_module()
class DenseIoUSoftRerankGroundingDINOHead(RerankGroundingDINOHead):
    """Rerank all final-layer matching queries with soft IoU targets.

    Queries with IoU >= ``positive_iou_threshold`` are soft positives,
    queries with IoU <= ``negative_iou_threshold`` are negatives, and the
    interval between both thresholds is ignored. CDN queries and encoder
    proposals keep their original losses but never enter this loss.
    """

    def __init__(self,
                 positive_iou_threshold: float = 0.5,
                 negative_iou_threshold: float = 0.3,
                 iou_target_temperature: float = 0.1,
                 **kwargs) -> None:
        if not 0 < positive_iou_threshold <= 1:
            raise ValueError('positive_iou_threshold must be in (0, 1].')
        if not 0 <= negative_iou_threshold < positive_iou_threshold:
            raise ValueError(
                'negative_iou_threshold must be non-negative and smaller '
                'than positive_iou_threshold.')
        if iou_target_temperature <= 0:
            raise ValueError('iou_target_temperature must be positive.')

        # The inherited negative-selection strategy is deliberately unused;
        # the dense loss consumes every unambiguous final-layer query.
        super().__init__(
            rerank_strategy='diverse',
            negative_iou_threshold=negative_iou_threshold,
            **kwargs)
        self.positive_iou_threshold = positive_iou_threshold
        self.iou_target_temperature = iou_target_temperature

    @staticmethod
    def _mean_or_zero(values, zero: Tensor) -> Tensor:
        if values:
            return torch.stack(values).float().mean()
        return zero.detach()

    def _loss_rerank(self, hidden_states: Tensor,
                     all_layers_cls_scores: Tensor,
                     all_layers_bbox_preds: Tensor,
                     text_token_mask: Tensor,
                     batch_data_samples: SampleList,
                     dn_meta: Optional[Dict[str, int]]) -> Dict[str, Tensor]:
        num_dn = 0 if dn_meta is None else dn_meta['num_denoising_queries']
        query_feats = hidden_states[-1, :, num_dn:, :]
        cls_scores = all_layers_cls_scores[-1, :, num_dn:, :]
        bbox_preds = all_layers_bbox_preds[-1, :, num_dn:, :]

        rerank_losses = []
        positive_counts = []
        negative_counts = []
        ignored_counts = []
        max_ious = []
        anchor_positive_ranks = []
        rerank_positive_ranks = []
        anchor_recalls_at_10 = []
        rerank_recalls_at_10 = []

        for image_id, data_sample in enumerate(batch_data_samples):
            gt_instances = data_sample.gt_instances
            if len(gt_instances) != 1:
                continue

            bbox_pred = bbox_preds[image_id]
            img_h, img_w = data_sample.metainfo['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
            pred_boxes_xyxy = bbox_cxcywh_to_xyxy(bbox_pred) * factor
            gt_box = gt_instances.bboxes[0].unsqueeze(0)
            # IoU defines discrete supervision and must not move boxes.
            gt_ious = bbox_overlaps(
                pred_boxes_xyxy.detach(), gt_box.detach()).squeeze(1)

            positive_mask = gt_ious >= self.positive_iou_threshold
            negative_mask = gt_ious <= self.negative_iou_threshold
            ignored_mask = ~(positive_mask | negative_mask)
            num_positive = positive_mask.sum()
            num_negative = negative_mask.sum()

            positive_counts.append(num_positive.float())
            negative_counts.append(num_negative.float())
            ignored_counts.append(ignored_mask.sum().float())
            max_ious.append(gt_ious.max())

            if num_positive == 0 or num_negative == 0:
                continue

            anchors = self._anchor_logits(
                cls_scores[image_id], text_token_mask[image_id])
            residual_logits = self._rerank_logits(
                query_feats[image_id], bbox_pred)
            combined_logits = (
                anchors.detach() +
                self.rerank_score_weight * residual_logits)

            valid_mask = positive_mask | negative_mask
            valid_logits = (
                combined_logits[valid_mask] / self.rerank_temperature)
            valid_positive_mask = positive_mask[valid_mask]
            positive_ious = gt_ious[positive_mask]
            soft_positive_targets = F.softmax(
                positive_ious / self.iou_target_temperature, dim=0)
            targets = torch.zeros_like(valid_logits)
            targets[valid_positive_mask] = soft_positive_targets.to(
                targets.dtype)
            rerank_losses.append(
                -(targets * F.log_softmax(valid_logits, dim=0)).sum())

            best_anchor_positive = anchors[positive_mask].max()
            best_rerank_positive = combined_logits[positive_mask].max()
            anchor_positive_ranks.append(
                1 + (anchors > best_anchor_positive).sum())
            rerank_positive_ranks.append(
                1 + (combined_logits > best_rerank_positive).sum())

            topk = min(10, anchors.numel())
            anchor_topk = anchors.topk(topk).indices
            rerank_topk = combined_logits.topk(topk).indices
            anchor_recalls_at_10.append(
                positive_mask[anchor_topk].any().float())
            rerank_recalls_at_10.append(
                positive_mask[rerank_topk].any().float())

        # Keep every dense-reranker parameter in the graph on ranks whose
        # local batch contains no IoU-qualified positive. This avoids DDP
        # unused-parameter failures without creating a non-zero gradient.
        zero = hidden_states.sum() * 0.
        for parameter in self.rerank_box_embed.parameters():
            zero = zero + parameter.sum() * 0.
        for parameter in self.rerank_head.parameters():
            zero = zero + parameter.sum() * 0.
        if rerank_losses:
            loss = torch.stack(rerank_losses).mean()
            valid_count = loss.new_tensor(float(len(rerank_losses)))
        else:
            loss = zero
            valid_count = zero.detach()

        return dict(
            loss_dense_rerank=loss * self.rerank_loss_weight,
            dense_positive_count=self._mean_or_zero(positive_counts, zero),
            dense_negative_count=self._mean_or_zero(negative_counts, zero),
            dense_ignore_count=self._mean_or_zero(ignored_counts, zero),
            dense_valid_samples=valid_count.detach(),
            dense_max_iou=self._mean_or_zero(max_ious, zero),
            dense_anchor_positive_rank=self._mean_or_zero(
                anchor_positive_ranks, zero),
            dense_rerank_positive_rank=self._mean_or_zero(
                rerank_positive_ranks, zero),
            dense_anchor_recall_at_10=self._mean_or_zero(
                anchor_recalls_at_10, zero),
            dense_rerank_recall_at_10=self._mean_or_zero(
                rerank_recalls_at_10, zero))

    def _predict_reranked_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                 query_feats: Tensor,
                                 text_token_mask: Tensor, img_meta: dict,
                                 rescale: bool) -> InstanceData:
        """Rerank all 900 matching queries before selecting final results."""
        anchor_logits = self._anchor_logits(cls_score, text_token_mask)
        residual_logits = self._rerank_logits(query_feats, bbox_pred)
        combined_logits = (
            anchor_logits + self.rerank_score_weight * residual_logits)

        max_per_img = min(self.test_cfg.get('max_per_img', len(bbox_pred)),
                          len(bbox_pred))
        scores, final_order = combined_logits.topk(max_per_img)
        scores = scores.sigmoid()

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred[final_order])
        img_h, img_w = img_meta['img_shape']
        det_bboxes[:, 0::2] *= img_w
        det_bboxes[:, 1::2] *= img_h
        det_bboxes[:, 0::2].clamp_(min=0, max=img_w)
        det_bboxes[:, 1::2].clamp_(min=0, max=img_h)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = scores.new_zeros(scores.shape, dtype=torch.long)
        return results
