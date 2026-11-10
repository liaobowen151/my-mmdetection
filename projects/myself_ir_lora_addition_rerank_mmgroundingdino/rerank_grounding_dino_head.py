"""Candidate-level full-expression reranking for Grounding DINO."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps


@MODELS.register_module()
class RerankGroundingDINOHead(GroundingDINOHead):
    """Grounding DINO head with a final-layer listwise reranking loss.

    Only normal matching queries are eligible. Encoder proposals and CDN
    denoising queries retain their original losses but never enter the
    candidate set.
    """

    _VALID_STRATEGIES = ('random', 'score', 'diverse')

    def __init__(self,
                 rerank_strategy: str = 'random',
                 num_rerank_negatives: int = 9,
                 negative_iou_threshold: float = 0.3,
                 diverse_nms_iou_threshold: float = 0.7,
                 rerank_loss_weight: float = 1.0,
                 rerank_temperature: float = 1.0,
                 rerank_score_weight: float = 1.0,
                 rerank_topk: int = 10,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if rerank_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f'Unknown rerank_strategy={rerank_strategy!r}; expected '
                f'one of {self._VALID_STRATEGIES}.')
        if num_rerank_negatives < 1:
            raise ValueError('num_rerank_negatives must be positive.')
        if not 0 <= negative_iou_threshold < 1:
            raise ValueError('negative_iou_threshold must be in [0, 1).')
        if not 0 < diverse_nms_iou_threshold <= 1:
            raise ValueError(
                'diverse_nms_iou_threshold must be in (0, 1].')
        if rerank_temperature <= 0:
            raise ValueError('rerank_temperature must be positive.')
        if rerank_topk < 2:
            raise ValueError('rerank_topk must be at least 2.')

        self.rerank_strategy = rerank_strategy
        self.num_rerank_negatives = num_rerank_negatives
        self.negative_iou_threshold = negative_iou_threshold
        self.diverse_nms_iou_threshold = diverse_nms_iou_threshold
        self.rerank_loss_weight = rerank_loss_weight
        self.rerank_temperature = rerank_temperature
        self.rerank_score_weight = rerank_score_weight
        self.rerank_topk = rerank_topk

        self.rerank_box_embed = nn.Sequential(
            nn.Linear(4, self.embed_dims), nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims))
        self.rerank_head = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.ReLU(inplace=True), nn.Linear(self.embed_dims, 1))

    def init_weights(self) -> None:
        super().init_weights()
        for module in self.rerank_box_embed:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.)
        nn.init.xavier_uniform_(self.rerank_head[0].weight)
        nn.init.constant_(self.rerank_head[0].bias, 0.)
        # Start from the original Grounding DINO ranking.
        nn.init.constant_(self.rerank_head[-1].weight, 0.)
        nn.init.constant_(self.rerank_head[-1].bias, 0.)

    def _rerank_logits(self, query_feats: Tensor,
                       bbox_preds: Tensor) -> Tensor:
        # Box coordinates describe a selected candidate; the reranking loss
        # should not move boxes merely to make their positional embedding more
        # discriminative. Query features still receive reranking gradients.
        box_feats = self.rerank_box_embed(bbox_preds.detach())
        return self.rerank_head(
            torch.cat([query_feats, box_feats], dim=-1)).squeeze(-1)

    @staticmethod
    def _anchor_logits(cls_scores: Tensor,
                       text_token_mask: Tensor) -> Tensor:
        valid_length = text_token_mask.numel()
        valid_mask = text_token_mask.bool()
        return cls_scores[:, :valid_length][:, valid_mask].max(dim=-1).values

    def _select_diverse(self, candidate_inds: Tensor, anchor_logits: Tensor,
                        boxes_xyxy: Tensor) -> Tensor:
        order = candidate_inds[torch.argsort(
            anchor_logits[candidate_inds], descending=True)]
        selected = []
        for index in order:
            if not selected:
                selected.append(index)
            else:
                selected_tensor = torch.stack(selected)
                overlaps = bbox_overlaps(
                    boxes_xyxy[index].unsqueeze(0),
                    boxes_xyxy[selected_tensor]).squeeze(0)
                if torch.all(overlaps <= self.diverse_nms_iou_threshold):
                    selected.append(index)
            if len(selected) == self.num_rerank_negatives:
                break

        if not selected:
            return candidate_inds.new_empty((0, ))
        return torch.stack(selected)

    def _select_negatives(self, candidate_inds: Tensor,
                          anchor_logits: Tensor,
                          boxes_xyxy: Tensor) -> Tensor:
        num_to_select = min(self.num_rerank_negatives,
                            candidate_inds.numel())
        if num_to_select == 0:
            return candidate_inds.new_empty((0, ))
        if self.rerank_strategy == 'random':
            permutation = torch.randperm(
                candidate_inds.numel(), device=candidate_inds.device)
            return candidate_inds[permutation[:num_to_select]]
        if self.rerank_strategy == 'score':
            scores = anchor_logits[candidate_inds]
            return candidate_inds[scores.topk(num_to_select).indices]
        return self._select_diverse(candidate_inds, anchor_logits,
                                    boxes_xyxy)[:num_to_select]

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
        positive_ranks = []
        negative_counts = []
        for image_id, data_sample in enumerate(batch_data_samples):
            gt_instances = data_sample.gt_instances
            if len(gt_instances) != 1:
                continue
            cls_score = cls_scores[image_id]
            bbox_pred = bbox_preds[image_id]
            img_meta = data_sample.metainfo
            targets = self._get_targets_single(cls_score, bbox_pred,
                                               gt_instances, img_meta)
            pos_inds = targets[4]
            if pos_inds.numel() != 1:
                continue
            positive_ind = pos_inds[0]

            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
            pred_boxes_xyxy = bbox_cxcywh_to_xyxy(bbox_pred) * factor
            gt_box = gt_instances.bboxes[0].unsqueeze(0)
            gt_ious = bbox_overlaps(pred_boxes_xyxy, gt_box).squeeze(1)

            valid_negative = gt_ious <= self.negative_iou_threshold
            valid_negative[positive_ind] = False
            negative_pool = torch.nonzero(
                valid_negative, as_tuple=False).squeeze(1)
            if negative_pool.numel() == 0:
                continue

            anchors = self._anchor_logits(
                cls_score, text_token_mask[image_id])
            negative_inds = self._select_negatives(
                negative_pool, anchors, pred_boxes_xyxy)
            if negative_inds.numel() == 0:
                continue
            candidate_inds = torch.cat(
                [positive_ind.view(1), negative_inds])

            context_logits = self._rerank_logits(
                query_feats[image_id, candidate_inds],
                bbox_pred[candidate_inds])
            candidate_logits = (
                anchors[candidate_inds].detach() +
                self.rerank_score_weight * context_logits)
            candidate_logits = candidate_logits / self.rerank_temperature
            target = candidate_logits.new_zeros((1, ), dtype=torch.long)
            rerank_losses.append(
                F.cross_entropy(candidate_logits.unsqueeze(0), target))
            positive_ranks.append(
                1 + (candidate_logits[1:] > candidate_logits[0]).sum())
            negative_counts.append(negative_inds.new_tensor(
                negative_inds.numel(), dtype=torch.float32))

        if rerank_losses:
            loss = torch.stack(rerank_losses).mean()
            positive_rank = torch.stack(positive_ranks).float().mean()
            negative_count = torch.stack(negative_counts).mean()
            valid_count = loss.new_tensor(float(len(rerank_losses)))
        else:
            loss = hidden_states.sum() * 0.
            positive_rank = loss.detach()
            negative_count = loss.detach()
            valid_count = loss.detach()
        return dict(
            loss_rerank=loss * self.rerank_loss_weight,
            rerank_positive_rank=positive_rank.detach(),
            rerank_negative_count=negative_count.detach(),
            rerank_valid_samples=valid_count.detach())

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int]) -> dict:
        losses = super().loss(
            hidden_states=hidden_states,
            references=references,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
            batch_data_samples=batch_data_samples,
            dn_meta=dn_meta)
        all_cls_scores, all_bbox_preds = self(
            hidden_states, references, memory_text, text_token_mask)
        losses.update(
            self._loss_rerank(hidden_states, all_cls_scores, all_bbox_preds,
                              text_token_mask, batch_data_samples, dn_meta))
        return losses

    def _predict_reranked_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                 query_feats: Tensor,
                                 text_token_mask: Tensor, img_meta: dict,
                                 rescale: bool) -> InstanceData:
        anchor_logits = self._anchor_logits(cls_score, text_token_mask)
        anchor_order = torch.argsort(anchor_logits, descending=True)
        max_per_img = min(self.test_cfg.get('max_per_img', len(bbox_pred)),
                          len(bbox_pred))
        topk = min(self.rerank_topk, anchor_order.numel(), max_per_img)
        rerank_inds = anchor_order[:topk]
        context_logits = self._rerank_logits(query_feats[rerank_inds],
                                             bbox_pred[rerank_inds])
        combined_logits = (
            anchor_logits[rerank_inds] +
            self.rerank_score_weight * context_logits)
        rerank_order = rerank_inds[torch.argsort(
            combined_logits, descending=True)]

        remaining = anchor_order[topk:max_per_img]
        final_order = torch.cat([rerank_order, remaining])
        scores = anchor_logits[final_order].sigmoid()
        if topk:
            scores[:topk] = combined_logits[torch.argsort(
                combined_logits, descending=True)].sigmoid()

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

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> List[InstanceData]:
        all_cls_scores, all_bbox_preds = self(
            hidden_states, references, memory_text, text_token_mask)
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        results = []
        for image_id, data_sample in enumerate(batch_data_samples):
            token_positive_map = data_sample.token_positive_map
            if token_positive_map is not None:
                results.append(
                    self._predict_by_feat_single(
                        cls_scores[image_id], bbox_preds[image_id],
                        token_positive_map, data_sample.metainfo, rescale))
            else:
                results.append(
                    self._predict_reranked_single(
                        cls_scores[image_id], bbox_preds[image_id],
                        hidden_states[-1, image_id],
                        text_token_mask[image_id], data_sample.metainfo,
                        rescale))
        return results
