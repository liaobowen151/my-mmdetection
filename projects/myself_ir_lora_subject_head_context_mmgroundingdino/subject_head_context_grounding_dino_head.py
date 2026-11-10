"""Token-wise semantic/geometric context reranking for Grounding DINO."""

from typing import Dict, List, Optional, Tuple

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
class SubjectHeadContextGroundingDINOHead(GroundingDINOHead):
    """Dense subject-head/context reranking with optional deep supervision."""

    def __init__(self,
                 context_fusion: str = 'fixed',
                 b1_supervision: str = 'last',
                 b1_loss_weight: float = 1.0,
                 positive_iou_threshold: float = 0.5,
                 negative_iou_threshold: float = 0.3,
                 iou_target_temperature: float = 0.1,
                 importance_temperature: float = 1.0,
                 evidence_temperature: float = 1.0,
                 rerank_temperature: float = 1.0,
                 score_standardize_eps: float = 1e-6,
                 score_standardize_min_std: float = 0.05,
                 context_text_dim: int = 768,
                 match_dim: int = 256,
                 context_hidden_dim: int = 256,
                 context_dropout: float = 0.1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if context_fusion not in {'fixed', 'gated'}:
            raise ValueError('context_fusion must be fixed or gated.')
        if b1_supervision not in {'last', 'all6'}:
            raise ValueError('b1_supervision must be last or all6.')
        if not 0 <= negative_iou_threshold < positive_iou_threshold <= 1:
            raise ValueError('Expected 0 <= negative < positive <= 1.')
        for name, value in {
                'b1_loss_weight': b1_loss_weight,
                'iou_target_temperature': iou_target_temperature,
                'importance_temperature': importance_temperature,
                'evidence_temperature': evidence_temperature,
                'rerank_temperature': rerank_temperature,
                'score_standardize_eps': score_standardize_eps,
                'score_standardize_min_std': score_standardize_min_std,
        }.items():
            if value <= 0:
                raise ValueError(f'{name} must be positive.')

        self.context_fusion = context_fusion
        self.b1_supervision = b1_supervision
        self.b1_loss_weight = b1_loss_weight
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.iou_target_temperature = iou_target_temperature
        self.importance_temperature = importance_temperature
        self.evidence_temperature = evidence_temperature
        self.rerank_temperature = rerank_temperature
        self.score_standardize_eps = score_standardize_eps
        self.score_standardize_min_std = score_standardize_min_std

        self.semantic_query_norm = nn.LayerNorm(self.embed_dims)
        self.semantic_query_proj = nn.Linear(self.embed_dims, match_dim)
        self.semantic_text_norm = nn.LayerNorm(context_text_dim)
        self.semantic_text_proj = nn.Linear(context_text_dim, match_dim)

        self.geometry_proj = nn.Sequential(
            nn.Linear(6, context_hidden_dim), nn.GELU(),
            nn.Dropout(context_dropout),
            nn.Linear(context_hidden_dim, match_dim), nn.LayerNorm(match_dim))
        self.geometry_text_norm = nn.LayerNorm(context_text_dim)
        self.geometry_text_proj = nn.Linear(context_text_dim, match_dim)

        self.importance_norm = nn.LayerNorm(context_text_dim)
        self.importance_mlp = nn.Sequential(
            nn.Linear(context_text_dim, context_hidden_dim), nn.GELU(),
            nn.Dropout(context_dropout), nn.Linear(context_hidden_dim, 1))
        if context_fusion == 'gated':
            self.gate_norm = nn.LayerNorm(context_text_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(context_text_dim, context_hidden_dim), nn.GELU(),
                nn.Dropout(context_dropout), nn.Linear(context_hidden_dim, 1))
        else:
            self.gate_norm = None
            self.gate_mlp = None

    def init_weights(self) -> None:
        super().init_weights()
        modules = [
            self.semantic_query_proj, self.semantic_text_proj,
            self.geometry_text_proj, self.geometry_proj[0],
            self.geometry_proj[3], self.importance_mlp[0]
        ]
        if self.gate_mlp is not None:
            modules.append(self.gate_mlp[0])
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.)
        # Start importance uniformly. The Gated variant starts from a 50/50
        # convex mixture, whereas Fixed deliberately sums both experts with
        # unit coefficients.
        nn.init.constant_(self.importance_mlp[-1].weight, 0.)
        nn.init.constant_(self.importance_mlp[-1].bias, 0.)
        if self.gate_mlp is not None:
            nn.init.constant_(self.gate_mlp[-1].weight, 0.)
            nn.init.constant_(self.gate_mlp[-1].bias, 0.)

    def _zero_with_all_b1_parameters(self, reference: Tensor) -> Tensor:
        zero = reference.sum() * 0.
        module_names = [
            'semantic_query_norm', 'semantic_query_proj',
            'semantic_text_norm', 'semantic_text_proj', 'geometry_proj',
            'geometry_text_norm', 'geometry_text_proj', 'importance_norm',
            'importance_mlp'
        ]
        if self.gate_mlp is not None:
            module_names.extend(['gate_norm', 'gate_mlp'])
        for module_name in module_names:
            for parameter in getattr(self, module_name).parameters():
                zero = zero + parameter.sum() * 0.
        return zero

    def _standardize(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        mean = scores.mean()
        std = torch.sqrt(
            scores.var(unbiased=False) + self.score_standardize_eps)
        safe_std = std.clamp_min(self.score_standardize_min_std)
        return (scores - mean) / safe_std, std

    @staticmethod
    def _relative_rank(values: Tensor) -> Tensor:
        """Average percentile ranks, including deterministic tie handling."""
        count = values.numel()
        if count <= 1:
            return values.new_zeros(values.shape)
        less = (values[:, None] > values[None, :]).sum(dim=1)
        equal = (values[:, None] == values[None, :]).sum(dim=1)
        return (less + (equal - 1) * 0.5).to(values.dtype) / (count - 1)

    @staticmethod
    def _head_anchor_logits(cls_scores: Tensor,
                            head_mask: Tensor) -> Tensor:
        text_length = head_mask.numel()
        logits = cls_scores[:, :text_length]
        return logits.masked_fill(
            ~head_mask[None, :].bool(), float('-inf')).max(dim=-1).values

    def _context_evidence_single(
            self, query_feats: Tensor, bbox_preds: Tensor,
            context_text: Tensor,
            context_mask: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        context_indices = torch.nonzero(
            context_mask, as_tuple=False).squeeze(1)
        if context_indices.numel() == 0:
            zeros = query_feats.new_zeros((query_feats.size(0), ))
            return zeros, dict(
                token_importance=query_feats.new_empty((0, )),
                channel_gate=query_feats.new_empty((0, )),
                semantic_evidence=query_feats.new_empty(
                    (query_feats.size(0), 0)),
                geometry_evidence=query_feats.new_empty(
                    (query_feats.size(0), 0)))

        token_feats = context_text[context_indices].detach()
        importance_logits = self.importance_mlp(
            self.importance_norm(token_feats)).squeeze(-1)
        token_importance = F.softmax(
            importance_logits / self.importance_temperature, dim=0)

        query_projected = F.normalize(
            self.semantic_query_proj(
                self.semantic_query_norm(query_feats.detach())), dim=-1)
        semantic_text = F.normalize(
            self.semantic_text_proj(self.semantic_text_norm(token_feats)),
            dim=-1)
        semantic_evidence = query_projected @ semantic_text.transpose(0, 1)

        boxes = bbox_preds.detach()
        rank_x = self._relative_rank(boxes[:, 0])
        rank_y = self._relative_rank(boxes[:, 1])
        geometry_input = torch.cat(
            [boxes, rank_x[:, None], rank_y[:, None]], dim=-1)
        geometry_projected = F.normalize(
            self.geometry_proj(geometry_input), dim=-1)
        geometry_text = F.normalize(
            self.geometry_text_proj(self.geometry_text_norm(token_feats)),
            dim=-1)
        geometry_evidence = geometry_projected @ geometry_text.transpose(0, 1)

        if self.context_fusion == 'gated':
            channel_gate = torch.sigmoid(
                self.gate_mlp(self.gate_norm(token_feats)).squeeze(-1))
            fused_evidence = (
                channel_gate[None, :] * semantic_evidence +
                (1. - channel_gate[None, :]) * geometry_evidence)
        else:
            # Fixed uses one full-strength semantic expert plus one
            # full-strength geometric expert. ``channel_gate`` is retained as
            # an analysis field and records the unit coefficient.
            channel_gate = token_feats.new_ones((context_indices.numel(), ))
            fused_evidence = semantic_evidence + geometry_evidence
        fused_evidence = fused_evidence / self.evidence_temperature
        context_scores = fused_evidence @ token_importance
        return context_scores, dict(
            token_importance=token_importance,
            channel_gate=channel_gate,
            semantic_evidence=semantic_evidence,
            geometry_evidence=geometry_evidence)

    @staticmethod
    def _mean_or_zero(values: List[Tensor], zero: Tensor) -> Tensor:
        if values:
            return torch.stack(values).float().mean()
        return zero.detach()

    def _loss_b1_layer(
            self, query_feats: Tensor, cls_scores: Tensor,
            bbox_preds: Tensor, context_text: Tensor,
            subject_head_mask: Tensor, context_token_mask: Tensor,
            subject_head_valid: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tensor, Dict[str, Tensor]]:
        losses = []
        positive_counts, negative_counts, ignored_counts = [], [], []
        anchor_stds, context_stds, low_std_flags = [], [], []
        anchor_positive_ranks, final_positive_ranks = [], []
        anchor_recalls, final_recalls = [], []

        zero = self._zero_with_all_b1_parameters(query_feats)
        for image_id, data_sample in enumerate(batch_data_samples):
            if not bool(subject_head_valid[image_id]):
                continue
            if not bool(context_token_mask[image_id].any()):
                continue
            gt_instances = data_sample.gt_instances
            if len(gt_instances) != 1:
                continue

            bbox_pred = bbox_preds[image_id]
            img_h, img_w = data_sample.metainfo['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
            pred_xyxy = bbox_cxcywh_to_xyxy(bbox_pred) * factor
            gt_ious = bbox_overlaps(
                pred_xyxy.detach(),
                gt_instances.bboxes[0].unsqueeze(0).detach()).squeeze(1)
            positive_mask = gt_ious >= self.positive_iou_threshold
            negative_mask = gt_ious <= self.negative_iou_threshold
            ignored_mask = ~(positive_mask | negative_mask)
            positive_counts.append(positive_mask.sum().float())
            negative_counts.append(negative_mask.sum().float())
            ignored_counts.append(ignored_mask.sum().float())
            if not bool(positive_mask.any()) or not bool(negative_mask.any()):
                continue

            anchors = self._head_anchor_logits(
                cls_scores[image_id], subject_head_mask[image_id])
            context_scores, _ = self._context_evidence_single(
                query_feats[image_id], bbox_pred, context_text[image_id],
                context_token_mask[image_id])
            standardized_anchor, anchor_std = self._standardize(anchors)
            standardized_context, context_std = self._standardize(
                context_scores)
            final_logits = standardized_anchor.detach() + \
                standardized_context

            valid_mask = positive_mask | negative_mask
            valid_logits = final_logits[valid_mask] / self.rerank_temperature
            valid_positive_mask = positive_mask[valid_mask]
            positive_ious = gt_ious[positive_mask]
            soft_targets = F.softmax(
                positive_ious / self.iou_target_temperature, dim=0)
            targets = torch.zeros_like(valid_logits)
            targets[valid_positive_mask] = soft_targets.to(targets.dtype)
            losses.append(
                -(targets * F.log_softmax(valid_logits, dim=0)).sum())

            anchor_stds.append(anchor_std)
            context_stds.append(context_std)
            low_std_flags.append(
                (context_std < self.score_standardize_min_std).float())
            best_anchor_positive = anchors[positive_mask].max()
            best_final_positive = final_logits[positive_mask].max()
            anchor_positive_ranks.append(
                1 + (anchors > best_anchor_positive).sum())
            final_positive_ranks.append(
                1 + (final_logits > best_final_positive).sum())
            topk = min(10, anchors.numel())
            anchor_recalls.append(
                positive_mask[anchors.topk(topk).indices].any().float())
            final_recalls.append(
                positive_mask[final_logits.topk(topk).indices].any().float())

        if losses:
            layer_loss = torch.stack(losses).mean()
            valid_count = layer_loss.new_tensor(float(len(losses)))
        else:
            layer_loss = zero
            valid_count = zero.detach()
        metrics = dict(
            valid_samples=valid_count.detach(),
            positive_count=self._mean_or_zero(positive_counts, zero),
            negative_count=self._mean_or_zero(negative_counts, zero),
            ignored_count=self._mean_or_zero(ignored_counts, zero),
            anchor_std=self._mean_or_zero(anchor_stds, zero),
            context_std=self._mean_or_zero(context_stds, zero),
            context_low_std_rate=self._mean_or_zero(low_std_flags, zero),
            anchor_positive_rank=self._mean_or_zero(
                anchor_positive_ranks, zero),
            final_positive_rank=self._mean_or_zero(
                final_positive_ranks, zero),
            anchor_recall_at_10=self._mean_or_zero(anchor_recalls, zero),
            final_recall_at_10=self._mean_or_zero(final_recalls, zero))
        return layer_loss, metrics

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             context_text: Tensor, subject_head_mask: Tensor,
             context_token_mask: Tensor, subject_head_valid: Tensor,
             subject_head_token_index: Tensor,
             subject_head_confidence: Tensor, subject_head_char_span: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int]) -> dict:
        del subject_head_token_index, subject_head_confidence, \
            subject_head_char_span
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
        num_dn = 0 if dn_meta is None else dn_meta['num_denoising_queries']
        layer_ids = ([hidden_states.size(0) - 1]
                     if self.b1_supervision == 'last'
                     else list(range(hidden_states.size(0))))
        final_metrics = None
        for layer_id in layer_ids:
            layer_loss, metrics = self._loss_b1_layer(
                hidden_states[layer_id, :, num_dn:, :],
                all_cls_scores[layer_id, :, num_dn:, :],
                all_bbox_preds[layer_id, :, num_dn:, :], context_text,
                subject_head_mask, context_token_mask, subject_head_valid,
                batch_data_samples)
            key = ('loss_b1_context' if layer_id == hidden_states.size(0) - 1
                   else f'd{layer_id}.loss_b1_context')
            losses[key] = layer_loss * self.b1_loss_weight
            if layer_id == hidden_states.size(0) - 1:
                final_metrics = metrics
        if final_metrics is not None:
            for name, value in final_metrics.items():
                losses[f'b1_{name}'] = value
        return losses

    def _predict_context_single(
            self, cls_score: Tensor, bbox_pred: Tensor, query_feats: Tensor,
            context_text: Tensor, subject_head_mask: Tensor,
            context_token_mask: Tensor, subject_head_valid: Tensor,
            subject_head_token_index: Tensor,
            subject_head_confidence: Tensor, subject_head_char_span: Tensor,
            img_meta: dict, rescale: bool) -> InstanceData:
        anchors = self._head_anchor_logits(cls_score, subject_head_mask)
        standardized_anchor, _ = self._standardize(anchors)
        has_context = bool(subject_head_valid) and bool(
            context_token_mask.any())
        if has_context:
            context_scores, _ = self._context_evidence_single(
                query_feats, bbox_pred, context_text, context_token_mask)
            standardized_context, _ = self._standardize(context_scores)
        else:
            context_scores = anchors.new_zeros(anchors.shape)
            standardized_context = anchors.new_zeros(anchors.shape)
        final_logits = standardized_anchor + standardized_context

        max_per_img = min(self.test_cfg.get('max_per_img', len(bbox_pred)),
                          len(bbox_pred))
        final_scores, final_order = final_logits.topk(max_per_img)
        anchor_order = torch.argsort(anchors, descending=True)
        pre_ranks = torch.empty_like(anchor_order)
        pre_ranks[anchor_order] = torch.arange(
            1, anchor_order.numel() + 1, device=anchor_order.device)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred[final_order])
        img_h, img_w = img_meta['img_shape']
        det_bboxes[:, 0::2] *= img_w
        det_bboxes[:, 1::2] *= img_h
        det_bboxes[:, 0::2].clamp_(min=0, max=img_w)
        det_bboxes[:, 1::2].clamp_(min=0, max=img_h)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        count = final_order.numel()
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = final_scores.sigmoid()
        results.labels = final_order.new_zeros((count, ))
        results.final_ranking_logits = final_scores
        results.head_anchor_logits = anchors[final_order]
        results.head_anchor_scores = anchors[final_order].sigmoid()
        results.head_anchor_standardized = standardized_anchor[final_order]
        results.context_scores = context_scores[final_order]
        results.context_standardized = standardized_context[final_order]
        results.query_indices = final_order
        results.pre_rerank_ranks = pre_ranks[final_order]
        results.subject_head_token_index = subject_head_token_index.repeat(
            count)
        results.subject_head_confidence = subject_head_confidence.repeat(count)
        results.subject_head_char_span = subject_head_char_span[None, :].repeat(
            count, 1)
        return results

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                context_text: Tensor,
                subject_head_mask: Tensor,
                context_token_mask: Tensor,
                subject_head_valid: Tensor,
                subject_head_token_index: Tensor,
                subject_head_confidence: Tensor,
                subject_head_char_span: Tensor,
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
                results.append(self._predict_by_feat_single(
                    cls_scores[image_id], bbox_preds[image_id],
                    token_positive_map, data_sample.metainfo, rescale))
            else:
                results.append(self._predict_context_single(
                    cls_scores[image_id], bbox_preds[image_id],
                    hidden_states[-1, image_id], context_text[image_id],
                    subject_head_mask[image_id], context_token_mask[image_id],
                    subject_head_valid[image_id],
                    subject_head_token_index[image_id],
                    subject_head_confidence[image_id],
                    subject_head_char_span[image_id], data_sample.metainfo,
                    rescale))
        return results
