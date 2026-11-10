#!/usr/bin/env python3
"""Measure encoder Top-K proposal IoU with GT, split by VIS/IR origin.

The competitive model concatenates candidates in [VIS, IR] order before its
global Top-K selection.  This tool wraps ``pre_decoder`` on the model instance,
reproduces that selection without changing inference, and compares every
selected reference box with all GT boxes in the same image.
"""

import argparse
import csv
import json
import os
import os.path as osp
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.utils import ProgressBar

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores, )


MATCH_FIELDNAMES = [
    'sample_idx', 'img_id', 'img_path', 'stage', 'threshold_table',
    'output_rank', 'query_rank', 'modality', 'candidate_index',
    'modality_candidate_index', 'score', 'matched_gt_index', 'max_gt_iou',
    'x1_norm', 'y1_norm', 'x2_norm', 'y2_norm', 'x1', 'y1', 'x2', 'y2'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count competitive encoder Top-K boxes whose maximum IoU '
        'with any GT is greater than a threshold, grouped by modality.')
    parser.add_argument('config', help='Model/test-data config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Defaults to <checkpoint_dir>/topk_gt_iou_analysis.')
    parser.add_argument(
        '--iou-thresholds',
        type=float,
        nargs='+',
        default=[0.8],
        help='Strict thresholds: a proposal is matched when max IoU > value.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Analyze only the first N samples (for smoke tests).')
    parser.add_argument(
        '--device', default='cuda:0', help='Inference device: cuda:N or cpu.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def threshold_key(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(array.mean()),
        std=float(array.std()),
        min=float(array.min()),
        median=float(np.median(array)),
        max=float(array.max()))


def get_gt_boxes(data_sample, device: torch.device) -> torch.Tensor:
    """Return resized-image GT boxes as a float tensor in xyxy format."""
    if data_sample is None or not hasattr(data_sample, 'gt_instances'):
        return torch.empty((0, 4), dtype=torch.float32, device=device)
    instances = data_sample.gt_instances
    if 'bboxes' not in instances:
        return torch.empty((0, 4), dtype=torch.float32, device=device)
    boxes = instances.bboxes
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    return boxes.to(device=device, dtype=torch.float32).reshape(-1, 4)


def normalize_gt_boxes(gt_boxes: torch.Tensor,
                       coordinate_shape: Sequence[int]) -> torch.Tensor:
    """Convert pixel xyxy GT boxes to normalized xyxy coordinates."""
    if len(coordinate_shape) < 2:
        raise ValueError(f'Invalid coordinate shape: {coordinate_shape}')
    img_h, img_w = int(coordinate_shape[0]), int(coordinate_shape[1])
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f'Invalid coordinate shape: {coordinate_shape}')
    factor = gt_boxes.new_tensor([img_w, img_h, img_w, img_h])
    return gt_boxes / factor


class TopKGTCollector:
    """Collect per-image and per-matched-proposal statistics."""

    def __init__(self, thresholds: Sequence[float]):
        self.thresholds = tuple(sorted(set(float(x) for x in thresholds)))
        if not self.thresholds:
            raise ValueError('At least one IoU threshold is required.')
        if any(x < 0 or x > 1 for x in self.thresholds):
            raise ValueError('IoU thresholds must be in [0, 1].')
        self.rows: List[Dict] = []
        self.matches: List[Dict] = []
        self.pending_batches: List[List[Dict]] = []

    @torch.no_grad()
    def observe(self, model, memory, memory_mask, spatial_shapes, memory_text,
                text_token_mask, batch_data_samples):
        if not hasattr(model, '_gen_competitive_proposals'):
            raise TypeError(
                f'{type(model).__name__} does not expose '
                '_gen_competitive_proposals(); candidate modality cannot be '
                'recovered safely.')

        output_memory, output_proposals = model._gen_competitive_proposals(
            memory, memory_mask, spatial_shapes)
        num_candidates = output_memory.size(1)
        if num_candidates % 2:
            raise RuntimeError(
                f'Expected [VIS, IR] candidates of equal length, got '
                f'{num_candidates}.')
        branch_tokens = num_candidates // 2

        enc_outputs_class = model.bbox_head.cls_branches[
            model.decoder.num_layers](output_memory, memory_text,
                                      text_token_mask)
        enc_outputs_coord_unact = model.bbox_head.reg_branches[
            model.decoder.num_layers](output_memory) + output_proposals
        ranking_scores = enc_outputs_class.max(dim=-1).values
        topk_scores, topk_indices = torch.topk(
            ranking_scores, k=model.num_queries, dim=1)
        topk_coords = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()
        topk_boxes = bbox_cxcywh_to_xyxy(topk_coords)

        if batch_data_samples is None:
            batch_data_samples = [None] * memory.size(0)
        if len(batch_data_samples) != memory.size(0):
            raise RuntimeError('Data-sample and memory batch sizes differ.')

        detail_threshold = min(self.thresholds)
        detail_key = threshold_key(detail_threshold)
        pending_batch = []
        for batch_id, data_sample in enumerate(batch_data_samples):
            indices = topk_indices[batch_id]
            scores = topk_scores[batch_id]
            boxes = topk_boxes[batch_id]
            is_vis = indices < branch_tokens
            vis_count = int(is_vis.sum().item())
            ir_count = int((~is_vis).sum().item())
            if vis_count + ir_count != model.num_queries:
                raise RuntimeError('VIS/IR counts do not sum to num_queries.')

            gt_boxes = get_gt_boxes(data_sample, boxes.device)
            img_shape = (data_sample.get('img_shape', None)
                         if data_sample is not None else None)
            if img_shape is None:
                raise RuntimeError('img_shape is required to normalize GT.')
            # This FLIR test pipeline loads annotations *after* Resize, so its
            # GT boxes remain in original-image pixels. Normalizing by
            # ori_shape maps them into the same [0, 1] frame as model boxes.
            gt_shape = data_sample.get('ori_shape', None)
            if gt_shape is None:
                raise RuntimeError('ori_shape is required to normalize GT.')
            gt_boxes_norm = normalize_gt_boxes(gt_boxes, gt_shape)
            if gt_boxes_norm.numel():
                ious = bbox_overlaps(boxes, gt_boxes_norm)
                max_gt_iou, matched_gt_indices = ious.max(dim=1)
            else:
                max_gt_iou = boxes.new_zeros((model.num_queries, ))
                matched_gt_indices = torch.full(
                    (model.num_queries, ), -1, dtype=torch.long,
                    device=boxes.device)

            img_id = (data_sample.get('img_id', len(self.rows))
                      if data_sample is not None else len(self.rows))
            img_path = (data_sample.get(
                'vis_img_path', data_sample.get('img_path', ''))
                        if data_sample is not None else '')
            sample_idx = (data_sample.get('sample_idx', None)
                          if data_sample is not None else None)
            row = dict(
                sample_idx=sample_idx,
                img_id=img_id,
                img_path=img_path,
                img_height=int(img_shape[0]),
                img_width=int(img_shape[1]),
                num_gt=int(gt_boxes.size(0)),
                num_queries=int(model.num_queries),
                branch_candidates=branch_tokens,
                vis_count=vis_count,
                ir_count=ir_count,
                max_topk_gt_iou=float(max_gt_iou.max().item()),
                mean_topk_gt_iou=float(max_gt_iou.mean().item()))

            for threshold in self.thresholds:
                key = threshold_key(threshold)
                matched = max_gt_iou > threshold
                vis_matched = matched & is_vis
                ir_matched = matched & ~is_vis
                row[f'matched_count_iou_gt_{key}'] = int(matched.sum())
                row[f'vis_matched_count_iou_gt_{key}'] = int(
                    vis_matched.sum())
                row[f'ir_matched_count_iou_gt_{key}'] = int(ir_matched.sum())
                row[f'matched_ratio_iou_gt_{key}'] = float(
                    matched.float().mean())
                row[f'vis_matched_ratio_iou_gt_{key}'] = (
                    int(vis_matched.sum()) / vis_count if vis_count else 0.0)
                row[f'ir_matched_ratio_iou_gt_{key}'] = (
                    int(ir_matched.sum()) / ir_count if ir_count else 0.0)
            self.rows.append(row)
            pending_batch.append(dict(
                row_index=len(self.rows) - 1,
                is_vis=is_vis.detach(),
                candidate_indices=indices.detach(),
                branch_tokens=branch_tokens))

            # Keep one detailed table at the least restrictive requested
            # threshold. Higher thresholds can be filtered using max_gt_iou.
            detail_mask = max_gt_iou > detail_threshold
            for query_rank in torch.nonzero(
                    detail_mask, as_tuple=False).flatten().tolist():
                candidate_index = int(indices[query_rank])
                modality = 'VIS' if candidate_index < branch_tokens else 'IR'
                local_index = (candidate_index if modality == 'VIS' else
                               candidate_index - branch_tokens)
                box_norm = boxes[query_rank]
                img_h, img_w = int(img_shape[0]), int(img_shape[1])
                self.matches.append(dict(
                    sample_idx=sample_idx,
                    img_id=img_id,
                    img_path=img_path,
                    stage='encoder_topk',
                    threshold_table=f'iou_gt_{detail_key}',
                    output_rank=query_rank,
                    query_rank=query_rank,
                    modality=modality,
                    candidate_index=candidate_index,
                    modality_candidate_index=local_index,
                    score=float(scores[query_rank]),
                    matched_gt_index=int(matched_gt_indices[query_rank]),
                    max_gt_iou=float(max_gt_iou[query_rank]),
                    x1_norm=float(box_norm[0]),
                    y1_norm=float(box_norm[1]),
                    x2_norm=float(box_norm[2]),
                    y2_norm=float(box_norm[3]),
                    x1=float(box_norm[0] * img_w),
                    y1=float(box_norm[1] * img_h),
                    x2=float(box_norm[2] * img_w),
                    y2=float(box_norm[3] * img_h)))
        self.pending_batches.append(pending_batch)

    @torch.no_grad()
    def observe_decoder_and_final(self, head, all_layers_cls_scores,
                                  all_layers_bbox_preds,
                                  batch_data_samples):
        """Record the last decoder layer and its normal Top-N postprocess."""
        if not self.pending_batches:
            raise RuntimeError('Decoder output arrived without encoder Top-K.')
        pending_batch = self.pending_batches.pop(0)
        if len(pending_batch) != len(batch_data_samples):
            raise RuntimeError('Pending Top-K and decoder batch sizes differ.')

        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        detail_threshold = min(self.thresholds)
        detail_key = threshold_key(detail_threshold)
        max_per_img = head.test_cfg.get('max_per_img', bbox_preds.size(1))

        for batch_id, data_sample in enumerate(batch_data_samples):
            pending = pending_batch[batch_id]
            row = self.rows[pending['row_index']]
            is_vis = pending['is_vis']
            candidate_indices = pending['candidate_indices']
            branch_tokens = pending['branch_tokens']
            img_shape = data_sample.get('img_shape')
            img_h, img_w = int(img_shape[0]), int(img_shape[1])
            gt_boxes = get_gt_boxes(data_sample, bbox_preds.device)
            gt_shape = data_sample.get('ori_shape', None)
            if gt_shape is None:
                raise RuntimeError('ori_shape is required to normalize GT.')
            gt_boxes_norm = normalize_gt_boxes(gt_boxes, gt_shape)

            decoder_boxes = bbox_cxcywh_to_xyxy(bbox_preds[batch_id])
            if gt_boxes_norm.numel():
                decoder_ious = bbox_overlaps(decoder_boxes, gt_boxes_norm)
                decoder_max_iou, decoder_gt = decoder_ious.max(dim=1)
            else:
                decoder_max_iou = decoder_boxes.new_zeros(
                    (decoder_boxes.size(0), ))
                decoder_gt = torch.full(
                    (decoder_boxes.size(0), ), -1, dtype=torch.long,
                    device=decoder_boxes.device)

            token_positive_map = data_sample.token_positive_map
            cls_score = cls_scores[batch_id]
            if token_positive_map is not None:
                class_scores = convert_grounding_to_cls_scores(
                    logits=cls_score.sigmoid()[None],
                    positive_maps=[token_positive_map])[0]
                final_scores, flat_indices = class_scores.reshape(-1).topk(
                    max_per_img)
                num_classes = class_scores.shape[-1]
                final_query_indices = flat_indices // num_classes
            else:
                query_scores = cls_score.sigmoid().max(dim=-1).values
                final_scores, final_query_indices = query_scores.topk(
                    max_per_img)
            final_boxes = decoder_boxes[final_query_indices].clamp(0, 1)
            final_is_vis = is_vis[final_query_indices]
            if gt_boxes_norm.numel():
                final_ious = bbox_overlaps(final_boxes, gt_boxes_norm)
                final_max_iou, final_gt = final_ious.max(dim=1)
            else:
                final_max_iou = final_boxes.new_zeros((max_per_img, ))
                final_gt = torch.full(
                    (max_per_img, ), -1, dtype=torch.long,
                    device=final_boxes.device)

            row['decoder_count'] = int(decoder_boxes.size(0))
            row['decoder_vis_count'] = int(is_vis.sum())
            row['decoder_ir_count'] = int((~is_vis).sum())
            row['decoder_max_gt_iou'] = float(decoder_max_iou.max())
            row['decoder_mean_gt_iou'] = float(decoder_max_iou.mean())
            row['final_count'] = int(final_boxes.size(0))
            row['final_vis_count'] = int(final_is_vis.sum())
            row['final_ir_count'] = int((~final_is_vis).sum())
            row['final_max_gt_iou'] = float(final_max_iou.max())
            row['final_mean_gt_iou'] = float(final_max_iou.mean())

            stage_values = (
                ('decoder', decoder_boxes, decoder_max_iou, decoder_gt,
                 is_vis, cls_score.sigmoid().max(dim=-1).values,
                 torch.arange(decoder_boxes.size(0), device=boxes_device(
                     decoder_boxes))),
                ('final', final_boxes, final_max_iou, final_gt, final_is_vis,
                 final_scores, final_query_indices))
            for stage, boxes, max_iou, matched_gt, stage_is_vis, stage_scores, \
                    query_indices in stage_values:
                for threshold in self.thresholds:
                    key = threshold_key(threshold)
                    matched = max_iou > threshold
                    vis_matched = matched & stage_is_vis
                    ir_matched = matched & ~stage_is_vis
                    row[f'{stage}_matched_count_iou_gt_{key}'] = int(
                        matched.sum())
                    row[f'{stage}_vis_matched_count_iou_gt_{key}'] = int(
                        vis_matched.sum())
                    row[f'{stage}_ir_matched_count_iou_gt_{key}'] = int(
                        ir_matched.sum())
                detail_mask = max_iou > detail_threshold
                for output_rank in torch.nonzero(
                        detail_mask, as_tuple=False).flatten().tolist():
                    query_rank = int(query_indices[output_rank])
                    candidate_index = int(candidate_indices[query_rank])
                    modality = 'VIS' if bool(stage_is_vis[output_rank]) else 'IR'
                    local_index = (candidate_index if modality == 'VIS' else
                                   candidate_index - branch_tokens)
                    box = boxes[output_rank]
                    self.matches.append(dict(
                        sample_idx=row['sample_idx'], img_id=row['img_id'],
                        img_path=row['img_path'], stage=stage,
                        threshold_table=f'iou_gt_{detail_key}',
                        output_rank=output_rank, query_rank=query_rank,
                        modality=modality, candidate_index=candidate_index,
                        modality_candidate_index=local_index,
                        score=float(stage_scores[output_rank]),
                        matched_gt_index=int(matched_gt[output_rank]),
                        max_gt_iou=float(max_iou[output_rank]),
                        x1_norm=float(box[0]), y1_norm=float(box[1]),
                        x2_norm=float(box[2]), y2_norm=float(box[3]),
                        x1=float(box[0] * img_w), y1=float(box[1] * img_h),
                        x2=float(box[2] * img_w), y2=float(box[3] * img_h)))


def boxes_device(boxes: torch.Tensor) -> torch.device:
    """Small helper kept separate to make arange device intent explicit."""
    return boxes.device


def attach_collector(model, collector: TopKGTCollector):
    original_pre_decoder = model.pre_decoder

    def wrapped_pre_decoder(self, memory, memory_mask, spatial_shapes,
                            memory_text, text_token_mask,
                            batch_data_samples=None):
        collector.observe(
            self, memory, memory_mask, spatial_shapes, memory_text,
            text_token_mask, batch_data_samples)
        return original_pre_decoder(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            batch_data_samples=batch_data_samples)

    model.pre_decoder = MethodType(wrapped_pre_decoder, model)

    def wrapped_predict(head_self, hidden_states, references, memory_text,
                        text_token_mask, batch_data_samples, rescale=True):
        batch_img_metas = [sample.metainfo for sample in batch_data_samples]
        batch_token_positive_maps = [
            sample.token_positive_map for sample in batch_data_samples
        ]
        outs = head_self(hidden_states, references, memory_text,
                         text_token_mask)
        collector.observe_decoder_and_final(
            head_self, outs[0], outs[1], batch_data_samples)
        return head_self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)

    model.bbox_head.predict = MethodType(wrapped_predict, model.bbox_head)


def gather_records(local_records: List[Dict]) -> List[Dict]:
    if not dist.is_available() or not dist.is_initialized():
        return local_records
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_records)
    return [record for rank_records in gathered for record in rank_records]


def record_identity(record: Dict) -> Tuple:
    if record.get('sample_idx') is not None:
        return ('sample_idx', record['sample_idx'])
    return ('image', record.get('img_id'), record.get('img_path'))


def deduplicate_rows(rows: List[Dict]) -> List[Dict]:
    unique, seen = [], set()
    for row in rows:
        identity = record_identity(row)
        if identity not in seen:
            seen.add(identity)
            unique.append(row)
    if all(row.get('sample_idx') is not None for row in unique):
        unique.sort(key=lambda row: row['sample_idx'])
    return unique


def build_summary(rows: List[Dict], thresholds: Sequence[float]) -> Dict:
    if not rows:
        raise RuntimeError('No samples were analyzed.')
    total_queries = sum(row['num_queries'] for row in rows)
    total_vis = sum(row['vis_count'] for row in rows)
    total_ir = sum(row['ir_count'] for row in rows)
    summary = dict(
        num_images=len(rows),
        total_gt=sum(row['num_gt'] for row in rows),
        num_queries=numeric_summary([row['num_queries'] for row in rows]),
        stages={})
    stage_specs = (
        ('encoder_topk', '', 'num_queries', 'vis_count', 'ir_count',
         'max_topk_gt_iou', 'mean_topk_gt_iou', False),
        ('decoder_last', 'decoder_', 'decoder_count', 'decoder_vis_count',
         'decoder_ir_count', 'decoder_max_gt_iou', 'decoder_mean_gt_iou',
         False),
        ('final_topk', 'final_', 'final_count', 'final_vis_count',
         'final_ir_count', 'final_max_gt_iou', 'final_mean_gt_iou', True))
    for (stage_name, prefix, count_field, vis_count_field, ir_count_field,
         max_field, mean_field, clipped) in stage_specs:
        stage_total = sum(row[count_field] for row in rows)
        stage_vis = sum(row[vis_count_field] for row in rows)
        stage_ir = sum(row[ir_count_field] for row in rows)
        stage_summary = dict(
            boxes_clipped_to_image=clipped,
            total_boxes=stage_total,
            count=numeric_summary([row[count_field] for row in rows]),
            vis_count=numeric_summary([row[vis_count_field] for row in rows]),
            ir_count=numeric_summary([row[ir_count_field] for row in rows]),
            max_gt_iou=numeric_summary([row[max_field] for row in rows]),
            mean_gt_iou=numeric_summary([row[mean_field] for row in rows]),
            totals=dict(vis_count=stage_vis, ir_count=stage_ir),
            thresholds={})
        for threshold in thresholds:
            key = threshold_key(threshold)
            total_field = f'{prefix}matched_count_iou_gt_{key}'
            vis_field = f'{prefix}vis_matched_count_iou_gt_{key}'
            ir_field = f'{prefix}ir_matched_count_iou_gt_{key}'
            total_matched = sum(row[total_field] for row in rows)
            vis_matched = sum(row[vis_field] for row in rows)
            ir_matched = sum(row[ir_field] for row in rows)
            stage_summary['thresholds'][str(threshold)] = dict(
                matched_count=numeric_summary(
                    [row[total_field] for row in rows]),
                vis_matched_count=numeric_summary(
                    [row[vis_field] for row in rows]),
                ir_matched_count=numeric_summary(
                    [row[ir_field] for row in rows]),
                totals=dict(
                    matched_count=total_matched,
                    vis_matched_count=vis_matched,
                    ir_matched_count=ir_matched),
                matched_ratio=(total_matched / stage_total
                               if stage_total else 0.0),
                vis_matched_ratio=(vis_matched / stage_vis
                                   if stage_vis else 0.0),
                ir_matched_ratio=(ir_matched / stage_ir
                                  if stage_ir else 0.0),
                matched_modality_share=dict(
                    vis=(vis_matched / total_matched
                         if total_matched else 0.0),
                    ir=(ir_matched / total_matched
                        if total_matched else 0.0)))
        summary['stages'][stage_name] = stage_summary
    return summary


def write_csv(path: Path, records: List[Dict], fieldnames=None):
    with path.open('w', newline='', encoding='utf-8') as file:
        if fieldnames is None and records:
            fieldnames = list(records[0].keys())
        if fieldnames is None:
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_outputs(out_dir: Path, rows: List[Dict], matches: List[Dict],
                  summary: Dict, run_info: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / 'per_image_topk_gt_stats.csv'
    matches_path = out_dir / 'matched_boxes.csv'
    summary_path = out_dir / 'summary.json'
    write_csv(per_image_path, rows)
    write_csv(matches_path, matches, MATCH_FIELDNAMES)
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (out_dir / 'run_info.json').open('w', encoding='utf-8') as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)
    return per_image_path, matches_path, summary_path


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.resume = False
    cfg.work_dir = args.out_dir or osp.join(
        osp.dirname(args.checkpoint), 'topk_gt_iou_analysis')
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.sampler.shuffle = False
    cfg.env_cfg = cfg.get('env_cfg', {})

    if args.launcher == 'pytorch':
        if args.device == 'cpu':
            raise ValueError('Distributed analysis requires CUDA.')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        cfg.device = f'cuda:{local_rank}'
    elif args.device == 'cpu':
        cfg.device = 'cpu'
    elif args.device.startswith('cuda:'):
        torch.cuda.set_device(int(args.device.split(':', 1)[1]))
        cfg.device = args.device
    else:
        raise ValueError('--device must be cpu or cuda:<index>.')

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    model = runner.model.module if is_model_wrapper(runner.model) else runner.model
    model.eval()
    collector = TopKGTCollector(args.iou_thresholds)
    attach_collector(model, collector)

    dataloader = runner.test_dataloader
    dataset_size = len(dataloader.dataset)
    sample_limit = dataset_size
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError('--max-samples must be positive.')
        sample_limit = min(sample_limit, args.max_samples)
    local_limit = ((sample_limit + world_size - 1) // world_size
                   if world_size > 1 else sample_limit)
    progress = ProgressBar(local_limit) if rank == 0 else None

    processed = 0
    with torch.no_grad():
        for data_batch in dataloader:
            runner.model.test_step(data_batch)
            new_processed = len(collector.rows)
            if progress is not None:
                progress.update(new_processed - processed)
            processed = new_processed
            if processed >= local_limit:
                break

    rows = deduplicate_rows(gather_records(collector.rows))[:sample_limit]
    allowed = {record_identity(row) for row in rows}
    matches = [record for record in gather_records(collector.matches)
               if record_identity(record) in allowed]
    if rank != 0:
        dist.destroy_process_group()
        return

    summary = build_summary(rows, collector.thresholds)
    run_info = dict(
        config=osp.abspath(args.config),
        checkpoint=osp.abspath(args.checkpoint),
        device=(f'{world_size} distributed CUDA processes'
                if world_size > 1 else args.device),
        launcher=args.launcher,
        world_size=world_size,
        dataset_size=dataset_size,
        analyzed_samples=len(rows),
        coordinate_system=(
            'model boxes normalized to the valid image; test GT normalized '
            'by ori_shape because this pipeline loads annotations after '
            'Resize and therefore retains original-pixel GT coordinates'),
        matching='each Top-K box matched to its maximum-IoU GT',
        category_aware=False,
        iou_rule='strict_greater_than',
        iou_thresholds=list(collector.thresholds),
        modality_rule='candidate index: [VIS, IR]',
        stage_rules=dict(
            encoder_topk='raw selected encoder reference boxes; not clipped',
            decoder_last='last decoder layer all queries; not clipped',
            final_topk='normal class-score Top-N; clipped to image bounds'),
        final_modality_semantics=(
            'VIS/IR identifies the encoder proposal that initialized the '
            'decoder query; decoder queries attend to both modalities'),
        query_rank_indexing='zero_based')
    paths = write_outputs(
        Path(cfg.work_dir), rows, matches, summary, run_info)

    print('\nTop-K versus GT analysis complete')
    print(f'Images analyzed: {len(rows)}')
    for stage_name, stage_stats in summary['stages'].items():
        print(f'{stage_name}: {stage_stats["total_boxes"]} boxes')
        for threshold, stats in stage_stats['thresholds'].items():
            totals = stats['totals']
            print(f'  IoU > {threshold}: {totals["matched_count"]} boxes '
                  f'(VIS {totals["vis_matched_count"]}, '
                  f'IR {totals["ir_matched_count"]})')
    print(f'Per-image CSV: {paths[0]}')
    print(f'Matched-box CSV: {paths[1]}')
    print(f'Summary JSON: {paths[2]}')
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
