#!/usr/bin/env python3
"""Analyze VIS/IR origins and cross-modal redundancy in competitive Top-K.

This tool is intentionally non-invasive: it wraps ``pre_decoder`` at runtime,
recomputes the encoder proposal selection from its inputs, records statistics,
and then calls the model's original ``pre_decoder`` unchanged.
"""

import argparse
import csv
import json
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.utils import ProgressBar

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze modality origins and cross-modal IoU of the '
        'encoder Top-K selected by competitive multimodal Grounding DINO.')
    parser.add_argument('config', help='Model/test-data config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output directory. Defaults to <checkpoint_dir>/topk_analysis.')
    parser.add_argument(
        '--iou-thresholds',
        type=float,
        nargs='+',
        default=[0.8],
        help='Strict cross-modal IoU thresholds (uses IoU > threshold).')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Analyze only the first N samples; useful for smoke tests.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Inference device, e.g. cuda:0 or cpu.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='Job launcher. Use pytorch with torchrun for multi-GPU analysis.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings, e.g. test_dataloader.batch_size=1.')
    return parser.parse_args()


def threshold_key(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def greedy_unique_pairs(ious: torch.Tensor, threshold: float) -> int:
    """Count one-to-one cross-modal pairs greedily by descending IoU."""
    candidate_indices = torch.nonzero(ious > threshold, as_tuple=False)
    if candidate_indices.numel() == 0:
        return 0
    candidate_ious = ious[candidate_indices[:, 0], candidate_indices[:, 1]]
    order = torch.argsort(candidate_ious, descending=True)
    used_vis = set()
    used_ir = set()
    count = 0
    for pair_id in order.tolist():
        vis_id, ir_id = candidate_indices[pair_id].tolist()
        if vis_id in used_vis or ir_id in used_ir:
            continue
        used_vis.add(vis_id)
        used_ir.add(ir_id)
        count += 1
    return count


class TopKStatisticsCollector:
    """Collect per-image statistics from competitive ``pre_decoder`` inputs."""

    def __init__(self, thresholds: Sequence[float]):
        self.thresholds = tuple(sorted(set(float(x) for x in thresholds)))
        if not self.thresholds:
            raise ValueError('At least one IoU threshold is required.')
        if any(x < 0 or x > 1 for x in self.thresholds):
            raise ValueError('IoU thresholds must be in [0, 1].')
        self.rows: List[Dict] = []

    @torch.no_grad()
    def observe(self, model, memory, memory_mask, spatial_shapes, memory_text,
                text_token_mask, batch_data_samples):
        if not hasattr(model, '_gen_competitive_proposals'):
            raise TypeError(
                f'{type(model).__name__} does not expose '
                '_gen_competitive_proposals(). This analyzer supports '
                'competitive VIS/IR models whose candidate order is '
                '[VIS, IR].')

        output_memory, output_proposals = model._gen_competitive_proposals(
            memory, memory_mask, spatial_shapes)
        num_candidates = output_memory.size(1)
        if num_candidates % 2 != 0:
            raise RuntimeError(
                f'Expected an even [VIS, IR] candidate pool, got '
                f'{num_candidates} candidates.')
        branch_tokens = num_candidates // 2

        enc_outputs_class = model.bbox_head.cls_branches[
            model.decoder.num_layers](output_memory, memory_text,
                                      text_token_mask)
        enc_outputs_coord_unact = model.bbox_head.reg_branches[
            model.decoder.num_layers](output_memory) + output_proposals
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0],
            k=model.num_queries,
            dim=1)[1]
        topk_coords = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4)).sigmoid()

        if batch_data_samples is None:
            batch_data_samples = [None] * memory.size(0)
        if len(batch_data_samples) != memory.size(0):
            raise RuntimeError('batch_data_samples and memory batch sizes differ.')

        for batch_id in range(memory.size(0)):
            indices = topk_indices[batch_id]
            coords = topk_coords[batch_id]
            is_vis = indices < branch_tokens
            is_ir = ~is_vis
            vis_boxes = bbox_cxcywh_to_xyxy(coords[is_vis])
            ir_boxes = bbox_cxcywh_to_xyxy(coords[is_ir])
            vis_count = int(is_vis.sum().item())
            ir_count = int(is_ir.sum().item())
            if vis_count + ir_count != model.num_queries:
                raise RuntimeError('VIS/IR counts do not sum to num_queries.')

            if vis_count and ir_count:
                cross_ious = bbox_overlaps(vis_boxes, ir_boxes)
                best_vis = cross_ious.max(dim=1).values
                best_ir = cross_ious.max(dim=0).values
                max_cross_iou = float(cross_ious.max().item())
                mean_best_vis = float(best_vis.mean().item())
                mean_best_ir = float(best_ir.mean().item())
            else:
                cross_ious = coords.new_zeros((vis_count, ir_count))
                best_vis = coords.new_zeros((vis_count, ))
                best_ir = coords.new_zeros((ir_count, ))
                max_cross_iou = 0.0
                mean_best_vis = 0.0
                mean_best_ir = 0.0

            data_sample = batch_data_samples[batch_id]
            if data_sample is None:
                img_id = len(self.rows)
                img_path = ''
                sample_idx = None
            else:
                img_id = data_sample.get('img_id', len(self.rows))
                img_path = data_sample.get(
                    'vis_img_path', data_sample.get('img_path', ''))
                sample_idx = data_sample.get('sample_idx', None)

            row = dict(
                sample_idx=sample_idx,
                img_id=img_id,
                img_path=img_path,
                num_queries=int(model.num_queries),
                branch_candidates=branch_tokens,
                vis_count=vis_count,
                ir_count=ir_count,
                vis_ratio=vis_count / model.num_queries,
                ir_ratio=ir_count / model.num_queries,
                max_cross_iou=max_cross_iou,
                mean_best_vis_to_ir_iou=mean_best_vis,
                mean_best_ir_to_vis_iou=mean_best_ir)

            for threshold in self.thresholds:
                key = threshold_key(threshold)
                row[f'vis_matched_count_iou_gt_{key}'] = int(
                    (best_vis > threshold).sum().item())
                row[f'ir_matched_count_iou_gt_{key}'] = int(
                    (best_ir > threshold).sum().item())
                row[f'cross_pair_count_iou_gt_{key}'] = int(
                    (cross_ious > threshold).sum().item())
                row[f'greedy_unique_pair_count_iou_gt_{key}'] = \
                    greedy_unique_pairs(cross_ious, threshold)
            self.rows.append(row)


def attach_collector(model, collector: TopKStatisticsCollector):
    """Wrap an instance's pre_decoder without modifying its class source."""
    original_pre_decoder = model.pre_decoder

    def wrapped_pre_decoder(self, memory, memory_mask, spatial_shapes,
                            memory_text, text_token_mask,
                            batch_data_samples=None):
        collector.observe(
            model=self,
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            batch_data_samples=batch_data_samples)
        return original_pre_decoder(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            batch_data_samples=batch_data_samples)

    model.pre_decoder = MethodType(wrapped_pre_decoder, model)


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(array.mean()),
        std=float(array.std()),
        min=float(array.min()),
        median=float(np.median(array)),
        max=float(array.max()))


def build_summary(rows: List[Dict], thresholds: Sequence[float]) -> Dict:
    if not rows:
        raise RuntimeError('No samples were analyzed.')
    summary = dict(
        num_images=len(rows),
        num_queries=numeric_summary([row['num_queries'] for row in rows]),
        vis_count=numeric_summary([row['vis_count'] for row in rows]),
        ir_count=numeric_summary([row['ir_count'] for row in rows]),
        vis_ratio=numeric_summary([row['vis_ratio'] for row in rows]),
        ir_ratio=numeric_summary([row['ir_ratio'] for row in rows]),
        max_cross_iou=numeric_summary(
            [row['max_cross_iou'] for row in rows]),
        mean_best_vis_to_ir_iou=numeric_summary(
            [row['mean_best_vis_to_ir_iou'] for row in rows]),
        mean_best_ir_to_vis_iou=numeric_summary(
            [row['mean_best_ir_to_vis_iou'] for row in rows]),
        thresholds={})

    total_vis = sum(row['vis_count'] for row in rows)
    total_ir = sum(row['ir_count'] for row in rows)
    summary['totals'] = dict(vis_count=total_vis, ir_count=total_ir)
    for threshold in thresholds:
        key = threshold_key(threshold)
        vis_field = f'vis_matched_count_iou_gt_{key}'
        ir_field = f'ir_matched_count_iou_gt_{key}'
        pair_field = f'cross_pair_count_iou_gt_{key}'
        unique_field = f'greedy_unique_pair_count_iou_gt_{key}'
        total_vis_matched = sum(row[vis_field] for row in rows)
        total_ir_matched = sum(row[ir_field] for row in rows)
        total_pairs = sum(row[pair_field] for row in rows)
        total_unique = sum(row[unique_field] for row in rows)
        summary['thresholds'][str(threshold)] = dict(
            vis_matched_count=numeric_summary(
                [row[vis_field] for row in rows]),
            ir_matched_count=numeric_summary(
                [row[ir_field] for row in rows]),
            cross_pair_count=numeric_summary(
                [row[pair_field] for row in rows]),
            greedy_unique_pair_count=numeric_summary(
                [row[unique_field] for row in rows]),
            totals=dict(
                vis_matched_count=total_vis_matched,
                ir_matched_count=total_ir_matched,
                cross_pair_count=total_pairs,
                greedy_unique_pair_count=total_unique),
            vis_matched_ratio=(total_vis_matched / total_vis
                               if total_vis else 0.0),
            ir_matched_ratio=(total_ir_matched / total_ir
                              if total_ir else 0.0),
            mean_removable_fraction_of_topk=float(np.mean([
                row[unique_field] / row['num_queries'] for row in rows
            ])))
    return summary


def write_outputs(out_dir: Path, rows: List[Dict], summary: Dict,
                  run_info: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'per_image_topk_stats.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (out_dir / 'run_info.json').open('w', encoding='utf-8') as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)
    return csv_path, out_dir / 'summary.json'


def gather_rows(local_rows: List[Dict]) -> List[Dict]:
    """Gather per-rank rows, restore dataset order and remove sampler padding."""
    if not dist.is_available() or not dist.is_initialized():
        return local_rows

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_rows)
    rows = [row for rank_rows in gathered for row in rank_rows]

    # DistributedSampler may repeat samples to make all ranks equally long.
    # sample_idx is the most reliable identity; fall back to image metadata.
    unique_rows = []
    seen = set()
    for row in rows:
        if row.get('sample_idx') is not None:
            identity = ('sample_idx', row['sample_idx'])
        else:
            identity = ('image', row.get('img_id'), row.get('img_path'))
        if identity in seen:
            continue
        seen.add(identity)
        unique_rows.append(row)

    if all(row.get('sample_idx') is not None for row in unique_rows):
        unique_rows.sort(key=lambda row: row['sample_idx'])
    return unique_rows


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
        osp.dirname(args.checkpoint), 'topk_analysis')
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.sampler.shuffle = False
    cfg.env_cfg = cfg.get('env_cfg', {})

    if args.launcher == 'pytorch':
        if args.device == 'cpu':
            raise ValueError('Distributed analysis currently requires CUDA.')
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

    collector = TopKStatisticsCollector(args.iou_thresholds)
    attach_collector(model, collector)

    dataloader = runner.test_dataloader
    dataset_size = len(dataloader.dataset)
    sample_limit = dataset_size
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError('--max-samples must be positive.')
        sample_limit = min(sample_limit, args.max_samples)
    # In distributed mode this is rank 0's local progress. The final number of
    # globally analyzed images is printed after gathering all ranks.
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
                if processed > local_limit:
                    collector.rows = collector.rows[:local_limit]
                break

    rows = gather_rows(collector.rows)
    if len(rows) > sample_limit:
        rows = rows[:sample_limit]
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
        iou_rule='strict_greater_than',
        iou_thresholds=list(collector.thresholds),
        unique_pair_method='greedy_descending_iou')
    csv_path, summary_path = write_outputs(
        Path(cfg.work_dir), rows, summary, run_info)

    print('\nTop-K modality analysis complete')
    print(f'Images analyzed: {len(rows)}')
    print(f"Mean VIS count: {summary['vis_count']['mean']:.4f}")
    print(f"Mean IR count: {summary['ir_count']['mean']:.4f}")
    for threshold, stats in summary['thresholds'].items():
        print(f'IoU > {threshold}:')
        print('  Mean VIS boxes with IR match: '
              f"{stats['vis_matched_count']['mean']:.4f}")
        print('  Mean IR boxes with VIS match: '
              f"{stats['ir_matched_count']['mean']:.4f}")
        print('  Mean cross-modal pairs: '
              f"{stats['cross_pair_count']['mean']:.4f}")
        print('  Mean greedy one-to-one pairs: '
              f"{stats['greedy_unique_pair_count']['mean']:.4f}")
    print(f'Per-image CSV: {csv_path}')
    print(f'Summary JSON: {summary_path}')
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
