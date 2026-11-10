#!/usr/bin/env python3
"""Four-stage GT-IoU analysis for balanced-selector multimodal models.

Stages:
1. Encoder per-modality preselection: VIS 600 + IR 600.
2. Balanced selector output: 900 queries.
3. Last decoder layer: 900 queries, retaining initialization modality.
4. Normal postprocessed Top-300, retaining initialization modality.
"""

import argparse
import os
import os.path as osp
from pathlib import Path

import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.utils import ProgressBar

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.utils import setup_cache_size_limit_of_dynamo

from analyze_multimodal_topk_gt_iou import (
    TopKGTCollector, attach_collector, deduplicate_rows, gather_records,
    get_gt_boxes, numeric_summary, record_identity, threshold_key,
    write_outputs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze balanced-selector pre-1200, selected-900, '
        'decoder-900 and final Top-300 boxes against GT by modality.')
    parser.add_argument('config', help='Model/test-data config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument(
        '--iou-thresholds', type=float, nargs='+', default=[0.8])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def normalize_original_gt(data_sample, device):
    gt_boxes = get_gt_boxes(data_sample, device)
    ori_shape = data_sample.get('ori_shape', None)
    if ori_shape is None or len(ori_shape) < 2:
        raise RuntimeError('ori_shape is required to normalize FLIR test GT.')
    ori_h, ori_w = int(ori_shape[0]), int(ori_shape[1])
    factor = gt_boxes.new_tensor([ori_w, ori_h, ori_w, ori_h])
    return gt_boxes, gt_boxes / factor


class BalancedSelectorGTCollector(TopKGTCollector):
    """Collect balanced preselection and selection before base decoder hook."""

    @torch.no_grad()
    def observe(self, model, memory, memory_mask, spatial_shapes, memory_text,
                text_token_mask, batch_data_samples):
        if not hasattr(model, '_preselect_branch') or not hasattr(
                model, 'balanced_selector'):
            raise TypeError('This analyzer requires a balanced-selector model.')

        output_memory, output_proposals = model._gen_competitive_proposals(
            memory, memory_mask, spatial_shapes)
        if output_memory.size(1) % 2 or memory_text.size(1) % 2:
            raise RuntimeError('VIS/IR branches must have equal lengths.')
        branch_tokens = output_memory.size(1) // 2
        vis_memory, ir_memory = output_memory.split(branch_tokens, dim=1)
        vis_proposals, ir_proposals = output_proposals.split(
            branch_tokens, dim=1)
        text_length = memory_text.size(1) // 2
        vis_text, ir_text = memory_text.split(text_length, dim=1)

        vis = model._preselect_branch(
            vis_memory, vis_proposals, vis_text, text_token_mask)
        ir = model._preselect_branch(
            ir_memory, ir_proposals, ir_text, text_token_mask)
        selected = model.balanced_selector(
            vis_scores=vis['scores'],
            vis_coords_unact=vis['coords_unact'],
            vis_features=vis['features'],
            vis_source_indices=vis['source_indices'],
            ir_scores=ir['scores'],
            ir_coords_unact=ir['coords_unact'],
            ir_features=ir['features'],
            ir_source_indices=ir['source_indices'])

        pre_topk = model.balanced_selector.pre_topk_per_modality
        if pre_topk != 600 or model.num_queries != 900:
            raise RuntimeError(
                f'Expected 600 candidates per modality and 900 selected, got '
                f'{pre_topk} and {model.num_queries}.')
        pre_boxes = bbox_cxcywh_to_xyxy(torch.cat([
            vis['coords_unact'], ir['coords_unact']], dim=1).sigmoid())
        pre_scores = torch.cat([
            vis['scores'].max(dim=-1).values,
            ir['scores'].max(dim=-1).values], dim=1)
        pre_is_vis = torch.cat([
            torch.ones((memory.size(0), pre_topk), dtype=torch.bool,
                       device=memory.device),
            torch.zeros((memory.size(0), pre_topk), dtype=torch.bool,
                        device=memory.device)], dim=1)
        pre_source = torch.cat([
            vis['source_indices'], ir['source_indices'] + branch_tokens],
            dim=1)

        selected_boxes = bbox_cxcywh_to_xyxy(
            selected['coords_unact'].sigmoid())
        selected_scores = selected['scores'].max(dim=-1).values
        selected_is_vis = selected['modality_ids'] == 0
        selected_global_source = selected['source_indices'] + (
            selected['modality_ids'] * branch_tokens)

        if batch_data_samples is None:
            raise RuntimeError('GT analysis requires batch_data_samples.')
        pending_batch = []
        detail_threshold = min(self.thresholds)
        detail_key = threshold_key(detail_threshold)
        for batch_id, data_sample in enumerate(batch_data_samples):
            gt_boxes, gt_norm = normalize_original_gt(
                data_sample, selected_boxes.device)
            img_shape = data_sample.get('img_shape')
            img_h, img_w = int(img_shape[0]), int(img_shape[1])

            stage_inputs = (
                ('encoder_pre600', pre_boxes[batch_id],
                 pre_scores[batch_id], pre_is_vis[batch_id],
                 pre_source[batch_id]),
                ('encoder_selected900', selected_boxes[batch_id],
                 selected_scores[batch_id], selected_is_vis[batch_id],
                 selected_global_source[batch_id]))
            stage_results = {}
            for stage, boxes, scores, is_vis, source_indices in stage_inputs:
                if gt_norm.numel():
                    ious = bbox_overlaps(boxes, gt_norm)
                    max_iou, matched_gt = ious.max(dim=1)
                else:
                    max_iou = boxes.new_zeros((boxes.size(0), ))
                    matched_gt = torch.full(
                        (boxes.size(0), ), -1, dtype=torch.long,
                        device=boxes.device)
                stage_results[stage] = (max_iou, matched_gt)

            sample_idx = data_sample.get('sample_idx', None)
            img_id = data_sample.get('img_id', len(self.rows))
            img_path = data_sample.get(
                'vis_img_path', data_sample.get('img_path', ''))
            pre_max_iou, _ = stage_results['encoder_pre600']
            selected_max_iou, _ = stage_results['encoder_selected900']
            selected_vis_count = int(selected_is_vis[batch_id].sum())
            selected_ir_count = model.num_queries - selected_vis_count
            row = dict(
                sample_idx=sample_idx,
                img_id=img_id,
                img_path=img_path,
                img_height=img_h,
                img_width=img_w,
                num_gt=int(gt_boxes.size(0)),
                pre_count=2 * pre_topk,
                pre_vis_count=pre_topk,
                pre_ir_count=pre_topk,
                pre_max_gt_iou=float(pre_max_iou.max()),
                pre_mean_gt_iou=float(pre_max_iou.mean()),
                num_queries=int(model.num_queries),
                branch_candidates=branch_tokens,
                vis_count=selected_vis_count,
                ir_count=selected_ir_count,
                max_topk_gt_iou=float(selected_max_iou.max()),
                mean_topk_gt_iou=float(selected_max_iou.mean()))

            for stage, prefix in (
                    ('encoder_pre600', 'pre_'),
                    ('encoder_selected900', '')):
                boxes, scores, is_vis, source_indices = next(
                    values[1:] for values in stage_inputs
                    if values[0] == stage)
                max_iou, matched_gt = stage_results[stage]
                for threshold in self.thresholds:
                    key = threshold_key(threshold)
                    matched = max_iou > threshold
                    row[f'{prefix}matched_count_iou_gt_{key}'] = int(
                        matched.sum())
                    row[f'{prefix}vis_matched_count_iou_gt_{key}'] = int(
                        (matched & is_vis).sum())
                    row[f'{prefix}ir_matched_count_iou_gt_{key}'] = int(
                        (matched & ~is_vis).sum())
                detail_mask = max_iou > detail_threshold
                for output_rank in torch.nonzero(
                        detail_mask, as_tuple=False).flatten().tolist():
                    global_source = int(source_indices[output_rank])
                    modality = 'VIS' if bool(is_vis[output_rank]) else 'IR'
                    local_source = (global_source if modality == 'VIS' else
                                    global_source - branch_tokens)
                    box = boxes[output_rank]
                    self.matches.append(dict(
                        sample_idx=sample_idx, img_id=img_id,
                        img_path=img_path, stage=stage,
                        threshold_table=f'iou_gt_{detail_key}',
                        output_rank=output_rank, query_rank=output_rank,
                        modality=modality,
                        candidate_index=global_source,
                        modality_candidate_index=local_source,
                        score=float(scores[output_rank]),
                        matched_gt_index=int(matched_gt[output_rank]),
                        max_gt_iou=float(max_iou[output_rank]),
                        x1_norm=float(box[0]), y1_norm=float(box[1]),
                        x2_norm=float(box[2]), y2_norm=float(box[3]),
                        x1=float(box[0] * img_w), y1=float(box[1] * img_h),
                        x2=float(box[2] * img_w), y2=float(box[3] * img_h)))

            self.rows.append(row)
            pending_batch.append(dict(
                row_index=len(self.rows) - 1,
                is_vis=selected_is_vis[batch_id].detach(),
                candidate_indices=selected_global_source[batch_id].detach(),
                branch_tokens=branch_tokens))
        self.pending_batches.append(pending_batch)


def summarize_stage(rows, thresholds, prefix, count_field, vis_field,
                    ir_field, max_field, mean_field, clipped):
    total = sum(row[count_field] for row in rows)
    total_vis = sum(row[vis_field] for row in rows)
    total_ir = sum(row[ir_field] for row in rows)
    result = dict(
        boxes_clipped_to_image=clipped,
        total_boxes=total,
        count=numeric_summary([row[count_field] for row in rows]),
        vis_count=numeric_summary([row[vis_field] for row in rows]),
        ir_count=numeric_summary([row[ir_field] for row in rows]),
        max_gt_iou=numeric_summary([row[max_field] for row in rows]),
        mean_gt_iou=numeric_summary([row[mean_field] for row in rows]),
        totals=dict(vis_count=total_vis, ir_count=total_ir),
        thresholds={})
    for threshold in thresholds:
        key = threshold_key(threshold)
        fields = (
            f'{prefix}matched_count_iou_gt_{key}',
            f'{prefix}vis_matched_count_iou_gt_{key}',
            f'{prefix}ir_matched_count_iou_gt_{key}')
        matched = sum(row[fields[0]] for row in rows)
        vis_matched = sum(row[fields[1]] for row in rows)
        ir_matched = sum(row[fields[2]] for row in rows)
        result['thresholds'][str(threshold)] = dict(
            matched_count=numeric_summary([row[fields[0]] for row in rows]),
            vis_matched_count=numeric_summary(
                [row[fields[1]] for row in rows]),
            ir_matched_count=numeric_summary(
                [row[fields[2]] for row in rows]),
            totals=dict(matched_count=matched,
                        vis_matched_count=vis_matched,
                        ir_matched_count=ir_matched),
            matched_ratio=matched / total if total else 0.0,
            vis_matched_ratio=(vis_matched / total_vis if total_vis else 0.0),
            ir_matched_ratio=(ir_matched / total_ir if total_ir else 0.0),
            matched_modality_share=dict(
                vis=vis_matched / matched if matched else 0.0,
                ir=ir_matched / matched if matched else 0.0))
    return result


def build_summary(rows, thresholds):
    if not rows:
        raise RuntimeError('No samples were analyzed.')
    specs = (
        ('encoder_pre600_each', 'pre_', 'pre_count', 'pre_vis_count',
         'pre_ir_count', 'pre_max_gt_iou', 'pre_mean_gt_iou', False),
        ('encoder_selected900', '', 'num_queries', 'vis_count', 'ir_count',
         'max_topk_gt_iou', 'mean_topk_gt_iou', False),
        ('decoder_last900', 'decoder_', 'decoder_count', 'decoder_vis_count',
         'decoder_ir_count', 'decoder_max_gt_iou', 'decoder_mean_gt_iou',
         False),
        ('final_top300', 'final_', 'final_count', 'final_vis_count',
         'final_ir_count', 'final_max_gt_iou', 'final_mean_gt_iou', True))
    return dict(
        num_images=len(rows),
        total_gt=sum(row['num_gt'] for row in rows),
        stages={name: summarize_stage(
            rows, thresholds, prefix, count, vis, ir, max_iou, mean_iou,
            clipped)
                for name, prefix, count, vis, ir, max_iou, mean_iou, clipped
                in specs})


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
        osp.dirname(args.checkpoint), 'balanced_topk_gt_iou_analysis')
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
    collector = BalancedSelectorGTCollector(args.iou_thresholds)
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
        config=osp.abspath(args.config), checkpoint=osp.abspath(args.checkpoint),
        device=(f'{world_size} distributed CUDA processes'
                if world_size > 1 else args.device),
        dataset_size=dataset_size, analyzed_samples=len(rows),
        iou_rule='strict_greater_than',
        iou_thresholds=list(collector.thresholds), category_aware=False,
        gt_coordinate_space='original pixels normalized by ori_shape',
        stages=dict(
            encoder_pre600_each='VIS 600 + IR 600 before balanced selection',
            encoder_selected900='C++ exact balanced selector output',
            decoder_last900='last decoder layer, raw and unclipped',
            final_top300='normal score selection and image-bound clipping'),
        modality_semantics=(
            'Decoder/final VIS or IR means the modality of the encoder '
            'proposal that initialized that query.'))
    paths = write_outputs(
        Path(cfg.work_dir), rows, matches, summary, run_info)
    print('\nBalanced-selector four-stage GT-IoU analysis complete')
    print(f'Images analyzed: {len(rows)}')
    for name, stats in summary['stages'].items():
        print(f'{name}: {stats["total_boxes"]} boxes')
        for threshold, values in stats['thresholds'].items():
            totals = values['totals']
            print(f'  IoU > {threshold}: {totals["matched_count"]} '
                  f'(VIS {totals["vis_matched_count"]}, '
                  f'IR {totals["ir_matched_count"]})')
    print(f'Per-image CSV: {paths[0]}')
    print(f'Matched-box CSV: {paths[1]}')
    print(f'Summary JSON: {paths[2]}')
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
