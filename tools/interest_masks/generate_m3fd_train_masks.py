#!/usr/bin/env python3
"""Generate the final M3FD training interest masks.

For each referring-expression training sample, Detectron2 Panoptic FPN is
run on VIS and three-channel IR images. All ``thing`` instances whose
intersection / instance area is strictly greater than the coverage threshold
are retained and merged across modalities. If neither modality has a qualified
instance, the complete GT box becomes the positive mask.

Outputs are deliberately method-agnostic:
  M3FD/mask/*.png
  M3FD/annotations/rgbtvg_m3fd_train_seg.jsonl
  M3FD/annotations/rgbtvg_m3fd_train_seg_statistics.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from generate_detectron_d1 import (
    DEFAULT_CONFIG,
    DEFAULT_WEIGHTS,
    PanopticInstanceCache,
    box_to_mask,
    build_predictor,
    iter_samples,
    mask_metrics,
    quality_flags,
    read_image,
    save_visualization,
    select_by_instance_coverage,
    write_json,
)


DEFAULT_DATA_ROOT = Path('/home/lbw/mmdetection/data/M3FD')
DEFAULT_ANNOTATIONS = (
    DEFAULT_DATA_ROOT / 'annotations/rgbtvg_m3fd_train_vg.json')
DEFAULT_MASK_DIR = DEFAULT_DATA_ROOT / 'mask'
DEFAULT_SEG_INDEX = (
    DEFAULT_DATA_ROOT / 'annotations/rgbtvg_m3fd_train_seg.jsonl')
DEFAULT_STATISTICS = (
    DEFAULT_DATA_ROOT /
    'annotations/rgbtvg_m3fd_train_seg_statistics.json')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate final M3FD training masks with dual-modal '
                    'Detectron2 instance filtering.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--mask-dir', type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument('--seg-index', type=Path, default=DEFAULT_SEG_INDEX)
    parser.add_argument('--statistics', type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--coverage-threshold', type=float, default=0.5)
    parser.add_argument('--cache-size', type=int, default=4)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--visualization-dir', type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, label in ((args.data_root, 'data root'),
                        (args.annotations, 'annotation file'),
                        (args.config, 'Detectron2 config'),
                        (args.weights, 'Detectron2 weights')):
        if not path.exists():
            raise FileNotFoundError(f'{label} does not exist: {path}')
    if args.start < 0:
        raise ValueError('--start must be non-negative.')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive when provided.')
    if args.cache_size <= 0:
        raise ValueError('--cache-size must be positive.')
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError('--confidence-threshold must be in [0, 1].')
    if not 0.0 <= args.coverage_threshold <= 1.0:
        raise ValueError('--coverage-threshold must be in [0, 1].')
    if args.visualize and args.visualization_dir is None:
        args.visualization_dir = args.data_root / 'mask_visualizations'


def relative_mask_path(mask_path: Path, data_root: Path) -> str:
    try:
        return mask_path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return str(mask_path.resolve())


def choose_final_mask(vis_mask: np.ndarray, ir_mask: np.ndarray,
                      vis_count: int, ir_count: int,
                      box_mask: np.ndarray):
    if vis_count and ir_count:
        return np.logical_or(vis_mask, ir_mask), 'dual_success', False
    if vis_count:
        return vis_mask, 'vis_only', False
    if ir_count:
        return ir_mask, 'ir_only', False
    return box_mask.copy(), 'bbox_fallback', True


def make_statistics(records: List[Dict], elapsed_seconds: float,
                    args: argparse.Namespace) -> Dict:
    statuses = Counter(record['status'] for record in records)
    numeric_fields = (
        'candidate_count', 'selected_count', 'mask_area', 'box_area',
        'mask_box_area_ratio', 'instance_coverage', 'box_coverage', 'iou',
        'elapsed_ms')
    result: Dict[str, object] = {
        'total_samples': len(records),
        'dual_success_count': statuses['dual_success'],
        'vis_only_count': statuses['vis_only'],
        'ir_only_count': statuses['ir_only'],
        'bbox_fallback_count': statuses['bbox_fallback'],
        'bbox_fallback_rate': (
            statuses['bbox_fallback'] / len(records) if records else 0.0),
        'empty_mask_count': sum(record['mask_area'] == 0 for record in records),
        'quality_flag_counts': dict(Counter(
            flag for record in records for flag in record['quality_flags'])),
        'total_elapsed_seconds': elapsed_seconds,
        'annotation_path': str(args.annotations),
        'mask_dir': str(args.mask_dir),
        'seg_index': str(args.seg_index),
        'confidence_threshold': args.confidence_threshold,
        'coverage_threshold_strictly_greater_than': args.coverage_threshold,
        'start': args.start,
        'limit': args.limit,
    }
    for key in numeric_fields:
        values = [float(record[key]) for record in records]
        result[f'average_{key}'] = float(np.mean(values)) if values else None
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    args.seg_index.parent.mkdir(parents=True, exist_ok=True)
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    if args.visualization_dir is not None:
        args.visualization_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args)
    cache = PanopticInstanceCache(predictor, args.cache_size)
    records: List[Dict] = []
    started = time.perf_counter()

    for processed_index, (sample_id, sample) in enumerate(iter_samples(
            args.annotations, args.start, args.limit)):
        item_started = time.perf_counter()
        filename = sample['filename']
        caption = sample['grounding']['caption']
        bbox = sample['grounding']['regions'][0]['bbox']
        vis_path = args.data_root / 'vis' / filename
        ir_path = args.data_root / 'ir' / filename
        vis_image = read_image(vis_path, 'vis')
        ir_image = read_image(ir_path, 'ir')
        if vis_image.shape[:2] != ir_image.shape[:2]:
            raise RuntimeError(
                f'VIS/IR shape mismatch for {filename}: '
                f'{vis_image.shape[:2]} vs {ir_image.shape[:2]}')
        height, width = vis_image.shape[:2]
        box_mask, clipped_box = box_to_mask(bbox, height, width)

        vis_candidates = cache.get(f'vis:{vis_path}', vis_image)
        ir_candidates = cache.get(f'ir:{ir_path}', ir_image)
        vis_union, vis_selected = select_by_instance_coverage(
            vis_candidates, box_mask, args.coverage_threshold)
        ir_union, ir_selected = select_by_instance_coverage(
            ir_candidates, box_mask, args.coverage_threshold)
        final_mask, status, fallback = choose_final_mask(
            vis_union, ir_union, len(vis_selected), len(ir_selected), box_mask)
        metrics = mask_metrics(final_mask, box_mask)
        if not final_mask.any():
            raise RuntimeError(f'Unexpected empty final mask at sample {sample_id}.')

        output_name = f'{sample_id:06d}_{Path(filename).stem}.png'
        mask_path = args.mask_dir / output_name
        if mask_path.exists() and not args.overwrite:
            raise FileExistsError(
                f'Mask already exists: {mask_path}. Use --overwrite.')
        if not cv2.imwrite(str(mask_path), final_mask.astype(np.uint8) * 255):
            raise OSError(f'Failed to save mask: {mask_path}')

        record = {
            'sample_id': sample_id,
            'image_id': sample_id,
            'filename': filename,
            'caption': caption,
            'bbox_xyxy': [float(value) for value in bbox],
            'mask_path': relative_mask_path(mask_path, args.data_root),
            'status': status,
            'fallback': fallback,
            'candidate_count': len(vis_candidates) + len(ir_candidates),
            'vis_candidate_count': len(vis_candidates),
            'ir_candidate_count': len(ir_candidates),
            'selected_count': len(vis_selected) + len(ir_selected),
            'vis_selected_count': len(vis_selected),
            'ir_selected_count': len(ir_selected),
            **metrics,
            'quality_flags': quality_flags(metrics, fallback),
            'elapsed_ms': (time.perf_counter() - item_started) * 1000.0,
        }
        records.append(record)

        if args.visualize and processed_index < args.visualize:
            assert args.visualization_dir is not None
            save_visualization(
                vis_image, final_mask, clipped_box, caption,
                args.visualization_dir / output_name, status)
        print(
            f'[{processed_index + 1}] sample={sample_id} file={filename} '
            f'vis_selected={len(vis_selected)} '
            f'ir_selected={len(ir_selected)} status={status}',
            flush=True)

    temporary_index = args.seg_index.with_suffix(args.seg_index.suffix + '.tmp')
    with temporary_index.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary_index.replace(args.seg_index)

    statistics = make_statistics(
        records, time.perf_counter() - started, args)
    write_json(args.statistics, statistics)
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
