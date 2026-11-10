#!/usr/bin/env python3
"""Regenerate masks for MFAD samples whose boxes were manually revised."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from generate_detectron_d1 import (
    DEFAULT_CONFIG,
    DEFAULT_WEIGHTS,
    PanopticInstanceCache,
    box_to_mask,
    build_predictor,
    mask_metrics,
    quality_flags,
    read_image,
    select_by_instance_coverage,
)
from generate_m3fd_train_masks import choose_final_mask


DEFAULT_IDS = (5902, 6198, 15778, 15841, 15842)
DEFAULT_ROOT = Path('/home/lbw/mmdetection/data/MFAD')
DEFAULT_ANNOTATIONS = (
    DEFAULT_ROOT / 'annotations/rgbtvg_mfad_train_vg.json')
DEFAULT_OUTPUT = Path('/tmp/mfad_manual_revision_masks')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--sample-ids', type=int, nargs='+', default=DEFAULT_IDS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--coverage-threshold', type=float, default=0.5)
    parser.add_argument('--cache-size', type=int, default=8)
    return parser.parse_args()


def load_selected(path, sample_ids):
    wanted = set(sample_ids)
    selected = {}
    with path.open(encoding='utf-8') as file:
        for sample_id, line in enumerate(file):
            if sample_id in wanted:
                selected[sample_id] = json.loads(line)
    missing = wanted - set(selected)
    if missing:
        raise ValueError(f'Sample IDs missing from annotations: {missing}')
    return selected


def main():
    args = parse_args()
    for path in (args.data_root, args.annotations, args.config, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    sample_ids = sorted(set(args.sample_ids))
    samples = load_selected(args.annotations, sample_ids)
    mask_dir = args.output_dir / 'mask'
    mask_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args)
    cache = PanopticInstanceCache(predictor, args.cache_size)
    records = []
    for sample_id in sample_ids:
        started = time.perf_counter()
        sample = samples[sample_id]
        filename = sample['filename']
        caption = sample['grounding']['caption']
        bbox = sample['grounding']['regions'][0]['bbox']
        vis_path = args.data_root / 'vis' / filename
        ir_path = args.data_root / 'ir' / filename
        vis_image = read_image(vis_path, 'vis')
        ir_image = read_image(ir_path, 'ir')
        if vis_image.shape[:2] != ir_image.shape[:2]:
            raise RuntimeError(f'VIS/IR size mismatch for {filename}')
        height, width = vis_image.shape[:2]
        box_mask, _ = box_to_mask(bbox, height, width)
        vis_candidates = cache.get(f'vis:{vis_path}', vis_image)
        ir_candidates = cache.get(f'ir:{ir_path}', ir_image)
        vis_union, vis_selected = select_by_instance_coverage(
            vis_candidates, box_mask, args.coverage_threshold)
        ir_union, ir_selected = select_by_instance_coverage(
            ir_candidates, box_mask, args.coverage_threshold)
        final_mask, status, fallback = choose_final_mask(
            vis_union, ir_union, len(vis_selected), len(ir_selected), box_mask)
        if not final_mask.any():
            raise RuntimeError(f'Empty final mask for sample {sample_id}')
        metrics = mask_metrics(final_mask, box_mask)
        output_name = f'{sample_id:06d}_{Path(filename).stem}.png'
        output_path = mask_dir / output_name
        if not cv2.imwrite(
                str(output_path), final_mask.astype(np.uint8) * 255):
            raise OSError(f'Failed to write {output_path}')
        record = {
            'sample_id': sample_id,
            'image_id': sample_id,
            'filename': filename,
            'caption': caption,
            'bbox_xyxy': [float(value) for value in bbox],
            'mask_path': f'mask/{output_name}',
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
            'elapsed_ms': (time.perf_counter() - started) * 1000.0,
        }
        records.append(record)
        print(
            f'sample={sample_id} status={status} '
            f'vis_selected={len(vis_selected)} '
            f'ir_selected={len(ir_selected)}', flush=True)

    index_path = args.output_dir / 'records.json'
    index_path.write_text(
        json.dumps({'records': records}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps({
        'output_dir': str(args.output_dir),
        'record_count': len(records),
        'sample_ids': sample_ids,
    }, indent=2))


if __name__ == '__main__':
    main()
