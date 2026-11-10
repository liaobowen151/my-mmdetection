#!/usr/bin/env python3
"""Regenerate FLIR masks for reviewed training samples with revised boxes."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from generate_detectron_d1 import (
    DEFAULT_CONFIG, DEFAULT_WEIGHTS, PanopticInstanceCache, box_to_mask,
    build_predictor, mask_metrics, quality_flags, read_image,
    select_by_instance_coverage)
from generate_m3fd_train_masks import choose_final_mask


ROOT = Path('/home/lbw/mmdetection')
DATA_ROOT = ROOT / 'data/FLIR'
ANNOTATIONS = DATA_ROOT / 'annotations/rgbtvg_flir_train_vg.json'
REVISIONS = ROOT / 'tools/analysis_tools/flir_test_text_anomaly_revisions_preview.json'
OUTPUT = Path('/tmp/flir_text_revision_masks')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    parser.add_argument('--revision-file', type=Path, default=REVISIONS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--coverage-threshold', type=float, default=0.5)
    parser.add_argument('--cache-size', type=int, default=16)
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    args = parse_args()
    for path in (args.data_root, args.annotations, args.revision_file,
                 args.config, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)
    annotations = load_jsonl(args.annotations)
    revisions = json.loads(args.revision_file.read_text(
        encoding='utf-8'))['revisions']
    selected = {
        int(key.split(':', 1)[1]): revision
        for key, revision in revisions.items()
        if key.startswith('train:') and 'bbox_xyxy' in revision
    }
    if not selected:
        raise ValueError('No revised FLIR training boxes found.')
    mask_dir = args.output_dir / 'mask'
    mask_dir.mkdir(parents=True, exist_ok=True)
    predictor = build_predictor(args)
    cache = PanopticInstanceCache(predictor, args.cache_size)
    records = []
    for sample_id in sorted(selected):
        started = time.perf_counter()
        sample = annotations[sample_id]
        revision = selected[sample_id]
        filename = sample['filename']
        caption = revision['proposed_text']
        bbox = [float(value) for value in revision['bbox_xyxy']]
        vis_image = read_image(args.data_root / 'vis' / filename, 'vis')
        ir_image = read_image(args.data_root / 'ir' / filename, 'ir')
        if vis_image.shape[:2] != ir_image.shape[:2]:
            raise RuntimeError(f'VIS/IR size mismatch for {filename}')
        height, width = vis_image.shape[:2]
        box_mask, _ = box_to_mask(bbox, height, width)
        vis_candidates = cache.get(f'vis:{filename}', vis_image)
        ir_candidates = cache.get(f'ir:{filename}', ir_image)
        vis_union, vis_selected = select_by_instance_coverage(
            vis_candidates, box_mask, args.coverage_threshold)
        ir_union, ir_selected = select_by_instance_coverage(
            ir_candidates, box_mask, args.coverage_threshold)
        final_mask, status, fallback = choose_final_mask(
            vis_union, ir_union, len(vis_selected), len(ir_selected), box_mask)
        if not final_mask.any():
            raise RuntimeError(f'Empty final mask for train:{sample_id}')
        metrics = mask_metrics(final_mask, box_mask)
        output_name = f'{sample_id:06d}_{Path(filename).stem}.png'
        output_path = mask_dir / output_name
        if not cv2.imwrite(str(output_path),
                           final_mask.astype(np.uint8) * 255):
            raise OSError(f'Failed to write {output_path}')
        record = {
            'sample_id': sample_id,
            'image_id': sample_id,
            'filename': filename,
            'caption': caption,
            'bbox_xyxy': bbox,
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
        print(f'train:{sample_id} {status}', flush=True)
    (args.output_dir / 'records.json').write_text(json.dumps(
        {'records': records}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps({'record_count': len(records),
                      'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
