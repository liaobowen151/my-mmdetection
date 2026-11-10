#!/usr/bin/env python3
"""Generate D1-V, D1-I, D2 and D3 masks with Detectron2 Panoptic FPN.

D1-V runs Panoptic FPN on the visible image, while D1-I runs it on the
infrared image converted to three channels.  For every referring-expression
sample, the COCO ``thing`` instance with the highest box-matching score is
selected.  If no valid overlapping instance exists, the target box mask is
used as a recorded fallback.

D2 is composed from the generated D1-V and D1-I masks.  Successful masks are
combined with a hard union; if only one modality succeeds, that modality is
used; if both fail, the target box mask is used.

D3 reuses all per-modality ``thing`` candidates produced during D1 inference.
Every candidate whose instance coverage by the GT box is greater than 0.5 is
retained, and retained VIS/IR masks are combined with a hard union.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


DEFAULT_DATA_ROOT = Path('/home/lbw/mmdetection/data/M3FD')
DEFAULT_ANNOTATIONS = DEFAULT_DATA_ROOT / 'annotations/rgbtvg_m3fd_train_vg.json'
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / 'interest_masks'
DEFAULT_CONFIG = Path(
    '/home/lbw/repos/detectron2/configs/COCO-PanopticSegmentation/'
    'panoptic_fpn_R_101_3x.yaml')
DEFAULT_WEIGHTS = Path(
    '/data3/lbw/models/detectron2/model_final_cafdb1.pkl')

METHODS = {
    'd1v': {
        'name': 'd1_detectron_vis_best',
        'modality': 'vis',
    },
    'd1i': {
        'name': 'd1_detectron_ir_best',
        'modality': 'ir',
    },
    'd2': {
        'name': 'd2_detectron_dual_best_union',
        'modality': 'dual',
    },
    'd3': {
        'name': 'd3_detectron_dual_all_union',
        'modality': 'dual',
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate Detectron2 D1-V/D1-I/D2/D3 interest masks.')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        '--methods', nargs='+', choices=tuple(METHODS), default=['d1v', 'd1i'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Number of annotation rows to process. Omit for the full set.')
    parser.add_argument(
        '--visualize', type=int, default=0,
        help='Save visualizations for the first N processed samples per method.')
    parser.add_argument('--cache-size', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
            (args.data_root, 'data root'),
            (args.annotations, 'annotations'),
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


def iter_samples(path: Path, start: int,
                 limit: Optional[int]) -> Iterable[Tuple[int, Dict]]:
    stop = None if limit is None else start + limit
    with path.open('r', encoding='utf-8') as handle:
        for sample_id, line in enumerate(handle):
            if sample_id < start:
                continue
            if stop is not None and sample_id >= stop:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            regions = sample.get('grounding', {}).get('regions', [])
            if len(regions) != 1:
                raise ValueError(
                    f'Sample {sample_id} has {len(regions)} regions; expected 1.')
            yield sample_id, sample


def build_predictor(args: argparse.Namespace) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(str(args.config))
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    threshold = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
    return DefaultPredictor(cfg)


def read_image(path: Path, modality: str) -> np.ndarray:
    if modality == 'vis':
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    else:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image = None if gray is None else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if image is None:
        raise FileNotFoundError(f'Unable to read image: {path}')
    return image


def box_to_mask(box: Sequence[float], height: int,
                width: int) -> Tuple[np.ndarray, List[int]]:
    if len(box) != 4:
        raise ValueError(f'Expected xyxy box with 4 values, got {box}.')
    x1 = int(np.floor(box[0]))
    y1 = int(np.floor(box[1]))
    x2 = int(np.ceil(box[2]))
    y2 = int(np.ceil(box[3]))
    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid clipped box: {[x1, y1, x2, y2]}.')
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask, [x1, y1, x2, y2]


def mask_metrics(mask: np.ndarray, box_mask: np.ndarray) -> Dict[str, float]:
    mask = mask.astype(bool, copy=False)
    box_mask = box_mask.astype(bool, copy=False)
    intersection = int(np.logical_and(mask, box_mask).sum())
    instance_area = int(mask.sum())
    box_area = int(box_mask.sum())
    union = int(np.logical_or(mask, box_mask).sum())
    instance_coverage = intersection / instance_area if instance_area else 0.0
    box_coverage = intersection / box_area if box_area else 0.0
    iou = intersection / union if union else 0.0
    return {
        'intersection': intersection,
        'mask_area': instance_area,
        'box_area': box_area,
        'instance_coverage': instance_coverage,
        'box_coverage': box_coverage,
        'iou': iou,
        'mask_box_area_ratio': instance_area / box_area if box_area else 0.0,
    }


def quality_flags(metrics: Dict[str, float], fallback: bool) -> List[str]:
    if fallback:
        return []
    flags = []
    area_ratio = metrics['mask_box_area_ratio']
    if area_ratio < 0.05:
        flags.append('mask_too_small')
    if area_ratio > 3.0:
        flags.append('mask_too_large')
    if metrics['box_coverage'] < 0.1:
        flags.append('low_box_coverage')
    return flags


class PanopticInstanceCache:
    """Small LRU cache because adjacent expressions often share an image."""

    def __init__(self, predictor: DefaultPredictor, max_size: int) -> None:
        self.predictor = predictor
        self.max_size = max_size
        self.cache: OrderedDict[str, List[Dict]] = OrderedDict()

    def get(self, key: str, image: np.ndarray) -> List[Dict]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with torch.inference_mode():
            predictions = self.predictor(image)
        panoptic_seg, segments_info = predictions['panoptic_seg']
        panoptic = panoptic_seg.detach().cpu().numpy()
        candidates: List[Dict] = []
        for segment in segments_info:
            if not segment.get('isthing', False):
                continue
            mask = panoptic == int(segment['id'])
            if not mask.any():
                continue
            candidates.append({
                'mask': mask,
                'segment_id': int(segment['id']),
                'category_id': int(segment['category_id']),
                'confidence': float(segment.get('score', 0.0)),
            })
        self.cache[key] = candidates
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return candidates


def select_best(candidates: Sequence[Dict], box_mask: np.ndarray
                ) -> Tuple[Optional[Dict], Optional[Dict[str, float]]]:
    best_candidate = None
    best_metrics = None
    best_rank = None
    for candidate in candidates:
        metrics = mask_metrics(candidate['mask'], box_mask)
        match_score = 0.5 * metrics['iou'] + 0.5 * metrics['instance_coverage']
        rank = (match_score, metrics['iou'], candidate['confidence'])
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_candidate = candidate
            best_metrics = metrics
            best_metrics['match_score'] = match_score
    return best_candidate, best_metrics


def select_by_instance_coverage(
        candidates: Sequence[Dict], box_mask: np.ndarray,
        threshold: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
    union_mask = np.zeros_like(box_mask, dtype=bool)
    selected = []
    for candidate in candidates:
        metrics = mask_metrics(candidate['mask'], box_mask)
        if metrics['instance_coverage'] <= threshold:
            continue
        union_mask = np.logical_or(union_mask, candidate['mask'])
        selected.append({
            'segment_id': candidate['segment_id'],
            'category_id': candidate['category_id'],
            'confidence': candidate['confidence'],
            'instance_coverage': metrics['instance_coverage'],
            'box_coverage': metrics['box_coverage'],
            'iou': metrics['iou'],
            'mask_area': metrics['mask_area'],
        })
    return union_mask, selected


def save_visualization(image: np.ndarray, mask: np.ndarray,
                       box: Sequence[int], caption: str, output_path: Path,
                       status: str) -> None:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 2] = 255
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 0.55, color[mask_bool], 0.45, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    x1, y1, x2, y2 = box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label = f'{status}: {caption}'
    cv2.putText(overlay, label[:120], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, label[:120], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f'Failed to save visualization: {output_path}')


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def write_comparison(output_root: Path, summaries: Dict[str, Dict]) -> None:
    summary_dir = output_root / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / 'detectron_comparison.json', summaries)
    columns = [
        'method_key', 'method', 'total_samples', 'success_count',
        'partial_success_count', 'fallback_count', 'fallback_rate',
        'success_average_iou', 'success_average_instance_coverage',
        'success_average_box_coverage', 'success_average_mask_box_area_ratio',
        'average_vis_ir_iou', 'total_elapsed_seconds'
    ]
    with (summary_dir / 'detectron_comparison.csv').open(
            'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for method_key, summary in summaries.items():
            row = {'method_key': method_key}
            for column in columns[1:]:
                row[column] = summary.get(column)
            writer.writerow(row)


def summarize(records: Sequence[Dict], elapsed_seconds: float) -> Dict:
    numeric_keys = (
        'candidate_count', 'selected_count', 'mask_area', 'box_area',
        'mask_box_area_ratio', 'instance_coverage', 'box_coverage', 'iou',
        'match_score', 'elapsed_ms')
    summary: Dict[str, object] = {
        'total_samples': len(records),
        'success_count': sum(r['status'] == 'success' for r in records),
        'partial_success_count': sum(
            r['status'] == 'partial_success' for r in records),
        'fallback_count': sum(r['fallback'] for r in records),
        'fallback_rate': (
            sum(r['fallback'] for r in records) / len(records) if records else 0.0),
        'total_elapsed_seconds': elapsed_seconds,
    }
    success_records = [r for r in records if not r['fallback']]
    for key in numeric_keys:
        all_values = [float(r[key]) for r in records if r.get(key) is not None]
        success_values = [
            float(r[key]) for r in success_records if r.get(key) is not None
        ]
        summary[f'output_average_{key}'] = (
            float(np.mean(all_values)) if all_values else None)
        summary[f'success_average_{key}'] = (
            float(np.mean(success_values)) if success_values else None)
    reasons: Dict[str, int] = {}
    for record in records:
        reason = record.get('fallback_reason')
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    summary['fallback_reasons'] = reasons
    flag_counts: Dict[str, int] = {}
    for record in records:
        for flag in record.get('quality_flags', []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary['quality_flag_counts'] = flag_counts
    return summary


def load_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f'Required manifest does not exist: {path}')
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def binary_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool, copy=False)
    second = second.astype(bool, copy=False)
    union = int(np.logical_or(first, second).sum())
    if not union:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def generate_d2(args: argparse.Namespace) -> Dict:
    """Compose D2 from D1-V and D1-I without repeating inference."""
    method_name = METHODS['d2']['name']
    method_dir = args.output_root / method_name
    masks_dir = method_dir / 'masks'
    vis_dir = method_dir / 'visualizations'
    method_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    d1v_dir = args.output_root / METHODS['d1v']['name']
    d1i_dir = args.output_root / METHODS['d1i']['name']
    vis_records = load_manifest(d1v_dir / 'manifest.jsonl')
    ir_records = load_manifest(d1i_dir / 'manifest.jsonl')
    if len(vis_records) != len(ir_records):
        raise RuntimeError(
            f'D1 manifest lengths differ: VIS={len(vis_records)}, '
            f'IR={len(ir_records)}.')

    records: List[Dict] = []
    started = time.perf_counter()
    for processed_index, (vis_record, ir_record) in enumerate(
            zip(vis_records, ir_records)):
        item_started = time.perf_counter()
        sample_id = int(vis_record['sample_id'])
        if sample_id != int(ir_record['sample_id']):
            raise RuntimeError(
                f'D1 sample mismatch: {sample_id} vs {ir_record["sample_id"]}.')
        if vis_record['filename'] != ir_record['filename']:
            raise RuntimeError(
                f'D1 filename mismatch at sample {sample_id}: '
                f'{vis_record["filename"]} vs {ir_record["filename"]}.')

        vis_mask = cv2.imread(vis_record['mask_path'], cv2.IMREAD_GRAYSCALE)
        ir_mask = cv2.imread(ir_record['mask_path'], cv2.IMREAD_GRAYSCALE)
        if vis_mask is None or ir_mask is None:
            raise FileNotFoundError(
                f'Unable to read D1 masks for sample {sample_id}.')
        if vis_mask.shape != ir_mask.shape:
            raise RuntimeError(
                f'D1 mask shapes differ at sample {sample_id}: '
                f'{vis_mask.shape} vs {ir_mask.shape}.')
        vis_mask = vis_mask > 0
        ir_mask = ir_mask > 0
        vis_success = not bool(vis_record['fallback'])
        ir_success = not bool(ir_record['fallback'])

        if vis_success and ir_success:
            final_mask = np.logical_or(vis_mask, ir_mask)
            status = 'success'
            fallback = False
            fallback_reason = None
            vis_ir_iou = binary_mask_iou(vis_mask, ir_mask)
        elif vis_success:
            final_mask = vis_mask
            status = 'partial_success'
            fallback = False
            fallback_reason = 'ir_failed_vis_used'
            vis_ir_iou = None
        elif ir_success:
            final_mask = ir_mask
            status = 'partial_success'
            fallback = False
            fallback_reason = 'vis_failed_ir_used'
            vis_ir_iou = None
        else:
            final_mask = vis_mask
            status = 'fallback'
            fallback = True
            fallback_reason = 'both_modalities_failed'
            vis_ir_iou = None

        height, width = final_mask.shape
        box_mask, clipped_box = box_to_mask(
            vis_record['bbox_xyxy'], height, width)
        metrics = mask_metrics(final_mask, box_mask)
        output_name = (
            f'{sample_id:06d}_{Path(vis_record["filename"]).stem}.png')
        mask_path = masks_dir / output_name
        if args.overwrite or not mask_path.exists():
            if not cv2.imwrite(
                    str(mask_path), final_mask.astype(np.uint8) * 255):
                raise OSError(f'Failed to save mask: {mask_path}')

        elapsed_ms = (time.perf_counter() - item_started) * 1000.0
        record = {
            'sample_id': sample_id,
            'filename': vis_record['filename'],
            'caption': vis_record['caption'],
            'bbox_xyxy': vis_record['bbox_xyxy'],
            'bbox_xyxy_clipped': clipped_box,
            'image_width': width,
            'image_height': height,
            'vis_path': vis_record['image_path'],
            'ir_path': ir_record['image_path'],
            'mask_path': str(mask_path),
            'method': method_name,
            'modality': 'dual',
            'status': status,
            'fallback': fallback,
            'fallback_reason': fallback_reason,
            'candidate_count': (
                int(vis_record['candidate_count'])
                + int(ir_record['candidate_count'])),
            'vis_candidate_count': int(vis_record['candidate_count']),
            'ir_candidate_count': int(ir_record['candidate_count']),
            'selected_count': int(vis_success) + int(ir_success),
            'vis_selected_instance': vis_record['selected_instance'],
            'ir_selected_instance': ir_record['selected_instance'],
            'vis_mask_area': int(vis_mask.sum()),
            'ir_mask_area': int(ir_mask.sum()),
            'vis_ir_iou': vis_ir_iou,
            **metrics,
            'match_score': None,
            'quality_flags': quality_flags(metrics, fallback),
            'elapsed_ms': elapsed_ms,
            'attributes': vis_record.get('attributes', {}),
        }
        records.append(record)

        if processed_index < args.visualize:
            image = read_image(Path(vis_record['image_path']), 'vis')
            save_visualization(
                image, final_mask, clipped_box, vis_record['caption'],
                vis_dir / output_name, status)
        print(
            f'[{method_name}] {processed_index + 1}: sample={sample_id} '
            f'file={vis_record["filename"]} status={status}', flush=True)

    manifest_path = method_dir / 'manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    elapsed_seconds = time.perf_counter() - started
    statistics = summarize(records, elapsed_seconds)
    valid_vis_ir_ious = [
        r['vis_ir_iou'] for r in records if r['vis_ir_iou'] is not None
    ]
    statistics.update({
        'method': method_name,
        'modality': 'dual',
        'annotation_path': str(args.annotations),
        'start': args.start,
        'limit': args.limit,
        'confidence_threshold': args.confidence_threshold,
        'average_vis_ir_iou': (
            float(np.mean(valid_vis_ir_ious)) if valid_vis_ir_ious else None),
    })
    write_json(method_dir / 'statistics.json', statistics)
    return statistics


def generate_d3(args: argparse.Namespace) -> Dict:
    """Compose D3 from D1 coverage-union sidecars."""
    method_name = METHODS['d3']['name']
    method_dir = args.output_root / method_name
    masks_dir = method_dir / 'masks'
    vis_dir = method_dir / 'visualizations'
    method_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    d1v_dir = args.output_root / METHODS['d1v']['name']
    d1i_dir = args.output_root / METHODS['d1i']['name']
    vis_records = load_manifest(d1v_dir / 'manifest.jsonl')
    ir_records = load_manifest(d1i_dir / 'manifest.jsonl')
    if len(vis_records) != len(ir_records):
        raise RuntimeError(
            f'D1 manifest lengths differ: VIS={len(vis_records)}, '
            f'IR={len(ir_records)}.')

    records: List[Dict] = []
    started = time.perf_counter()
    for processed_index, (vis_record, ir_record) in enumerate(
            zip(vis_records, ir_records)):
        item_started = time.perf_counter()
        sample_id = int(vis_record['sample_id'])
        if sample_id != int(ir_record['sample_id']):
            raise RuntimeError(
                f'D1 sample mismatch: {sample_id} vs {ir_record["sample_id"]}.')
        if vis_record['filename'] != ir_record['filename']:
            raise RuntimeError(
                f'D1 filename mismatch at sample {sample_id}.')

        vis_union_path = vis_record.get('coverage_union_path')
        ir_union_path = ir_record.get('coverage_union_path')
        if not vis_union_path or not ir_union_path:
            raise RuntimeError(
                'D1 manifests do not contain D3 coverage sidecars. '
                'Regenerate D1 with the current script.')
        vis_mask = cv2.imread(vis_union_path, cv2.IMREAD_GRAYSCALE)
        ir_mask = cv2.imread(ir_union_path, cv2.IMREAD_GRAYSCALE)
        if vis_mask is None or ir_mask is None:
            raise FileNotFoundError(
                f'Unable to read D3 sidecars for sample {sample_id}.')
        if vis_mask.shape != ir_mask.shape:
            raise RuntimeError(
                f'D3 sidecar shapes differ at sample {sample_id}: '
                f'{vis_mask.shape} vs {ir_mask.shape}.')
        vis_mask = vis_mask > 0
        ir_mask = ir_mask > 0
        vis_count = int(vis_record['coverage_selected_count'])
        ir_count = int(ir_record['coverage_selected_count'])

        if vis_count and ir_count:
            final_mask = np.logical_or(vis_mask, ir_mask)
            status = 'success'
            fallback = False
            fallback_reason = None
            vis_ir_iou = binary_mask_iou(vis_mask, ir_mask)
        elif vis_count:
            final_mask = vis_mask
            status = 'partial_success'
            fallback = False
            fallback_reason = 'ir_no_qualified_instance_vis_used'
            vis_ir_iou = None
        elif ir_count:
            final_mask = ir_mask
            status = 'partial_success'
            fallback = False
            fallback_reason = 'vis_no_qualified_instance_ir_used'
            vis_ir_iou = None
        else:
            height = int(vis_record['image_height'])
            width = int(vis_record['image_width'])
            final_mask, _ = box_to_mask(
                vis_record['bbox_xyxy'], height, width)
            status = 'fallback'
            fallback = True
            fallback_reason = 'no_qualified_instance'
            vis_ir_iou = None

        height, width = final_mask.shape
        box_mask, clipped_box = box_to_mask(
            vis_record['bbox_xyxy'], height, width)
        metrics = mask_metrics(final_mask, box_mask)
        output_name = (
            f'{sample_id:06d}_{Path(vis_record["filename"]).stem}.png')
        mask_path = masks_dir / output_name
        if args.overwrite or not mask_path.exists():
            if not cv2.imwrite(
                    str(mask_path), final_mask.astype(np.uint8) * 255):
                raise OSError(f'Failed to save mask: {mask_path}')

        elapsed_ms = (time.perf_counter() - item_started) * 1000.0
        record = {
            'sample_id': sample_id,
            'filename': vis_record['filename'],
            'caption': vis_record['caption'],
            'bbox_xyxy': vis_record['bbox_xyxy'],
            'bbox_xyxy_clipped': clipped_box,
            'image_width': width,
            'image_height': height,
            'vis_path': vis_record['image_path'],
            'ir_path': ir_record['image_path'],
            'mask_path': str(mask_path),
            'method': method_name,
            'modality': 'dual',
            'status': status,
            'fallback': fallback,
            'fallback_reason': fallback_reason,
            'coverage_threshold': 0.5,
            'candidate_count': (
                int(vis_record['candidate_count'])
                + int(ir_record['candidate_count'])),
            'vis_candidate_count': int(vis_record['candidate_count']),
            'ir_candidate_count': int(ir_record['candidate_count']),
            'selected_count': vis_count + ir_count,
            'vis_selected_count': vis_count,
            'ir_selected_count': ir_count,
            'vis_selected_instances': vis_record['coverage_selected_instances'],
            'ir_selected_instances': ir_record['coverage_selected_instances'],
            'vis_mask_area': int(vis_mask.sum()),
            'ir_mask_area': int(ir_mask.sum()),
            'vis_ir_iou': vis_ir_iou,
            **metrics,
            'match_score': None,
            'quality_flags': quality_flags(metrics, fallback),
            'elapsed_ms': elapsed_ms,
            'attributes': vis_record.get('attributes', {}),
        }
        records.append(record)

        if processed_index < args.visualize:
            image = read_image(Path(vis_record['image_path']), 'vis')
            save_visualization(
                image, final_mask, clipped_box, vis_record['caption'],
                vis_dir / output_name, status)
        print(
            f'[{method_name}] {processed_index + 1}: sample={sample_id} '
            f'file={vis_record["filename"]} selected={vis_count + ir_count} '
            f'status={status}', flush=True)

    manifest_path = method_dir / 'manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    elapsed_seconds = time.perf_counter() - started
    statistics = summarize(records, elapsed_seconds)
    valid_vis_ir_ious = [
        r['vis_ir_iou'] for r in records if r['vis_ir_iou'] is not None
    ]
    statistics.update({
        'method': method_name,
        'modality': 'dual',
        'annotation_path': str(args.annotations),
        'start': args.start,
        'limit': args.limit,
        'confidence_threshold': args.confidence_threshold,
        'coverage_threshold': 0.5,
        'average_vis_ir_iou': (
            float(np.mean(valid_vis_ir_ious)) if valid_vis_ir_ious else None),
    })
    write_json(method_dir / 'statistics.json', statistics)
    return statistics


def generate_method(args: argparse.Namespace, method_key: str,
                    predictor: DefaultPredictor) -> Dict:
    spec = METHODS[method_key]
    modality = spec['modality']
    method_name = spec['name']
    method_dir = args.output_root / method_name
    masks_dir = method_dir / 'masks'
    coverage_dir = method_dir / 'coverage_unions'
    vis_dir = method_dir / 'visualizations'
    method_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    coverage_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    cache = PanopticInstanceCache(predictor, args.cache_size)
    started = time.perf_counter()
    for processed_index, (sample_id, sample) in enumerate(
            iter_samples(args.annotations, args.start, args.limit)):
        item_started = time.perf_counter()
        filename = sample['filename']
        caption = sample['grounding']['caption']
        bbox = sample['grounding']['regions'][0]['bbox']
        image_path = args.data_root / modality / filename
        image = read_image(image_path, modality)
        height, width = image.shape[:2]
        box_mask, clipped_box = box_to_mask(bbox, height, width)
        cache_key = f'{modality}:{image_path}'
        candidates = cache.get(cache_key, image)
        candidate, metrics = select_best(candidates, box_mask)
        coverage_union, coverage_instances = select_by_instance_coverage(
            candidates, box_mask, threshold=0.5)

        fallback = candidate is None or metrics is None or metrics['intersection'] == 0
        if fallback:
            selected_mask = box_mask
            metrics = mask_metrics(selected_mask, box_mask)
            metrics['match_score'] = None
            status = 'fallback'
            fallback_reason = 'no_candidate' if not candidates else 'no_box_overlap'
            selected_info = None
        else:
            selected_mask = candidate['mask']
            status = 'success'
            fallback_reason = None
            selected_info = {
                'segment_id': candidate['segment_id'],
                'category_id': candidate['category_id'],
                'confidence': candidate['confidence'],
            }

        output_name = f'{sample_id:06d}_{Path(filename).stem}.png'
        mask_path = masks_dir / output_name
        coverage_union_path = coverage_dir / output_name
        if args.overwrite or not mask_path.exists():
            if not cv2.imwrite(str(mask_path), selected_mask.astype(np.uint8) * 255):
                raise OSError(f'Failed to save mask: {mask_path}')
        if args.overwrite or not coverage_union_path.exists():
            if not cv2.imwrite(
                    str(coverage_union_path),
                    coverage_union.astype(np.uint8) * 255):
                raise OSError(
                    f'Failed to save coverage union: {coverage_union_path}')
        elapsed_ms = (time.perf_counter() - item_started) * 1000.0
        record = {
            'sample_id': sample_id,
            'filename': filename,
            'caption': caption,
            'bbox_xyxy': [float(v) for v in bbox],
            'bbox_xyxy_clipped': clipped_box,
            'image_width': width,
            'image_height': height,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'method': method_name,
            'modality': modality,
            'status': status,
            'fallback': fallback,
            'fallback_reason': fallback_reason,
            'candidate_count': len(candidates),
            'selected_count': 0 if fallback else 1,
            'selected_instance': selected_info,
            'coverage_threshold': 0.5,
            'coverage_selected_count': len(coverage_instances),
            'coverage_selected_instances': coverage_instances,
            'coverage_union_path': str(coverage_union_path),
            **metrics,
            'quality_flags': quality_flags(metrics, fallback),
            'elapsed_ms': elapsed_ms,
            'attributes': sample.get('attributes', {}),
        }
        records.append(record)

        if processed_index < args.visualize:
            save_visualization(
                image, selected_mask, clipped_box, caption,
                vis_dir / output_name, status)
        print(
            f'[{method_name}] {processed_index + 1}: sample={sample_id} '
            f'file={filename} candidates={len(candidates)} status={status}',
            flush=True)

    manifest_path = method_dir / 'manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    elapsed_seconds = time.perf_counter() - started
    statistics = summarize(records, elapsed_seconds)
    statistics.update({
        'method': method_name,
        'modality': modality,
        'annotation_path': str(args.annotations),
        'start': args.start,
        'limit': args.limit,
        'confidence_threshold': args.confidence_threshold,
    })
    write_json(method_dir / 'statistics.json', statistics)
    return statistics


def main() -> None:
    args = parse_args()
    validate_args(args)
    summaries = {}
    requested_methods = list(dict.fromkeys(args.methods))
    inference_methods = [m for m in requested_methods if m in ('d1v', 'd1i')]
    if 'd2' in requested_methods or 'd3' in requested_methods:
        for dependency in ('d1v', 'd1i'):
            if dependency not in inference_methods:
                inference_methods.append(dependency)
    if inference_methods:
        predictor = build_predictor(args)
        for method_key in inference_methods:
            summaries[method_key] = generate_method(args, method_key, predictor)
    if 'd2' in requested_methods:
        summaries['d2'] = generate_d2(args)
    if 'd3' in requested_methods:
        summaries['d3'] = generate_d3(args)
    write_comparison(args.output_root, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
