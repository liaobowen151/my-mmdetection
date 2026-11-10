#!/usr/bin/env python3
"""Install regenerated FLIR masks and synchronize train segmentation index."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path('/home/lbw/mmdetection')
DATA_ROOT = ROOT / 'data/FLIR'
ANNOTATIONS = DATA_ROOT / 'annotations/rgbtvg_flir_train_vg.json'
SEG_INDEX = DATA_ROOT / 'annotations/rgbtvg_flir_train_seg.jsonl'
STATISTICS = DATA_ROOT / 'annotations/rgbtvg_flir_train_seg_statistics.json'
MASK_DIR = DATA_ROOT / 'mask'
REVISIONS = ROOT / 'tools/analysis_tools/flir_test_text_anomaly_revisions_preview.json'
GENERATED_ROOT = Path('/tmp/flir_text_revision_masks')
NUMERIC_FIELDS = ('candidate_count', 'selected_count', 'mask_area',
                  'box_area', 'mask_box_area_ratio', 'instance_coverage',
                  'box_coverage', 'iou', 'elapsed_ms')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)
                         + '\n', encoding='utf-8')
    temporary.replace(path)


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    args = parse_args()
    annotations = load_jsonl(ANNOTATIONS)
    existing = load_jsonl(SEG_INDEX)
    revisions = json.loads(REVISIONS.read_text(
        encoding='utf-8'))['revisions']
    train_revisions = {int(k.split(':', 1)[1]): v for k, v in revisions.items()
                       if k.startswith('train:')}
    regenerated_ids = {sample_id for sample_id, revision
                       in train_revisions.items() if 'bbox_xyxy' in revision}
    generated_rows = json.loads((GENERATED_ROOT / 'records.json').read_text(
        encoding='utf-8'))['records']
    generated = {int(row['sample_id']): row for row in generated_rows}
    if set(generated) != regenerated_ids:
        raise ValueError('Generated mask IDs do not match revised train boxes.')
    if not (len(annotations) == len(existing) == 7000):
        raise ValueError('Unexpected FLIR train population.')
    merged = [dict(row) for row in existing]
    for sample_id, revision in train_revisions.items():
        annotation = annotations[sample_id]
        caption = annotation['grounding']['caption']
        bbox = [float(v) for v in
                annotation['grounding']['regions'][0]['bbox']]
        if caption != revision['proposed_text']:
            raise ValueError(f'Annotation caption not applied at train:{sample_id}')
        if sample_id in regenerated_ids:
            row = dict(generated[sample_id])
            if row['caption'] != caption or row['bbox_xyxy'] != bbox:
                raise ValueError(f'Generated row mismatch at train:{sample_id}')
            source = GENERATED_ROOT / row['mask_path']
            mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (512, 640):
                raise ValueError(f'Invalid mask at train:{sample_id}')
            if set(int(v) for v in set(mask.ravel())) - {0, 255}:
                raise ValueError(f'Non-binary mask at train:{sample_id}')
            if int((mask > 0).sum()) != int(row['mask_area']):
                raise ValueError(f'Mask area mismatch at train:{sample_id}')
            row['manual_revision'] = {
                'mask_regenerated': True, 'caption_synchronized': True}
            merged[sample_id] = row
        else:
            if [float(v) for v in merged[sample_id]['bbox_xyxy']] != bbox:
                raise ValueError(f'Unexpected bbox change at train:{sample_id}')
            merged[sample_id]['caption'] = caption
            merged[sample_id]['manual_revision'] = {
                'mask_regenerated': False, 'caption_synchronized': True}
    old_statistics = json.loads(STATISTICS.read_text(encoding='utf-8'))
    statuses = Counter(row['status'] for row in merged)
    statistics = dict(old_statistics)
    statistics.update({
        'total_samples': len(merged),
        'dual_success_count': statuses['dual_success'],
        'vis_only_count': statuses['vis_only'],
        'ir_only_count': statuses['ir_only'],
        'bbox_fallback_count': statuses['bbox_fallback'],
        'bbox_fallback_rate': statuses['bbox_fallback'] / len(merged),
        'empty_mask_count': sum(int(row['mask_area']) == 0 for row in merged),
        'quality_flag_counts': dict(Counter(
            flag for row in merged for flag in row['quality_flags'])),
    })
    for field in NUMERIC_FIELDS:
        statistics[f'average_{field}'] = (
            sum(float(row[field]) for row in merged) / len(merged))
    statistics['manual_text_revision'] = {
        'caption_synchronized_count': len(train_revisions),
        'mask_regenerated_count': len(regenerated_ids),
        'mask_regenerated_sample_ids': sorted(regenerated_ids),
        'method': ('Detectron2 Panoptic FPN on paired VIS/IR using the same '
                   'thresholds as the original mask generation'),
    }
    summary = {
        'mode': 'apply' if args.apply else 'dry_run',
        'caption_synchronized_count': len(train_revisions),
        'mask_regenerated_count': len(regenerated_ids),
        'status_counts': dict(statuses),
    }
    if args.apply:
        for sample_id in sorted(regenerated_ids):
            row = generated[sample_id]
            source = GENERATED_ROOT / row['mask_path']
            destination = MASK_DIR / Path(row['mask_path']).name
            temporary = destination.with_suffix('.png.tmp')
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        atomic_jsonl(SEG_INDEX, merged)
        atomic_json(STATISTICS, statistics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
