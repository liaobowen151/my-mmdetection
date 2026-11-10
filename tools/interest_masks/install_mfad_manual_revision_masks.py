#!/usr/bin/env python3
"""Install regenerated MFAD masks and merge their segmentation records."""

import argparse
import filecmp
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path('/home/lbw/mmdetection')
DATA_ROOT = ROOT / 'data/MFAD'
ANNOTATIONS = DATA_ROOT / 'annotations/rgbtvg_mfad_train_vg.json'
SEG_INDEX = DATA_ROOT / 'annotations/rgbtvg_mfad_train_seg.jsonl'
STATISTICS = (
    DATA_ROOT / 'annotations/rgbtvg_mfad_train_seg_statistics.json')
MASK_DIR = DATA_ROOT / 'mask'
GENERATED_ROOT = Path('/tmp/mfad_manual_revision_masks')
GENERATED_RECORDS = GENERATED_ROOT / 'records.json'
BACKUP_DIR = ROOT / 'analysis_demo/mfad_manual_revision_backup_20261109'

REVISED_IDS = (1, 2, 5478, 5902, 6198, 13496, 15255, 15518, 15778,
               15841, 15842)
REGENERATED_IDS = (5902, 6198, 15778, 15841, 15842)
NUMERIC_FIELDS = (
    'candidate_count', 'selected_count', 'mask_area', 'box_area',
    'mask_box_area_ratio', 'instance_coverage', 'box_coverage', 'iou',
    'elapsed_ms')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_generated(records):
    by_id = {int(record['sample_id']): record for record in records}
    if set(by_id) != set(REGENERATED_IDS):
        raise ValueError('Generated record IDs do not match revised boxes.')
    for sample_id, record in by_id.items():
        source = GENERATED_ROOT / record['mask_path']
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (960, 1280):
            raise ValueError(f'Invalid generated mask for {sample_id}.')
        values = set(int(value) for value in set(mask.ravel()))
        if not values <= {0, 255} or 255 not in values:
            raise ValueError(
                f'Generated mask is not non-empty binary for {sample_id}.')
        if int((mask > 0).sum()) != int(record['mask_area']):
            raise ValueError(f'Mask-area mismatch for {sample_id}.')
    return by_id


def merge_records(existing, generated, annotations):
    if not (len(existing) == len(annotations) == 16000):
        raise ValueError('Unexpected segmentation/annotation population.')
    if any(int(record['sample_id']) != index
           for index, record in enumerate(existing)):
        raise ValueError('Existing segmentation index is not ordered.')
    merged = [dict(record) for record in existing]
    for sample_id in REVISED_IDS:
        sample = annotations[sample_id]
        caption = sample['grounding']['caption']
        bbox = [float(value) for value in
                sample['grounding']['regions'][0]['bbox']]
        if sample_id in generated:
            record = dict(generated[sample_id])
            if record['caption'] != caption or record['bbox_xyxy'] != bbox:
                raise ValueError(
                    f'Generated record differs from annotation {sample_id}.')
            record['manual_revision'] = {
                'mask_regenerated': True,
                'caption_synchronized': True,
            }
            merged[sample_id] = record
        else:
            if merged[sample_id]['bbox_xyxy'] != bbox:
                raise ValueError(
                    f'Unexpected bbox change without regeneration: {sample_id}')
            merged[sample_id]['caption'] = caption
            merged[sample_id]['manual_revision'] = {
                'mask_regenerated': False,
                'caption_synchronized': True,
            }
    return merged


def recompute_statistics(records, old_statistics, generated):
    statuses = Counter(record['status'] for record in records)
    result = dict(old_statistics)
    result.update({
        'total_samples': len(records),
        'dual_success_count': statuses['dual_success'],
        'vis_only_count': statuses['vis_only'],
        'ir_only_count': statuses['ir_only'],
        'bbox_fallback_count': statuses['bbox_fallback'],
        'bbox_fallback_rate': statuses['bbox_fallback'] / len(records),
        'empty_mask_count': sum(record['mask_area'] == 0
                                for record in records),
        'quality_flag_counts': dict(Counter(
            flag for record in records for flag in record['quality_flags'])),
    })
    for key in NUMERIC_FIELDS:
        result[f'average_{key}'] = (
            sum(float(record[key]) for record in records) / len(records))
    result['manual_revision'] = {
        'caption_synchronized_sample_ids': list(REVISED_IDS),
        'mask_regenerated_sample_ids': list(REGENERATED_IDS),
        'incremental_regeneration_elapsed_seconds': (
            sum(float(record['elapsed_ms']) for record in generated.values())
            / 1000.0),
        'method': ('Detectron2 Panoptic FPN on paired VIS/IR, identical '
                   'thresholds to the original generation'),
    }
    return result


def backup(source):
    relative = source.relative_to(ROOT)
    destination = BACKUP_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not filecmp.cmp(source, destination, shallow=False):
            raise RuntimeError(f'Existing backup differs: {destination}')
        return
    shutil.copy2(source, destination)


def atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    temporary.replace(path)


def atomic_write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def install_mask(source, destination):
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main():
    args = parse_args()
    annotations = load_jsonl(ANNOTATIONS)
    existing = load_jsonl(SEG_INDEX)
    generated_payload = json.loads(
        GENERATED_RECORDS.read_text(encoding='utf-8'))
    generated = validate_generated(generated_payload['records'])
    old_statistics = json.loads(STATISTICS.read_text(encoding='utf-8'))
    merged = merge_records(existing, generated, annotations)
    statistics = recompute_statistics(merged, old_statistics, generated)

    summary = {
        'mode': 'apply' if args.apply else 'dry-run',
        'record_count': len(merged),
        'caption_synchronized_sample_ids': list(REVISED_IDS),
        'mask_regenerated_sample_ids': list(REGENERATED_IDS),
        'new_statuses': {
            str(sample_id): generated[sample_id]['status']
            for sample_id in REGENERATED_IDS
        },
        'global_counts': {
            key: statistics[key] for key in (
                'dual_success_count', 'vis_only_count', 'ir_only_count',
                'bbox_fallback_count')
        },
    }
    if not args.apply:
        print(json.dumps(summary, indent=2))
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup(SEG_INDEX)
    backup(STATISTICS)
    for sample_id in REGENERATED_IDS:
        mask_name = Path(generated[sample_id]['mask_path']).name
        backup(MASK_DIR / mask_name)
    for sample_id in REGENERATED_IDS:
        mask_name = Path(generated[sample_id]['mask_path']).name
        install_mask(
            GENERATED_ROOT / generated[sample_id]['mask_path'],
            MASK_DIR / mask_name)
    atomic_write_jsonl(SEG_INDEX, merged)
    atomic_write_json(STATISTICS, statistics)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
