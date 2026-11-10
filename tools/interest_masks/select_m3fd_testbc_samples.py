#!/usr/bin/env python3
"""Select deterministic M3FD testB/testC samples for mask comparison.

testB contributes small targets (scale_cls == ``small``). testC contributes
weak-light targets, preferring lighting level 1 and then level 2. The selected
COCO records are converted to the JSONL format consumed by the mask generators.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


ROOT = Path('/home/lbw/mmdetection/data/M3FD')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=(
        ROOT / 'interest_masks_eval_testbc' / 'selections'))
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261102)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict]:
    dataset = json.loads(path.read_text(encoding='utf-8'))
    annotations = {ann['image_id']: ann for ann in dataset['annotations']}
    rows = []
    for image in dataset['images']:
        ann = annotations[image['id']]
        x, y, width, height = ann['bbox']
        rows.append({
            'image': image,
            'annotation': ann,
            'bbox_xyxy': [x, y, x + width, y + height],
        })
    return rows


def choose_testb(rows: List[Dict], count: int, rng: random.Random) -> List[Dict]:
    candidates = [r for r in rows if r['image'].get('scale_cls') == 'small']
    if len(candidates) < count:
        raise RuntimeError(f'Only {len(candidates)} small testB samples found.')
    return sorted(rng.sample(candidates, count), key=lambda r: r['image']['id'])


def choose_testc(rows: List[Dict], count: int, rng: random.Random) -> List[Dict]:
    # In this local M3FD annotation, level 1 is darkest and level 4 brightest.
    # Prefer level 1, then fill from level 2.
    chosen: List[Dict] = []
    for lighting in (1, 2, 3, 4):
        candidates = [r for r in rows if r['image'].get('lighting') == lighting]
        need = count - len(chosen)
        if need <= 0:
            break
        chosen.extend(rng.sample(candidates, min(need, len(candidates))))
    if len(chosen) < count:
        raise RuntimeError(f'Only {len(chosen)} weak-light testC samples found.')
    return sorted(chosen, key=lambda r: r['image']['id'])


def convert(row: Dict, split: str, selection_index: int) -> Dict:
    image = row['image']
    return {
        'filename': image['file_name'],
        'grounding': {
            'caption': image['caption'],
            'regions': [{'bbox': row['bbox_xyxy']}],
        },
        'attributes': {
            'lighting': image.get('lighting'),
            'scale_cls': image.get('scale_cls'),
            'crowded': image.get('crowded'),
            'scene': image.get('scene'),
            'weather': image.get('weather'),
            'occlusion': image.get('occlusion'),
            'source_split': split,
            'source_image_id': image['id'],
            'selection_index': selection_index,
        },
    }


def write_jsonl(path: Path, rows: List[Dict], split: str) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps(
                convert(row, split, index), ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError('--count must be positive.')
    annotations = args.data_root / 'annotations'
    testb = load_rows(annotations / 'rgbtvg_m3fd_testB.json')
    testc = load_rows(annotations / 'rgbtvg_m3fd_testC.json')
    rng = random.Random(args.seed)
    selected_b = choose_testb(testb, args.count, rng)
    selected_c = choose_testc(testc, args.count, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / 'testB_small_20.jsonl', selected_b, 'testB')
    write_jsonl(args.output_dir / 'testC_lowlight_20.jsonl', selected_c, 'testC')
    metadata = {
        'seed': args.seed,
        'count_per_split': args.count,
        'testB_rule': 'scale_cls == small; deterministic random sample',
        'testC_rule': 'prefer lighting 1, then 2; deterministic random sample',
        'testB_filenames': [r['image']['file_name'] for r in selected_b],
        'testC_filenames': [r['image']['file_name'] for r in selected_c],
        'testB_lighting': [r['image'].get('lighting') for r in selected_b],
        'testC_lighting': [r['image'].get('lighting') for r in selected_c],
    }
    (args.output_dir / 'selection_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
