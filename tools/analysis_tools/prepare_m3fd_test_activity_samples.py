#!/usr/bin/env python3
"""Select deterministic M3FD COCO samples and export ODVG JSONL."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', default=10, type=int)
    parser.add_argument('--seed', default=20261102, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding='utf-8'))
    images = {item['id']: item for item in coco['images']}
    candidates = []
    for ann in coco['annotations']:
        image = images[ann['image_id']]
        x, y, width, height = ann['bbox']
        caption = image['caption']
        candidates.append({
            'filename': image['file_name'],
            'height': image['height'],
            'width': image['width'],
            'grounding': {
                'caption': caption,
                'regions': [{
                    'bbox': [x, y, x + width, y + height],
                    'phrase': caption,
                    'tokens_positive': ann.get(
                        'tokens_positive', [[0, len(caption)]])
                }]
            },
            'source_image_id': image['id'],
            'source_annotation_id': ann['id'],
            'attributes': {
                key: image.get(key) for key in
                ('lighting', 'scale_cls', 'crowded', 'scene', 'weather',
                 'occlusion')
            }
        })
    if args.count > len(candidates):
        raise ValueError('Requested more samples than available.')
    selected = random.Random(args.seed).sample(candidates, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Wrote {len(selected)} samples to {args.output}')
    for index, item in enumerate(selected):
        print(index, item['filename'], item['attributes'])


if __name__ == '__main__':
    main()
