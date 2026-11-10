#!/usr/bin/env python3
"""Apply reviewed FLIR caption/box revisions to canonical annotations."""

import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path('/home/lbw/mmdetection')
ANNOTATION_DIR = ROOT / 'data/FLIR/annotations'
REVIEW_FILE = ROOT / 'analysis_demo/flir_text_anomalies_reviewed.json'
REVISION_FILE = ROOT / 'tools/analysis_tools/flir_test_text_anomaly_revisions_preview.json'
SMALL_AREA_THRESHOLD = 0.01 * 640 * 512
SUBSETS = ('testA', 'testB', 'testC')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-dir', type=Path, default=ANNOTATION_DIR)
    parser.add_argument('--review-file', type=Path, default=REVIEW_FILE)
    parser.add_argument('--revision-file', type=Path, default=REVISION_FILE)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_json(path, payload):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def atomic_jsonl(path, rows):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def annotation_map(payload):
    result = {ann['image_id']: ann for ann in payload['annotations']}
    if len(result) != len(payload['annotations']):
        raise ValueError('Annotations do not have unique image_id values.')
    return result


def scale_class(area):
    return 'small' if area < SMALL_AREA_THRESHOLD else 'normal'


def subset_filter(name, image):
    if name == 'testA':
        return image['lighting'] == 3 and image['scale_cls'] == 'normal'
    if name == 'testB':
        return image['lighting'] != 3
    if name == 'testC':
        return image['scale_cls'] == 'small'
    raise ValueError(name)


def build_subset(name, canonical, template):
    anns = annotation_map(canonical)
    result = copy.deepcopy(template)
    result['images'] = []
    result['annotations'] = []
    for source_image in canonical['images']:
        if not subset_filter(name, source_image):
            continue
        new_id = len(result['images'])
        image = copy.deepcopy(source_image)
        ann = copy.deepcopy(anns[source_image['id']])
        image['id'] = new_id
        ann['id'] = new_id
        ann['image_id'] = new_id
        ann['original_id'] = new_id
        if name in {'testA', 'testB'}:
            for attribute in ('crowded', 'scene', 'weather', 'occlusion'):
                image[attribute] = None
        result['images'].append(image)
        result['annotations'].append(ann)
    return result


def validate_source(payloads, review_by_key, revisions):
    if set(review_by_key) != set(revisions):
        raise ValueError('Revision keys do not exactly match reviewed keys.')
    for key, reviewed in review_by_key.items():
        split, index_text = key.split(':', 1)
        index = int(index_text)
        image = payloads[split]['images'][index]
        ann = annotation_map(payloads[split])[image['id']]
        if image['id'] != index or image['file_name'] != reviewed['file_name']:
            raise ValueError(f'Identity mismatch at {key}')
        if image['caption'] != reviewed['caption']:
            raise ValueError(f'Source caption changed at {key}')
        if [float(v) for v in ann['bbox']] != [
                float(v) for v in reviewed['bbox_xywh']]:
            raise ValueError(f'Source bbox changed at {key}')


def apply_revisions(payloads, revisions):
    changed_boxes = []
    for key, revision in revisions.items():
        split, index_text = key.split(':', 1)
        index = int(index_text)
        payload = payloads[split]
        image = payload['images'][index]
        ann = annotation_map(payload)[image['id']]
        caption = revision['proposed_text']
        if not caption or not caption.isascii():
            raise ValueError(f'Caption must be non-empty English at {key}')
        image['caption'] = caption
        ann['tokens_positive'] = [[0, len(caption)]]
        if 'bbox_xyxy' in revision:
            x1, y1, x2, y2 = [float(v) for v in revision['bbox_xyxy']]
            if not (0 <= x1 < x2 <= image['width'] and
                    0 <= y1 < y2 <= image['height']):
                raise ValueError(f'Invalid bbox at {key}')
            width, height = x2 - x1, y2 - y1
            ann['bbox'] = [x1, y1, width, height]
            ann['area'] = width * height
            image['scale_cls'] = scale_class(ann['area'])
            changed_boxes.append(key)
    return changed_boxes


def synchronize_train_vg(train, vg):
    if len(train['images']) != len(vg):
        raise ValueError('Train/VG population mismatch.')
    anns = annotation_map(train)
    result = copy.deepcopy(vg)
    for index, image in enumerate(train['images']):
        ann = anns[image['id']]
        x, y, width, height = ann['bbox']
        row = result[index]
        if row['filename'] != image['file_name']:
            raise ValueError(f'Train/VG filename mismatch at {index}')
        row['grounding']['caption'] = image['caption']
        region = row['grounding']['regions'][0]
        region['bbox'] = [x, y, x + width, y + height]
        region['phrase'] = image['caption']
        region['tokens_positive'] = [[0, len(image['caption'])]]
        row['attributes']['scale_cls'] = image['scale_cls']
    return result


def validate(payloads, vg, subsets, revisions):
    for split, payload in payloads.items():
        anns = annotation_map(payload)
        if len(payload['images']) != len(payload['annotations']):
            raise ValueError(f'Population mismatch in {split}')
        for image in payload['images']:
            ann = anns[image['id']]
            x, y, width, height = ann['bbox']
            if not (0 <= x < x + width <= image['width'] and
                    0 <= y < y + height <= image['height']):
                raise ValueError(f'Out-of-bounds bbox in {split}:{image["id"]}')
            if abs(float(ann['area']) - width * height) > 1e-6:
                raise ValueError(f'Area mismatch in {split}:{image["id"]}')
            if image['scale_cls'] != scale_class(ann['area']):
                raise ValueError(f'Scale mismatch in {split}:{image["id"]}')
            if ann['tokens_positive'] != [[0, len(image['caption'])]]:
                raise ValueError(f'Token mismatch in {split}:{image["id"]}')
    anns = annotation_map(payloads['train'])
    for index, row in enumerate(vg):
        image = payloads['train']['images'][index]
        ann = anns[image['id']]
        x, y, width, height = ann['bbox']
        region = row['grounding']['regions'][0]
        if (row['filename'] != image['file_name'] or
                row['grounding']['caption'] != image['caption'] or
                region['bbox'] != [x, y, x + width, y + height]):
            raise ValueError(f'VG mismatch at train:{index}')
    for name, subset in subsets.items():
        expected = sum(subset_filter(name, im)
                       for im in payloads['test']['images'])
        if len(subset['images']) != expected:
            raise ValueError(f'Subset population mismatch in {name}')
    for key, revision in revisions.items():
        split, index_text = key.split(':', 1)
        image = payloads[split]['images'][int(index_text)]
        if image['caption'] != revision['proposed_text']:
            raise ValueError(f'Revision missing at {key}')


def main():
    args = parse_args()
    paths = {split: args.annotation_dir / f'rgbtvg_flir_{split}.json'
             for split in ('train', 'val', 'test', *SUBSETS)}
    payloads = {split: load(paths[split])
                for split in ('train', 'val', 'test')}
    templates = {name: load(paths[name]) for name in SUBSETS}
    review = load(args.review_file)
    revisions = load(args.revision_file)['revisions']
    review_by_key = {f"{s['split']}:{s['index']}": s
                     for s in review['samples']}
    validate_source(payloads, review_by_key, revisions)
    changed_boxes = apply_revisions(payloads, revisions)
    vg_path = args.annotation_dir / 'rgbtvg_flir_train_vg.json'
    vg = synchronize_train_vg(payloads['train'], load_jsonl(vg_path))
    subsets = {name: build_subset(name, payloads['test'], templates[name])
               for name in SUBSETS}
    validate(payloads, vg, subsets, revisions)
    summary = {
        'mode': 'apply' if args.apply else 'dry_run',
        'changed_text_count': len(revisions),
        'changed_box_count': len(changed_boxes),
        'changed_box_counts_by_split': dict(Counter(
            key.split(':', 1)[0] for key in changed_boxes)),
        'final_counts': {**{k: len(v['images']) for k, v in payloads.items()},
                         **{k: len(v['images']) for k, v in subsets.items()}},
    }
    if args.apply:
        for split, payload in payloads.items():
            atomic_json(paths[split], payload)
        for name, payload in subsets.items():
            atomic_json(paths[name], payload)
        atomic_jsonl(vg_path, vg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
