#!/usr/bin/env python3
"""Apply confirmed MFAD manual revisions to annotations and span catalogs."""

import argparse
import copy
import filecmp
import json
import shutil
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path('/home/lbw/mmdetection')
REFERENT_DIR = ROOT / 'analysis_demo/referent_spans'
ANNOTATION_DIR = ROOT / 'data/MFAD/annotations'
REVISION_FILE = (
    ROOT / 'tools/analysis_tools/mfad_manual_review_revisions_draft.json')
BACKUP_DIR = (
    ROOT / 'analysis_demo/mfad_manual_revision_backup_20261109')
TOKENIZER_DIR = (
    ROOT / 'huggingface-hub/transformers/bert-base-uncased')

FILES_TO_BACK_UP = (
    ANNOTATION_DIR / 'rgbtvg_mfad_train_vg.json',
    ANNOTATION_DIR / 'rgbtvg_mfad_train.json',
    REFERENT_DIR / 'mfad_train_referent_spans_low.json',
    REFERENT_DIR / 'mfad_train_referent_spans_low_fix.json',
    REFERENT_DIR / 'mfad_train_referent_spans_all.json',
    REFERENT_DIR / 'mfad_train_referent_spans_manual_review.json',
)

HEADS = {
    1: 'car',
    2: 'bus',
    5478: 'truck',
    5902: 'board',
    6198: 'building',
    13496: 'pedestrian',
    15255: 'bus',
    15518: 'vehicle',
    15778: 'pedestrian',
    15841: 'person',
    15842: 'person',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true',
        help='Write changes. Without this flag, only validate and summarize.')
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def referent_span(caption):
    if caption.startswith('The '):
        return [[4, len(caption)]]
    return [[0, len(caption)]]


def bert_evidence(tokenizer, caption, spans):
    encoding = tokenizer(
        caption, add_special_tokens=True, max_length=256, truncation=True,
        return_offsets_mapping=True)
    evidence = []
    for start, end in spans:
        begin_token = encoding.char_to_token(start)
        end_token = encoding.char_to_token(end - 1)
        if begin_token is None or end_token is None:
            raise ValueError(
                f'BERT mapping failed for {[start, end]} in {caption!r}')
        evidence.append({
            'char_span': [start, end],
            'token_span': [begin_token, end_token + 1],
            'bert_tokens': tokenizer.convert_ids_to_tokens(
                encoding['input_ids'][begin_token:end_token + 1]),
        })
    return evidence


def load_revisions():
    payload = json.loads(REVISION_FILE.read_text(encoding='utf-8'))
    revisions = {
        int(index): dict(revision)
        for index, revision in payload['revisions'].items()
    }
    if set(revisions) != set(HEADS):
        raise ValueError('Revision indices differ from the confirmed set.')
    for index, revision in revisions.items():
        caption = revision['proposed_text']
        if not caption or not caption.isascii():
            raise ValueError(f'Caption for {index} is not non-empty English.')
    return revisions


def updated_box(revision, old_box):
    return [float(value) for value in revision.get('bbox_xyxy', old_box)]


def update_vg(rows, revisions):
    before = {}
    for index, revision in revisions.items():
        row = rows[index]
        region = row['grounding']['regions'][0]
        before[index] = {
            'caption': row['grounding']['caption'],
            'bbox_xyxy': list(region['bbox']),
            'tokens_positive': copy.deepcopy(region['tokens_positive']),
        }
        caption = revision['proposed_text']
        box = updated_box(revision, region['bbox'])
        spans = referent_span(caption)
        row['grounding']['caption'] = caption
        region['phrase'] = caption
        region['bbox'] = box
        region['tokens_positive'] = spans
    return before


def update_coco(payload, revisions, vg_rows):
    images = payload['images']
    annotations = payload['annotations']
    if not (len(images) == len(annotations) == 16000):
        raise ValueError('Unexpected COCO annotation size.')
    for index, revision in revisions.items():
        image = images[index]
        annotation = annotations[index]
        vg_row = vg_rows[index]
        if image['id'] != index or annotation['image_id'] != index:
            raise ValueError(f'COCO index mismatch at {index}.')
        if image['file_name'] != vg_row['filename']:
            raise ValueError(f'COCO/VG filename mismatch at {index}.')
        caption = vg_row['grounding']['caption']
        x1, y1, x2, y2 = vg_row['grounding']['regions'][0]['bbox']
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            raise ValueError(f'Invalid revised box at {index}.')
        image['caption'] = caption
        annotation['bbox'] = [x1, y1, width, height]
        annotation['area'] = width * height
        annotation['tokens_positive'] = referent_span(caption)


def update_span_sample(sample, index, revision, before, tokenizer,
                       reviewed):
    old_caption = sample['caption']
    old_box = list(sample['bbox_xyxy'])
    caption = revision['proposed_text']
    box = updated_box(revision, before[index]['bbox_xyxy'])
    spans = referent_span(caption)
    texts = [caption[start:end] for start, end in spans]
    evidence = bert_evidence(tokenizer, caption, spans)
    head = HEADS[index]
    head_start = caption.lower().rfind(head.lower())
    if head_start < 0:
        raise ValueError(f'Could not find head {head!r} in {caption!r}')
    head_span = [head_start, head_start + len(head)]

    sample['pre_manual_caption'] = old_caption
    sample['pre_manual_bbox_xyxy'] = old_box
    sample['pre_manual_referent_positive_spans'] = copy.deepcopy(
        sample['referent_positive_spans'])
    sample['pre_manual_referent_positive_texts'] = copy.deepcopy(
        sample['referent_positive_texts'])
    sample['caption'] = caption
    sample['bbox_xyxy'] = box
    sample['original_tokens_positive'] = copy.deepcopy(spans)
    sample['subject_head'] = caption[head_span[0]:head_span[1]]
    sample['subject_head_lemma'] = head
    sample['subject_head_pos'] = 'NOUN'
    sample['subject_head_dependency'] = 'manual'
    sample['subject_head_char_span'] = head_span
    sample['subject_phrase'] = texts[0]
    sample['subject_phrase_char_span'] = list(spans[0])
    sample['referent_positive_spans'] = copy.deepcopy(spans)
    sample['referent_positive_texts'] = list(texts)
    sample['bert_span_evidence'] = copy.deepcopy(evidence)
    sample['extraction_method'] = 'manual_review'
    sample['review_flags'] = ['manually_revised_source']
    if reviewed:
        sample['original_referent_positive_spans'] = copy.deepcopy(spans)
        sample['original_referent_positive_texts'] = list(texts)
        sample['fixed_referent_positive_spans'] = copy.deepcopy(spans)
        sample['fixed_referent_positive_texts'] = list(texts)
        sample['fixed_subject_phrase_char_span'] = list(spans[0])
        sample['fixed_subject_phrase'] = texts[0]
        sample['fixed_bert_span_evidence'] = copy.deepcopy(evidence)
        sample['review_status'] = 'corrected'
        sample['review_note'] = (
            'Corrected using the user-confirmed visual inspection of the '
            'paired visible and infrared images.')
        sample.pop('manual_review_reason', None)
    sample['manual_revision'] = {
        'confirmed': True,
        'caption_changed': old_caption != caption,
        'bbox_changed': old_box != box,
        'source': 'user-confirmed paired VIS/IR visualization',
    }


def refresh_payload_counts(payload):
    samples = payload['samples']
    payload['method_counts'] = dict(sorted(Counter(
        sample['extraction_method'] for sample in samples).items()))
    payload['review_flag_counts'] = dict(sorted(Counter(
        flag for sample in samples for flag in sample['review_flags']).items()))


def update_referent_files(revisions, before, tokenizer):
    low_path = REFERENT_DIR / 'mfad_train_referent_spans_low.json'
    fix_path = REFERENT_DIR / 'mfad_train_referent_spans_low_fix.json'
    low = json.loads(low_path.read_text(encoding='utf-8'))
    fix = json.loads(fix_path.read_text(encoding='utf-8'))
    low_by_index = {sample['sample_index']: sample for sample in low['samples']}
    fix_by_index = {sample['sample_index']: sample for sample in fix['samples']}
    if not set(revisions) <= set(low_by_index) or not set(revisions) <= set(
            fix_by_index):
        raise ValueError('Confirmed samples are missing from low catalogs.')
    for index, revision in revisions.items():
        update_span_sample(
            low_by_index[index], index, revision, before, tokenizer,
            reviewed=False)
        update_span_sample(
            fix_by_index[index], index, revision, before, tokenizer,
            reviewed=True)
    refresh_payload_counts(low)
    refresh_payload_counts(fix)
    statuses = Counter(sample['review_status'] for sample in fix['samples'])
    fix['semantic_review']['status_counts'] = {
        key: statuses.get(key, 0)
        for key in ('confirmed', 'corrected', 'needs_manual_review')
    }
    fix['semantic_review']['all_samples_semantically_resolved'] = (
        statuses.get('needs_manual_review', 0) == 0)
    return low, fix


def build_all(fixed_low):
    high = json.loads((
        REFERENT_DIR / 'mfad_train_referent_spans_high.json').read_text(
            encoding='utf-8'))['samples']
    medium = json.loads((
        REFERENT_DIR / 'mfad_train_referent_spans_medium_fix.json').read_text(
            encoding='utf-8'))['samples']
    samples = [dict(sample) for sample in high + medium + fixed_low['samples']]
    for sample in samples:
        if 'fixed_referent_positive_spans' in sample:
            sample['referent_positive_spans'] = copy.deepcopy(
                sample['fixed_referent_positive_spans'])
            sample['referent_positive_texts'] = list(
                sample['fixed_referent_positive_texts'])
    samples.sort(key=lambda sample: sample['sample_index'])
    if [sample['sample_index'] for sample in samples] != list(range(16000)):
        raise ValueError('Merged all-catalog indices are invalid.')
    return {'samples': samples}


def resolve_manual_report(revisions, before):
    path = REFERENT_DIR / 'mfad_train_referent_spans_manual_review.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    old_by_index = {
        sample['sample_index']: sample for sample in payload['samples']
    }
    resolved = []
    for index in sorted(revisions):
        revision = revisions[index]
        item = dict(old_by_index[index])
        item['pre_manual_caption'] = item['caption']
        item['pre_manual_bbox_xyxy'] = item['bbox_xyxy']
        item['caption'] = revision['proposed_text']
        item['bbox_xyxy'] = updated_box(
            revision, before[index]['bbox_xyxy'])
        item['resolved_referent_spans'] = referent_span(item['caption'])
        item['resolved_referent_texts'] = [
            item['caption'][start:end]
            for start, end in item['resolved_referent_spans']
        ]
        item['resolution_status'] = 'resolved_by_user'
        resolved.append(item)
    return {
        'sample_count': 0,
        'resolved_count': len(resolved),
        'instructions': 'All previously ambiguous samples were resolved.',
        'samples': [],
        'resolved_samples': resolved,
    }


def validate(vg_rows, coco, low, fix, all_payload, revisions):
    if not (len(vg_rows) == len(coco['images']) ==
            len(coco['annotations']) == 16000):
        raise ValueError('Population sizes differ after revision.')
    low_by_index = {sample['sample_index']: sample for sample in low['samples']}
    fix_by_index = {sample['sample_index']: sample for sample in fix['samples']}
    all_by_index = {
        sample['sample_index']: sample for sample in all_payload['samples']
    }
    for index, revision in revisions.items():
        caption = revision['proposed_text']
        vg = vg_rows[index]
        region = vg['grounding']['regions'][0]
        expected_box = updated_box(revision, region['bbox'])
        spans = referent_span(caption)
        texts = [caption[start:end] for start, end in spans]
        if vg['grounding']['caption'] != caption:
            raise ValueError(f'VG caption mismatch at {index}.')
        if region['bbox'] != expected_box or region['tokens_positive'] != spans:
            raise ValueError(f'VG region mismatch at {index}.')
        image = coco['images'][index]
        annotation = coco['annotations'][index]
        if image['caption'] != caption or annotation['tokens_positive'] != spans:
            raise ValueError(f'COCO text mismatch at {index}.')
        for catalog in (low_by_index, fix_by_index, all_by_index):
            sample = catalog[index]
            if sample['caption'] != caption:
                raise ValueError(f'Catalog caption mismatch at {index}.')
            if sample['bbox_xyxy'] != region['bbox']:
                raise ValueError(f'Catalog box mismatch at {index}.')
            if sample['referent_positive_spans'] != spans:
                raise ValueError(f'Catalog span mismatch at {index}.')
            if sample['referent_positive_texts'] != texts:
                raise ValueError(f'Catalog text mismatch at {index}.')
    statuses = fix['semantic_review']['status_counts']
    if statuses.get('needs_manual_review') != 0:
        raise ValueError('Manual-review samples remain in low_fix.')


def atomic_write_json(path, payload, indent=None):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent) + '\n',
        encoding='utf-8')
    temporary.replace(path)


def atomic_write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def backup_files():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for source in FILES_TO_BACK_UP:
        # ``all.json`` may not exist yet; it is a generated catalog and will
        # be created by this application step.
        if not source.exists():
            continue
        relative = source.relative_to(ROOT)
        destination = BACKUP_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not filecmp.cmp(source, destination, shallow=False):
                raise RuntimeError(
                    f'Existing backup differs from source: {destination}')
            continue
        shutil.copy2(source, destination)


def main():
    args = parse_args()
    revisions = load_revisions()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True, use_fast=True)
    vg_path = ANNOTATION_DIR / 'rgbtvg_mfad_train_vg.json'
    coco_path = ANNOTATION_DIR / 'rgbtvg_mfad_train.json'
    vg_rows = load_jsonl(vg_path)
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    if len(vg_rows) != 16000:
        raise ValueError(f'Expected 16000 VG rows, found {len(vg_rows)}.')
    before = update_vg(vg_rows, revisions)
    update_coco(coco, revisions, vg_rows)
    low, fix = update_referent_files(revisions, before, tokenizer)
    all_payload = build_all(fix)
    manual_report = resolve_manual_report(revisions, before)
    validate(vg_rows, coco, low, fix, all_payload, revisions)

    bbox_changed = [
        index for index, revision in revisions.items()
        if 'bbox_xyxy' in revision
    ]
    summary = {
        'mode': 'apply' if args.apply else 'dry-run',
        'revised_samples': sorted(revisions),
        'bbox_changed_samples': sorted(bbox_changed),
        'caption_changed_count': len(revisions),
        'low_fix_status_counts': fix['semantic_review']['status_counts'],
        'backup_dir': str(BACKUP_DIR) if args.apply else None,
        'segmentation_artifacts_modified': False,
    }
    if not args.apply:
        print(json.dumps(summary, indent=2))
        return

    backup_files()
    atomic_write_jsonl(vg_path, vg_rows)
    atomic_write_json(coco_path, coco)
    atomic_write_json(
        REFERENT_DIR / 'mfad_train_referent_spans_low.json', low, indent=2)
    atomic_write_json(
        REFERENT_DIR / 'mfad_train_referent_spans_low_fix.json', fix,
        indent=2)
    atomic_write_json(
        REFERENT_DIR / 'mfad_train_referent_spans_all.json', all_payload,
        indent=2)
    atomic_write_json(
        REFERENT_DIR / 'mfad_train_referent_spans_manual_review.json',
        manual_report, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
