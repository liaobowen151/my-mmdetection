#!/usr/bin/env python3
"""Apply the three resolved FLIR referent captions and refresh fix spans."""

import json
import sys
from collections import Counter
from pathlib import Path

import spacy
from transformers import AutoTokenizer


ROOT = Path('/home/lbw/mmdetection')
sys.path.insert(0, str(ROOT / 'tools/analysis_tools'))
import extract_m3fd_referent_spans_by_confidence as extractor  # noqa: E402


REVISIONS = {
    895: {
        'filename': 'FLIR_04781.jpg',
        'old': "Red car's rear tire and fender partially visible.",
        'new': 'The red car on the left with only its rear wheel visible.',
    },
    6383: {
        'filename': 'FLIR_03190.jpg',
        'old': "Person's leg and bicycle partially visible.",
        'new': 'The partially visible person on the left.',
    },
    6506: {
        'filename': 'FLIR_04584.jpg',
        'old': "White car's rear tire and fender visible on the right.",
        'new': 'The rightmost white car with its rear partially occluded.',
    },
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def replace_exact_strings(value, replacements, counts):
    if isinstance(value, str):
        if value in replacements:
            counts[value] += 1
            return replacements[value]
        return value
    if isinstance(value, list):
        return [replace_exact_strings(item, replacements, counts)
                for item in value]
    if isinstance(value, dict):
        return {key: replace_exact_strings(item, replacements, counts)
                for key, item in value.items()}
    return value


def main():
    ann_dir = ROOT / 'data/FLIR/annotations'
    vg_path = ann_dir / 'rgbtvg_flir_train_vg.json'
    seg_path = ann_dir / 'rgbtvg_flir_train_seg.jsonl'
    coco_path = ann_dir / 'rgbtvg_flir_train.json'
    referent_dir = ROOT / 'analysis_demo/referent_spans'
    fix_path = referent_dir / 'flir_train_referent_spans_low_fix.json'
    manual_path = referent_dir / 'flir_train_referent_spans_manual_review.json'

    vg_rows = load_jsonl(vg_path)
    seg_rows = load_jsonl(seg_path)
    if len(vg_rows) != 7000 or len(seg_rows) != 7000:
        raise RuntimeError('Unexpected FLIR train annotation size.')

    for index, revision in REVISIONS.items():
        vg = vg_rows[index]
        seg = seg_rows[index]
        if vg['filename'] != revision['filename']:
            raise RuntimeError(f'VG filename mismatch at {index}.')
        if seg['filename'] != revision['filename']:
            raise RuntimeError(f'SEG filename mismatch at {index}.')
        if vg['grounding']['caption'] not in {revision['old'], revision['new']}:
            raise RuntimeError(f'Unexpected VG caption at {index}.')
        if seg['caption'] not in {revision['old'], revision['new']}:
            raise RuntimeError(f'Unexpected SEG caption at {index}.')
        caption = revision['new']
        vg['grounding']['caption'] = caption
        region = vg['grounding']['regions'][0]
        region['phrase'] = caption
        region['tokens_positive'] = [[0, len(caption)]]
        seg['caption'] = caption

    coco = json.loads(coco_path.read_text())
    replacements = {item['old']: item['new'] for item in REVISIONS.values()}
    replacement_counts = Counter()
    coco = replace_exact_strings(coco, replacements, replacement_counts)
    if any(replacement_counts[old] not in {0, 1} for old in replacements):
        raise RuntimeError(
            f'Unexpected COCO replacement counts: {dict(replacement_counts)}')

    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'huggingface-hub/transformers/bert-base-uncased',
        local_files_only=True, use_fast=True)
    fix = json.loads(fix_path.read_text())
    by_index = {item['sample_index']: item for item in fix['samples']}
    extracted = {}
    for index, revision in REVISIONS.items():
        fresh = extractor.analyze_sample(
            index, vg_rows[index], nlp(revision['new']), tokenizer)
        extracted[index] = fresh
        target = by_index[index]
        bucket_confidence = target['confidence']
        target.clear()
        target.update(fresh)
        # Keep membership in the original low-confidence review partition,
        # while recording the confidence obtained from the revised caption.
        target['reextracted_confidence'] = fresh['confidence']
        target['confidence'] = bucket_confidence
        spans = fresh['referent_positive_spans']
        texts = fresh['referent_positive_texts']
        target['original_referent_positive_spans'] = spans
        target['original_referent_positive_texts'] = texts
        target['fixed_referent_positive_spans'] = spans
        target['fixed_referent_positive_texts'] = texts
        target['fixed_subject_phrase_char_span'] = [
            spans[0][0], spans[-1][1]]
        target['fixed_subject_phrase'] = revision['new'][
            spans[0][0]:spans[-1][1]]
        target['fixed_bert_span_evidence'] = fresh['bert_span_evidence']
        target['review_status'] = 'corrected'
        target['review_note'] = (
            'Corrected by human caption revision; subject spans were '
            're-extracted with spaCy and aligned with the BERT tokenizer.')

    statuses = Counter(item['review_status'] for item in fix['samples'])
    fix['semantic_review']['status_counts'] = dict(statuses)
    fix['semantic_review']['all_samples_semantically_resolved'] = (
        statuses['needs_manual_review'] == 0)

    manual = {
        'sample_count': 0,
        'instructions': (
            'All previously ambiguous FLIR low-confidence captions have been '
            'resolved by human-provided caption revisions.'),
        'samples': [],
    }

    write_jsonl(vg_path, vg_rows)
    write_jsonl(seg_path, seg_rows)
    coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2) + '\n')
    fix_path.write_text(json.dumps(fix, ensure_ascii=False, indent=2) + '\n')
    manual_path.write_text(json.dumps(manual, ensure_ascii=False, indent=2) + '\n')

    print(json.dumps({
        'updated_indices': sorted(REVISIONS),
        'captions': {str(i): item['new'] for i, item in REVISIONS.items()},
        'subjects': {
            str(i): extracted[i]['referent_positive_texts']
            for i in sorted(extracted)
        },
        'reextracted_confidences': {
            str(i): extracted[i]['confidence'] for i in sorted(extracted)
        },
        'low_fix_status_counts': dict(statuses),
        'coco_replacement_counts': dict(replacement_counts),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
