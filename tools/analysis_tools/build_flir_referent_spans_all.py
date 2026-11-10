#!/usr/bin/env python3
"""Merge reviewed FLIR high/medium/low referent-span catalogs."""

import copy
import json
from pathlib import Path


ROOT = Path('/home/lbw/mmdetection')
REFERENT_DIR = ROOT / 'analysis_demo/referent_spans'
SOURCE_ANN = ROOT / 'data/FLIR/annotations/rgbtvg_flir_train_vg.json'
OUTPUT = REFERENT_DIR / 'flir_train_referent_spans_all.json'


def main():
    inputs = {
        'high': REFERENT_DIR / 'flir_train_referent_spans_high.json',
        'medium_fix': (
            REFERENT_DIR / 'flir_train_referent_spans_medium_fix.json'),
        'low_fix': REFERENT_DIR / 'flir_train_referent_spans_low_fix.json',
    }
    parts = {
        name: json.loads(path.read_text())['samples']
        for name, path in inputs.items()
    }
    samples = [
        copy.deepcopy(sample)
        for name in ('high', 'medium_fix', 'low_fix')
        for sample in parts[name]
    ]
    for sample in samples:
        if 'fixed_referent_positive_spans' in sample:
            sample['referent_positive_spans'] = copy.deepcopy(
                sample['fixed_referent_positive_spans'])
            sample['referent_positive_texts'] = list(
                sample['fixed_referent_positive_texts'])
            if 'fixed_bert_span_evidence' in sample:
                sample['bert_span_evidence'] = copy.deepcopy(
                    sample['fixed_bert_span_evidence'])
            if 'fixed_subject_phrase_char_span' in sample:
                sample['subject_phrase_char_span'] = copy.deepcopy(
                    sample['fixed_subject_phrase_char_span'])
                sample['subject_phrase'] = sample['fixed_subject_phrase']

    samples.sort(key=lambda item: item['sample_index'])
    indices = [item['sample_index'] for item in samples]
    expected = list(range(7000))
    if indices != expected:
        duplicates = sorted({i for i in indices if indices.count(i) > 1})
        missing = sorted(set(expected) - set(indices))
        raise RuntimeError(
            f'Invalid merged indices: missing={missing[:20]}, '
            f'duplicates={duplicates[:20]}')

    source = [
        json.loads(line) for line in SOURCE_ANN.read_text().splitlines()
        if line.strip()
    ]
    if len(source) != len(samples):
        raise RuntimeError(
            f'Population mismatch: source={len(source)}, merged={len(samples)}')
    errors = []
    for sample, annotation in zip(samples, source):
        grounding = annotation['grounding']
        if sample['filename'] != annotation['filename']:
            errors.append((sample['sample_index'], 'filename'))
        if sample['caption'] != grounding['caption']:
            errors.append((sample['sample_index'], 'caption'))
        if sample['bbox_xyxy'] != grounding['regions'][0]['bbox']:
            errors.append((sample['sample_index'], 'bbox'))
        spans = sample['referent_positive_spans']
        texts = sample['referent_positive_texts']
        if not spans or len(spans) != len(texts):
            errors.append((sample['sample_index'], 'span_count'))
        for (start, end), text in zip(spans, texts):
            if sample['caption'][start:end] != text:
                errors.append((sample['sample_index'], 'span_text'))
    if errors:
        raise RuntimeError(f'Merged validation failed: {errors[:20]}')

    OUTPUT.write_text(json.dumps({'samples': samples}, ensure_ascii=False,
                                 indent=2) + '\n')
    print(json.dumps({
        'output': str(OUTPUT),
        'counts': {name: len(items) for name, items in parts.items()},
        'merged_count': len(samples),
        'validation_errors': len(errors),
    }, indent=2))


if __name__ == '__main__':
    main()
