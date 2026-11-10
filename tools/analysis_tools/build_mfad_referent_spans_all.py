#!/usr/bin/env python3
"""Merge and validate MFAD referent-span confidence files.

The high-confidence file is used directly.  Medium/low files may either be
the extractor outputs or reviewed ``*_fix.json`` files.  When a reviewed
sample contains ``fixed_referent_positive_*`` fields, those values replace
the automatic extraction in the merged catalog.
"""

import argparse
import json
from pathlib import Path


CONFIDENCES = ('high', 'medium', 'low')
EXPECTED_SAMPLES = 16000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', type=Path,
        default=Path('/home/lbw/mmdetection/analysis_demo/referent_spans'))
    parser.add_argument('--prefix', default='mfad_train_referent_spans')
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def input_path(input_dir, prefix, confidence):
    reviewed = input_dir / f'{prefix}_{confidence}_fix.json'
    automatic = input_dir / f'{prefix}_{confidence}.json'
    if confidence != 'high' and reviewed.exists():
        return reviewed
    return automatic


def normalized_sample(sample):
    result = dict(sample)
    fixed_spans = result.get('fixed_referent_positive_spans')
    fixed_texts = result.get('fixed_referent_positive_texts')
    if fixed_spans is not None or fixed_texts is not None:
        if fixed_spans is None or fixed_texts is None:
            raise ValueError(
                f'Incomplete fixed fields for sample {sample["sample_index"]}')
        result['referent_positive_spans'] = fixed_spans
        result['referent_positive_texts'] = fixed_texts
    return result


def validate_sample(sample):
    caption = sample['caption']
    spans = sample['referent_positive_spans']
    texts = sample['referent_positive_texts']
    if not spans:
        raise ValueError(f'Empty span for sample {sample["sample_index"]}')
    if len(spans) != len(texts):
        raise ValueError(
            f'Span/text count mismatch for sample {sample["sample_index"]}')
    actual = []
    previous_end = -1
    for start, end in spans:
        if not 0 <= start < end <= len(caption):
            raise ValueError(
                f'Invalid span {[start, end]} for sample '
                f'{sample["sample_index"]}')
        if start < previous_end:
            raise ValueError(
                f'Overlapping/out-of-order spans for sample '
                f'{sample["sample_index"]}')
        actual.append(caption[start:end])
        previous_end = end
    if actual != texts:
        raise ValueError(
            f'Span text mismatch for sample {sample["sample_index"]}: '
            f'{actual!r} != {texts!r}')


def main():
    args = parse_args()
    output = args.output or args.input_dir / f'{args.prefix}_all.json'
    samples = []
    sources = {}
    for confidence in CONFIDENCES:
        source = input_path(args.input_dir, args.prefix, confidence)
        if not source.exists():
            raise FileNotFoundError(source)
        payload = json.loads(source.read_text(encoding='utf-8'))
        if payload.get('confidence') != confidence:
            raise ValueError(
                f'{source} has confidence={payload.get("confidence")!r}')
        if payload.get('population_size') != EXPECTED_SAMPLES:
            raise ValueError(
                f'{source} has population_size='
                f'{payload.get("population_size")!r}')
        sources[confidence] = str(source)
        samples.extend(normalized_sample(item) for item in payload['samples'])

    samples.sort(key=lambda item: item['sample_index'])
    indices = [sample['sample_index'] for sample in samples]
    if indices != list(range(EXPECTED_SAMPLES)):
        raise ValueError('Merged sample indices are missing or duplicated.')
    for sample in samples:
        validate_sample(sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({'samples': samples}, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps({
        'output': str(output),
        'sample_count': len(samples),
        'sources': sources,
    }, indent=2))


if __name__ == '__main__':
    main()
