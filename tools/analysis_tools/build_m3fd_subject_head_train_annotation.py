#!/usr/bin/env python3
"""Build M3FD ODVG training annotations with subject-head positives."""

import argparse
import copy
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--referents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def without_tokens_positive(sample):
    sample = copy.deepcopy(sample)
    for region in sample['grounding']['regions']:
        region.pop('tokens_positive')
    return sample


def main():
    args = parse_args()
    source_samples = load_jsonl(args.source)
    with args.referents.open(encoding='utf-8') as file:
        referent_samples = json.load(file)['samples']

    if len(source_samples) != len(referent_samples):
        raise AssertionError(
            f'Length mismatch: source={len(source_samples)}, '
            f'referents={len(referent_samples)}')
    if [x['sample_index'] for x in referent_samples] != list(
            range(len(referent_samples))):
        raise AssertionError('Referent samples are not in sample_index order.')

    output_samples = copy.deepcopy(source_samples)
    changed = 0
    positive_characters = 0

    for index, (source, referent, output) in enumerate(
            zip(source_samples, referent_samples, output_samples)):
        if referent['sample_index'] != index:
            raise AssertionError(f'sample_index mismatch at {index}')
        if referent['source_line'] != index + 1:
            raise AssertionError(f'source_line mismatch at {index}')
        if source['filename'] != referent['filename']:
            raise AssertionError(f'filename mismatch at {index}')
        if source['grounding']['caption'] != referent['caption']:
            raise AssertionError(f'caption mismatch at {index}')
        if len(source['grounding']['regions']) != 1:
            raise AssertionError(f'Expected one region at {index}')
        if source['grounding']['regions'][0]['bbox'] != referent['bbox_xyxy']:
            raise AssertionError(f'bbox mismatch at {index}')

        caption = source['grounding']['caption']
        start, end = referent['subject_head_char_span']
        subject_head = referent['subject_head']
        if not 0 <= start < end <= len(caption):
            raise AssertionError(f'Invalid subject-head span at {index}')
        if caption[start:end] != subject_head:
            raise AssertionError(
                f'Subject-head text mismatch at {index}: '
                f'{caption[start:end]!r} != {subject_head!r}')

        spans = [[start, end]]
        old_spans = output['grounding']['regions'][0]['tokens_positive']
        output['grounding']['regions'][0]['tokens_positive'] = spans
        changed += old_spans != spans
        positive_characters += end - start

        if without_tokens_positive(source) != without_tokens_positive(output):
            raise AssertionError(f'Non-positive fields changed at {index}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for sample in output_samples:
            file.write(json.dumps(sample, ensure_ascii=False) + '\n')
    temporary.replace(args.output)

    reloaded = load_jsonl(args.output)
    if len(reloaded) != len(source_samples):
        raise AssertionError('Reloaded output length changed.')
    for index, (source, referent, output) in enumerate(
            zip(source_samples, referent_samples, reloaded)):
        if without_tokens_positive(source) != without_tokens_positive(output):
            raise AssertionError(
                f'Reloaded non-positive fields changed at {index}')
        expected = [referent['subject_head_char_span']]
        actual = output['grounding']['regions'][0]['tokens_positive']
        if actual != expected:
            raise AssertionError(
                f'Reloaded subject-head span mismatch at {index}')

    print(f'output={args.output}')
    print(f'samples={len(reloaded)}')
    print(f'changed_tokens_positive={changed}')
    print('multi_span_samples=0')
    print(
        'average_positive_characters='
        f'{positive_characters / max(len(reloaded), 1):.4f}')


if __name__ == '__main__':
    main()
