#!/usr/bin/env python3
"""Build the M3FD ODVG training annotation with referent-only positives."""

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

    assert len(source_samples) == len(referent_samples) == 3604
    assert [x['sample_index'] for x in referent_samples] == list(range(3604))

    output_samples = copy.deepcopy(source_samples)
    changed = 0
    multi_span = 0
    positive_characters = 0

    for index, (source, referent, output) in enumerate(
            zip(source_samples, referent_samples, output_samples)):
        assert referent['sample_index'] == index
        assert referent['source_line'] == index + 1
        assert source['filename'] == referent['filename']
        assert source['grounding']['caption'] == referent['caption']
        assert len(source['grounding']['regions']) == 1
        assert source['grounding']['regions'][0]['bbox'] == referent['bbox_xyxy']

        caption = source['grounding']['caption']
        spans = referent['referent_positive_spans']
        texts = referent['referent_positive_texts']
        assert [caption[start:end] for start, end in spans] == texts
        assert all(0 <= start < end <= len(caption) for start, end in spans)

        old_spans = output['grounding']['regions'][0]['tokens_positive']
        output['grounding']['regions'][0]['tokens_positive'] = spans
        changed += old_spans != spans
        multi_span += len(spans) > 1
        positive_characters += sum(end - start for start, end in spans)

        assert without_tokens_positive(source) == without_tokens_positive(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        for sample in output_samples:
            file.write(json.dumps(sample, ensure_ascii=False) + '\n')
    temporary.replace(args.output)

    reloaded = load_jsonl(args.output)
    assert len(reloaded) == 3604
    assert all(
        without_tokens_positive(source) == without_tokens_positive(output)
        for source, output in zip(source_samples, reloaded))
    assert all(
        output['grounding']['regions'][0]['tokens_positive'] ==
        referent['referent_positive_spans']
        for output, referent in zip(reloaded, referent_samples))

    print(f'output={args.output}')
    print(f'samples={len(reloaded)}')
    print(f'changed_tokens_positive={changed}')
    print(f'multi_span_samples={multi_span}')
    print(f'average_positive_characters={positive_characters / len(reloaded):.4f}')


if __name__ == '__main__':
    main()
