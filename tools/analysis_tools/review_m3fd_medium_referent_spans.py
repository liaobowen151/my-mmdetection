"""Apply a semantic review to the current M3FD medium-confidence spans."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


# Exact ordered substrings. Separate entries exclude commas and coordinators.
CORRECTIONS = {
    15: ['faint', 'indistinct figure'],
    18: ['person'],
    36: ['gray sedan'],
    60: ['Motorcyclist'],
    113: ['silver sedan'],
    124: ['faint', 'indistinct figure'],
    180: ['white sedan'],
    192: ['silver sedan'],
    291: ['silver sedan'],
    346: ['Small white bus'],
    362: ['silver sedan'],
    367: ['white van'],
    455: ['white sedan'],
    483: ['large white', 'black double-decker bus'],
    505: ['white sedan'],
    518: ['green', 'white bus'],
    559: ['green', 'white bus'],
    828: ['white sedan'],
    842: ['Partially obscured dark-colored car'],
    861: ['silver sedan'],
    1001: ['silver sedan'],
    1013: ['Faint', 'indistinct figure'],
    1046: ['silver sedan'],
    1062: ['silver sedan'],
    1091: ['white sedan'],
    1105: ['Small white truck'],
    1140: ['person'],
    1176: ['green', 'white bus'],
    1210: ['Small white bus'],
    1225: ['Tall', 'slender plant stalks'],
    1273: ['white sedan'],
    1334: ['Tall', 'slender plant stalks'],
    1500: ['white minivan'],
    1581: ['small', 'indistinct figure'],
    1690: ['green bus'],
    1698: ['person'],
    1751: ['silver sedan'],
    1792: ['Motorcyclist'],
    1847: ['Yellow metal railing'],
    1851: ['red bus'],
    1939: ['silver minivan'],
    1976: ['Small white bus'],
    2018: ['white sedan'],
    2060: ['silver sedan'],
    2110: ['white sedan'],
    2138: ['silver sedan'],
    2139: ['Motorcyclist'],
    2310: ['faint', 'indistinct figure'],
    2319: ['silver sedan'],
    2332: ['Blue flatbed truck'],
    2386: ['large', 'dark-colored bus'],
    2546: ['bright', 'upright street lamp'],
    2584: ['green', 'white bus'],
    2629: ['red bus'],
    2736: ['distant red bus'],
    2802: ['distant white truck'],
    2803: ['white sedan'],
    2828: ['white sedan'],
    2857: ['distant', 'partially obscured bus'],
    2906: ['Person'],
    3067: ['green', 'white bus'],
    3068: ['silver sedan'],
    3070: ['Motorcyclist'],
    3096: ['distant', 'glowing orange lamp'],
    3126: ['green', 'white bus'],
    3262: ['person'],
    3348: ['Tall', 'slender figure'],
    3379: ['red bus'],
    3436: ['solitary figure'],
    3536: ['white sedan'],
    3576: ['silver sedan'],
    3581: ['silver sedan'],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        default=(
            '/home/lbw/mmdetection/analysis_demo/'
            'm3fd_train_referent_spans_medium.json'))
    parser.add_argument(
        '--output',
        default=(
            '/home/lbw/mmdetection/analysis_demo/'
            'm3fd_train_referent_spans_medium_fix.json'))
    parser.add_argument(
        '--tokenizer-dir',
        default=(
            '/home/lbw/mmdetection/huggingface-hub/transformers/'
            'bert-base-uncased/'))
    return parser.parse_args()


def locate_ordered_texts(caption, texts):
    spans = []
    cursor = 0
    for text in texts:
        start = caption.find(text, cursor)
        if start < 0:
            raise ValueError(f'Could not locate {text!r} in {caption!r}')
        end = start + len(text)
        spans.append([start, end])
        cursor = end
    return spans


def bert_evidence(tokenizer, caption, spans):
    encoding = tokenizer(
        caption,
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        return_offsets_mapping=True)
    evidence = []
    for start, end in spans:
        begin_token = encoding.char_to_token(start)
        end_token = encoding.char_to_token(end - 1)
        if begin_token is None or end_token is None:
            raise ValueError(
                f'BERT mapping failed for span {[start, end]} in {caption!r}')
        evidence.append({
            'char_span': [start, end],
            'token_span': [begin_token, end_token + 1],
            'bert_tokens': tokenizer.convert_ids_to_tokens(
                encoding['input_ids'][begin_token:end_token + 1]),
        })
    return evidence


def main():
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    payload = json.loads(source.read_text())
    samples = payload['samples']
    if payload['sample_count'] != 784 or len(samples) != 784:
        raise RuntimeError(
            f'Expected 784 medium samples, found {len(samples)}')
    if len(CORRECTIONS) != 72:
        raise RuntimeError(
            f'Expected 72 reviewed corrections, found {len(CORRECTIONS)}')
    sample_by_index = {sample['sample_index']: sample for sample in samples}
    missing = sorted(set(CORRECTIONS) - set(sample_by_index))
    if missing:
        raise RuntimeError(f'Correction indices missing from input: {missing}')
    flagged_non_initial = {
        sample['sample_index'] for sample in samples
        if 'non_initial_subject_phrase' in sample['review_flags']
    }
    if flagged_non_initial != set(CORRECTIONS):
        raise RuntimeError(
            'The non-initial review set changed: '
            f'missing={sorted(flagged_non_initial - set(CORRECTIONS))}, '
            f'unexpected={sorted(set(CORRECTIONS) - flagged_non_initial)}')

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for char_to_token.')

    status_counts = {'confirmed': 0, 'corrected': 0}
    for sample in samples:
        index = sample['sample_index']
        caption = sample['caption']
        original_spans = sample['referent_positive_spans']
        original_texts = sample['referent_positive_texts']
        fixed_texts = CORRECTIONS.get(index, original_texts)
        fixed_spans = locate_ordered_texts(caption, fixed_texts)
        status = 'corrected' if index in CORRECTIONS else 'confirmed'

        sample['original_referent_positive_spans'] = original_spans
        sample['original_referent_positive_texts'] = original_texts
        sample['fixed_referent_positive_spans'] = fixed_spans
        sample['fixed_referent_positive_texts'] = fixed_texts
        sample['fixed_subject_phrase_char_span'] = [
            fixed_spans[0][0], fixed_spans[-1][1]
        ]
        sample['fixed_subject_phrase'] = caption[
            fixed_spans[0][0]:fixed_spans[-1][1]]
        sample['fixed_bert_span_evidence'] = bert_evidence(
            tokenizer, caption, fixed_spans)
        sample['review_status'] = status
        sample['review_note'] = (
            ('Corrected: replaced a later noun phrase selected by spaCy with '
             'the sentence-initial referent noun phrase.')
            if status == 'corrected'
            else ('Confirmed: the existing positive spans already identify '
                  'the sentence-initial referent noun phrase.'))
        status_counts[status] += 1

    payload['source_medium_confidence_file'] = str(source)
    payload['semantic_review'] = {
        'reviewed_count': len(samples),
        'status_counts': status_counts,
        'all_samples_reviewed': True,
        'training_annotation_modified': False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'input': str(source),
        'output': str(destination),
        'sample_count': len(samples),
        'status_counts': status_counts,
    }, indent=2))


if __name__ == '__main__':
    main()
