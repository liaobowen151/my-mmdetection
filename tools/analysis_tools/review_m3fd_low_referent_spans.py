"""Apply a semantic review to the current M3FD low-confidence spans."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


EXPECTED_LOW_INDICES = {
    75, 99, 112, 130, 139, 144, 150, 181, 209, 273, 328, 428, 471,
    473, 478, 488, 623, 624, 626, 627, 630, 694, 700, 714, 888, 983,
    988, 994, 1035, 1144, 1184, 1276, 1280, 1299, 1305, 1324, 1361,
    1380, 1438, 1492, 1496, 1522, 1543, 1588, 1606, 1615, 1617, 1671,
    1676, 1837, 1856, 1900, 1901, 1927, 1980, 1985, 1987, 1992, 2086,
    2127, 2149, 2182, 2237, 2289, 2303, 2346, 2418, 2476, 2479, 2503,
    2568, 2577, 2660, 2670, 2726, 2733, 2772, 2793, 2836, 2922, 2929,
    3055, 3065, 3076, 3086, 3123, 3148, 3157, 3183, 3189, 3305, 3353,
    3510, 3542, 3570,
}

# Each list contains exact, ordered substrings from the caption. Separate
# entries intentionally omit punctuation and coordinators such as "and".
CORRECTIONS = {
    75: ['Dark', 'partially obscured car'],
    112: ['faint', 'obscured figure'],
    130: ['sleek silver sedan'],
    150: ['bright red taillight'],
    209: ['Red', 'white van'],
    328: ['sleek silver sedan'],
    471: ['sleek black Honda sedan'],
    478: ['white compact car'],
    624: ['sleek silver sedan'],
    630: ['large brown semi-truck'],
    700: ['dark-colored sedan'],
    888: ['small', 'light-colored car'],
    983: ['white compact car'],
    1035: ['dark-colored sedan'],
    1184: ['sleek silver sedan'],
    1280: ['Green', 'red bus'],
    1361: ['dark-colored sedan'],
    1438: ['dark-colored sedan'],
    1496: ['dark-colored sedan'],
    1522: ['small', 'bright motorbike headlight'],
    1588: ['faint', 'obscured figure'],
    1615: ['white compact car'],
    1617: ['silver sedan'],
    1676: ['silver compact car'],
    1856: ['dark-colored sedan'],
    1901: ['solitary figure'],
    1927: ['white compact car'],
    1980: ['sleek silver sedan'],
    2127: ['white sedan'],
    2289: ['sleek silver sedan'],
    2476: ['white car'],
    2660: ['dark-colored sedan'],
    2670: ['faint', 'ghostly figure'],
    2922: ['sleek silver sedan'],
    3157: ['silver sedan'],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        default=(
            '/home/lbw/mmdetection/analysis_demo/'
            'm3fd_train_referent_spans_low.json'))
    parser.add_argument(
        '--output',
        default=(
            '/home/lbw/mmdetection/analysis_demo/'
            'm3fd_train_referent_spans_low_fix.json'))
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


def correction_note(sample):
    flags = set(sample['review_flags'])
    if 'entity_coordination' in flags:
        return ('Corrected: recovered coordinated modifiers omitted by the '
                'spaCy fallback while excluding the coordinator token.')
    if 'suspicious_subject_head' in flags:
        return ('Corrected: replaced a parser-selected predicate or '
                'non-referent head with the sentence-initial referent noun '
                'phrase.')
    return 'Corrected after semantic review of the complete caption.'


def main():
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    payload = json.loads(source.read_text())
    samples = payload['samples']
    actual_indices = {sample['sample_index'] for sample in samples}
    if actual_indices != EXPECTED_LOW_INDICES:
        missing = sorted(EXPECTED_LOW_INDICES - actual_indices)
        unexpected = sorted(actual_indices - EXPECTED_LOW_INDICES)
        raise RuntimeError(
            f'Low-confidence input changed; missing={missing}, '
            f'unexpected={unexpected}')

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
            correction_note(sample) if status == 'corrected'
            else ('Confirmed: the existing positive spans already identify '
                  'the main referent noun phrase.'))
        status_counts[status] += 1

    payload['source_low_confidence_file'] = str(source)
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
