#!/usr/bin/env python3
"""Review and repair FLIR medium/low referent spans.

The FLIR captions normally introduce the grounded object at sentence start.
This reviewer repairs later noun chunks selected by spaCy, removes predicate
verbs accidentally attached to noun phrases, and retains genuinely ambiguous
coordinated-object captions for human review.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import spacy
from transformers import AutoTokenizer

import review_mfad_referent_spans as common


TRAILING_PREDICATES = {
    'appears', 'approaches', 'cruises', 'drives', 'glides', 'glows', 'leans',
    'moves', 'rides', 'speeds', 'travels', 'walks',
}

# These captions name two coordinate targets rather than one unambiguous
# object.  Their current proposal is retained and explicitly left unresolved.
MANUAL_REVIEW_REASONS = {
    895: "The caption jointly refers to the car's rear tire and fender.",
    6383: "The caption jointly refers to a person's leg and a bicycle.",
    6506: "The caption jointly refers to the car's rear tire and fender.",
}

# Corrections whose coordination or hyphenation cannot be recovered reliably
# by the generic sentence-initial-head rule.
EXACT_TEXT_CORRECTIONS = {
    4586: ['small', 'light-colored car'],
    5732: ['vintage beige', 'wood-paneled van'],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', type=Path,
        default=Path('/home/lbw/mmdetection/analysis_demo/referent_spans'))
    parser.add_argument(
        '--tokenizer-dir',
        default=('/home/lbw/mmdetection/huggingface-hub/transformers/'
                 'bert-base-uncased/'))
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument(
        '--manual-report', type=Path,
        default=Path('/home/lbw/mmdetection/analysis_demo/referent_spans/'
                     'flir_train_referent_spans_manual_review.json'))
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


def trim_trailing_predicate(sample):
    """Remove a predicate verb appended to an otherwise valid span."""
    texts = list(sample['referent_positive_texts'])
    spans = [list(span) for span in sample['referent_positive_spans']]
    if not texts:
        return spans
    words = texts[-1].split()
    if not words or words[-1].lower() not in TRAILING_PREDICATES:
        return spans
    predicate = words[-1]
    new_end = spans[-1][1] - len(predicate)
    while new_end > spans[-1][0] and sample['caption'][new_end - 1].isspace():
        new_end -= 1
    if new_end <= spans[-1][0]:
        return spans[:-1]
    spans[-1][1] = new_end
    return spans


def initial_object_spans(sample, doc):
    head = common.select_initial_head(doc)
    if head is None:
        return None
    start = 1 if doc[0].text.lower() in common.ARTICLES else 0
    return common.tokens_to_spans(doc[start:head.i + 1], sample['caption'], doc)


def review_sample(sample, doc, tokenizer):
    result = dict(sample)
    index = result['sample_index']
    caption = result['caption']
    original_spans = [list(span) for span in result['referent_positive_spans']]
    original_texts = list(result['referent_positive_texts'])
    flags = set(result.get('review_flags', []))
    reason = MANUAL_REVIEW_REASONS.get(index)

    if reason is not None:
        fixed_spans = original_spans
        status = 'needs_manual_review'
        note = f'Needs manual review: {reason}'
    elif index in EXACT_TEXT_CORRECTIONS:
        fixed_spans = locate_ordered_texts(
            caption, EXACT_TEXT_CORRECTIONS[index])
        status = 'corrected'
        note = ('Corrected after semantic review of coordination and '
                'hyphenation in the sentence-initial referent phrase.')
    elif {'non_initial_subject_phrase', 'suspicious_subject_head'} & flags:
        fixed_spans = initial_object_spans(result, doc)
        if not fixed_spans:
            fixed_spans = original_spans
            reason = 'No reliable sentence-initial object head was found.'
            status = 'needs_manual_review'
            note = f'Needs manual review: {reason}'
        else:
            status = 'corrected' if fixed_spans != original_spans else 'confirmed'
            note = (
                'Corrected: replaced a predicate or later scene noun phrase '
                'with the sentence-initial object phrase.'
                if status == 'corrected' else
                'Confirmed: the existing spans identify the main referent.')
    else:
        fixed_spans = trim_trailing_predicate(result)
        status = 'corrected' if fixed_spans != original_spans else 'confirmed'
        note = (
            'Corrected: removed a predicate verb attached to the referent '
            'noun phrase by the spaCy fallback.'
            if status == 'corrected' else
            'Confirmed: the existing spans identify the main referent.')

    fixed_texts = [caption[start:end] for start, end in fixed_spans]
    result['original_referent_positive_spans'] = original_spans
    result['original_referent_positive_texts'] = original_texts
    result['fixed_referent_positive_spans'] = fixed_spans
    result['fixed_referent_positive_texts'] = fixed_texts
    result['fixed_subject_phrase_char_span'] = [
        fixed_spans[0][0], fixed_spans[-1][1]]
    result['fixed_subject_phrase'] = caption[
        fixed_spans[0][0]:fixed_spans[-1][1]]
    result['fixed_bert_span_evidence'] = common.bert_evidence(
        tokenizer, caption, fixed_spans)
    result['review_status'] = status
    result['review_note'] = note
    if reason is not None:
        result['manual_review_reason'] = reason
    return result


def write_manual_report(path, samples):
    records = [{
        'sample_index': item['sample_index'],
        'source_line': item['source_line'],
        'confidence': item['confidence'],
        'filename': item['filename'],
        'bbox_xyxy': item['bbox_xyxy'],
        'caption': item['caption'],
        'current_proposed_spans': item['fixed_referent_positive_spans'],
        'current_proposed_texts': item['fixed_referent_positive_texts'],
        'reason': item['manual_review_reason'],
    } for item in samples]
    payload = {
        'sample_count': len(records),
        'instructions': (
            'These captions do not identify one unambiguous textual referent. '
            'Their fix-file entries retain the original proposal and use '
            'review_status needs_manual_review.'),
        'samples': records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def main():
    args = parse_args()
    # Extend the concrete-head vocabulary for FLIR spelling and object types.
    common.MANUAL_REVIEW_REASONS = {}
    common.OBJECT_HEADS.update({
        'box', 'handlebar', 'limousine', 'motocycle', 'unicycle',
    })
    nlp = spacy.load(args.spacy_model, disable=['ner'])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for char_to_token.')

    summaries = {}
    manual = []
    for confidence in ('medium', 'low'):
        source = args.input_dir / f'flir_train_referent_spans_{confidence}.json'
        destination = (
            args.input_dir /
            f'flir_train_referent_spans_{confidence}_fix.json')
        payload = json.loads(source.read_text())
        samples = payload['samples']
        docs = nlp.pipe(
            [item['caption'] for item in samples],
            batch_size=args.batch_size)
        reviewed = [
            review_sample(item, doc, tokenizer)
            for item, doc in zip(samples, docs)
        ]
        if len(reviewed) != len(samples):
            raise RuntimeError('spaCy returned an unexpected number of docs.')
        counts = Counter(item['review_status'] for item in reviewed)
        manual.extend(
            item for item in reviewed
            if item['review_status'] == 'needs_manual_review')
        payload['samples'] = reviewed
        payload[f'source_{confidence}_confidence_file'] = str(source)
        payload['semantic_review'] = {
            'reviewed_count': len(reviewed),
            'status_counts': dict(counts),
            'automatic_rules_applied': True,
            'all_samples_structurally_reviewed': True,
            'all_samples_semantically_resolved': not counts[
                'needs_manual_review'],
            'training_annotation_modified': False,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        summaries[confidence] = {
            'input': str(source),
            'output': str(destination),
            'sample_count': len(reviewed),
            'status_counts': dict(counts),
        }

    manual.sort(key=lambda item: item['sample_index'])
    write_manual_report(args.manual_report, manual)
    print(json.dumps({
        'summaries': summaries,
        'manual_report': str(args.manual_report),
        'manual_review_count': len(manual),
    }, indent=2))


if __name__ == '__main__':
    main()
