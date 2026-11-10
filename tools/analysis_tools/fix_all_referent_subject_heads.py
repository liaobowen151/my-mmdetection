#!/usr/bin/env python3
"""Synchronize subject-head fields with final referent spans in all catalogs."""

import json
import sys
from collections import Counter
from pathlib import Path

import spacy
from transformers import AutoTokenizer


ROOT = Path('/home/lbw/mmdetection')
REFERENT_DIR = ROOT / 'analysis_demo/referent_spans'
sys.path.insert(0, str(ROOT / 'tools/analysis_tools'))
import review_mfad_referent_spans as common  # noqa: E402


# Parser errors can label these finite predicates as nominal heads.
PREDICATE_HEADS = {
    'appears', 'approaches', 'cruises', 'displays', 'drives', 'glides',
    'glows', 'leans', 'moves', 'rides', 'speeds', 'travels', 'walks',
}

# Concrete terminal heads used across M3FD, MFAD, and FLIR.  Modifier-like
# words such as "pickup", "semi", "transit", and colors are deliberately not
# included; the desired heads are truck, bus, car, etc.
SEMANTIC_HEADS = set(common.OBJECT_HEADS) | {
    'handlebar', 'jeep', 'limousine', 'motocycle', 'railing', 'stalk',
    'unicycle', 'volkswagen',
}
SEMANTIC_HEADS -= {'display', 'light'}

# A previous MFAD repair rule treated "light" and hyphenated compounds as
# terminal heads, truncating these ten final referent phrases.  Repairing the
# span is necessary before a correct subject head can be placed inside it.
MFAD_REFERENT_TEXT_CORRECTIONS = {
    1275: ['light blue sedan'],
    2409: ['small red fire hydrant'],
    3689: ['light blue sedan'],
    8070: ['light-colored sedan'],
    10905: ['light green sedan'],
    12017: ['light green sedan'],
    12604: ['light blue sedan'],
    13034: ['light blue sedan'],
    15013: ['large red semi-truck'],
    15555: ['light blue sedan'],
}


def locate_texts(caption, texts):
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


def selected_tokens(sample, doc):
    spans = sample['referent_positive_spans']
    return [
        token for token in doc
        if any(start <= token.idx
               and token.idx + len(token.text) <= end
               for start, end in spans)
    ]


def is_semantic_head(token):
    return (token.text.lower() in SEMANTIC_HEADS
            or token.lemma_.lower() in SEMANTIC_HEADS)


def choose_head(sample, doc):
    tokens = selected_tokens(sample, doc)
    if not tokens:
        raise ValueError(
            f'No tokens in referent spans for sample {sample["sample_index"]}')
    candidates = [token for token in tokens if is_semantic_head(token)]
    if candidates:
        chosen = candidates[0]
        # Advance across a compound nominal (Mazda sedan, bicycle rider,
        # car-like vehicle), but never across a relation or predicate as in
        # "person riding a bicycle".
        for candidate in candidates[1:]:
            between = doc[chosen.i + 1:candidate.i]
            if any(token.pos_ in {'VERB', 'AUX', 'ADP'} for token in between):
                break
            if any(token.text.lower() in {'and', 'or'} for token in between):
                break
            chosen = candidate
        return chosen

    nominal = [
        token for token in tokens
        if token.pos_ in {'NOUN', 'PROPN', 'PRON'}
    ]
    token_ids = {token.i for token in tokens}
    roots = [
        token for token in nominal
        if token.head.i not in token_ids or token.head.i == token.i
    ]
    preferred = [
        token for token in roots
        if token.dep_ in {'nsubj', 'nsubjpass', 'ROOT'}
    ]
    choices = preferred or roots or nominal or tokens
    return choices[0]


def head_is_inside(sample):
    span = sample.get('subject_head_char_span')
    if not span:
        return False
    return any(
        start <= span[0] and span[1] <= end
        for start, end in sample['referent_positive_spans'])


def main():
    paths = sorted(REFERENT_DIR.glob('*_train_referent_spans_all.json'))
    if {path.name for path in paths} != {
            'flir_train_referent_spans_all.json',
            'm3fd_train_referent_spans_all.json',
            'mfad_train_referent_spans_all.json'}:
        raise RuntimeError('Expected exactly the three dataset all catalogs.')

    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'huggingface-hub/transformers/bert-base-uncased',
        local_files_only=True, use_fast=True)
    summary = {}
    for path in paths:
        payload = json.loads(path.read_text())
        samples = payload['samples']
        span_corrected = 0
        if path.name == 'mfad_train_referent_spans_all.json':
            by_index = {sample['sample_index']: sample for sample in samples}
            for index, texts in MFAD_REFERENT_TEXT_CORRECTIONS.items():
                sample = by_index[index]
                spans = locate_texts(sample['caption'], texts)
                sample['referent_positive_spans'] = spans
                sample['referent_positive_texts'] = texts
                sample['bert_span_evidence'] = common.bert_evidence(
                    tokenizer, sample['caption'], spans)
                sample['subject_phrase_char_span'] = [
                    spans[0][0], spans[-1][1]]
                sample['subject_phrase'] = sample['caption'][
                    spans[0][0]:spans[-1][1]]
                span_corrected += 1
        changed = []
        docs = nlp.pipe(
            [sample['caption'] for sample in samples], batch_size=256)
        for sample, doc in zip(samples, docs):
            old_head = sample.get('subject_head')
            needs_fix = (
                not head_is_inside(sample)
                or str(old_head).lower() in PREDICATE_HEADS
                or (path.name == 'mfad_train_referent_spans_all.json'
                    and sample['sample_index']
                    in MFAD_REFERENT_TEXT_CORRECTIONS))
            if not needs_fix:
                continue
            head = choose_head(sample, doc)
            old_span = sample.get('subject_head_char_span')
            new_span = [head.idx, head.idx + len(head.text)]
            sample['subject_head'] = head.text
            sample['subject_head_lemma'] = head.lemma_
            sample['subject_head_pos'] = head.pos_
            sample['subject_head_dependency'] = head.dep_
            sample['subject_head_char_span'] = new_span
            changed.append({
                'sample_index': sample['sample_index'],
                'old_head': old_head,
                'old_span': old_span,
                'new_head': head.text,
                'new_span': new_span,
            })

        errors = []
        for sample in samples:
            head = sample.get('subject_head')
            span = sample.get('subject_head_char_span')
            if not head or not span:
                errors.append((sample['sample_index'], 'missing'))
                continue
            start, end = span
            if sample['caption'][start:end] != head:
                errors.append((sample['sample_index'], 'text_mismatch'))
            if not head_is_inside(sample):
                errors.append((sample['sample_index'], 'outside_referent'))
            if str(head).lower() in PREDICATE_HEADS:
                errors.append((sample['sample_index'], 'predicate_head'))
        if errors:
            raise RuntimeError(f'{path.name} validation errors: {errors[:20]}')

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        summary[path.name] = {
            'sample_count': len(samples),
            'corrected_count': len(changed),
            'referent_span_corrected_count': span_corrected,
            'new_head_counts': dict(Counter(
                item['new_head'].lower() for item in changed)),
            'validation_errors': len(errors),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
