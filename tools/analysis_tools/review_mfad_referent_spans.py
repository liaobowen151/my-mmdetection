#!/usr/bin/env python3
"""Review and repair MFAD medium/low referent spans.

MFAD captions overwhelmingly introduce the referred object at the beginning
of the caption.  spaCy frequently mis-tags predicate words such as ``drives``
as nouns in this dataset, so this reviewer finds the first explicit object
head and rebuilds the initial base noun phrase.  Captions without a real or
unambiguous object are retained but marked for manual review.
"""

import argparse
import json
from pathlib import Path

import spacy
from transformers import AutoTokenizer


ARTICLES = {'a', 'an', 'the'}
COORDINATORS = {'and', 'or'}
LEXICAL_JOINERS = {'-', '‐', '‑', '‒', '–', '—', "'", '’'}

# Concrete heads observed at the beginning of MFAD captions.  Inflections are
# handled with token lemmas; surface forms are also checked because the small
# spaCy model occasionally assigns unusual lemmas/POS tags to brand names.
OBJECT_HEADS = {
    'arrow', 'barrier', 'beam', 'bed', 'benz', 'beetle', 'bicycle', 'building', 'bus',
    'bush', 'cab', 'camry', 'car', 'cart', 'child', 'cone', 'container',
    'cooper', 'coupe', 'cyclist', 'display', 'dog', 'dot', 'ebikerider',
    'ecosport', 'fiesta', 'figure', 'focus', 'golf', 'gx', 'hatchback',
    'headlight', 'hydrant', 'individual', 'light', 'man', 'mark', 'marking',
    'material', 'mazda', 'minibus', 'minivan', 'motorcycle', 'motorcyclist',
    'officer', 'patch', 'pedestrian', 'person', 'pipe', 'plate', 'polo',
    'post', 'reflection', 'rider', 'scooter', 'sedan', 'sign', 'signboard',
    'surface', 'suv', 'symbol', 'taillight', 'taxi', 'tire', 'trailer',
    'train', 'truck', 'umbrella', 'van', 'vehicle', 'wagon', 'woman',
    'worker', 'y',
}

COMPOUND_TERMINAL_HEADS = {
    'bus', 'car', 'hatchback', 'headlight', 'minibus', 'minivan', 'rider',
    'sedan', 'suv', 'taillight', 'tire', 'truck', 'van', 'vehicle',
}

# These samples do not provide one reliably identifiable referred object.
# Keep their current proposal in the fix files, but never label it confirmed.
MANUAL_REVIEW_REASONS = {
    1: ('The caption describes bounding-box coordinates and explicitly says '
        'the box covers road rather than the mentioned car.'),
    2: 'The caption explicitly says that no bus is present in the box.',
    5478: ('The caption discusses two cars and several competing vehicle '
           'identities, so the intended box referent is ambiguous.'),
    5902: 'The caption explicitly says that no visible bus was detected.',
    6198: 'The caption explicitly says that no visible cars were detected.',
    13496: ('The caption alternates between a bicycle and a vehicle component '
            'and says that no rider can be identified.'),
    15255: ('The caption contains prompt/answer leakage and several competing '
            'car/bus interpretations for the supplied box.'),
    15518: 'The caption says no cars are visible and the road is empty.',
    15778: 'The caption explicitly says that no visible buses were detected.',
    15841: 'The caption explicitly says that no visible cars were detected.',
    15842: 'The caption explicitly says that no visible bus was detected.',
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
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument(
        '--manual-report', type=Path,
        default=Path('/home/lbw/mmdetection/analysis_demo/referent_spans/'
                     'mfad_train_referent_spans_manual_review.json'))
    return parser.parse_args()


def is_lexical_joiner(token, doc):
    if token.text not in LEXICAL_JOINERS:
        return False
    if token.i == 0 or token.i + 1 >= len(doc):
        return False
    left = doc[token.i - 1]
    right = doc[token.i + 1]
    if (left.text.lower() in COORDINATORS
            or right.text.lower() in COORDINATORS):
        return False
    return (left.idx + len(left.text) == token.idx
            and token.idx + len(token.text) == right.idx
            and left.is_alpha and right.is_alpha)


def keep_phrase_token(token, doc):
    if token.text.lower() in ARTICLES:
        return False
    if token.text.lower() in COORDINATORS or token.pos_ == 'CCONJ':
        return False
    if token.is_space:
        return False
    if token.is_punct and not is_lexical_joiner(token, doc):
        return False
    return True


def tokens_to_spans(tokens, caption, doc):
    kept = [token for token in tokens if keep_phrase_token(token, doc)]
    if not kept:
        return []
    spans = []
    start = kept[0].idx
    end = kept[0].idx + len(kept[0].text)
    for token in kept[1:]:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        gap = caption[end:token_start]
        if not gap or gap.isspace():
            end = token_end
        else:
            spans.append([start, end])
            start, end = token_start, token_end
    spans.append([start, end])
    return spans


def is_object_head(token):
    return (token.text.lower() in OBJECT_HEADS
            or token.lemma_.lower() in OBJECT_HEADS)


def select_initial_head(doc):
    candidates = [token for token in doc[:20] if is_object_head(token)]
    if not candidates:
        return None
    head = candidates[0]
    # In "a red vehicle's taillight", the possessed light, rather than the
    # vehicle, is the described target.  Advance to that possessed object.
    if head.i + 1 < len(doc) and doc[head.i + 1].text in {"'s", '’s'}:
        later = [token for token in candidates if head.i < token.i <= head.i + 4]
        if later:
            head = later[0]
    else:
        # Prefer the terminal head of an adjacent compound: "bicycle rider",
        # "Toyota Camry sedan", and "Buick Encore GX SUV" should end at
        # rider/sedan/SUV rather than at the first recognized noun.
        for candidate in candidates[1:]:
            candidate_name = candidate.lemma_.lower()
            if candidate_name not in COMPOUND_TERMINAL_HEADS:
                candidate_name = candidate.text.lower()
            if (candidate.i == head.i + 1
                    and candidate_name in COMPOUND_TERMINAL_HEADS):
                head = candidate
            elif candidate.i > head.i + 1:
                break
    return head


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
                f'BERT mapping failed for span {[start, end]} in {caption!r}')
        evidence.append({
            'char_span': [start, end],
            'token_span': [begin_token, end_token + 1],
            'bert_tokens': tokenizer.convert_ids_to_tokens(
                encoding['input_ids'][begin_token:end_token + 1]),
        })
    return evidence


def review_sample(sample, doc, tokenizer):
    result = dict(sample)
    caption = result['caption']
    original_spans = result['referent_positive_spans']
    original_texts = result['referent_positive_texts']
    reason = MANUAL_REVIEW_REASONS.get(result['sample_index'])

    if reason is None:
        head = select_initial_head(doc)
        if head is None:
            reason = 'No reliable sentence-initial object head was found.'
            fixed_spans = original_spans
        else:
            start = 1 if doc[0].text.lower() in ARTICLES else 0
            fixed_spans = tokens_to_spans(doc[start:head.i + 1], caption, doc)
            if not fixed_spans:
                reason = 'The proposed sentence-initial object phrase is empty.'
                fixed_spans = original_spans
    else:
        fixed_spans = original_spans

    fixed_texts = [caption[start:end] for start, end in fixed_spans]
    if reason is not None:
        status = 'needs_manual_review'
        note = f'Needs manual review: {reason}'
    elif fixed_spans == original_spans:
        status = 'confirmed'
        note = ('Confirmed: the existing spans identify the sentence-initial '
                'single-object referent.')
    else:
        status = 'corrected'
        note = ('Corrected: rebuilt the sentence-initial object base noun '
                'phrase and removed predicate/scene words selected by spaCy.')

    result['original_referent_positive_spans'] = original_spans
    result['original_referent_positive_texts'] = original_texts
    result['fixed_referent_positive_spans'] = fixed_spans
    result['fixed_referent_positive_texts'] = fixed_texts
    result['fixed_subject_phrase_char_span'] = [
        fixed_spans[0][0], fixed_spans[-1][1]
    ]
    result['fixed_subject_phrase'] = caption[
        fixed_spans[0][0]:fixed_spans[-1][1]]
    result['fixed_bert_span_evidence'] = bert_evidence(
        tokenizer, caption, fixed_spans)
    result['review_status'] = status
    result['review_note'] = note
    if reason is not None:
        result['manual_review_reason'] = reason
    return result


def write_manual_report(path, samples):
    records = [{
        'sample_index': sample['sample_index'],
        'source_line': sample['source_line'],
        'confidence': sample['confidence'],
        'filename': sample['filename'],
        'bbox_xyxy': sample['bbox_xyxy'],
        'caption': sample['caption'],
        'current_proposed_spans': sample['fixed_referent_positive_spans'],
        'current_proposed_texts': sample['fixed_referent_positive_texts'],
        'reason': sample['manual_review_reason'],
    } for sample in samples]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'sample_count': len(records),
        'instructions': (
            'These samples need a human decision. The corresponding fix-file '
            'entries retain the automatic proposal and use review_status '
            'needs_manual_review.'),
        'samples': records,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    nlp = spacy.load(args.spacy_model, disable=['ner'])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for char_to_token.')

    manual = []
    summaries = {}
    for confidence in ('medium', 'low'):
        source = args.input_dir / f'mfad_train_referent_spans_{confidence}.json'
        destination = (
            args.input_dir / f'mfad_train_referent_spans_{confidence}_fix.json')
        payload = json.loads(source.read_text(encoding='utf-8'))
        samples = payload['samples']
        docs = nlp.pipe(
            [sample['caption'] for sample in samples],
            batch_size=args.batch_size)
        reviewed = [
            review_sample(sample, doc, tokenizer)
            for sample, doc in zip(samples, docs)
        ]
        if len(reviewed) != len(samples):
            raise RuntimeError('spaCy returned an unexpected number of docs.')
        status_counts = {
            status: sum(item['review_status'] == status for item in reviewed)
            for status in ('confirmed', 'corrected', 'needs_manual_review')
        }
        manual.extend(
            item for item in reviewed
            if item['review_status'] == 'needs_manual_review')
        payload['samples'] = reviewed
        payload[f'source_{confidence}_confidence_file'] = str(source)
        payload['semantic_review'] = {
            'reviewed_count': len(reviewed),
            'status_counts': status_counts,
            'automatic_rules_applied': True,
            'all_samples_structurally_reviewed': True,
            'all_samples_semantically_resolved': not status_counts[
                'needs_manual_review'],
            'training_annotation_modified': False,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
        summaries[confidence] = {
            'input': str(source),
            'output': str(destination),
            'sample_count': len(reviewed),
            'status_counts': status_counts,
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
