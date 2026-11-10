"""Extract M3FD training referent spans and split them by confidence.

This is an analysis-only utility.  It never rewrites the source annotation.
The output keeps enough parser and tokenizer evidence for manual review before
any final ODVG training annotation is produced.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import spacy
from transformers import AutoTokenizer


ARTICLES = {'a', 'an', 'the'}
COORDINATORS = {'and', 'or'}
LEXICAL_JOINERS = {'-', '‐', '‑', '‒', '–', '—', "'", '’'}
RELATION_WORDS = {
    'above', 'behind', 'below', 'beside', 'between', 'near', 'under', 'with'
}
SUSPICIOUS_HEADS = {
    'casting', 'dark', 'drive', 'drives', 'faint', 'glow', 'glows', 'side',
    'silver', 'utility', 'white'
}
HARD_LOW_FLAGS = {
    'bert_mapping_failed',
    'empty_positive_spans',
    'entity_coordination',
    'head_not_in_positive_spans',
    'multiple_syntactic_subjects',
    'negative_caption',
    'no_reliable_noun_phrase',
    'pronoun_subject',
    'relation_inside_subject_phrase',
    'suspicious_subject_head',
    'unresolved_subject',
}
MEDIUM_FLAGS = {
    'first_noun_chunk_fallback',
    'long_subject_phrase',
    'non_initial_subject_phrase',
    'reconstructed_subject_phrase',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file',
        default=(
            '/home/lbw/mmdetection/data/M3FD/annotations/'
            'rgbtvg_m3fd_train_vg.json'))
    parser.add_argument(
        '--tokenizer-dir',
        default=(
            '/home/lbw/mmdetection/huggingface-hub/transformers/'
            'bert-base-uncased/'))
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    parser.add_argument(
        '--output-dir', default='/home/lbw/mmdetection/analysis_demo')
    parser.add_argument(
        '--output-prefix', default='m3fd_train_referent_spans')
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def is_lexical_joiner(token, doc) -> bool:
    """Keep a hyphen/apostrophe only when it joins adjacent word pieces."""
    if token.text not in LEXICAL_JOINERS:
        return False
    if token.i == 0 or token.i + 1 >= len(doc):
        return False
    left = doc[token.i - 1]
    right = doc[token.i + 1]
    touches_left = left.idx + len(left.text) == token.idx
    touches_right = token.idx + len(token.text) == right.idx
    return touches_left and touches_right and left.is_alpha and right.is_alpha


def keep_phrase_token(token, doc) -> bool:
    lower = token.text.lower()
    if lower in ARTICLES:
        return False
    if lower in COORDINATORS or token.pos_ == 'CCONJ':
        return False
    if token.is_space:
        return False
    if token.is_punct and not is_lexical_joiner(token, doc):
        return False
    return True


def tokens_to_spans(tokens: Sequence, caption: str, doc) -> List[List[int]]:
    """Group kept tokens when the original gap contains only whitespace."""
    kept = [token for token in tokens if keep_phrase_token(token, doc)]
    if not kept:
        return []

    spans: List[List[int]] = []
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


def find_matching_chunk(chunks: Sequence, head) -> Optional[object]:
    for chunk in chunks:
        if chunk.root.i == head.i:
            return chunk
    return None


def reconstruct_pre_head_phrase(doc, head) -> Sequence:
    """Conservative candidate for an explicit subject lacking a noun chunk."""
    start = head.left_edge.i
    return doc[start:head.i + 1]


def has_entity_coordination(head, phrase, doc) -> bool:
    """Detect coordinated referents without trusting noisy long attachments.

    The small English parser sometimes attaches nouns from a following
    prepositional modifier (for example, ``person in a jacket and jeans``) as
    conjuncts of the subject.  A real coordinated referent either lies inside
    the selected base noun phrase or starts with a coordinator immediately
    after it (for example, ``a person and a motorcycle``).
    """
    if head.dep_ == 'conj' and head.pos_ in {'NOUN', 'PROPN', 'PRON'}:
        return True
    entity_conjuncts = [
        token for token in head.conjuncts
        if token.pos_ in {'NOUN', 'PROPN', 'PRON'}
    ]
    if not entity_conjuncts:
        return False
    if any(phrase.start <= token.i < phrase.end
           for token in entity_conjuncts):
        return True
    following = phrase.end
    while following < len(doc) and doc[following].is_punct:
        following += 1
    return (following < len(doc)
            and doc[following].text.lower() in COORDINATORS)


def bert_span_evidence(tokenizer, caption: str,
                       spans: Sequence[Sequence[int]]) -> Tuple[List[dict], bool]:
    encoding = tokenizer(
        caption,
        add_special_tokens=True,
        max_length=256,
        truncation=True,
        return_offsets_mapping=True)
    input_ids = encoding['input_ids']
    evidence = []
    valid = True
    for start, end in spans:
        begin_token = encoding.char_to_token(start)
        end_token = encoding.char_to_token(end - 1)
        if begin_token is None or end_token is None:
            valid = False
            evidence.append({
                'char_span': [start, end],
                'token_span': None,
                'bert_tokens': [],
            })
            continue
        evidence.append({
            'char_span': [start, end],
            'token_span': [begin_token, end_token + 1],
            'bert_tokens': tokenizer.convert_ids_to_tokens(
                input_ids[begin_token:end_token + 1]),
        })
    return evidence, valid


def choose_subject_phrase(doc) -> Tuple[Optional[object], Sequence, str,
                                        List[str]]:
    """Return head, phrase tokens, extraction method, and initial flags."""
    flags: List[str] = []
    chunks = list(doc.noun_chunks)
    subjects = [
        token for token in doc
        if token.dep_ in {'nsubj', 'nsubjpass'}
        and token.pos_ in {'NOUN', 'PROPN', 'PRON'}
    ]

    if subjects:
        head = subjects[0]
        if len(subjects) > 1:
            flags.append('multiple_syntactic_subjects')
        if head.pos_ == 'PRON':
            flags.append('pronoun_subject')
        chunk = find_matching_chunk(chunks, head)
        if chunk is not None:
            return head, chunk, 'syntactic_subject_noun_chunk', flags
        flags.append('reconstructed_subject_phrase')
        return (head, reconstruct_pre_head_phrase(doc, head),
                'syntactic_subject_reconstructed', flags)

    if chunks:
        chunk = chunks[0]
        flags.append('first_noun_chunk_fallback')
        return chunk.root, chunk, 'first_noun_chunk', flags

    nouns = [token for token in doc if token.pos_ in {'NOUN', 'PROPN'}]
    if nouns:
        flags.append('no_reliable_noun_phrase')
        head = nouns[0]
        return head, doc[head.i:head.i + 1], 'first_noun_token', flags

    flags.extend(['unresolved_subject', 'no_reliable_noun_phrase'])
    return None, doc[0:0], 'unresolved', flags


def unique_flags(flags: Iterable[str]) -> List[str]:
    return sorted(set(flags))


def assign_confidence(flags: Sequence[str]) -> str:
    flag_set = set(flags)
    if flag_set & HARD_LOW_FLAGS:
        return 'low'
    if flag_set & MEDIUM_FLAGS:
        return 'medium'
    return 'high'


def analyze_sample(index: int, data: dict, doc, tokenizer) -> dict:
    caption = data['grounding']['caption']
    region = data['grounding']['regions'][0]
    head, phrase, method, flags = choose_subject_phrase(doc)

    if caption.lstrip().lower().startswith('no '):
        flags.append('negative_caption')
    if head is not None and has_entity_coordination(head, phrase, doc):
        flags.append('entity_coordination')
    if head is not None and head.lemma_.lower() in SUSPICIOUS_HEADS:
        flags.append('suspicious_subject_head')
    if len(phrase) > 10:
        flags.append('long_subject_phrase')
    if len(phrase) and phrase.start > 1:
        flags.append('non_initial_subject_phrase')
    if any(token.text.lower() in RELATION_WORDS for token in phrase):
        flags.append('relation_inside_subject_phrase')

    spans = tokens_to_spans(list(phrase), caption, doc)
    if not spans:
        flags.append('empty_positive_spans')
    if head is not None:
        head_start = head.idx
        head_end = head.idx + len(head.text)
        if not any(start <= head_start and head_end <= end
                   for start, end in spans):
            flags.append('head_not_in_positive_spans')

    bert_evidence, bert_valid = bert_span_evidence(
        tokenizer, caption, spans)
    if not bert_valid:
        flags.append('bert_mapping_failed')

    flags = unique_flags(flags)
    confidence = assign_confidence(flags)
    phrase_start = phrase.start_char if len(phrase) else None
    phrase_end = phrase.end_char if len(phrase) else None
    return {
        'sample_index': index,
        'source_line': index + 1,
        'filename': data['filename'],
        'caption': caption,
        'bbox_xyxy': region['bbox'],
        'original_tokens_positive': region['tokens_positive'],
        'subject_head': None if head is None else head.text,
        'subject_head_lemma': None if head is None else head.lemma_,
        'subject_head_pos': None if head is None else head.pos_,
        'subject_head_dependency': None if head is None else head.dep_,
        'subject_head_char_span': (
            None if head is None
            else [head.idx, head.idx + len(head.text)]),
        'subject_phrase': (
            None if not len(phrase) else caption[phrase_start:phrase_end]),
        'subject_phrase_char_span': (
            None if not len(phrase) else [phrase_start, phrase_end]),
        'referent_positive_spans': spans,
        'referent_positive_texts': [caption[start:end]
                                    for start, end in spans],
        'bert_span_evidence': bert_evidence,
        'extraction_method': method,
        'confidence': confidence,
        'review_flags': flags,
    }


def output_payload(source: Path, confidence: str, samples: Sequence[dict],
                   total: int, spacy_model: str,
                   tokenizer_dir: str) -> Dict:
    flag_counts = Counter(
        flag for sample in samples for flag in sample['review_flags'])
    method_counts = Counter(
        sample['extraction_method'] for sample in samples)
    return {
        'source_annotation': str(source),
        'analysis_only': True,
        'confidence': confidence,
        'sample_count': len(samples),
        'population_size': total,
        'spacy_model': spacy_model,
        'tokenizer': tokenizer_dir,
        'span_policy': {
            'base': 'spaCy subject base noun phrase',
            'removed_articles': sorted(ARTICLES),
            'removed_coordinators': sorted(COORDINATORS),
            'remove_independent_punctuation': True,
            'preserve_internal_hyphens_and_apostrophes': True,
            'group_across_whitespace_only': True,
        },
        'method_counts': dict(sorted(method_counts.items())),
        'review_flag_counts': dict(sorted(flag_counts.items())),
        'samples': list(samples),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.ann_file)
    output_dir = Path(args.output_dir)
    rows = load_jsonl(source)

    nlp = spacy.load(args.spacy_model, disable=['ner'])
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for char_to_token.')

    captions = [row['grounding']['caption'] for row in rows]
    docs = nlp.pipe(captions, batch_size=args.batch_size)
    grouped = {'high': [], 'medium': [], 'low': []}
    for index, (row, doc) in enumerate(zip(rows, docs)):
        sample = analyze_sample(index, row, doc, tokenizer)
        grouped[sample['confidence']].append(sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for confidence in ('high', 'medium', 'low'):
        path = output_dir / f'{args.output_prefix}_{confidence}.json'
        payload = output_payload(
            source, confidence, grouped[confidence], len(rows),
            args.spacy_model, args.tokenizer_dir)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        output_paths[confidence] = str(path)

    print(json.dumps({
        'population_size': len(rows),
        'confidence_counts': {
            key: len(value) for key, value in grouped.items()
        },
        'outputs': output_paths,
    }, indent=2))


if __name__ == '__main__':
    main()
