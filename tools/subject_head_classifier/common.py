import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


DEFAULT_ANNOTATIONS = (
    '/home/lbw/mmdetection/analysis_demo/referent_spans/'
    'm3fd_train_referent_spans_all.json')
DEFAULT_BERT = (
    '/home/lbw/mmdetection/huggingface-hub/transformers/'
    'bert-base-uncased/')
SPECIAL_TOKENS = ['[CLS]', '[SEP]', '.', '?']


def load_samples(path: str) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    samples = data['samples'] if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise TypeError('Annotation file must contain a list or {"samples": [...]}')
    return samples


def write_json(path: str, data) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def grouped_split(samples: Sequence[dict], seed: int = 3407,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1) -> Dict[str, List[int]]:
    """Split sample indices by filename; a source image cannot leak."""
    groups = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[sample['filename']].append(i)
    filenames = sorted(groups)
    random.Random(seed).shuffle(filenames)
    n = len(filenames)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    name_splits = {
        'train': filenames[:n_train],
        'val': filenames[n_train:n_train + n_val],
        'test': filenames[n_train + n_val:],
    }
    return {
        split: [i for name in names for i in groups[name]]
        for split, names in name_splits.items()
    }


def candidate_mask_and_target(caption: str, offsets: Sequence[Sequence[int]],
                              tokens: Sequence[str],
                              head_span: Sequence[int]) -> Tuple[List[bool], int]:
    """Build word-start candidates and align a character span to one token."""
    start, end = map(int, head_span)
    candidate_mask = []
    target = None
    for i, ((left, right), token) in enumerate(zip(offsets, tokens)):
        surface = caption[left:right] if right > left else ''
        is_candidate = (
            right > left and not token.startswith('##')
            and any(ch.isalnum() for ch in surface))
        candidate_mask.append(is_candidate)
        if target is None and is_candidate and left < end and right > start:
            target = i
    if target is None:
        raise ValueError(
            f'Cannot align span {head_span} ({caption[start:end]!r}) in {caption!r}')
    return candidate_mask, target


def audit_samples(samples: Sequence[dict], tokenizer) -> dict:
    issues = []
    head_counts = Counter()
    for i, sample in enumerate(samples):
        caption = sample['caption']
        span = sample['subject_head_char_span']
        head = sample['subject_head']
        head_counts[head] += 1
        sample_issues = []
        if len(span) != 2 or not (0 <= span[0] < span[1] <= len(caption)):
            sample_issues.append('invalid_char_span')
        elif caption[span[0]:span[1]] != head:
            sample_issues.append('span_text_mismatch')
        try:
            encoded = tokenizer(caption, return_offsets_mapping=True,
                                truncation=True, max_length=256)
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
            candidate_mask_and_target(caption, encoded['offset_mapping'],
                                      tokens, span)
        except Exception as exc:  # report all bad annotations together
            sample_issues.append(f'token_alignment: {exc}')
        if sample_issues:
            issues.append({
                'sample_index': i,
                'filename': sample.get('filename'),
                'caption': caption,
                'subject_head': head,
                'subject_head_char_span': span,
                'issues': sample_issues,
            })
    suspicious = {
        word: head_counts[word] for word in ('drives', 'travels')
        if head_counts[word]
    }
    return {
        'num_samples': len(samples),
        'num_issues': len(issues),
        'issues': issues,
        'suspicious_head_counts': suspicious,
        'num_unique_heads_case_sensitive': len(head_counts),
        'top_heads': head_counts.most_common(30),
    }


def freeze_bert(bert) -> None:
    bert.requires_grad_(False)
    bert.eval()


def resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)
