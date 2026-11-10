#!/usr/bin/env python3
import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DEFAULT_BERT, SPECIAL_TOKENS, audit_samples,
                    candidate_mask_and_target, freeze_bert, load_samples,
                    resolve_device, write_json)
from mmdet.models.language_models.bert import BertModel


DEFAULT_INPUTS = {
    'm3fd': str(REPO / 'analysis_demo/referent_spans/m3fd_train_referent_spans_all.json'),
    'flir': str(REPO / 'analysis_demo/referent_spans/flir_train_referent_spans_all.json'),
    'mfad': str(REPO / 'analysis_demo/referent_spans/mfad_train_referent_spans_all.json'),
}
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def normalize_caption(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


def repair_unambiguous_hyphen_head(sample, dataset_name, source_index):
    """Repair the known MFAD `semi-truck` head marked as the hyphen."""
    row = dict(sample)
    span = row.get('subject_head_char_span', [])
    if (row.get('subject_head') == '-' and len(span) == 2
            and row['caption'][span[0]:span[1]] == '-'):
        suffix = row['caption'][span[1]:span[1] + 5]
        prefix = row['caption'][max(0, span[0] - 4):span[0]]
        if prefix.lower() == 'semi' and suffix.lower() == 'truck':
            old = {
                'dataset_name': dataset_name,
                'source_sample_index': source_index,
                'filename': row.get('filename'),
                'caption': row['caption'],
                'old_subject_head': '-',
                'old_subject_head_char_span': list(span),
                'new_subject_head': row['caption'][span[1]:span[1] + 5],
                'new_subject_head_char_span': [span[1], span[1] + 5],
                'reason': 'hyphen inside semi-truck cannot be a lexical head',
            }
            row['subject_head'] = old['new_subject_head']
            row['subject_head_char_span'] = old['new_subject_head_char_span']
            return row, old
    return row, None


def build_components(samples):
    """Keep both a source image and an exact normalized caption in one split."""
    union_find = UnionFind(len(samples))
    image_owner = {}
    caption_owner = {}
    for index, sample in enumerate(samples):
        image_key = f"{sample['dataset_name']}::{sample['filename']}"
        caption_key = normalize_caption(sample['caption'])
        for mapping, key in ((image_owner, image_key),
                             (caption_owner, caption_key)):
            if key in mapping:
                union_find.union(index, mapping[key])
            else:
                mapping[key] = index
    components = defaultdict(list)
    for index in range(len(samples)):
        components[union_find.find(index)].append(index)
    return list(components.values())


def assign_components(samples, components, seed):
    """Greedily balance indivisible components against per-domain targets."""
    datasets = sorted({sample['dataset_name'] for sample in samples})
    totals = Counter(sample['dataset_name'] for sample in samples)
    targets = {
        split: {dataset: totals[dataset] * ratio
                for dataset in datasets}
        for split, ratio in SPLIT_RATIOS.items()
    }
    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    counts = {split: Counter() for split in SPLIT_RATIOS}
    assigned = {split: [] for split in SPLIT_RATIOS}
    for component in components:
        component_counts = Counter(
            samples[index]['dataset_name'] for index in component)

        def incremental_cost(split):
            cost = 0.0
            for dataset, amount in component_counts.items():
                target = max(targets[split][dataset], 1.0)
                before = counts[split][dataset]
                after = before + amount
                cost += ((after - target) ** 2
                         - (before - target) ** 2) / target
            return cost

        chosen = min(SPLIT_RATIOS, key=incremental_cost)
        assigned[chosen].extend(component)
        counts[chosen].update(component_counts)
    for split in assigned:
        assigned[split].sort()
    return assigned, counts, totals


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare leakage-safe combined M3FD/FLIR/MFAD features.')
    parser.add_argument('--m3fd', default=DEFAULT_INPUTS['m3fd'])
    parser.add_argument('--flir', default=DEFAULT_INPUTS['flir'])
    parser.add_argument('--mfad', default=DEFAULT_INPUTS['mfad'])
    parser.add_argument('--bert', default=DEFAULT_BERT)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cache-dtype', choices=['float16', 'float32'],
                        default='float16')
    parser.add_argument('--split-only', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for offset_mapping')

    source_paths = {'m3fd': args.m3fd, 'flir': args.flir, 'mfad': args.mfad}
    samples = []
    automatic_corrections = []
    audit_by_dataset = {}
    for dataset_name, path in source_paths.items():
        raw_dataset_samples = load_samples(path)
        dataset_samples = []
        for source_index, sample in enumerate(raw_dataset_samples):
            repaired, correction = repair_unambiguous_hyphen_head(
                sample, dataset_name, source_index)
            dataset_samples.append(repaired)
            if correction is not None:
                automatic_corrections.append(correction)
        audit = audit_samples(dataset_samples, tokenizer)
        audit_by_dataset[dataset_name] = audit
        for source_index, sample in enumerate(dataset_samples):
            row = dict(sample)
            row['dataset_name'] = dataset_name
            row['source_sample_index'] = source_index
            samples.append(row)
    write_json(str(output_dir / 'automatic_corrections.json'),
               automatic_corrections)
    write_json(str(output_dir / 'annotation_audit.json'), audit_by_dataset)
    invalid = sum(audit['num_issues'] for audit in audit_by_dataset.values())
    if invalid:
        raise RuntimeError(f'Found {invalid} invalid annotations; see audit file')

    caption_heads = defaultdict(set)
    for sample in samples:
        caption_heads[normalize_caption(sample['caption'])].add(
            sample['subject_head'].lower())
    conflicts = [
        {'caption': caption, 'subject_heads': sorted(heads)}
        for caption, heads in caption_heads.items() if len(heads) > 1
    ]
    write_json(str(output_dir / 'caption_label_conflicts.json'), conflicts)
    if conflicts:
        raise RuntimeError(
            f'Found {len(conflicts)} identical captions with conflicting heads')

    components = build_components(samples)
    splits, split_counts, totals = assign_components(
        samples, components, args.seed)
    sample_to_split = {
        index: split for split, indices in splits.items() for index in indices}
    caption_splits = defaultdict(set)
    image_splits = defaultdict(set)
    for index, sample in enumerate(samples):
        split = sample_to_split[index]
        caption_splits[normalize_caption(sample['caption'])].add(split)
        image_splits[f"{sample['dataset_name']}::{sample['filename']}"] .add(split)
    if any(len(value) != 1 for value in caption_splits.values()):
        raise AssertionError('Normalized caption leakage detected')
    if any(len(value) != 1 for value in image_splits.values()):
        raise AssertionError('Image leakage detected')

    manifest = {
        'seed': args.seed,
        'ratios': SPLIT_RATIOS,
        'source_files': source_paths,
        'total_samples': len(samples),
        'total_by_dataset': dict(totals),
        'num_components': len(components),
        'largest_component': max(map(len, components)),
        'split_counts': {
            split: {
                'total': len(indices),
                'by_dataset': dict(split_counts[split]),
            } for split, indices in splits.items()
        },
        'indices': splits,
        'leakage_checks': {
            'normalized_caption_cross_split': 0,
            'dataset_image_cross_split': 0,
        },
    }
    write_json(str(output_dir / 'split_manifest.json'), manifest)
    print('split_counts:', manifest['split_counts'])
    print('components:', len(components), 'largest:', max(map(len, components)))
    if args.split_only:
        return

    device = resolve_device(args.device)
    bert = BertModel(
        name=args.bert, max_tokens=256, pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=SPECIAL_TOKENS,
        add_pooling_layer=False).to(device)
    freeze_bert(bert)
    cache_dtype = torch.float16 if args.cache_dtype == 'float16' else torch.float32
    for split_name, indices in splits.items():
        records = []
        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            captions = [samples[index]['caption'] for index in batch_indices]
            encoded = tokenizer(
                captions, padding=True, truncation=True, max_length=256,
                return_offsets_mapping=True, return_tensors='pt')
            with torch.inference_mode():
                hidden = bert(captions)['hidden'].detach().cpu().to(cache_dtype)
            lengths = encoded['attention_mask'].sum(dim=1).tolist()
            for row, (index, length) in enumerate(zip(batch_indices, lengths)):
                sample = samples[index]
                offsets = encoded['offset_mapping'][row, :length].tolist()
                input_ids = encoded['input_ids'][row, :length].tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                candidate_mask, target = candidate_mask_and_target(
                    sample['caption'], offsets, tokens,
                    sample['subject_head_char_span'])
                records.append({
                    'sample_index': index,
                    'source_sample_index': sample['source_sample_index'],
                    'dataset_name': sample['dataset_name'],
                    'filename': sample['filename'],
                    'caption': sample['caption'],
                    'subject_head': sample['subject_head'],
                    'subject_head_char_span': sample['subject_head_char_span'],
                    'hidden': hidden[row, :length].clone(),
                    'candidate_mask': torch.tensor(candidate_mask, dtype=torch.bool),
                    'target': target,
                    'offsets': offsets,
                    'tokens': tokens,
                })
        torch.save({
            'records': records,
            'metadata': {
                'bert': args.bert,
                'feature_key': 'hidden',
                'use_sub_sentence_represent': True,
                'special_tokens_list': SPECIAL_TOKENS,
                'cache_dtype': args.cache_dtype,
                'combined_datasets': sorted(source_paths),
            },
        }, output_dir / f'{split_name}_features.pt')
        print(f'{split_name}: cached {len(records)} samples')


if __name__ == '__main__':
    main()
