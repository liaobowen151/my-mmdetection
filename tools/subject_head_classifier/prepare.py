#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DEFAULT_ANNOTATIONS, DEFAULT_BERT, SPECIAL_TOKENS,
                    audit_samples, candidate_mask_and_target, freeze_bert,
                    grouped_split, load_samples, resolve_device, write_json)
from mmdet.models.language_models.bert import BertModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit labels, make image-grouped splits and cache frozen BERT features.')
    parser.add_argument('--annotations', default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--bert', default=DEFAULT_BERT)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cache-dtype', choices=['float16', 'float32'],
                        default='float16')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.annotations)
    tokenizer = AutoTokenizer.from_pretrained(args.bert, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError('A fast tokenizer is required for offset_mapping')

    audit = audit_samples(samples, tokenizer)
    write_json(str(output_dir / 'annotation_audit.json'), audit)
    if audit['num_issues']:
        raise RuntimeError(
            f"Found {audit['num_issues']} invalid annotations; see annotation_audit.json")

    splits = grouped_split(samples, seed=args.seed)
    split_manifest = {
        'seed': args.seed,
        'annotation_file': args.annotations,
        'splits': splits,
        'filenames': {
            name: sorted({samples[i]['filename'] for i in indices})
            for name, indices in splits.items()
        },
    }
    write_json(str(output_dir / 'split_manifest.json'), split_manifest)

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
            captions = [samples[i]['caption'] for i in batch_indices]
            encoded = tokenizer(
                captions, padding=True, truncation=True, max_length=256,
                return_offsets_mapping=True, return_tensors='pt')
            with torch.inference_mode():
                hidden = bert(captions)['hidden'].detach().cpu().to(cache_dtype)
            lengths = encoded['attention_mask'].sum(dim=1).tolist()
            for row, (sample_index, length) in enumerate(zip(batch_indices, lengths)):
                sample = samples[sample_index]
                offsets = encoded['offset_mapping'][row, :length].tolist()
                input_ids = encoded['input_ids'][row, :length].tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                candidate_mask, target = candidate_mask_and_target(
                    sample['caption'], offsets, tokens,
                    sample['subject_head_char_span'])
                records.append({
                    'sample_index': sample_index,
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
            },
        }, output_dir / f'{split_name}_features.pt')
        print(f'{split_name}: cached {len(records)} samples')


if __name__ == '__main__':
    main()
