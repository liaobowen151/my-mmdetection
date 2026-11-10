#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DEFAULT_BERT, SPECIAL_TOKENS, freeze_bert,
                    resolve_device)
from model import SubjectHeadClassifier
from mmdet.models.language_models.bert import BertModel


def parse_args():
    parser = argparse.ArgumentParser(description='Predict one subject head.')
    parser.add_argument('caption')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bert', default=DEFAULT_BERT)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--device', default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert, use_fast=True)
    encoded = tokenizer(
        args.caption, truncation=True, max_length=256,
        return_offsets_mapping=True, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    offsets = encoded['offset_mapping'][0].tolist()
    candidate_mask = torch.tensor([[
        right > left and not token.startswith('##')
        and any(ch.isalnum() for ch in args.caption[left:right])
        for (left, right), token in zip(offsets, tokens)
    ]], dtype=torch.bool, device=device)

    bert = BertModel(
        name=args.bert, max_tokens=256, pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=SPECIAL_TOKENS,
        add_pooling_layer=False).to(device)
    freeze_bert(bert)
    model = SubjectHeadClassifier(768, args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    with torch.inference_mode():
        hidden = bert([args.caption])['hidden']
        logits = model(hidden, candidate_mask)
        probabilities = logits.softmax(-1)
        index = logits.argmax(-1).item()
    left, right = offsets[index]
    print(json.dumps({
        'caption': args.caption,
        'subject_head': args.caption[left:right],
        'subject_head_char_span': [left, right],
        'token_index': index,
        'confidence': probabilities[0, index].item(),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
