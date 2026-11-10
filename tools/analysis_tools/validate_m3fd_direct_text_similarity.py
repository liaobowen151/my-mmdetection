#!/usr/bin/env python3
"""Test direct pretrained neck/text cosine alignment on all four levels."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmdet.registry import MODELS

from validate_m3fd_neck_abs_activity import (
    image_tensor, read_jsonl, resize_and_pad, resize_bool_map)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


TOPK_FRACTIONS = (0.05, 0.10)
TEMPERATURES = (0.05, 0.10, 0.20, 0.50)
PUNCTUATION = {'.', ',', '?', '!', ';', ':', '-', '--'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=20261102)
    return parser.parse_args()


def prefixed_state(state, prefix):
    return {key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)}


def build_components(config_path, checkpoint_path, device):
    import projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino  # noqa: F401

    cfg = Config.fromfile(config_path)
    backbone_cfg = cfg.model.backbone.copy()
    backbone_cfg['init_cfg'] = None
    backbone = MODELS.build(backbone_cfg)
    neck = MODELS.build(cfg.model.neck)
    language_model = MODELS.build(cfg.model.language_model)
    language_dim = language_model.language_backbone.body.language_dim
    # Grounding DINO's shared embedding width equals the neck output width;
    # this config stores embed_dims inside sublayers rather than encoder root.
    text_feat_map = nn.Linear(language_dim, cfg.model.neck.out_channels)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    reports = {}
    for name, module, prefix in (
            ('backbone', backbone, 'backbone.'),
            ('neck', neck, 'neck.'),
            ('language_model', language_model, 'language_model.'),
            ('text_feat_map', text_feat_map, 'text_feat_map.')):
        report = module.load_state_dict(prefixed_state(state, prefix), strict=False)
        reports[name] = {
            'missing_keys': list(report.missing_keys),
            'unexpected_keys': list(report.unexpected_keys),
        }
        module.to(device).eval()
    if hasattr(backbone, 'set_ir_lora_enabled'):
        backbone.set_ir_lora_enabled(False)
    reports['note'] = ('IR LoRA disabled. Direct cosine uses pretrained neck '
                       'features and pretrained text_feat_map output; no Wv/Wt.')
    return backbone, neck, language_model, text_feat_map, reports


def encode_caption(caption, language_model, text_feat_map):
    if not caption.endswith('.'):
        caption = caption + '.'
    tokenizer = language_model.tokenizer
    encoded = tokenizer.batch_encode_plus(
        [caption], max_length=language_model.max_tokens,
        padding='max_length' if language_model.pad_to_max else 'longest',
        return_special_tokens_mask=True, return_tensors='pt', truncation=True)
    device = next(language_model.parameters()).device
    with torch.inference_mode():
        result = language_model([caption])
        tokens = text_feat_map(result['embedded'])[0]
    input_ids = encoded['input_ids'][0]
    attention = encoded['attention_mask'][0].bool()
    special = encoded['special_tokens_mask'][0].bool()
    token_names = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    valid = attention & ~special
    for index, token in enumerate(token_names):
        if token in PUNCTUATION:
            valid[index] = False
    valid = valid.to(tokens.device)
    if not valid.any():
        raise RuntimeError(f'No valid tokens for caption: {caption}')
    return F.normalize(tokens[valid], dim=-1), [
        token for token, keep in zip(token_names, valid.cpu().tolist()) if keep]


def similarity_maps(feature, token_features):
    visual = F.normalize(feature[0], dim=0)
    token_similarity = torch.einsum('chw,tc->thw', visual, token_features)
    mean_text = F.normalize(token_features.mean(dim=0), dim=0)
    mean_similarity = torch.einsum('chw,c->hw', visual, mean_text)
    max_similarity, max_indices = token_similarity.max(dim=0)
    return {
        'mean': (mean_similarity, None),
        'max': (max_similarity, max_indices),
    }


def average_precision(scores, labels):
    order = torch.argsort(scores, descending=True)
    ranked = labels[order].float()
    total_positive = ranked.sum()
    if total_positive == 0:
        return float('nan')
    precision = ranked.cumsum(0) / torch.arange(
        1, ranked.numel() + 1, device=scores.device)
    return float((precision * ranked).sum() / total_positive)


def roc_auc(scores, labels):
    positive = scores[labels]
    negative = scores[~labels]
    if not positive.numel() or not negative.numel():
        return float('nan')
    # Feature maps are at most 28x28, so exact pairwise AUC is inexpensive.
    greater = (positive[:, None] > negative[None, :]).float()
    equal = (positive[:, None] == negative[None, :]).float()
    return float((greater + 0.5 * equal).mean())


def calculate_metrics(similarity, mask, valid):
    values = similarity[valid]
    labels = mask[valid]
    positives = int(labels.sum())
    total = int(labels.numel())
    if positives == 0 or positives == total:
        return None
    inside = float(values[labels].mean())
    outside = float(values[~labels].mean())
    result = {
        'valid_pixels': total,
        'mask_pixels': positives,
        'mask_ratio': positives / total,
        'inside_mean': inside,
        'outside_mean': outside,
        'delta_similarity': inside - outside,
        'pixel_ap': average_precision(values, labels),
        'roc_auc': roc_auc(values, labels),
    }
    for fraction in TOPK_FRACTIONS:
        count = max(1, math.ceil(fraction * total))
        indices = torch.topk(values, count, largest=True, sorted=False).indices
        precision = float(labels[indices].float().mean())
        suffix = f'{int(fraction * 100):02d}'
        result[f'P_{suffix}'] = precision
        result[f'G_{suffix}'] = precision / (result['mask_ratio'] + 1e-12)
    return result


def render(output_path, vis_rgb, ir_rgb, mask, maps, sample, method, condition):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes[0, 0].imshow(vis_rgb)
    axes[0, 0].contour(mask > 0, levels=[0.5], colors='lime', linewidths=1)
    axes[0, 0].set_title('VIS + mask')
    axes[0, 1].imshow(ir_rgb)
    axes[0, 1].contour(mask > 0, levels=[0.5], colors='lime', linewidths=1)
    axes[0, 1].set_title('IR + mask')
    axes[0, 2].imshow(mask, cmap='gray')
    axes[0, 2].set_title('D3 mask')
    axes[0, 3].axis('off')
    axes[0, 3].text(
        0, 1, f"{condition}/{method}\n{sample['filename']}\n"
        f"{sample['caption']}", va='top', wrap=True)
    for row_index, modality in enumerate(('vis', 'ir'), start=1):
        for level in range(4):
            similarity, metric = maps[(condition, method, modality, level)]
            axes[row_index, level].imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1)
            if metric is None:
                title = f'{modality.upper()} L{level}: mask vanished'
            else:
                title = (f"{modality.upper()} L{level}  d={metric['delta_similarity']:.3f}\n"
                         f"AP={metric['pixel_ap']:.3f}, G5={metric['G_05']:.2f}")
            axes[row_index, level].set_title(title)
    for axis in axes.flat:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def numeric_summary(rows, keys):
    result = {'valid_samples': len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            'mean': float(values.mean()),
            'median': float(np.median(values)),
            'std': float(values.std()),
        }
        if key in ('delta_similarity', 'G_05', 'G_10'):
            threshold = 0 if key == 'delta_similarity' else 1
            result[key]['fraction_above_threshold'] = float((values > threshold).mean())
    return result


def summarize(rows):
    keys = ('delta_similarity', 'pixel_ap', 'roc_auc', 'P_05', 'G_05',
            'P_10', 'G_10')
    summary = {}
    for condition in ('correct', 'wrong'):
        summary[condition] = {}
        for method in ('mean', 'max'):
            summary[condition][method] = {}
            for modality in ('vis', 'ir'):
                summary[condition][method][modality] = {}
                for level in range(4):
                    selected = [row for row in rows if
                                row['condition'] == condition and
                                row['method'] == method and
                                row['modality'] == modality and
                                row['level'] == level and
                                row['status'] == 'valid']
                    summary[condition][method][modality][f'level_{level}'] = \
                        numeric_summary(selected, keys) if selected else {
                            'valid_samples': 0}
    return summary


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    visualization_dir = output_dir / 'text_similarity_visualizations'
    visualization_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.annotations)
    if len(records) != 10:
        raise ValueError(f'Expected exactly 10 selected samples, got {len(records)}')
    # A one-step cyclic shift is a deterministic derangement for wrong captions.
    wrong_captions = [records[(i + 1) % len(records)]['caption']
                      for i in range(len(records))]
    backbone, neck, language_model, text_feat_map, load_report = build_components(
        args.config, args.checkpoint, args.device)
    (output_dir / 'text_checkpoint_load_report.json').write_text(
        json.dumps(load_report, indent=2, ensure_ascii=False), encoding='utf-8')

    rows = []
    dominant_tokens = Counter()
    for sample_index, sample in enumerate(records):
        filename = sample['filename']
        vis_bgr = cv2.imread(str(data_root / 'vis' / filename), cv2.IMREAD_COLOR)
        ir_bgr = cv2.imread(str(data_root / 'ir' / filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(Path(sample['mask_path'])), cv2.IMREAD_GRAYSCALE)
        if vis_bgr is None or ir_bgr is None or mask is None:
            raise FileNotFoundError(f'Input missing for {filename}')
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
        vis_input, mask_pad, valid_pad, _ = resize_and_pad(
            vis_rgb, mask, args.input_size)
        ir_input, _, _, _ = resize_and_pad(ir_rgb, mask, args.input_size)
        with torch.inference_mode():
            if hasattr(backbone, 'set_ir_lora_enabled'):
                backbone.set_ir_lora_enabled(False)
            features = {
                'vis': neck(backbone(image_tensor(vis_input, args.device))),
                'ir': neck(backbone(image_tensor(ir_input, args.device))),
            }
            text_data = {}
            for condition, caption in (
                    ('correct', sample['caption']),
                    ('wrong', wrong_captions[sample_index])):
                text_data[condition] = encode_caption(
                    caption, language_model, text_feat_map)

        plot_maps = {}
        for condition, (token_features, token_names) in text_data.items():
            for modality in ('vis', 'ir'):
                for level, feature in enumerate(features[modality]):
                    maps = similarity_maps(feature, token_features)
                    shape = feature.shape[-2:]
                    level_mask = resize_bool_map(mask_pad, shape, args.device)
                    level_valid = resize_bool_map(valid_pad, shape, args.device)
                    for method, (similarity, max_indices) in maps.items():
                        metric = calculate_metrics(similarity, level_mask, level_valid)
                        row = {
                            'sample_id': sample['sample_id'],
                            'filename': filename,
                            'condition': condition,
                            'method': method,
                            'modality': modality,
                            'level': level,
                            'feature_shape': list(feature.shape[1:]),
                            'status': 'valid' if metric else 'mask_vanished_or_invalid',
                            'caption': sample['caption'],
                            'used_caption': sample['caption'] if condition == 'correct'
                            else wrong_captions[sample_index],
                            'valid_tokens': token_names,
                        }
                        if metric:
                            row.update(metric)
                            if method == 'max':
                                target_indices = max_indices[level_mask & level_valid]
                                counts = Counter(int(x) for x in target_indices.cpu())
                                if counts:
                                    best_index, best_count = counts.most_common(1)[0]
                                    row['dominant_target_token'] = token_names[best_index]
                                    row['dominant_target_token_fraction'] = (
                                        best_count / sum(counts.values()))
                                    dominant_tokens[(condition, modality, level,
                                                     token_names[best_index])] += 1
                        rows.append(row)
                        plot_maps[(condition, method, modality, level)] = (
                            similarity.cpu().numpy(), metric)
        stem = f"{sample['sample_id']:06d}_{Path(filename).stem}"
        for condition in ('correct', 'wrong'):
            for method in ('mean', 'max'):
                render(visualization_dir / f'{stem}_{condition}_{method}.png',
                       vis_rgb, ir_rgb, mask, plot_maps, sample, method, condition)
        print(f'[{sample_index + 1:02d}/10] {filename}', flush=True)

    with (output_dir / 'text_similarity_metrics.jsonl').open(
            'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {
        'num_samples': len(records),
        'levels': [0, 1, 2, 3],
        'methods': ['mean_token_embedding_then_cosine',
                    'per_token_cosine_then_max'],
        'conditions': ['correct_caption', 'cyclically_mismatched_caption'],
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'ir_lora_enabled': False,
        'summary': summarize(rows),
        'dominant_target_tokens': [
            {'condition': key[0], 'modality': key[1], 'level': key[2],
             'token': key[3], 'sample_count': count}
            for key, count in dominant_tokens.most_common()],
    }
    (output_dir / 'text_similarity_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Finished: {output_dir}')


if __name__ == '__main__':
    main()
