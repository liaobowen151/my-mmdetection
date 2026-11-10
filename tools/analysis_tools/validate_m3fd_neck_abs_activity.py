#!/usr/bin/env python3
"""Validate |feature|-mean activity against M3FD D3 pseudo masks.

This is deliberately an offline analysis: it builds only the visual backbone
and neck, loads their weights from a Grounding DINO checkpoint, disables the
IR LoRA residual, and never runs multimodal fusion or the detector head.
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmdet.registry import MODELS

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


FORCED_MASKS = (
    '000001_03220.png',
    '000005_00686.png',
    '000002_03220.png',
    '000015_00337.png',
    '000020_04170.png',
)
TOPK_FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='data/M3FD')
    parser.add_argument(
        '--annotations',
        default='data/M3FD/annotations/rgbtvg_m3fd_train_seg.jsonl')
    parser.add_argument(
        '--output-dir',
        default='analysis_demo/swin_neck_abs_activity_m3fd_20')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument(
        '--use-all-annotations', action='store_true',
        help='Use every row in --annotations in file order; useful for an '
             'already selected test subset and does not require forced masks.')
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=224)
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_samples(records, data_root, count, seed, use_all_annotations=False):
    if use_all_annotations:
        if count != len(records):
            raise ValueError(
                '--use-all-annotations requires --num-samples to equal the '
                f'annotation row count ({len(records)}).')
        return records
    by_mask = {Path(item['mask_path']).name: item for item in records}
    missing = [name for name in FORCED_MASKS if name not in by_mask]
    if missing:
        raise KeyError(f'Forced masks absent from annotations: {missing}')
    forced = [by_mask[name] for name in FORCED_MASKS]
    forced_ids = {item['sample_id'] for item in forced}
    candidates = []
    for item in records:
        if item['sample_id'] in forced_ids:
            continue
        paths = (
            data_root / 'vis' / item['filename'],
            data_root / 'ir' / item['filename'],
            data_root / item['mask_path'])
        if all(path.is_file() for path in paths):
            candidates.append(item)
    if count < len(forced):
        raise ValueError(f'num-samples must be at least {len(forced)}')
    rng = random.Random(seed)
    selected = forced + rng.sample(candidates, count - len(forced))
    return selected


def build_backbone_and_neck(config_path, checkpoint_path, device):
    # Registration imports are intentional and local to this analysis model.
    import projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino  # noqa: F401

    cfg = Config.fromfile(config_path)
    backbone_cfg = cfg.model.backbone.copy()
    backbone_cfg['init_cfg'] = None  # The explicit checkpoint below is authoritative.
    backbone = MODELS.build(backbone_cfg)
    neck = MODELS.build(cfg.model.neck)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    backbone_state = {
        key[len('backbone.'):]: value
        for key, value in state.items() if key.startswith('backbone.')
    }
    neck_state = {
        key[len('neck.'):]: value
        for key, value in state.items() if key.startswith('neck.')
    }
    backbone_report = backbone.load_state_dict(backbone_state, strict=False)
    neck_report = neck.load_state_dict(neck_state, strict=False)
    backbone.to(device).eval()
    neck.to(device).eval()
    if hasattr(backbone, 'set_ir_lora_enabled'):
        backbone.set_ir_lora_enabled(False)
    report = {
        'backbone_missing_keys': list(backbone_report.missing_keys),
        'backbone_unexpected_keys': list(backbone_report.unexpected_keys),
        'neck_missing_keys': list(neck_report.missing_keys),
        'neck_unexpected_keys': list(neck_report.unexpected_keys),
        'note': ('IR LoRA residual disabled: VIS and IR use the same pretrained '
                 'Swin base, followed by the same pretrained neck.'),
    }
    return backbone, neck, report


def resize_and_pad(image_rgb, mask, size):
    height, width = image_rgb.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = cv2.resize(image_rgb, (new_width, new_height),
                       interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_width, new_height),
                              interpolation=cv2.INTER_NEAREST)
    image_pad = np.zeros((size, size, 3), dtype=np.float32)
    mask_pad = np.zeros((size, size), dtype=np.uint8)
    valid_pad = np.zeros((size, size), dtype=np.uint8)
    image_pad[:new_height, :new_width] = image.astype(np.float32)
    mask_pad[:new_height, :new_width] = resized_mask > 0
    valid_pad[:new_height, :new_width] = 1
    return image_pad, mask_pad, valid_pad, (new_height, new_width)


def image_tensor(image, device):
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
    normalized = (image - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)


def calculate_metrics(activity, mask, valid):
    values = activity[valid]
    labels = mask[valid]
    positives = int(labels.sum())
    total = int(labels.numel())
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return None
    inside = float(values[labels].mean())
    outside = float(values[~labels].mean())
    result = {
        'valid_pixels': total,
        'mask_pixels': positives,
        'mask_ratio': positives / total,
        'inside_mean': inside,
        'outside_mean': outside,
        'R': inside / (outside + 1e-12),
    }
    for fraction in TOPK_FRACTIONS:
        count = max(1, math.ceil(fraction * total))
        indices = torch.topk(values, count, largest=True, sorted=False).indices
        precision = float(labels[indices].float().mean())
        gain = precision / (result['mask_ratio'] + 1e-12)
        suffix = f'{int(fraction * 100):02d}'
        result[f'P_{suffix}'] = precision
        result[f'G_{suffix}'] = gain
    return result


def resize_bool_map(array, shape, device):
    tensor = torch.from_numpy(array.astype(np.float32))[None, None].to(device)
    return F.interpolate(tensor, size=shape, mode='nearest')[0, 0] > 0.5


def render_visualization(output_path, vis_rgb, ir_rgb, original_mask,
                         activities, sample):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes[0, 0].imshow(vis_rgb)
    axes[0, 0].contour(original_mask > 0, levels=[0.5], colors='lime', linewidths=1)
    axes[0, 0].set_title('VIS + D3 mask')
    axes[0, 1].imshow(ir_rgb)
    axes[0, 1].contour(original_mask > 0, levels=[0.5], colors='lime', linewidths=1)
    axes[0, 1].set_title('IR + D3 mask')
    axes[0, 2].imshow(original_mask, cmap='gray')
    axes[0, 2].set_title('D3 pseudo mask')
    axes[0, 3].axis('off')
    axes[0, 3].text(
        0, 1,
        f"sample={sample['sample_id']}\n{sample['filename']}\n"
        f"{sample.get('caption', '')}", va='top', wrap=True)
    for row, modality in enumerate(('vis', 'ir'), start=1):
        for level in range(4):
            activity, metric = activities[(modality, level)]
            axes[row, level].imshow(activity, cmap='inferno')
            if metric is None:
                title = f'{modality.upper()} L{level}: mask vanished'
            else:
                title = (f"{modality.upper()} L{level}  R={metric['R']:.3f}\n"
                         f"P5={metric['P_05']:.3f}, G5={metric['G_05']:.3f}")
            axes[row, level].set_title(title)
    for axis in axes.flat:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def summarize(rows):
    summary = {}
    for modality in ('vis', 'ir'):
        summary[modality] = {}
        for level in range(4):
            subset = [row for row in rows
                      if row['modality'] == modality and row['level'] == level
                      and row['status'] == 'valid']
            vanished = sum(row['modality'] == modality and row['level'] == level
                           and row['status'] != 'valid' for row in rows)
            level_summary = {'valid_samples': len(subset),
                             'mask_vanished_or_invalid': vanished}
            for key in ('R', 'P_01', 'G_01', 'P_05', 'G_05',
                        'P_10', 'G_10', 'P_20', 'G_20'):
                values = np.asarray([row[key] for row in subset], dtype=np.float64)
                if values.size:
                    level_summary[key] = {
                        'mean': float(values.mean()),
                        'median': float(np.median(values)),
                        'std': float(values.std()),
                        'fraction_gt_1': float((values > 1).mean())
                        if key.startswith(('R', 'G_')) else None,
                    }
            summary[modality][f'level_{level}'] = level_summary
    return summary


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.annotations)
    selected = select_samples(records, data_root, args.num_samples, args.seed,
                              args.use_all_annotations)
    backbone, neck, load_report = build_backbone_and_neck(
        args.config, args.checkpoint, args.device)
    with open(output_dir / 'checkpoint_load_report.json', 'w', encoding='utf-8') as f:
        json.dump(load_report, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'selected_samples.json', 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    rows = []
    for index, sample in enumerate(selected, start=1):
        filename = sample['filename']
        vis_bgr = cv2.imread(str(data_root / 'vis' / filename), cv2.IMREAD_COLOR)
        ir_bgr = cv2.imread(str(data_root / 'ir' / filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(data_root / sample['mask_path']), cv2.IMREAD_GRAYSCALE)
        if vis_bgr is None or ir_bgr is None or mask is None:
            raise FileNotFoundError(f'Failed to read inputs for sample {sample["sample_id"]}')
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
        if vis_rgb.shape[:2] != ir_rgb.shape[:2] or vis_rgb.shape[:2] != mask.shape:
            raise ValueError(f'Unaligned inputs for sample {sample["sample_id"]}')

        vis_input, mask_pad, valid_pad, resized_shape = resize_and_pad(
            vis_rgb, mask, args.input_size)
        ir_input, _, _, _ = resize_and_pad(ir_rgb, mask, args.input_size)
        with torch.inference_mode():
            if hasattr(backbone, 'set_ir_lora_enabled'):
                backbone.set_ir_lora_enabled(False)
            vis_features = neck(backbone(image_tensor(vis_input, args.device)))
            ir_features = neck(backbone(image_tensor(ir_input, args.device)))

        plot_data = {}
        for modality, features in (('vis', vis_features), ('ir', ir_features)):
            for level, feature in enumerate(features):
                activity = feature[0].abs().mean(dim=0)
                level_mask = resize_bool_map(mask_pad, activity.shape, args.device)
                level_valid = resize_bool_map(valid_pad, activity.shape, args.device)
                metric = calculate_metrics(activity, level_mask, level_valid)
                base = {
                    'sample_id': sample['sample_id'],
                    'filename': filename,
                    'mask_path': sample['mask_path'],
                    'forced': (not args.use_all_annotations and
                               Path(sample['mask_path']).name in FORCED_MASKS),
                    'modality': modality,
                    'level': level,
                    'feature_shape': list(feature.shape[1:]),
                    'resized_valid_shape': list(resized_shape),
                    'status': 'valid' if metric is not None else 'mask_vanished_or_invalid',
                }
                if metric is not None:
                    base.update(metric)
                rows.append(base)
                plot_data[(modality, level)] = (activity.cpu().numpy(), metric)
        render_visualization(
            vis_dir / f"{sample['sample_id']:06d}_{Path(filename).stem}.png",
            vis_rgb, ir_rgb, mask, plot_data, sample)
        print(f'[{index:02d}/{len(selected):02d}] sample={sample["sample_id"]} {filename}')

    with open(output_dir / 'per_sample_metrics.jsonl', 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {
        'experiment': 'mean_channel_absolute_neck_activity',
        'num_samples': len(selected),
        'seed': args.seed,
        'input_size': args.input_size,
        'forced_masks': [] if args.use_all_annotations else list(FORCED_MASKS),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'ir_lora_enabled': False,
        'summary': summarize(rows),
    }
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'Finished. Results: {output_dir}')


if __name__ == '__main__':
    main()
