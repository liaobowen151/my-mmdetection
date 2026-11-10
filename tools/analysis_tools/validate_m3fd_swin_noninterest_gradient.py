#!/usr/bin/env python3
"""Validate Swin/neck non-interest gradient modality weights.

The validation follows TextFusion's fixed 3x3 high-pass operator, computes a
VIS/IR softmax independently at every neck level, then averages the four
level-wise weights (method B). Image-space high-pass energy is the independent
reference; VGG is intentionally not used.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

from validate_m3fd_neck_abs_activity import (
    build_backbone_and_neck, image_tensor, read_jsonl, resize_and_pad,
    resize_bool_map)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


TEMPERATURES = (0.1, 0.5, 1.0, 2.0, 5.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--subset', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=224)
    return parser.parse_args()


def high_pass_energy(tensor):
    """TextFusion high-pass squared, channel-averaged; input is BCHW."""
    channels = tensor.shape[1]
    kernel = tensor.new_tensor([
        [1 / 8, 1 / 8, 1 / 8],
        [1 / 8, -1.0, 1 / 8],
        [1 / 8, 1 / 8, 1 / 8],
    ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    padded = F.pad(tensor, (1, 1, 1, 1), mode='reflect')
    response = F.conv2d(padded, kernel, groups=channels)
    return response.square().mean(dim=1)[0]


def masked_mean(values, weights):
    denominator = weights.sum()
    if denominator <= 0:
        return None
    return float((values * weights).sum() / denominator)


def modality_weight(vis_energy, ir_energy, temperature):
    energies = torch.tensor([vis_energy, ir_energy], dtype=torch.float64)
    return torch.softmax(energies / temperature, dim=0).tolist()


def rankdata(values):
    """Average ranks for ties, sufficient for small validation arrays."""
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def correlation(x, y, rank=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if rank:
        x, y = rankdata(x), rankdata(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def entropy_binary(probability):
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def render(output_path, vis_rgb, ir_rgb, mask, valid_shape, vis_maps, ir_maps,
           record):
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes[0, 0].imshow(vis_rgb)
    axes[0, 0].contour(mask > 0, levels=[0.5], colors='lime', linewidths=1)
    axes[0, 0].set_title('VIS + D3 mask')
    axes[0, 1].imshow(ir_rgb)
    axes[0, 1].contour(mask > 0, levels=[0.5], colors='lime', linewidths=1)
    axes[0, 1].set_title('IR + D3 mask')
    axes[0, 2].imshow(1 - (mask > 0), cmap='gray')
    axes[0, 2].set_title('non-interest region')
    axes[0, 3].axis('off')
    level_text = '\n'.join(
        f"L{x['level']}: VIS={x['g_vis']:.3e}, IR={x['g_ir']:.3e}, "
        f"wIR@1={x['weights']['1.0']['ir']:.3f}"
        for x in record['levels'])
    axes[0, 3].text(
        0, 1, f"{record['subset']} / {record['filename']}\n"
        f"image log-ratio={record['image_log_ratio']:+.3f}\n"
        f"Swin log-ratio={record['swin_log_ratio_mean']:+.3f}\n"
        f"final wIR@1={record['temperatures']['1.0']['ir']:.3f}\n"
        f"{level_text}", va='top', family='monospace')
    for level in range(4):
        axes[1, level].imshow(vis_maps[level], cmap='inferno')
        axes[1, level].set_title(f'VIS L{level} gradient')
        axes[2, level].imshow(ir_maps[level], cmap='inferno')
        axes[2, level].set_title(f'IR L{level} gradient')
    temperatures = [str(x) for x in TEMPERATURES]
    vis_weights = [record['temperatures'][t]['vis'] for t in temperatures]
    ir_weights = [record['temperatures'][t]['ir'] for t in temperatures]
    axes[3, 0].plot(TEMPERATURES, vis_weights, marker='o', label='VIS')
    axes[3, 0].plot(TEMPERATURES, ir_weights, marker='o', label='IR')
    axes[3, 0].set_xscale('log')
    axes[3, 0].set_ylim(0, 1)
    axes[3, 0].set_title('averaged method-B weights')
    axes[3, 0].legend()
    level_ir = [x['weights']['1.0']['ir'] for x in record['levels']]
    axes[3, 1].bar(range(4), level_ir)
    axes[3, 1].axhline(0.5, color='black', linestyle='--')
    axes[3, 1].set_ylim(0, 1)
    axes[3, 1].set_title('per-level IR weight @ tau=1')
    axes[3, 2].bar(['image', 'Swin'], [
        record['image_ir_energy_fraction'],
        record['swin_ir_energy_fraction_geomean']])
    axes[3, 2].set_ylim(0, 1)
    axes[3, 2].set_title('scale-free IR energy fraction')
    axes[3, 3].axis('off')
    axes[3, 3].text(0, 1, f'valid resized shape={valid_shape}', va='top')
    for axis in axes[:3].flat:
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def summarize(records):
    summary = {'num_samples': len(records), 'temperatures': {}}
    image_ratios = [x['image_log_ratio'] for x in records]
    swin_ratios = [x['swin_log_ratio_mean'] for x in records]
    signs = [np.sign(x) for x in image_ratios]
    swin_signs = [np.sign(x) for x in swin_ratios]
    summary['reference_comparison'] = {
        'direction_agreement': float(np.mean(np.asarray(signs) == np.asarray(swin_signs))),
        'pearson': correlation(swin_ratios, image_ratios, rank=False),
        'spearman': correlation(swin_ratios, image_ratios, rank=True),
        'image_ir_dominant_fraction': float(np.mean(np.asarray(image_ratios) > 0)),
        'swin_ir_dominant_fraction': float(np.mean(np.asarray(swin_ratios) > 0)),
    }
    all_level_signs = np.asarray([
        [np.sign(level['log_ratio']) for level in x['levels']]
        for x in records])
    majority = []
    complete = []
    for row in all_level_signs:
        positive = int((row > 0).sum())
        negative = int((row < 0).sum())
        majority.append(max(positive, negative) >= 3)
        complete.append(positive == 4 or negative == 4)
    summary['cross_level_consistency'] = {
        'at_least_three_agree_fraction': float(np.mean(majority)),
        'all_four_agree_fraction': float(np.mean(complete)),
    }
    for temperature in TEMPERATURES:
        key = str(temperature)
        ir_weights = np.asarray([x['temperatures'][key]['ir'] for x in records])
        summary['temperatures'][key] = {
            'ir_mean': float(ir_weights.mean()),
            'ir_median': float(np.median(ir_weights)),
            'ir_dominant_fraction': float((ir_weights > 0.5).mean()),
            'saturated_fraction': float(((ir_weights < 0.05) |
                                         (ir_weights > 0.95)).mean()),
            'entropy_mean': float(np.mean([entropy_binary(x) for x in ir_weights])),
        }
    summary['per_level'] = {}
    for level in range(4):
        level_swin = [x['levels'][level]['log_ratio'] for x in records]
        summary['per_level'][f'level_{level}'] = {
            'direction_agreement_with_image': float(np.mean(
                np.sign(level_swin) == np.sign(image_ratios))),
            'pearson_with_image': correlation(level_swin, image_ratios),
            'spearman_with_image': correlation(level_swin, image_ratios, rank=True),
            'ir_dominant_fraction': float(np.mean(np.asarray(level_swin) > 0)),
        }
    return summary


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    vis_output = output_dir / 'noninterest_gradient_visualizations'
    vis_output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.annotations)
    if len(records) != 10:
        raise ValueError(f'Expected 10 selected samples, got {len(records)}')
    backbone, neck, load_report = build_backbone_and_neck(
        args.config, args.checkpoint, args.device)
    (output_dir / 'noninterest_checkpoint_load_report.json').write_text(
        json.dumps(load_report, indent=2, ensure_ascii=False), encoding='utf-8')

    results = []
    for index, sample in enumerate(records, start=1):
        filename = sample['filename']
        vis_bgr = cv2.imread(str(data_root / 'vis' / filename), cv2.IMREAD_COLOR)
        ir_bgr = cv2.imread(str(data_root / 'ir' / filename), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(Path(sample['mask_path'])), cv2.IMREAD_GRAYSCALE)
        if vis_bgr is None or ir_bgr is None or mask is None:
            raise FileNotFoundError(f'Inputs missing for {filename}')
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)
        vis_input, mask_pad, valid_pad, valid_shape = resize_and_pad(
            vis_rgb, mask, args.input_size)
        ir_input, _, _, _ = resize_and_pad(ir_rgb, mask, args.input_size)

        with torch.inference_mode():
            if hasattr(backbone, 'set_ir_lora_enabled'):
                backbone.set_ir_lora_enabled(False)
            vis_features = neck(backbone(image_tensor(vis_input, args.device)))
            ir_features = neck(backbone(image_tensor(ir_input, args.device)))

        level_records = []
        vis_maps, ir_maps = [], []
        eps = 1e-12
        for level, (vis_feature, ir_feature) in enumerate(
                zip(vis_features, ir_features)):
            vis_gradient = high_pass_energy(vis_feature)
            ir_gradient = high_pass_energy(ir_feature)
            shape = vis_gradient.shape
            # area interpolation preserves partial target occupancy.
            mask_tensor = torch.from_numpy(mask_pad.astype(np.float32))[None, None].to(args.device)
            soft_mask = F.interpolate(mask_tensor, size=shape, mode='area')[0, 0]
            valid = resize_bool_map(valid_pad, shape, args.device).float()
            noninterest = valid * (1 - soft_mask)
            g_vis = masked_mean(vis_gradient, noninterest)
            g_ir = masked_mean(ir_gradient, noninterest)
            if g_vis is None or g_ir is None:
                raise RuntimeError(f'Empty non-interest region for {filename}, level {level}')
            weights = {}
            for temperature in TEMPERATURES:
                vis_weight, ir_weight = modality_weight(g_vis, g_ir, temperature)
                weights[str(temperature)] = {'vis': vis_weight, 'ir': ir_weight}
            level_records.append({
                'level': level,
                'shape': list(shape),
                'g_vis': g_vis,
                'g_ir': g_ir,
                'log_ratio': math.log((g_ir + eps) / (g_vis + eps)),
                'ir_energy_fraction': g_ir / (g_vis + g_ir + eps),
                'weights': weights,
            })
            vis_maps.append(vis_gradient.cpu().numpy())
            ir_maps.append(ir_gradient.cpu().numpy())

        # Independent reference at original image resolution, normalized [0, 1].
        vis_gray = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        ir_gray = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        image_noninterest = torch.from_numpy((mask == 0).astype(np.float32)).to(args.device)
        vis_gray_t = torch.from_numpy(vis_gray)[None, None].to(args.device)
        ir_gray_t = torch.from_numpy(ir_gray)[None, None].to(args.device)
        with torch.inference_mode():
            image_vis_gradient = high_pass_energy(vis_gray_t)
            image_ir_gradient = high_pass_energy(ir_gray_t)
        image_g_vis = masked_mean(image_vis_gradient, image_noninterest)
        image_g_ir = masked_mean(image_ir_gradient, image_noninterest)
        if image_g_vis is None or image_g_ir is None:
            raise RuntimeError(f'Empty image-space non-interest region for {filename}')

        temperatures = {}
        for temperature in TEMPERATURES:
            key = str(temperature)
            temperatures[key] = {
                'vis': float(np.mean([x['weights'][key]['vis'] for x in level_records])),
                'ir': float(np.mean([x['weights'][key]['ir'] for x in level_records])),
            }
        log_ratios = [x['log_ratio'] for x in level_records]
        record = {
            'subset': args.subset,
            'sample_id': sample['sample_id'],
            'filename': filename,
            'caption': sample.get('caption', ''),
            'mask_path': sample['mask_path'],
            'mask_status': sample.get('status'),
            'mask_fallback': sample.get('fallback'),
            'image_g_vis': image_g_vis,
            'image_g_ir': image_g_ir,
            'image_log_ratio': math.log((image_g_ir + eps) / (image_g_vis + eps)),
            'image_ir_energy_fraction': image_g_ir / (image_g_vis + image_g_ir + eps),
            'swin_log_ratio_mean': float(np.mean(log_ratios)),
            'swin_ir_energy_fraction_geomean': 1 / (1 + math.exp(-float(np.mean(log_ratios)))),
            'levels': level_records,
            'temperatures': temperatures,
        }
        results.append(record)
        render(vis_output / f"{sample['sample_id']:06d}_{Path(filename).stem}.png",
               vis_rgb, ir_rgb, mask, valid_shape, vis_maps, ir_maps, record)
        print(f'[{index:02d}/10] {args.subset} {filename}', flush=True)

    with (output_dir / 'noninterest_gradient_metrics.jsonl').open(
            'w', encoding='utf-8') as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    output = {
        'subset': args.subset,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'aggregation': 'method_B_level_softmax_then_equal_average',
        'reference': 'normalized_input_image_high_pass_noninterest_energy',
        'ir_lora_enabled': False,
        'summary': summarize(results),
    }
    (output_dir / 'noninterest_gradient_summary.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Finished: {output_dir}')


if __name__ == '__main__':
    main()
