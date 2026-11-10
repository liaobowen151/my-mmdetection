"""Probe text/image alignment against resized GT boxes.

The script intentionally measures alignment before the Grounding DINO encoder.
It writes one JSON summary and one heat-map figure for every sample/model pair.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet.apis import init_detector
from mmdet.registry import DATASETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='append', nargs=3,
                        metavar=('NAME', 'CONFIG', 'CHECKPOINT'), required=True)
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def auc_score(score, target):
    score = score.flatten()
    target = target.flatten().bool()
    pos, neg = int(target.sum()), int((~target).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float32)
    return float((ranks[target].sum() - pos * (pos + 1) / 2) / (pos * neg))


def overlay(image, heat, box, title, path):
    heat = heat.astype(np.float32)
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-6)
    color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    shown = np.uint8(.55 * image + .45 * color)
    x1, y1, x2, y2 = [round(v) for v in box]
    cv2.rectangle(shown, (x1, y1), (x2, y2), (255, 255, 255), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(shown)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for model_name, config_path, checkpoint_path in args.model:
        cfg = Config.fromfile(config_path)
        if cfg.get('custom_imports'):
            import_modules_from_strings(**cfg.custom_imports)
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        dataset = DATASETS.build(cfg.test_dataloader.dataset)
        model = init_detector(cfg, checkpoint_path, device='cpu')
        fusion = model.multimodal_fusion

        for index in args.indices:
            item = dataset[index]
            sample = item['data_samples']
            captured = {}

            def hook(module, hook_args, kwargs):
                captured.update({
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                })

            handle = fusion.register_forward_pre_hook(hook, with_kwargs=True)
            with torch.no_grad():
                prediction = model.test_step(default_collate([item]))[0]
            handle.remove()

            vis, ir = captured['vis_feat'].float(), captured['ir_feat'].float()
            text = captured['text_feat'].float()
            valid, valid_tokens = fusion._validate_masks(
                vis, text, captured['feat_mask'].bool(),
                captured['text_token_mask'].bool())
            ranges = fusion._level_ranges(captured['spatial_shapes'],
                                          captured['level_start_index'],
                                          vis.size(1))
            vis_sims, ir_sims, vis_evidence, ir_evidence = [], [], [], []
            for start, end in ranges:
                vs, ve = fusion._similarity_and_evidence(
                    vis[:, start:end], text, valid[:, start:end])
                rs, re = fusion._similarity_and_evidence(
                    ir[:, start:end], text, valid[:, start:end])
                vis_sims.append(vs); ir_sims.append(rs)
                vis_evidence.append(ve); ir_evidence.append(re)
            vis_weight, _ = fusion._token_weight(
                torch.stack(vis_evidence, 1).mean(1), valid_tokens)
            ir_weight, _ = fusion._token_weight(
                torch.stack(ir_evidence, 1).mean(1), valid_tokens)

            tokenized = model.language_model.tokenizer(
                sample.text, return_tensors='pt')
            tokens = model.language_model.tokenizer.convert_ids_to_tokens(
                tokenized['input_ids'][0])
            top_tokens = {}
            for modality, weight in [('vis', vis_weight), ('ir', ir_weight)]:
                values, ids = weight[0].topk(min(5, int(valid_tokens[0].sum())))
                top_tokens[modality] = [
                    [tokens[int(i)], float(v)] for v, i in zip(values, ids)]

            img_h, img_w = sample.img_shape[:2]
            batch_h, batch_w = prediction.batch_input_shape
            box = sample.gt_instances.bboxes.tensor[0].float()
            yy, xx = torch.meshgrid(torch.arange(img_h), torch.arange(img_w),
                                    indexing='ij')
            gt = ((xx >= box[0]) & (xx <= box[2]) &
                  (yy >= box[1]) & (yy <= box[3]))
            image = cv2.cvtColor(cv2.imread(sample.img_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_w, img_h))

            for level, ((start, end), shape, vs, rs) in enumerate(zip(
                    ranges, captured['spatial_shapes'], vis_sims, ir_sims)):
                h, w = map(int, shape)
                for modality, sim, weight in (
                        ('vis', vs, vis_weight), ('ir', rs, ir_weight)):
                    raw = (sim * weight[:, None, :]).sum(-1)
                    gate_input, _, _ = fusion._normalize_spatial_alignment(
                        raw, valid[:, start:end])
                    raw_map = F.interpolate(raw.reshape(1, 1, h, w),
                                            (batch_h, batch_w),
                                            mode='bilinear',
                                            align_corners=False)[0, 0, :img_h, :img_w]
                    values_in, values_out = raw_map[gt], raw_map[~gt]
                    k = max(1, round(raw_map.numel() * .1))
                    top = raw_map.flatten().topk(k).indices
                    top_mask = torch.zeros(raw_map.numel(), dtype=torch.bool)
                    top_mask[top] = True
                    record = dict(
                        model=model_name, index=index, level=level,
                        modality=modality, caption=sample.text,
                        top_tokens=top_tokens[modality],
                        gt_area_ratio=float(gt.float().mean()),
                        inside_mean=float(values_in.mean()),
                        outside_mean=float(values_out.mean()),
                        inside_minus_outside=float(values_in.mean() - values_out.mean()),
                        pointing_hit=bool(gt.flatten()[raw_map.argmax()]),
                        top10_inside_ratio=float(gt.flatten()[top_mask].float().mean()),
                        top10_enrichment=float(
                            gt.flatten()[top_mask].float().mean() /
                            gt.float().mean().clamp_min(1e-8)),
                        auc=auc_score(raw_map, gt))
                    summary.append(record)
                    overlay(
                        image, raw_map.numpy(), box.tolist(),
                        f'{model_name} sample {index} L{level} {modality.upper()} '
                        f'delta={record["inside_minus_outside"]:.3f} '
                        f'AUC={record["auc"]:.3f}',
                        out_dir / f'{model_name}_sample{index}_L{level}_{modality}.png')

            pred = prediction.pred_instances
            pred_box = pred.bboxes[0].tolist() if len(pred) else None
            pred_score = float(pred.scores[0]) if len(pred) else None
            (out_dir / f'{model_name}_sample{index}_tokens.json').write_text(
                json.dumps(dict(caption=sample.text, top_tokens=top_tokens,
                                pred_box=pred_box, pred_score=pred_score),
                           ensure_ascii=False, indent=2))

        del model, dataset
        gc.collect()
        (out_dir / 'alignment_metrics.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2))

    (out_dir / 'alignment_metrics.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    for model_name, _, _ in args.model:
        rows = [r for r in summary if r['model'] == model_name]
        print(model_name, 'mean_delta=', np.mean([
            r['inside_minus_outside'] for r in rows]),
              'pointing=', np.mean([r['pointing_hit'] for r in rows]),
              'top10_enrichment=', np.mean([r['top10_enrichment'] for r in rows]),
              'auc=', np.mean([r['auc'] for r in rows]))


if __name__ == '__main__':
    main()
