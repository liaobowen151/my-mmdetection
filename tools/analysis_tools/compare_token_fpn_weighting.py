"""Compare level-balanced, position-balanced, and per-level token weights."""

import argparse
import gc
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import default_collate
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet.apis import init_detector
from mmdet.registry import DATASETS


TARGETS = {788: 'bus', 1548: 'person', 1806: 'person',
           1643: 'sedan', 1312: 'suv'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='append', nargs=3, required=True,
                        metavar=('NAME', 'CONFIG', 'CHECKPOINT'))
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--subjects-json',
                        help='Output of select_subject_samples.py. Overrides '
                        '--indices and the built-in five-sample targets.')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def distribution(tokens, evidence, valid_tokens, temperature, target):
    masked = evidence.masked_fill(~valid_tokens, float('-inf'))
    weights = torch.softmax(masked.float() / temperature, dim=-1)
    valid_ids = torch.where(valid_tokens)[0]
    ordered = valid_ids[weights[valid_ids].argsort(descending=True)]
    rows = [dict(token_index=int(i), token=tokens[int(i)],
                 evidence=float(evidence[i]), weight=float(weights[i]))
            for i in ordered]
    target_rows = [row for row in rows if row['token'] == target]
    target_row = target_rows[0] if target_rows else None
    if target_row is not None:
        target_rank = next(i + 1 for i, row in enumerate(rows)
                           if row['token_index'] == target_row['token_index'])
        margin = target_row['evidence'] - rows[0]['evidence']
    else:
        target_rank, margin = None, None
    return dict(top_tokens=rows, target_token=target,
                target_rank=target_rank,
                target_weight=None if target_row is None else target_row['weight'],
                target_evidence=None if target_row is None else target_row['evidence'],
                target_minus_top1_evidence=margin)


def main():
    args = parse_args()
    if args.subjects_json:
        sample_spec = json.loads(Path(args.subjects_json).read_text())['samples']
        targets = {item['sample_index']: item['subject'] for item in sample_spec}
        indices = list(targets)
    else:
        if not args.indices:
            raise ValueError('Provide --indices or --subjects-json.')
        targets = TARGETS
        indices = args.indices
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

        for index in indices:
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
                model.test_step(default_collate([item]))
            handle.remove()

            vis = captured['vis_feat'].float()
            ir = captured['ir_feat'].float()
            text = captured['text_feat'].float()
            valid, valid_tokens = fusion._validate_masks(
                vis, text, captured['feat_mask'].bool(),
                captured['text_token_mask'].bool())
            ranges = fusion._level_ranges(
                captured['spatial_shapes'], captured['level_start_index'],
                vis.size(1))
            temperature = float(fusion._token_temperature(valid_tokens)[0, 0])
            encoded = model.language_model.tokenizer(
                sample.text, return_tensors='pt')
            tokens = model.language_model.tokenizer.convert_ids_to_tokens(
                encoded['input_ids'][0])
            target = targets[index]
            result = dict(model=model_name, sample_index=index,
                          caption=sample.text, target_token=target,
                          temperature=temperature, modalities={})

            for modality, feature in [('vis', vis), ('ir', ir)]:
                evidences, counts, per_level = [], [], []
                for level, (start, end) in enumerate(ranges):
                    _, evidence = fusion._similarity_and_evidence(
                        feature[:, start:end], text, valid[:, start:end])
                    count = int(valid[0, start:end].sum())
                    evidences.append(evidence[0])
                    counts.append(count)
                    per_level.append(dict(
                        level=level,
                        spatial_shape=captured['spatial_shapes'][level].tolist(),
                        valid_position_count=count,
                        distribution=distribution(
                            tokens, evidence[0], valid_tokens[0],
                            temperature, target)))

                stacked = torch.stack(evidences)
                level_balanced = stacked.mean(0)
                position_weights = torch.tensor(
                    counts, dtype=stacked.dtype) / sum(counts)
                position_balanced = (
                    stacked * position_weights[:, None]).sum(0)
                modes = dict(
                    level_balanced_shared=distribution(
                        tokens, level_balanced, valid_tokens[0],
                        temperature, target),
                    position_balanced_shared=distribution(
                        tokens, position_balanced, valid_tokens[0],
                        temperature, target),
                    per_level=per_level)
                modes['position_balanced_shared']['level_weights'] = \
                    position_weights.tolist()
                result['modalities'][modality] = modes

                for mode_name in ('level_balanced_shared',
                                  'position_balanced_shared'):
                    dist = modes[mode_name]
                    summary.append(dict(
                        model=model_name, sample_index=index,
                        modality=modality, mode=mode_name,
                        target=target,
                        top1=dist['top_tokens'][0]['token'],
                        target_rank=dist['target_rank'],
                        target_weight=dist['target_weight'],
                        target_minus_top1_evidence=
                        dist['target_minus_top1_evidence']))
                for level_result in per_level:
                    dist = level_result['distribution']
                    summary.append(dict(
                        model=model_name, sample_index=index,
                        modality=modality,
                        mode=f"per_level_L{level_result['level']}",
                        target=target,
                        top1=dist['top_tokens'][0]['token'],
                        target_rank=dist['target_rank'],
                        target_weight=dist['target_weight'],
                        target_minus_top1_evidence=
                        dist['target_minus_top1_evidence']))

            (out_dir / f'{model_name}_sample{index}.json').write_text(
                json.dumps(result, ensure_ascii=False, indent=2))

        del model, dataset
        gc.collect()

    (out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print('model sample mod mode | top1 target rank weight margin')
    for row in summary:
        print(row['model'], row['sample_index'], row['modality'], row['mode'],
              '|', row['top1'], row['target'], row['target_rank'],
              f"{row['target_weight']:.3g}",
              f"{row['target_minus_top1_evidence']:.3f}")


if __name__ == '__main__':
    main()
