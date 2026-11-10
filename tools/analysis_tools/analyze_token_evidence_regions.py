"""Compare token evidence/weights globally, inside GT, and outside GT."""

import argparse
import gc
import json
from pathlib import Path

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
    parser.add_argument('--model', action='append', nargs=3, required=True,
                        metavar=('NAME', 'CONFIG', 'CHECKPOINT'))
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def region_evidence(similarity, shape, image_shape, batch_shape, region):
    """Average every token's upsampled similarity within a pixel region."""
    h, w = map(int, shape)
    image_h, image_w = image_shape
    batch_h, batch_w = batch_shape
    token_maps = similarity[0].transpose(0, 1).reshape(1, -1, h, w)
    token_maps = F.interpolate(
        token_maps.float(), (batch_h, batch_w), mode='bilinear',
        align_corners=False)[0, :, :image_h, :image_w]
    return token_maps[:, region].mean(dim=1)


def serialize_distribution(tokens, evidence, weight, valid_tokens):
    rows = []
    for token_id in torch.where(valid_tokens)[0].tolist():
        rows.append(dict(
            token_index=token_id,
            token=tokens[token_id],
            evidence=float(evidence[token_id]),
            weight=float(weight[token_id])))
    rows.sort(key=lambda row: row['weight'], reverse=True)
    return rows


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

            vis = captured['vis_feat'].float()
            ir = captured['ir_feat'].float()
            text = captured['text_feat'].float()
            valid, valid_tokens = fusion._validate_masks(
                vis, text, captured['feat_mask'].bool(),
                captured['text_token_mask'].bool())
            ranges = fusion._level_ranges(
                captured['spatial_shapes'], captured['level_start_index'],
                vis.size(1))

            image_h, image_w = sample.img_shape[:2]
            box = sample.gt_instances.bboxes.tensor[0].float()
            yy, xx = torch.meshgrid(
                torch.arange(image_h), torch.arange(image_w), indexing='ij')
            inside = ((xx >= box[0]) & (xx <= box[2]) &
                      (yy >= box[1]) & (yy <= box[3]))
            outside = ~inside

            tokenized = model.language_model.tokenizer(
                sample.text, return_tensors='pt')
            tokens = model.language_model.tokenizer.convert_ids_to_tokens(
                tokenized['input_ids'][0])
            result = dict(
                model=model_name,
                sample_index=index,
                caption=sample.text,
                resized_image_shape=[image_h, image_w],
                resized_gt_box=box.tolist(),
                gt_area_ratio=float(inside.float().mean()),
                modalities={})

            for modality, feature in [('vis', vis), ('ir', ir)]:
                global_levels = []
                inside_levels = []
                outside_levels = []
                per_level = []
                for level, ((start, end), shape) in enumerate(zip(
                        ranges, captured['spatial_shapes'])):
                    similarity, global_evidence = \
                        fusion._similarity_and_evidence(
                            feature[:, start:end], text, valid[:, start:end])
                    inside_evidence = region_evidence(
                        similarity, shape, (image_h, image_w),
                        prediction.batch_input_shape, inside)
                    outside_evidence = region_evidence(
                        similarity, shape, (image_h, image_w),
                        prediction.batch_input_shape, outside)
                    global_levels.append(global_evidence[0])
                    inside_levels.append(inside_evidence)
                    outside_levels.append(outside_evidence)
                    per_level.append(dict(
                        level=level,
                        global_evidence=global_evidence[0].tolist(),
                        inside_evidence=inside_evidence.tolist(),
                        outside_evidence=outside_evidence.tolist()))

                evidences = {
                    'global': torch.stack(global_levels).mean(0),
                    'inside_gt': torch.stack(inside_levels).mean(0),
                    'outside_gt': torch.stack(outside_levels).mean(0),
                }
                distributions = {}
                for region_name, evidence in evidences.items():
                    weight, temperature = fusion._token_weight(
                        evidence[None], valid_tokens)
                    distributions[region_name] = dict(
                        temperature=float(temperature[0, 0]),
                        tokens=serialize_distribution(
                            tokens, evidence, weight[0], valid_tokens[0]))

                result['modalities'][modality] = dict(
                    distributions=distributions,
                    per_level=per_level)

                valid_ids = torch.where(valid_tokens[0])[0]
                global_top = int(valid_ids[
                    evidences['global'][valid_ids].argmax()])
                inside_top = int(valid_ids[
                    evidences['inside_gt'][valid_ids].argmax()])
                outside_top = int(valid_ids[
                    evidences['outside_gt'][valid_ids].argmax()])
                summary.append(dict(
                    model=model_name, sample_index=index, modality=modality,
                    global_top=tokens[global_top],
                    inside_top=tokens[inside_top],
                    outside_top=tokens[outside_top],
                    target_candidates=[row['token'] for row in
                                       distributions['inside_gt']['tokens'][:5]]))

            output_path = out_dir / f'{model_name}_sample{index}_regions.json'
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2))

        del model, dataset
        gc.collect()

    (out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print('model sample modality | global_top inside_top outside_top')
    for row in summary:
        print(row['model'], row['sample_index'], row['modality'], '|',
              row['global_top'], row['inside_top'], row['outside_top'])


if __name__ == '__main__':
    main()
