import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert RGBT-Ground .pth annotations to MM-GDINO format.')

    parser.add_argument(
        'input',
        help='Input RGBT-Ground .pth annotation file.')

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory used to save converted annotation files.')

    parser.add_argument(
        '--image-dir',
        required=True,
        help='Image directory used to validate image existence and size.')

    parser.add_argument(
        '--split',
        required=True,
        help='Split name, such as train, val, test, testA.')

    parser.add_argument(
        '--dataset-prefix',
        default='rgbtvg_flir',
        help='Prefix of output annotation files.')

    parser.add_argument(
        '--generate-train-vg',
        action='store_true',
        help='Also generate ODVG JSONL annotation for training.')

    return parser.parse_args()


def parse_record(item, sample_id, image_dir):
    if not isinstance(item, (list, tuple)):
        raise TypeError(
            f'Sample {sample_id} must be list/tuple, but got {type(item)}.')

    if len(item) == 6:
        (
            file_name,
            image_size,
            bbox_xywh,
            expression,
            lighting,
            scale_cls,
        ) = item

        crowded = None
        scene = None
        weather = None
        occlusion = None
    elif len(item) == 10:
        (
            file_name,
            image_size,
            bbox_xywh,
            expression,
            lighting,
            scale_cls,
            crowded,
            scene,
            weather,
            occlusion,
        ) = item
    else:
        raise ValueError(
            f'Sample {sample_id} must contain 6 or 10 fields, '
            f'but got {len(item)}.')

    if not isinstance(file_name, str):
        raise TypeError(f'Invalid file name at sample {sample_id}.')

    if not isinstance(image_size, dict):
        raise TypeError(f'Invalid image size at sample {sample_id}.')

    height = int(image_size['height'])
    width = int(image_size['width'])

    if len(bbox_xywh) != 4:
        raise ValueError(f'Invalid bbox at sample {sample_id}: {bbox_xywh}')

    x, y, bbox_width, bbox_height = [
        float(value) for value in bbox_xywh
    ]

    caption = expression.strip()

    if not caption:
        raise ValueError(f'Empty expression at sample {sample_id}.')

    if width <= 0 or height <= 0:
        raise ValueError(
            f'Invalid image size at sample {sample_id}: '
            f'{width}x{height}')

    if x < 0 or y < 0 or bbox_width <= 0 or bbox_height <= 0:
        raise ValueError(
            f'Invalid bbox at sample {sample_id}: {bbox_xywh}')

    if x + bbox_width > width or y + bbox_height > height:
        raise ValueError(
            f'BBox outside image at sample {sample_id}: '
            f'bbox={bbox_xywh}, image={width}x{height}')

    image_path = os.path.join(image_dir, file_name)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f'Image not found at sample {sample_id}: {image_path}')

    with Image.open(image_path) as image:
        real_width, real_height = image.size

    if real_width != width or real_height != height:
        raise ValueError(
            f'Image size mismatch at sample {sample_id}: '
            f'annotation={width}x{height}, '
            f'image={real_width}x{real_height}, '
            f'path={image_path}')

    return {
        'sample_id': sample_id,
        'file_name': file_name,
        'height': height,
        'width': width,
        'caption': caption,

        # COCO/MDETR 使用 xywh。
        'bbox_xywh': [x, y, bbox_width, bbox_height],

        # ODVG 使用 xyxy。
        'bbox_xyxy': [
            x,
            y,
            x + bbox_width,
            y + bbox_height,
        ],

        # 方案 A：整句均为正文本区域。
        'tokens_positive': [[0, len(caption)]],

        'attributes': {
            'lighting': lighting,
            'scale_cls': scale_cls,
            'crowded': crowded,
            'scene': scene,
            'weather': weather,
            'occlusion': occlusion,
        }
    }


def build_mdetr_annotation(samples, split):
    images = []
    annotations = []

    for sample in samples:
        sample_id = sample['sample_id']
        x, y, bbox_width, bbox_height = sample['bbox_xywh']

        image_info = {
            'id': sample_id,
            'file_name': sample['file_name'],
            'height': sample['height'],
            'width': sample['width'],
            'original_id': sample['file_name'],
            'caption': sample['caption'],

            # 当前 RefExpMetric 只认识 refcoco/refcoco+/refcocog。
            'dataset_name': 'refcoco',

            # 与官方 MDETR 标注字段保持相似；当前读取器不使用。
            'tokens_negative': [],

            # 保留 RGBT-Ground 条件属性。
            **sample['attributes'],
        }

        annotation = {
            'id': sample_id,
            'image_id': sample_id,
            'category_id': 1,

            # MDETR/COCO 测试标注必须使用 xywh。
            'bbox': [x, y, bbox_width, bbox_height],

            'area': bbox_width * bbox_height,
            'iscrowd': 0,
            'original_id': sample_id,

            # 训练源标注使用；测试读取器会自动改为 -1。
            'tokens_positive': sample['tokens_positive'],
        }

        images.append(image_info)
        annotations.append(annotation)

    return {
        'info': {
            'description': f'RGBT-Ground RefFLIR REC {split} set',
            'version': '1.0',
        },
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': [
            {
                'id': 1,
                'name': 'referent',
                'supercategory': 'referent',
            }
        ],
    }


def build_odvg_record(sample):
    return {
        'filename': sample['file_name'],
        'height': sample['height'],
        'width': sample['width'],
        'grounding': {
            'caption': sample['caption'],
            'regions': [
                {
                    # ODVG 训练标注必须使用 xyxy。
                    'bbox': sample['bbox_xyxy'],

                    # 方案 A：完整表达式作为 phrase。
                    'phrase': sample['caption'],
                    'tokens_positive': sample['tokens_positive'],
                }
            ],
        },
        'attributes': sample['attributes'],
    }


def validate_mdetr(data, expected_count):
    images = data['images']
    annotations = data['annotations']

    assert len(images) == expected_count
    assert len(annotations) == expected_count

    image_ids = [item['id'] for item in images]
    annotation_ids = [item['id'] for item in annotations]
    annotation_image_ids = [
        item['image_id'] for item in annotations
    ]

    assert len(set(image_ids)) == expected_count
    assert len(set(annotation_ids)) == expected_count
    assert set(annotation_image_ids) == set(image_ids)

    annotation_count = {}
    for image_id in annotation_image_ids:
        annotation_count[image_id] = annotation_count.get(image_id, 0) + 1

    assert all(count == 1 for count in annotation_count.values())


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    records = torch.load(
        input_path,
        map_location='cpu',
        weights_only=True,
    )

    if not isinstance(records, list):
        raise TypeError(
            f'Top-level annotation must be list, got {type(records)}.')

    samples = []

    for sample_id, item in enumerate(records):
        sample = parse_record(
            item=item,
            sample_id=sample_id,
            image_dir=str(image_dir),
        )
        samples.append(sample)

    # 为 train/val/test 都生成 MDETR/COCO JSON。
    mdetr_data = build_mdetr_annotation(samples, args.split)
    validate_mdetr(mdetr_data, len(records))

    mdetr_output = output_dir / (
        f'{args.dataset_prefix}_{args.split}.json'
    )

    with open(mdetr_output, 'w', encoding='utf-8') as file:
        json.dump(
            mdetr_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f'MDETR annotation saved to: {mdetr_output}')
    print(f'Number of samples: {len(samples)}')

    # 训练集额外生成 ODVG JSONL。
    if args.generate_train_vg:
        odvg_output = output_dir / (
            f'{args.dataset_prefix}_{args.split}_vg.json'
        )

        with open(odvg_output, 'w', encoding='utf-8') as file:
            for sample in samples:
                record = build_odvg_record(sample)
                file.write(
                    json.dumps(record, ensure_ascii=False) + '\n'
                )

        print(f'ODVG training annotation saved to: {odvg_output}')

        with open(odvg_output, 'r', encoding='utf-8') as file:
            line_count = sum(1 for line in file if line.strip())

        if line_count != len(samples):
            raise RuntimeError(
                f'ODVG line count mismatch: '
                f'{line_count} != {len(samples)}')

        print(f'ODVG line count: {line_count}')

    print('Conversion and validation completed successfully.')


if __name__ == '__main__':
    main()
