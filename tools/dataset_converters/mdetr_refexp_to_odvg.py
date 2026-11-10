import argparse
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MDETR-style REC JSON to ODVG JSONL.')

    parser.add_argument(
        'input',
        help='Input MDETR/COCO-style JSON file.')

    parser.add_argument(
        'output',
        help='Output ODVG JSONL file.')

    parser.add_argument(
        '--image-dir',
        required=True,
        help='Image directory used to verify image files.')

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, 'r', encoding='utf-8') as file:
        data = json.load(file)

    images = data['images']
    annotations = data['annotations']

    annotations_by_image = defaultdict(list)

    for annotation in annotations:
        annotations_by_image[annotation['image_id']].append(annotation)

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converted_count = 0

    with open(args.output, 'w', encoding='utf-8') as output_file:
        for image_info in images:
            image_id = image_info['id']
            file_name = image_info['file_name']
            width = int(image_info['width'])
            height = int(image_info['height'])
            caption = image_info['caption'].strip()

            image_path = os.path.join(args.image_dir, file_name)

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f'Image does not exist: {image_path}')

            image_annotations = annotations_by_image.get(image_id, [])

            # REC：一条表达式只能对应一个 GT。
            if len(image_annotations) != 1:
                raise ValueError(
                    f'Image id {image_id} has '
                    f'{len(image_annotations)} annotations; expected 1.')

            annotation = image_annotations[0]

            # 原始 MDETR/COCO bbox 为 xywh。
            x, y, bbox_width, bbox_height = [
                float(value) for value in annotation['bbox']
            ]

            if bbox_width <= 0 or bbox_height <= 0:
                raise ValueError(
                    f'Invalid bbox for image id {image_id}: '
                    f'{annotation["bbox"]}')

            if (
                x < 0
                or y < 0
                or x + bbox_width > width
                or y + bbox_height > height
            ):
                raise ValueError(
                    f'BBox outside image for image id {image_id}: '
                    f'bbox={annotation["bbox"]}, '
                    f'image_size=({width}, {height})')

            # ODVG 必须使用 xyxy。
            bbox_xyxy = [
                x,
                y,
                x + bbox_width,
                y + bbox_height,
            ]

            # 方案 A：完整句子作为正文本区域。
            expected_tokens_positive = [[0, len(caption)]]

            source_tokens_positive = annotation.get(
                'tokens_positive',
                expected_tokens_positive,
            )

            if source_tokens_positive != expected_tokens_positive:
                raise ValueError(
                    f'tokens_positive does not cover the complete caption '
                    f'for image id {image_id}: '
                    f'caption={caption!r}, '
                    f'tokens_positive={source_tokens_positive}, '
                    f'expected={expected_tokens_positive}')

            output_record = {
                'filename': file_name,
                'height': height,
                'width': width,
                'grounding': {
                    'caption': caption,
                    'regions': [
                        {
                            'bbox': bbox_xyxy,
                            'phrase': caption,
                            'tokens_positive': expected_tokens_positive,
                        }
                    ],
                },
            }

            output_file.write(
                json.dumps(output_record, ensure_ascii=False) + '\n'
            )

            converted_count += 1

    if converted_count != len(images):
        raise RuntimeError(
            f'Converted count mismatch: '
            f'{converted_count} != {len(images)}')

    print(f'Input: {args.input}')
    print(f'Output: {args.output}')
    print(f'Converted samples: {converted_count}')
    print('Conversion completed successfully.')


if __name__ == '__main__':
    main()