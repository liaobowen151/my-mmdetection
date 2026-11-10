#!/usr/bin/env python3
"""Visualize MFAD manual-review boxes on paired visible/infrared images."""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (225, 225, 225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--review-file', type=Path,
        default=Path('/home/lbw/mmdetection/analysis_demo/referent_spans/'
                     'mfad_train_referent_spans_manual_review.json'))
    parser.add_argument(
        '--data-root', type=Path,
        default=Path('/home/lbw/mmdetection/data/MFAD'))
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('/home/lbw/mmdetection/analysis_demo/'
                     'mfad_manual_review_visualizations'))
    parser.add_argument(
        '--revision-file', type=Path, default=None,
        help='Optional JSON mapping sample indices to proposed bbox/text.')
    return parser.parse_args()


def load_font(size, bold=False):
    name = 'NotoSansCJK-Bold.ttc' if bold else 'NotoSansCJK-Regular.ttc'
    path = Path('/usr/share/fonts/opentype/noto') / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    path = Path('/usr/share/fonts/truetype/dejavu') / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(1, min(x2, width)),
        max(1, min(y2, height)),
    )


def expanded_crop(box, width, height):
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = max(100, box_width * 3)
    margin_y = max(100, box_height * 3)
    crop = [
        max(0, x1 - margin_x), max(0, y1 - margin_y),
        min(width, x2 + margin_x), min(height, y2 + margin_y),
    ]
    # Use the same square-ish context on both modalities and avoid tiny crops.
    crop_width = crop[2] - crop[0]
    crop_height = crop[3] - crop[1]
    side = max(crop_width, crop_height, 240)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = max(0, min(int(center_x - side / 2), width - side))
    top = max(0, min(int(center_y - side / 2), height - side))
    right = min(width, int(left + side))
    bottom = min(height, int(top + side))
    return (int(left), int(top), int(right), int(bottom))


def draw_box(image, box, label, width=5):
    result = image.copy().convert('RGB')
    draw = ImageDraw.Draw(result)
    x1, y1, x2, y2 = box
    for offset in range(width):
        draw.rectangle(
            (x1 - offset, y1 - offset, x2 + offset, y2 + offset),
            outline=RED)
    font = load_font(26, bold=True)
    label_box = draw.textbbox((0, 0), label, font=font)
    label_width = label_box[2] - label_box[0] + 14
    label_height = label_box[3] - label_box[1] + 12
    label_y = max(0, y1 - label_height)
    draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height),
                   fill=RED)
    draw.text((x1 + 7, label_y + 3), label, fill=WHITE, font=font)
    return result


def fit_panel(image, width, height):
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new('RGB', (width, height), BLACK)
    left = (width - result.width) // 2
    top = (height - result.height) // 2
    panel.paste(result, (left, top))
    return panel


def zoom_panel(image, box, crop_box, width, height, label):
    crop = image.crop(crop_box).convert('RGB')
    relative = (
        box[0] - crop_box[0], box[1] - crop_box[1],
        box[2] - crop_box[0], box[3] - crop_box[1],
    )
    scale = min(width / crop.width, height / crop.height)
    resized = crop.resize(
        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
        Image.Resampling.LANCZOS)
    scaled_box = tuple(int(value * scale) for value in relative)
    resized = draw_box(resized, scaled_box, label, width=7)
    return fit_panel(resized, width, height)


def wrapped_lines(text, width=145):
    return textwrap.wrap(text, width=width, break_long_words=False,
                         break_on_hyphens=False) or ['']


def render_sample(sample, data_root, destination):
    visible_path = data_root / 'vis' / sample['filename']
    infrared_path = data_root / 'ir' / sample['filename']
    if not visible_path.exists() or not infrared_path.exists():
        raise FileNotFoundError(
            f'Missing pair: visible={visible_path}, infrared={infrared_path}')

    visible = Image.open(visible_path).convert('RGB')
    infrared = Image.open(infrared_path).convert('RGB')
    if visible.size != infrared.size:
        raise ValueError(
            f'Pair size mismatch for {sample["filename"]}: '
            f'{visible.size} != {infrared.size}')
    image_width, image_height = visible.size
    box = clamp_box(sample['bbox_xyxy'], image_width, image_height)
    crop_box = expanded_crop(box, image_width, image_height)

    panel_width = 1280
    full_height = 960
    zoom_height = 720
    header_height = 92
    caption_lines = wrapped_lines('Caption: ' + sample['caption'])
    reason_lines = wrapped_lines('Reason: ' + sample['reason'])
    footer_height = 70 + 37 * (len(caption_lines) + len(reason_lines))
    canvas = Image.new(
        'RGB', (panel_width * 2,
                header_height + full_height + zoom_height + footer_height),
        WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    text_font = load_font(28)
    small_font = load_font(25)

    title = (f'MFAD manual review | sample_index={sample["sample_index"]} | '
             f'file={sample["filename"]} | bbox={sample["bbox_xyxy"]}')
    draw.text((24, 20), title, fill=BLACK, font=title_font)

    label = f'index {sample["sample_index"]}'
    full_visible = fit_panel(
        draw_box(visible, box, label), panel_width, full_height)
    full_infrared = fit_panel(
        draw_box(infrared, box, label), panel_width, full_height)
    top = header_height
    canvas.paste(full_visible, (0, top))
    canvas.paste(full_infrared, (panel_width, top))

    zoom_visible = zoom_panel(
        visible, box, crop_box, panel_width, zoom_height, label)
    zoom_infrared = zoom_panel(
        infrared, box, crop_box, panel_width, zoom_height, label)
    zoom_top = top + full_height
    canvas.paste(zoom_visible, (0, zoom_top))
    canvas.paste(zoom_infrared, (panel_width, zoom_top))

    # Panel labels remain outside the imagery to avoid obscuring small boxes.
    for x, text in ((20, 'VISIBLE: full + zoom'),
                    (panel_width + 20, 'INFRARED: full + zoom')):
        draw.rectangle((x - 8, top + 8, x + 340, top + 51), fill=BLACK)
        draw.text((x, top + 11), text, fill=WHITE, font=small_font)

    footer_top = zoom_top + zoom_height
    draw.rectangle((0, footer_top, panel_width * 2,
                    canvas.height), fill=GRAY)
    current = ', '.join(sample['current_proposed_texts']) or '<empty>'
    draw.text((24, footer_top + 16),
              f'Current proposal: {current}', fill=RED, font=text_font)
    y = footer_top + 58
    for line in caption_lines:
        draw.text((24, y), line, fill=BLACK, font=small_font)
        y += 37
    for line in reason_lines:
        draw.text((24, y), line, fill=BLACK, font=small_font)
        y += 37

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94, subsampling=0)


def make_contact_sheet(rendered, destination):
    thumb_width = 800
    thumb_height = 620
    columns = 2
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * thumb_width, rows * thumb_height),
                      WHITE)
    for index, (sample, path) in enumerate(rendered):
        image = Image.open(path).convert('RGB')
        # The full/zoom imagery occupies the upper portion of each render.
        crop_height = min(image.height, 1772)
        thumb = fit_panel(image.crop((0, 0, image.width, crop_height)),
                          thumb_width, thumb_height - 48)
        left = (index % columns) * thumb_width
        top = (index // columns) * thumb_height
        sheet.paste(thumb, (left, top))
        draw = ImageDraw.Draw(sheet)
        draw.text((left + 12, top + thumb_height - 42),
                  f'sample_index={sample["sample_index"]}', fill=BLACK,
                  font=load_font(27, bold=True))
    sheet.save(destination, quality=92, subsampling=0)


def main():
    args = parse_args()
    payload = json.loads(args.review_file.read_text(encoding='utf-8'))
    samples = [dict(sample) for sample in payload['samples']]
    if args.revision_file is not None:
        revision_payload = json.loads(
            args.revision_file.read_text(encoding='utf-8'))
        revisions = {
            int(index): revision
            for index, revision in revision_payload['revisions'].items()
        }
        for sample in samples:
            revision = revisions.get(sample['sample_index'])
            if revision is None:
                continue
            sample['original_bbox_xyxy'] = sample['bbox_xyxy']
            sample['original_proposed_texts'] = sample[
                'current_proposed_texts']
            if 'bbox_xyxy' in revision:
                sample['bbox_xyxy'] = revision['bbox_xyxy']
            if 'proposed_text' in revision:
                sample['current_proposed_texts'] = [
                    revision['proposed_text']]
            if 'reason' in revision:
                sample['reason'] = revision['reason']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for sample in samples:
        safe_stem = Path(sample['filename']).stem
        destination = args.output_dir / (
            f'sample_{sample["sample_index"]:05d}_{safe_stem}.jpg')
        render_sample(sample, args.data_root, destination)
        rendered.append((sample, destination))
    contact_sheet = args.output_dir / 'contact_sheet.jpg'
    make_contact_sheet(rendered, contact_sheet)
    manifest = {
        'sample_count': len(rendered),
        'contact_sheet': str(contact_sheet),
        'images': [{
            'sample_index': sample['sample_index'],
            'filename': sample['filename'],
            'bbox_xyxy': sample['bbox_xyxy'],
            'proposed_texts': sample['current_proposed_texts'],
            'visualization': str(path),
        } for sample, path in rendered],
    }
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
