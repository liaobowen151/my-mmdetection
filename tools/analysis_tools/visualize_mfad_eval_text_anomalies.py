#!/usr/bin/env python3
"""Visualize reviewed MFAD validation/test caption anomalies."""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from visualize_mfad_manual_review import (
    BLACK, GRAY, RED, WHITE, clamp_box, draw_box, expanded_crop, fit_panel,
    load_font, zoom_panel,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-file", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "mfad_eval_text_anomalies_reviewed.json"))
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/home/lbw/mmdetection/data/MFAD"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "mfad_eval_text_anomaly_visualizations"))
    parser.add_argument(
        "--revision-file", type=Path, default=None,
        help="Optional preview revisions keyed by split:index.")
    return parser.parse_args()


def display_caption(caption, limit=560):
    if len(caption) <= limit:
        return caption
    return caption[:limit].rstrip() + " ... [TRUNCATED; full text is in manifest.json]"


def wrapped_lines(text, width=142):
    return textwrap.wrap(text, width=width, break_long_words=False,
                         break_on_hyphens=False) or [""]


def render_sample(sample, data_root, destination):
    visible_path = data_root / "vis" / sample["file_name"]
    infrared_path = data_root / "ir" / sample["file_name"]
    if not visible_path.exists() or not infrared_path.exists():
        raise FileNotFoundError(
            f"Missing pair: visible={visible_path}, infrared={infrared_path}")

    visible = Image.open(visible_path).convert("RGB")
    infrared = Image.open(infrared_path).convert("RGB")
    if visible.size != infrared.size:
        raise ValueError(f"Pair size mismatch for {sample['file_name']}")

    image_width, image_height = visible.size
    if "preview_bbox_xyxy" in sample:
        box = clamp_box(
            sample["preview_bbox_xyxy"], image_width, image_height)
    else:
        x, y, width, height = sample["bbox_xywh"]
        box = clamp_box((x, y, x + width, y + height),
                        image_width, image_height)
    crop_box = expanded_crop(box, image_width, image_height)

    panel_width = 1280
    full_height = 960
    zoom_height = 720
    header_height = 118
    caption_lines = wrapped_lines(
        "Original caption: " + display_caption(sample["caption"]))
    proposed_lines = wrapped_lines(
        "Proposed English text: " + sample.get(
            "preview_text", "<no proposed revision>"))
    note_lines = wrapped_lines(
        "Review: " + sample["review_category"].replace("_", " "))
    footer_height = 76 + 38 * (
        len(proposed_lines) + len(caption_lines) + len(note_lines))
    canvas = Image.new(
        "RGB",
        (panel_width * 2,
         header_height + full_height + zoom_height + footer_height),
        WHITE,
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    text_font = load_font(27)
    small_font = load_font(25)

    sample_key = f"{sample['split']}:{sample['index']}"
    title = (f"MFAD evaluation caption anomaly | {sample_key} | "
             f"severity={sample['review_severity']}")
    draw.text((24, 14), title, fill=BLACK, font=title_font)
    draw.text(
        (24, 62),
        f"file={sample['file_name']} | proposed_bbox_xyxy={list(box)} | "
        f"test subsets={sample['test_subsets'] or 'N/A'}",
        fill=BLACK, font=text_font,
    )

    full_visible = fit_panel(
        draw_box(visible, box, sample_key), panel_width, full_height)
    full_infrared = fit_panel(
        draw_box(infrared, box, sample_key), panel_width, full_height)
    top = header_height
    canvas.paste(full_visible, (0, top))
    canvas.paste(full_infrared, (panel_width, top))

    zoom_visible = zoom_panel(
        visible, box, crop_box, panel_width, zoom_height, sample_key)
    zoom_infrared = zoom_panel(
        infrared, box, crop_box, panel_width, zoom_height, sample_key)
    zoom_top = top + full_height
    canvas.paste(zoom_visible, (0, zoom_top))
    canvas.paste(zoom_infrared, (panel_width, zoom_top))

    for left, label in ((20, "VISIBLE: full + zoom"),
                        (panel_width + 20, "INFRARED: full + zoom")):
        draw.rectangle((left - 8, top + 8, left + 340, top + 51), fill=BLACK)
        draw.text((left, top + 11), label, fill=WHITE, font=small_font)

    footer_top = zoom_top + zoom_height
    draw.rectangle((0, footer_top, panel_width * 2, canvas.height), fill=GRAY)
    y_pos = footer_top + 16
    for line in proposed_lines:
        draw.text((24, y_pos), line, fill=RED, font=text_font)
        y_pos += 38
    for line in note_lines:
        draw.text((24, y_pos), line, fill=BLACK, font=text_font)
        y_pos += 38
    for line in caption_lines:
        draw.text((24, y_pos), line, fill=BLACK, font=small_font)
        y_pos += 38

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=94, subsampling=0)
    return list(box)


def make_contact_sheet(rendered, destination):
    thumb_width = 800
    thumb_height = 620
    columns = 2
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), WHITE)
    draw = ImageDraw.Draw(sheet)
    font = load_font(27, bold=True)
    for order, (sample, path) in enumerate(rendered):
        image = Image.open(path).convert("RGB")
        imagery_height = min(image.height, 1798)
        thumb = fit_panel(
            image.crop((0, 0, image.width, imagery_height)),
            thumb_width, thumb_height - 48,
        )
        left = (order % columns) * thumb_width
        top = (order // columns) * thumb_height
        sheet.paste(thumb, (left, top))
        draw.text(
            (left + 12, top + thumb_height - 42),
            f"{sample['split']}:{sample['index']} | {sample['review_category']}",
            fill=BLACK, font=font,
        )
    sheet.save(destination, quality=92, subsampling=0)


def main():
    args = parse_args()
    payload = json.loads(args.review_file.read_text(encoding="utf-8"))
    samples = [dict(sample) for sample in payload["samples"]]
    revision_payload = None
    if args.revision_file is not None:
        revision_payload = json.loads(
            args.revision_file.read_text(encoding="utf-8"))
        revisions = revision_payload["revisions"]
        for sample in samples:
            key = f"{sample['split']}:{sample['index']}"
            revision = revisions.get(key)
            if revision is None:
                continue
            sample["preview_text"] = revision["proposed_text"]
            if "bbox_xyxy" in revision:
                sample["preview_bbox_xyxy"] = revision["bbox_xyxy"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    manifest_samples = []
    for sample in samples:
        destination = args.output_dir / (
            f"{sample['split']}_{sample['index']:05d}_"
            f"{Path(sample['file_name']).stem}.jpg")
        bbox_xyxy = render_sample(sample, args.data_root, destination)
        rendered.append((sample, destination))
        manifest_item = dict(sample)
        manifest_item["proposed_bbox_xyxy"] = bbox_xyxy
        manifest_item["visualization"] = str(destination)
        manifest_samples.append(manifest_item)

    contact_sheet = args.output_dir / "contact_sheet.jpg"
    make_contact_sheet(rendered, contact_sheet)
    manifest = {
        "sample_count": len(rendered),
        "status": "visualization_preview_only_not_applied_to_source_annotations",
        "revision_file": str(args.revision_file) if args.revision_file else None,
        "assumptions": revision_payload.get("assumptions", [])
        if revision_payload else [],
        "contact_sheet": str(contact_sheet),
        "samples": manifest_samples,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_count": len(rendered),
        "output_dir": str(args.output_dir),
        "contact_sheet": str(contact_sheet),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
