#!/usr/bin/env python3
"""Visualize reviewed FLIR caption anomalies on paired RGB/IR images."""

import argparse
import json
import math
import textwrap
from collections import defaultdict
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
                     "flir_text_anomalies_reviewed.json"))
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/home/lbw/mmdetection/data/FLIR"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "flir_text_anomaly_visualizations"))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional preview limit; omitted means all reviewed samples.")
    parser.add_argument(
        "--revision-file", type=Path, default=None,
        help="Optional preview revisions keyed by split:index.")
    parser.add_argument(
        "--revisions-only", action="store_true",
        help="Render only samples present in --revision-file.")
    return parser.parse_args()


def wrapped_lines(text, width=104):
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
        box = clamp_box(sample["preview_bbox_xyxy"], image_width, image_height)
    else:
        x, y, width, height = sample["bbox_xywh"]
        box = clamp_box((x, y, x + width, y + height),
                        image_width, image_height)
    crop_box = expanded_crop(box, image_width, image_height)

    panel_width = 960
    full_height = 768
    zoom_height = 640
    header_height = 112
    caption_lines = wrapped_lines("Original caption: " + sample["caption"])
    proposed_lines = wrapped_lines(
        "Proposed English text: " + sample.get(
            "preview_text", "<no proposed revision>"))
    footer_height = 94 + 36 * (len(proposed_lines) + len(caption_lines))
    canvas = Image.new(
        "RGB",
        (panel_width * 2,
         header_height + full_height + zoom_height + footer_height),
        WHITE,
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(32, bold=True)
    text_font = load_font(25)
    small_font = load_font(23)

    sample_key = f"{sample['split']}:{sample['index']}"
    draw.text(
        (22, 12),
        f"FLIR caption anomaly | {sample_key} | severity={sample['severity']}",
        fill=BLACK, font=title_font,
    )
    draw.text(
        (22, 58),
        f"file={sample['file_name']} | proposed_bbox_xyxy={list(box)} | "
        f"test subsets={sample['test_subsets'] or 'N/A'}",
        fill=BLACK, font=text_font,
    )

    full_visible = fit_panel(
        draw_box(visible, box, sample_key, width=4),
        panel_width, full_height,
    )
    full_infrared = fit_panel(
        draw_box(infrared, box, sample_key, width=4),
        panel_width, full_height,
    )
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

    for left, label in ((18, "VISIBLE: full + zoom"),
                        (panel_width + 18, "INFRARED: full + zoom")):
        draw.rectangle((left - 7, top + 7, left + 325, top + 47), fill=BLACK)
        draw.text((left, top + 9), label, fill=WHITE, font=small_font)

    footer_top = zoom_top + zoom_height
    draw.rectangle((0, footer_top, canvas.width, canvas.height), fill=GRAY)
    category = sample["category"].replace("_", " ")
    draw.text(
        (22, footer_top + 14), f"Review category: {category}",
        fill=BLACK, font=text_font,
    )
    y_pos = footer_top + 54
    for line in proposed_lines:
        draw.text((22, y_pos), line, fill=RED, font=text_font)
        y_pos += 36
    for line in caption_lines:
        draw.text((22, y_pos), line, fill=BLACK, font=small_font)
        y_pos += 36

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=93, subsampling=0)
    return list(box)


def make_contact_sheets(rendered, output_dir, per_page=18):
    thumb_width = 640
    thumb_height = 350
    columns = 3
    paths = []
    by_split = defaultdict(list)
    for item in rendered:
        by_split[item[0]["split"]].append(item)

    for split in ("train", "val", "test"):
        items = by_split.get(split, [])
        for page_index, start in enumerate(range(0, len(items), per_page), 1):
            page = items[start:start + per_page]
            rows_on_page = math.ceil(len(page) / columns)
            sheet = Image.new(
                "RGB", (columns * thumb_width, rows_on_page * thumb_height),
                WHITE,
            )
            draw = ImageDraw.Draw(sheet)
            font = load_font(22, bold=True)
            for order, (sample, path) in enumerate(page):
                image = Image.open(path).convert("RGB")
                imagery_height = min(image.height, 1520)
                thumb = fit_panel(
                    image.crop((0, 0, image.width, imagery_height)),
                    thumb_width, thumb_height - 42,
                )
                left = (order % columns) * thumb_width
                top = (order // columns) * thumb_height
                sheet.paste(thumb, (left, top))
                label = (f"{sample['split']}:{sample['index']} | "
                         f"{sample['category']}")
                draw.text((left + 9, top + thumb_height - 36), label,
                          fill=BLACK, font=font)
            destination = output_dir / (
                f"contact_sheet_{split}_{page_index:02d}.jpg")
            sheet.save(destination, quality=91, subsampling=0)
            paths.append(str(destination))
    return paths


def main():
    args = parse_args()
    payload = json.loads(args.review_file.read_text(encoding="utf-8"))
    samples = [dict(sample) for sample in payload["samples"]]
    revision_payload = None
    if args.revision_file is not None:
        revision_payload = json.loads(
            args.revision_file.read_text(encoding="utf-8"))
        revisions = revision_payload["revisions"]
        selected = []
        for sample in samples:
            key = f"{sample['split']}:{sample['index']}"
            revision = revisions.get(key)
            if revision is None:
                if not args.revisions_only:
                    selected.append(sample)
                continue
            sample["preview_text"] = revision["proposed_text"]
            if "bbox_xyxy" in revision:
                sample["preview_bbox_xyxy"] = revision["bbox_xyxy"]
            if revision.get("inferred_from_ambiguous_user_description"):
                sample["preview_inference"] = True
            selected.append(sample)
        samples = selected
    if args.limit is not None:
        samples = samples[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    manifest_samples = []
    for sample in samples:
        destination = args.output_dir / (
            f"{sample['split']}_{sample['index']:05d}_"
            f"{Path(sample['file_name']).stem}.jpg")
        bbox_xyxy = render_sample(sample, args.data_root, destination)
        rendered.append((sample, destination))
        item = dict(sample)
        item["bbox_xyxy"] = bbox_xyxy
        item["visualization"] = str(destination)
        manifest_samples.append(item)

    contact_sheets = make_contact_sheets(rendered, args.output_dir)
    manifest = {
        "sample_count": len(rendered),
        "status": "visualization_preview_only_not_applied_to_source_annotations"
        if args.revision_file else "current_annotations",
        "revision_file": str(args.revision_file) if args.revision_file else None,
        "assumptions": revision_payload.get("assumptions", [])
        if revision_payload else [],
        "source_report": str(args.review_file),
        "contact_sheets": contact_sheets,
        "samples": manifest_samples,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sample_count": len(rendered),
        "output_dir": str(args.output_dir),
        "contact_sheet_count": len(contact_sheets),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
