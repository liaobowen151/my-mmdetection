#!/usr/bin/env python3
"""Find conspicuous caption defects in FLIR train/val/test annotations."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir", type=Path,
        default=Path("/home/lbw/mmdetection/data/FLIR/annotations"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "flir_text_anomalies_reviewed.json"))
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify(caption):
    if not isinstance(caption, str) or not caption.strip():
        return "critical", "empty_or_non_string"
    if re.search(r"^\s*no\b", caption, re.IGNORECASE):
        if re.search(r"\b(?:bounding box|bbox|coordinates?)\b",
                     caption, re.IGNORECASE):
            return "critical", "target_absent_and_box_meta_leakage"
        return "critical", "target_explicitly_absent"
    if re.search(r"\bno motorcycle visible\b", caption, re.IGNORECASE):
        return "high", "irrelevant_absence_clause"
    if re.search(r"\bnot a motorcycle\b", caption, re.IGNORECASE):
        return "high", "correction_leakage"
    if any(ord(char) < 32 and char not in "\t\n\r" for char in caption):
        return "critical", "invalid_control_character"
    if len(re.findall(r"\b\w+(?:[-']\w+)*\b", caption)) >= 60:
        return "high", "extreme_length_60plus_words"
    return None


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_derived_consistency(annotation_dir, train):
    anns = {ann["image_id"]: ann for ann in train["annotations"]}
    vg = load_jsonl(annotation_dir / "rgbtvg_flir_train_vg.json")
    seg = load_jsonl(annotation_dir / "rgbtvg_flir_train_seg.jsonl")
    vg_mismatches = []
    seg_mismatches = []
    for index, image in enumerate(train["images"]):
        ann = anns[image["id"]]
        x, y, width, height = ann["bbox"]
        expected_xyxy = [x, y, x + width, y + height]
        vg_item = vg[index]
        vg_region = vg_item["grounding"]["regions"][0]
        if not (vg_item["filename"] == image["file_name"]
                and vg_item["grounding"]["caption"] == image["caption"]
                and vg_region["bbox"] == expected_xyxy):
            vg_mismatches.append(index)
        seg_item = seg[index]
        if not (seg_item["filename"] == image["file_name"]
                and seg_item["caption"] == image["caption"]
                and seg_item["bbox_xyxy"] == expected_xyxy):
            seg_mismatches.append(index)
    return {
        "train_vg_count": len(vg),
        "train_seg_count": len(seg),
        "train_vg_mismatch_count": len(vg_mismatches),
        "train_seg_mismatch_count": len(seg_mismatches),
        "train_vg_mismatch_indices": vg_mismatches[:50],
        "train_seg_mismatch_indices": seg_mismatches[:50],
    }


def main():
    args = parse_args()
    payloads = {
        split: load(args.annotation_dir / f"rgbtvg_flir_{split}.json")
        for split in ("train", "val", "test")
    }
    subset_memberships = {}
    for subset in ("testA", "testB", "testC"):
        payload = load(args.annotation_dir / f"rgbtvg_flir_{subset}.json")
        anns = {ann["image_id"]: ann for ann in payload["annotations"]}
        for image in payload["images"]:
            key = (image["file_name"], image["caption"],
                   tuple(anns[image["id"]]["bbox"]))
            subset_memberships.setdefault(key, []).append(subset)

    records = []
    structural = Counter()
    for split, payload in payloads.items():
        anns = {ann["image_id"]: ann for ann in payload["annotations"]}
        if len(payload["images"]) != len(payload["annotations"]):
            structural[f"{split}_image_annotation_count_mismatch"] += 1
        for index, image in enumerate(payload["images"]):
            ann = anns.get(image["id"])
            if ann is None:
                structural[f"{split}_missing_annotation"] += 1
                continue
            caption = image.get("caption")
            result = classify(caption)
            if isinstance(caption, str):
                if ann.get("tokens_positive") != [[0, len(caption)]]:
                    structural[f"{split}_token_span_mismatch"] += 1
            if result is None:
                continue
            severity, category = result
            key = (image["file_name"], caption, tuple(ann["bbox"]))
            records.append({
                "split": split,
                "index": index,
                "image_id": image["id"],
                "file_name": image["file_name"],
                "bbox_xywh": ann["bbox"],
                "caption": caption,
                "severity": severity,
                "category": category,
                "test_subsets": subset_memberships.get(key, [])
                if split == "test" else [],
            })

    summary = {
        "screened_unique_samples": {
            split: len(payload["images"])
            for split, payload in payloads.items()
        },
        "anomaly_count": len(records),
        "count_by_split": dict(Counter(r["split"] for r in records)),
        "count_by_severity": dict(Counter(r["severity"] for r in records)),
        "count_by_category": dict(Counter(r["category"] for r in records)),
        "structural_issue_counts": dict(structural),
        "train_derived_consistency": train_derived_consistency(
            args.annotation_dir, payloads["train"]),
        "test_subset_note": ("testA/testB/testC are memberships of the "
                             "canonical test split and are not double-counted."),
        "scope_note": ("This report flags conspicuous caption defects. High "
                       "severity correction clauses may retain a usable noun "
                       "phrase but still contain generation/meta leakage."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"summary": summary, "samples": records},
        ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
