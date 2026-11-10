#!/usr/bin/env python3
"""Find suspicious referring captions in the MFAD validation/test splits."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RULES = {
    "explicit_absence": [
        r"^\s*no\b",
        r"\bno visible\b",
        r"\bno (?:\w+\s+){0,3}(?:is|are|was|were|can be) (?:present|visible|detected|found|shown)\b",
        r"\b(?:does not|doesn't|do not|don't) (?:contain|show|appear|include|seem)\b",
        r"\b(?:not|isn't|aren't) (?:present|visible|shown|detected|identifiable)\b",
        r"\blikely not actually present\b",
    ],
    "annotation_meta_leakage": [
        r"\bbounding box(?:es)?\b",
        r"\bbox coordinates?\b",
        r"\bcoordinates?\s*\[",
        r"\bspecified box\b",
        r"\bprovided (?:box|bounding|coordinates)\b",
        r"\bprompt\b",
        r"\banswer\b",
    ],
    "recognition_failure": [
        r"\b(?:cannot|can't|could not|couldn't|unable to) (?:be )?(?:identify|determine|discern|detect|see|locate|confirm)\b",
        r"\b(?:cannot|can't|could not|couldn't) be (?:identified|determined|discerned|detected|seen|located|confirmed)\b",
        r"\b(?:unclear|unknown) (?:whether|what|which)\b",
        r"\bnot enough (?:visual )?(?:information|detail|evidence)\b",
    ],
    "competing_interpretations": [
        r"\b(?:could|might|may) be either\b",
        r"\beither (?:a|an|the)\b.+\bor (?:a|an|the)\b",
        r"\b(?:rather than|instead of)\b",
        r"\b(?:difficult|hard) to (?:tell|determine|distinguish|identify)\b",
        r"\b(?:appears|seems) to be .{0,80}\bbut\b",
    ],
}

# Manually checked after the broad rule-based pass.  These are the samples
# whose captions are clearly unsuitable as ordinary referring expressions.
REVIEWED_ANOMALIES = {
    ("val", 772): ("critical", "runaway_generation"),
    ("val", 984): ("critical", "target_explicitly_absent"),
    ("val", 1026): ("critical", "target_explicitly_absent"),
    ("test", 2): ("critical", "target_absent_and_box_leakage"),
    ("test", 3): ("critical", "target_absent_and_box_leakage"),
    ("test", 4): ("critical", "target_absent_and_box_leakage"),
    ("test", 320): ("high", "box_coordinate_leakage"),
    ("test", 770): ("critical", "referent_semantic_contradiction"),
    ("test", 894): ("critical", "target_absent_and_box_leakage"),
    ("test", 944): ("high", "referent_semantic_contradiction"),
    ("test", 1751): ("high", "box_coordinate_leakage"),
    ("test", 2517): ("critical", "runaway_generation"),
    ("test", 2560): ("critical", "target_absent_and_box_leakage"),
    ("test", 2651): ("critical", "target_absent_and_box_leakage"),
    ("test", 3605): ("high", "box_dimension_leakage"),
    ("test", 4074): ("high", "box_dimension_leakage"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir", type=Path,
        default=Path("/home/lbw/mmdetection/data/MFAD/annotations"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "mfad_eval_text_anomalies.json"))
    parser.add_argument(
        "--reviewed-output", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "mfad_eval_text_anomalies_reviewed.json"))
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    subset_membership = {}
    for subset in ("testA", "testB", "testC"):
        payload = load(args.annotation_dir / f"rgbtvg_mfad_{subset}.json")
        for image in payload["images"]:
            key = (image["file_name"], image["caption"])
            subset_membership.setdefault(key, []).append(subset)

    records = []
    split_counts = {}
    for split in ("val", "test"):
        payload = load(args.annotation_dir / f"rgbtvg_mfad_{split}.json")
        anns = {ann["image_id"]: ann for ann in payload["annotations"]}
        split_counts[split] = len(payload["images"])
        for index, image in enumerate(payload["images"]):
            caption = image.get("caption")
            reasons = []
            matched = {}
            if not isinstance(caption, str) or not caption.strip():
                reasons.append("empty_or_non_string")
            else:
                for rule, patterns in RULES.items():
                    hits = [pattern for pattern in patterns
                            if re.search(pattern, caption, re.IGNORECASE | re.DOTALL)]
                    if hits:
                        reasons.append(rule)
                        matched[rule] = hits
                word_count = len(re.findall(r"\b\w+(?:[-']\w+)*\b", caption))
                if word_count >= 70:
                    reasons.append("extreme_length_70plus_words")
                if any(ord(char) < 32 and char not in "\t\n\r" for char in caption):
                    reasons.append("invalid_control_character")
            if not reasons:
                continue
            ann = anns.get(image["id"], {})
            records.append({
                "split": split,
                "index": index,
                "image_id": image["id"],
                "file_name": image["file_name"],
                "bbox_xywh": ann.get("bbox"),
                "caption": caption,
                "word_count": len(caption.split()) if isinstance(caption, str) else None,
                "reasons": reasons,
                "matched_patterns": matched,
                "test_subsets": subset_membership.get(
                    (image["file_name"], caption), []) if split == "test" else [],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "screened_unique_samples": split_counts,
        "candidate_count": len(records),
        "candidate_count_by_split": dict(Counter(r["split"] for r in records)),
        "reason_counts": dict(Counter(
            reason for record in records for reason in record["reasons"])),
        "note": ("testA/testB/testC are treated as memberships of the canonical "
                 "test split and are not counted as additional unique samples."),
    }
    args.output.write_text(json.dumps(
        {"summary": summary, "samples": records}, ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")

    reviewed = []
    for record in records:
        review = REVIEWED_ANOMALIES.get((record["split"], record["index"]))
        if review is None:
            continue
        item = dict(record)
        item["review_severity"], item["review_category"] = review
        reviewed.append(item)
    reviewed_summary = {
        "screened_unique_samples": split_counts,
        "reviewed_anomaly_count": len(reviewed),
        "count_by_split": dict(Counter(r["split"] for r in reviewed)),
        "count_by_severity": dict(Counter(
            r["review_severity"] for r in reviewed)),
        "count_by_category": dict(Counter(
            r["review_category"] for r in reviewed)),
        "test_subset_note": ("testA/testB/testC memberships are reported but "
                             "not counted as additional unique samples."),
        "scope_note": ("This report flags conspicuous caption defects only. "
                       "It does not assert that every remaining caption is "
                       "visually or semantically perfect."),
    }
    args.reviewed_output.parent.mkdir(parents=True, exist_ok=True)
    args.reviewed_output.write_text(json.dumps(
        {"summary": reviewed_summary, "samples": reviewed},
        ensure_ascii=False, indent=2
    ) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Reviewed:")
    print(json.dumps(reviewed_summary, ensure_ascii=False, indent=2))
    for record in records:
        print(f"{record['split']}[{record['index']}] {record['file_name']} "
              f"{record['reasons']}\n  {record['caption']}")


if __name__ == "__main__":
    main()
