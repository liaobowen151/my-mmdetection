#!/usr/bin/env python3
"""Apply reviewed MFAD validation/test caption and box revisions."""

import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


# 1% of the 1280x960 MFAD image area.  Existing annotations classify areas
# below this boundary as small and areas at/above it as normal.
SMALL_AREA_THRESHOLD = 12288.0
SUBSETS = ("testA", "testB", "testC")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir", type=Path,
        default=Path("/home/lbw/mmdetection/data/MFAD/annotations"))
    parser.add_argument(
        "--review-file", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "mfad_eval_text_anomalies_reviewed.json"))
    parser.add_argument(
        "--revision-file", type=Path,
        default=Path("/home/lbw/mmdetection/tools/analysis_tools/"
                     "mfad_eval_text_anomaly_revisions_preview.json"))
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes. Without this flag, perform preflight only.")
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_atomic(path, payload):
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def scale_class(area):
    return "small" if area < SMALL_AREA_THRESHOLD else "normal"


def annotation_map(payload):
    result = {ann["image_id"]: ann for ann in payload["annotations"]}
    if len(result) != len(payload["annotations"]):
        raise ValueError("Annotations do not have unique image_id values.")
    return result


def identity_counter(payload):
    anns = annotation_map(payload)
    return Counter(
        (image["file_name"], image["caption"],
         tuple(anns[image["id"]]["bbox"]))
        for image in payload["images"]
    )


def subset_filter(name, image):
    if name == "testA":
        return image["lighting"] == 3 and image["scale_cls"] == "normal"
    if name == "testB":
        return image["lighting"] != 3
    if name == "testC":
        return image["scale_cls"] == "small"
    raise ValueError(name)


def build_subset(name, canonical, template):
    canonical_anns = annotation_map(canonical)
    images = []
    annotations = []
    for source_image in canonical["images"]:
        if not subset_filter(name, source_image):
            continue
        new_id = len(images)
        image = copy.deepcopy(source_image)
        image["id"] = new_id
        if name in {"testA", "testB"}:
            for attribute in ("crowded", "scene", "weather", "occlusion"):
                image[attribute] = None
        ann = copy.deepcopy(canonical_anns[source_image["id"]])
        ann["id"] = new_id
        ann["image_id"] = new_id
        ann["original_id"] = new_id
        images.append(image)
        annotations.append(ann)
    result = copy.deepcopy(template)
    result["images"] = images
    result["annotations"] = annotations
    return result


def validate_scale_rule(payload, split):
    anns = annotation_map(payload)
    mismatches = []
    for image in payload["images"]:
        area = float(anns[image["id"]]["area"])
        expected = scale_class(area)
        if image["scale_cls"] != expected:
            mismatches.append((image["id"], image["scale_cls"], expected, area))
    if mismatches:
        raise ValueError(
            f"Scale rule mismatch in {split}: {mismatches[:5]}")


def old_revision_identity(sample):
    return (sample["file_name"], sample["caption"],
            tuple(float(value) for value in sample["bbox_xywh"]))


def preflight_revised_subset_memberships(existing, review):
    test_samples = {
        f"test:{sample['index']}": sample
        for sample in review["samples"] if sample["split"] == "test"
    }
    for name in SUBSETS:
        identities = identity_counter(existing[name])
        for key, sample in test_samples.items():
            actual = identities[old_revision_identity(sample)]
            expected = 1 if name in sample["test_subsets"] else 0
            if actual != expected:
                raise ValueError(
                    f"Unexpected pre-revision membership for {key} in "
                    f"{name}: {actual} != {expected}")


def replacement_row(name, canonical, index):
    image = copy.deepcopy(canonical["images"][index])
    ann = copy.deepcopy(annotation_map(canonical)[image["id"]])
    if name in {"testA", "testB"}:
        for attribute in ("crowded", "scene", "weather", "occlusion"):
            image[attribute] = None
    return image, ann


def reindex_subset(template, rows):
    result = copy.deepcopy(template)
    result["images"] = []
    result["annotations"] = []
    for new_id, (source_image, source_ann) in enumerate(rows):
        image = copy.deepcopy(source_image)
        ann = copy.deepcopy(source_ann)
        image["id"] = new_id
        ann["id"] = new_id
        ann["image_id"] = new_id
        ann["original_id"] = new_id
        result["images"].append(image)
        result["annotations"].append(ann)
    return result


def apply_subset_revisions(name, existing, canonical, review, revisions):
    test_samples = {
        f"test:{sample['index']}": sample
        for sample in review["samples"] if sample["split"] == "test"
    }
    identity_to_key = {
        old_revision_identity(sample): key
        for key, sample in test_samples.items()
    }
    existing_anns = annotation_map(existing)
    rows = []
    seen = Counter()
    for image in existing["images"]:
        ann = existing_anns[image["id"]]
        identity = (image["file_name"], image["caption"], tuple(ann["bbox"]))
        key = identity_to_key.get(identity)
        if key is None:
            rows.append((image, ann))
            continue
        seen[key] += 1
        index = int(key.split(":", 1)[1])
        revised_image = canonical["images"][index]
        if subset_filter(name, revised_image):
            rows.append(replacement_row(name, canonical, index))

    # Add samples newly entering a subset after their revised box changes the
    # scale class.  Existing ordering is retained; new entries are appended.
    for key in revisions:
        if not key.startswith("test:"):
            continue
        index = int(key.split(":", 1)[1])
        revised_image = canonical["images"][index]
        if subset_filter(name, revised_image) and seen[key] == 0:
            rows.append(replacement_row(name, canonical, index))
    return reindex_subset(existing, rows)


def apply_revisions(payloads, review, revisions):
    review_by_key = {
        f"{sample['split']}:{sample['index']}": sample
        for sample in review["samples"]
    }
    if set(review_by_key) != set(revisions):
        raise ValueError(
            "Revision keys do not exactly match reviewed anomaly keys.")

    changed_boxes = []
    changed_texts = []
    for key, revision in revisions.items():
        split, index_text = key.split(":", 1)
        index = int(index_text)
        if split not in {"val", "test"}:
            raise ValueError(f"Unexpected split in revision: {key}")
        payload = payloads[split]
        image = payload["images"][index]
        if image["id"] != index:
            raise ValueError(f"Image ordering/id mismatch at {key}")
        reviewed = review_by_key[key]
        if image["file_name"] != reviewed["file_name"]:
            raise ValueError(f"Filename mismatch at {key}")
        if image["caption"] != reviewed["caption"]:
            raise ValueError(f"Source caption changed unexpectedly at {key}")
        ann = annotation_map(payload)[image["id"]]

        caption = revision["proposed_text"]
        if not caption or not caption.isascii():
            raise ValueError(f"Proposed caption must be non-empty English: {key}")
        image["caption"] = caption
        ann["tokens_positive"] = [[0, len(caption)]]
        changed_texts.append(key)

        if "bbox_xyxy" in revision:
            x1, y1, x2, y2 = [float(value) for value in revision["bbox_xyxy"]]
            if not (0 <= x1 < x2 <= image["width"]
                    and 0 <= y1 < y2 <= image["height"]):
                raise ValueError(f"Invalid proposed bbox at {key}")
            width = x2 - x1
            height = y2 - y1
            area = width * height
            ann["bbox"] = [x1, y1, width, height]
            ann["area"] = area
            image["scale_cls"] = scale_class(area)
            changed_boxes.append(key)
    return changed_texts, changed_boxes


def validate_final(payloads, subsets, revisions):
    for split, payload in payloads.items():
        validate_scale_rule(payload, split)
        anns = annotation_map(payload)
        if len(payload["images"]) != len(payload["annotations"]):
            raise ValueError(f"Image/annotation count mismatch in {split}")
        for image in payload["images"]:
            ann = anns[image["id"]]
            x, y, width, height = ann["bbox"]
            if not (0 <= x < x + width <= image["width"]
                    and 0 <= y < y + height <= image["height"]):
                raise ValueError(f"Out-of-bounds bbox in {split}:{image['id']}")
            if abs(ann["area"] - width * height) > 1e-6:
                raise ValueError(f"Area mismatch in {split}:{image['id']}")
            if ann["tokens_positive"] != [[0, len(image["caption"])]]:
                raise ValueError(
                    f"Token span mismatch in {split}:{image['id']}")

    for key, revision in revisions.items():
        split, index_text = key.split(":", 1)
        index = int(index_text)
        image = payloads[split]["images"][index]
        ann = annotation_map(payloads[split])[image["id"]]
        if image["caption"] != revision["proposed_text"]:
            raise ValueError(f"Caption revision missing at {key}")
        if "bbox_xyxy" in revision:
            x1, y1, x2, y2 = revision["bbox_xyxy"]
            expected = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            if ann["bbox"] != expected:
                raise ValueError(f"BBox revision missing at {key}")

    for name, subset in subsets.items():
        identities = identity_counter(subset)
        for key, revision in revisions.items():
            if not key.startswith("test:"):
                continue
            index = int(key.split(":", 1)[1])
            image = payloads["test"]["images"][index]
            ann = annotation_map(payloads["test"])[image["id"]]
            identity = (image["file_name"], image["caption"],
                        tuple(ann["bbox"]))
            expected = 1 if subset_filter(name, image) else 0
            if identities[identity] != expected:
                raise ValueError(
                    f"Final membership mismatch for {key} in {name}")


def main():
    args = parse_args()
    paths = {
        split: args.annotation_dir / f"rgbtvg_mfad_{split}.json"
        for split in ("val", "test", *SUBSETS)
    }
    payloads = {split: load(paths[split]) for split in ("val", "test")}
    existing_subsets = {name: load(paths[name]) for name in SUBSETS}
    review = load(args.review_file)
    revision_payload = load(args.revision_file)
    revisions = revision_payload["revisions"]

    validate_scale_rule(payloads["val"], "val")
    validate_scale_rule(payloads["test"], "test")
    preflight_revised_subset_memberships(existing_subsets, review)

    changed_texts, changed_boxes = apply_revisions(
        payloads, review, revisions)
    rebuilt_subsets = {
        name: apply_subset_revisions(
            name, existing_subsets[name], payloads["test"], review, revisions)
        for name in SUBSETS
    }
    validate_final(payloads, rebuilt_subsets, revisions)

    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "changed_text_count": len(changed_texts),
        "changed_box_count": len(changed_boxes),
        "changed_text_keys": changed_texts,
        "changed_box_keys": changed_boxes,
        "final_counts": {
            "val": len(payloads["val"]["images"]),
            "test": len(payloads["test"]["images"]),
            **{name: len(payload["images"])
               for name, payload in rebuilt_subsets.items()},
        },
        "subset_membership_changes": {
            name: len(rebuilt_subsets[name]["images"])
                  - len(existing_subsets[name]["images"])
            for name in SUBSETS
        },
    }
    if args.apply:
        dump_atomic(paths["val"], payloads["val"])
        dump_atomic(paths["test"], payloads["test"])
        for name in SUBSETS:
            dump_atomic(paths[name], rebuilt_subsets[name])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
