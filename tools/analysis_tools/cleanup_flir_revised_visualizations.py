#!/usr/bin/env python3
"""Remove superseded FLIR anomaly previews and rebuild folder indexes."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from visualize_flir_text_anomalies import make_contact_sheets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "flir_text_anomaly_visualizations"))
    parser.add_argument(
        "--revised-dir", type=Path,
        default=Path("/home/lbw/mmdetection/analysis_demo/"
                     "flir_text_anomaly_visualizations_revised"))
    parser.add_argument(
        "--revision-file", type=Path,
        default=Path("/home/lbw/mmdetection/tools/analysis_tools/"
                     "flir_test_text_anomaly_revisions_preview.json"))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path, payload):
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


def main():
    args = parse_args()
    manifest_path = args.source_dir / "manifest.json"
    manifest = load(manifest_path)
    revised_manifest = load(args.revised_dir / "manifest.json")
    revision_keys = set(load(args.revision_file)["revisions"])
    revised_keys = {
        f"{sample['split']}:{sample['index']}"
        for sample in revised_manifest["samples"]
    }
    if revision_keys != revised_keys:
        raise ValueError("Revised visualization set does not match revisions.")

    removed = []
    retained = []
    for sample in manifest["samples"]:
        key = f"{sample['split']}:{sample['index']}"
        if key in revision_keys:
            removed.append(sample)
        else:
            retained.append(sample)
    source_keys = {
        f"{sample['split']}:{sample['index']}" for sample in manifest["samples"]
    }
    removable_keys = revision_keys & source_keys
    already_removed_keys = revision_keys - source_keys
    if {f"{sample['split']}:{sample['index']}" for sample in removed} != removable_keys:
        raise ValueError("Source partition does not match removable revisions.")
    for sample in removed:
        if not Path(sample["visualization"]).is_file():
            raise FileNotFoundError(sample["visualization"])
    for sample in revised_manifest["samples"]:
        if not Path(sample["visualization"]).is_file():
            raise FileNotFoundError(sample["visualization"])

    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "removed_old_visualizations": len(removed),
        "retained_old_visualizations": len(retained),
        "newly_removed_keys": sorted(removable_keys),
        "already_removed_keys": sorted(already_removed_keys),
    }
    if args.apply:
        for sample in removed:
            Path(sample["visualization"]).unlink()
        for contact in manifest["contact_sheets"]:
            path = Path(contact)
            if path.is_file():
                path.unlink()
        rendered = [(sample, Path(sample["visualization"]))
                    for sample in retained]
        contact_sheets = make_contact_sheets(rendered, args.source_dir)
        manifest["sample_count"] = len(retained)
        manifest["contact_sheets"] = contact_sheets
        manifest["samples"] = retained
        manifest["removed_as_revised"] = sorted(
            set(manifest.get("removed_as_revised", [])) | revision_keys)
        manifest["revised_visualization_dir"] = str(args.revised_dir)
        write_atomic(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
