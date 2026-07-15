#!/usr/bin/env python3

"""Freeze source-derived linguistic controls for the J-Space G1 sensor."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


AXES = (
    "sadness",
    "surprise",
    "joy",
    "disgust",
    "fear",
    "anger",
    "curiosity",
    "neutral",
)
SEED = "treebeard.jspace.g1.controls.v1"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
PATTERNS = {
    "negation": re.compile(
        r"\b(?:not|never|no|nothing|nobody|neither|cannot|can't|won't|wouldn't|"
        r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|haven't|hasn't|hadn't)\b",
        re.IGNORECASE,
    ),
    "quotation": re.compile(r'''["“”]|\b(?:said|says|wrote|quoted|quote)\b''', re.IGNORECASE),
    "third_person": re.compile(
        r"\b(?:he|she|they|him|her|them|his|hers|their|theirs)\b", re.IGNORECASE
    ),
    "sarcasm_marker": re.compile(
        r"(?:/s\b|\byeah[, ]+right\b|\bas if\b|\bsure[, ]|\btotally\b|\bobviously\b)",
        re.IGNORECASE,
    ),
}
BALANCED_COUNTS = {
    "lexical_echo": 20,
    "negation": 8,
    "quotation": 2,
    "third_person": 15,
}


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anchor_words(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "treebeard.jspace.anchors.v0":
        raise ValueError("unsupported anchor manifest")
    by_axis = {
        axis: {row["word"].casefold() for row in rows}
        for axis, rows in document["axes"].items()
    }
    return by_axis, set().union(*by_axis.values())


def rank(control, row):
    payload = "\0".join((SEED, control, row["source_id"], row["text"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_test(data_dir: Path, anchors_path: Path):
    labels = (data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines()
    by_axis, all_anchors = anchor_words(anchors_path)
    rows = []
    for line_number, line in enumerate(
            (data_dir / "test.tsv").read_text(encoding="utf-8").splitlines(), 1):
        columns = line.split("\t")
        if len(columns) != 3:
            raise ValueError(f"test.tsv:{line_number}: expected three columns")
        text, label_ids, source_id = columns
        source_labels = [labels[int(value)] for value in label_ids.split(",")]
        selected_labels = [axis for axis in AXES if axis in source_labels]
        words = set(WORD_RE.findall(text.casefold()))
        rows.append({
            "source_id": source_id,
            "text": text,
            "labels": selected_labels,
            "single_target": len(source_labels) == 1 and len(selected_labels) == 1,
            "anchor_echo": bool(words & all_anchors),
            "neutral_tedium_echo": bool(words & by_axis["neutral"]),
            "matches": {
                name: bool(pattern.search(text))
                for name, pattern in PATTERNS.items()
            },
        })
    return rows


def choose_balanced(rows, control, count):
    selected = []
    for axis in AXES:
        candidates = []
        for row in rows:
            if not row["single_target"] or row["labels"] != [axis]:
                continue
            if control == "lexical_echo":
                matches = row["anchor_echo"]
            else:
                matches = row["matches"][control]
            if matches:
                candidates.append(row)
        if len(candidates) < count:
            raise ValueError(f"{control}/{axis} has {len(candidates)} rows; {count} required")
        selected.extend(sorted(candidates, key=lambda row: rank(control, row))[:count])
    return selected


def freeze_controls(data_dir: Path, anchors_path: Path, primary_manifest_path: Path):
    rows = parse_test(data_dir, anchors_path)
    primary = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    primary_test_ids = {
        row["source_id"] for row in primary["rows"] if row["split"] == "test"
    }
    memberships = defaultdict(set)
    rows_by_id = {row["source_id"]: row for row in rows}

    for control, count in BALANCED_COUNTS.items():
        for row in choose_balanced(rows, control, count):
            memberships[row["source_id"]].add(control)

    for row in rows:
        if len(row["labels"]) >= 2:
            memberships[row["source_id"]].add("mixed_mood")
        if row["matches"]["sarcasm_marker"] and row["single_target"]:
            memberships[row["source_id"]].add("sarcasm_marker")
        if row["single_target"] and row["labels"] == ["neutral"] and row["neutral_tedium_echo"]:
            memberships[row["source_id"]].add("narrated_tedium")

    neutral_flat = [
        row for row in rows
        if row["single_target"] and row["labels"] == ["neutral"] and
        not row["anchor_echo"] and row["source_id"] not in primary_test_ids
    ]
    for row in sorted(neutral_flat, key=lambda item: rank("neutral_flat", item))[:128]:
        memberships[row["source_id"]].add("neutral_flat")

    output_rows = []
    counts = Counter()
    for source_id in sorted(memberships, key=lambda value: rank("union", rows_by_id[value])):
        row = rows_by_id[source_id]
        controls = sorted(memberships[source_id])
        for control in controls:
            counts[control] += 1
        text_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        output_rows.append({
            "sample_id": f"control:{source_id}",
            "split": "control",
            "label": row["labels"][0] if row["labels"] else "neutral",
            "labels": row["labels"],
            "control_types": controls,
            "source_id": source_id,
            "text_sha256": text_sha,
            "anchor_echo": row["anchor_echo"],
            "text": row["text"],
        })

    for control, per_axis in BALANCED_COUNTS.items():
        observed = Counter()
        for row in output_rows:
            if control in row["control_types"]:
                observed[row["label"]] += 1
        if any(observed[axis] != per_axis for axis in AXES):
            raise AssertionError(f"{control} is not balanced: {observed}")
    if len({row["source_id"] for row in output_rows}) != len(output_rows):
        raise AssertionError("control union contains duplicate source IDs")

    return {
        "schema": "treebeard.jspace.g1.controls.v1",
        "source": {
            "name": "GoEmotions agreement-filtered official test split",
            "project": "https://github.com/google-research/google-research/tree/master/goemotions",
            "test_sha256": sha256_file(data_dir / "test.tsv"),
            "emotions_sha256": sha256_file(data_dir / "emotions.txt"),
        },
        "policy": {
            "selection_seed": SEED,
            "balanced_per_axis": BALANCED_COUNTS,
            "mixed_mood": "all official test rows with at least two target labels",
            "sarcasm_marker": PATTERNS["sarcasm_marker"].pattern,
            "narrated_tedium": "all single-neutral rows containing a frozen neutral/tedium anchor",
            "neutral_flat": "128 single-neutral non-echo rows not in the primary balanced test",
            "overlap": "one activation row may belong to multiple named controls",
            "sparse_controls": "sarcasm_marker and narrated_tedium are report-only because source counts are small",
            "primary_manifest_sha256": sha256_file(primary_manifest_path),
            "anchors_sha256": sha256_file(anchors_path),
        },
        "audit": {
            "unique_rows": len(output_rows),
            "memberships": dict(sorted(counts.items())),
        },
        "rows": output_rows,
    }


def self_test():
    assert PATTERNS["negation"].search("This is not correct.")
    assert PATTERNS["third_person"].search("They entered quietly.")
    assert PATTERNS["sarcasm_marker"].search("Yeah, right, that worked.")
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.controls-self-test.v1"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--primary-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if any(value is None for value in (
            args.data_dir, args.anchors, args.primary_manifest, args.out)):
        parser.error("data, anchors, primary manifest, and output are required")
    document = freeze_controls(args.data_dir, args.anchors, args.primary_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "out": str(args.out),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "unique_rows": document["audit"]["unique_rows"],
        "memberships": document["audit"]["memberships"],
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
