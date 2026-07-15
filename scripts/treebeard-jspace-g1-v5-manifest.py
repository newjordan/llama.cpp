#!/usr/bin/env python3

"""Freeze the fresh-validation J-Space G1 v5 holdout manifest."""

import argparse
import hashlib
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path


V4 = runpy.run_path(Path(__file__).with_name("treebeard-jspace-g1-v4-manifest.py"))
AXES = tuple(V4["AXES"])
ED_LABELS = V4["ED_LABELS"]
DD_LABELS = V4["DD_LABELS"]
MAX_WORDS = V4["MAX_WORDS"]
SELECTION_SEED = "treebeard.jspace.g1.hierarchical-routing.v5"
ED_COUNTS = {axis: 64 for axis in ED_LABELS}
DD_COUNTS = {
    "sadness": 32,
    "surprise": 32,
    "joy": 32,
    "disgust": 1,
    "fear": 8,
    "anger": 32,
    "neutral": 128,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(row):
    payload = "\0".join((
        SELECTION_SEED,
        row["source"],
        row["source_split"],
        row["label"],
        row["source_id"],
        row["normalized_text"],
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_consumed_hashes(paths):
    hashes = set()
    documents = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        documents.append({"path": str(path), "sha256": sha256_file(path)})
        hashes.update(row["text_sha256"] for row in document["rows"])
    return hashes, documents


def freeze(rows, roots, source_files, anchors_path, consumed_paths):
    pools = defaultdict(list)
    for row in rows:
        wanted_split = "valid" if row["source"] == "empatheticdialogues" else "validation"
        if row["source_split"] == wanted_split:
            pools[(row["source"], row["label"])].append(row)

    selected = []
    available = {}
    for source, counts in (("empatheticdialogues", ED_COUNTS), ("dailydialog", DD_COUNTS)):
        for axis, needed in counts.items():
            pool = pools[(source, axis)]
            available[f"{source}:{axis}"] = len(pool)
            if len(pool) < needed:
                raise ValueError(f"{source}/{axis}: {len(pool)} available, {needed} required")
            selected.extend(sorted(pool, key=rank)[:needed])

    output_rows = []
    for row in selected:
        echo = V4["echoes"](row["text"], roots)
        output_rows.append({
            "sample_id": f"v5:holdout:{row['source']}:{row['label']}:{row['source_id']}",
            "split": "holdout",
            "scope": "fresh_validation_holdout",
            "source": row["source"],
            "source_split": row["source_split"],
            "source_id": row["source_id"],
            "source_label": row["source_label"],
            "label": row["label"],
            "anchor_echo": bool(echo),
            "anchor_echo_tokens": echo,
            "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            "text": row["text"],
        })

    if len({row["sample_id"] for row in output_rows}) != len(output_rows):
        raise AssertionError("duplicate sample IDs")
    if len({row["text_sha256"] for row in output_rows}) != len(output_rows):
        raise AssertionError("duplicate selected texts")
    consumed_hashes, consumed_documents = load_consumed_hashes(consumed_paths)
    overlap = consumed_hashes & {row["text_sha256"] for row in output_rows}
    if overlap:
        raise AssertionError("fresh holdout overlaps a consumed v4 manifest")

    axis_counts = Counter(row["label"] for row in output_rows)
    source_axis_counts = Counter((row["source"], row["label"]) for row in output_rows)
    echo_counts = Counter((row["source"], row["label"])
                          for row in output_rows if row["anchor_echo"])
    return {
        "schema": "treebeard.jspace.g1.dataset.v5",
        "source": {
            "name": "EmpatheticDialogues plus ConvLab-normalized DailyDialog",
            "files_sha256": source_files,
            "use": "private research; this freezer does not assert redistribution rights",
        },
        "policy": {
            "kind": "fresh_validation_holdout",
            "axes": list(AXES),
            "representation": "fixed instruction-probed ordered label-token logits",
            "routing_prompt_prefix": (
                "You are an emotion router. Choose exactly one label: sadness, surprise, joy, "
                "disgust, fear, anger, neutral.\nText:\n"
            ),
            "routing_prompt_suffix": "\nLabel:",
            "verbalizer_token_ids": [49166, 12395, 15420, 64797, 8415, 18654, 20002],
            "selection": "lowest fixed-seed SHA-256 rank within source and mapped axis",
            "selection_seed": SELECTION_SEED,
            "counts_per_source_axis": {
                "empatheticdialogues": ED_COUNTS,
                "dailydialog": DD_COUNTS,
            },
            "maximum_whitespace_words": MAX_WORDS,
            "cross_source_normalized_text_deduplication": True,
            "anchor_echo": {
                "retained_and_stratified": True,
                "anchor_manifest_sha256": sha256_file(anchors_path),
            },
            "empatheticdialogues_mapping": ED_LABELS,
            "dailydialog_mapping": DD_LABELS,
            "holdout_use": (
                "untouched until the v5 decision rule, fitted artifact, and gates are frozen"
            ),
            "daily_dialog_disgust_use": (
                "report-only because validation contains one eligible positive row"
            ),
            "consumed_manifests": consumed_documents,
        },
        "audit": {
            "available": dict(sorted(available.items())),
            "axis_counts": dict(sorted(axis_counts.items())),
            "source_axis_counts": {
                f"{source}:{axis}": count
                for (source, axis), count in sorted(source_axis_counts.items())
            },
            "anchor_echo_counts": {
                f"{source}:{axis}": count
                for (source, axis), count in sorted(echo_counts.items())
            },
            "rows": len(output_rows),
        },
        "rows": output_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empathetic-dir", type=Path, required=True)
    parser.add_argument("--dailydialog-json", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--consumed-manifest", action="append", type=Path, required=True)
    parser.add_argument("--holdout-out", type=Path, required=True)
    args = parser.parse_args()

    words, roots = V4["anchor_roots"](args.anchors)
    rows = V4["load_empathetic"](args.empathetic_dir) + \
        V4["load_dailydialog"](args.dailydialog_json)
    normalized_counts = Counter(V4["normalized_text"](row["text"]) for row in rows)
    filtered = []
    rejected = Counter()
    for row in rows:
        row["normalized_text"] = V4["normalized_text"](row["text"])
        if not row["normalized_text"]:
            rejected["empty"] += 1
        elif len(row["text"].split()) > MAX_WORDS:
            rejected["too_long"] += 1
        elif normalized_counts[row["normalized_text"]] != 1:
            rejected["duplicate_text"] += 1
        else:
            filtered.append(row)

    source_files = {
        "empatheticdialogues/train.csv": sha256_file(args.empathetic_dir / "train.csv"),
        "empatheticdialogues/valid.csv": sha256_file(args.empathetic_dir / "valid.csv"),
        "empatheticdialogues/test.csv": sha256_file(args.empathetic_dir / "test.csv"),
        "dailydialog/dialogues.json": sha256_file(args.dailydialog_json),
    }
    document = freeze(filtered, roots, source_files, args.anchors,
                      args.consumed_manifest)
    document["audit"]["global_rejections"] = dict(sorted(rejected.items()))
    document["audit"]["anchor_word_count"] = len(words)
    args.holdout_out.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.holdout_out),
        "rows": len(document["rows"]),
        "sha256": sha256_file(args.holdout_out),
    }))


if __name__ == "__main__":
    main()
