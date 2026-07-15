#!/usr/bin/env python3

"""Freeze the multi-source instruction-routing J-Space G1 v4 manifests."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


AXES = ("sadness", "surprise", "joy", "disgust", "fear", "anger", "neutral")
ED_LABELS = {
    "sadness": ("sad", "devastated", "lonely", "disappointed"),
    "surprise": ("surprised",),
    "joy": ("joyful", "excited", "content", "proud", "grateful", "hopeful"),
    "disgust": ("disgusted",),
    "fear": ("afraid", "terrified", "anxious", "apprehensive"),
    "anger": ("angry", "furious", "annoyed"),
}
DD_LABELS = {
    "sadness": "sadness",
    "surprise": "surprise",
    "joy": "happiness",
    "disgust": "disgust",
    "fear": "fear",
    "anger": "anger",
    "neutral": "no emotion",
}
CAL_COUNTS = {"empatheticdialogues": 64, "dailydialog": 64}
TEST_COUNTS = {"empatheticdialogues": 32, "dailydialog": 8}
SELECTION_SEED = "treebeard.jspace.g1.instruction-routing.v4"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SPACE_RE = re.compile(r"\s+")
MORPH_SUFFIXES = ("inesses", "nesses", "ingly", "iness", "ness", "edly", "ing", "ed", "es", "s")
MAX_WORDS = 64


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(text):
    return SPACE_RE.sub(" ", text.casefold()).strip()


def morphology_roots(word):
    roots = {word}
    for suffix in MORPH_SUFFIXES:
        if not word.endswith(suffix) or len(word) <= len(suffix) + 2:
            continue
        root = word[: -len(suffix)]
        roots.add(root)
        if root.endswith("i"):
            roots.add(root[:-1] + "y")
        if len(root) > 2 and root[-1] == root[-2]:
            roots.add(root[:-1])
    return roots


def anchor_roots(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "treebeard.jspace.anchors.v0":
        raise ValueError("unsupported anchor manifest")
    words = {
        row["word"].casefold()
        for rows in document["axes"].values()
        for row in rows
    }
    words.update(AXES)
    roots = set()
    for word in words:
        roots.update(morphology_roots(word))
    return words, roots


def echoes(text, roots):
    return sorted({
        token for token in WORD_RE.findall(text.casefold())
        if morphology_roots(token) & roots
    })


def clean_ed(text):
    return SPACE_RE.sub(" ", text.replace("_comma_", ",")).strip()


def load_empathetic(directory):
    reverse = {
        source_label: axis
        for axis, source_labels in ED_LABELS.items()
        for source_label in source_labels
    }
    result = []
    csv.field_size_limit(10_000_000)
    for split in ("train", "valid", "test"):
        with (directory / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["utterance_idx"] != "1" or row["context"] not in reverse:
                    continue
                text = clean_ed(row["prompt"])
                result.append({
                    "source": "empatheticdialogues",
                    "source_split": split,
                    "source_id": row["conv_id"],
                    "source_label": row["context"],
                    "label": reverse[row["context"]],
                    "text": text,
                })
    return result


def load_dailydialog(path):
    reverse = {source_label: axis for axis, source_label in DD_LABELS.items()}
    result = []
    for dialogue in json.loads(path.read_text(encoding="utf-8")):
        for turn in dialogue["turns"]:
            source_label = turn["emotion"]
            if source_label not in reverse:
                continue
            result.append({
                "source": "dailydialog",
                "source_split": dialogue["data_split"],
                "source_id": f"{dialogue['dialogue_id']}:turn-{turn['utt_idx']}",
                "source_label": source_label,
                "label": reverse[source_label],
                "text": SPACE_RE.sub(" ", turn["utterance"]).strip(),
            })
    return result


def rank(row, split):
    payload = "\0".join((
        SELECTION_SEED, split, row["source"], row["source_split"], row["label"],
        row["source_id"], row["normalized_text"],
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze(rows, split, roots, source_files, anchors_path):
    source_splits = {
        "calibration": {"empatheticdialogues": "train", "dailydialog": "train"},
        "test": {"empatheticdialogues": "test", "dailydialog": "test"},
    }[split]
    counts = CAL_COUNTS if split == "calibration" else TEST_COUNTS
    pools = defaultdict(list)
    for row in rows:
        if row["source_split"] == source_splits[row["source"]]:
            pools[(row["source"], row["label"])].append(row)

    selected = []
    available = {}
    for source in ("empatheticdialogues", "dailydialog"):
        supported = tuple(ED_LABELS) if source == "empatheticdialogues" else AXES
        for axis in supported:
            pool = pools[(source, axis)]
            available[f"{source}:{axis}"] = len(pool)
            needed = counts[source]
            if len(pool) < needed:
                raise ValueError(f"{split}/{source}/{axis}: {len(pool)} available, {needed} required")
            selected.extend(sorted(pool, key=lambda row: rank(row, split))[:needed])

    output_rows = []
    for row in selected:
        echo = echoes(row["text"], roots)
        output_rows.append({
            "sample_id": f"v4:{split}:{row['source']}:{row['label']}:{row['source_id']}",
            "split": split,
            "scope": "source_calibration" if split == "calibration" else "source_holdout",
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
    axis_counts = Counter(row["label"] for row in output_rows)
    source_axis_counts = Counter((row["source"], row["label"]) for row in output_rows)
    echo_counts = Counter((row["source"], row["label"]) for row in output_rows if row["anchor_echo"])
    return {
        "schema": "treebeard.jspace.g1.dataset.v4",
        "source": {
            "name": "EmpatheticDialogues plus ConvLab-normalized DailyDialog",
            "files_sha256": source_files,
            "use": "private research; this freezer does not assert redistribution rights",
        },
        "policy": {
            "kind": "source_calibration" if split == "calibration" else "source_holdout",
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
            "counts_per_source_axis": counts,
            "maximum_whitespace_words": MAX_WORDS,
            "cross_source_normalized_text_deduplication": True,
            "anchor_echo": {
                "retained_and_stratified": True,
                "anchor_manifest_sha256": sha256_file(anchors_path),
            },
            "empatheticdialogues_mapping": ED_LABELS,
            "dailydialog_mapping": DD_LABELS,
            "test_use": (
                "untouched until the v4 representation, calibration protocol, and gates are committed"
                if split == "test" else "fit calibration only; never select from v4 holdout rows"
            ),
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
    parser.add_argument("--calibration-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    args = parser.parse_args()

    words, roots = anchor_roots(args.anchors)
    rows = load_empathetic(args.empathetic_dir) + load_dailydialog(args.dailydialog_json)
    normalized_counts = Counter(normalized_text(row["text"]) for row in rows)
    filtered = []
    rejected = Counter()
    for row in rows:
        row["normalized_text"] = normalized_text(row["text"])
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
    calibration = freeze(filtered, "calibration", roots, source_files, args.anchors)
    test = freeze(filtered, "test", roots, source_files, args.anchors)
    calibration["audit"]["global_rejections"] = dict(sorted(rejected.items()))
    test["audit"]["global_rejections"] = dict(sorted(rejected.items()))
    calibration["audit"]["anchor_word_count"] = len(words)
    test["audit"]["anchor_word_count"] = len(words)

    overlap = {row["text_sha256"] for row in calibration["rows"]} & {
        row["text_sha256"] for row in test["rows"]
    }
    if overlap:
        raise AssertionError("calibration and test manifests overlap")
    for path, document in ((args.calibration_out, calibration), (args.test_out, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"out": str(path), "rows": len(document["rows"]), "sha256": sha256_file(path)}))


if __name__ == "__main__":
    main()
