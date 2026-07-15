#!/usr/bin/env python3

"""Freeze source-disjoint MELD calibration and test rows for J-Space G1 v3."""

import argparse
import csv
import hashlib
import importlib.util
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


AXES = (
    "sadness",
    "surprise",
    "joy",
    "disgust",
    "fear",
    "anger",
    "neutral",
)
UNSUPPORTED_AXES = ("curiosity",)
SOURCE_FILES = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}
SOURCE_REVISION = "e8cedf27b5d2877e198332c957127e16eb214afe"
SELECTION_SEED = "treebeard.jspace.g1.meld.v3"
CALIBRATION_PER_LENGTH_BIN = 32
PRIMARY_PER_LENGTH_BIN = 10
LEXICAL_ECHO_PER_AXIS = 4


def load_g1_helpers():
    path = Path(__file__).with_name("treebeard-jspace-g1-manifest.py")
    spec = importlib.util.spec_from_file_location("treebeard_jspace_g1_manifest_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G1 = load_g1_helpers()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def length_bin(text: str) -> str:
    words = len(text.split())
    if words <= 5:
        return "short"
    if words <= 12:
        return "medium"
    return "long"


def source_id(row) -> str:
    return f"dialogue-{row['Dialogue_ID']}:utterance-{row['Utterance_ID']}"


def selection_rank(row, scope: str) -> str:
    payload = "\0".join((
        SELECTION_SEED,
        scope,
        row["source_split"],
        row["label"],
        row["source_id"],
        row["normalized_text"],
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def excluded_normalized_texts(paths):
    excluded = set()
    audit = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        rows = document.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"exclusion manifest has no rows: {path}")
        texts = {
            G1.normalized_text(row["text"])
            for row in rows
            if isinstance(row.get("text"), str) and G1.normalized_text(row["text"])
        }
        excluded.update(texts)
        audit.append({
            "manifest": path.name,
            "sha256": sha256_file(path),
            "normalized_texts": len(texts),
        })
    return excluded, audit


def parse_meld(data_dir: Path, anchors_path: Path, excluded_texts):
    _, anchor_roots = G1.build_anchor_roots(anchors_path)
    rows = []
    normalized_counts = Counter()
    for split, filename in SOURCE_FILES.items():
        path = data_dir / filename
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "Utterance", "Emotion", "Dialogue_ID", "Utterance_ID", "Speaker",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"unexpected MELD columns in {filename}")
            for line, source_row in enumerate(reader, 2):
                label = source_row["Emotion"]
                if label not in AXES:
                    raise ValueError(f"{filename}:{line}: unexpected emotion {label!r}")
                text = source_row["Utterance"].strip()
                normalized = G1.normalized_text(text)
                if not normalized:
                    continue
                row = {
                    "source_split": split,
                    "source_id": source_id(source_row),
                    "label": label,
                    "speaker": source_row["Speaker"],
                    "text": text,
                    "normalized_text": normalized,
                    "length_bin": length_bin(text),
                    "anchor_echo_tokens": G1.echo_tokens(text, anchor_roots),
                }
                rows.append(row)
                normalized_counts[normalized] += 1

    eligible = []
    rejection_counts = Counter()
    for row in rows:
        split = row["source_split"]
        if normalized_counts[row["normalized_text"]] != 1:
            rejection_counts[f"{split}:duplicate_text"] += 1
            continue
        if row["normalized_text"] in excluded_texts:
            rejection_counts[f"{split}:prior_g1_text"] += 1
            continue
        eligible.append(row)
    return eligible, rejection_counts


def manifest_row(row, split: str, scope: str):
    return {
        "sample_id": f"meld:{split}:{scope}:{row['label']}:{row['source_id']}",
        "split": split,
        "scope": scope,
        "source_split": row["source_split"],
        "source_id": row["source_id"],
        "speaker": row["speaker"],
        "label": row["label"],
        "length_bin": row["length_bin"],
        "anchor_echo": bool(row["anchor_echo_tokens"]),
        "anchor_echo_tokens": row["anchor_echo_tokens"],
        "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
        "text": row["text"],
    }


def select_calibration(eligible):
    rows = []
    available = {}
    for label in AXES:
        for band in ("short", "medium", "long"):
            pool = [
                row for row in eligible
                if row["source_split"] == "train" and
                row["label"] == label and row["length_bin"] == band
            ]
            available[f"{label}:{band}"] = len(pool)
            if len(pool) < CALIBRATION_PER_LENGTH_BIN:
                raise ValueError(
                    f"MELD train {label}/{band} has {len(pool)} rows; "
                    f"{CALIBRATION_PER_LENGTH_BIN} required"
                )
            chosen = sorted(
                pool, key=lambda row: selection_rank(row, "calibration")
            )[:CALIBRATION_PER_LENGTH_BIN]
            rows.extend(manifest_row(row, "calibration", "source_calibration") for row in chosen)
    return rows, available


def select_test(eligible):
    rows = []
    available = {}
    used_ids = set()
    for label in AXES:
        for band in ("short", "medium", "long"):
            pool = [
                row for row in eligible
                if row["source_split"] == "test" and
                row["label"] == label and row["length_bin"] == band and
                not row["anchor_echo_tokens"]
            ]
            available[f"primary:{label}:{band}"] = len(pool)
            if len(pool) < PRIMARY_PER_LENGTH_BIN:
                raise ValueError(
                    f"MELD test primary {label}/{band} has {len(pool)} rows; "
                    f"{PRIMARY_PER_LENGTH_BIN} required"
                )
            chosen = sorted(
                pool, key=lambda row: selection_rank(row, "primary")
            )[:PRIMARY_PER_LENGTH_BIN]
            for row in chosen:
                used_ids.add(row["source_id"])
                rows.append(manifest_row(row, "test", "primary_anchor_free"))

        echo_pool = [
            row for row in eligible
            if row["source_split"] == "test" and row["label"] == label and
            row["anchor_echo_tokens"] and row["source_id"] not in used_ids
        ]
        available[f"lexical_echo:{label}"] = len(echo_pool)
        if len(echo_pool) < LEXICAL_ECHO_PER_AXIS:
            raise ValueError(
                f"MELD test lexical echo {label} has {len(echo_pool)} rows; "
                f"{LEXICAL_ECHO_PER_AXIS} required"
            )
        chosen = sorted(
            echo_pool, key=lambda row: selection_rank(row, "lexical_echo")
        )[:LEXICAL_ECHO_PER_AXIS]
        rows.extend(manifest_row(row, "test", "lexical_echo") for row in chosen)
    return rows, available


def freeze_manifest(kind: str, data_dir: Path, anchors_path: Path,
                    exclusion_paths, source_revision: str):
    if source_revision != SOURCE_REVISION:
        raise ValueError(
            f"MELD source revision mismatch: {source_revision} != {SOURCE_REVISION}"
        )
    excluded_texts, exclusions = excluded_normalized_texts(exclusion_paths)
    eligible, rejection_counts = parse_meld(data_dir, anchors_path, excluded_texts)
    if kind == "calibration":
        rows, available = select_calibration(eligible)
        expected = len(AXES) * 3 * CALIBRATION_PER_LENGTH_BIN
        source_split = "train"
    elif kind == "test":
        rows, available = select_test(eligible)
        expected = len(AXES) * (3 * PRIMARY_PER_LENGTH_BIN + LEXICAL_ECHO_PER_AXIS)
        source_split = "test"
    else:
        raise ValueError(f"unsupported manifest kind: {kind}")

    if len(rows) != expected:
        raise AssertionError(f"wrong row count: {len(rows)} != {expected}")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise AssertionError("selected MELD rows contain duplicate source IDs")
    if len({row["text_sha256"] for row in rows}) != len(rows):
        raise AssertionError("selected MELD rows contain duplicate text")
    if any(G1.normalized_text(row["text"]) in excluded_texts for row in rows):
        raise AssertionError("selected MELD rows overlap prior G1 text")

    return {
        "schema": "treebeard.jspace.g1.dataset.v3",
        "source": {
            "name": "MELD text utterance emotion annotations",
            "project": "https://github.com/declare-lab/MELD",
            "paper": "https://arxiv.org/abs/1810.02508",
            "license": "GPL-3.0 (source repository)",
            "revision": source_revision,
            "files_sha256": {
                filename: sha256_file(data_dir / filename)
                for filename in SOURCE_FILES.values()
            },
        },
        "policy": {
            "kind": kind,
            "axes": list(AXES),
            "unsupported_axes": list(UNSUPPORTED_AXES),
            "source_split": source_split,
            "selection": "lowest fixed SHA-256 rank within label, scope, and length band",
            "selection_seed": SELECTION_SEED,
            "length_bands_whitespace_words": {
                "short": "1..5", "medium": "6..12", "long": "13+",
            },
            "representation": "arithmetic mean of every literal prompt-token layer-35 l_out residual",
            "calibration_per_axis_per_length_band": (
                CALIBRATION_PER_LENGTH_BIN if kind == "calibration" else None
            ),
            "test_primary_per_axis_per_length_band": (
                PRIMARY_PER_LENGTH_BIN if kind == "test" else None
            ),
            "test_lexical_echo_per_axis": (
                LEXICAL_ECHO_PER_AXIS if kind == "test" else None
            ),
            "global_meld_normalized_text_unique": True,
            "prior_g1_normalized_text_exclusions": exclusions,
            "curiosity_policy": (
                "unsupported: MELD has no curiosity label; no proxy or relabeling is permitted"
            ),
            "test_use": (
                "not captured or indexed until the v3 fit artifact and exact gates are committed"
                if kind == "test" else
                "may be captured and indexed only for v3 source calibration"
            ),
        },
        "audit": {
            "eligible_selected_scope": available,
            "excluded_prior_g1_normalized_texts": len(excluded_texts),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rows": len(rows),
        },
        "rows": rows,
    }


def self_test():
    assert length_bin("one two three") == "short"
    assert length_bin("one two three four five six") == "medium"
    assert length_bin(" ".join(str(i) for i in range(13))) == "long"
    row = {
        "source_split": "test", "label": "joy", "source_id": "d:1",
        "normalized_text": "a row",
    }
    assert selection_rank(row, "primary") == selection_rank(dict(row), "primary")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps({"rows": [{"text": "  Same\tText "}]}), encoding="utf-8")
        excluded, audit = excluded_normalized_texts([path])
        assert excluded == {"same text"}
        assert audit[0]["normalized_texts"] == 1
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.meld-manifest-self-test.v3"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("calibration", "test"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.kind is None or args.data_dir is None or args.anchors is None or args.out is None:
        parser.error("--kind, --data-dir, --anchors, and --out are required")
    if not args.exclude_manifest:
        parser.error("at least one --exclude-manifest is required")

    document = freeze_manifest(
        args.kind, args.data_dir, args.anchors, args.exclude_manifest,
        args.source_revision,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "status": "pass",
        "out": str(args.out),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "rows": len(document["rows"]),
        "kind": args.kind,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
