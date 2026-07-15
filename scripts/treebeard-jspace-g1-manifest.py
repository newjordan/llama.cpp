#!/usr/bin/env python3

"""Freeze the anchor-free GoEmotions subset used by the J-Space G1 gate."""

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
SOURCE_SPLITS = {
    "train": "train.tsv",
    "calibration": "dev.tsv",
    "test": "test.tsv",
}
DEFAULT_COUNTS = {
    "train": 224,
    "calibration": 32,
    "test": 32,
}
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SPACE_RE = re.compile(r"\s+")
MORPH_SUFFIXES = (
    "inesses",
    "nesses",
    "ingly",
    "iness",
    "ness",
    "edly",
    "ing",
    "ed",
    "es",
    "s",
)
SELECTION_SEED = "treebeard.jspace.g1.goemotions.v1"
V2_SELECTION_SEED = "treebeard.jspace.g1.goemotions.residual-test.v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def morphology_roots(word: str):
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


def build_anchor_roots(anchor_path: Path):
    document = json.loads(anchor_path.read_text(encoding="utf-8"))
    if document.get("schema") != "treebeard.jspace.anchors.v0":
        raise ValueError("unsupported J-Space anchor manifest schema")
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


def echo_tokens(text: str, anchor_roots):
    echoes = []
    for token in WORD_RE.findall(text.casefold()):
        if morphology_roots(token) & anchor_roots:
            echoes.append(token)
    return sorted(set(echoes))


def parse_source(data_dir: Path, anchors_path: Path):
    labels_path = data_dir / "emotions.txt"
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    if len(labels) != 28 or labels[-1] != "neutral":
        raise ValueError("unexpected GoEmotions label manifest")

    anchor_words, anchor_roots = build_anchor_roots(anchors_path)
    candidates = defaultdict(list)
    rejection_counts = Counter()
    normalized_counts = Counter()

    for output_split, filename in SOURCE_SPLITS.items():
        path = data_dir / filename
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            columns = raw_line.split("\t")
            if len(columns) != 3:
                raise ValueError(f"{filename}:{line_number}: expected three TSV columns")
            text, label_ids, source_id = columns
            row_labels = tuple(labels[int(value)] for value in label_ids.split(","))
            selected = tuple(label for label in row_labels if label in AXES)
            if len(row_labels) != 1 or len(selected) != 1:
                rejection_counts[f"{output_split}:not_single_target"] += 1
                continue
            echoes = echo_tokens(text, anchor_roots)
            if echoes:
                rejection_counts[f"{output_split}:anchor_echo"] += 1
                continue
            key = normalized_text(text)
            if not key:
                rejection_counts[f"{output_split}:empty"] += 1
                continue
            row = {
                "source_id": source_id,
                "source_split": output_split,
                "label": selected[0],
                "text": text,
                "normalized_text": key,
            }
            candidates[(output_split, selected[0])].append(row)
            normalized_counts[key] += 1

    for key, rows in list(candidates.items()):
        unique_rows = []
        for row in rows:
            if normalized_counts[row["normalized_text"]] != 1:
                rejection_counts[f"{key[0]}:duplicate_text"] += 1
                continue
            unique_rows.append(row)
        candidates[key] = unique_rows

    return candidates, rejection_counts, anchor_words, anchor_roots


def selection_rank(row, seed=SELECTION_SEED):
    payload = "\0".join((
        seed,
        row["source_split"],
        row["label"],
        row["source_id"],
        row["normalized_text"],
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def excluded_source_ids(manifest_paths):
    excluded = set()
    manifests = []
    for path in manifest_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document.get("rows"), list):
            raise ValueError(f"exclusion manifest has no rows: {path}")
        source_ids = {
            row["source_id"] for row in document["rows"]
            if isinstance(row.get("source_id"), str) and row["source_id"]
        }
        if len(source_ids) != len(document["rows"]):
            raise ValueError(f"exclusion manifest has missing or duplicate source IDs: {path}")
        excluded.update(source_ids)
        manifests.append({
            "manifest": path.name,
            "sha256": sha256_file(path),
            "source_ids": len(source_ids),
        })
    return excluded, manifests


def freeze_residual_test_manifest(
        data_dir: Path, anchors_path: Path, exclude_paths, count_per_axis: int):
    candidates, rejection_counts, anchor_words, anchor_roots = parse_source(
        data_dir, anchors_path
    )
    excluded, exclusions = excluded_source_ids(exclude_paths)
    rows = []
    available = {}
    source_split_counts = Counter()
    for label in AXES:
        pool = [
            row
            for split in SOURCE_SPLITS
            for row in candidates[(split, label)]
            if row["source_id"] not in excluded
        ]
        available[label] = len(pool)
        if len(pool) < count_per_axis:
            raise ValueError(
                f"residual pool/{label} has {len(pool)} eligible rows; "
                f"{count_per_axis} required"
            )
        chosen = sorted(
            pool, key=lambda row: selection_rank(row, V2_SELECTION_SEED)
        )[:count_per_axis]
        for row in chosen:
            source_split_counts[row["source_split"]] += 1
            rows.append({
                "sample_id": f"residual-test:{label}:{row['source_split']}:{row['source_id']}",
                "split": "test",
                "source_split": row["source_split"],
                "label": label,
                "source_id": row["source_id"],
                "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                "anchor_echo": False,
                "text": row["text"],
            })

    if len(rows) != len(AXES) * count_per_axis:
        raise AssertionError("residual test manifest has the wrong row count")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise AssertionError("residual test manifest contains duplicate source IDs")
    if {row["source_id"] for row in rows} & excluded:
        raise AssertionError("residual test manifest overlaps an exclusion manifest")
    if len({row["text_sha256"] for row in rows}) != len(rows):
        raise AssertionError("residual test manifest contains duplicate text")
    if any(echo_tokens(row["text"], anchor_roots) for row in rows):
        raise AssertionError("residual test manifest retained an anchor echo")

    source_files = {
        filename: sha256_file(data_dir / filename)
        for filename in (*SOURCE_SPLITS.values(), "emotions.txt")
    }
    return {
        "schema": "treebeard.jspace.g1.dataset.v2",
        "source": {
            "name": "GoEmotions simplified agreement-filtered residual pool",
            "project": "https://github.com/google-research/google-research/tree/master/goemotions",
            "paper": "https://aclanthology.org/2020.acl-main.372/",
            "license": "Apache-2.0 (google-research repository)",
            "files_sha256": source_files,
        },
        "policy": {
            "axes": list(AXES),
            "single_label_only": True,
            "selection": "lowest SHA-256 rank across all residual upstream splits within class",
            "selection_seed": V2_SELECTION_SEED,
            "counts_per_axis": count_per_axis,
            "upstream_split_handling": (
                "pooled only after excluding every v1 primary and control source ID; "
                "source_split remains recorded for audit"
            ),
            "representation": "arithmetic mean of every literal prompt-token l_out residual",
            "anchor_scrub": {
                "scope": "all eight axis anchor clusters plus axis names",
                "casefolded_word_count": len(anchor_words),
                "morphology_root_count": len(anchor_roots),
                "suffixes": list(MORPH_SUFFIXES),
                "anchor_manifest_sha256": sha256_file(anchors_path),
            },
            "cross_split_normalized_text_deduplication": True,
            "exclusions": exclusions,
            "test_use": "untouched until the v2 representation, model, calibration, and gates are frozen",
        },
        "audit": {
            "eligible_after_v1_exclusion": available,
            "excluded_source_ids": len(excluded),
            "selected_source_splits": dict(sorted(source_split_counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rows": len(rows),
        },
        "rows": rows,
    }


def freeze_manifest(data_dir: Path, anchors_path: Path, counts):
    candidates, rejection_counts, anchor_words, anchor_roots = parse_source(
        data_dir, anchors_path
    )
    rows = []
    available = {}
    for split in SOURCE_SPLITS:
        for label in AXES:
            pool = candidates[(split, label)]
            available[f"{split}:{label}"] = len(pool)
            needed = counts[split]
            if len(pool) < needed:
                raise ValueError(
                    f"{split}/{label} has {len(pool)} eligible rows; {needed} required"
                )
            chosen = sorted(pool, key=selection_rank)[:needed]
            for row in chosen:
                text_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                rows.append({
                    "sample_id": f"{split}:{label}:{row['source_id']}",
                    "split": split,
                    "label": label,
                    "source_id": row["source_id"],
                    "text_sha256": text_sha,
                    "anchor_echo": False,
                    "text": row["text"],
                })

    observed = Counter((row["split"], row["label"]) for row in rows)
    for split in SOURCE_SPLITS:
        for label in AXES:
            if observed[(split, label)] != counts[split]:
                raise AssertionError("frozen manifest is not class balanced")
    if len({row["text_sha256"] for row in rows}) != len(rows):
        raise AssertionError("frozen manifest contains duplicate text")
    if any(echo_tokens(row["text"], anchor_roots) for row in rows):
        raise AssertionError("frozen manifest retained an anchor echo")

    source_files = {
        filename: sha256_file(data_dir / filename)
        for filename in (*SOURCE_SPLITS.values(), "emotions.txt")
    }
    return {
        "schema": "treebeard.jspace.g1.dataset.v1",
        "source": {
            "name": "GoEmotions simplified agreement-filtered splits",
            "project": "https://github.com/google-research/google-research/tree/master/goemotions",
            "paper": "https://aclanthology.org/2020.acl-main.372/",
            "license": "Apache-2.0 (google-research repository)",
            "files_sha256": source_files,
        },
        "policy": {
            "axes": list(AXES),
            "single_label_only": True,
            "upstream_split_mapping": SOURCE_SPLITS,
            "selection": "lowest SHA-256 rank within split and class",
            "selection_seed": SELECTION_SEED,
            "counts_per_axis": counts,
            "score_position": "last token of raw comment; no chat template",
            "anchor_scrub": {
                "scope": "all eight axis anchor clusters plus axis names",
                "casefolded_word_count": len(anchor_words),
                "morphology_root_count": len(anchor_roots),
                "suffixes": list(MORPH_SUFFIXES),
                "anchor_manifest_sha256": sha256_file(anchors_path),
            },
            "cross_split_normalized_text_deduplication": True,
            "test_use": "untouched until feature and calibration choices are frozen",
        },
        "audit": {
            "eligible_after_scrub_and_dedup": available,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rows": len(rows),
        },
        "rows": rows,
    }


def self_test():
    roots = set()
    for word in ("joy", "crying", "curious", "anger"):
        roots.update(morphology_roots(word))
    assert echo_tokens("They cried after waiting.", roots) == ["cried"]
    assert echo_tokens("A plain technical sentence.", roots) == []
    assert normalized_text("  A\tTest\n") == "a test"
    row = {
        "source_split": "test",
        "label": "joy",
        "source_id": "abc",
        "normalized_text": "content",
    }
    assert selection_rank(row) == selection_rank(dict(row))
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.manifest-self-test.v1"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--anchors", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--train-per-axis", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--calibration-per-axis", type=int, default=DEFAULT_COUNTS["calibration"])
    parser.add_argument("--test-per-axis", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--residual-test-v2", action="store_true")
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--residual-test-per-axis", type=int, default=20)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.data_dir is None or args.anchors is None or args.out is None:
        parser.error("--data-dir, --anchors, and --out are required")
    if args.residual_test_v2:
        if not args.exclude_manifest:
            parser.error("--residual-test-v2 requires at least one --exclude-manifest")
        if args.residual_test_per_axis <= 0:
            parser.error("--residual-test-per-axis must be positive")
        document = freeze_residual_test_manifest(
            args.data_dir, args.anchors, args.exclude_manifest,
            args.residual_test_per_axis,
        )
        counts = {"test": args.residual_test_per_axis}
    else:
        if args.exclude_manifest:
            parser.error("--exclude-manifest is only valid with --residual-test-v2")
        counts = {
            "train": args.train_per_axis,
            "calibration": args.calibration_per_axis,
            "test": args.test_per_axis,
        }
        if any(value <= 0 for value in counts.values()):
            parser.error("all per-axis counts must be positive")
        document = freeze_manifest(args.data_dir, args.anchors, counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(encoded, encoding="utf-8")
    summary = {
        "status": "pass",
        "out": str(args.out),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "rows": len(document["rows"]),
        "counts_per_axis": counts,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
