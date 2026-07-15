#!/usr/bin/env python3

"""Freeze the routing-blinded G2b development sample-size extension."""

import argparse
import json
import runpy
from pathlib import Path


G2 = runpy.run_path(Path(__file__).with_name("treebeard-jspace-g2-manifest.py"))
AXES = tuple(G2["AXES"])
EXTENDED_COUNTS = {
    "empatheticdialogues": {axis: 8 for axis in AXES},
    "dailydialog": {
        "sadness": 6,
        "surprise": 6,
        "joy": 6,
        "disgust": 6,
        "fear": 4,
        "anger": 6,
        "neutral": 16,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empathetic-dir", type=Path, required=True)
    parser.add_argument("--dailydialog-json", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--consumed-manifest", action="append", type=Path, required=True)
    parser.add_argument("--base-development", type=Path, required=True)
    parser.add_argument("--frozen-holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--routing-out", type=Path, required=True)
    args = parser.parse_args()

    _, roots = G2["V4"]["anchor_roots"](args.anchors)
    consumed_conversations, consumed_hashes, consumed_documents = G2["load_consumed"](
        args.consumed_manifest)
    source_rows = G2["load_empathetic"](args.empathetic_dir) + G2["load_dailydialog"](
        args.dailydialog_json)
    eligible, rejected = G2["eligible_rows"](
        source_rows, consumed_conversations, consumed_hashes)
    source_files = {
        "empatheticdialogues/train.csv": G2["sha256_file"](
            args.empathetic_dir / "train.csv"),
        "empatheticdialogues/valid.csv": G2["sha256_file"](
            args.empathetic_dir / "valid.csv"),
        "empatheticdialogues/test.csv": G2["sha256_file"](
            args.empathetic_dir / "test.csv"),
        "dailydialog/dialogues.json": G2["sha256_file"](args.dailydialog_json),
    }

    # Keep the original seed and deterministic ranking. Only the frozen counts
    # increase, before any response outcome has been observed.
    G2["COUNTS"]["development"] = EXTENDED_COUNTS
    rows, available = G2["freeze"](eligible, "development", roots)
    response = G2["response_document"](
        "development", rows, available, source_files, args.anchors,
        consumed_documents, rejected)
    response["policy"]["sample_size_extension"] = {
        "schema": "treebeard.jspace.g2b.routing-blinded-extension.v1",
        "reason": "original frozen 50-row set yielded 16 active rows below the frozen minimum of 20",
        "outcome_blinding": (
            "no control, candidate, judge, embedding, evaluation, or holdout output existed"
        ),
        "base_development_path": str(args.base_development),
        "base_development_sha256": G2["sha256_file"](args.base_development),
        "frozen_holdout_path": str(args.frozen_holdout),
        "frozen_holdout_sha256": G2["sha256_file"](args.frozen_holdout),
        "rule": "double every original source-label count where available; use all four eligible DailyDialog fear rows",
        "active_row_gate_unchanged": 20,
    }

    base = json.loads(args.base_development.read_text(encoding="utf-8"))
    holdout = json.loads(args.frozen_holdout.read_text(encoding="utf-8"))
    base_by_id = {row["sample_id"]: row for row in base["rows"]}
    extended_by_id = {row["sample_id"]: row for row in response["rows"]}
    if len(response["rows"]) != 98:
        raise AssertionError("G2b development extension must contain 98 rows")
    if not set(base_by_id) < set(extended_by_id):
        raise AssertionError("original G2 development rows are not a strict subset")
    if any(extended_by_id[sample_id] != row for sample_id, row in base_by_id.items()):
        raise AssertionError("an original G2 development row changed")
    response_hashes = {row["text_sha256"] for row in response["rows"]}
    holdout_hashes = {row["text_sha256"] for row in holdout["rows"]}
    if response_hashes & holdout_hashes:
        raise AssertionError("G2b development and frozen holdout overlap")

    routing = G2["routing_document"](response)
    G2["write_json"](args.out, response)
    G2["write_json"](args.routing_out, routing)
    print(json.dumps({
        "status": "frozen",
        "rows": len(response["rows"]),
        "original_rows": len(base["rows"]),
        "additional_rows": len(response["rows"]) - len(base["rows"]),
        "response_sha256": G2["sha256_file"](args.out),
        "routing_sha256": G2["sha256_file"](args.routing_out),
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
