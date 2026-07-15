#!/usr/bin/env python3

"""Apply the frozen G1 v5 router to a hash-bound G2 response set."""

import argparse
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np


V5 = runpy.run_path(Path(__file__).with_name("treebeard-jspace-g1-v5-evaluate.py"))
AXES = tuple(V5["AXES"])
VERBALIZERS = tuple(V5["VERBALIZERS"])
PROMPT_PREFIX = V5["PROMPT_PREFIX"]
PROMPT_SUFFIX = V5["PROMPT_SUFFIX"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def load_capture(args):
    for path, expected, label in (
            (args.response_manifest, args.response_manifest_sha256, "response manifest"),
            (args.routing_manifest, args.routing_manifest_sha256, "routing manifest"),
            (args.metadata, args.metadata_sha256, "routing metadata"),
            (args.raw, args.raw_sha256, "routing logits"),
            (args.artifact, args.artifact_sha256, "G1 v5 artifact")):
        require_sha(path, expected, label)

    response = json.loads(args.response_manifest.read_text(encoding="utf-8"))
    routing = json.loads(args.routing_manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if response.get("schema") != "treebeard.jspace.g2.response-set.v1" or \
            response.get("status") != "frozen_before_response_generation":
        raise ValueError("unexpected G2 response manifest")
    if routing.get("schema") != "treebeard.jspace.g1.dataset.v5" or \
            routing.get("policy", {}).get("kind") != "g2_response_routing":
        raise ValueError("unexpected G2 routing manifest")
    if metadata.get("schema") != "treebeard.jspace.g1.routing-logits.v1" or \
            metadata.get("status") != "exact_runtime_routing_verbalizer_logits":
        raise ValueError("unexpected routing capture")
    if metadata.get("dataset", {}).get("runner_verified_sha256") != \
            args.routing_manifest_sha256:
        raise ValueError("routing capture does not attest the routing manifest")

    response_rows = response["rows"]
    routing_rows = routing["rows"]
    metadata_rows = metadata["rows"]
    response_ids = [row["sample_id"] for row in response_rows]
    if response_ids != [row["sample_id"] for row in routing_rows] or \
            response_ids != [row["sample_id"] for row in metadata_rows]:
        raise ValueError("response, routing, and capture row orders differ")
    if [row["text_sha256"] for row in response_rows] != \
            [row["text_sha256"] for row in routing_rows]:
        raise ValueError("response and routing inputs differ")

    capture = metadata.get("capture", {})
    if capture.get("prompt_prefix") != PROMPT_PREFIX or \
            capture.get("prompt_suffix") != PROMPT_SUFFIX or \
            [row["id"] for row in capture.get("verbalizers", [])] != list(VERBALIZERS):
        raise ValueError("capture does not use the frozen G1 v5 representation")
    raw = metadata.get("raw", {})
    shape = tuple(raw.get("shape", ()))
    if raw.get("dtype") != "little_endian_float32" or \
            shape != (len(response_rows), len(AXES)) or \
            args.raw.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError("invalid G2 routing raw array")
    scores = np.memmap(args.raw, dtype="<f4", mode="r", shape=shape)
    return response, metadata, np.asarray(scores, dtype=np.float64)


def route(args):
    response, metadata, scores = load_capture(args)
    artifact = np.load(args.artifact)
    model = {
        "neutral_platt": artifact["neutral_platt"],
        "affect_platt": artifact["affect_platt"],
        "neutral_threshold": float(artifact["neutral_threshold"]),
        "abstention_threshold": float(artifact["abstention_threshold"]),
    }
    probabilities, gate_affect, retained = V5["apply_model"](scores, model)
    affect_routes = probabilities[:, :6].argmax(axis=1)

    rows = []
    for index, source in enumerate(response["rows"]):
        protected = source["scope"] == "protected_task"
        if protected:
            decision = "protected_noop"
            active = False
        elif not gate_affect[index]:
            decision = "neutral_noop"
            active = False
        elif not retained[index]:
            decision = "abstain_noop"
            active = False
        else:
            decision = "actuate"
            active = True
        routed_axis = AXES[int(affect_routes[index])] if gate_affect[index] else "neutral"
        rows.append({
            "sample_id": source["sample_id"],
            "source": source["source"],
            "label": source["label"],
            "scope": source["scope"],
            "probabilities": {
                axis: float(probabilities[index, axis_index])
                for axis_index, axis in enumerate(AXES)
            },
            "neutral_probability": float(probabilities[index, 6]),
            "maximum_probability": float(probabilities[index].max()),
            "top_label": AXES[int(probabilities[index].argmax())],
            "affect_gate": bool(gate_affect[index]),
            "retained": bool(retained[index]),
            "routed_axis": routed_axis,
            "active": active,
            "decision": decision,
        })

    decisions = Counter(row["decision"] for row in rows)
    active_sources = Counter(row["source"] for row in rows if row["active"])
    active_axes = Counter(row["routed_axis"] for row in rows if row["active"])
    report = {
        "schema": "treebeard.jspace.g2.routes.v1",
        "status": "frozen_g1_v5_applied",
        "inputs": {
            "response_manifest_sha256": args.response_manifest_sha256,
            "routing_manifest_sha256": args.routing_manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact_sha256": args.artifact_sha256,
            "captured_model_sha256": metadata["model"]["runner_verified_sha256"],
        },
        "policy": {
            "neutral_threshold": model["neutral_threshold"],
            "abstention_threshold": model["abstention_threshold"],
            "protected_policy": "always_noop",
            "active_policy": "affect_gate_and_retained",
            "affect_route": "argmax_frozen_conditional_affect_probability",
        },
        "audit": {
            "rows": len(rows),
            "active_rows": sum(row["active"] for row in rows),
            "decisions": dict(sorted(decisions.items())),
            "active_sources": dict(sorted(active_sources.items())),
            "active_axes": dict(sorted(active_axes.items())),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "rows": len(rows),
        "active_rows": report["audit"]["active_rows"],
        "out": str(args.out),
    }, separators=(",", ":")))
    return 0


def self_test():
    scores = np.zeros((4, 7), dtype=np.float64)
    scores[0, 2] = 4.0
    scores[1, 6] = 4.0
    scores[2, 5] = 1.0
    scores[3, 0] = 4.0
    model = {
        "neutral_platt": np.array([1.0, 0.0]),
        "affect_platt": np.column_stack((np.ones(6), np.zeros(6))),
        "neutral_threshold": 0.25,
        "abstention_threshold": 0.70,
    }
    probabilities, gate, retained = V5["apply_model"](scores, model)
    if probabilities.shape != (4, 7) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise AssertionError("probability self-test failed")
    if not gate[0] or gate[1] or not retained[0] or not retained[1]:
        raise AssertionError("gate self-test failed")
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g2.route-self-test.v1"}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--response-manifest", type=Path)
    parser.add_argument("--response-manifest-sha256")
    parser.add_argument("--routing-manifest", type=Path)
    parser.add_argument("--routing-manifest-sha256")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--metadata-sha256")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--raw-sha256")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    required = (
        "response_manifest", "response_manifest_sha256", "routing_manifest",
        "routing_manifest_sha256", "metadata", "metadata_sha256", "raw",
        "raw_sha256", "artifact", "artifact_sha256", "out",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing arguments: " + ", ".join(missing))
    return route(args)


if __name__ == "__main__":
    raise SystemExit(main())
