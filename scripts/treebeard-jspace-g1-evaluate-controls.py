#!/usr/bin/env python3

"""Evaluate the frozen G1 sensor on the source-derived linguistic controls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
GATES = {
    "lexical_echo_macro_auroc_min": 0.80,
    "negation_macro_auroc_min": 0.70,
    "quotation_macro_auroc_min": 0.65,
    "third_person_macro_auroc_min": 0.75,
    "mixed_mood_top1_hit_min": 0.50,
    "neutral_flat_fpr_max": 0.10,
}


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="stable")
    values = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and values[end] == values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end + 1)
        begin = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def axis_aurocs(labels, scores):
    return np.array([
        binary_auroc(labels == axis, scores[:, axis])
        for axis in range(len(AXES))
    ])


def equal_mass_ece(labels, probabilities, bins=10):
    groups = np.array_split(np.argsort(probabilities, kind="stable"), bins)
    return float(sum(
        len(group) / len(labels) * abs(probabilities[group].mean() - labels[group].mean())
        for group in groups if len(group)
    ))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def balanced_metrics(indices, labels, scores, probabilities, abstention_threshold):
    subset_labels = labels[indices]
    subset_scores = scores[indices]
    subset_probabilities = probabilities[indices]
    aurocs = axis_aurocs(subset_labels, subset_scores)
    eces = np.array([
        equal_mass_ece(subset_labels == axis, subset_probabilities[:, axis])
        for axis in range(len(AXES))
    ])
    confidence = subset_probabilities.max(axis=1)
    retained = confidence >= abstention_threshold
    correct = subset_probabilities.argmax(axis=1) == subset_labels
    return {
        "rows": len(indices),
        "macro_auroc": float(aurocs.mean()),
        "axis_auroc": dict(zip(AXES, map(float, aurocs))),
        "macro_ece": float(eces.mean()),
        "top1_accuracy": float(correct.mean()),
        "abstention_coverage": float(retained.mean()),
        "abstention_precision": float(correct[retained].mean()) if retained.any() else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--metadata-sha256")
    parser.add_argument("--raw-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert binary_auroc([0, 1, 0, 1], [0.0, 1.0, 0.1, 0.9]) == 1.0
        print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.controls-eval-self-test.v1"}))
        return 0
    if any(value is None for value in (
            args.metadata, args.raw, args.manifest, args.artifact,
            args.metadata_sha256, args.raw_sha256, args.manifest_sha256,
            args.artifact_sha256, args.out)):
        parser.error("all artifact paths and SHA-256 values are required")

    require_sha(args.metadata, args.metadata_sha256, "control metadata")
    require_sha(args.raw, args.raw_sha256, "control activations")
    require_sha(args.manifest, args.manifest_sha256, "control manifest")
    require_sha(args.artifact, args.artifact_sha256, "sensor artifact")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if metadata.get("schema") != "treebeard.jspace.g1.activations.v1" or \
            metadata["dataset"]["schema"] != "treebeard.jspace.g1.controls.v1" or \
            manifest.get("schema") != "treebeard.jspace.g1.controls.v1":
        raise ValueError("unsupported control artifact schema")
    if [row["sample_id"] for row in metadata["rows"]] != \
            [row["sample_id"] for row in manifest["rows"]]:
        raise ValueError("control activation rows do not match the frozen manifest")
    shape = tuple(metadata["raw"]["shape"])
    activations = np.memmap(args.raw, dtype="<f4", mode="r", shape=shape)

    with np.load(args.artifact, allow_pickle=False) as artifact:
        if str(artifact["schema"]) != "treebeard.jspace.g1.sensor.v1":
            raise ValueError("unsupported sensor artifact")
        primary_index = int(artifact["primary_index"])
        weights = np.array(artifact["weights"][primary_index], dtype=np.float64)
        intercepts = np.array(artifact["intercepts"][primary_index], dtype=np.float64)
        means = np.array(artifact["score_means"][primary_index], dtype=np.float64)
        scales = np.array(artifact["score_scales"][primary_index], dtype=np.float64)
        platt = np.array(artifact["platt"], dtype=np.float64)
        deadband_threshold = float(artifact["deadband_threshold"])
        abstention_threshold = float(artifact["abstention_threshold"])
        primary_layer = int(artifact["layers"][primary_index])

    scores = (
        np.asarray(activations[:, primary_index, :], dtype=np.float64) @ weights.T +
        intercepts - means
    ) / scales
    probabilities = sigmoid(scores * platt[:, 0] + platt[:, 1])
    labels = np.array([AXES.index(row["label"]) for row in manifest["rows"]])
    memberships = {
        control: np.array([
            index for index, row in enumerate(manifest["rows"])
            if control in row["control_types"]
        ], dtype=np.int64)
        for control in manifest["audit"]["memberships"]
    }

    balanced = {
        control: balanced_metrics(
            memberships[control], labels, scores, probabilities, abstention_threshold
        )
        for control in ("lexical_echo", "negation", "quotation", "third_person")
    }

    mixed_indices = memberships["mixed_mood"]
    mixed_targets = np.zeros((len(mixed_indices), len(AXES)), dtype=bool)
    for output_index, row_index in enumerate(mixed_indices):
        for label in manifest["rows"][int(row_index)]["labels"]:
            mixed_targets[output_index, AXES.index(label)] = True
    mixed_scores = scores[mixed_indices]
    mixed_probabilities = probabilities[mixed_indices]
    mixed_aurocs = np.array([
        binary_auroc(mixed_targets[:, axis], mixed_scores[:, axis])
        for axis in range(len(AXES))
    ])
    mixed_top1 = mixed_probabilities.argmax(axis=1)
    mixed_hit = mixed_targets[np.arange(len(mixed_indices)), mixed_top1]
    mixed_retained = mixed_probabilities.max(axis=1) >= abstention_threshold
    mixed = {
        "rows": len(mixed_indices),
        "macro_multilabel_auroc": float(mixed_aurocs.mean()),
        "axis_auroc": dict(zip(AXES, map(float, mixed_aurocs))),
        "top1_in_annotated_labels": float(mixed_hit.mean()),
        "abstention_coverage": float(mixed_retained.mean()),
        "retained_top1_hit": float(mixed_hit[mixed_retained].mean()) if mixed_retained.any() else None,
    }

    neutral_indices = memberships["neutral_flat"]
    neutral_affect = probabilities[neutral_indices, :7].max(axis=1)
    neutral_flat = {
        "rows": len(neutral_indices),
        "deadband_false_positive_rate": float((neutral_affect > deadband_threshold).mean()),
        "neutral_top1_rate": float((probabilities[neutral_indices].argmax(axis=1) == 7).mean()),
        "mean_neutral_probability": float(probabilities[neutral_indices, 7].mean()),
    }

    sarcasm_indices = memberships["sarcasm_marker"]
    sarcasm_correct = probabilities[sarcasm_indices].argmax(axis=1) == labels[sarcasm_indices]
    sarcasm_retained = probabilities[sarcasm_indices].max(axis=1) >= abstention_threshold
    sarcasm = {
        "rows": len(sarcasm_indices),
        "label_distribution": {
            axis: int((labels[sarcasm_indices] == index).sum())
            for index, axis in enumerate(AXES)
        },
        "top1_accuracy": float(sarcasm_correct.mean()),
        "abstention_coverage": float(sarcasm_retained.mean()),
        "retained_precision": float(sarcasm_correct[sarcasm_retained].mean())
            if sarcasm_retained.any() else None,
        "interpretation": "report_only_heuristic_marker_not_a_sarcasm_annotation",
    }

    tedium_indices = memberships["narrated_tedium"]
    tedium = {
        "rows": len(tedium_indices),
        "neutral_top1_rate": float((probabilities[tedium_indices].argmax(axis=1) == 7).mean()),
        "mean_neutral_probability": float(probabilities[tedium_indices, 7].mean()),
        "interpretation": "report_only_sparse_explicit_neutral_tedium_anchor_subset",
    }

    echo_indices = memberships["lexical_echo"]
    hashes = np.array([
        hashlib.sha256(manifest["rows"][int(index)]["sample_id"].encode("utf-8")).hexdigest()
        for index in echo_indices
    ])
    random_clusters = np.empty(len(echo_indices), dtype=np.int64)
    random_clusters[np.argsort(hashes)] = np.repeat(np.arange(len(AXES)), len(echo_indices) // len(AXES))
    random_aurocs = axis_aurocs(random_clusters, scores[echo_indices])
    random_cluster = {
        "construction": "frequency-matched SHA-256 clusters independent of source labels",
        "macro_auroc": float(random_aurocs.mean()),
        "axis_auroc": dict(zip(AXES, map(float, random_aurocs))),
    }

    gate_checks = {
        "lexical_echo": balanced["lexical_echo"]["macro_auroc"] >= GATES["lexical_echo_macro_auroc_min"],
        "negation": balanced["negation"]["macro_auroc"] >= GATES["negation_macro_auroc_min"],
        "quotation": balanced["quotation"]["macro_auroc"] >= GATES["quotation_macro_auroc_min"],
        "third_person": balanced["third_person"]["macro_auroc"] >= GATES["third_person_macro_auroc_min"],
        "mixed_mood": mixed["top1_in_annotated_labels"] >= GATES["mixed_mood_top1_hit_min"],
        "neutral_flat": neutral_flat["deadband_false_positive_rate"] <= GATES["neutral_flat_fpr_max"],
        "random_cluster": 0.40 <= random_cluster["macro_auroc"] <= 0.60,
    }
    passed = all(gate_checks.values())
    report = {
        "schema": "treebeard.jspace.g1.controls-eval.v1",
        "status": "pass" if passed else "fail",
        "inputs": {
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "manifest_sha256": args.manifest_sha256,
            "artifact_sha256": args.artifact_sha256,
            "rows": len(manifest["rows"]),
            "primary_layer": primary_layer,
        },
        "structural_anchor_controls": {
            "leave_one_anchor_out": "invariant: sensor fit consumes no anchor identity or anchor-logit feature",
            "split_anchor": "invariant: sensor fit consumes no anchor identity or anchor-logit feature",
        },
        "gates": GATES,
        "gate_checks": gate_checks,
        "balanced_single_label": balanced,
        "mixed_mood": mixed,
        "neutral_flat": neutral_flat,
        "sarcasm_marker": sarcasm,
        "narrated_tedium": tedium,
        "frequency_matched_random_cluster": random_cluster,
        "control_gate_pass": passed,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "out": str(args.out),
        "gate_checks": gate_checks,
    }, separators=(",", ":")))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
