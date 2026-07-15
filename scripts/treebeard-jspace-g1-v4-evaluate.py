#!/usr/bin/env python3

"""Fit and evaluate the frozen instruction-routing J-Space G1 v4 sensor."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


AXES = ("sadness", "surprise", "joy", "disgust", "fear", "anger", "neutral")
VERBALIZERS = (49166, 12395, 15420, 64797, 8415, 18654, 20002)
PROMPT_PREFIX = (
    "You are an emotion router. Choose exactly one label: sadness, surprise, joy, "
    "disgust, fear, anger, neutral.\nText:\n"
)
PROMPT_SUFFIX = "\nLabel:"
CAL_AUROC_GATE = 0.80
CAL_SOURCE_AUROC_GATE = 0.75
CAL_ECE_GATE = 0.08
CAL_TOP1_GATE = 0.55
CAL_NEUTRAL_FPR_GATE = 0.10
CAL_COVERAGE_GATE = 0.20
CAL_PRECISION_FLOOR = 0.75
TEST_AUROC_GATE = 0.80
TEST_ED_AUROC_GATE = 0.80
TEST_DD_AUROC_GATE = 0.75
TEST_ECE_GATE = 0.10
TEST_TOP1_GATE = 0.55
TEST_NEUTRAL_FPR_GATE = 0.15
TEST_COVERAGE_GATE = 0.15
TEST_PRECISION_GATE = 0.70
TEST_ECHO_AUROC_GATE = 0.75
SEED = 0x4A535034


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


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    begin = 0
    while begin < len(scores):
        end = begin + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end + 1)
        begin = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def axis_aurocs(labels, scores, axes):
    return np.array([
        binary_auroc(labels == axis, scores[:, axis]) for axis in axes
    ])


def equal_mass_ece(labels, probabilities, bins=10):
    groups = np.array_split(np.argsort(probabilities, kind="stable"), bins)
    return float(sum(
        len(group) / len(labels) *
        abs(probabilities[group].mean() - labels[group].mean())
        for group in groups if len(group)
    ))


def fit_platt(scores, labels):
    design = np.column_stack((scores, np.ones(len(scores))))
    positive = int(labels.sum())
    theta = np.array([1.0, math.log((positive + 0.5) /
                                   (len(labels) - positive + 0.5))])
    penalty = np.diag((1e-3, 1e-3))
    for _ in range(100):
        probabilities = sigmoid(design @ theta)
        variance = probabilities * (1.0 - probabilities)
        gradient = design.T @ (probabilities - labels) + penalty @ theta
        hessian = design.T @ (design * variance[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return theta


def choose_abstention(labels, probabilities, precision_floor):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    best = None
    for threshold in np.unique(confidence):
        retained = confidence >= threshold
        precision = float(correct[retained].mean())
        coverage = float(retained.mean())
        if precision >= precision_floor and (best is None or coverage > best[0]):
            best = (coverage, float(threshold), precision)
    if best is None:
        return {"threshold": 1.0, "coverage": 0.0, "precision": None}
    return {"threshold": best[1], "coverage": best[0], "precision": best[2]}


def choose_deadband(labels, probabilities, fpr_gate):
    neutral = labels == AXES.index("neutral")
    affect = probabilities[:, : AXES.index("neutral")].max(axis=1)
    values = np.sort(affect[neutral])
    allowed = math.floor(fpr_gate * len(values))
    if allowed == 0:
        threshold = float(np.nextafter(values[-1], np.inf))
    else:
        boundary = len(values) - allowed
        threshold = float(0.5 * (values[boundary - 1] + values[boundary]))
    return threshold, float((affect[neutral] > threshold).mean())


def validate_capture(manifest_path, manifest_sha, metadata_path, metadata_sha,
                     raw_path, raw_sha, expected_kind):
    require_sha(manifest_path, manifest_sha, "manifest")
    require_sha(metadata_path, metadata_sha, "routing metadata")
    require_sha(raw_path, raw_sha, "routing logits")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "treebeard.jspace.g1.dataset.v4" or \
            manifest.get("policy", {}).get("kind") != expected_kind:
        raise ValueError("unexpected v4 manifest kind")
    if metadata.get("schema") != "treebeard.jspace.g1.routing-logits.v1" or \
            metadata.get("status") != "exact_runtime_routing_verbalizer_logits":
        raise ValueError("unexpected routing capture schema")
    if metadata.get("dataset", {}).get("runner_verified_sha256") != manifest_sha:
        raise ValueError("capture does not attest the manifest")
    if [row["sample_id"] for row in metadata["rows"]] != \
            [row["sample_id"] for row in manifest["rows"]]:
        raise ValueError("capture row order does not match manifest")
    capture = metadata.get("capture", {})
    if capture.get("prompt_prefix") != PROMPT_PREFIX or \
            capture.get("prompt_suffix") != PROMPT_SUFFIX or \
            [row["id"] for row in capture.get("verbalizers", [])] != list(VERBALIZERS):
        raise ValueError("capture routing representation does not match v4")
    shape = tuple(metadata["raw"]["shape"])
    if shape != (len(manifest["rows"]), len(AXES)) or \
            raw_path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError("routing raw shape is invalid")
    scores = np.memmap(raw_path, dtype="<f4", mode="r", shape=shape)
    labels = np.array([AXES.index(row["label"]) for row in manifest["rows"]])
    return manifest, metadata, np.asarray(scores, dtype=np.float64), labels


def subset_metrics(indices, labels, logits, probabilities, deadband, abstention, axes):
    selected_labels = labels[indices]
    selected_logits = logits[indices]
    selected_probabilities = probabilities[indices]
    aurocs = axis_aurocs(selected_labels, selected_logits, axes)
    eces = np.array([
        equal_mass_ece(selected_labels == axis, selected_probabilities[:, axis])
        for axis in axes
    ])
    confidence = selected_probabilities.max(axis=1)
    correct = selected_probabilities.argmax(axis=1) == selected_labels
    retained = confidence >= abstention
    neutral = selected_labels == AXES.index("neutral")
    affect = selected_probabilities[:, : AXES.index("neutral")].max(axis=1)
    return {
        "rows": len(indices),
        "axes": [AXES[axis] for axis in axes],
        "macro_auroc": float(aurocs.mean()),
        "axis_auroc": {AXES[axis]: float(value) for axis, value in zip(axes, aurocs)},
        "macro_ece": float(eces.mean()),
        "top1_accuracy": float(correct.mean()),
        "neutral_fpr": float((affect[neutral] > deadband).mean()) if neutral.any() else None,
        "abstention_coverage": float(retained.mean()),
        "abstention_precision": float(correct[retained].mean()) if retained.any() else None,
    }


def fit_command(args):
    manifest, metadata, scores, labels = validate_capture(
        args.manifest, args.manifest_sha256, args.metadata, args.metadata_sha256,
        args.raw, args.raw_sha256, "source_calibration")
    platt = np.array([fit_platt(scores[:, axis], labels == axis) for axis in range(len(AXES))])
    if np.any(platt[:, 0] <= 0):
        raise ValueError("v4 requires every calibrated verbalizer slope to remain positive")
    logits = scores * platt[:, 0] + platt[:, 1]
    probabilities = sigmoid(logits)
    deadband, realized_fpr = choose_deadband(labels, probabilities, CAL_NEUTRAL_FPR_GATE)
    abstention = choose_abstention(labels, probabilities, CAL_PRECISION_FLOOR)
    all_indices = np.arange(len(labels))
    overall = subset_metrics(all_indices, labels, logits, probabilities,
                             deadband, abstention["threshold"], range(len(AXES)))
    sources = {}
    for source in ("empatheticdialogues", "dailydialog"):
        indices = np.array([i for i, row in enumerate(manifest["rows"])
                            if row["source"] == source])
        axes = range(6) if source == "empatheticdialogues" else range(7)
        sources[source] = subset_metrics(indices, labels, logits, probabilities,
                                         deadband, abstention["threshold"], axes)
    passed = all((
        overall["macro_auroc"] >= CAL_AUROC_GATE,
        min(row["macro_auroc"] for row in sources.values()) >= CAL_SOURCE_AUROC_GATE,
        overall["macro_ece"] <= CAL_ECE_GATE,
        overall["top1_accuracy"] >= CAL_TOP1_GATE,
        realized_fpr <= CAL_NEUTRAL_FPR_GATE,
        abstention["coverage"] >= CAL_COVERAGE_GATE,
        abstention["precision"] is not None and abstention["precision"] >= CAL_PRECISION_FLOOR,
    ))
    artifact_path = args.out_prefix.with_suffix(".npz")
    report_path = args.out_prefix.with_suffix(".fit.json")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(artifact_path, platt=platt, deadband=np.array(deadband),
             abstention=np.array(abstention["threshold"]))
    report = {
        "schema": "treebeard.jspace.g1.fit.v4",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
        },
        "representation": "fixed instruction-probed ordered label-token logits",
        "gates": {
            "macro_auroc_min": CAL_AUROC_GATE,
            "each_source_macro_auroc_min": CAL_SOURCE_AUROC_GATE,
            "macro_ece_max": CAL_ECE_GATE,
            "top1_accuracy_min": CAL_TOP1_GATE,
            "neutral_fpr_max": CAL_NEUTRAL_FPR_GATE,
            "abstention_coverage_min": CAL_COVERAGE_GATE,
            "abstention_precision_min": CAL_PRECISION_FLOOR,
        },
        "overall": overall,
        "sources": sources,
        "deadband": {"threshold": deadband, "realized_neutral_fpr": realized_fpr},
        "abstention": abstention,
        "platt_positive_slopes": bool(np.all(platt[:, 0] > 0)),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "calibration_gate_pass": passed,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "macro_auroc": overall["macro_auroc"],
                      "top1_accuracy": overall["top1_accuracy"], "out": str(report_path)}))
    return 0 if passed else 2


def test_command(args):
    manifest, metadata, scores, labels = validate_capture(
        args.manifest, args.manifest_sha256, args.metadata, args.metadata_sha256,
        args.raw, args.raw_sha256, "source_holdout")
    require_sha(args.artifact, args.artifact_sha256, "v4 artifact")
    artifact = np.load(args.artifact)
    platt = artifact["platt"]
    deadband = float(artifact["deadband"])
    abstention = float(artifact["abstention"])
    logits = scores * platt[:, 0] + platt[:, 1]
    probabilities = sigmoid(logits)
    all_indices = np.arange(len(labels))
    overall = subset_metrics(all_indices, labels, logits, probabilities,
                             deadband, abstention, range(7))
    source_metrics = {}
    for source in ("empatheticdialogues", "dailydialog"):
        indices = np.array([i for i, row in enumerate(manifest["rows"])
                            if row["source"] == source])
        axes = range(6) if source == "empatheticdialogues" else range(7)
        source_metrics[source] = subset_metrics(indices, labels, logits, probabilities,
                                                deadband, abstention, axes)
    echo_metrics = {}
    for echo in (False, True):
        indices = np.array([i for i, row in enumerate(manifest["rows"])
                            if row["source"] == "empatheticdialogues" and
                            row["anchor_echo"] is echo])
        echo_metrics["echo" if echo else "non_echo"] = subset_metrics(
            indices, labels, logits, probabilities, deadband, abstention, range(6))
    rng = np.random.default_rng(SEED)
    shuffled = [float(axis_aurocs(rng.permutation(labels), logits, range(7)).mean())
                for _ in range(32)]
    passed = all((
        overall["macro_auroc"] >= TEST_AUROC_GATE,
        source_metrics["empatheticdialogues"]["macro_auroc"] >= TEST_ED_AUROC_GATE,
        source_metrics["dailydialog"]["macro_auroc"] >= TEST_DD_AUROC_GATE,
        overall["macro_ece"] <= TEST_ECE_GATE,
        overall["top1_accuracy"] >= TEST_TOP1_GATE,
        overall["neutral_fpr"] <= TEST_NEUTRAL_FPR_GATE,
        overall["abstention_coverage"] >= TEST_COVERAGE_GATE,
        overall["abstention_precision"] is not None and
            overall["abstention_precision"] >= TEST_PRECISION_GATE,
        min(row["macro_auroc"] for row in echo_metrics.values()) >= TEST_ECHO_AUROC_GATE,
        0.40 <= float(np.mean(shuffled)) <= 0.60,
    ))
    report = {
        "schema": "treebeard.jspace.g1.test.v4",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact_sha256": args.artifact_sha256,
        },
        "overall": overall,
        "sources": source_metrics,
        "empatheticdialogues_anchor_strata": echo_metrics,
        "label_shuffle_mean_macro_auroc": float(np.mean(shuffled)),
        "test_gate_pass": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "macro_auroc": overall["macro_auroc"],
                      "top1_accuracy": overall["top1_accuracy"], "out": str(args.out)}))
    return 0 if passed else 2


def self_test():
    labels = np.repeat(np.arange(7), 12)
    scores = np.full((len(labels), 7), -1.0)
    scores[np.arange(len(labels)), labels] = 2.0
    platt = np.array([fit_platt(scores[:, axis], labels == axis) for axis in range(7)])
    probabilities = sigmoid(scores * platt[:, 0] + platt[:, 1])
    assert np.all(platt[:, 0] > 0)
    assert np.allclose(axis_aurocs(labels, scores, range(7)), 1.0)
    abstention = choose_abstention(labels, probabilities, 0.75)
    assert abstention["coverage"] == 1.0 and abstention["precision"] == 1.0
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.v4-eval-self-test.v4"}))


def add_capture_args(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-sha256", required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    fit = subparsers.add_parser("fit")
    add_capture_args(fit)
    fit.add_argument("--out-prefix", type=Path, required=True)
    test = subparsers.add_parser("test")
    add_capture_args(test)
    test.add_argument("--artifact", type=Path, required=True)
    test.add_argument("--artifact-sha256", required=True)
    test.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    if args.command == "fit":
        return fit_command(args)
    if args.command == "test":
        return test_command(args)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
