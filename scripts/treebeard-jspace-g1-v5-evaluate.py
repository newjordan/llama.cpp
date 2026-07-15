#!/usr/bin/env python3

"""Fit and evaluate the frozen hierarchical J-Space G1 v5 router."""

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path

import numpy as np


V4 = runpy.run_path(Path(__file__).with_name("treebeard-jspace-g1-v4-evaluate.py"))
AXES = tuple(V4["AXES"])
VERBALIZERS = tuple(V4["VERBALIZERS"])
PROMPT_PREFIX = V4["PROMPT_PREFIX"]
PROMPT_SUFFIX = V4["PROMPT_SUFFIX"]
FOLDS = 8
CV_SEED = "treebeard.jspace.g1.hierarchical-routing.v5.cv"
SHUFFLE_SEED = 0x4A535035

CAL_AUROC_GATE = 0.85
CAL_SOURCE_AUROC_GATE = 0.82
CAL_ECE_GATE = 0.08
CAL_TOP1_GATE = 0.55
CAL_NEUTRAL_FPR_GATE = 0.15
CAL_AFFECT_RECALL_GATE = 0.60
CAL_COVERAGE_GATE = 0.40
CAL_PRECISION_GATE = 0.72
FIT_NEUTRAL_FPR = 0.10
FIT_PRECISION_FLOOR = 0.75

TEST_AUROC_GATE = 0.82
TEST_ED_AUROC_GATE = 0.82
TEST_DD_AUROC_GATE = 0.75
TEST_ECE_GATE = 0.10
TEST_TOP1_GATE = 0.55
TEST_NEUTRAL_FPR_GATE = 0.15
TEST_AFFECT_RECALL_GATE = 0.55
TEST_COVERAGE_GATE = 0.30
TEST_PRECISION_GATE = 0.70
TEST_ECHO_AUROC_GATE = 0.78


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


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def logsumexp(values):
    maximum = values.max(axis=1)
    return maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))


def neutral_contrast(scores):
    return scores[:, 6] - logsumexp(scores[:, :6])


def choose_neutral_threshold(labels, neutral_probability, fpr_gate):
    values = np.sort(neutral_probability[labels == 6])
    if not len(values):
        raise ValueError("neutral threshold requires neutral calibration rows")
    allowed = math.floor(fpr_gate * len(values))
    if allowed == 0:
        threshold = float(np.nextafter(values[0], -np.inf))
    else:
        threshold = float(0.5 * (values[allowed - 1] + values[allowed]))
    realized = float((neutral_probability[labels == 6] < threshold).mean())
    return threshold, realized


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


def validate_capture(manifest_path, manifest_sha, metadata_path, metadata_sha,
                     raw_path, raw_sha, expected_kind):
    require_sha(manifest_path, manifest_sha, "manifest")
    require_sha(metadata_path, metadata_sha, "routing metadata")
    require_sha(raw_path, raw_sha, "routing logits")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_schema = (
        "treebeard.jspace.g1.dataset.v4"
        if expected_kind == "source_calibration"
        else "treebeard.jspace.g1.dataset.v5"
    )
    if manifest.get("schema") != expected_schema or \
            manifest.get("policy", {}).get("kind") != expected_kind:
        raise ValueError("unexpected manifest schema or kind")
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
        raise ValueError("capture routing representation does not match v5")
    shape = tuple(metadata["raw"]["shape"])
    if shape != (len(manifest["rows"]), len(AXES)) or \
            raw_path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError("routing raw shape is invalid")
    scores = np.memmap(raw_path, dtype="<f4", mode="r", shape=shape)
    labels = np.array([AXES.index(row["label"]) for row in manifest["rows"]])
    sources = np.array([row["source"] for row in manifest["rows"]])
    return manifest, metadata, np.asarray(scores, dtype=np.float64), labels, sources


def fit_model(scores, labels, sources):
    daily = sources == "dailydialog"
    neutral_indices = np.flatnonzero(daily & (labels == 6))
    affect_indices = np.flatnonzero(daily & (labels != 6))
    if len(affect_indices) != 6 * len(neutral_indices):
        raise ValueError("v5 class balancing requires six DailyDialog affect rows per neutral row")
    balanced_indices = np.concatenate((affect_indices, np.repeat(neutral_indices, 6)))
    contrast = neutral_contrast(scores)
    neutral_platt = V4["fit_platt"](
        contrast[balanced_indices], labels[balanced_indices] == 6)
    conditional_affect = labels != 6
    affect_platt = np.array([
        V4["fit_platt"](scores[conditional_affect, axis],
                        labels[conditional_affect] == axis)
        for axis in range(6)
    ])
    if neutral_platt[0] <= 0 or np.any(affect_platt[:, 0] <= 0):
        raise ValueError("v5 requires every fitted slope to remain positive")
    provisional = apply_probabilities(scores, neutral_platt, affect_platt)
    neutral_threshold, realized_fpr = choose_neutral_threshold(
        labels, provisional[:, 6], FIT_NEUTRAL_FPR)
    abstention = choose_abstention(labels, provisional, FIT_PRECISION_FLOOR)
    return {
        "neutral_platt": neutral_platt,
        "affect_platt": affect_platt,
        "neutral_threshold": neutral_threshold,
        "calibration_neutral_fpr": realized_fpr,
        "abstention_threshold": abstention["threshold"],
        "calibration_abstention": abstention,
    }


def apply_probabilities(scores, neutral_platt, affect_platt):
    contrast = neutral_contrast(scores)
    p_neutral = V4["sigmoid"](contrast * neutral_platt[0] + neutral_platt[1])
    affect_logits = scores[:, :6] * affect_platt[:, 0] + affect_platt[:, 1]
    p_affect = softmax(affect_logits)
    probabilities = np.column_stack(
        ((1.0 - p_neutral[:, None]) * p_affect, p_neutral))
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise AssertionError("hierarchical probabilities do not sum to one")
    return probabilities


def apply_model(scores, model):
    probabilities = apply_probabilities(
        scores, model["neutral_platt"], model["affect_platt"])
    gate_affect = probabilities[:, 6] < model["neutral_threshold"]
    retained = probabilities.max(axis=1) >= model["abstention_threshold"]
    return probabilities, gate_affect, retained


def subset_metrics(indices, labels, probabilities, gate_affect, retained, axes):
    selected_labels = labels[indices]
    selected_probabilities = probabilities[indices]
    selected_gate = gate_affect[indices]
    selected_retained = retained[indices]
    aurocs = V4["axis_aurocs"](selected_labels, selected_probabilities, axes)
    eces = np.array([
        V4["equal_mass_ece"](
            selected_labels == axis, selected_probabilities[:, axis])
        for axis in axes
    ])
    predictions = selected_probabilities.argmax(axis=1)
    correct = predictions == selected_labels
    neutral = selected_labels == 6
    affect = ~neutral
    routed = selected_probabilities[:, :6].argmax(axis=1)
    routed[~selected_gate] = 6
    return {
        "rows": len(indices),
        "axes": [AXES[axis] for axis in axes],
        "macro_auroc": float(aurocs.mean()),
        "axis_auroc": {AXES[axis]: float(value) for axis, value in zip(axes, aurocs)},
        "macro_ece": float(eces.mean()),
        "top1_accuracy": float(correct.mean()),
        "neutral_fpr": float(selected_gate[neutral].mean()) if neutral.any() else None,
        "affect_recall": float(selected_gate[affect].mean()) if affect.any() else None,
        "routed_accuracy": float((routed == selected_labels).mean()),
        "abstention_coverage": float(selected_retained.mean()),
        "abstention_precision": (
            float(correct[selected_retained].mean()) if selected_retained.any() else None
        ),
    }


def all_metrics(manifest, labels, probabilities, gate_affect, retained):
    overall = subset_metrics(
        np.arange(len(labels)), labels, probabilities, gate_affect, retained, range(7))
    sources = {}
    for source in ("empatheticdialogues", "dailydialog"):
        indices = np.array([i for i, row in enumerate(manifest["rows"])
                            if row["source"] == source])
        axes = range(6) if source == "empatheticdialogues" else range(7)
        sources[source] = subset_metrics(
            indices, labels, probabilities, gate_affect, retained, axes)
    return overall, sources


def fold_ids(rows):
    strata = {}
    result = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        strata.setdefault((row["source"], row["label"]), []).append(index)
    for indices in strata.values():
        ranked = sorted(indices, key=lambda index: hashlib.sha256(
            (CV_SEED + "\0" + rows[index]["sample_id"]).encode("utf-8")
        ).hexdigest())
        for rank, index in enumerate(ranked):
            result[index] = rank % FOLDS
    return result


def cross_validate(manifest, scores, labels, sources):
    folds = fold_ids(manifest["rows"])
    probabilities = np.empty((len(labels), 7), dtype=np.float64)
    gate_affect = np.empty(len(labels), dtype=bool)
    retained = np.empty(len(labels), dtype=bool)
    thresholds = []
    for fold in range(FOLDS):
        train = folds != fold
        test = folds == fold
        model = fit_model(scores[train], labels[train], sources[train])
        fold_probabilities, fold_gate, fold_retained = apply_model(scores[test], model)
        probabilities[test] = fold_probabilities
        gate_affect[test] = fold_gate
        retained[test] = fold_retained
        thresholds.append({
            "fold": fold,
            "neutral": float(model["neutral_threshold"]),
            "abstention": float(model["abstention_threshold"]),
        })
    overall, source_metrics = all_metrics(
        manifest, labels, probabilities, gate_affect, retained)
    return {
        "folds": FOLDS,
        "fold_thresholds": thresholds,
        "overall": overall,
        "sources": source_metrics,
    }


def fit_command(args):
    manifest, _, scores, labels, sources = validate_capture(
        args.manifest, args.manifest_sha256, args.metadata, args.metadata_sha256,
        args.raw, args.raw_sha256, "source_calibration")
    cross_validation = cross_validate(manifest, scores, labels, sources)
    overall = cross_validation["overall"]
    source_metrics = cross_validation["sources"]
    passed = all((
        overall["macro_auroc"] >= CAL_AUROC_GATE,
        min(row["macro_auroc"] for row in source_metrics.values()) >=
            CAL_SOURCE_AUROC_GATE,
        overall["macro_ece"] <= CAL_ECE_GATE,
        overall["top1_accuracy"] >= CAL_TOP1_GATE,
        overall["neutral_fpr"] <= CAL_NEUTRAL_FPR_GATE,
        overall["affect_recall"] >= CAL_AFFECT_RECALL_GATE,
        overall["abstention_coverage"] >= CAL_COVERAGE_GATE,
        overall["abstention_precision"] is not None and
            overall["abstention_precision"] >= CAL_PRECISION_GATE,
    ))
    model = fit_model(scores, labels, sources)
    probabilities, gate_affect, retained = apply_model(scores, model)
    refit_overall, refit_sources = all_metrics(
        manifest, labels, probabilities, gate_affect, retained)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = args.out_prefix.with_suffix(".npz")
    report_path = args.out_prefix.with_suffix(".fit.json")
    np.savez(
        artifact_path,
        neutral_platt=model["neutral_platt"],
        affect_platt=model["affect_platt"],
        neutral_threshold=np.array(model["neutral_threshold"]),
        abstention_threshold=np.array(model["abstention_threshold"]),
    )
    report = {
        "schema": "treebeard.jspace.g1.fit.v5",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
        },
        "representation": "hierarchical joint-logit neutral gate plus conditional affect route",
        "gates": {
            "oof_macro_auroc_min": CAL_AUROC_GATE,
            "oof_each_source_macro_auroc_min": CAL_SOURCE_AUROC_GATE,
            "oof_macro_ece_max": CAL_ECE_GATE,
            "oof_top1_accuracy_min": CAL_TOP1_GATE,
            "oof_neutral_fpr_max": CAL_NEUTRAL_FPR_GATE,
            "oof_affect_recall_min": CAL_AFFECT_RECALL_GATE,
            "oof_abstention_coverage_min": CAL_COVERAGE_GATE,
            "oof_abstention_precision_min": CAL_PRECISION_GATE,
        },
        "cross_validation": cross_validation,
        "refit": {
            "overall": refit_overall,
            "sources": refit_sources,
            "neutral_threshold": model["neutral_threshold"],
            "calibration_neutral_fpr": model["calibration_neutral_fpr"],
            "abstention": model["calibration_abstention"],
            "neutral_platt_positive_slope": bool(model["neutral_platt"][0] > 0),
            "affect_platt_positive_slopes": bool(np.all(model["affect_platt"][:, 0] > 0)),
        },
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "calibration_gate_pass": passed,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "oof_macro_auroc": overall["macro_auroc"],
        "oof_top1_accuracy": overall["top1_accuracy"],
        "oof_neutral_fpr": overall["neutral_fpr"],
        "oof_affect_recall": overall["affect_recall"],
        "out": str(report_path),
    }))
    return 0 if passed else 2


def test_command(args):
    manifest, _, scores, labels, _ = validate_capture(
        args.manifest, args.manifest_sha256, args.metadata, args.metadata_sha256,
        args.raw, args.raw_sha256, "fresh_validation_holdout")
    require_sha(args.artifact, args.artifact_sha256, "v5 artifact")
    artifact = np.load(args.artifact)
    model = {
        "neutral_platt": artifact["neutral_platt"],
        "affect_platt": artifact["affect_platt"],
        "neutral_threshold": float(artifact["neutral_threshold"]),
        "abstention_threshold": float(artifact["abstention_threshold"]),
    }
    probabilities, gate_affect, retained = apply_model(scores, model)
    overall, source_metrics = all_metrics(
        manifest, labels, probabilities, gate_affect, retained)
    dd_indices = np.array([i for i, row in enumerate(manifest["rows"])
                           if row["source"] == "dailydialog"])
    dd_gated = subset_metrics(
        dd_indices, labels, probabilities, gate_affect, retained, (0, 1, 2, 4, 5, 6))
    echo_metrics = {}
    for echo in (False, True):
        indices = np.array([i for i, row in enumerate(manifest["rows"])
                            if row["source"] == "empatheticdialogues" and
                            row["anchor_echo"] is echo])
        echo_metrics["echo" if echo else "non_echo"] = subset_metrics(
            indices, labels, probabilities, gate_affect, retained, range(6))
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = [float(V4["axis_aurocs"](
        rng.permutation(labels), probabilities, range(7)).mean()) for _ in range(32)]
    passed = all((
        overall["macro_auroc"] >= TEST_AUROC_GATE,
        source_metrics["empatheticdialogues"]["macro_auroc"] >= TEST_ED_AUROC_GATE,
        dd_gated["macro_auroc"] >= TEST_DD_AUROC_GATE,
        overall["macro_ece"] <= TEST_ECE_GATE,
        overall["top1_accuracy"] >= TEST_TOP1_GATE,
        overall["neutral_fpr"] <= TEST_NEUTRAL_FPR_GATE,
        overall["affect_recall"] >= TEST_AFFECT_RECALL_GATE,
        overall["abstention_coverage"] >= TEST_COVERAGE_GATE,
        overall["abstention_precision"] is not None and
            overall["abstention_precision"] >= TEST_PRECISION_GATE,
        min(row["macro_auroc"] for row in echo_metrics.values()) >=
            TEST_ECHO_AUROC_GATE,
        0.40 <= float(np.mean(shuffled)) <= 0.60,
    ))
    report = {
        "schema": "treebeard.jspace.g1.test.v5",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact_sha256": args.artifact_sha256,
        },
        "overall": overall,
        "sources": source_metrics,
        "dailydialog_gated_axes_excluding_single_disgust": dd_gated,
        "empatheticdialogues_anchor_strata": echo_metrics,
        "label_shuffle_mean_macro_auroc": float(np.mean(shuffled)),
        "test_gate_pass": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "macro_auroc": overall["macro_auroc"],
        "top1_accuracy": overall["top1_accuracy"],
        "neutral_fpr": overall["neutral_fpr"],
        "affect_recall": overall["affect_recall"],
        "out": str(args.out),
    }))
    return 0 if passed else 2


def self_test():
    labels = np.repeat(np.arange(7), 16)
    scores = np.zeros((len(labels), 7), dtype=np.float64)
    scores[np.arange(len(labels)), labels] = 3.0
    sources = np.repeat("dailydialog", len(labels))
    model = fit_model(scores, labels, sources)
    probabilities, gate_affect, retained = apply_model(scores, model)
    assert np.all(probabilities.argmax(axis=1) == labels)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert float(gate_affect[labels == 6].mean()) <= FIT_NEUTRAL_FPR
    assert float(gate_affect[labels != 6].mean()) == 1.0
    assert retained.all()
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.v5-eval-self-test.v5"}))


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
