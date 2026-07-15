#!/usr/bin/env python3

"""Fit and evaluate the frozen J-Space G1 semantic sensor without test leakage."""

import argparse
import hashlib
import json
import math
import os
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
RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
CAL_AUROC_GATE = 0.80
TEST_AUROC_GATE = 0.80
ECE_GATE = 0.05
STABILITY_MEDIAN_GATE = 0.80
STABILITY_P05_GATE = 0.60
PAIR_CORRELATION_GATE = 0.70
TEST_NEUTRAL_FPR_GATE = 0.10
ABSTENTION_CAL_PRECISION = 0.80
ABSTENTION_TEST_PRECISION_GATE = 0.70
ABSTENTION_TEST_COVERAGE_GATE = 0.15
BOOTSTRAP_REPLICATES = 32
TEST_BOOTSTRAP_REPLICATES = 1000
SEED = 0x4A535031


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def load_capture(metadata_path: Path, raw_path: Path, metadata_sha: str, raw_sha: str):
    require_sha(metadata_path, metadata_sha, "activation metadata")
    require_sha(raw_path, raw_sha, "activation raw data")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "treebeard.jspace.g1.activations.v1":
        raise ValueError("unsupported activation metadata schema")
    shape = tuple(metadata["raw"]["shape"])
    if shape != (len(metadata["rows"]), len(metadata["capture"]["layers"]), 2048):
        raise ValueError("unexpected activation shape")
    if raw_path.stat().st_size != int(np.prod(shape, dtype=np.int64)) * 4:
        raise ValueError("activation file size does not match metadata")
    rows = metadata["rows"]
    labels = np.array([AXES.index(row["label"]) for row in rows], dtype=np.int64)
    splits = np.array([row["split"] for row in rows])
    activations = np.memmap(raw_path, dtype="<f4", mode="r", shape=shape)
    return metadata, activations, labels, splits


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


def axis_aurocs(labels, scores):
    return np.array([
        binary_auroc(labels == axis, scores[:, axis])
        for axis in range(len(AXES))
    ])


def equal_mass_ece(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    groups = np.array_split(np.argsort(probabilities, kind="stable"), bins)
    return float(sum(
        len(group) / len(labels) * abs(probabilities[group].mean() - labels[group].mean())
        for group in groups if len(group)
    ))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_platt(scores, labels):
    design = np.column_stack((scores, np.ones(len(scores))))
    positive = int(labels.sum())
    theta = np.array([
        1.0,
        math.log((positive + 0.5) / (len(labels) - positive + 0.5)),
    ])
    regularization = 1e-3
    for _ in range(100):
        probabilities = sigmoid(design @ theta)
        weights = probabilities * (1.0 - probabilities)
        gradient = design.T @ (probabilities - labels) + regularization * theta
        hessian = design.T @ (design * weights[:, None]) + regularization * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return theta


def fit_directions(features, labels, ridge):
    features = np.asarray(features, dtype=np.float64)
    variance = features.var(axis=0) + 1e-12
    positive_variance = variance[variance > 0]
    scale = float(np.median(positive_variance))
    denominator = variance + ridge * scale
    weights = np.empty((len(AXES), features.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(AXES), dtype=np.float64)
    for axis in range(len(AXES)):
        positive_mean = features[labels == axis].mean(axis=0)
        negative_mean = features[labels != axis].mean(axis=0)
        direction = (positive_mean - negative_mean) / denominator
        direction /= np.linalg.norm(direction) + 1e-30
        weights[axis] = direction
        intercepts[axis] = -0.5 * direction @ (positive_mean + negative_mean)
    return weights, intercepts


def score(features, weights, intercepts):
    return np.asarray(features, dtype=np.float64) @ weights.T + intercepts


def standardize_scores(train_scores, other_scores):
    means = train_scores.mean(axis=0)
    scales = train_scores.std(axis=0)
    if np.any(scales <= 0):
        raise ValueError("one or more sensor scores have zero training variance")
    return means, scales, (other_scores - means) / scales


def choose_layer_models(activations, labels, train_indices, calibration_indices, layers):
    records = []
    stored = []
    for layer_index, layer in enumerate(layers):
        train_x = np.asarray(activations[train_indices, layer_index, :], dtype=np.float64)
        calibration_x = np.asarray(activations[calibration_indices, layer_index, :], dtype=np.float64)
        best = None
        ridge_records = []
        for ridge in RIDGES:
            weights, intercepts = fit_directions(train_x, labels[train_indices], ridge)
            train_scores = score(train_x, weights, intercepts)
            calibration_scores = score(calibration_x, weights, intercepts)
            means, scales, standardized = standardize_scores(train_scores, calibration_scores)
            aurocs = axis_aurocs(labels[calibration_indices], standardized)
            ridge_record = {
                "ridge": ridge,
                "macro_auroc": float(aurocs.mean()),
                "axis_auroc": dict(zip(AXES, map(float, aurocs))),
            }
            ridge_records.append(ridge_record)
            candidate = (float(aurocs.mean()), -ridge, weights, intercepts, means, scales, standardized)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        macro, negative_ridge, weights, intercepts, means, scales, standardized = best
        record = {
            "layer": int(layer),
            "type": "full_attention" if (layer + 1) % 4 == 0 else "deltanet",
            "selected_ridge": float(-negative_ridge),
            "calibration_macro_auroc": macro,
            "calibration_axis_auroc": dict(zip(
                AXES, map(float, axis_aurocs(labels[calibration_indices], standardized)))),
            "ridge_sweep": ridge_records,
        }
        records.append(record)
        stored.append((weights, intercepts, means, scales, standardized))
    return records, stored


def stratified_resample(indices, labels, rng):
    pieces = []
    for axis in range(len(AXES)):
        class_indices = indices[labels[indices] == axis]
        pieces.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    return np.concatenate(pieces)


def bootstrap_stability(activations, labels, train_indices, layer_index, ridge, reference_weights, rng):
    cosines = np.empty((BOOTSTRAP_REPLICATES, len(AXES)), dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = stratified_resample(train_indices, labels, rng)
        weights, _ = fit_directions(activations[sampled, layer_index, :], labels[sampled], ridge)
        cosines[replicate] = (
            np.sum(weights * reference_weights, axis=1) /
            (np.linalg.norm(weights, axis=1) * np.linalg.norm(reference_weights, axis=1))
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "axis_median_cosine": dict(zip(AXES, map(float, np.median(cosines, axis=0)))),
        "axis_p05_cosine": dict(zip(AXES, map(float, np.quantile(cosines, 0.05, axis=0)))),
        "median_cosine": float(np.median(cosines)),
        "p05_cosine": float(np.quantile(cosines, 0.05)),
    }


def pearson_by_axis(left, right):
    correlations = []
    for axis in range(len(AXES)):
        x = left[:, axis] - left[:, axis].mean()
        y = right[:, axis] - right[:, axis].mean()
        correlations.append(float(x @ y / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-30)))
    return np.array(correlations)


def choose_abstention_threshold(labels, probabilities):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    best = None
    for threshold in np.unique(confidence):
        retained = confidence >= threshold
        if not retained.any():
            continue
        precision = float(correct[retained].mean())
        coverage = float(retained.mean())
        if precision >= ABSTENTION_CAL_PRECISION and (best is None or coverage > best[0]):
            best = (coverage, float(threshold), precision)
    if best is None:
        return {"threshold": 1.0, "coverage": 0.0, "precision": None}
    return {"threshold": best[1], "coverage": best[0], "precision": best[2]}


def fit_command(args):
    metadata, activations, labels, splits = load_capture(
        args.metadata, args.raw, args.metadata_sha256, args.raw_sha256
    )
    train_indices = np.flatnonzero(splits == "train")
    calibration_indices = np.flatnonzero(splits == "calibration")
    test_indices = np.flatnonzero(splits == "test")
    if len(train_indices) != 1792 or len(calibration_indices) != 256 or len(test_indices) != 256:
        raise ValueError("unexpected frozen split sizes")

    layers = np.array(metadata["capture"]["layers"], dtype=np.int32)
    layer_records, stored = choose_layer_models(
        activations, labels, train_indices, calibration_indices, layers
    )
    primary_index = max(range(len(layers)), key=lambda i: (
        layer_records[i]["calibration_macro_auroc"], -int(layers[i])
    ))
    primary_type = layer_records[primary_index]["type"]
    companion_type = "full_attention" if primary_type == "deltanet" else "deltanet"
    companion_index = max(
        (i for i in range(len(layers)) if layer_records[i]["type"] == companion_type),
        key=lambda i: (layer_records[i]["calibration_macro_auroc"], -int(layers[i])),
    )

    weights = np.stack([item[0] for item in stored])
    intercepts = np.stack([item[1] for item in stored])
    score_means = np.stack([item[2] for item in stored])
    score_scales = np.stack([item[3] for item in stored])
    primary_calibration_scores = stored[primary_index][4]
    platt = np.stack([
        fit_platt(primary_calibration_scores[:, axis], labels[calibration_indices] == axis)
        for axis in range(len(AXES))
    ])
    probabilities = sigmoid(
        primary_calibration_scores * platt[:, 0] + platt[:, 1]
    )
    axis_ece = np.array([
        equal_mass_ece(labels[calibration_indices] == axis, probabilities[:, axis])
        for axis in range(len(AXES))
    ])

    neutral = labels[calibration_indices] == AXES.index("neutral")
    affect_score = probabilities[:, :7].max(axis=1)
    neutral_scores = np.sort(affect_score[neutral])
    allowed_false_positives = math.floor(0.05 * len(neutral_scores))
    if allowed_false_positives == 0:
        deadband_threshold = float(np.nextafter(neutral_scores[-1], np.inf))
    else:
        boundary = len(neutral_scores) - allowed_false_positives
        deadband_threshold = float(0.5 * (neutral_scores[boundary - 1] + neutral_scores[boundary]))
    calibration_neutral_fpr = float((affect_score[neutral] > deadband_threshold).mean())
    abstention = choose_abstention_threshold(labels[calibration_indices], probabilities)

    rng = np.random.default_rng(SEED)
    primary_stability = bootstrap_stability(
        activations, labels, train_indices, primary_index,
        layer_records[primary_index]["selected_ridge"], weights[primary_index], rng
    )
    companion_stability = bootstrap_stability(
        activations, labels, train_indices, companion_index,
        layer_records[companion_index]["selected_ridge"], weights[companion_index], rng
    )
    correlations = pearson_by_axis(stored[primary_index][4], stored[companion_index][4])

    shuffled_aurocs = []
    for _ in range(8):
        shuffled_labels = rng.permutation(labels[train_indices])
        shuffled_weights, shuffled_intercepts = fit_directions(
            activations[train_indices, primary_index, :],
            shuffled_labels,
            layer_records[primary_index]["selected_ridge"],
        )
        shuffled_scores = score(
            activations[calibration_indices, primary_index, :],
            shuffled_weights,
            shuffled_intercepts,
        )
        shuffled_aurocs.append(float(axis_aurocs(labels[calibration_indices], shuffled_scores).mean()))

    calibration_macro_auroc = layer_records[primary_index]["calibration_macro_auroc"]
    calibration_macro_ece = float(axis_ece.mean())
    stability_pass = all((
        primary_stability["median_cosine"] >= STABILITY_MEDIAN_GATE,
        primary_stability["p05_cosine"] >= STABILITY_P05_GATE,
        companion_stability["median_cosine"] >= STABILITY_MEDIAN_GATE,
        companion_stability["p05_cosine"] >= STABILITY_P05_GATE,
    ))
    calibration_pass = all((
        calibration_macro_auroc >= CAL_AUROC_GATE,
        calibration_macro_ece <= ECE_GATE,
        calibration_neutral_fpr <= 0.05,
        stability_pass,
        float(np.median(correlations)) >= PAIR_CORRELATION_GATE,
        0.40 <= float(np.mean(shuffled_aurocs)) <= 0.60,
        abstention["coverage"] > 0,
    ))

    artifact_path = args.out_prefix.with_suffix(".npz")
    artifact_tmp = Path(str(artifact_path) + ".tmp")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.array("treebeard.jspace.g1.sensor.v1"),
            axes=np.array(AXES),
            layers=layers,
            weights=weights.astype("<f4"),
            intercepts=intercepts.astype("<f4"),
            score_means=score_means.astype("<f4"),
            score_scales=score_scales.astype("<f4"),
            selected_ridges=np.array([row["selected_ridge"] for row in layer_records], dtype="<f4"),
            primary_index=np.array(primary_index, dtype=np.int32),
            companion_index=np.array(companion_index, dtype=np.int32),
            platt=platt.astype("<f8"),
            deadband_threshold=np.array(deadband_threshold, dtype="<f8"),
            abstention_threshold=np.array(abstention["threshold"], dtype="<f8"),
            metadata_sha256=np.array(args.metadata_sha256),
            raw_sha256=np.array(args.raw_sha256),
        )
    os.replace(artifact_tmp, artifact_path)

    report = {
        "schema": "treebeard.jspace.g1.fit.v1",
        "status": "pass" if calibration_pass else "fail",
        "evidence_boundary": "train_and_calibration_only_test_partition_unread",
        "inputs": {
            "metadata": str(args.metadata),
            "metadata_sha256": args.metadata_sha256,
            "raw": str(args.raw),
            "raw_sha256": args.raw_sha256,
            "train_rows": len(train_indices),
            "calibration_rows": len(calibration_indices),
            "sealed_test_rows": len(test_indices),
        },
        "method": {
            "family": "diagonal_regularized_one_vs_rest_mean_discriminant",
            "ridge_grid": RIDGES,
            "layer_selection": "maximum calibration macro AUROC, shallow layer tie break",
            "calibration": "per-axis Platt scaling on calibration split",
            "ece": "macro one-vs-rest 10-bin equal-mass ECE",
            "deadband": "conservative empirical 95th percentile of neutral calibration max-affect probability",
            "abstention": f"maximum coverage at calibration precision >= {ABSTENTION_CAL_PRECISION}",
        },
        "gates": {
            "macro_auroc_min": CAL_AUROC_GATE,
            "macro_ece_max": ECE_GATE,
            "bootstrap_median_cosine_min": STABILITY_MEDIAN_GATE,
            "bootstrap_p05_cosine_min": STABILITY_P05_GATE,
            "paired_stratum_score_correlation_median_min": PAIR_CORRELATION_GATE,
            "neutral_calibration_fpr_max": 0.05,
        },
        "layers": layer_records,
        "primary": {
            "layer": int(layers[primary_index]),
            "type": layer_records[primary_index]["type"],
            "calibration_macro_auroc": calibration_macro_auroc,
            "calibration_axis_auroc": layer_records[primary_index]["calibration_axis_auroc"],
            "calibration_macro_ece": calibration_macro_ece,
            "calibration_axis_ece": dict(zip(AXES, map(float, axis_ece))),
            "platt": dict(zip(AXES, platt.tolist())),
            "bootstrap_stability": primary_stability,
        },
        "companion": {
            "layer": int(layers[companion_index]),
            "type": layer_records[companion_index]["type"],
            "calibration_macro_auroc": layer_records[companion_index]["calibration_macro_auroc"],
            "bootstrap_stability": companion_stability,
        },
        "stratum_pair_score_correlation": {
            "axis_pearson": dict(zip(AXES, map(float, correlations))),
            "median": float(np.median(correlations)),
        },
        "deadband": {
            "threshold": deadband_threshold,
            "neutral_rows": int(neutral.sum()),
            "allowed_false_positives": allowed_false_positives,
            "realized_neutral_fpr": calibration_neutral_fpr,
        },
        "abstention": abstention,
        "label_shuffle_control": {
            "replicates": len(shuffled_aurocs),
            "macro_auroc": shuffled_aurocs,
            "mean": float(np.mean(shuffled_aurocs)),
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
        },
        "calibration_gate_pass": calibration_pass,
    }
    report_path = args.out_prefix.with_suffix(".fit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "artifact": str(artifact_path),
        "primary_layer": int(layers[primary_index]),
        "calibration_macro_auroc": calibration_macro_auroc,
        "calibration_macro_ece": calibration_macro_ece,
    }, separators=(",", ":")))
    return 0 if calibration_pass else 2


def test_command(args):
    metadata, activations, labels, splits = load_capture(
        args.metadata, args.raw, args.metadata_sha256, args.raw_sha256
    )
    require_sha(args.artifact, args.artifact_sha256, "sensor artifact")
    with np.load(args.artifact, allow_pickle=False) as artifact:
        if str(artifact["schema"]) != "treebeard.jspace.g1.sensor.v1":
            raise ValueError("unsupported sensor artifact schema")
        if str(artifact["metadata_sha256"]) != args.metadata_sha256 or \
                str(artifact["raw_sha256"]) != args.raw_sha256:
            raise ValueError("sensor artifact input identity mismatch")
        primary_index = int(artifact["primary_index"])
        weights = np.array(artifact["weights"][primary_index], dtype=np.float64)
        intercepts = np.array(artifact["intercepts"][primary_index], dtype=np.float64)
        means = np.array(artifact["score_means"][primary_index], dtype=np.float64)
        scales = np.array(artifact["score_scales"][primary_index], dtype=np.float64)
        platt = np.array(artifact["platt"], dtype=np.float64)
        deadband_threshold = float(artifact["deadband_threshold"])
        abstention_threshold = float(artifact["abstention_threshold"])
        layer = int(artifact["layers"][primary_index])

    # This is the first and only point where the sealed test rows are indexed.
    test_indices = np.flatnonzero(splits == "test")
    if len(test_indices) != 256:
        raise ValueError("unexpected frozen test size")
    test_labels = labels[test_indices]
    test_scores = (score(
        activations[test_indices, primary_index, :], weights, intercepts
    ) - means) / scales
    probabilities = sigmoid(test_scores * platt[:, 0] + platt[:, 1])
    aurocs = axis_aurocs(test_labels, test_scores)
    eces = np.array([
        equal_mass_ece(test_labels == axis, probabilities[:, axis])
        for axis in range(len(AXES))
    ])

    neutral = test_labels == AXES.index("neutral")
    affect_score = probabilities[:, :7].max(axis=1)
    neutral_fpr = float((affect_score[neutral] > deadband_threshold).mean())
    affect_detection = float((affect_score[~neutral] > deadband_threshold).mean())
    confidence = probabilities.max(axis=1)
    retained = confidence >= abstention_threshold
    correct = probabilities.argmax(axis=1) == test_labels
    abstention_coverage = float(retained.mean())
    abstention_precision = float(correct[retained].mean()) if retained.any() else None

    rng = np.random.default_rng(SEED + 1)
    bootstrap = []
    for _ in range(TEST_BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(test_indices), size=len(test_indices))
        try:
            bootstrap.append(float(axis_aurocs(test_labels[sample], test_scores[sample]).mean()))
        except ValueError:
            continue
    shuffled = []
    for _ in range(32):
        shuffled_labels = rng.permutation(test_labels)
        shuffled.append(float(axis_aurocs(shuffled_labels, test_scores).mean()))

    macro_auroc = float(aurocs.mean())
    macro_ece = float(eces.mean())
    test_pass = all((
        macro_auroc >= TEST_AUROC_GATE,
        macro_ece <= ECE_GATE,
        neutral_fpr <= TEST_NEUTRAL_FPR_GATE,
        abstention_coverage >= ABSTENTION_TEST_COVERAGE_GATE,
        abstention_precision is not None and abstention_precision >= ABSTENTION_TEST_PRECISION_GATE,
        0.40 <= float(np.mean(shuffled)) <= 0.60,
    ))
    report = {
        "schema": "treebeard.jspace.g1.test.v1",
        "status": "pass" if test_pass else "fail",
        "inputs": {
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact": str(args.artifact),
            "artifact_sha256": args.artifact_sha256,
            "test_rows": len(test_indices),
        },
        "primary_layer": layer,
        "gates": {
            "macro_auroc_min": TEST_AUROC_GATE,
            "macro_ece_max": ECE_GATE,
            "neutral_fpr_max": TEST_NEUTRAL_FPR_GATE,
            "abstention_coverage_min": ABSTENTION_TEST_COVERAGE_GATE,
            "abstention_precision_min": ABSTENTION_TEST_PRECISION_GATE,
        },
        "macro_auroc": macro_auroc,
        "axis_auroc": dict(zip(AXES, map(float, aurocs))),
        "macro_ece": macro_ece,
        "axis_ece": dict(zip(AXES, map(float, eces))),
        "macro_auroc_bootstrap": {
            "replicates": len(bootstrap),
            "p025": float(np.quantile(bootstrap, 0.025)),
            "median": float(np.median(bootstrap)),
            "p975": float(np.quantile(bootstrap, 0.975)),
        },
        "deadband": {
            "threshold": deadband_threshold,
            "neutral_fpr": neutral_fpr,
            "affect_detection_rate": affect_detection,
        },
        "abstention": {
            "threshold": abstention_threshold,
            "coverage": abstention_coverage,
            "precision": abstention_precision,
        },
        "label_shuffle_control": {
            "replicates": len(shuffled),
            "mean_macro_auroc": float(np.mean(shuffled)),
            "minimum": float(np.min(shuffled)),
            "maximum": float(np.max(shuffled)),
        },
        "test_gate_pass": test_pass,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(args.out),
        "macro_auroc": macro_auroc,
        "macro_ece": macro_ece,
    }, separators=(",", ":")))
    return 0 if test_pass else 2


def add_capture_arguments(parser):
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--raw-sha256", required=True)


def self_test():
    labels = np.array([False, True, False, True])
    assert binary_auroc(labels, np.array([0.0, 1.0, 0.1, 0.9])) == 1.0
    assert binary_auroc(labels, np.array([1.0, 0.0, 0.9, 0.1])) == 0.0
    theta = fit_platt(np.array([-2.0, -1.0, 1.0, 2.0]), labels)
    assert np.isfinite(theta).all()
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.fit-self-test.v1"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    fit_parser = subparsers.add_parser("fit")
    add_capture_arguments(fit_parser)
    fit_parser.add_argument("--out-prefix", type=Path, required=True)
    test_parser = subparsers.add_parser("test")
    add_capture_arguments(test_parser)
    test_parser.add_argument("--artifact", type=Path, required=True)
    test_parser.add_argument("--artifact-sha256", required=True)
    test_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in ("metadata_sha256", "raw_sha256"):
        value = getattr(args, name, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    if args.command == "fit":
        return fit_command(args)
    if args.command == "test":
        if len(args.artifact_sha256) != 64:
            parser.error("--artifact-sha256 must be a SHA-256")
        return test_command(args)
    parser.error("choose fit or test")


if __name__ == "__main__":
    raise SystemExit(main())
