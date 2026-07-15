#!/usr/bin/env python3

"""Fit and evaluate the frozen source-calibrated J-Space G1 v3 sensor."""

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
    "neutral",
)
UNSUPPORTED_AXES = ("curiosity",)
PRIMARY_LAYER = 35
RIDGE = 1.0
POOLING_INDEX = 1
CAL_AUROC_GATE = 0.80
CAL_ECE_GATE = 0.05
CAL_NEUTRAL_FPR_GATE = 0.05
CAL_ABSTENTION_PRECISION = 0.75
CAL_ABSTENTION_COVERAGE_GATE = 0.15
TEST_AUROC_GATE = 0.80
TEST_ECE_GATE = 0.05
TEST_NEUTRAL_FPR_GATE = 0.10
TEST_ABSTENTION_PRECISION_GATE = 0.70
TEST_ABSTENTION_COVERAGE_GATE = 0.15
TEST_LENGTH_BAND_AUROC_GATE = 0.70
LEXICAL_ECHO_AUROC_GATE = 0.70
LENGTH_REGULARIZATION = 0.1
SEED = 0x4A535033


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
        len(group) / len(labels) *
        abs(probabilities[group].mean() - labels[group].mean())
        for group in groups if len(group)
    ))


def fit_directions(features, labels, ridge):
    features = np.asarray(features, dtype=np.float64)
    variance = features.var(axis=0) + 1e-12
    scale = float(np.median(variance[variance > 0]))
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


def sensor_scores(features, weights, intercepts, means, scales):
    raw = np.asarray(features, dtype=np.float64) @ weights.T + intercepts
    return (raw - means) / scales


def fit_length_platt(scores, length_z, labels):
    design = np.column_stack((scores, length_z, np.ones(len(scores))))
    positive = int(np.asarray(labels, dtype=bool).sum())
    theta = np.array([
        1.0,
        0.0,
        math.log((positive + 0.5) / (len(labels) - positive + 0.5)),
    ], dtype=np.float64)
    penalty = np.diag((1e-3, LENGTH_REGULARIZATION, 1e-3))
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


def calibrated_logits(scores, token_counts, length_mean, length_scale, platt):
    length_z = (np.log1p(token_counts) - length_mean) / length_scale
    return scores * platt[:, 0] + length_z[:, None] * platt[:, 1] + platt[:, 2]


def choose_abstention_threshold(labels, probabilities, precision_floor):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    best = None
    for threshold in np.unique(confidence):
        retained = confidence >= threshold
        if not retained.any():
            continue
        precision = float(correct[retained].mean())
        coverage = float(retained.mean())
        if precision >= precision_floor and (best is None or coverage > best[0]):
            best = (coverage, float(threshold), precision)
    if best is None:
        return {"threshold": 1.0, "coverage": 0.0, "precision": None}
    return {"threshold": best[1], "coverage": best[0], "precision": best[2]}


def load_mean_capture(metadata_path: Path, raw_path: Path,
                      metadata_sha: str, raw_sha: str):
    require_sha(metadata_path, metadata_sha, "activation metadata")
    require_sha(raw_path, raw_sha, "activation raw data")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "treebeard.jspace.g1.activations.v1" or \
            metadata.get("status") != "exact_runtime_last_mean_token_residuals":
        raise ValueError("v3 requires an exact last+mean activation capture")
    layers = metadata.get("capture", {}).get("layers", [])
    shape = tuple(metadata["raw"]["shape"])
    expected = (len(metadata["rows"]), len(layers), 2, 2048)
    if shape != expected:
        raise ValueError(f"unexpected activation shape: {shape} != {expected}")
    if raw_path.stat().st_size != int(np.prod(shape, dtype=np.int64)) * 4:
        raise ValueError("activation file size does not match metadata")
    activations = np.memmap(raw_path, dtype="<f4", mode="r", shape=shape)
    return metadata, activations[:, :, POOLING_INDEX, :]


def validate_manifest_capture(metadata, manifest_path: Path,
                              manifest_sha: str, kind: str):
    require_sha(manifest_path, manifest_sha, f"v3 {kind} manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "treebeard.jspace.g1.dataset.v3" or \
            manifest.get("policy", {}).get("kind") != kind:
        raise ValueError(f"unexpected v3 {kind} manifest")
    dataset = metadata.get("dataset", {})
    if dataset.get("schema") != manifest["schema"] or \
            dataset.get("runner_verified_sha256") != manifest_sha:
        raise ValueError("capture does not attest the frozen v3 manifest")
    if [row["sample_id"] for row in metadata["rows"]] != \
            [row["sample_id"] for row in manifest["rows"]]:
        raise ValueError("activation row order does not match the v3 manifest")
    return manifest


def layer_index(metadata):
    layers = np.asarray(metadata["capture"]["layers"], dtype=np.int64)
    matches = np.flatnonzero(layers == PRIMARY_LAYER)
    if len(matches) != 1:
        raise ValueError("frozen layer 35 is absent or duplicated")
    return int(matches[0])


def labels_for_rows(rows):
    try:
        return np.array([AXES.index(row["label"]) for row in rows], dtype=np.int64)
    except ValueError as error:
        raise ValueError("capture contains a label unsupported by v3") from error


def deadband_threshold(labels, probabilities):
    neutral = labels == AXES.index("neutral")
    affect_scores = probabilities[:, : AXES.index("neutral")].max(axis=1)
    neutral_scores = np.sort(affect_scores[neutral])
    allowed = math.floor(CAL_NEUTRAL_FPR_GATE * len(neutral_scores))
    if allowed == 0:
        threshold = float(np.nextafter(neutral_scores[-1], np.inf))
    else:
        boundary = len(neutral_scores) - allowed
        threshold = float(
            0.5 * (neutral_scores[boundary - 1] + neutral_scores[boundary])
        )
    realized = float((affect_scores[neutral] > threshold).mean())
    return threshold, realized, int(neutral.sum()), allowed


def subset_metrics(indices, labels, logits, probabilities,
                   deadband, abstention):
    selected_labels = labels[indices]
    selected_logits = logits[indices]
    selected_probabilities = probabilities[indices]
    aurocs = axis_aurocs(selected_labels, selected_logits)
    eces = np.array([
        equal_mass_ece(selected_labels == axis, selected_probabilities[:, axis])
        for axis in range(len(AXES))
    ])
    confidence = selected_probabilities.max(axis=1)
    retained = confidence >= abstention
    correct = selected_probabilities.argmax(axis=1) == selected_labels
    neutral = selected_labels == AXES.index("neutral")
    affect_scores = selected_probabilities[:, : AXES.index("neutral")].max(axis=1)
    return {
        "rows": len(indices),
        "macro_auroc": float(aurocs.mean()),
        "axis_auroc": dict(zip(AXES, map(float, aurocs))),
        "macro_ece": float(eces.mean()),
        "axis_ece": dict(zip(AXES, map(float, eces))),
        "top1_accuracy": float(correct.mean()),
        "neutral_fpr": float((affect_scores[neutral] > deadband).mean()),
        "abstention_coverage": float(retained.mean()),
        "abstention_precision": (
            float(correct[retained].mean()) if retained.any() else None
        ),
    }


def fit_command(args):
    training_metadata, training_activations = load_mean_capture(
        args.training_metadata, args.training_raw,
        args.training_metadata_sha256, args.training_raw_sha256,
    )
    if training_metadata.get("dataset", {}).get("schema") != \
            "treebeard.jspace.g1.dataset.v1":
        raise ValueError("v3 directions require the frozen v1 training corpus")
    training_rows = training_metadata["rows"]
    training_indices = np.array([
        i for i, row in enumerate(training_rows)
        if row["split"] == "train" and row["label"] in AXES
    ], dtype=np.int64)
    if len(training_indices) != 1568:
        raise ValueError(f"expected 1568 seven-axis training rows, got {len(training_indices)}")
    training_labels = labels_for_rows([training_rows[int(i)] for i in training_indices])
    training_layer = layer_index(training_metadata)
    training_x = np.asarray(
        training_activations[training_indices, training_layer, :], dtype=np.float64
    )
    weights, intercepts = fit_directions(training_x, training_labels, RIDGE)
    raw_training_scores = training_x @ weights.T + intercepts
    score_means = raw_training_scores.mean(axis=0)
    score_scales = raw_training_scores.std(axis=0)
    if np.any(score_scales <= 0):
        raise ValueError("one or more v3 sensor scores have zero training variance")

    calibration_metadata, calibration_activations = load_mean_capture(
        args.calibration_metadata, args.calibration_raw,
        args.calibration_metadata_sha256, args.calibration_raw_sha256,
    )
    manifest = validate_manifest_capture(
        calibration_metadata, args.calibration_manifest,
        args.calibration_manifest_sha256, "calibration",
    )
    if len(manifest["rows"]) != 672 or \
            {row["split"] for row in manifest["rows"]} != {"calibration"}:
        raise ValueError("v3 calibration must contain exactly 672 rows")
    calibration_labels = labels_for_rows(calibration_metadata["rows"])
    counts = np.bincount(calibration_labels, minlength=len(AXES))
    if not np.all(counts == 96):
        raise ValueError("v3 source calibration is not balanced at 96 rows per axis")
    calibration_layer = layer_index(calibration_metadata)
    calibration_scores = sensor_scores(
        calibration_activations[:, calibration_layer, :], weights,
        intercepts, score_means, score_scales,
    )
    token_counts = np.array([
        row["token_count"] for row in calibration_metadata["rows"]
    ], dtype=np.float64)
    log_lengths = np.log1p(token_counts)
    length_mean = float(log_lengths.mean())
    length_scale = float(log_lengths.std())
    if length_scale <= 0:
        raise ValueError("v3 calibration token lengths have zero variance")
    length_z = (log_lengths - length_mean) / length_scale
    platt = np.stack([
        fit_length_platt(
            calibration_scores[:, axis], length_z,
            calibration_labels == axis,
        )
        for axis in range(len(AXES))
    ])
    logits = calibrated_logits(
        calibration_scores, token_counts, length_mean, length_scale, platt
    )
    probabilities = sigmoid(logits)
    calibration_aurocs = axis_aurocs(calibration_labels, logits)
    calibration_eces = np.array([
        equal_mass_ece(calibration_labels == axis, probabilities[:, axis])
        for axis in range(len(AXES))
    ])
    threshold, neutral_fpr, neutral_rows, allowed_false_positives = \
        deadband_threshold(calibration_labels, probabilities)
    abstention = choose_abstention_threshold(
        calibration_labels, probabilities, CAL_ABSTENTION_PRECISION
    )
    length_band_aurocs = {}
    for band in ("short", "medium", "long"):
        indices = np.array([
            i for i, row in enumerate(manifest["rows"])
            if row["length_bin"] == band
        ], dtype=np.int64)
        length_band_aurocs[band] = float(
            axis_aurocs(calibration_labels[indices], logits[indices]).mean()
        )
    rng = np.random.default_rng(SEED)
    shuffled = [
        float(axis_aurocs(rng.permutation(calibration_labels), logits).mean())
        for _ in range(32)
    ]

    macro_auroc = float(calibration_aurocs.mean())
    macro_ece = float(calibration_eces.mean())
    passed = all((
        macro_auroc >= CAL_AUROC_GATE,
        macro_ece <= CAL_ECE_GATE,
        neutral_fpr <= CAL_NEUTRAL_FPR_GATE,
        abstention["coverage"] >= CAL_ABSTENTION_COVERAGE_GATE,
        abstention["precision"] is not None and
            abstention["precision"] >= CAL_ABSTENTION_PRECISION,
        min(length_band_aurocs.values()) >= TEST_LENGTH_BAND_AUROC_GATE,
        0.40 <= float(np.mean(shuffled)) <= 0.60,
    ))

    artifact_path = args.out_prefix.with_suffix(".npz")
    artifact_tmp = Path(str(artifact_path) + ".tmp")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.array("treebeard.jspace.g1.sensor.v3"),
            axes=np.array(AXES),
            unsupported_axes=np.array(UNSUPPORTED_AXES),
            pooling=np.array("mean_all_literal_prompt_tokens"),
            primary_layer=np.array(PRIMARY_LAYER, dtype=np.int32),
            ridge=np.array(RIDGE, dtype="<f8"),
            weights=weights.astype("<f4"),
            intercepts=intercepts.astype("<f4"),
            score_means=score_means.astype("<f4"),
            score_scales=score_scales.astype("<f4"),
            length_log_mean=np.array(length_mean, dtype="<f8"),
            length_log_scale=np.array(length_scale, dtype="<f8"),
            length_platt=platt.astype("<f8"),
            deadband_threshold=np.array(threshold, dtype="<f8"),
            abstention_threshold=np.array(abstention["threshold"], dtype="<f8"),
            training_metadata_sha256=np.array(args.training_metadata_sha256),
            training_raw_sha256=np.array(args.training_raw_sha256),
            calibration_manifest_sha256=np.array(args.calibration_manifest_sha256),
            calibration_metadata_sha256=np.array(args.calibration_metadata_sha256),
            calibration_raw_sha256=np.array(args.calibration_raw_sha256),
        )
    os.replace(artifact_tmp, artifact_path)

    report = {
        "schema": "treebeard.jspace.g1.fit.v3",
        "status": "pass" if passed else "fail",
        "evidence_boundary": (
            "fit has no test-manifest argument and indexes only v1 train plus "
            "the frozen MELD-train source-calibration capture"
        ),
        "inputs": {
            "training_metadata_sha256": args.training_metadata_sha256,
            "training_raw_sha256": args.training_raw_sha256,
            "training_rows": len(training_indices),
            "calibration_manifest_sha256": args.calibration_manifest_sha256,
            "calibration_metadata_sha256": args.calibration_metadata_sha256,
            "calibration_raw_sha256": args.calibration_raw_sha256,
            "calibration_rows": len(calibration_labels),
        },
        "method": {
            "axes": list(AXES),
            "unsupported_axes": list(UNSUPPORTED_AXES),
            "representation": "mean all literal prompt tokens",
            "layer": PRIMARY_LAYER,
            "ridge": RIDGE,
            "directions": "seven-axis diagonal-variance one-vs-rest mean discriminant",
            "calibration": (
                "per-axis logistic calibration on standardized sensor score plus "
                "standardized log1p exact Qwen token count"
            ),
            "length_coefficient_l2": LENGTH_REGULARIZATION,
        },
        "gates": {
            "macro_auroc_min": CAL_AUROC_GATE,
            "macro_ece_max": CAL_ECE_GATE,
            "neutral_fpr_max": CAL_NEUTRAL_FPR_GATE,
            "abstention_coverage_min": CAL_ABSTENTION_COVERAGE_GATE,
            "abstention_precision_min": CAL_ABSTENTION_PRECISION,
            "each_length_band_macro_auroc_min": TEST_LENGTH_BAND_AUROC_GATE,
            "label_shuffle_mean_macro_auroc": [0.40, 0.60],
        },
        "calibration": {
            "macro_auroc": macro_auroc,
            "axis_auroc": dict(zip(AXES, map(float, calibration_aurocs))),
            "macro_ece": macro_ece,
            "axis_ece": dict(zip(AXES, map(float, calibration_eces))),
            "length_band_macro_auroc": length_band_aurocs,
            "length_platt": dict(zip(AXES, platt.tolist())),
        },
        "deadband": {
            "threshold": threshold,
            "neutral_rows": neutral_rows,
            "allowed_false_positives": allowed_false_positives,
            "realized_neutral_fpr": neutral_fpr,
        },
        "abstention": abstention,
        "label_shuffle_control": {
            "replicates": len(shuffled),
            "mean_macro_auroc": float(np.mean(shuffled)),
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": sha256_file(artifact_path),
        },
        "calibration_gate_pass": passed,
    }
    report_path = args.out_prefix.with_suffix(".fit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "artifact": str(artifact_path),
        "macro_auroc": macro_auroc,
        "macro_ece": macro_ece,
    }, separators=(",", ":")))
    return 0 if passed else 2


def load_artifact(path: Path, expected_sha: str):
    require_sha(path, expected_sha, "v3 sensor artifact")
    artifact = np.load(path, allow_pickle=False)
    if str(artifact["schema"]) != "treebeard.jspace.g1.sensor.v3" or \
            tuple(map(str, artifact["axes"])) != AXES or \
            tuple(map(str, artifact["unsupported_axes"])) != UNSUPPORTED_AXES or \
            int(artifact["primary_layer"]) != PRIMARY_LAYER or \
            float(artifact["ridge"]) != RIDGE:
        artifact.close()
        raise ValueError("unsupported or mismatched v3 sensor artifact")
    return artifact


def test_command(args):
    metadata, activations = load_mean_capture(
        args.metadata, args.raw, args.metadata_sha256, args.raw_sha256
    )
    manifest = validate_manifest_capture(
        metadata, args.manifest, args.manifest_sha256, "test"
    )
    if len(manifest["rows"]) != 238 or \
            {row["split"] for row in manifest["rows"]} != {"test"}:
        raise ValueError("v3 test must contain exactly 238 rows")
    labels = labels_for_rows(metadata["rows"])
    artifact = load_artifact(args.artifact, args.artifact_sha256)
    weights = np.array(artifact["weights"], dtype=np.float64)
    intercepts = np.array(artifact["intercepts"], dtype=np.float64)
    means = np.array(artifact["score_means"], dtype=np.float64)
    scales = np.array(artifact["score_scales"], dtype=np.float64)
    length_mean = float(artifact["length_log_mean"])
    length_scale = float(artifact["length_log_scale"])
    platt = np.array(artifact["length_platt"], dtype=np.float64)
    deadband = float(artifact["deadband_threshold"])
    abstention = float(artifact["abstention_threshold"])
    scores = sensor_scores(
        activations[:, layer_index(metadata), :], weights,
        intercepts, means, scales,
    )
    token_counts = np.array([
        row["token_count"] for row in metadata["rows"]
    ], dtype=np.float64)
    logits = calibrated_logits(
        scores, token_counts, length_mean, length_scale, platt
    )
    probabilities = sigmoid(logits)
    artifact.close()

    primary_indices = np.array([
        i for i, row in enumerate(manifest["rows"])
        if row["scope"] == "primary_anchor_free"
    ], dtype=np.int64)
    lexical_indices = np.array([
        i for i, row in enumerate(manifest["rows"])
        if row["scope"] == "lexical_echo"
    ], dtype=np.int64)
    if len(primary_indices) != 210 or len(lexical_indices) != 28:
        raise ValueError("v3 test scope sizes do not match the frozen protocol")
    primary = subset_metrics(
        primary_indices, labels, logits, probabilities, deadband, abstention
    )
    lexical = subset_metrics(
        lexical_indices, labels, logits, probabilities, deadband, abstention
    )
    length_bands = {}
    for band in ("short", "medium", "long"):
        indices = np.array([
            i for i in primary_indices
            if manifest["rows"][int(i)]["length_bin"] == band
        ], dtype=np.int64)
        length_bands[band] = subset_metrics(
            indices, labels, logits, probabilities, deadband, abstention
        )

    rng = np.random.default_rng(SEED + 1)
    by_axis = [
        primary_indices[labels[primary_indices] == axis]
        for axis in range(len(AXES))
    ]
    bootstrap = []
    for _ in range(1000):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in by_axis
        ])
        bootstrap.append(float(axis_aurocs(labels[sample], logits[sample]).mean()))
    shuffled = [
        float(axis_aurocs(rng.permutation(labels[primary_indices]),
                          logits[primary_indices]).mean())
        for _ in range(32)
    ]
    centered_lengths = token_counts[primary_indices] - token_counts[primary_indices].mean()
    length_correlations = {}
    for axis, name in enumerate(AXES):
        centered_logits = logits[primary_indices, axis] - logits[primary_indices, axis].mean()
        length_correlations[name] = float(
            centered_lengths @ centered_logits /
            (np.linalg.norm(centered_lengths) * np.linalg.norm(centered_logits) + 1e-30)
        )

    passed = all((
        primary["macro_auroc"] >= TEST_AUROC_GATE,
        primary["macro_ece"] <= TEST_ECE_GATE,
        primary["neutral_fpr"] <= TEST_NEUTRAL_FPR_GATE,
        primary["abstention_coverage"] >= TEST_ABSTENTION_COVERAGE_GATE,
        primary["abstention_precision"] is not None and
            primary["abstention_precision"] >= TEST_ABSTENTION_PRECISION_GATE,
        min(row["macro_auroc"] for row in length_bands.values()) >=
            TEST_LENGTH_BAND_AUROC_GATE,
        lexical["macro_auroc"] >= LEXICAL_ECHO_AUROC_GATE,
        0.40 <= float(np.mean(shuffled)) <= 0.60,
    ))
    report = {
        "schema": "treebeard.jspace.g1.test.v3",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact_sha256": args.artifact_sha256,
            "rows": len(labels),
        },
        "evidence_boundary": (
            "source-disjoint MELD test; curiosity unsupported and never proxied"
        ),
        "gates": {
            "primary_macro_auroc_min": TEST_AUROC_GATE,
            "primary_macro_ece_max": TEST_ECE_GATE,
            "primary_neutral_fpr_max": TEST_NEUTRAL_FPR_GATE,
            "primary_abstention_coverage_min": TEST_ABSTENTION_COVERAGE_GATE,
            "primary_abstention_precision_min": TEST_ABSTENTION_PRECISION_GATE,
            "each_primary_length_band_macro_auroc_min": TEST_LENGTH_BAND_AUROC_GATE,
            "lexical_echo_macro_auroc_min": LEXICAL_ECHO_AUROC_GATE,
            "label_shuffle_mean_macro_auroc": [0.40, 0.60],
        },
        "primary": primary,
        "primary_length_bands": length_bands,
        "lexical_echo": lexical,
        "macro_auroc_bootstrap": {
            "replicates": len(bootstrap),
            "p025": float(np.quantile(bootstrap, 0.025)),
            "median": float(np.median(bootstrap)),
            "p975": float(np.quantile(bootstrap, 0.975)),
        },
        "token_count_diagnostic": {
            "axis_pearson": length_correlations,
            "maximum_absolute": float(max(map(abs, length_correlations.values()))),
            "interpretation": "report-only; no post-test calibration change is permitted",
        },
        "label_shuffle_control": {
            "replicates": len(shuffled),
            "mean_macro_auroc": float(np.mean(shuffled)),
        },
        "unsupported_axes": list(UNSUPPORTED_AXES),
        "test_gate_pass": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "out": str(args.out),
        "primary_macro_auroc": primary["macro_auroc"],
        "primary_macro_ece": primary["macro_ece"],
        "lexical_macro_auroc": lexical["macro_auroc"],
    }, separators=(",", ":")))
    return 0 if passed else 2


def self_test():
    labels = np.repeat(np.arange(len(AXES)), 8)
    scores = np.full((len(labels), len(AXES)), -1.0)
    scores[np.arange(len(labels)), labels] = 1.0
    assert np.allclose(axis_aurocs(labels, scores), 1.0)
    lengths = np.linspace(-1.0, 1.0, len(labels))
    theta = fit_length_platt(scores[:, 0], lengths, labels == 0)
    assert theta.shape == (3,) and np.all(np.isfinite(theta))
    probabilities = sigmoid(scores)
    abstention = choose_abstention_threshold(labels, probabilities, 0.75)
    assert abstention["coverage"] == 1.0 and abstention["precision"] == 1.0
    rng = np.random.default_rng(7)
    features = rng.normal(size=(70, 16))
    feature_labels = np.repeat(np.arange(len(AXES)), 10)
    features[np.arange(70), feature_labels] += 3.0
    weights, intercepts = fit_directions(features, feature_labels, RIDGE)
    assert weights.shape == (len(AXES), 16) and intercepts.shape == (len(AXES),)
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.v3-eval-self-test.v3"}))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")

    fit = subparsers.add_parser("fit")
    fit.add_argument("--training-metadata", type=Path, required=True)
    fit.add_argument("--training-raw", type=Path, required=True)
    fit.add_argument("--training-metadata-sha256", required=True)
    fit.add_argument("--training-raw-sha256", required=True)
    fit.add_argument("--calibration-manifest", type=Path, required=True)
    fit.add_argument("--calibration-manifest-sha256", required=True)
    fit.add_argument("--calibration-metadata", type=Path, required=True)
    fit.add_argument("--calibration-raw", type=Path, required=True)
    fit.add_argument("--calibration-metadata-sha256", required=True)
    fit.add_argument("--calibration-raw-sha256", required=True)
    fit.add_argument("--out-prefix", type=Path, required=True)

    test = subparsers.add_parser("test")
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--manifest-sha256", required=True)
    test.add_argument("--metadata", type=Path, required=True)
    test.add_argument("--raw", type=Path, required=True)
    test.add_argument("--metadata-sha256", required=True)
    test.add_argument("--raw-sha256", required=True)
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
