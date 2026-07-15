#!/usr/bin/env python3

"""Fit and evaluate the preregistered mean-pooled J-Space G1 v2 sensor."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np


def load_core():
    path = Path(__file__).with_name("treebeard-jspace-g1-fit.py")
    spec = importlib.util.spec_from_file_location("treebeard_jspace_g1_fit_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_core()
AXES = CORE.AXES
PRIMARY_LAYER = 35
RIDGE = 1.0
POOLING_INDEX = 1
CAL_AUROC_GATE = 0.80
TEST_AUROC_GATE = 0.80
ECE_GATE = 0.05
TEST_NEUTRAL_FPR_GATE = 0.10
ABSTENTION_PRECISION_GATE = 0.70
ABSTENTION_COVERAGE_GATE = 0.15
CONTROL_GATES = {
    "direct_macro_auroc_min": 0.80,
    "quotation_macro_auroc_min": 0.70,
    "third_person_macro_auroc_min": 0.75,
    "negation_pair_suppression_min": 0.70,
    "neutral_flat_fpr_max": 0.10,
}
SEED = 0x4A535032


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


def load_mean_capture(metadata_path: Path, raw_path: Path, metadata_sha: str, raw_sha: str):
    require_sha(metadata_path, metadata_sha, "activation metadata")
    require_sha(raw_path, raw_sha, "activation raw data")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "treebeard.jspace.g1.activations.v1":
        raise ValueError("unsupported activation metadata schema")
    if metadata.get("status") != "exact_runtime_last_mean_token_residuals" or \
            metadata.get("capture", {}).get("pooling") != ["last", "mean"]:
        raise ValueError("v2 requires an exact last+mean runtime capture")
    shape = tuple(metadata["raw"]["shape"])
    expected = (len(metadata["rows"]), len(metadata["capture"]["layers"]), 2, 2048)
    if shape != expected:
        raise ValueError(f"unexpected activation shape: {shape} != {expected}")
    if raw_path.stat().st_size != int(np.prod(shape, dtype=np.int64)) * 4:
        raise ValueError("activation file size does not match metadata")
    raw = np.memmap(raw_path, dtype="<f4", mode="r", shape=shape)
    labels = np.array([AXES.index(row["label"]) for row in metadata["rows"]], dtype=np.int64)
    splits = np.array([row["split"] for row in metadata["rows"]])
    return metadata, raw[:, :, POOLING_INDEX, :], labels, splits


def load_artifact(path: Path, expected_sha: str):
    require_sha(path, expected_sha, "sensor artifact")
    artifact = np.load(path, allow_pickle=False)
    if str(artifact["schema"]) != "treebeard.jspace.g1.sensor.v2":
        artifact.close()
        raise ValueError("unsupported v2 sensor artifact")
    if int(artifact["primary_layer"]) != PRIMARY_LAYER or float(artifact["ridge"]) != RIDGE:
        artifact.close()
        raise ValueError("sensor artifact does not match the preregistered model")
    return artifact


def parameters(artifact):
    return (
        np.array(artifact["weights"], dtype=np.float64),
        np.array(artifact["intercepts"], dtype=np.float64),
        np.array(artifact["score_means"], dtype=np.float64),
        np.array(artifact["score_scales"], dtype=np.float64),
        np.array(artifact["platt"], dtype=np.float64),
        float(artifact["deadband_threshold"]),
        float(artifact["abstention_threshold"]),
    )


def model_scores(features, artifact):
    weights, intercepts, means, scales, platt, _, _ = parameters(artifact)
    scores = (CORE.score(features, weights, intercepts) - means) / scales
    probabilities = CORE.sigmoid(scores * platt[:, 0] + platt[:, 1])
    return scores, probabilities


def fit_command(args):
    metadata, activations, labels, splits = load_mean_capture(
        args.metadata, args.raw, args.metadata_sha256, args.raw_sha256
    )
    if metadata["dataset"]["schema"] != "treebeard.jspace.g1.dataset.v1":
        raise ValueError("v2 fit requires the frozen v1 primary corpus")
    train_indices = np.flatnonzero(splits == "train")
    calibration_indices = np.flatnonzero(splits == "calibration")
    sealed_v1_test_indices = np.flatnonzero(splits == "test")
    if (len(train_indices), len(calibration_indices), len(sealed_v1_test_indices)) != (1792, 256, 256):
        raise ValueError("unexpected frozen v1 corpus shape")
    layers = np.array(metadata["capture"]["layers"], dtype=np.int32)
    matches = np.flatnonzero(layers == PRIMARY_LAYER)
    if len(matches) != 1:
        raise ValueError("preregistered layer 35 is absent or duplicated")
    layer_index = int(matches[0])

    train_x = np.asarray(activations[train_indices, layer_index, :], dtype=np.float64)
    calibration_x = np.asarray(
        activations[calibration_indices, layer_index, :], dtype=np.float64
    )
    weights, intercepts = CORE.fit_directions(train_x, labels[train_indices], RIDGE)
    train_scores = CORE.score(train_x, weights, intercepts)
    raw_calibration_scores = CORE.score(calibration_x, weights, intercepts)
    score_means, score_scales, calibration_scores = CORE.standardize_scores(
        train_scores, raw_calibration_scores
    )
    calibration_aurocs = CORE.axis_aurocs(labels[calibration_indices], calibration_scores)
    platt = np.stack([
        CORE.fit_platt(
            calibration_scores[:, axis], labels[calibration_indices] == axis
        )
        for axis in range(len(AXES))
    ])
    probabilities = CORE.sigmoid(calibration_scores * platt[:, 0] + platt[:, 1])
    calibration_eces = np.array([
        CORE.equal_mass_ece(
            labels[calibration_indices] == axis, probabilities[:, axis]
        )
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
        deadband_threshold = float(
            0.5 * (neutral_scores[boundary - 1] + neutral_scores[boundary])
        )
    neutral_fpr = float((affect_score[neutral] > deadband_threshold).mean())
    abstention = CORE.choose_abstention_threshold(labels[calibration_indices], probabilities)

    rng = np.random.default_rng(SEED)
    stability = CORE.bootstrap_stability(
        activations, labels, train_indices, layer_index, RIDGE, weights, rng
    )
    shuffled_aurocs = []
    for _ in range(8):
        shuffled_labels = rng.permutation(labels[train_indices])
        shuffled_weights, shuffled_intercepts = CORE.fit_directions(
            train_x, shuffled_labels, RIDGE
        )
        shuffled_aurocs.append(float(CORE.axis_aurocs(
            labels[calibration_indices],
            CORE.score(calibration_x, shuffled_weights, shuffled_intercepts),
        ).mean()))

    macro_auroc = float(calibration_aurocs.mean())
    macro_ece = float(calibration_eces.mean())
    passed = all((
        macro_auroc >= CAL_AUROC_GATE,
        macro_ece <= ECE_GATE,
        neutral_fpr <= 0.05,
        stability["median_cosine"] >= CORE.STABILITY_MEDIAN_GATE,
        stability["p05_cosine"] >= CORE.STABILITY_P05_GATE,
        0.40 <= float(np.mean(shuffled_aurocs)) <= 0.60,
        abstention["coverage"] > 0,
    ))

    artifact_path = args.out_prefix.with_suffix(".npz")
    artifact_tmp = Path(str(artifact_path) + ".tmp")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.array("treebeard.jspace.g1.sensor.v2"),
            axes=np.array(AXES),
            pooling=np.array("mean_all_literal_prompt_tokens"),
            primary_layer=np.array(PRIMARY_LAYER, dtype=np.int32),
            layer_index=np.array(layer_index, dtype=np.int32),
            ridge=np.array(RIDGE, dtype="<f8"),
            weights=weights.astype("<f4"),
            intercepts=intercepts.astype("<f4"),
            score_means=score_means.astype("<f4"),
            score_scales=score_scales.astype("<f4"),
            platt=platt.astype("<f8"),
            deadband_threshold=np.array(deadband_threshold, dtype="<f8"),
            abstention_threshold=np.array(abstention["threshold"], dtype="<f8"),
            training_metadata_sha256=np.array(args.metadata_sha256),
            training_raw_sha256=np.array(args.raw_sha256),
        )
    os.replace(artifact_tmp, artifact_path)

    report = {
        "schema": "treebeard.jspace.g1.fit.v2",
        "status": "pass" if passed else "fail",
        "evidence_boundary": (
            "fit indexes only v1 train/calibration; no v1 or v2 test activation is indexed"
        ),
        "inputs": {
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "train_rows": len(train_indices),
            "calibration_rows": len(calibration_indices),
            "unindexed_v1_test_rows": len(sealed_v1_test_indices),
        },
        "method": {
            "representation": "arithmetic mean across every literal prompt token",
            "layer": PRIMARY_LAYER,
            "ridge": RIDGE,
            "family": "diagonal_regularized_one_vs_rest_mean_discriminant",
            "calibration": "per-axis Platt scaling on v1 calibration",
        },
        "calibration": {
            "macro_auroc": macro_auroc,
            "axis_auroc": dict(zip(AXES, map(float, calibration_aurocs))),
            "macro_ece": macro_ece,
            "axis_ece": dict(zip(AXES, map(float, calibration_eces))),
            "neutral_fpr": neutral_fpr,
        },
        "bootstrap_stability": stability,
        "abstention": abstention,
        "label_shuffle_control": {
            "macro_auroc": shuffled_aurocs,
            "mean": float(np.mean(shuffled_aurocs)),
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
        "calibration_macro_auroc": macro_auroc,
        "calibration_macro_ece": macro_ece,
    }, separators=(",", ":")))
    return 0 if passed else 2


def validate_evaluation_manifest(metadata, manifest_path, manifest_sha, schema):
    require_sha(manifest_path, manifest_sha, "evaluation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema or metadata["dataset"]["schema"] != schema:
        raise ValueError("evaluation manifest schema mismatch")
    if metadata["dataset"]["runner_verified_sha256"] != manifest_sha:
        raise ValueError("capture does not attest the requested manifest")
    if [row["sample_id"] for row in manifest["rows"]] != \
            [row["sample_id"] for row in metadata["rows"]]:
        raise ValueError("capture row order does not match the frozen manifest")
    return manifest


def test_command(args):
    metadata, activations, labels, splits = load_mean_capture(
        args.metadata, args.raw, args.metadata_sha256, args.raw_sha256
    )
    manifest = validate_evaluation_manifest(
        metadata, args.manifest, args.manifest_sha256,
        "treebeard.jspace.g1.dataset.v2",
    )
    if len(labels) != 160 or set(splits) != {"test"}:
        raise ValueError("v2 primary test must contain exactly 160 test rows")
    counts = np.bincount(labels, minlength=len(AXES))
    if not np.all(counts == 20):
        raise ValueError("v2 primary test is not balanced at 20 rows per axis")

    artifact = load_artifact(args.artifact, args.artifact_sha256)
    layer_index = int(artifact["layer_index"])
    scores, probabilities = model_scores(activations[:, layer_index, :], artifact)
    _, _, _, _, _, deadband_threshold, abstention_threshold = parameters(artifact)
    artifact.close()

    aurocs = CORE.axis_aurocs(labels, scores)
    eces = np.array([
        CORE.equal_mass_ece(labels == axis, probabilities[:, axis])
        for axis in range(len(AXES))
    ])
    neutral = labels == AXES.index("neutral")
    affect_score = probabilities[:, :7].max(axis=1)
    neutral_fpr = float((affect_score[neutral] > deadband_threshold).mean())
    confidence = probabilities.max(axis=1)
    retained = confidence >= abstention_threshold
    correct = probabilities.argmax(axis=1) == labels
    coverage = float(retained.mean())
    precision = float(correct[retained].mean()) if retained.any() else None

    rng = np.random.default_rng(SEED + 1)
    bootstrap = []
    by_axis = [np.flatnonzero(labels == axis) for axis in range(len(AXES))]
    for _ in range(1000):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in by_axis
        ])
        bootstrap.append(float(CORE.axis_aurocs(labels[sample], scores[sample]).mean()))
    shuffled = [
        float(CORE.axis_aurocs(rng.permutation(labels), scores).mean())
        for _ in range(32)
    ]
    token_counts = np.array([row["token_count"] for row in metadata["rows"]], dtype=np.float64)
    centered_tokens = token_counts - token_counts.mean()
    length_correlations = []
    for axis in range(len(AXES)):
        centered_scores = scores[:, axis] - scores[:, axis].mean()
        length_correlations.append(float(
            centered_tokens @ centered_scores /
            (np.linalg.norm(centered_tokens) * np.linalg.norm(centered_scores) + 1e-30)
        ))

    macro_auroc = float(aurocs.mean())
    macro_ece = float(eces.mean())
    passed = all((
        macro_auroc >= TEST_AUROC_GATE,
        macro_ece <= ECE_GATE,
        neutral_fpr <= TEST_NEUTRAL_FPR_GATE,
        coverage >= ABSTENTION_COVERAGE_GATE,
        precision is not None and precision >= ABSTENTION_PRECISION_GATE,
        0.40 <= float(np.mean(shuffled)) <= 0.60,
    ))
    report = {
        "schema": "treebeard.jspace.g1.test.v2",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact_sha256": args.artifact_sha256,
            "rows": len(labels),
            "source_split_counts": manifest["audit"]["selected_source_splits"],
        },
        "gates": {
            "macro_auroc_min": TEST_AUROC_GATE,
            "macro_ece_max": ECE_GATE,
            "neutral_fpr_max": TEST_NEUTRAL_FPR_GATE,
            "abstention_coverage_min": ABSTENTION_COVERAGE_GATE,
            "abstention_precision_min": ABSTENTION_PRECISION_GATE,
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
        "deadband": {"threshold": deadband_threshold, "neutral_fpr": neutral_fpr},
        "abstention": {
            "threshold": abstention_threshold,
            "coverage": coverage,
            "precision": precision,
        },
        "token_count_diagnostic": {
            "axis_pearson": dict(zip(AXES, length_correlations)),
            "maximum_absolute": float(np.max(np.abs(length_correlations))),
            "interpretation": "report-only; no post-test representation change is permitted",
        },
        "label_shuffle_control": {"mean_macro_auroc": float(np.mean(shuffled))},
        "test_gate_pass": passed,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "out": str(args.out),
        "macro_auroc": macro_auroc,
        "macro_ece": macro_ece,
    }, separators=(",", ":")))
    return 0 if passed else 2


def seven_axis_aurocs(indices, manifest, scores):
    targets = np.array([AXES.index(manifest["rows"][int(i)]["target_axis"]) for i in indices])
    return np.array([
        CORE.binary_auroc(targets == axis, scores[indices, axis]) for axis in range(7)
    ])


def controls_command(args):
    metadata, activations, _, splits = load_mean_capture(
        args.metadata, args.raw, args.metadata_sha256, args.raw_sha256
    )
    manifest = validate_evaluation_manifest(
        metadata, args.manifest, args.manifest_sha256,
        "treebeard.jspace.g1.controls.v2",
    )
    if len(splits) != 128 or set(splits) != {"control"}:
        raise ValueError("v2 synthetic controls must contain exactly 128 control rows")
    artifact = load_artifact(args.artifact, args.artifact_sha256)
    layer_index = int(artifact["layer_index"])
    scores, probabilities = model_scores(activations[:, layer_index, :], artifact)
    _, _, _, _, _, deadband_threshold, _ = parameters(artifact)
    artifact.close()

    memberships = {
        control: np.array([
            i for i, row in enumerate(manifest["rows"])
            if row["control_types"] == [control]
        ], dtype=np.int64)
        for control in ("direct", "negation", "quotation", "third_person", "neutral_flat")
    }
    balanced = {}
    for control in ("direct", "quotation", "third_person"):
        aurocs = seven_axis_aurocs(memberships[control], manifest, scores)
        targets = np.array([
            AXES.index(manifest["rows"][int(i)]["target_axis"])
            for i in memberships[control]
        ])
        balanced[control] = {
            "rows": len(targets),
            "macro_auroc": float(aurocs.mean()),
            "axis_auroc": dict(zip(AXES[:7], map(float, aurocs))),
            "top1_accuracy": float((probabilities[memberships[control]].argmax(axis=1) == targets).mean()),
        }

    direct_by_pair = {
        manifest["rows"][int(i)]["pair_id"]: int(i) for i in memberships["direct"]
    }
    negated_deltas = []
    negated_neutral = []
    for index in memberships["negation"]:
        row = manifest["rows"][int(index)]
        direct_index = direct_by_pair[row["pair_id"]]
        target = AXES.index(row["target_axis"])
        negated_deltas.append(float(scores[direct_index, target] - scores[index, target]))
        negated_neutral.append(int(probabilities[index].argmax() == AXES.index("neutral")))
    negated_deltas = np.array(negated_deltas)
    negation = {
        "rows": len(negated_deltas),
        "paired_target_score_suppression_rate": float((negated_deltas > 0).mean()),
        "paired_target_score_delta_median": float(np.median(negated_deltas)),
        "neutral_top1_rate": float(np.mean(negated_neutral)),
    }
    neutral_affect = probabilities[memberships["neutral_flat"], :7].max(axis=1)
    neutral_flat = {
        "rows": len(neutral_affect),
        "deadband_false_positive_rate": float((neutral_affect > deadband_threshold).mean()),
        "neutral_top1_rate": float((
            probabilities[memberships["neutral_flat"]].argmax(axis=1) == AXES.index("neutral")
        ).mean()),
    }

    rng = np.random.default_rng(SEED + 2)
    direct_indices = memberships["direct"]
    direct_targets = np.array([
        AXES.index(manifest["rows"][int(i)]["target_axis"]) for i in direct_indices
    ])
    shuffled = []
    for _ in range(64):
        random_targets = rng.permutation(direct_targets)
        shuffled.append(float(np.mean([
            CORE.binary_auroc(random_targets == axis, scores[direct_indices, axis])
            for axis in range(7)
        ])))

    gate_checks = {
        "direct": balanced["direct"]["macro_auroc"] >= CONTROL_GATES["direct_macro_auroc_min"],
        "quotation": balanced["quotation"]["macro_auroc"] >= CONTROL_GATES["quotation_macro_auroc_min"],
        "third_person": balanced["third_person"]["macro_auroc"] >= CONTROL_GATES["third_person_macro_auroc_min"],
        "negation": negation["paired_target_score_suppression_rate"] >= CONTROL_GATES["negation_pair_suppression_min"],
        "neutral_flat": neutral_flat["deadband_false_positive_rate"] <= CONTROL_GATES["neutral_flat_fpr_max"],
        "label_shuffle": 0.40 <= float(np.mean(shuffled)) <= 0.60,
    }
    passed = all(gate_checks.values())
    report = {
        "schema": "treebeard.jspace.g1.controls-eval.v2",
        "status": "pass" if passed else "fail",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "metadata_sha256": args.metadata_sha256,
            "raw_sha256": args.raw_sha256,
            "artifact_sha256": args.artifact_sha256,
            "rows": len(splits),
        },
        "gates": CONTROL_GATES,
        "gate_checks": gate_checks,
        "balanced_semantic_retention": balanced,
        "negation_scope": negation,
        "neutral_flat": neutral_flat,
        "label_shuffle_control": {"replicates": len(shuffled), "mean_macro_auroc": float(np.mean(shuffled))},
        "control_gate_pass": passed,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "out": str(args.out), "gate_checks": gate_checks
    }, separators=(",", ":")))
    return 0 if passed else 2


def add_capture_arguments(parser):
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--raw-sha256", required=True)


def add_evaluation_arguments(parser):
    add_capture_arguments(parser)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)


def self_test():
    assert PRIMARY_LAYER == 35 and RIDGE == 1.0 and POOLING_INDEX == 1
    assert CORE.binary_auroc([0, 1, 0, 1], [0.0, 1.0, 0.1, 0.9]) == 1.0
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g1.v2-eval-self-test.v2"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    fit_parser = subparsers.add_parser("fit")
    add_capture_arguments(fit_parser)
    fit_parser.add_argument("--out-prefix", type=Path, required=True)
    test_parser = subparsers.add_parser("test")
    add_evaluation_arguments(test_parser)
    controls_parser = subparsers.add_parser("controls")
    add_evaluation_arguments(controls_parser)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for name in ("metadata_sha256", "raw_sha256", "manifest_sha256", "artifact_sha256"):
        value = getattr(args, name, None)
        if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256")
    if args.command == "fit":
        return fit_command(args)
    if args.command == "test":
        return test_command(args)
    if args.command == "controls":
        return controls_command(args)
    parser.error("choose fit, test, or controls")


if __name__ == "__main__":
    raise SystemExit(main())
