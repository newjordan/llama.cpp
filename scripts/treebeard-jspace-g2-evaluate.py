#!/usr/bin/env python3

"""Evaluate and select preregistered Treebeard J-Space G2 candidates."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
REFUSAL_RE = re.compile(
    r"\b(?:i (?:cannot|can't|won't|am unable to)|as an ai|i must refuse|i'm sorry,? but)\b",
    re.IGNORECASE,
)
DEV_SCALES = (
    0.008641079027104324,
    0.017282158054208648,
    0.034564316108417296,
)
POLICIES = ("engage_curiosity", "match_route")


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


def load_json(path, expected, label):
    require_sha(path, expected, label)
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return {"lower": None, "upper": None}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"lower": center - spread, "upper": center + spread}


def preference_metrics(rows):
    counts = Counter(row["preference"] for row in rows)
    total = len(rows)
    successes = counts["candidate"] + 0.5 * counts["tie"]
    return {
        "comparisons": total,
        "candidate_wins": counts["candidate"],
        "control_wins": counts["control"],
        "ties": counts["tie"],
        "score": successes / total if total else None,
        "wilson_95": wilson(successes, total),
    }


def repeated_ngram(text, size=4):
    words = WORD_RE.findall(text.casefold())
    if len(words) < size * 2:
        return False
    grams = [tuple(words[index:index + size]) for index in range(len(words) - size + 1)]
    return len(grams) != len(set(grams))


def malformed(text):
    return not text.strip() or "\ufffd" in text


def anchor_hit(text, anchor_words):
    return bool(set(WORD_RE.findall(text.casefold())) & anchor_words)


def validate_protected(text, validator):
    stripped = text.strip()
    kind = validator["kind"]
    if kind == "exact":
        return stripped == validator["value"]
    if kind == "exact_casefold":
        return stripped.casefold() == validator["value"].casefold()
    if kind == "json_equal":
        try:
            return json.loads(stripped) == validator["value"]
        except json.JSONDecodeError:
            return False
    if kind == "contains_any_casefold":
        lowered = stripped.casefold()
        return any(value.casefold() in lowered for value in validator["values"])
    raise ValueError(f"unknown protected validator: {kind}")


def rate(values):
    return sum(bool(value) for value in values) / len(values) if values else None


def metric_delta(candidate, control):
    return candidate - control


def load_evaluation_inputs(args):
    manifest = load_json(args.manifest, args.manifest_sha256, "response manifest")
    routes = load_json(args.routes, args.routes_sha256, "routes")
    judgments = load_json(args.judgments, args.judgments_sha256, "judgments")
    embeddings = load_json(args.embeddings, args.embeddings_sha256, "embeddings")
    require_sha(args.embedding_raw, args.embedding_raw_sha256, "raw embeddings")
    anchors = load_json(args.anchors, args.anchors_sha256, "anchor manifest")
    if manifest.get("schema") != "treebeard.jspace.g2.response-set.v1" or \
            routes.get("schema") != "treebeard.jspace.g2.routes.v1" or \
            judgments.get("schema") != "treebeard.jspace.g2.blind-judgments.v1" or \
            embeddings.get("schema") != "treebeard.jspace.g2.embedding-scores.v1":
        raise ValueError("unexpected evaluation input schema")
    if judgments.get("status") != "complete" or embeddings.get("status") != "complete":
        raise ValueError("judge or embedding capture is incomplete")
    if routes.get("inputs", {}).get("response_manifest_sha256") != args.manifest_sha256:
        raise ValueError("routes do not attest the response manifest")
    for document, label in ((judgments, "judgments"), (embeddings, "embeddings")):
        if document.get("inputs", {}).get("manifest_sha256") != args.manifest_sha256 or \
                document.get("inputs", {}).get("routes_sha256") != args.routes_sha256:
            raise ValueError(f"{label} do not attest the manifest and routes")
    if embeddings.get("embedding", {}).get("raw_sha256") != args.embedding_raw_sha256:
        raise ValueError("embedding score document does not attest the raw embeddings")

    generated = {}
    response_inputs = []
    candidate_policies = set()
    candidate_scales = set()
    attestation_shas = set()
    for index, path in enumerate(args.responses):
        expected = args.responses_sha256[index]
        document = load_json(path, expected, f"generated response file {index}")
        if document.get("schema") != "treebeard.jspace.g2.generated-responses.v1" or \
                document.get("status") != "complete" or \
                document.get("inputs", {}).get("manifest_sha256") != args.manifest_sha256 or \
                document.get("inputs", {}).get("routes_sha256") != args.routes_sha256:
            raise ValueError(f"unexpected or unbound generated responses: {path}")
        vector_sha = document.get("inputs", {}).get("vector_sha256")
        if not vector_sha or document.get("generation", {}).get("vector_sha256") != vector_sha:
            raise ValueError(f"response vector provenance is internally inconsistent: {path}")
        attestation_shas.add(document.get("inputs", {}).get("attestation_sha256"))
        for record in document["records"]:
            if record.get("vector_sha256") != vector_sha:
                raise ValueError(f"response record vector provenance differs: {path}")
            key = (record["sample_id"], int(record["seed"]), record["arm"])
            if key in generated:
                raise ValueError(f"duplicate generated response: {key}")
            generated[key] = record
            if record["arm"] == "candidate":
                candidate_policies.add(record["policy"])
                if record["active"]:
                    candidate_scales.add(float(record["scale"]))
        response_inputs.append({"path": str(path), "sha256": expected})
    if None in attestation_shas or len(attestation_shas) != 1:
        raise ValueError("response files do not share one run attestation")
    for document, label in ((judgments, "judgments"), (embeddings, "embeddings")):
        if document.get("inputs", {}).get("response_files") != response_inputs:
            raise ValueError(f"{label} do not attest the exact evaluated response files")
    if len(candidate_policies) != 1 or len(candidate_scales) != 1:
        raise ValueError("candidate policy or active scale is not unique")
    policy = next(iter(candidate_policies))
    scale = next(iter(candidate_scales))
    if policy not in POLICIES:
        raise ValueError("candidate policy is not preregistered")
    if args.mode == "development" and not any(math.isclose(scale, value) for value in DEV_SCALES):
        raise ValueError("development scale is not preregistered")

    rows = {row["sample_id"]: row for row in manifest["rows"]}
    route_rows = {row["sample_id"]: row for row in routes["rows"]}
    if list(rows) != list(route_rows):
        raise ValueError("manifest and route orders differ")
    control_seeds = defaultdict(set)
    candidate_seeds = defaultdict(set)
    for sample_id, seed, arm in generated:
        (control_seeds if arm == "control" else candidate_seeds)[sample_id].add(seed)
    for sample_id in rows:
        if not control_seeds[sample_id] or control_seeds[sample_id] != candidate_seeds[sample_id]:
            raise ValueError(f"missing or asymmetric control/candidate seeds: {sample_id}")
    pair_keys = sorted(
        (sample_id, seed) for sample_id in rows for seed in control_seeds[sample_id])
    for sample_id, seed in pair_keys:
        control = generated[(sample_id, seed, "control")]
        candidate = generated[(sample_id, seed, "candidate")]
        if control.get("vector_axis") != candidate.get("vector_axis") or \
                control.get("vector_sha256") != candidate.get("vector_sha256"):
            raise ValueError(
                f"control/candidate loaded-vector provenance differs: {(sample_id, seed)}")

    judge_rows = {(row["sample_id"], int(row["seed"])): row
                  for row in judgments["records"]}
    embedding_rows = {(row["sample_id"], int(row["seed"])): row
                      for row in embeddings["rows"]}
    dialogue_keys = [key for key in pair_keys if rows[key[0]]["scope"] == "dialogue_response"]
    if set(judge_rows) != set(dialogue_keys) or set(embedding_rows) != set(dialogue_keys):
        raise ValueError("judge or embedding rows do not exactly cover dialogue pairs")

    anchor_words = {
        row["word"].casefold()
        for axis_rows in anchors["axes"].values()
        for row in axis_rows
    }
    return {
        "manifest": manifest,
        "routes": routes,
        "rows": rows,
        "route_rows": route_rows,
        "generated": generated,
        "pair_keys": pair_keys,
        "dialogue_keys": dialogue_keys,
        "judge_rows": judge_rows,
        "embedding_rows": embedding_rows,
        "anchor_words": anchor_words,
        "policy": policy,
        "scale": scale,
        "response_inputs": response_inputs,
    }


def evaluate_command(args):
    data = load_evaluation_inputs(args)
    rows = data["rows"]
    routes = data["route_rows"]
    generated = data["generated"]
    judge_rows = data["judge_rows"]
    embeddings = data["embedding_rows"]

    active_keys = [key for key in data["dialogue_keys"] if routes[key[0]]["active"]]
    neutral_keys = [key for key in data["dialogue_keys"] if rows[key[0]]["label"] == "neutral"]
    judge_active = [judge_rows[key] for key in active_keys]
    judge_neutral = [judge_rows[key] for key in neutral_keys]
    source_groups = {}
    for source in sorted({rows[key[0]]["source"] for key in active_keys}):
        source_groups[source] = preference_metrics([
            judge_rows[key] for key in active_keys if rows[key[0]]["source"] == source
        ])
    axis_groups = {}
    for axis in sorted({routes[key[0]]["routed_axis"] for key in active_keys}):
        axis_groups[axis] = preference_metrics([
            judge_rows[key] for key in active_keys if routes[key[0]]["routed_axis"] == axis
        ])
    label_groups = {}
    for label in sorted({
            rows[key[0]]["label"] for key in active_keys
            if rows[key[0]]["label"] != "neutral"}):
        label_groups[label] = preference_metrics([
            judge_rows[key] for key in active_keys if rows[key[0]]["label"] == label
        ])
    source_label_groups = {}
    for source, label in sorted({
            (rows[key[0]]["source"], rows[key[0]]["label"])
            for key in active_keys if rows[key[0]]["label"] != "neutral"}):
        source_label_groups[f"{source}:{label}"] = preference_metrics([
            judge_rows[key] for key in active_keys
            if rows[key[0]]["source"] == source and rows[key[0]]["label"] == label
        ])

    changed = []
    lengths = []
    output_flags = {arm: defaultdict(list) for arm in ("control", "candidate")}
    inactive_exact = []
    for key in data["dialogue_keys"]:
        sample_id, seed = key
        control = generated[(sample_id, seed, "control")]["response"]
        candidate = generated[(sample_id, seed, "candidate")]["response"]
        if routes[sample_id]["active"]:
            changed.append(control["tokens"] != candidate["tokens"])
        else:
            inactive_exact.append(control["tokens"] == candidate["tokens"])
        lengths.append(candidate["counts"]["generated_tokens"] /
                       max(1, control["counts"]["generated_tokens"]))
        for arm, response in (("control", control), ("candidate", candidate)):
            text = response["content"]
            output_flags[arm]["anchor"].append(anchor_hit(text, data["anchor_words"]))
            output_flags[arm]["repeated_4gram"].append(repeated_ngram(text))
            output_flags[arm]["malformed"].append(malformed(text))
            output_flags[arm]["unsolicited_refusal"].append(bool(REFUSAL_RE.search(text)))

    output_metrics = {}
    for name in output_flags["control"]:
        control_rate = rate(output_flags["control"][name])
        candidate_rate = rate(output_flags["candidate"][name])
        output_metrics[name] = {
            "control_rate": control_rate,
            "candidate_rate": candidate_rate,
            "candidate_minus_control": metric_delta(candidate_rate, control_rate),
        }
    embedding_delta = float(np.mean([
        embeddings[key]["candidate_minus_control"] for key in data["dialogue_keys"]
    ]))
    violations = Counter(
        violation for row in judge_rows.values()
        for violation in row["candidate_only_violations"])

    protected_keys = [key for key in data["pair_keys"]
                      if rows[key[0]]["scope"] == "protected_task"]
    protected_exact = []
    protected_outcomes = defaultdict(lambda: {"control": [], "candidate": []})
    for sample_id, seed in protected_keys:
        control = generated[(sample_id, seed, "control")]["response"]
        candidate = generated[(sample_id, seed, "candidate")]["response"]
        protected_exact.append(control["tokens"] == candidate["tokens"])
        validator = rows[sample_id]["validator"]
        protected_outcomes[sample_id]["control"].append(
            validate_protected(control["content"], validator))
        protected_outcomes[sample_id]["candidate"].append(
            validate_protected(candidate["content"], validator))
    protected_cases_control = sum(
        all(outcomes["control"]) for outcomes in protected_outcomes.values())
    protected_outcome_match = all(
        outcomes["control"] == outcomes["candidate"]
        for outcomes in protected_outcomes.values())

    neutral_samples = {key[0] for key in neutral_keys}
    neutral_active_rate = (
        sum(routes[sample_id]["active"] for sample_id in neutral_samples) /
        len(neutral_samples) if neutral_samples else None
    )
    primary = preference_metrics(judge_active)
    metrics = {
        "dialogue_pairs": len(data["dialogue_keys"]),
        "active_pairs": len(active_keys),
        "active_rows": len({key[0] for key in active_keys}),
        "active_preference": primary,
        "source_preference": source_groups,
        "routed_axis_preference": axis_groups,
        "source_label_preference": source_label_groups,
        "label_preference": label_groups,
        "neutral_preference": preference_metrics(judge_neutral),
        "neutral_active_rate": neutral_active_rate,
        "changed_active_token_rate": rate(changed),
        "inactive_token_exact_rate": rate(inactive_exact),
        "mean_candidate_control_length_ratio": float(np.mean(lengths)),
        "mean_reference_cosine_delta": embedding_delta,
        "output_pathologies": output_metrics,
        "candidate_only_judge_violations": dict(sorted(violations.items())),
        "candidate_only_judge_violation_total": sum(violations.values()),
        "protected": {
            "pairs": len(protected_keys),
            "token_exact_rate": rate(protected_exact),
            "control_cases_passing_all_seeds": protected_cases_control,
            "cases": len(protected_outcomes),
            "candidate_outcomes_match_control": protected_outcome_match,
        },
    }

    if args.mode == "development":
        gates = {
            "active_rows_min_20": metrics["active_rows"] >= 20,
            "changed_active_rate_min_0_10": metrics["changed_active_token_rate"] >= 0.10,
            "active_preference_min_0_52": primary["score"] >= 0.52,
            "each_source_preference_min_0_45": bool(source_groups) and
                min(value["score"] for value in source_groups.values()) >= 0.45,
            "reference_cosine_delta_min_minus_0_005": embedding_delta >= -0.005,
            "anchor_delta_max_0_02": output_metrics["anchor"]["candidate_minus_control"] <= 0.02,
            "repeated_4gram_delta_max_0_02":
                output_metrics["repeated_4gram"]["candidate_minus_control"] <= 0.02,
            "malformed_delta_max_0_02":
                output_metrics["malformed"]["candidate_minus_control"] <= 0.02,
            "length_ratio_0_75_to_1_33":
                0.75 <= metrics["mean_candidate_control_length_ratio"] <= 1.33,
            "zero_candidate_only_judge_violations":
                metrics["candidate_only_judge_violation_total"] == 0,
            "inactive_token_exact": metrics["inactive_token_exact_rate"] in (None, 1.0),
        }
    else:
        neutral_score = metrics["neutral_preference"]["score"]
        gates = {
            "active_preference_min_0_55": primary["score"] >= 0.55,
            "active_wilson_lower_above_0_50": primary["wilson_95"]["lower"] > 0.50,
            "each_source_preference_min_0_47": bool(source_groups) and
                min(value["score"] for value in source_groups.values()) >= 0.47,
            "each_axis_preference_min_0_40": bool(axis_groups) and
                min(value["score"] for value in axis_groups.values()) >= 0.40,
            "each_source_axis_preference_min_0_40_except_dailydialog_fear":
                bool(source_label_groups) and all(
                    value["score"] >= 0.40
                    for key, value in source_label_groups.items()
                    if key != "dailydialog:fear"),
            "reference_cosine_delta_min_minus_0_005": embedding_delta >= -0.005,
            "neutral_preference_min_0_50": neutral_score is not None and neutral_score >= 0.50,
            "neutral_active_rate_max_0_15": neutral_active_rate is not None and
                neutral_active_rate <= 0.15,
            "anchor_delta_max_0_01": output_metrics["anchor"]["candidate_minus_control"] <= 0.01,
            "repeated_4gram_delta_max_0_01":
                output_metrics["repeated_4gram"]["candidate_minus_control"] <= 0.01,
            "malformed_delta_max_0_01":
                output_metrics["malformed"]["candidate_minus_control"] <= 0.01,
            "refusal_delta_max_0_01":
                output_metrics["unsolicited_refusal"]["candidate_minus_control"] <= 0.01,
            "length_ratio_0_80_to_1_25":
                0.80 <= metrics["mean_candidate_control_length_ratio"] <= 1.25,
            "zero_candidate_only_judge_violations":
                metrics["candidate_only_judge_violation_total"] == 0,
            "protected_token_exact": metrics["protected"]["token_exact_rate"] == 1.0,
            "protected_control_cases_min_10":
                metrics["protected"]["control_cases_passing_all_seeds"] >= 10,
            "protected_outcomes_match":
                metrics["protected"]["candidate_outcomes_match_control"],
            "inactive_token_exact": metrics["inactive_token_exact_rate"] in (None, 1.0),
        }
    passed = all(gates.values())
    report = {
        "schema": "treebeard.jspace.g2.evaluation.v1",
        "status": "pass" if passed else "fail",
        "mode": args.mode,
        "candidate": {
            "policy": data["policy"],
            "scale": data["scale"],
            "id": f"{data['policy']}:{data['scale']:.18g}",
        },
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "routes_sha256": args.routes_sha256,
            "response_files": data["response_inputs"],
            "judgments_sha256": args.judgments_sha256,
            "embeddings_sha256": args.embeddings_sha256,
            "embedding_raw_sha256": args.embedding_raw_sha256,
            "anchors_sha256": args.anchors_sha256,
        },
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate": report["candidate"]["id"],
        "active_preference": primary["score"],
        "out": str(args.out),
    }, separators=(",", ":")))
    return 0 if passed else 2


def select_command(args):
    if len(args.report) != len(args.report_sha256):
        raise ValueError("--report and --report-sha256 counts differ")
    expected = {(policy, scale) for policy in POLICIES for scale in DEV_SCALES}
    candidates = []
    inputs = []
    seen = set()
    for index, path in enumerate(args.report):
        expected_sha = args.report_sha256[index]
        report = load_json(path, expected_sha, f"development report {index}")
        if report.get("schema") != "treebeard.jspace.g2.evaluation.v1" or \
                report.get("mode") != "development":
            raise ValueError("selection input is not a development evaluation")
        key = (report["candidate"]["policy"], float(report["candidate"]["scale"]))
        matched = next((item for item in expected
                        if item[0] == key[0] and math.isclose(item[1], key[1])), None)
        if matched is None or matched in seen:
            raise ValueError("selection input candidate is unexpected or duplicated")
        seen.add(matched)
        inputs.append({"path": str(path), "sha256": expected_sha})
        if report["all_gates_pass"]:
            candidates.append(report)
    if seen != expected:
        raise ValueError("selection requires all six preregistered development candidates")
    if candidates:
        selected = sorted(
            candidates,
            key=lambda report: (
                -report["metrics"]["active_preference"]["score"],
                report["candidate"]["scale"],
                0 if report["candidate"]["policy"] == "engage_curiosity" else 1,
            ),
        )[0]
        status = "selected"
        selected_value = selected["candidate"]
        selected_metrics = selected["metrics"]
    else:
        status = "stop_no_eligible_candidate"
        selected_value = None
        selected_metrics = None
    output = {
        "schema": "treebeard.jspace.g2.selected-policy.v1",
        "status": status,
        "selection_rule": (
            "highest eligible active preference; exact tie: lower scale, then engage_curiosity"
        ),
        "inputs": inputs,
        "selected": selected_value,
        "selected_metrics": selected_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected": selected_value, "out": str(args.out)},
                     separators=(",", ":")))
    return 0 if candidates else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--mode", choices=("development", "holdout"), required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--manifest-sha256", required=True)
    evaluate.add_argument("--routes", type=Path, required=True)
    evaluate.add_argument("--routes-sha256", required=True)
    evaluate.add_argument("--responses", action="append", type=Path, required=True)
    evaluate.add_argument("--responses-sha256", action="append", required=True)
    evaluate.add_argument("--judgments", type=Path, required=True)
    evaluate.add_argument("--judgments-sha256", required=True)
    evaluate.add_argument("--embeddings", type=Path, required=True)
    evaluate.add_argument("--embeddings-sha256", required=True)
    evaluate.add_argument("--embedding-raw", type=Path, required=True)
    evaluate.add_argument("--embedding-raw-sha256", required=True)
    evaluate.add_argument("--anchors", type=Path, required=True)
    evaluate.add_argument("--anchors-sha256", required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.set_defaults(func=evaluate_command)

    select = subparsers.add_parser("select")
    select.add_argument("--report", action="append", type=Path, required=True)
    select.add_argument("--report-sha256", action="append", required=True)
    select.add_argument("--out", type=Path, required=True)
    select.set_defaults(func=select_command)
    args = parser.parse_args()
    if hasattr(args, "responses") and len(args.responses) != len(args.responses_sha256):
        parser.error("--responses and --responses-sha256 counts differ")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
