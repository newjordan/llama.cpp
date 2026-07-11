#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare saved Turbo multi-agent Pareto result JSONs without contacting a server."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


RESULT_KIND = "turbo-multiagent-pareto"
RESULT_SCHEMA_VERSION = 1
WORKLOAD_CONFIG_KEYS = ("ctx", "n_predict", "parallel", "prompt", "repeats")
METRIC_SPECS = {
    "aggregate_wall_tps": "higher",
    "mean_slot_tps": "higher",
    "min_slot_tps": "higher",
    "p95_client_ms": "lower",
}
SUMMARY_STAT_KEYS = ("n", "min", "mean", "p50", "p95", "max")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def add_gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    reference: Any = None,
    candidate: Any = None,
    detail: str | None = None,
) -> None:
    gate: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "reference": reference,
        "candidate": candidate,
    }
    if detail is not None:
        gate["detail"] = detail
    gates.append(gate)


def workload_identity(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config")
    if not isinstance(config, dict):
        return {key: None for key in WORKLOAD_CONFIG_KEYS}
    return {key: config.get(key) for key in WORKLOAD_CONFIG_KEYS}


def result_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("kind") != RESULT_KIND:
        errors.append(f"kind must be {RESULT_KIND!r}")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RESULT_SCHEMA_VERSION}")
    if result.get("passed") is not True:
        errors.append("result did not pass its own benchmark gate")

    config = result.get("config")
    if not isinstance(config, dict):
        errors.append("config must be an object")
    else:
        for key in WORKLOAD_CONFIG_KEYS:
            if key not in config:
                errors.append(f"config.{key} is missing")

    summary = result.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("summary.rows must be a non-empty array")
        return errors

    seen_agents: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"summary.rows[{index}] must be an object")
            continue
        agents = row.get("active_agents")
        if type(agents) is not int or agents <= 0:
            errors.append(f"summary.rows[{index}].active_agents must be positive")
            continue
        if agents in seen_agents:
            errors.append(f"summary.rows has duplicate active_agents={agents}")
        seen_agents.add(agents)
    return errors


def rows_by_agent(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    summary = result.get("summary")
    rows = summary.get("rows") if isinstance(summary, dict) else None
    if not isinstance(rows, list):
        return {}
    result_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        agents = row.get("active_agents")
        if type(agents) is int and agents > 0 and agents not in result_rows:
            result_rows[agents] = row
    return result_rows


def metric_summary(row: dict[str, Any], metric: str) -> dict[str, Any] | None:
    value = row.get(metric)
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in SUMMARY_STAT_KEYS}


def relative_delta_pct(reference: float, candidate: float) -> float | None:
    if reference == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - reference) / reference * 100.0


def compare_metric(
    reference_row: dict[str, Any],
    candidate_row: dict[str, Any],
    metric: str,
    direction: str,
    max_regression_pct: float,
) -> tuple[dict[str, Any], str | None]:
    reference = metric_summary(reference_row, metric)
    candidate = metric_summary(candidate_row, metric)
    reference_p50 = finite_number(reference.get("p50")) if reference else None
    candidate_p50 = finite_number(candidate.get("p50")) if candidate else None

    output: dict[str, Any] = {
        "direction": "higher_is_better" if direction == "higher" else "lower_is_better",
        "reference": reference,
        "candidate": candidate,
        "reference_p50": reference_p50,
        "candidate_p50": candidate_p50,
        "absolute_delta": None,
        "percent_delta": None,
        "regression_pct": None,
        "max_regression_pct": max_regression_pct,
        "passed": False,
    }
    if reference_p50 is None or candidate_p50 is None or reference_p50 <= 0 or candidate_p50 <= 0:
        return output, "metric requires positive finite p50 values on both results"

    percent_delta = relative_delta_pct(reference_p50, candidate_p50)
    if percent_delta is None:
        return output, "cannot calculate a percent delta from the reference p50"

    regression_pct = max(0.0, -percent_delta) if direction == "higher" else max(0.0, percent_delta)
    output.update(
        {
            "absolute_delta": candidate_p50 - reference_p50,
            "percent_delta": percent_delta,
            "regression_pct": regression_pct,
            "passed": regression_pct <= max_regression_pct,
        }
    )
    return output, None


def compare(
    reference: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []
    reference_errors = result_errors(reference)
    candidate_errors = result_errors(candidate)
    add_gate(
        gates,
        "reference_result_valid",
        not reference_errors,
        reference=reference_errors,
        detail="kind, schema, benchmark pass state, workload fields, and unique rows are required",
    )
    add_gate(
        gates,
        "candidate_result_valid",
        not candidate_errors,
        candidate=candidate_errors,
        detail="kind, schema, benchmark pass state, workload fields, and unique rows are required",
    )

    reference_workload = workload_identity(reference)
    candidate_workload = workload_identity(candidate)
    workload_complete = all(value is not None and value != "" for value in reference_workload.values()) and all(
        value is not None and value != "" for value in candidate_workload.values()
    )
    add_gate(
        gates,
        "workload_identity",
        workload_complete and reference_workload == candidate_workload,
        reference=reference_workload,
        candidate=candidate_workload,
        detail="ctx, n_predict, parallel, prompt, and repeats must match exactly",
    )

    reference_rows = rows_by_agent(reference)
    candidate_rows = rows_by_agent(candidate)
    shared_agents = sorted(set(reference_rows) & set(candidate_rows))
    add_gate(
        gates,
        "intersecting_agents",
        bool(shared_agents),
        reference=sorted(reference_rows),
        candidate=sorted(candidate_rows),
        detail="at least one active-agent count must occur in both results",
    )

    rows: list[dict[str, Any]] = []
    for agents in shared_agents:
        reference_row = reference_rows[agents]
        candidate_row = candidate_rows[agents]
        row: dict[str, Any] = {
            "active_agents": agents,
            "reference": {
                "samples": reference_row.get("samples"),
                "failed_samples": reference_row.get("failed_samples"),
            },
            "candidate": {
                "samples": candidate_row.get("samples"),
                "failed_samples": candidate_row.get("failed_samples"),
            },
            "metrics": {},
        }
        for metric, direction in METRIC_SPECS.items():
            metric_result, error = compare_metric(
                reference_row,
                candidate_row,
                metric,
                direction,
                args.max_regression_pct,
            )
            row["metrics"][metric] = metric_result
            gate_name = f"agents_{agents}_{metric}_regression"
            add_gate(
                gates,
                gate_name,
                metric_result["passed"],
                reference=metric_result["reference_p50"],
                candidate=metric_result["candidate_p50"],
                detail=(
                    error
                    or f"percent_delta={metric_result['percent_delta']}; "
                    f"regression_pct={metric_result['regression_pct']}; "
                    f"maximum={args.max_regression_pct}"
                ),
            )
        rows.append(row)

    passed = all(gate["passed"] for gate in gates)
    return {
        "kind": "turbo-pareto-comparison",
        "schema_version": 1,
        "passed": passed,
        "config": {
            "max_regression_pct": args.max_regression_pct,
            "workload_config_keys": list(WORKLOAD_CONFIG_KEYS),
            "metrics": list(METRIC_SPECS),
        },
        "reference": {
            "label": reference.get("label"),
            "workload": reference_workload,
            "agents": sorted(reference_rows),
        },
        "candidate": {
            "label": candidate.get("label"),
            "workload": candidate_workload,
            "agents": sorted(candidate_rows),
        },
        "intersecting_agents": shared_agents,
        "reference_only_agents": sorted(set(reference_rows) - set(candidate_rows)),
        "candidate_only_agents": sorted(set(candidate_rows) - set(reference_rows)),
        "rows": rows,
        "gates": gates,
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference turbo-multiagent-pareto result JSON")
    parser.add_argument("candidate", type=Path, help="candidate turbo-multiagent-pareto result JSON")
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=1.0,
        help="maximum allowed regression for each p50 metric (default: 1.0)",
    )
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args()
    if not math.isfinite(args.max_regression_pct) or args.max_regression_pct < 0:
        parser.error("--max-regression-pct must be finite and non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        reference = load_json(args.reference)
        candidate = load_json(args.candidate)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = compare(reference, candidate, args)
    result["reference_path"] = str(args.reference)
    result["candidate_path"] = str(args.candidate)
    text = json.dumps(result, indent=None if args.compact else 2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
