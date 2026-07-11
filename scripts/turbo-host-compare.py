#!/usr/bin/env python3
"""Compare two Turbo host baselines with strict identity and regression gates."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = 1
BASELINE_KIND = "turbo-host-baseline"
BASELINE_SCHEMA_VERSION = 1


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def dig(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def identity_complete(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(identity_complete(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(identity_complete(item) for item in value)
    return value is not None and value != ""


def relative_delta_pct(reference: float, candidate: float) -> float | None:
    if reference == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - reference) / reference * 100.0


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
    if detail:
        gate["detail"] = detail
    gates.append(gate)


def baseline_validity(baseline: dict[str, Any]) -> tuple[bool, str]:
    kind = baseline.get("kind")
    schema = baseline.get("schema_version")
    if kind != BASELINE_KIND:
        return False, f"expected kind {BASELINE_KIND!r}, got {kind!r}"
    if schema != BASELINE_SCHEMA_VERSION:
        return False, f"expected schema {BASELINE_SCHEMA_VERSION}, got {schema!r}"
    return True, "valid"


def props(baseline: dict[str, Any]) -> dict[str, Any]:
    value = dig(baseline, "production", "http", "props", "value")
    return value if isinstance(value, dict) else {}


def health_ok(baseline: dict[str, Any]) -> bool:
    return (
        dig(baseline, "production", "http", "health", "ok") is True
        and dig(baseline, "production", "http", "health", "status") == 200
        and dig(baseline, "production", "http", "health", "value", "status") == "ok"
    )


def service_active(baseline: dict[str, Any]) -> bool:
    properties = dig(baseline, "production", "service", "properties")
    return (
        isinstance(properties, dict)
        and properties.get("ActiveState") == "active"
        and properties.get("SubState") == "running"
    )


def workload_identity(baseline: dict[str, Any]) -> dict[str, Any]:
    value = props(baseline)
    return {
        "build_info": value.get("build_info"),
        "model_alias": value.get("model_alias"),
        "model_path": value.get("model_path"),
        "total_slots": value.get("total_slots"),
        "n_ctx": dig(value, "default_generation_settings", "n_ctx"),
    }


def hardware_identity(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": dig(baseline, "host", "architecture"),
        "cpu_model": dig(baseline, "cpu", "model_name"),
        "cpu_online": dig(baseline, "cpu", "online"),
        "board_vendor": dig(baseline, "host", "dmi", "board_vendor"),
        "board_name": dig(baseline, "host", "dmi", "board_name"),
        "board_version": dig(baseline, "host", "dmi", "board_version"),
        "xe_device": xe_device_identity(baseline),
    }


def xe_device_identity(baseline: dict[str, Any]) -> dict[str, Any] | None:
    devices = dig(baseline, "pci", "devices")
    if not isinstance(devices, list):
        return None
    for row in devices:
        if isinstance(row, dict) and row.get("driver") == "xe":
            return {
                key: row.get(key)
                for key in ("vendor", "device", "subsystem_vendor", "subsystem_device")
            }
    return None


def host_pcie_link(baseline: dict[str, Any]) -> dict[str, Any] | None:
    devices = dig(baseline, "pci", "devices")
    if not isinstance(devices, list):
        return None
    candidates: list[dict[str, Any]] = []
    for row in devices:
        if not isinstance(row, dict):
            continue
        width_text = row.get("current_link_width")
        speed_text = row.get("current_link_speed")
        try:
            width = int(width_text)
        except (TypeError, ValueError):
            continue
        match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", str(speed_text or ""))
        if not match:
            continue
        candidates.append(
            {
                "bdf": row.get("bdf"),
                "width": width,
                "speed_gtps": float(match.group(1)),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["width"], row["speed_gtps"]))


def scheduler_sample(baseline: dict[str, Any]) -> dict[str, Any] | None:
    value = dig(baseline, "microbenchmarks", "sched_pipe")
    return value if isinstance(value, dict) else None


def energy_watts(baseline: dict[str, Any], counter: str) -> float | None:
    return finite_number(dig(baseline, "energy", "average_watts", counter))


def memory_psi_avg10(baseline: dict[str, Any], category: str) -> float | None:
    text = dig(baseline, "memory", "pressure", "memory")
    if not isinstance(text, str):
        return None
    pattern = rf"^{re.escape(category)}\s+avg10=([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def policy_summary(baseline: dict[str, Any]) -> dict[str, Any]:
    policies = dig(baseline, "cpu", "cpufreq_policies")
    first = policies[0] if isinstance(policies, list) and policies else {}
    idle_states = dig(baseline, "cpu", "cpu0_idle_states")
    idle_names = (
        [row.get("name") for row in idle_states if isinstance(row, dict)]
        if isinstance(idle_states, list)
        else []
    )
    return {
        "kernel_release": dig(baseline, "host", "kernel_release"),
        "boot_id": dig(baseline, "host", "boot_id"),
        "cmdline": dig(baseline, "host", "cmdline"),
        "governor": first.get("scaling_governor") if isinstance(first, dict) else None,
        "energy_performance_preference": (
            first.get("energy_performance_preference") if isinstance(first, dict) else None
        ),
        "scaling_min_freq": first.get("scaling_min_freq") if isinstance(first, dict) else None,
        "scaling_max_freq": first.get("scaling_max_freq") if isinstance(first, dict) else None,
        "idle_states": idle_names,
        "sysctls": dig(baseline, "memory", "sysctls"),
    }


def compare(
    reference: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    for label, baseline in (("reference", reference), ("candidate", candidate)):
        valid, detail = baseline_validity(baseline)
        add_gate(gates, f"{label}_baseline_valid", valid, candidate=detail)

    reference_hardware = hardware_identity(reference)
    candidate_hardware = hardware_identity(candidate)
    add_gate(
        gates,
        "hardware_identity",
        reference_hardware == candidate_hardware and identity_complete(reference_hardware),
        reference=reference_hardware,
        candidate=candidate_hardware,
        detail="architecture, CPU topology, board, and Xe identity must match",
    )

    reference_workload = workload_identity(reference)
    candidate_workload = workload_identity(candidate)
    add_gate(
        gates,
        "workload_identity",
        reference_workload == candidate_workload and identity_complete(reference_workload),
        reference=reference_workload,
        candidate=candidate_workload,
        detail="build, model, alias, slot count, and context must match",
    )

    add_gate(gates, "reference_health", health_ok(reference), candidate=health_ok(reference))
    add_gate(gates, "candidate_health", health_ok(candidate), candidate=health_ok(candidate))
    add_gate(
        gates,
        "reference_service_active",
        service_active(reference),
        candidate=service_active(reference),
    )
    add_gate(
        gates,
        "candidate_service_active",
        service_active(candidate),
        candidate=service_active(candidate),
    )

    reference_link = host_pcie_link(reference)
    candidate_link = host_pcie_link(candidate)
    link_passed = (
        reference_link is not None
        and candidate_link is not None
        and candidate_link["width"] >= reference_link["width"]
        and candidate_link["speed_gtps"] >= reference_link["speed_gtps"]
    )
    add_gate(
        gates,
        "host_pcie_link_not_regressed",
        link_passed,
        reference=reference_link,
        candidate=candidate_link,
    )

    reference_sched = scheduler_sample(reference)
    candidate_sched = scheduler_sample(candidate)
    reference_sched_median = finite_number(
        dig(reference_sched or {}, "summary", "median_usecs_per_op")
    )
    candidate_sched_median = finite_number(
        dig(candidate_sched or {}, "summary", "median_usecs_per_op")
    )
    sched_identity = (
        reference_sched is not None
        and candidate_sched is not None
        and reference_sched.get("loops") == candidate_sched.get("loops")
        and finite_number(reference_sched.get("accepted_samples")) is not None
        and finite_number(candidate_sched.get("accepted_samples")) is not None
        and int(reference_sched["accepted_samples"]) >= args.min_sched_samples
        and int(candidate_sched["accepted_samples"]) >= args.min_sched_samples
    )
    add_gate(
        gates,
        "scheduler_sample_integrity",
        sched_identity,
        reference={
            "loops": reference_sched.get("loops") if reference_sched else None,
            "accepted_samples": (
                reference_sched.get("accepted_samples") if reference_sched else None
            ),
        },
        candidate={
            "loops": candidate_sched.get("loops") if candidate_sched else None,
            "accepted_samples": (
                candidate_sched.get("accepted_samples") if candidate_sched else None
            ),
        },
    )
    sched_delta = (
        relative_delta_pct(reference_sched_median, candidate_sched_median)
        if reference_sched_median is not None and candidate_sched_median is not None
        else None
    )
    add_gate(
        gates,
        "scheduler_median_regression",
        sched_delta is not None and sched_delta <= args.max_sched_regression_pct,
        reference=reference_sched_median,
        candidate=candidate_sched_median,
        detail=(
            f"delta_pct={sched_delta}; maximum={args.max_sched_regression_pct}"
            if sched_delta is not None
            else "missing finite scheduler median"
        ),
    )

    energy: dict[str, Any] = {}
    for counter in args.energy_counter:
        reference_watts = energy_watts(reference, counter)
        candidate_watts = energy_watts(candidate, counter)
        delta = (
            relative_delta_pct(reference_watts, candidate_watts)
            if reference_watts is not None and candidate_watts is not None
            else None
        )
        energy[counter] = {
            "reference_watts": reference_watts,
            "candidate_watts": candidate_watts,
            "delta_pct": delta,
        }
        add_gate(
            gates,
            f"energy_{counter.replace('/', '_')}_regression",
            delta is not None and delta <= args.max_energy_regression_pct,
            reference=reference_watts,
            candidate=candidate_watts,
            detail=(
                f"delta_pct={delta}; maximum={args.max_energy_regression_pct}"
                if delta is not None
                else "missing finite energy sample"
            ),
        )

    candidate_memory_some = memory_psi_avg10(candidate, "some")
    candidate_memory_full = memory_psi_avg10(candidate, "full")
    add_gate(
        gates,
        "candidate_memory_psi",
        candidate_memory_some is not None
        and candidate_memory_full is not None
        and candidate_memory_some <= args.max_memory_psi_avg10
        and candidate_memory_full <= args.max_memory_psi_avg10,
        candidate={"some_avg10": candidate_memory_some, "full_avg10": candidate_memory_full},
        detail=f"maximum avg10={args.max_memory_psi_avg10}",
    )

    reference_boot_id = dig(reference, "host", "boot_id")
    candidate_boot_id = dig(candidate, "host", "boot_id")
    if args.require_different_boot:
        add_gate(
            gates,
            "different_boot",
            bool(
                reference_boot_id
                and candidate_boot_id
                and reference_boot_id != candidate_boot_id
            ),
            reference=reference_boot_id,
            candidate=candidate_boot_id,
        )

    reference_cmdline = dig(reference, "host", "cmdline")
    candidate_cmdline = dig(candidate, "host", "cmdline")
    if args.require_cmdline_change:
        add_gate(
            gates,
            "cmdline_changed",
            bool(
                reference_cmdline
                and candidate_cmdline
                and reference_cmdline != candidate_cmdline
            ),
            reference=reference_cmdline,
            candidate=candidate_cmdline,
        )
    for token in args.expect_removed_cmdline_token:
        add_gate(
            gates,
            f"cmdline_removed_{token}",
            isinstance(reference_cmdline, str)
            and isinstance(candidate_cmdline, str)
            and token in reference_cmdline.split()
            and token not in candidate_cmdline.split(),
            reference=reference_cmdline,
            candidate=candidate_cmdline,
        )

    wall_power: dict[str, Any] | None = None
    if (
        args.require_wall_power
        or args.reference_wall_watts is not None
        or args.candidate_wall_watts is not None
    ):
        wall_delta = (
            relative_delta_pct(args.reference_wall_watts, args.candidate_wall_watts)
            if args.reference_wall_watts is not None and args.candidate_wall_watts is not None
            else None
        )
        improvement = -wall_delta if wall_delta is not None else None
        wall_power = {
            "reference_watts": args.reference_wall_watts,
            "candidate_watts": args.candidate_wall_watts,
            "delta_pct": wall_delta,
            "improvement_pct": improvement,
        }
        add_gate(
            gates,
            "wall_power_improvement",
            improvement is not None and improvement >= args.min_wall_power_improvement_pct,
            reference=args.reference_wall_watts,
            candidate=args.candidate_wall_watts,
            detail=(
                f"improvement_pct={improvement}; minimum={args.min_wall_power_improvement_pct}"
                if improvement is not None
                else "both wall-power measurements are required"
            ),
        )

    passed = all(gate["passed"] for gate in gates)
    return {
        "kind": "turbo-host-comparison",
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": passed,
        "reference_captured_at_utc": reference.get("captured_at_utc"),
        "candidate_captured_at_utc": candidate.get("captured_at_utc"),
        "gates": gates,
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "measurements": {
            "scheduler": {
                "reference_median_usecs_per_op": reference_sched_median,
                "candidate_median_usecs_per_op": candidate_sched_median,
                "delta_pct": sched_delta,
            },
            "energy": energy,
            "wall_power": wall_power,
        },
        "policy": {
            "reference": policy_summary(reference),
            "candidate": policy_summary(candidate),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--min-sched-samples", type=int, default=3)
    parser.add_argument("--max-sched-regression-pct", type=float, default=10.0)
    parser.add_argument("--max-energy-regression-pct", type=float, default=5.0)
    parser.add_argument("--max-memory-psi-avg10", type=float, default=0.0)
    parser.add_argument("--energy-counter", action="append", default=[])
    parser.add_argument("--require-different-boot", action="store_true")
    parser.add_argument("--require-cmdline-change", action="store_true")
    parser.add_argument("--expect-removed-cmdline-token", action="append", default=[])
    parser.add_argument("--require-wall-power", action="store_true")
    parser.add_argument("--reference-wall-watts", type=float)
    parser.add_argument("--candidate-wall-watts", type=float)
    parser.add_argument("--min-wall-power-improvement-pct", type=float, default=0.0)
    args = parser.parse_args()
    if not args.energy_counter:
        args.energy_counter = ["xe/card", "xe/pkg"]
    if args.min_sched_samples < 1:
        parser.error("--min-sched-samples must be positive")
    for name in (
        "max_sched_regression_pct",
        "max_energy_regression_pct",
        "max_memory_psi_avg10",
        "min_wall_power_improvement_pct",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("reference_wall_watts", "candidate_wall_watts"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (args.reference_wall_watts is None) != (args.candidate_wall_watts is None):
        parser.error("reference and candidate wall-power measurements must be supplied together")
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
    output = json.dumps(result, indent=None if args.compact else 2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        sys.stdout.write(output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
