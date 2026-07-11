#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def load_benchmark_module() -> ModuleType:
    path = Path(__file__).with_name("turbo-statetree-bench.py")
    spec = importlib.util.spec_from_file_location("turbo_statetree_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_benchmark_module()


def read_slots(port: int, timeout: float) -> list[dict[str, Any]]:
    body = bench.http_json("GET", f"http://127.0.0.1:{port}/slots", timeout=timeout)
    if not isinstance(body, list) or any(not isinstance(row, dict) for row in body):
        raise RuntimeError("slots response is not an array of objects")
    return body


def read_metrics(port: int, timeout: float) -> dict[str, float]:
    return bench.parse_prometheus(bench.http_text(f"http://127.0.0.1:{port}/metrics", timeout=timeout))


def state_bytes(rows: list[dict[str, Any]], slot_id: int) -> int:
    value = rows[slot_id].get("n_prompt_state_bytes")
    if type(value) is not int or value < 0:
        raise RuntimeError(f"slot {slot_id} is missing exact prompt-state bytes")
    return value


def assert_clean(rows: list[dict[str, Any]]) -> None:
    if any(row.get("is_reserved") for row in rows):
        raise RuntimeError("cleanup left a reserved slot")
    if any(int(row.get("n_prompt_state_bytes", 0)) != 0 for row in rows):
        raise RuntimeError("cleanup left prompt-state bytes")


def wait_clean(port: int, timeout: float, poll_ms: float) -> tuple[list[dict[str, Any]], float]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = read_slots(port, timeout)
        if not any(row.get("is_reserved") for row in rows):
            assert_clean(rows)
            return rows, time.monotonic()
        time.sleep(poll_ms / 1000.0)
    raise RuntimeError("retained family did not become clean before timeout")


def server_args(
    args: argparse.Namespace,
    label: str,
    lease_ms: int,
    budget_bytes: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        bin=args.bin,
        model=args.model,
        ngl=args.ngl,
        ncmoe=args.ncmoe,
        ctx=args.ctx,
        parallel=args.parallel,
        flash_attn=args.flash_attn,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        batch=args.batch,
        ubatch=args.ubatch,
        threads=args.threads,
        port=args.port,
        label=label,
        statetree_lease_ms=lease_ms,
        statetree_max_state_bytes=budget_bytes,
        extra_server_args="",
        source_oneapi=args.source_oneapi,
        request_timeout=args.request_timeout,
        startup_timeout=args.startup_timeout,
    )


class ManagedServer:
    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.args = args
        self.log_path = log_path
        self.proc: subprocess.Popen[Any] | None = None

    def __enter__(self) -> subprocess.Popen[Any]:
        self.proc = bench.launch_server(self.args, self.log_path)
        bench.wait_healthy(self.args.port, self.proc, self.args.startup_timeout)
        bench.erase_current_slots(
            self.args.port,
            list(range(self.args.parallel)),
            self.args.request_timeout,
        )
        return self.proc

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.proc is not None:
            try:
                bench.erase_current_slots(
                    self.args.port,
                    list(range(self.args.parallel)),
                    self.args.request_timeout,
                )
            except Exception:
                pass
            bench.stop_process(self.proc)
            log_file = getattr(self.proc, "_turbo_log_file", None)
            if log_file is not None:
                log_file.close()


def snapshot(args: argparse.Namespace, pid: int) -> dict[str, Any]:
    return bench.capture_snapshot(args.port, pid, list(range(args.parallel)))


def probe_checkpoint_bytes(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    launch = server_args(args, f"{args.label}-probe", 0, 0)
    with ManagedServer(launch, out_dir / "probe.server.log") as proc:
        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "retention-probe")
        request = bench.completion(
            args.port,
            prefix,
            0,
            0,
            args.seed,
            args.request_timeout,
        )
        measured = snapshot(args, proc.pid)
        rows = read_slots(args.port, args.request_timeout)
        checkpoint_bytes = state_bytes(rows, 0)
        if checkpoint_bytes <= 0:
            raise RuntimeError("development model did not expose retained prompt-state bytes")
        return {
            "checkpoint_bytes": checkpoint_bytes,
            "request": request,
            "snapshot": measured,
        }


def run_pressure_gate(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    budget = checkpoint_bytes * 3
    launch = server_args(args, f"{args.label}-pressure", 0, budget)
    with ManagedServer(launch, out_dir / "pressure.server.log") as proc:
        prompts = {
            slot_id: bench.exact_tokens(args.port, args.prefix_tokens, f"pressure-{slot_id}")
            for slot_id in range(args.parallel)
        }
        family_suffixes = {
            slot_id: bench.exact_tokens(args.port, 8, f"pressure-family-{slot_id}")
            for slot_id in (0, 1)
        }

        bench.completion(args.port, prompts[0], 0, 0, args.seed, args.request_timeout)
        family = bench.fork_slot(args.port, 0, [1], args.request_timeout)
        family_id = family["fork_id"]
        if type(family_id) is not int:
            raise RuntimeError("pressure fork did not return a generation")
        for slot_id in (0, 1):
            bench.completion(
                args.port,
                prompts[0] + family_suffixes[slot_id],
                slot_id,
                0,
                args.seed + slot_id,
                args.request_timeout,
                fork_id=family_id,
            )
        bench.completion(args.port, prompts[2], 2, 0, args.seed + 2, args.request_timeout)
        before = snapshot(args, proc.pid)
        rows = read_slots(args.port, args.request_timeout)
        fixture_bytes = [state_bytes(rows, slot_id) for slot_id in range(args.parallel)]
        fixture_expected = checkpoint_bytes * 3
        if sum(fixture_bytes) != fixture_expected:
            raise RuntimeError(
                "pressure fixture did not retain exactly three checkpoints: "
                f"slots={fixture_bytes}, expected={fixture_expected}"
            )

        bench.completion(args.port, prompts[3], 3, 0, args.seed + 3, args.request_timeout)
        after_family = snapshot(args, proc.pid)
        rows = read_slots(args.port, args.request_timeout)
        if any(state_bytes(rows, slot_id) != 0 or rows[slot_id].get("is_reserved") for slot_id in (0, 1)):
            raise RuntimeError("oldest fork family was not reclaimed atomically")
        if any(state_bytes(rows, slot_id) != checkpoint_bytes for slot_id in (2, 3)):
            raise RuntimeError("family pressure eviction damaged surviving slots")
        metrics = read_metrics(args.port, args.request_timeout)
        if metrics.get("statetree_evicted_total") != 1:
            raise RuntimeError("family pressure eviction counter is not exactly one")
        if metrics.get("statetree_reclaimed_bytes_total") != checkpoint_bytes * 2:
            raise RuntimeError("family pressure reclamation bytes are not exact")

        bench.completion(args.port, prompts[0], 0, 0, args.seed + 4, args.request_timeout)
        at_budget = snapshot(args, proc.pid)
        bench.completion(args.port, prompts[1], 1, 0, args.seed + 5, args.request_timeout)
        after_slot = snapshot(args, proc.pid)
        rows = read_slots(args.port, args.request_timeout)
        if state_bytes(rows, 2) != 0:
            raise RuntimeError("oldest ordinary slot was not the second pressure victim")
        if any(state_bytes(rows, slot_id) != checkpoint_bytes for slot_id in (0, 1, 3)):
            raise RuntimeError("ordinary pressure eviction damaged newer slots")
        metrics = read_metrics(args.port, args.request_timeout)
        if metrics.get("statetree_evicted_total") != 2:
            raise RuntimeError("combined pressure eviction counter is not exactly two")
        if metrics.get("statetree_reclaimed_bytes_total") != checkpoint_bytes * 3:
            raise RuntimeError("combined pressure reclamation bytes are not exact")
        if metrics.get("statetree_state_high_water_bytes", budget + 1) > budget:
            raise RuntimeError("post-enforcement state high-water exceeded the exact budget")

        bench.erase_current_slots(args.port, list(range(args.parallel)), args.request_timeout)
        final = snapshot(args, proc.pid)
        assert_clean(read_slots(args.port, args.request_timeout))
        return {
            "budget_bytes": budget,
            "fork_id": family_id,
            "before": before,
            "after_family_victim": after_family,
            "at_budget": at_budget,
            "after_slot_victim": after_slot,
            "final": final,
            "metrics": metrics,
            "passed": True,
        }


def run_rejection_gate(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    budget = checkpoint_bytes - 1
    launch = server_args(args, f"{args.label}-rejection", 0, budget)
    with ManagedServer(launch, out_dir / "rejection.server.log") as proc:
        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "retention-rejection")
        request = bench.completion(
            args.port,
            prefix,
            0,
            0,
            args.seed,
            args.request_timeout,
        )
        measured = snapshot(args, proc.pid)
        rows = read_slots(args.port, args.request_timeout)
        metrics = read_metrics(args.port, args.request_timeout)
        if sum(state_bytes(rows, slot_id) for slot_id in range(args.parallel)) != 0:
            raise RuntimeError("failed admission retained optional checkpoint bytes")
        if metrics.get("statetree_pressure_rejected_total", 0) < 1:
            raise RuntimeError("failed admission did not increment pressure rejection")
        if metrics.get("statetree_state_high_water_bytes", budget + 1) > budget:
            raise RuntimeError("rejection lane exceeded the exact state budget")
        return {
            "budget_bytes": budget,
            "request": request,
            "snapshot": measured,
            "metrics": metrics,
            "passed": True,
        }


def run_pressure_width_gate(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for width in args.pressure_widths:
        full_width = width == args.parallel
        budget_checkpoints = width - 1 if full_width else width
        budget = checkpoint_bytes * budget_checkpoints
        launch = server_args(args, f"{args.label}-width-{width}", 0, budget)
        with ManagedServer(launch, out_dir / f"pressure-width-{width}.server.log") as proc:
            prefix = bench.exact_tokens(args.port, args.prefix_tokens, f"pressure-width-{width}")
            family_id: int | None = None

            if width == 1:
                bench.completion(args.port, prefix, 0, 0, args.seed, args.request_timeout)
            else:
                bench.completion(args.port, prefix, 0, 0, args.seed, args.request_timeout)
                members = list(range(width))
                fork = bench.fork_slot(args.port, 0, members[1:], args.request_timeout)
                family_id = fork["fork_id"]
                if type(family_id) is not int:
                    raise RuntimeError(f"width-{width} fork did not return a generation")
                for slot_id in members:
                    suffix = bench.exact_tokens(
                        args.port,
                        8,
                        f"pressure-width-{width}-member-{slot_id}",
                    )
                    bench.completion(
                        args.port,
                        prefix + suffix,
                        slot_id,
                        0,
                        args.seed + slot_id,
                        args.request_timeout,
                        fork_id=family_id,
                    )

            before = snapshot(args, proc.pid)
            rows = read_slots(args.port, args.request_timeout)
            exact_before = sum(state_bytes(rows, slot_id) for slot_id in range(args.parallel))
            metrics_before = read_metrics(args.port, args.request_timeout)

            if full_width:
                if exact_before != budget:
                    raise RuntimeError(
                        f"full-width pressure retained {exact_before} bytes, expected {budget}"
                    )
                if metrics_before.get("statetree_pressure_rejected_total", 0) < 1:
                    raise RuntimeError("full-width pressure did not reject the non-evictable checkpoint")
                if metrics_before.get("statetree_evicted_total", 0) != 0:
                    raise RuntimeError("full-width pressure evicted part of the active family")
                if sum(state_bytes(rows, slot_id) == 0 for slot_id in range(width)) != 1:
                    raise RuntimeError("full-width pressure did not skip exactly one checkpoint")
                after = before
            else:
                expected_before = checkpoint_bytes * width
                if exact_before != expected_before:
                    raise RuntimeError(
                        f"width-{width} fixture retained {exact_before} bytes, expected {expected_before}"
                    )
                trigger_id = width
                trigger = bench.exact_tokens(args.port, args.prefix_tokens, f"pressure-trigger-{width}")
                bench.completion(
                    args.port,
                    trigger,
                    trigger_id,
                    0,
                    args.seed + 100 + width,
                    args.request_timeout,
                )
                after = snapshot(args, proc.pid)
                rows = read_slots(args.port, args.request_timeout)
                if any(state_bytes(rows, slot_id) != 0 for slot_id in range(width)):
                    raise RuntimeError(f"width-{width} pressure did not reclaim every victim checkpoint")
                if any(rows[slot_id].get("is_reserved") for slot_id in range(width)):
                    raise RuntimeError(f"width-{width} pressure left a victim reservation")
                if state_bytes(rows, trigger_id) != checkpoint_bytes:
                    raise RuntimeError(f"width-{width} pressure damaged the admitted checkpoint")
                metrics_after = read_metrics(args.port, args.request_timeout)
                if metrics_after.get("statetree_evicted_total") != 1:
                    raise RuntimeError(f"width-{width} pressure did not record one atomic victim")
                if metrics_after.get("statetree_reclaimed_bytes_total") != expected_before:
                    raise RuntimeError(f"width-{width} pressure reclaimed the wrong byte count")

            metrics = read_metrics(args.port, args.request_timeout)
            if metrics.get("statetree_state_high_water_bytes", budget + 1) > budget:
                raise RuntimeError(f"width-{width} pressure exceeded its exact budget")
            bench.erase_current_slots(args.port, list(range(args.parallel)), args.request_timeout)
            final = snapshot(args, proc.pid)
            assert_clean(read_slots(args.port, args.request_timeout))
            results[str(width)] = {
                "width": width,
                "full_width": full_width,
                "budget_bytes": budget,
                "family_id": family_id,
                "before": before,
                "after": after,
                "final": final,
                "metrics": metrics,
                "passed": True,
            }

    return {
        "widths": args.pressure_widths,
        "results": results,
        "passed": True,
    }


def run_expiry_and_race_gate(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    launch = server_args(args, f"{args.label}-expiry", args.lease_ms, 0)
    expiry_samples: list[dict[str, Any]] = []
    race_samples: list[dict[str, Any]] = []
    with ManagedServer(launch, out_dir / "expiry-race.server.log") as proc:
        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "retention-expiry")
        suffix = bench.exact_tokens(args.port, 8, "retention-race")

        for repeat in range(args.expiry_repeats):
            assert_clean(read_slots(args.port, args.request_timeout))
            bench.completion(args.port, prefix, 0, 0, args.seed + repeat, args.request_timeout)
            fork = bench.fork_slot(args.port, 0, [1], args.request_timeout)
            retention = fork["response"].get("retention")
            if not isinstance(retention, dict) or type(retention.get("lease_remaining_ms")) is not int:
                raise RuntimeError("fork response is missing lease timing")
            received = time.monotonic()
            expected = received + retention["lease_remaining_ms"] / 1000.0
            _, observed = wait_clean(
                args.port,
                args.request_timeout,
                args.poll_ms,
            )
            jitter_ms = (observed - expected) * 1000.0
            if jitter_ms < -2.0 or jitter_ms > args.max_expiry_jitter_ms:
                raise RuntimeError(f"expiry jitter {jitter_ms:.3f} ms is outside the gate")
            expiry_samples.append({
                "repeat": repeat,
                "fork_id": fork["fork_id"],
                "lease_remaining_ms": retention["lease_remaining_ms"],
                "jitter_ms": jitter_ms,
            })

        offsets_ms = (-10.0, -2.0, 0.0, 2.0, 10.0)
        for repeat in range(args.race_repeats):
            assert_clean(read_slots(args.port, args.request_timeout))
            bench.completion(args.port, prefix, 0, 0, args.seed + 1000 + repeat, args.request_timeout)
            fork = bench.fork_slot(args.port, 0, [1], args.request_timeout)
            fork_id = fork["fork_id"]
            if type(fork_id) is not int:
                raise RuntimeError("race fork did not return a generation")
            retention = fork["response"].get("retention")
            remaining_ms = retention.get("lease_remaining_ms") if isinstance(retention, dict) else None
            if type(remaining_ms) is not int:
                raise RuntimeError("race fork response is missing lease timing")
            offset_ms = offsets_ms[repeat % len(offsets_ms)]
            time.sleep(max(0.0, (remaining_ms + offset_ms) / 1000.0))
            payload = {
                "prompt": prefix + suffix,
                "id_slot": 0,
                "fork_id": fork_id,
                "n_predict": 1,
                "temperature": 0.0,
                "top_k": 1,
                "cache_prompt": True,
                "ignore_eos": True,
                "stop": [],
            }
            try:
                body = bench.http_json(
                    "POST",
                    f"http://127.0.0.1:{args.port}/completion",
                    payload,
                    timeout=args.request_timeout,
                )
                outcome = "continued"
                if not isinstance(body, dict) or body.get("id_slot") != 0:
                    raise RuntimeError("successful boundary continuation used the wrong slot")
                rows = read_slots(args.port, args.request_timeout)
                if not all(rows[slot_id].get("fork_id") == fork_id for slot_id in (0, 1)):
                    raise RuntimeError("successful boundary continuation lost its generation")
                bench.erase_current_slots(args.port, [0, 1], args.request_timeout)
                assert_clean(read_slots(args.port, args.request_timeout))
            except bench.HttpStatusError as exc:
                if exc.status != 503:
                    raise
                outcome = "expired"
                wait_clean(args.port, args.request_timeout, args.poll_ms)
            race_samples.append({
                "repeat": repeat,
                "fork_id": fork_id,
                "offset_ms": offset_ms,
                "outcome": outcome,
            })

        outcomes = {sample["outcome"] for sample in race_samples}
        if outcomes != {"continued", "expired"}:
            raise RuntimeError(f"boundary race did not exercise both outcomes: {sorted(outcomes)}")
        final = snapshot(args, proc.pid)
        metrics = read_metrics(args.port, args.request_timeout)

    jitter_values = [sample["jitter_ms"] for sample in expiry_samples]
    return {
        "lease_ms": args.lease_ms,
        "expiry": {
            "samples": expiry_samples,
            "jitter_ms": bench.summarize_values(jitter_values),
            "max_allowed_ms": args.max_expiry_jitter_ms,
        },
        "race": {
            "samples": race_samples,
            "continued": sum(sample["outcome"] == "continued" for sample in race_samples),
            "expired": sum(sample["outcome"] == "expired" for sample in race_samples),
        },
        "final": final,
        "metrics": metrics,
        "passed": True,
    }


def run_churn_gate(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    budget = checkpoint_bytes * 3
    launch = server_args(args, f"{args.label}-churn", args.lease_ms, budget)
    samples: list[dict[str, Any]] = []
    with ManagedServer(launch, out_dir / "churn.server.log") as proc:
        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "retention-churn")
        suffixes = {
            slot_id: bench.exact_tokens(args.port, 8, f"retention-churn-branch-{slot_id}")
            for slot_id in (0, 1, 2)
        }
        for iteration in range(args.churn_iterations):
            assert_clean(read_slots(args.port, args.request_timeout))
            bench.completion(
                args.port,
                prefix,
                0,
                0,
                args.seed + 2000 + iteration,
                args.request_timeout,
            )
            fork = bench.fork_slot(args.port, 0, [1, 2], args.request_timeout)
            fork_id = fork["fork_id"]
            if type(fork_id) is not int:
                raise RuntimeError("churn fork did not return a generation")
            for slot_id in (0, 1, 2):
                bench.completion(
                    args.port,
                    prefix + suffixes[slot_id],
                    slot_id,
                    0,
                    args.seed + 2500 + iteration * 3 + slot_id,
                    args.request_timeout,
                    fork_id=fork_id,
                )

            if iteration % 5 == 0:
                wait_clean(args.port, args.request_timeout, args.poll_ms)
                outcome = "expired"
            else:
                winner = iteration % 3
                bench.completion(
                    args.port,
                    prefix + suffixes[winner],
                    winner,
                    1,
                    args.seed + 3000 + iteration,
                    args.request_timeout,
                    fork_id=fork_id,
                )
                bench.commit_slot(args.port, winner, fork_id, args.request_timeout)
                destinations = [slot_id for slot_id in (0, 1, 2) if slot_id != winner]
                refork = bench.fork_slot(
                    args.port,
                    winner,
                    destinations,
                    args.request_timeout,
                    fork_id=fork_id,
                )
                refork_id = refork["fork_id"]
                if type(refork_id) is not int or refork_id == fork_id:
                    raise RuntimeError("churn refork did not advance the generation")
                next_winner = destinations[iteration % len(destinations)]
                bench.commit_slot(args.port, next_winner, refork_id, args.request_timeout)
                bench.erase_current_slots(args.port, [0, 1, 2], args.request_timeout)
                assert_clean(read_slots(args.port, args.request_timeout))
                outcome = "committed"

            metrics = read_metrics(args.port, args.request_timeout)
            if metrics.get("statetree_state_bytes", budget + 1) != 0:
                raise RuntimeError(f"churn iteration {iteration} leaked prompt-state bytes")
            if metrics.get("statetree_state_high_water_bytes", budget + 1) > budget:
                raise RuntimeError(f"churn iteration {iteration} exceeded the state budget")
            samples.append({
                "iteration": iteration,
                "fork_id": fork_id,
                "outcome": outcome,
                "expired_total": metrics.get("statetree_expired_total"),
                "rejected_total": metrics.get("statetree_pressure_rejected_total"),
            })

        final = snapshot(args, proc.pid)
        metrics = read_metrics(args.port, args.request_timeout)
        assert_clean(read_slots(args.port, args.request_timeout))
        expected_expired = sum(sample["outcome"] == "expired" for sample in samples)
        if metrics.get("statetree_expired_total") != expected_expired:
            raise RuntimeError("churn expiry counter does not match completed expiry cycles")

    return {
        "iterations": args.churn_iterations,
        "budget_bytes": budget,
        "committed": sum(sample["outcome"] == "committed" for sample in samples),
        "expired": sum(sample["outcome"] == "expired" for sample in samples),
        "samples": samples,
        "final": final,
        "metrics": metrics,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate bounded StateTree retention under expiry and pressure.")
    parser.add_argument("--bin", default=bench.DEFAULT_BIN)
    parser.add_argument("--model", default=bench.DEFAULT_MODEL)
    parser.add_argument("--label", default="statetree-retention")
    parser.add_argument("--commit", default="dirty")
    parser.add_argument("--out-dir", default="/tmp/turbo-statetree-retention-gate")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--ctx", type=bench.positive_int, default=2048)
    parser.add_argument("--parallel", type=bench.positive_int, default=4)
    parser.add_argument("--prefix-tokens", type=bench.positive_int, default=128)
    parser.add_argument("--lease-ms", type=bench.positive_int, default=100)
    parser.add_argument("--expiry-repeats", type=bench.positive_int, default=20)
    parser.add_argument("--race-repeats", type=bench.positive_int, default=20)
    parser.add_argument("--churn-iterations", type=bench.positive_int, default=50)
    parser.add_argument("--pressure-widths", type=bench.parse_csv_ints, default=[])
    parser.add_argument("--only-pressure-widths", action="store_true")
    parser.add_argument("--poll-ms", type=float, default=2.0)
    parser.add_argument("--max-expiry-jitter-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--batch", type=bench.positive_int, default=512)
    parser.add_argument("--ubatch", type=bench.positive_int, default=256)
    parser.add_argument("--threads", type=bench.positive_int, default=16)
    parser.add_argument("--ngl", type=bench.non_negative_int, default=0)
    parser.add_argument("--ncmoe", type=bench.non_negative_int, default=0)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="auto")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--source-oneapi", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.port == 8093:
        raise SystemExit("refusing production port 8093; use an isolated managed server")
    if args.parallel < 4:
        raise SystemExit("retention pressure and churn gates require at least four slots")
    if any(width > args.parallel for width in args.pressure_widths):
        raise SystemExit("pressure widths cannot exceed the configured parallel slots")
    if args.pressure_widths != sorted(args.pressure_widths):
        raise SystemExit("pressure widths must be in ascending order")
    if args.only_pressure_widths and not args.pressure_widths:
        raise SystemExit("--only-pressure-widths requires --pressure-widths")
    if args.poll_ms <= 0 or args.max_expiry_jitter_ms <= 0:
        raise SystemExit("poll and jitter thresholds must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{args.label}.result.json"
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result: dict[str, Any] = {
        "kind": "turbo-statetree-retention-gate",
        "schema_version": 1,
        "started": started,
        "label": args.label,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"bin", "model"}
        },
        "server": {
            "commit": args.commit,
            "binary_identity": bench.file_identity(args.bin, include_sha256=True),
            "bundled_runtime_identities": bench.bundled_runtime_identities(args.bin),
            "cmake_cache": bench.cmake_cache_identity(args.bin),
            "model_identity": bench.file_identity(args.model),
        },
        "failures": [],
    }

    try:
        result["probe"] = probe_checkpoint_bytes(args, out_dir)
        checkpoint_bytes = result["probe"]["checkpoint_bytes"]
        if args.pressure_widths:
            result["pressure_widths"] = run_pressure_width_gate(args, out_dir, checkpoint_bytes)
        if not args.only_pressure_widths:
            result["pressure"] = run_pressure_gate(args, out_dir, checkpoint_bytes)
            result["rejection"] = run_rejection_gate(args, out_dir, checkpoint_bytes)
            result["expiry_and_race"] = run_expiry_and_race_gate(args, out_dir)
            result["churn"] = run_churn_gate(args, out_dir, checkpoint_bytes)
    except Exception as exc:  # noqa: BLE001
        result["failures"].append(str(exc))

    result["passed"] = not result["failures"]
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "failures": result["failures"],
        "result": str(result_path),
    }, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
