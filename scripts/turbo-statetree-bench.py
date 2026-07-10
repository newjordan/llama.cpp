#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BIN = "/home/frosty40/turbo/turbo-combined/build/bin/llama-server"
DEFAULT_MODEL = "/home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf"
DEFAULT_OUT_DIR = "/tmp/turbo-statetree-bench"
RESULT_SCHEMA_VERSION = 1


class HttpStatusError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def parse_csv_ints(value: str) -> list[int]:
    values = [positive_int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one positive integer")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("duplicate integers are not allowed")
    return values


def parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("expected at least one cleanup mode")
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("duplicate cleanup modes are not allowed")
    unknown = sorted(set(modes) - {"manual", "commit"})
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown cleanup modes: {', '.join(unknown)}")
    return modes


def resolve_family_ids(parallel: int, fanout: int, layout: str) -> list[int]:
    count = fanout + 1
    if layout == "dense":
        if count > parallel:
            raise ValueError("fanout must be smaller than the number of parallel slots")
        return list(range(count))
    family_ids = list(range(1, parallel, 2))[:count]
    if len(family_ids) != count:
        raise ValueError("fragmented layout needs at least 2 * (fanout + 1) slots")
    return family_ids


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
    }


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpStatusError(exc.code, body) from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def http_text(url: str, timeout: float = 30.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HttpStatusError(exc.code, body) from exc


def timed_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    body = http_json(method, url, payload, timeout)
    return {
        "client_ms": (time.perf_counter() - started) * 1000.0,
        "response": body,
    }


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_healthy(port: int, proc: subprocess.Popen[Any] | None, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy on port {port}: {last_error}")


def stop_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def launch_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[Any]:
    if port_open(args.port):
        raise RuntimeError(f"port {args.port} is already listening")

    command = [
        args.bin,
        "-m", args.model,
        "-ngl", str(args.ngl),
        "--no-op-offload",
        "-c", str(args.ctx),
        "-np", str(args.parallel),
        "-kvu",
        "-fa", args.flash_attn,
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--slots",
        "--metrics",
        "--no-cache-idle-slots",
        "-a", f"turbo-statetree-bench-{args.label}",
    ]
    if args.extra_server_args:
        command.extend(shlex.split(args.extra_server_args))

    launch_command = command
    if args.source_oneapi:
        launch_command = [
            "bash",
            "-lc",
            "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec " + shlex.join(command),
        ]

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(launch_command, stdout=log_file, stderr=subprocess.STDOUT, env=os.environ.copy())
    proc._turbo_log_file = log_file  # type: ignore[attr-defined]
    return proc


def read_process_memory(pid: int | None) -> dict[str, int] | None:
    if pid is None:
        return None
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    wanted = {
        "VmRSS": "rss_bytes",
        "VmHWM": "rss_hwm_bytes",
        "VmSize": "virtual_bytes",
        "RssAnon": "rss_anon_bytes",
    }
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, rest = line.partition(":")
        if key not in wanted:
            continue
        parts = rest.strip().split()
        if parts:
            result[wanted[key]] = int(parts[0]) * 1024
    return result


def parse_prometheus(body: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in body.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        name_and_labels, separator, value = line.rpartition(" ")
        if not separator:
            continue
        name = name_and_labels[len("llamacpp:") :].split("{", 1)[0]
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def optional_sum(rows: list[dict[str, Any]], key: str) -> int | None:
    if any(key not in row for row in rows):
        return None
    return sum(int(row[key]) for row in rows)


def summarize_slots(rows: list[dict[str, Any]]) -> dict[str, Any]:
    byte_keys = (
        "n_prompt_data_bytes",
        "n_prompt_checkpoint_bytes",
        "n_prompt_state_bytes",
    )
    selected_keys = (
        "id",
        "is_processing",
        "is_reserved",
        "fork_source_id",
        "fork_id",
        "n_prompt_checkpoints",
        "n_prompt_tokens",
        *byte_keys,
    )
    by_id = {
        str(int(row["id"])): {key: row.get(key) for key in selected_keys if key in row}
        for row in rows
    }
    return {
        "slots": len(rows),
        "processing": sum(bool(row.get("is_processing")) for row in rows),
        "reserved": sum(bool(row.get("is_reserved")) for row in rows),
        "prompt_tokens": sum(int(row.get("n_prompt_tokens") or 0) for row in rows),
        "prompt_checkpoints": sum(int(row.get("n_prompt_checkpoints") or 0) for row in rows),
        "prompt_data_bytes": optional_sum(rows, "n_prompt_data_bytes"),
        "prompt_checkpoint_bytes": optional_sum(rows, "n_prompt_checkpoint_bytes"),
        "prompt_state_bytes": optional_sum(rows, "n_prompt_state_bytes"),
        "by_id": by_id,
    }


def capture_snapshot(port: int, pid: int | None, family_ids: list[int]) -> dict[str, Any]:
    body = http_json("GET", f"http://127.0.0.1:{port}/slots", timeout=60.0)
    if not isinstance(body, list) or any(not isinstance(row, dict) for row in body):
        raise RuntimeError("slots response is not an array of objects")
    rows = list(body)
    family = [row for row in rows if int(row.get("id", -1)) in family_ids]
    metrics = parse_prometheus(http_text(f"http://127.0.0.1:{port}/metrics"))
    return {
        "captured_at": time.time(),
        "all_slots": summarize_slots(rows),
        "family_slots": summarize_slots(family),
        "process": read_process_memory(pid),
        "server_metrics": {
            key: metrics.get(key)
            for key in (
                "kv_cache_usage_ratio",
                "requests_processing",
                "requests_deferred",
                "requests_reserved",
            )
            if key in metrics
        },
    }


def tokenize(port: int, content: str) -> list[int]:
    body = http_json(
        "POST",
        f"http://127.0.0.1:{port}/tokenize",
        {"content": content, "add_special": True, "parse_special": False},
        timeout=120.0,
    )
    if not isinstance(body, dict) or not isinstance(body.get("tokens"), list):
        raise RuntimeError("tokenize response is missing tokens")
    tokens = body["tokens"]
    if any(type(token) is not int for token in tokens):
        raise RuntimeError("tokenize response contains non-integer tokens")
    return tokens


def exact_tokens(port: int, count: int, salt: str) -> list[int]:
    unit = f" {salt} StateTree benchmark 0123456789 abcdefghijklmnopqrstuvwxyz."
    repeats = max(8, count // 4)
    for _ in range(8):
        tokens = tokenize(port, unit * repeats)
        if len(tokens) >= count:
            return tokens[:count]
        repeats *= 2
    raise RuntimeError(f"failed to tokenize at least {count} tokens")


def completion(
    port: int,
    prompt: list[int],
    slot_id: int,
    n_predict: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "id_slot": slot_id,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": True,
        "ignore_eos": True,
        "stop": [],
        "stream": False,
        "return_tokens": True,
    }
    timed = timed_json(
        "POST",
        f"http://127.0.0.1:{port}/completion",
        payload,
        timeout=timeout,
    )
    body = timed["response"]
    if not isinstance(body, dict):
        raise RuntimeError("completion response is not an object")
    if body.get("id_slot") != slot_id:
        raise RuntimeError(f"completion used slot {body.get('id_slot')!r}, expected {slot_id}")
    timings = body.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("completion response is missing timings")
    tokens = body.get("tokens", [])
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise RuntimeError("completion response is missing integer tokens")
    predicted_n = int(timings.get("predicted_n") or 0)
    if len(tokens) != predicted_n:
        raise RuntimeError(f"completion returned {len(tokens)} tokens, timings reported {predicted_n}")
    return {
        "client_ms": timed["client_ms"],
        "slot_id": slot_id,
        "prompt_n": int(timings.get("prompt_n") or 0),
        "cache_n": int(timings.get("cache_n") or 0),
        "predicted_n": predicted_n,
        "prompt_tps": float(timings["prompt_per_second"]) if timings.get("prompt_per_second") is not None else None,
        "predicted_tps": float(timings["predicted_per_second"]) if timings.get("predicted_per_second") is not None else None,
        "tokens_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("ascii")).hexdigest(),
    }


def fork_slot(
    port: int,
    source_id: int,
    destinations: list[int],
    timeout: float,
    fork_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"destinations": destinations}
    if fork_id is not None:
        payload["fork_id"] = fork_id
    timed = timed_json(
        "POST",
        f"http://127.0.0.1:{port}/slots/{source_id}?action=fork",
        payload,
        timeout=timeout,
    )
    body = timed["response"]
    if not isinstance(body, dict):
        raise RuntimeError("fork response is not an object")
    if body.get("id_slot") != source_id or body.get("destinations") != destinations:
        raise RuntimeError("fork response does not match requested slots")
    return {
        "client_ms": timed["client_ms"],
        "server_ms": float(body.get("timings", {}).get("fork_ms") or 0.0),
        "fork_id": body.get("fork_id"),
        "n_tokens": int(body.get("n_tokens") or 0),
        "response": body,
    }


def commit_slot(port: int, winner_id: int, fork_id: int, timeout: float) -> dict[str, Any]:
    timed = timed_json(
        "POST",
        f"http://127.0.0.1:{port}/slots/{winner_id}?action=commit",
        {"fork_id": fork_id},
        timeout=timeout,
    )
    body = timed["response"]
    if not isinstance(body, dict):
        raise RuntimeError("commit response is not an object")
    return {
        "client_ms": timed["client_ms"],
        "server_ms": float(body.get("timings", {}).get("commit_ms") or 0.0),
        "response": body,
    }


def erase_slot(port: int, slot_id: int, timeout: float) -> dict[str, Any]:
    return timed_json(
        "POST",
        f"http://127.0.0.1:{port}/slots/{slot_id}?action=erase",
        {},
        timeout=timeout,
    )


def erase_slots(port: int, slot_ids: list[int], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    operations = []
    for slot_id in slot_ids:
        operation = erase_slot(port, slot_id, timeout)
        operations.append({"slot_id": slot_id, "client_ms": operation["client_ms"]})
    return {
        "client_ms": (time.perf_counter() - started) * 1000.0,
        "operations": operations,
    }


def branch_wave(
    port: int,
    prefix: list[int],
    suffixes: dict[int, list[int]],
    family_ids: list[int],
    n_predict: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(len(family_ids))

    def one(slot_id: int) -> dict[str, Any]:
        barrier.wait()
        return completion(
            port,
            prefix + suffixes[slot_id],
            slot_id,
            n_predict,
            seed + slot_id,
            timeout,
        )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(family_ids)) as executor:
        requests = list(executor.map(one, family_ids))
    wall_s = time.perf_counter() - started
    predicted = sum(int(request["predicted_n"]) for request in requests)
    return {
        "wall_ms": wall_s * 1000.0,
        "aggregate_predicted_tps": predicted / wall_s if wall_s else None,
        "total_predicted_tokens": predicted,
        "min_cache_n": min(int(request["cache_n"]) for request in requests),
        "requests": requests,
    }


def slot_row(snapshot: dict[str, Any], slot_id: int) -> dict[str, Any]:
    return snapshot["all_slots"]["by_id"].get(str(slot_id), {})


def expected_shared_kv_cells(snapshot: dict[str, Any], prefix_tokens: int, family_ids: list[int]) -> int:
    prompt_lengths = [
        int(slot_row(snapshot, slot_id).get("n_prompt_tokens") or 0)
        for slot_id in family_ids
    ]
    active = [length for length in prompt_lengths if length > 0]
    if not active:
        return 0
    shared = min(prefix_tokens, min(active))
    return shared + sum(max(0, length - shared) for length in active)


def validate_sample(sample: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    family_ids = sample["family_ids"]
    prefix_tokens = sample["prefix_tokens"]
    snapshots = sample["snapshots"]

    if snapshots["after_fork"]["family_slots"]["reserved"] != len(family_ids):
        failures.append("fork did not reserve the exact family")
    if sample["branch_wave"]["min_cache_n"] < prefix_tokens - 1:
        failures.append("branch cache reuse fell below the shared prefix")

    if not sample.get("supported", True):
        return failures

    winner_id = sample["winner_id"]
    before = slot_row(snapshots["before_cleanup"], winner_id)
    after = slot_row(snapshots["after_cleanup"], winner_id)
    if snapshots["after_cleanup"]["family_slots"]["reserved"] != 1:
        failures.append("cleanup did not leave exactly one protected winner")
    for key in ("n_prompt_tokens", "n_prompt_checkpoints", "n_prompt_state_bytes"):
        if key in before and before.get(key) != after.get(key):
            failures.append(f"winner {key} changed across cleanup")

    if sample["cleanup_mode"] == "commit":
        released = sample["cleanup"]["response"].get("released")
        expected = sorted(slot_id for slot_id in family_ids if slot_id != winner_id)
        if sorted(released or []) != expected:
            failures.append("commit released the wrong slot set")
        for slot_id in expected:
            loser = slot_row(snapshots["after_cleanup"], slot_id)
            if int(loser.get("n_prompt_tokens") or 0) != 0:
                failures.append(f"loser slot {slot_id} retained prompt tokens")
            if int(loser.get("n_prompt_state_bytes") or 0) != 0:
                failures.append(f"loser slot {slot_id} retained prompt state bytes")

    refork = sample.get("refork")
    if refork and isinstance(refork.get("fork_id"), int):
        if refork["fork_id"] == sample["fork"]["fork_id"]:
            failures.append("refork did not advance the generation")

    final_snapshot = snapshots.get("final")
    if final_snapshot and final_snapshot["family_slots"]["reserved"] != 0:
        failures.append("final cleanup left reserved slots")
    return failures


def run_sample(
    args: argparse.Namespace,
    prefix: list[int],
    suffixes: dict[int, list[int]],
    filler_prompts: dict[int, list[int]],
    family_ids: list[int],
    repeat: int,
    cleanup_mode: str,
    pid: int | None,
) -> dict[str, Any]:
    source_id = family_ids[0]
    destinations = family_ids[1:]
    winner_id = source_id if cleanup_mode == "manual" else family_ids[-1]
    sample: dict[str, Any] = {
        "repeat": repeat,
        "prefix_tokens": len(prefix),
        "cleanup_mode": cleanup_mode,
        "family_ids": family_ids,
        "winner_id": winner_id,
        "layout": args.layout,
        "supported": True,
        "snapshots": {},
    }

    erase_slots(args.port, list(range(args.parallel)), args.request_timeout)
    try:
        if args.layout == "fragmented":
            fill_requests = []
            for slot_id in range(args.parallel):
                fill_requests.append(completion(
                    args.port,
                    filler_prompts[slot_id],
                    slot_id,
                    0,
                    args.seed + repeat * 1000 + 500 + slot_id,
                    args.request_timeout,
                ))
            erase_slots(args.port, family_ids, args.request_timeout)
            sample["fragmentation"] = {
                "fill_tokens": args.fragment_fill_tokens,
                "survivor_slots": [slot_id for slot_id in range(args.parallel) if slot_id not in family_ids],
                "fill_requests": fill_requests,
            }
            sample["snapshots"]["fragmented_layout"] = capture_snapshot(args.port, pid, family_ids)
        else:
            sample["fragmentation"] = {"fill_tokens": 0, "survivor_slots": [], "fill_requests": []}
        sample["snapshots"]["before_prefill"] = capture_snapshot(args.port, pid, family_ids)
        sample["prefill"] = completion(
            args.port,
            prefix,
            source_id,
            0,
            args.seed + repeat * 1000,
            args.request_timeout,
        )
        sample["snapshots"]["after_prefill"] = capture_snapshot(args.port, pid, family_ids)
        sample["fork"] = fork_slot(
            args.port,
            source_id,
            destinations,
            args.request_timeout,
        )
        sample["snapshots"]["after_fork"] = capture_snapshot(args.port, pid, family_ids)

        fork_id = sample["fork"].get("fork_id")
        if cleanup_mode == "commit" and not isinstance(fork_id, int):
            sample["supported"] = False
            sample["unsupported_reason"] = "fork response has no generation; commit is unavailable"
            sample["branch_wave"] = branch_wave(
                args.port,
                prefix,
                suffixes,
                family_ids,
                args.branch_tokens,
                args.seed + repeat * 1000 + 100,
                args.request_timeout,
            )
            sample["snapshots"]["before_cleanup"] = capture_snapshot(args.port, pid, family_ids)
            sample["failures"] = validate_sample(sample)
            return sample

        sample["branch_wave"] = branch_wave(
            args.port,
            prefix,
            suffixes,
            family_ids,
            args.branch_tokens,
            args.seed + repeat * 1000 + 100,
            args.request_timeout,
        )
        sample["snapshots"]["before_cleanup"] = capture_snapshot(args.port, pid, family_ids)
        sample["expected_shared_kv_cells_before_cleanup"] = expected_shared_kv_cells(
            sample["snapshots"]["before_cleanup"], len(prefix), family_ids
        )

        if cleanup_mode == "commit":
            sample["cleanup"] = commit_slot(args.port, winner_id, fork_id, args.request_timeout)
        else:
            losers = [slot_id for slot_id in family_ids if slot_id != winner_id]
            sample["cleanup"] = erase_slots(args.port, losers, args.request_timeout)
        sample["snapshots"]["after_cleanup"] = capture_snapshot(args.port, pid, family_ids)
        sample["expected_shared_kv_cells_after_cleanup"] = expected_shared_kv_cells(
            sample["snapshots"]["after_cleanup"], len(prefix), family_ids
        )

        if isinstance(fork_id, int):
            freed = [slot_id for slot_id in family_ids if slot_id != winner_id]
            sample["refork"] = fork_slot(
                args.port,
                winner_id,
                freed,
                args.request_timeout,
                fork_id=fork_id,
            )
            sample["snapshots"]["after_refork"] = capture_snapshot(args.port, pid, family_ids)
            if cleanup_mode == "commit":
                next_fork_id = sample["refork"].get("fork_id")
                if not isinstance(next_fork_id, int):
                    raise RuntimeError("refork response is missing a generation")
                sample["second_cleanup"] = commit_slot(
                    args.port,
                    winner_id,
                    next_fork_id,
                    args.request_timeout,
                )
            else:
                sample["second_cleanup"] = erase_slots(
                    args.port,
                    [slot_id for slot_id in family_ids if slot_id != winner_id],
                    args.request_timeout,
                )
        else:
            sample["refork"] = None
            sample["refork_unsupported_reason"] = "parent fork families cannot be reforked in place"

        erase_slots(args.port, family_ids, args.request_timeout)
        sample["snapshots"]["final"] = capture_snapshot(args.port, pid, family_ids)
        sample["failures"] = validate_sample(sample)
        return sample
    finally:
        erase_slots(args.port, list(range(args.parallel)), args.request_timeout)


def nested_value(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "prefill_client_ms": "prefill.client_ms",
        "prefill_prompt_tps": "prefill.prompt_tps",
        "fork_client_ms": "fork.client_ms",
        "fork_server_ms": "fork.server_ms",
        "branch_wall_ms": "branch_wave.wall_ms",
        "branch_aggregate_predicted_tps": "branch_wave.aggregate_predicted_tps",
        "cleanup_client_ms": "cleanup.client_ms",
        "cleanup_server_ms": "cleanup.server_ms",
        "refork_client_ms": "refork.client_ms",
        "refork_server_ms": "refork.server_ms",
        "rss_before_cleanup_bytes": "snapshots.before_cleanup.process.rss_bytes",
        "rss_after_cleanup_bytes": "snapshots.after_cleanup.process.rss_bytes",
        "state_before_cleanup_bytes": "snapshots.before_cleanup.family_slots.prompt_state_bytes",
        "state_after_cleanup_bytes": "snapshots.after_cleanup.family_slots.prompt_state_bytes",
        "checkpoint_before_cleanup_bytes": "snapshots.before_cleanup.family_slots.prompt_checkpoint_bytes",
        "checkpoint_after_cleanup_bytes": "snapshots.after_cleanup.family_slots.prompt_checkpoint_bytes",
        "expected_shared_kv_cells_before_cleanup": "expected_shared_kv_cells_before_cleanup",
        "expected_shared_kv_cells_after_cleanup": "expected_shared_kv_cells_after_cleanup",
    }
    groups: dict[str, Any] = {}
    group_keys = sorted({(sample["prefix_tokens"], sample["cleanup_mode"]) for sample in samples})
    for prefix_tokens, cleanup_mode in group_keys:
        selected = [
            sample
            for sample in samples
            if sample["prefix_tokens"] == prefix_tokens and sample["cleanup_mode"] == cleanup_mode
        ]
        usable = [sample for sample in selected if sample.get("supported", True) and not sample.get("error")]
        metrics: dict[str, Any] = {}
        for name, path in metric_paths.items():
            values = [nested_value(sample, path) for sample in usable]
            metrics[name] = summarize_values([float(value) for value in values if isinstance(value, (int, float))])
        groups[f"p{prefix_tokens}:{cleanup_mode}"] = {
            "prefix_tokens": prefix_tokens,
            "cleanup_mode": cleanup_mode,
            "samples": len(selected),
            "usable_samples": len(usable),
            "unsupported_samples": sum(not sample.get("supported", True) for sample in selected),
            "failed_samples": sum(bool(sample.get("error") or sample.get("failures")) for sample in selected),
            "metrics": metrics,
        }
    return {
        "groups": groups,
        "samples": len(samples),
        "supported_samples": sum(sample.get("supported", True) for sample in samples),
        "failed_samples": sum(bool(sample.get("error") or sample.get("failures")) for sample in samples),
        "commit_supported": any(
            sample["cleanup_mode"] == "commit" and sample.get("supported", True)
            for sample in samples
        ),
    }


def metric_p50(result: dict[str, Any], group: str, metric: str) -> float | None:
    value = nested_value(result, f"summary.groups.{group}.metrics.{metric}.p50")
    return float(value) if isinstance(value, (int, float)) else None


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    throughput_floor: float,
    latency_ratio: float,
    latency_slack_ms: float,
    rss_slack_mib: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    baseline_groups = baseline.get("summary", {}).get("groups", {})
    candidate_groups = candidate.get("summary", {}).get("groups", {})
    common_manual = sorted(set(baseline_groups) & set(candidate_groups))
    common_manual = [group for group in common_manual if group.endswith(":manual")]

    for group in common_manual:
        base_fork = metric_p50(baseline, group, "fork_server_ms")
        cand_fork = metric_p50(candidate, group, "fork_server_ms")
        if base_fork is not None and cand_fork is not None:
            limit = base_fork * latency_ratio + latency_slack_ms
            checks.append({
                "group": group,
                "metric": "fork_server_ms",
                "baseline": base_fork,
                "candidate": cand_fork,
                "limit": limit,
                "passed": cand_fork <= limit,
            })

        base_tps = metric_p50(baseline, group, "branch_aggregate_predicted_tps")
        cand_tps = metric_p50(candidate, group, "branch_aggregate_predicted_tps")
        if base_tps is not None and cand_tps is not None:
            limit = base_tps * throughput_floor
            checks.append({
                "group": group,
                "metric": "branch_aggregate_predicted_tps",
                "baseline": base_tps,
                "candidate": cand_tps,
                "limit": limit,
                "passed": cand_tps >= limit,
            })

        base_rss = metric_p50(baseline, group, "rss_before_cleanup_bytes")
        cand_rss = metric_p50(candidate, group, "rss_before_cleanup_bytes")
        if base_rss is not None and cand_rss is not None:
            limit = base_rss + rss_slack_mib * 1024.0 * 1024.0
            checks.append({
                "group": group,
                "metric": "rss_before_cleanup_bytes",
                "baseline": base_rss,
                "candidate": cand_rss,
                "limit": limit,
                "passed": cand_rss <= limit,
            })

    commit_deltas: list[dict[str, Any]] = []
    for group in sorted(candidate_groups):
        if not group.endswith(":commit"):
            continue
        manual_group = group.removesuffix(":commit") + ":manual"
        commit_ms = metric_p50(candidate, group, "cleanup_client_ms")
        manual_ms = metric_p50(candidate, manual_group, "cleanup_client_ms")
        if commit_ms is not None and manual_ms is not None:
            commit_deltas.append({
                "prefix_tokens": candidate_groups[group]["prefix_tokens"],
                "commit_client_ms": commit_ms,
                "manual_client_ms": manual_ms,
                "speedup": manual_ms / commit_ms if commit_ms else None,
            })

    return {
        "kind": "turbo-statetree-benchmark-comparison",
        "schema_version": RESULT_SCHEMA_VERSION,
        "baseline": {
            "label": baseline.get("label"),
            "commit": baseline.get("server", {}).get("commit"),
        },
        "candidate": {
            "label": candidate.get("label"),
            "commit": candidate.get("server", {}).get("commit"),
        },
        "thresholds": {
            "throughput_floor": throughput_floor,
            "latency_ratio": latency_ratio,
            "latency_slack_ms": latency_slack_ms,
            "rss_slack_mib": rss_slack_mib,
        },
        "checks": checks,
        "commit_vs_manual": commit_deltas,
        "candidate_commit_supported": bool(candidate.get("summary", {}).get("commit_supported")),
        "passed": bool(checks)
            and all(check["passed"] for check in checks)
            and baseline.get("summary", {}).get("failed_samples", 0) == 0
            and candidate.get("summary", {}).get("failed_samples", 0) == 0
            and bool(candidate.get("summary", {}).get("commit_supported")),
    }


def server_version(args: argparse.Namespace) -> str | None:
    command = [args.bin, "--version"]
    if args.source_oneapi:
        command = [
            "bash",
            "-lc",
            "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; " + shlex.join(command),
        ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout + completed.stderr).strip()
    return output[:2000] or None


def run_benchmark(args: argparse.Namespace) -> int:
    if args.attach and not args.allow_destructive_attach:
        raise SystemExit("--attach erases slot state; add --allow-destructive-attach for an isolated server")
    if args.port == 8093:
        raise SystemExit("refusing production port 8093; use an isolated managed server")
    try:
        family_ids = resolve_family_ids(args.parallel, args.fanout, args.layout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dense_budget = max(args.prefix_tokens) + (args.branch_suffix_tokens + args.branch_tokens) * len(family_ids)
    fragmented_budget = max(
        args.parallel * args.fragment_fill_tokens,
        (args.parallel - len(family_ids)) * args.fragment_fill_tokens + dense_budget,
    ) if args.layout == "fragmented" else dense_budget
    if fragmented_budget > args.ctx:
        raise SystemExit("workload exceeds the unified KV context budget")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.label}.server.log"
    samples_path = out_dir / f"{args.label}.samples.jsonl"
    result_path = out_dir / f"{args.label}.result.json"
    proc: subprocess.Popen[Any] | None = None
    samples: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    try:
        if args.attach:
            wait_healthy(args.port, None, args.startup_timeout)
            pid = args.server_pid
        else:
            proc = launch_server(args, log_path)
            wait_healthy(args.port, proc, args.startup_timeout)
            pid = proc.pid

        erase_slots(args.port, list(range(args.parallel)), args.request_timeout)
        warmup_tokens = exact_tokens(args.port, min(32, min(args.prefix_tokens)), "warmup")
        completion(args.port, warmup_tokens, 0, min(2, args.branch_tokens), args.seed, args.request_timeout)
        erase_slots(args.port, list(range(args.parallel)), args.request_timeout)

        prefix_pool = exact_tokens(args.port, max(args.prefix_tokens), "prefix")
        suffixes = {
            slot_id: exact_tokens(args.port, args.branch_suffix_tokens, f"branch-{slot_id}")
            if args.branch_suffix_tokens else []
            for slot_id in family_ids
        }
        filler_prompts = {
            slot_id: exact_tokens(args.port, args.fragment_fill_tokens, f"filler-{slot_id}")
            for slot_id in range(args.parallel)
        } if args.layout == "fragmented" else {}

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for repeat in range(args.repeats):
                for prefix_index, prefix_count in enumerate(args.prefix_tokens):
                    modes = list(args.modes)
                    if (repeat + prefix_index) % 2:
                        modes.reverse()
                    for cleanup_mode in modes:
                        print(
                            f"repeat={repeat} prefix={prefix_count} mode={cleanup_mode}",
                            flush=True,
                        )
                        try:
                            sample = run_sample(
                                args,
                                prefix_pool[:prefix_count],
                                suffixes,
                                filler_prompts,
                                family_ids,
                                repeat,
                                cleanup_mode,
                                pid,
                            )
                        except Exception as exc:  # noqa: BLE001
                            sample = {
                                "repeat": repeat,
                                "prefix_tokens": prefix_count,
                                "cleanup_mode": cleanup_mode,
                                "supported": True,
                                "error": str(exc),
                                "failures": [],
                            }
                        samples.append(sample)
                        sample_file.write(json.dumps(sample, sort_keys=True) + "\n")
                        sample_file.flush()
    finally:
        if not args.attach:
            stop_process(proc)
            log_file = getattr(proc, "_turbo_log_file", None)
            if log_file is not None:
                log_file.close()

    summary = summarize_samples(samples)
    result = {
        "kind": "turbo-statetree-benchmark",
        "schema_version": RESULT_SCHEMA_VERSION,
        "started": started,
        "label": args.label,
        "config": {
            "bin": args.bin,
            "model": args.model,
            "port": args.port,
            "ctx": args.ctx,
            "parallel": args.parallel,
            "fanout": args.fanout,
            "family_ids": family_ids,
            "layout": args.layout,
            "fragment_fill_tokens": args.fragment_fill_tokens,
            "prefix_tokens": args.prefix_tokens,
            "branch_suffix_tokens": args.branch_suffix_tokens,
            "branch_tokens": args.branch_tokens,
            "repeats": args.repeats,
            "modes": args.modes,
            "ngl": args.ngl,
            "attach": args.attach,
        },
        "server": {
            "version": server_version(args),
            "commit": args.commit,
            "log_path": str(log_path) if not args.attach else None,
        },
        "summary": summary,
        "samples": samples,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"result: {result_path}")
    print(f"samples: {samples_path}")

    failed = summary["failed_samples"] > 0
    missing_commit = args.require_commit and not summary["commit_supported"]
    return 1 if args.fail_on_error and (failed or missing_commit) else 0


def compare_benchmarks(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    comparison = compare_results(
        baseline,
        candidate,
        args.throughput_floor,
        args.latency_ratio,
        args.latency_slack_ms,
        args.rss_slack_mib,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 1 if args.fail_on_regression and not comparison["passed"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Turbo StateTree fork, commit, reclamation, and refork.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an isolated StateTree benchmark")
    run.add_argument("--bin", default=DEFAULT_BIN)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--label", required=True)
    run.add_argument("--commit", default="dirty")
    run.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    run.add_argument("--port", type=int, default=8098)
    run.add_argument("--ctx", type=positive_int, default=32768)
    run.add_argument("--parallel", type=positive_int, default=4)
    run.add_argument("--fanout", type=positive_int, default=3)
    run.add_argument("--layout", choices=("dense", "fragmented"), default="dense")
    run.add_argument("--fragment-fill-tokens", type=positive_int, default=128)
    run.add_argument("--prefix-tokens", type=parse_csv_ints, default=parse_csv_ints("128,1024,8192"))
    run.add_argument("--branch-suffix-tokens", type=non_negative_int, default=8)
    run.add_argument("--branch-tokens", type=positive_int, default=8)
    run.add_argument("--repeats", type=positive_int, default=5)
    run.add_argument("--modes", type=parse_modes, default=parse_modes("manual,commit"))
    run.add_argument("--seed", type=int, default=1709)
    run.add_argument("--batch", type=positive_int, default=2048)
    run.add_argument("--ubatch", type=positive_int, default=512)
    run.add_argument("--threads", type=positive_int, default=16)
    run.add_argument("--ngl", type=non_negative_int, default=0)
    run.add_argument("--cache-type-k", default="f16")
    run.add_argument("--cache-type-v", default="f16")
    run.add_argument("--flash-attn", choices=("on", "off", "auto"), default="auto")
    run.add_argument("--request-timeout", type=float, default=900.0)
    run.add_argument("--startup-timeout", type=float, default=900.0)
    run.add_argument("--extra-server-args", default="")
    run.add_argument("--source-oneapi", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--attach", action="store_true")
    run.add_argument("--server-pid", type=positive_int)
    run.add_argument("--allow-destructive-attach", action="store_true")
    run.add_argument("--require-commit", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument("--fail-on-error", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(func=run_benchmark)

    compare = subparsers.add_parser("compare", help="Compare parent and candidate result files")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--out")
    compare.add_argument("--throughput-floor", type=float, default=0.95)
    compare.add_argument("--latency-ratio", type=float, default=1.10)
    compare.add_argument("--latency-slack-ms", type=float, default=0.25)
    compare.add_argument("--rss-slack-mib", type=float, default=64.0)
    compare.add_argument("--fail-on-regression", action=argparse.BooleanOptionalAction, default=True)
    compare.set_defaults(func=compare_benchmarks)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
