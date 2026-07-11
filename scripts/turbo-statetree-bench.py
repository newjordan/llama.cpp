#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BIN = "/home/frosty40/turbo/turbo-combined/build/bin/llama-server"
DEFAULT_MODEL = "/home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf"
DEFAULT_OUT_DIR = "/tmp/turbo-statetree-bench"
RESULT_SCHEMA_VERSION = 3

BUILD_CACHE_KEYS = (
    "GGML_SYCL",
    "GGML_SYCL_DNN",
    "GGML_SYCL_F16",
    "GGML_SYCL_GRAPH",
    "GGML_SYCL_HOST_MEM_FALLBACK",
    "GGML_SYCL_TARGET",
    "CMAKE_BUILD_TYPE",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
)

# Binary identity, port, attach mode, cleanup modes, and candidate-only
# retention controls intentionally differ between the accepted parent and a
# bounded-retention candidate. Every other recorded config field is part of
# the comparable workload.
NON_WORKLOAD_CONFIG_KEYS = frozenset({
    "bin",
    "port",
    "attach",
    "modes",
    "statetree_lease_ms",
    "statetree_max_state_bytes",
})
MANDATORY_COMPARISON_METRICS = (
    "fork_server_ms",
    "branch_aggregate_predicted_tps",
)
CONTROLLED_SERVER_OPTIONS = frozenset({
    "-m", "--model", "--model-url", "-c", "--ctx-size", "-np", "--parallel",
    "--host", "--port", "-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified",
    "--slots", "--no-slots", "--metrics", "--no-metrics",
    "--statetree-lease-ms", "--statetree-max-state-bytes",
})


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


def transaction_slot_plan(family_ids: list[int], repeat: int) -> tuple[int, list[int], int]:
    if not family_ids:
        raise ValueError("StateTree transaction requires at least one family slot")
    source_id = family_ids[repeat % len(family_ids)]
    destinations = [slot_id for slot_id in family_ids if slot_id != source_id]
    # Manual loser erasure leaves only the original fork root reforkable. By
    # rotating the root itself, manual and commit lanes use the same physical
    # winner without sacrificing slot-position coverage.
    return source_id, destinations, source_id


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


def required_nonnegative_number(data: dict[str, Any], key: str, label: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is missing or is not numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise RuntimeError(f"{label} must be finite and non-negative")
    return parsed


def required_nonnegative_int(data: dict[str, Any], key: str, label: str) -> int:
    value = data.get(key)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{label} is missing or is not a non-negative integer")
    return value


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


def file_identity(path: str, include_sha256: bool = False) -> dict[str, Any] | None:
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return None
    result: dict[str, Any] = {"realpath": str(resolved), "size_bytes": stat.st_size}
    if include_sha256:
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def bundled_runtime_identities(binary: str) -> dict[str, dict[str, Any]]:
    binary_dir = Path(binary).expanduser().resolve().parent
    result: dict[str, dict[str, Any]] = {}
    for candidate in sorted(binary_dir.glob("*.so")):
        if candidate.name == "libllama-bench-impl.so":
            continue
        identity = file_identity(str(candidate), include_sha256=True)
        if identity is not None:
            result[candidate.name] = identity
    return result


def cmake_cache_identity(binary: str) -> dict[str, str] | None:
    resolved = Path(binary).expanduser().resolve()
    cache_path = next(
        (parent / "CMakeCache.txt" for parent in resolved.parents if (parent / "CMakeCache.txt").is_file()),
        None,
    )
    if cache_path is None:
        return None
    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        key_and_type, separator, value = line.partition("=")
        if not separator:
            continue
        key = key_and_type.split(":", 1)[0]
        if key in BUILD_CACHE_KEYS:
            values[key] = value
    return values


def launch_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[Any]:
    if port_open(args.port):
        raise RuntimeError(f"port {args.port} is already listening")

    command = [
        args.bin,
        "-m", args.model,
        "-ngl", str(args.ngl),
        "-ncmoe", str(args.ncmoe),
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
    if args.statetree_lease_ms > 0:
        command.extend(["--statetree-lease-ms", str(args.statetree_lease_ms)])
    if args.statetree_max_state_bytes > 0:
        command.extend(["--statetree-max-state-bytes", str(args.statetree_max_state_bytes)])
    if args.extra_server_args:
        command.extend(shlex.split(args.extra_server_args))

    launch_command = command
    if args.source_oneapi:
        launch_command = [
            "bash",
            "-lc",
            "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec " + shlex.join(command),
        ]

    env = os.environ.copy()
    env["GGML_SYCL_ENABLE_FUSION"] = "1"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(launch_command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
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


def parse_drm_fdinfo(body: str) -> dict[str, Any] | None:
    values: dict[str, Any] = {}
    for line in body.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator or not key.startswith("drm-"):
            continue
        raw_value = raw_value.strip()
        if key in {"drm-driver", "drm-pdev", "drm-client-id"}:
            values[key.removeprefix("drm-").replace("-", "_")] = raw_value
            continue
        parts = raw_value.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and parts[1] == "KiB":
            value *= 1024
        values[key.removeprefix("drm-").replace("-", "_") + "_bytes"] = value
    return values or None


def read_drm_memory(pid: int | None) -> dict[str, Any] | None:
    if pid is None:
        return None
    fd_dir = Path(f"/proc/{pid}/fd")
    fdinfo_dir = Path(f"/proc/{pid}/fdinfo")
    if not fd_dir.exists() or not fdinfo_dir.exists():
        return None

    clients: dict[str, dict[str, Any]] = {}
    try:
        fds = list(fd_dir.iterdir())
    except OSError:
        return None
    for fd in fds:
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if not target.startswith("/dev/dri/"):
            continue
        try:
            parsed = parse_drm_fdinfo((fdinfo_dir / fd.name).read_text(encoding="utf-8"))
        except OSError:
            continue
        if not parsed:
            continue
        client_id = str(parsed.get("client_id") or fd.name)
        clients.setdefault(client_id, parsed)

    if not clients:
        return None
    totals: dict[str, int] = {}
    for client in clients.values():
        for key, value in client.items():
            if key.endswith("_bytes") and isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return {
        "clients": list(clients.values()),
        "totals": totals,
    }


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
        "state_id",
        "node_id",
        "parent_node_id",
        "retention_touch",
        "lease_pinned",
        "lease_remaining_ms",
        "lease_expired",
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
    slot_ids = [row.get("id") for row in rows]
    if any(type(slot_id) is not int for slot_id in slot_ids) or len(set(slot_ids)) != len(slot_ids):
        raise RuntimeError("slots response has missing or duplicate integer IDs")
    family = [row for row in rows if int(row.get("id", -1)) in family_ids]
    metrics = parse_prometheus(http_text(f"http://127.0.0.1:{port}/metrics"))
    return {
        "captured_at": time.time(),
        "all_slots": summarize_slots(rows),
        "family_slots": summarize_slots(family),
        "process": read_process_memory(pid),
        "gpu": read_drm_memory(pid),
        "server_metrics": {
            key: metrics.get(key)
            for key in (
                "kv_cache_usage_ratio",
                "requests_processing",
                "requests_deferred",
                "requests_reserved",
                "statetree_state_bytes",
                "statetree_retained_bytes",
                "statetree_active_bytes",
                "statetree_state_budget_bytes",
                "statetree_state_high_water_bytes",
                "statetree_retained_high_water_bytes",
                "statetree_families",
                "statetree_active_families",
                "statetree_expired_total",
                "statetree_evicted_total",
                "statetree_reclaimed_bytes_total",
                "statetree_renewed_total",
                "statetree_pressure_rejected_total",
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
    fork_id: int | None = None,
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
    if fork_id is not None:
        payload["fork_id"] = fork_id
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
    prompt_n = required_nonnegative_int(timings, "prompt_n", "completion prompt_n")
    cache_n = required_nonnegative_int(timings, "cache_n", "completion cache_n")
    predicted_n = required_nonnegative_int(timings, "predicted_n", "completion predicted_n")
    # Native completion with n_predict=0 still samples the mandatory prompt
    # logit token, while retaining only the evaluated prompt in the slot.
    if n_predict > 0 and predicted_n != n_predict:
        raise RuntimeError(f"completion predicted {predicted_n} tokens, expected {n_predict}")
    if len(tokens) != predicted_n:
        raise RuntimeError(f"completion returned {len(tokens)} tokens, timings reported {predicted_n}")
    return {
        "client_ms": timed["client_ms"],
        "slot_id": slot_id,
        "prompt_n": prompt_n,
        "cache_n": cache_n,
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
    timings = body.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("fork response is missing timings")
    fork_id_response = body.get("fork_id")
    if fork_id_response is not None and type(fork_id_response) is not int:
        raise RuntimeError("fork response generation is not an integer")
    return {
        "client_ms": timed["client_ms"],
        "server_ms": required_nonnegative_number(timings, "fork_ms", "fork server timing"),
        "fork_id": fork_id_response,
        "n_tokens": required_nonnegative_int(body, "n_tokens", "fork n_tokens"),
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
    if body.get("id_slot") != winner_id or body.get("fork_id") != fork_id:
        raise RuntimeError("commit response does not match the requested winner generation")
    timings = body.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("commit response is missing timings")
    return {
        "client_ms": timed["client_ms"],
        "server_ms": required_nonnegative_number(timings, "commit_ms", "commit server timing"),
        "response": body,
    }


def erase_slot(
    port: int,
    slot_id: int,
    timeout: float,
    fork_id: int | None = None,
) -> dict[str, Any]:
    payload = {"fork_id": fork_id} if fork_id is not None else {}
    return timed_json(
        "POST",
        f"http://127.0.0.1:{port}/slots/{slot_id}?action=erase",
        payload,
        timeout=timeout,
    )


def erase_slots(
    port: int,
    slot_ids: list[int],
    timeout: float,
    fork_id: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    operations = []
    for slot_id in slot_ids:
        operation = erase_slot(port, slot_id, timeout, fork_id=fork_id)
        operations.append({"slot_id": slot_id, "client_ms": operation["client_ms"]})
    return {
        "client_ms": (time.perf_counter() - started) * 1000.0,
        "operations": operations,
    }


def erase_current_slots(port: int, slot_ids: list[int], timeout: float) -> dict[str, Any]:
    rows = http_json("GET", f"http://127.0.0.1:{port}/slots", timeout=timeout)
    if not isinstance(rows, list):
        raise RuntimeError("slots response is not an array")
    fork_ids = {
        int(row["id"]): int(row["fork_id"])
        for row in rows
        if isinstance(row, dict)
        and type(row.get("id")) is int
        and row.get("is_reserved") is True
        and type(row.get("fork_id")) is int
        and int(row["fork_id"]) >= 0
    }
    started = time.perf_counter()
    operations = []
    for slot_id in slot_ids:
        operation = erase_slot(port, slot_id, timeout, fork_id=fork_ids.get(slot_id))
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
    fork_id: int | None = None,
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
            fork_id,
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

    for snapshot_name, snapshot in snapshots.items():
        if not isinstance(snapshot, dict) or "all_slots" not in snapshot:
            continue
        rows = snapshot["all_slots"].get("by_id", {})
        for slot_id, row in rows.items():
            byte_values = [row.get(key) for key in (
                "n_prompt_data_bytes", "n_prompt_checkpoint_bytes", "n_prompt_state_bytes"
            )]
            if all(type(value) is int for value in byte_values):
                data_bytes, checkpoint_bytes, state_bytes = byte_values
                if min(data_bytes, checkpoint_bytes, state_bytes) < 0:
                    failures.append(f"snapshot {snapshot_name} slot {slot_id} reported negative state bytes")
                if state_bytes != data_bytes + checkpoint_bytes:
                    failures.append(f"snapshot {snapshot_name} slot {slot_id} state byte accounting is inconsistent")

        metrics = snapshot.get("server_metrics", {})
        metric_state = metrics.get("statetree_state_bytes")
        exact_state = snapshot["all_slots"].get("prompt_state_bytes")
        if isinstance(metric_state, (int, float)) and isinstance(exact_state, int):
            if int(metric_state) != exact_state:
                failures.append(f"snapshot {snapshot_name} StateTree state gauge disagrees with slot bytes")
        budget = metrics.get("statetree_state_budget_bytes")
        high_water = metrics.get("statetree_state_high_water_bytes")
        if isinstance(budget, (int, float)) and budget > 0:
            if isinstance(metric_state, (int, float)) and metric_state > budget:
                failures.append(f"snapshot {snapshot_name} exceeded the StateTree state budget")
            if isinstance(high_water, (int, float)) and high_water > budget:
                failures.append(f"snapshot {snapshot_name} StateTree high-water exceeded the state budget")

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

    loser_ids = sorted(slot_id for slot_id in family_ids if slot_id != winner_id)
    if sample["cleanup_mode"] == "commit":
        released = sample["cleanup"]["response"].get("released")
        if sorted(released or []) != loser_ids:
            failures.append("commit released the wrong slot set")

    retained_fields = (
        ("n_prompt_tokens", "prompt tokens"),
        ("n_prompt_checkpoints", "prompt checkpoints"),
        ("n_prompt_data_bytes", "prompt data bytes"),
        ("n_prompt_checkpoint_bytes", "prompt checkpoint bytes"),
        ("n_prompt_state_bytes", "prompt state bytes"),
    )
    for slot_id in loser_ids:
        loser = slot_row(snapshots["after_cleanup"], slot_id)
        for key, description in retained_fields:
            if int(loser.get(key) or 0) != 0:
                failures.append(f"loser slot {slot_id} retained {description}")

    refork = sample.get("refork")
    if refork and isinstance(refork.get("fork_id"), int):
        if refork["fork_id"] == sample["fork"]["fork_id"]:
            failures.append("refork did not advance the generation")

    final_snapshot = snapshots.get("final")
    if final_snapshot:
        if final_snapshot["family_slots"]["reserved"] != 0:
            failures.append("final cleanup left reserved slots")
        family_summary = final_snapshot["family_slots"]
        summary_keys = {
            "n_prompt_tokens": "prompt_tokens",
            "n_prompt_checkpoints": "prompt_checkpoints",
            "n_prompt_data_bytes": "prompt_data_bytes",
            "n_prompt_checkpoint_bytes": "prompt_checkpoint_bytes",
            "n_prompt_state_bytes": "prompt_state_bytes",
        }
        for key, description in retained_fields:
            summary_value = family_summary.get(summary_keys[key])
            row_has_state = any(
                int(slot_row(final_snapshot, slot_id).get(key) or 0) != 0
                for slot_id in family_ids
            )
            if int(summary_value or 0) != 0 or row_has_state:
                failures.append(f"final cleanup left {description}")
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
    source_id, destinations, winner_id = transaction_slot_plan(family_ids, repeat)
    sample: dict[str, Any] = {
        "repeat": repeat,
        "prefix_tokens": len(prefix),
        "cleanup_mode": cleanup_mode,
        "family_ids": family_ids,
        "winner_id": winner_id,
        "layout": args.layout,
        "persistent_fragmentation": args.persistent_fragmentation,
        "supported": True,
        "snapshots": {},
    }

    persistent_fragmentation = args.layout == "fragmented" and args.persistent_fragmentation
    reset_ids = family_ids if persistent_fragmentation else list(range(args.parallel))
    erase_current_slots(args.port, reset_ids, args.request_timeout)
    try:
        if args.layout == "fragmented":
            fill_requests = []
            if not persistent_fragmentation:
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
                "persistent": persistent_fragmentation,
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
                fork_id if isinstance(fork_id, int) else None,
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
            fork_id if isinstance(fork_id, int) else None,
        )
        sample["snapshots"]["before_cleanup"] = capture_snapshot(args.port, pid, family_ids)
        sample["expected_shared_kv_cells_before_cleanup"] = expected_shared_kv_cells(
            sample["snapshots"]["before_cleanup"], len(prefix), family_ids
        )

        if cleanup_mode == "commit":
            sample["cleanup"] = commit_slot(args.port, winner_id, fork_id, args.request_timeout)
        else:
            losers = [slot_id for slot_id in family_ids if slot_id != winner_id]
            sample["cleanup"] = erase_slots(
                args.port, losers, args.request_timeout, fork_id=fork_id if isinstance(fork_id, int) else None
            )
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
            next_fork_id = sample["refork"].get("fork_id")
            if not isinstance(next_fork_id, int):
                raise RuntimeError("refork response is missing a generation")
            if cleanup_mode == "commit":
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
                    fork_id=next_fork_id,
                )
        else:
            sample["refork"] = None
            sample["refork_unsupported_reason"] = "parent fork families cannot be reforked in place"

        erase_current_slots(args.port, family_ids, args.request_timeout)
        sample["snapshots"]["final"] = capture_snapshot(args.port, pid, family_ids)
        sample["failures"] = validate_sample(sample)
        return sample
    finally:
        erase_current_slots(args.port, reset_ids, args.request_timeout)


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
        "gpu_total_vram_before_cleanup_bytes": "snapshots.before_cleanup.gpu.totals.total_vram0_bytes",
        "gpu_total_vram_after_cleanup_bytes": "snapshots.after_cleanup.gpu.totals.total_vram0_bytes",
        "gpu_resident_vram_before_cleanup_bytes": "snapshots.before_cleanup.gpu.totals.resident_vram0_bytes",
        "gpu_resident_vram_after_cleanup_bytes": "snapshots.after_cleanup.gpu.totals.resident_vram0_bytes",
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
        usable = [
            sample
            for sample in selected
            if sample.get("supported", True)
            and not sample.get("error")
            and not sample.get("failures")
        ]
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
    commit_samples = [sample for sample in samples if sample["cleanup_mode"] == "commit"]
    return {
        "groups": groups,
        "samples": len(samples),
        "supported_samples": sum(sample.get("supported", True) for sample in samples),
        "failed_samples": sum(bool(sample.get("error") or sample.get("failures")) for sample in samples),
        "commit_supported": bool(commit_samples) and all(
            sample.get("supported", True)
            and not sample.get("error")
            and not sample.get("failures")
            for sample in commit_samples
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
    vram_slack_mib: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    compatibility_errors: list[str] = []
    baseline_groups = baseline.get("summary", {}).get("groups", {})
    candidate_groups = candidate.get("summary", {}).get("groups", {})

    if baseline.get("kind") != candidate.get("kind"):
        compatibility_errors.append(
            f"result kind mismatch: baseline={baseline.get('kind')!r}, candidate={candidate.get('kind')!r}"
        )
    if baseline.get("schema_version") != candidate.get("schema_version"):
        compatibility_errors.append(
            "result schema mismatch: "
            f"baseline={baseline.get('schema_version')!r}, candidate={candidate.get('schema_version')!r}"
        )

    baseline_config = baseline.get("config") if isinstance(baseline.get("config"), dict) else {}
    candidate_config = candidate.get("config") if isinstance(candidate.get("config"), dict) else {}
    config_keys = sorted((set(baseline_config) | set(candidate_config)) - NON_WORKLOAD_CONFIG_KEYS)
    for key in config_keys:
        baseline_has_key = key in baseline_config
        candidate_has_key = key in candidate_config
        if baseline_has_key != candidate_has_key or baseline_config.get(key) != candidate_config.get(key):
            compatibility_errors.append(
                f"workload config mismatch for {key}: "
                f"baseline={baseline_config.get(key)!r}, candidate={candidate_config.get(key)!r}"
            )

    baseline_manual = sorted(group for group in baseline_groups if group.endswith(":manual"))
    common_manual = [group for group in baseline_manual if group in candidate_groups]
    for group in baseline_manual:
        if group not in candidate_groups:
            compatibility_errors.append(f"candidate is missing baseline manual group {group}")
            continue
        for metric in MANDATORY_COMPARISON_METRICS:
            if metric_p50(baseline, group, metric) is None:
                compatibility_errors.append(f"baseline group {group} is missing mandatory metric {metric}")
            if metric_p50(candidate, group, metric) is None:
                compatibility_errors.append(f"candidate group {group} is missing mandatory metric {metric}")

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

        base_vram = metric_p50(baseline, group, "gpu_total_vram_before_cleanup_bytes")
        cand_vram = metric_p50(candidate, group, "gpu_total_vram_before_cleanup_bytes")
        if base_vram is not None and cand_vram is not None:
            limit = base_vram + vram_slack_mib * 1024.0 * 1024.0
            checks.append({
                "group": group,
                "metric": "gpu_total_vram_before_cleanup_bytes",
                "baseline": base_vram,
                "candidate": cand_vram,
                "limit": limit,
                "passed": cand_vram <= limit,
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
            "vram_slack_mib": vram_slack_mib,
        },
        "checks": checks,
        "compatibility": {
            "passed": not compatibility_errors,
            "errors": compatibility_errors,
        },
        "commit_vs_manual": commit_deltas,
        "candidate_commit_supported": bool(candidate.get("summary", {}).get("commit_supported")),
        "passed": bool(checks)
            and not compatibility_errors
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
    controlled_overrides = sorted({
        token.split("=", 1)[0]
        for token in shlex.split(getattr(args, "extra_server_args", ""))
        if token.split("=", 1)[0] in CONTROLLED_SERVER_OPTIONS
    })
    if controlled_overrides:
        raise SystemExit(
            "--extra-server-args cannot override controlled options: " + ", ".join(controlled_overrides)
        )
    if args.persistent_fragmentation and args.layout != "fragmented":
        raise SystemExit("--persistent-fragmentation requires --layout fragmented")
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
    server_ready = False
    live_props: dict[str, Any] | None = None
    persistent_setup: dict[str, Any] | None = None
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    try:
        if args.attach:
            wait_healthy(args.port, None, args.startup_timeout)
            pid = args.server_pid
        else:
            proc = launch_server(args, log_path)
            wait_healthy(args.port, proc, args.startup_timeout)
            pid = proc.pid
        server_ready = True
        props = http_json("GET", f"http://127.0.0.1:{args.port}/props", timeout=30.0)
        if isinstance(props, dict):
            live_props = {
                key: props.get(key)
                for key in ("build_info", "model_alias", "model_path", "total_slots", "statetree")
            }
            live_props["n_ctx"] = nested_value(props, "default_generation_settings.n_ctx")

        erase_current_slots(args.port, list(range(args.parallel)), args.request_timeout)
        warmup_tokens = exact_tokens(args.port, min(32, min(args.prefix_tokens)), "warmup")
        completion(args.port, warmup_tokens, 0, min(2, args.branch_tokens), args.seed, args.request_timeout)
        erase_current_slots(args.port, list(range(args.parallel)), args.request_timeout)

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

        if args.layout == "fragmented" and args.persistent_fragmentation:
            erase_current_slots(args.port, list(range(args.parallel)), args.request_timeout)
            fill_requests = []
            for slot_id in range(args.parallel):
                fill_requests.append(completion(
                    args.port,
                    filler_prompts[slot_id],
                    slot_id,
                    0,
                    args.seed + 500 + slot_id,
                    args.request_timeout,
                ))
            erase_slots(args.port, family_ids, args.request_timeout)
            persistent_setup = {
                "fill_tokens": args.fragment_fill_tokens,
                "survivor_slots": [
                    slot_id for slot_id in range(args.parallel) if slot_id not in family_ids
                ],
                "fill_requests": fill_requests,
                "snapshot": capture_snapshot(args.port, pid, family_ids),
            }

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
        if server_ready:
            try:
                erase_current_slots(args.port, list(range(args.parallel)), args.request_timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: final slot cleanup failed: {exc}", file=sys.stderr, flush=True)
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
            "persistent_fragmentation": args.persistent_fragmentation,
            "prefix_tokens": args.prefix_tokens,
            "branch_suffix_tokens": args.branch_suffix_tokens,
            "branch_tokens": args.branch_tokens,
            "repeats": args.repeats,
            "modes": args.modes,
            "seed": args.seed,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "threads": args.threads,
            "ngl": args.ngl,
            "ncmoe": args.ncmoe,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "flash_attn": args.flash_attn,
            "extra_server_args": args.extra_server_args,
            "source_oneapi": args.source_oneapi,
            "ggml_sycl_enable_fusion": "1",
            "statetree_lease_ms": args.statetree_lease_ms,
            "statetree_max_state_bytes": args.statetree_max_state_bytes,
            "model_identity": file_identity(args.model),
            "cmake_cache": cmake_cache_identity(args.bin),
            "attach": args.attach,
        },
        "server": {
            "version": server_version(args),
            "commit": args.commit,
            "log_path": str(log_path) if not args.attach else None,
            "binary_identity": file_identity(args.bin, include_sha256=True),
            "bundled_runtime_identities": bundled_runtime_identities(args.bin),
            "live_props": live_props,
        },
        "persistent_fragmentation_setup": persistent_setup,
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
        args.vram_slack_mib,
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
    run.add_argument("--persistent-fragmentation", action=argparse.BooleanOptionalAction, default=False)
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
    run.add_argument("--ncmoe", type=non_negative_int, default=0)
    run.add_argument("--cache-type-k", default="f16")
    run.add_argument("--cache-type-v", default="f16")
    run.add_argument("--flash-attn", choices=("on", "off", "auto"), default="auto")
    run.add_argument("--request-timeout", type=float, default=900.0)
    run.add_argument("--startup-timeout", type=float, default=900.0)
    run.add_argument("--extra-server-args", default="")
    run.add_argument("--statetree-lease-ms", type=non_negative_int, default=0)
    run.add_argument("--statetree-max-state-bytes", type=non_negative_int, default=0)
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
    compare.add_argument("--vram-slack-mib", type=float, default=64.0)
    compare.add_argument("--fail-on-regression", action=argparse.BooleanOptionalAction, default=True)
    compare.set_defaults(func=compare_benchmarks)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
