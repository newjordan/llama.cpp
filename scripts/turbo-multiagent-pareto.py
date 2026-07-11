#!/usr/bin/env python3
"""Measure a fixed-server multi-agent decode speed frontier.

The runner intentionally attaches to one already-configured server.  This
keeps the KV/context shape constant while varying only the number of active
clients, which is the meaningful multi-agent Pareto dimension.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PROMPT = (
    "Write a compact but production-quality Python class implementing a "
    "thread-safe LRU cache with per-item TTL expiry. Include type hints, "
    "docstrings, and a short usage example."
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def http_json(method: str, url: str, payload: object | None = None, timeout: float = 900.0) -> Any:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc


def wait_healthy(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            health = http_json("GET", f"{base_url}/health", timeout=5.0)
            if isinstance(health, dict):
                return
            last_error = f"unexpected health payload: {health!r}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def metrics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def erase_slots(base_url: str, slots: range, timeout: float) -> None:
    for slot in slots:
        try:
            http_json("POST", f"{base_url}/slots/{slot}?action=erase", {}, timeout)
        except RuntimeError as exc:
            # Empty/never-used slots can legitimately reject an erase on some
            # server builds.  A reserved slot or any other error is not hidden.
            if "not found" not in str(exc).lower() and "empty" not in str(exc).lower():
                raise


def completion(base_url: str, slot: int, n_predict: int, seed: int, timeout: float) -> dict[str, Any]:
    payload = {
        "prompt": PROMPT,
        "id_slot": slot,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": False,
        "ignore_eos": True,
        "stop": [],
        "stream": False,
        "return_tokens": True,
    }
    started = time.perf_counter()
    response = http_json("POST", f"{base_url}/completion", payload, timeout)
    ended = time.perf_counter()
    if not isinstance(response, dict):
        raise RuntimeError("completion response is not an object")
    if response.get("id_slot") != slot:
        raise RuntimeError(f"completion used slot {response.get('id_slot')!r}, expected {slot}")
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("completion response is missing timings")
    predicted_n = timings.get("predicted_n")
    if type(predicted_n) is not int or predicted_n != n_predict:
        raise RuntimeError(f"completion predicted {predicted_n!r}, expected {n_predict}")
    predicted_tps = timings.get("predicted_per_second")
    if not isinstance(predicted_tps, (int, float)) or predicted_tps <= 0:
        raise RuntimeError("completion response is missing a positive predicted_per_second")
    return {
        "slot": slot,
        "started": started,
        "ended": ended,
        "client_ms": (ended - started) * 1000.0,
        "prompt_n": timings.get("prompt_n"),
        "predicted_n": predicted_n,
        "prompt_tps": timings.get("prompt_per_second"),
        "predicted_tps": float(predicted_tps),
    }


def run_wave(
    base_url: str,
    active_agents: int,
    n_predict: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(active_agents)

    def one(slot: int) -> dict[str, Any]:
        barrier.wait()
        return completion(base_url, slot, n_predict, seed + slot, timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=active_agents) as executor:
        requests = list(executor.map(one, range(active_agents)))
    starts = [float(request["started"]) for request in requests]
    ends = [float(request["ended"]) for request in requests]
    wall_s = max(ends) - min(starts)
    total_predicted = sum(int(request["predicted_n"]) for request in requests)
    return {
        "active_agents": active_agents,
        "total_predicted_tokens": total_predicted,
        "wall_ms": wall_s * 1000.0,
        "aggregate_wall_tps": total_predicted / wall_s if wall_s else None,
        "mean_slot_tps": statistics.mean(float(request["predicted_tps"]) for request in requests),
        "min_slot_tps": min(float(request["predicted_tps"]) for request in requests),
        "max_slot_tps": max(float(request["predicted_tps"]) for request in requests),
        "p95_client_ms": percentile([float(request["client_ms"]) for request in requests], 0.95),
        "requests": requests,
    }


def pareto_front(rows: list[dict[str, Any]]) -> list[int]:
    front: list[int] = []
    for row in rows:
        aggregate = row["aggregate_wall_tps"]["p50"]
        per_agent = row["mean_slot_tps"]["p50"]
        if aggregate is None or per_agent is None:
            continue
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_aggregate = other["aggregate_wall_tps"]["p50"]
            other_per_agent = other["mean_slot_tps"]["p50"]
            if other_aggregate is None or other_per_agent is None:
                continue
            if (
                other_aggregate >= aggregate
                and other_per_agent >= per_agent
                and (other_aggregate > aggregate or other_per_agent > per_agent)
            ):
                dominated = True
                break
        if not dominated:
            front.append(int(row["active_agents"]))
    return front


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parallel", type=int, default=12, help="fixed server slot count")
    parser.add_argument("--ctx", type=int, default=262144, help="expected server context")
    parser.add_argument("--agents", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--seed", type=int, default=262144)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.parallel <= 0 or args.repeats <= 0 or args.n_predict <= 0:
        raise SystemExit("parallel, repeats, and n-predict must be positive")
    agents = sorted(set(args.agents))
    if not agents or any(agent <= 0 or agent > args.parallel for agent in agents):
        raise SystemExit("agents must be positive and no greater than --parallel")

    base_url = f"http://127.0.0.1:{args.port}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / f"{args.label}.samples.jsonl"
    result_path = args.out_dir / f"{args.label}.result.json"
    started = now()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "turbo-multiagent-pareto",
        "label": args.label,
        "started": started,
        "config": {
            "port": args.port,
            "parallel": args.parallel,
            "ctx": args.ctx,
            "agents": agents,
            "repeats": args.repeats,
            "n_predict": args.n_predict,
            "prompt": PROMPT,
            "metric": "aggregate predicted tokens per second over concurrent client wall time",
        },
        "samples": [],
        "summary": {},
        "passed": False,
        "failures": [],
    }
    try:
        wait_healthy(base_url, args.startup_timeout)
        props = http_json("GET", f"{base_url}/props", timeout=30.0)
        result["server"] = {
            "props": props,
            "expected_parallel": args.parallel,
        }
        if not isinstance(props, dict):
            raise RuntimeError("/props response is not an object")
        if props.get("total_slots") != args.parallel:
            raise RuntimeError(
                f"server reports total_slots={props.get('total_slots')!r}, expected {args.parallel}"
            )
        generation_settings = props.get("default_generation_settings")
        live_ctx = generation_settings.get("n_ctx") if isinstance(generation_settings, dict) else None
        if live_ctx != args.ctx:
            raise RuntimeError(f"server reports n_ctx={live_ctx!r}, expected {args.ctx}")

        erase_slots(base_url, range(args.parallel), args.request_timeout)
        warmups: list[dict[str, Any]] = []
        for active_agents in agents:
            erase_slots(base_url, range(args.parallel), args.request_timeout)
            warmup = run_wave(
                base_url,
                active_agents,
                min(16, args.n_predict),
                args.seed - active_agents,
                args.request_timeout,
            )
            warmups.append({
                "active_agents": active_agents,
                "n_predict": min(16, args.n_predict),
                "aggregate_wall_tps": warmup["aggregate_wall_tps"],
            })
        erase_slots(base_url, range(args.parallel), args.request_timeout)
        result["warmups"] = warmups

        with samples_path.open("w", encoding="utf-8") as sample_file:
            for repeat in range(args.repeats):
                agent_order = list(agents)
                random.Random(args.seed + repeat).shuffle(agent_order)
                for order_index, active_agents in enumerate(agent_order):
                    print(f"agents={active_agents} repeat={repeat + 1}/{args.repeats}", flush=True)
                    erase_slots(base_url, range(args.parallel), args.request_timeout)
                    sample: dict[str, Any] = {
                        "active_agents": active_agents,
                        "repeat": repeat,
                        "order_index": order_index,
                    }
                    try:
                        sample.update(
                            run_wave(
                                base_url,
                                active_agents,
                                args.n_predict,
                                args.seed + active_agents * 1000 + repeat * 100,
                                args.request_timeout,
                            )
                        )
                        sample["failures"] = []
                    except Exception as exc:  # noqa: BLE001
                        sample["error"] = str(exc)
                        sample["failures"] = [str(exc)]
                        result["failures"].append(
                            f"agents={active_agents} repeat={repeat}: {exc}"
                        )
                    result["samples"].append(sample)
                    sample_file.write(json.dumps(sample, sort_keys=True) + "\n")
                    sample_file.flush()

        rows: list[dict[str, Any]] = []
        for active_agents in agents:
            samples = [
                sample
                for sample in result["samples"]
                if sample["active_agents"] == active_agents and not sample.get("failures")
            ]
            rows.append(
                {
                    "active_agents": active_agents,
                    "samples": len(samples),
                    "failed_samples": args.repeats - len(samples),
                    "aggregate_wall_tps": metrics(
                        [float(sample["aggregate_wall_tps"]) for sample in samples]
                    ),
                    "mean_slot_tps": metrics(
                        [float(sample["mean_slot_tps"]) for sample in samples]
                    ),
                    "min_slot_tps": metrics(
                        [float(sample["min_slot_tps"]) for sample in samples]
                    ),
                    "p95_client_ms": metrics(
                        [float(sample["p95_client_ms"]) for sample in samples]
                    ),
                }
            )
        result["summary"] = {
            "rows": rows,
            "pareto_front_active_agents": pareto_front(rows),
            "samples": len(result["samples"]),
            "failed_samples": len(result["failures"]),
        }
        result["passed"] = not result["failures"]
    except Exception as exc:  # noqa: BLE001
        result["failures"].append(str(exc))
    finally:
        try:
            erase_slots(base_url, range(args.parallel), args.request_timeout)
        except Exception as exc:  # noqa: BLE001
            result["failures"].append(f"cleanup: {exc}")
        result["finished"] = now()
        result["passed"] = not result["failures"]
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"result": str(result_path), "passed": result["passed"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
