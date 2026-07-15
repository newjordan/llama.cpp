#!/usr/bin/env python3
"""Capture production-shaped MoE routing occupancy with heterogeneous agents."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPTS = [
    "Implement a bounded lock-free MPMC queue in Rust and explain its memory ordering.",
    "Diagnose a PostgreSQL query plan that scans 200 million rows despite a composite index.",
    "Write a concise proof that the square root of 2 is irrational using contradiction.",
    "Design a Kubernetes incident runbook for pods stuck terminating after a node failure.",
    "Translate this sentence into idiomatic French and explain two word choices: The release was delayed by a subtle race condition.",
    "Return a JSON Schema for an invoice with line items, taxes, currency, and payment status.",
    "Explain photosynthesis to a curious twelve-year-old without using equations.",
    "Optimize a C++ matrix transpose for cache locality and discuss when SIMD helps.",
    "Write a villanelle about a radio telescope listening through a winter storm.",
    "Compare event sourcing and change-data capture for rebuilding materialized views.",
    "Create a vegetarian dinner recipe using chickpeas, spinach, lemon, and pantry spices.",
    "Summarize the likely root causes of rising p99 latency with flat CPU and saturated memory bandwidth.",
]


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


def erase_slot(base_url: str, slot: int, timeout: float) -> None:
    try:
        http_json("POST", f"{base_url}/slots/{slot}?action=erase", {}, timeout)
    except RuntimeError as exc:
        if "not found" not in str(exc).lower() and "empty" not in str(exc).lower():
            raise


def complete(base_url: str, slot: int, prompt: str, n_predict: int, timeout: float) -> dict[str, Any]:
    response = http_json(
        "POST",
        f"{base_url}/completion",
        {
            "prompt": prompt,
            "id_slot": slot,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 9000 + slot,
            "cache_prompt": False,
            "ignore_eos": True,
            "stop": [],
            "stream": False,
            "return_tokens": True,
            "speculative.n_max": 0,
            "speculative.serial_anchor": False,
        },
        timeout,
    )
    if not isinstance(response, dict) or response.get("id_slot") != slot:
        raise RuntimeError(f"slot {slot} received an invalid completion response")
    timings = response.get("timings")
    tokens = response.get("tokens")
    if not isinstance(timings, dict) or not isinstance(tokens, list):
        raise RuntimeError(f"slot {slot} response omitted timings or tokens")
    if timings.get("predicted_n") != n_predict or len(tokens) != n_predict:
        raise RuntimeError(f"slot {slot} did not generate exactly {n_predict} tokens")
    token_bytes = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return {
        "slot": slot,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_n": timings.get("prompt_n"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.n_predict <= 0:
        raise SystemExit("--n-predict must be positive")

    base_url = f"http://127.0.0.1:{args.port}"
    for slot in range(len(PROMPTS)):
        erase_slot(base_url, slot, args.timeout)

    barrier = threading.Barrier(len(PROMPTS))

    def one(item: tuple[int, str]) -> dict[str, Any]:
        slot, prompt = item
        barrier.wait()
        return complete(base_url, slot, prompt, args.n_predict, args.timeout)

    started_at = now()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PROMPTS)) as executor:
        requests = list(executor.map(one, enumerate(PROMPTS)))
    wall_s = time.perf_counter() - started

    for slot in range(len(PROMPTS)):
        erase_slot(base_url, slot, args.timeout)

    result = {
        "schema_version": 1,
        "kind": "treebeard-moe-reuse-probe",
        "started_at": started_at,
        "finished_at": now(),
        "agents": len(PROMPTS),
        "heterogeneous_prompt_hashes": len({row["prompt_sha256"] for row in requests}),
        "distinct_output_hashes": len({row["tokens_sha256"] for row in requests}),
        "n_predict_per_agent": args.n_predict,
        "wall_seconds": wall_s,
        "aggregate_tokens_per_second": len(PROMPTS) * args.n_predict / wall_s,
        "requests": requests,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "agents", "heterogeneous_prompt_hashes", "distinct_output_hashes",
        "n_predict_per_agent", "aggregate_tokens_per_second",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
