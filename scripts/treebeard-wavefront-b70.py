#!/usr/bin/env python3
"""Matched B70 benchmark for per-request speculative wavefront widths."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_WIDTHS = "0,1,2,4,8,12,24,48"
DEFAULT_CONCURRENCY_WIDTHS = "0,1,2,4,8,12,24,48"
DEFAULT_DEPTHS = "512,32768,131072,256000"
CASE_SUFFIXES = {
    "structured-copy": (
        "\n\nThe archive above is unrelated padding. The canonical service telemetry "
        "stream continues by repeating the established record:\n"
        + (
            '{"service":"treebeard","state":"active","slots":12,'
            '"context":262144,"wave":"verified"}\n'
        ) * 8
    ),
    "code-edit": (
        "\n\nThe archive above is unrelated padding. Return only the edited Python "
        "function. Change every occurrence of old_state to new_state while "
        "preserving all other text exactly.\n"
        "def update(old_state):\n"
        "    old_state[\"count\"] = old_state.get(\"count\", 0) + 1\n"
        "    old_state[\"ready\"] = old_state[\"count\"] > 3\n"
        "    old_state[\"label\"] = \"ready\" if old_state[\"ready\"] else \"waiting\"\n"
        "    old_state[\"history\"] = old_state.get(\"history\", []) + [old_state[\"label\"]]\n"
        "    old_state[\"version\"] = old_state.get(\"version\", 0) + 1\n"
        "    return old_state\n"
        "Assistant:\n"
    ),
    "free-prose": (
        "\n\nThe archive above is unrelated padding. Explain in one compact paragraph "
        "why aggregate throughput from twelve independent agents does not "
        "automatically reduce latency for one autoregressive answer, and state "
        "what verified lookahead changes.\nAssistant:\n"
    ),
}


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


def parse_int_list(value: str, *, require_zero: bool = False) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be unique non-negative integers")
    if require_zero and 0 not in values:
        raise argparse.ArgumentTypeError("values must include zero")
    return values


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


def tokenize(base_url: str, content: str, add_special: bool, timeout: float) -> list[int]:
    response = http_json(
        "POST",
        f"{base_url}/tokenize",
        {"content": content, "add_special": add_special, "parse_special": True},
        timeout,
    )
    tokens = response.get("tokens") if isinstance(response, dict) else None
    if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
        raise RuntimeError("tokenize response did not contain integer tokens")
    return tokens


def build_filler(base_url: str, required: int, timeout: float) -> list[int]:
    tokens: list[int] = []
    record = 0
    while len(tokens) < required:
        lines: list[str] = []
        for _ in range(512):
            digest = hashlib.sha256(f"treebeard-archive-{record}".encode("ascii")).hexdigest()
            lines.append(f"Archive record {record:08d} checksum {digest} status sealed.\n")
            record += 1
        tokens.extend(tokenize(base_url, "".join(lines), not tokens, timeout))
    return tokens


def build_prompt(filler: list[int], suffix: list[int], depth: int) -> list[int]:
    if len(suffix) >= depth:
        raise ValueError(f"case suffix has {len(suffix)} tokens, which does not fit depth {depth}")
    needed = depth - len(suffix)
    if needed > len(filler):
        raise ValueError(f"filler has {len(filler)} tokens, expected at least {needed}")
    prompt = [*filler[:needed], *suffix]
    if len(prompt) != depth:
        raise AssertionError("constructed prompt depth mismatch")
    return prompt


def erase_slot(base_url: str, slot: int, timeout: float) -> None:
    try:
        http_json("POST", f"{base_url}/slots/{slot}?action=erase", {}, timeout)
    except RuntimeError as exc:
        if "not found" not in str(exc).lower() and "empty" not in str(exc).lower():
            raise


def validate_rounds(width: int, proposed: list[Any], accepted: list[Any]) -> None:
    if len(proposed) != len(accepted):
        raise RuntimeError("draft proposal and acceptance arrays are not aligned")
    for draft_n, accepted_n in zip(proposed, accepted):
        if type(draft_n) is not int or type(accepted_n) is not int:
            raise RuntimeError("draft telemetry contains a non-integer width")
        if draft_n <= 0 or accepted_n < 0 or accepted_n > draft_n:
            raise RuntimeError("draft telemetry contains an invalid round")
        if draft_n > width:
            raise RuntimeError(f"draft round width {draft_n} exceeded request cap {width}")


def validate_anchor_telemetry(width: int, enabled: bool, timings: dict[str, Any]) -> dict[str, Any]:
    anchor_n = int(timings.get("draft_anchor_n") or 0)
    match_n = int(timings.get("draft_anchor_match_n") or 0)
    fallback_n = int(timings.get("draft_anchor_fallback_n") or 0)
    serial_tokens = list(timings.get("draft_anchor_serial_tokens") or [])
    batched_tokens = list(timings.get("draft_anchor_batched_tokens") or [])
    values = [anchor_n, match_n, fallback_n, *serial_tokens, *batched_tokens]
    if not all(type(value) is int for value in values):
        raise RuntimeError("serial anchor telemetry contains a non-integer")
    if anchor_n != len(serial_tokens) or anchor_n != len(batched_tokens):
        raise RuntimeError("serial anchor token arrays are not aligned")
    if match_n + fallback_n != anchor_n:
        raise RuntimeError("serial anchor outcomes do not match the anchor total")
    observed_matches = sum(serial == batched for serial, batched in zip(serial_tokens, batched_tokens))
    if observed_matches != match_n:
        raise RuntimeError("serial anchor token comparisons do not match the reported outcomes")
    if (width == 0 or not enabled) and anchor_n:
        raise RuntimeError("serial anchor telemetry was emitted for an unanchored request")
    return {
        "draft_anchor_n": anchor_n,
        "draft_anchor_match_n": match_n,
        "draft_anchor_fallback_n": fallback_n,
        "draft_anchor_serial_tokens": serial_tokens,
        "draft_anchor_batched_tokens": batched_tokens,
    }


def completion(
    base_url: str,
    prompt: list[int],
    slot: int,
    width: int,
    n_predict: int,
    seed: int,
    timeout: float,
    serial_anchor: bool,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "id_slot": slot,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": True,
        "return_tokens": True,
        "ignore_eos": True,
        "stop": [],
        "stream": False,
        "speculative.n_max": width,
        "speculative.serial_anchor": serial_anchor,
    }
    started = time.perf_counter()
    response = http_json("POST", f"{base_url}/completion", payload, timeout)
    ended = time.perf_counter()
    if not isinstance(response, dict):
        raise RuntimeError("completion response is not an object")
    if response.get("id_slot") != slot:
        raise RuntimeError(f"completion used slot {response.get('id_slot')!r}, expected {slot}")
    timings = response.get("timings")
    tokens = response.get("tokens")
    if not isinstance(timings, dict) or not isinstance(tokens, list):
        raise RuntimeError("completion response is missing timings or tokens")
    if not all(type(token) is int for token in tokens):
        raise RuntimeError("completion returned non-integer tokens")
    predicted_n = timings.get("predicted_n")
    if type(predicted_n) is not int or predicted_n != n_predict or len(tokens) != n_predict:
        raise RuntimeError(
            f"completion returned predicted_n={predicted_n!r}, tokens={len(tokens)}, expected {n_predict}"
        )
    predicted_tps = timings.get("predicted_per_second")
    if not isinstance(predicted_tps, (int, float)) or predicted_tps <= 0:
        raise RuntimeError("completion is missing positive predicted_per_second")
    proposed = list(timings.get("draft_n_per_round") or [])
    accepted = list(timings.get("draft_n_accepted_per_round") or [])
    if width == 0 and (proposed or accepted or timings.get("draft_n") or timings.get("draft_n_accepted")):
        raise RuntimeError("width-zero control unexpectedly drafted tokens")
    validate_rounds(width, proposed, accepted)
    draft_n = int(timings.get("draft_n") or 0)
    draft_n_accepted = int(timings.get("draft_n_accepted") or 0)
    if sum(proposed) != draft_n or sum(accepted) != draft_n_accepted:
        raise RuntimeError("per-round draft telemetry does not match totals")
    anchor = validate_anchor_telemetry(width, serial_anchor, timings)
    if serial_anchor and draft_n > 0 and anchor["draft_anchor_n"] == 0:
        raise RuntimeError("drafted request did not execute a serial anchor")
    if response.get("truncated") is True:
        raise RuntimeError("completion unexpectedly truncated its prompt")
    wall_s = ended - started
    content = response.get("content", "")
    if not isinstance(content, str):
        raise RuntimeError("completion content is not a string")
    return {
        "slot": slot,
        "width": width,
        "wall_s": wall_s,
        "wall_tps": n_predict / wall_s,
        "predicted_tps": float(predicted_tps),
        "predicted_n": predicted_n,
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "tokens_cached": response.get("tokens_cached"),
        "tokens_evaluated": response.get("tokens_evaluated"),
        "tokens": tokens,
        "content": content,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "draft_n_per_round": proposed,
        "draft_n_accepted_per_round": accepted,
        **anchor,
    }


def midpoint_gain(candidate: float, control_a: float, control_b: float) -> float:
    midpoint = (control_a + control_b) / 2.0
    return 100.0 * (candidate / midpoint - 1.0)


def summarize_matched(samples: list[dict[str, Any]], widths: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controls = [sample for sample in samples if sample["width"] == 0]
    for width in widths:
        selected = [sample for sample in samples if sample["width"] == width]
        gains_server: list[float] = []
        gains_wall: list[float] = []
        if width != 0:
            for sample in selected:
                repeat = sample["repeat"]
                before = next(
                    control for control in controls
                    if control["repeat"] == repeat and control["control_phase"] == "before"
                )
                after = next(
                    control for control in controls
                    if control["repeat"] == repeat and control["control_phase"] == "after"
                )
                gains_server.append(
                    midpoint_gain(sample["predicted_tps"], before["predicted_tps"], after["predicted_tps"])
                )
                gains_wall.append(midpoint_gain(sample["wall_tps"], before["wall_tps"], after["wall_tps"]))
        drafted = sum(int(sample["draft_n"]) for sample in selected)
        accepted = sum(int(sample["draft_n_accepted"]) for sample in selected)
        anchor_n = sum(int(sample.get("draft_anchor_n") or 0) for sample in selected)
        anchor_match_n = sum(int(sample.get("draft_anchor_match_n") or 0) for sample in selected)
        anchor_fallback_n = sum(int(sample.get("draft_anchor_fallback_n") or 0) for sample in selected)
        rows.append({
            "width": width,
            "samples": len(selected),
            "server_tps": metrics([float(sample["predicted_tps"]) for sample in selected]),
            "wall_tps": metrics([float(sample["wall_tps"]) for sample in selected]),
            "paired_server_gain_pct": metrics(gains_server),
            "paired_wall_gain_pct": metrics(gains_wall),
            "draft_n": drafted,
            "draft_n_accepted": accepted,
            "acceptance": accepted / drafted if drafted else None,
            "draft_anchor_n": anchor_n,
            "draft_anchor_match_n": anchor_match_n,
            "draft_anchor_fallback_n": anchor_fallback_n,
            "draft_anchor_match_rate": anchor_match_n / anchor_n if anchor_n else None,
            "proposal_coverage": (
                sum(int(sample["draft_n"] > 0) for sample in selected) / len(selected)
                if selected else None
            ),
            "greedy_parity_rate": (
                sum(int(bool(sample.get("greedy_parity"))) for sample in selected) / len(selected)
                if selected else None
            ),
            "greedy_parity": all(bool(sample.get("greedy_parity")) for sample in selected),
        })
    return rows


def run_single_suite(
    base_url: str,
    filler: list[int],
    suffixes: dict[str, list[int]],
    depths: list[int],
    widths: list[int],
    repeats: int,
    n_predict: int,
    seed: int,
    timeout: float,
    sample_file: Any,
    reuse_case_prefix: bool,
    serial_anchor: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_samples: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {}
    nonzero = [width for width in widths if width != 0]
    for depth in depths:
        if reuse_case_prefix:
            erase_slot(base_url, 0, timeout)
        for case_index, (case_id, suffix) in enumerate(suffixes.items()):
            prompt = build_prompt(filler, suffix, depth)
            if not reuse_case_prefix:
                erase_slot(base_url, 0, timeout)
            warmup = completion(base_url, prompt, 0, 0, min(16, n_predict), seed, timeout, serial_anchor)
            if warmup["draft_n"] != 0:
                raise RuntimeError("single-suite warmup unexpectedly drafted tokens")
            case_samples: list[dict[str, Any]] = []
            for repeat in range(repeats):
                offset = repeat % max(1, len(nonzero))
                ordered = nonzero[offset:] + nonzero[:offset]
                request_widths = [0, *ordered, 0]
                control_tokens: list[int] | None = None
                for order_index, width in enumerate(request_widths):
                    print(
                        f"single depth={depth} case={case_id} repeat={repeat + 1}/{repeats} "
                        f"width={width}",
                        flush=True,
                    )
                    sample = completion(
                        base_url,
                        prompt,
                        0,
                        width,
                        n_predict,
                        seed + case_index,
                        timeout,
                        serial_anchor,
                    )
                    phase = (
                        "before" if order_index == 0
                        else "after" if order_index == len(request_widths) - 1
                        else None
                    )
                    if control_tokens is None:
                        control_tokens = sample["tokens"]
                    sample.update({
                        "suite": "single-depth",
                        "case_id": case_id,
                        "depth": depth,
                        "repeat": repeat,
                        "order": order_index,
                        "control_phase": phase,
                        "greedy_parity": sample["tokens"] == control_tokens,
                    })
                    case_samples.append(sample)
                    all_samples.append(sample)
                    sample_file.write(json.dumps(sample, sort_keys=True) + "\n")
                    sample_file.flush()
            key = f"{depth}:{case_id}"
            summaries[key] = summarize_matched(case_samples, widths)
            if not reuse_case_prefix:
                erase_slot(base_url, 0, timeout)
        if reuse_case_prefix:
            erase_slot(base_url, 0, timeout)
    return all_samples, summaries


def run_wave(
    base_url: str,
    prompt: list[int],
    active_agents: int,
    width: int,
    n_predict: int,
    seed: int,
    timeout: float,
    serial_anchor: bool,
) -> dict[str, Any]:
    barrier = threading.Barrier(active_agents)

    def one(slot: int) -> dict[str, Any]:
        barrier.wait()
        return completion(base_url, prompt, slot, width, n_predict, seed + slot, timeout, serial_anchor)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=active_agents) as executor:
        requests = list(executor.map(one, range(active_agents)))
    wall_s = time.perf_counter() - started
    return {
        "width": width,
        "active_agents": active_agents,
        "wall_s": wall_s,
        "wall_tps": active_agents * n_predict / wall_s,
        "predicted_tps": statistics.mean(float(request["predicted_tps"]) for request in requests),
        "predicted_n": active_agents * n_predict,
        "draft_n": sum(int(request["draft_n"]) for request in requests),
        "draft_n_accepted": sum(int(request["draft_n_accepted"]) for request in requests),
        "draft_anchor_n": sum(int(request["draft_anchor_n"]) for request in requests),
        "draft_anchor_match_n": sum(int(request["draft_anchor_match_n"]) for request in requests),
        "draft_anchor_fallback_n": sum(int(request["draft_anchor_fallback_n"]) for request in requests),
        "requests": requests,
    }


def run_concurrency_suite(
    base_url: str,
    filler: list[int],
    suffixes: dict[str, list[int]],
    depth: int,
    widths: list[int],
    repeats: int,
    n_predict: int,
    seed: int,
    active_agents: int,
    timeout: float,
    sample_file: Any,
    reuse_case_prefix: bool,
    serial_anchor: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_samples: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {}
    nonzero = [width for width in widths if width != 0]
    if reuse_case_prefix:
        for slot in range(active_agents):
            erase_slot(base_url, slot, timeout)
    for case_index, (case_id, suffix) in enumerate(suffixes.items()):
        prompt = build_prompt(filler, suffix, depth)
        if not reuse_case_prefix:
            for slot in range(active_agents):
                erase_slot(base_url, slot, timeout)
        run_wave(base_url, prompt, active_agents, 0, min(16, n_predict), seed, timeout, serial_anchor)
        case_samples: list[dict[str, Any]] = []
        for repeat in range(repeats):
            offset = repeat % max(1, len(nonzero))
            ordered = nonzero[offset:] + nonzero[:offset]
            request_widths = [0, *ordered, 0]
            control_tokens: list[list[int]] | None = None
            for order_index, width in enumerate(request_widths):
                print(
                    f"concurrency agents={active_agents} case={case_id} repeat={repeat + 1}/{repeats} "
                    f"width={width}",
                    flush=True,
                )
                sample = run_wave(
                    base_url,
                    prompt,
                    active_agents,
                    width,
                    n_predict,
                    seed + case_index * 1000,
                    timeout,
                    serial_anchor,
                )
                phase = (
                    "before" if order_index == 0
                    else "after" if order_index == len(request_widths) - 1
                    else None
                )
                tokens = [request["tokens"] for request in sample["requests"]]
                if control_tokens is None:
                    control_tokens = tokens
                sample.update({
                    "suite": "concurrency",
                    "case_id": case_id,
                    "depth": depth,
                    "repeat": repeat,
                    "order": order_index,
                    "control_phase": phase,
                    "greedy_parity": tokens == control_tokens,
                })
                case_samples.append(sample)
                all_samples.append(sample)
                sample_file.write(json.dumps(sample, sort_keys=True) + "\n")
                sample_file.flush()
        summaries[case_id] = summarize_matched(case_samples, widths)
        if not reuse_case_prefix:
            for slot in range(active_agents):
                erase_slot(base_url, slot, timeout)
    if reuse_case_prefix:
        for slot in range(active_agents):
            erase_slot(base_url, slot, timeout)
    return all_samples, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--depths", default=DEFAULT_DEPTHS)
    parser.add_argument("--widths", default=DEFAULT_WIDTHS)
    parser.add_argument("--concurrency-widths", default=DEFAULT_CONCURRENCY_WIDTHS)
    parser.add_argument("--cases", default=",".join(CASE_SUFFIXES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency-repeats", type=int, default=5)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict-parity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-concurrency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-case-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serial-anchor", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    depths = parse_int_list(args.depths)
    widths = parse_int_list(args.widths, require_zero=True)
    concurrency_widths = parse_int_list(args.concurrency_widths, require_zero=True)
    case_ids = [case.strip() for case in args.cases.split(",") if case.strip()]
    if any(case not in CASE_SUFFIXES for case in case_ids) or not case_ids:
        raise SystemExit(f"--cases must select from {','.join(CASE_SUFFIXES)}")
    if min(depths) <= 0 or max(depths) + args.n_predict >= args.ctx:
        raise SystemExit("depths must be positive and leave room for generated tokens")
    if args.repeats <= 0 or args.concurrency_repeats <= 0 or args.n_predict <= 0:
        raise SystemExit("repeat counts and n-predict must be positive")
    if args.parallel <= 0:
        raise SystemExit("parallel must be positive")

    base_url = f"http://127.0.0.1:{args.port}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples_path = args.out.with_suffix(".samples.jsonl")
    result: dict[str, Any] = {
        "kind": "treebeard-wavefront-b70",
        "started": now(),
        "config": {
            "port": args.port,
            "ctx": args.ctx,
            "parallel": args.parallel,
            "depths": depths,
            "widths": widths,
            "concurrency_widths": concurrency_widths,
            "cases": case_ids,
            "repeats": args.repeats,
            "concurrency_repeats": args.concurrency_repeats,
            "n_predict": args.n_predict,
            "seed": args.seed,
            "strict_parity": args.strict_parity,
            "run_concurrency": args.run_concurrency,
            "reuse_case_prefix": args.reuse_case_prefix,
            "serial_anchor": args.serial_anchor,
        },
        "passed": False,
        "failures": [],
    }
    try:
        props = http_json("GET", f"{base_url}/props", timeout=30.0)
        if not isinstance(props, dict):
            raise RuntimeError("props response is not an object")
        live_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        if props.get("total_slots") != args.parallel or live_ctx != args.ctx:
            raise RuntimeError(
                f"server shape was slots={props.get('total_slots')!r}, ctx={live_ctx!r}; "
                f"expected slots={args.parallel}, ctx={args.ctx}"
            )
        result["server_props"] = props
        suffixes = {
            case: tokenize(base_url, CASE_SUFFIXES[case], False, args.timeout)
            for case in case_ids
        }
        result["suffix_tokens"] = {case: len(tokens) for case, tokens in suffixes.items()}
        filler = build_filler(base_url, max(depths), args.timeout)
        result["filler_tokens_built"] = len(filler)

        with samples_path.open("w", encoding="utf-8") as sample_file:
            single_samples, single_summary = run_single_suite(
                base_url,
                filler,
                suffixes,
                depths,
                widths,
                args.repeats,
                args.n_predict,
                args.seed,
                args.timeout,
                sample_file,
                args.reuse_case_prefix,
                args.serial_anchor,
            )
            if args.run_concurrency:
                concurrency_samples, concurrency_summary = run_concurrency_suite(
                    base_url,
                    filler,
                    suffixes,
                    min(depths),
                    concurrency_widths,
                    args.concurrency_repeats,
                    args.n_predict,
                    args.seed + 100000,
                    args.parallel,
                    args.timeout,
                    sample_file,
                    args.reuse_case_prefix,
                    args.serial_anchor,
                )
            else:
                concurrency_samples, concurrency_summary = [], {}
        result["single_samples"] = single_samples
        result["single_summary"] = single_summary
        result["concurrency_samples"] = concurrency_samples
        result["concurrency_summary"] = concurrency_summary
        result["parity_failures"] = sum(
            int(not bool(sample.get("greedy_parity")))
            for sample in [*single_samples, *concurrency_samples]
        )
        result["promotion_parity_passed"] = result["parity_failures"] == 0
        if args.strict_parity and result["parity_failures"]:
            raise RuntimeError(f"exact greedy parity failed in {result['parity_failures']} measured waves")
        result["passed"] = True
    except Exception as exc:  # noqa: BLE001
        result["failures"].append(str(exc))
    finally:
        for slot in range(args.parallel):
            try:
                erase_slot(base_url, slot, args.timeout)
            except Exception as exc:  # noqa: BLE001
                result["failures"].append(f"cleanup slot {slot}: {exc}")
        result["finished"] = now()
        result["passed"] = not result["failures"]
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"result": str(args.out), "samples": str(samples_path), "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
