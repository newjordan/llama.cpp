#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_WIDTHS = "0,1,2,4,8,12,24,48"
DEFAULT_PROMPTS = [
    {
        "id": "structured-copy",
        "prompt": (
            "Repeat only the payload below, byte for byte, with no heading or commentary.\n"
            "<payload>\n"
            "{\"service\":\"treebeard\",\"state\":\"active\",\"slots\":12,\"context\":262144}\n"
            "{\"service\":\"treebeard\",\"state\":\"active\",\"slots\":12,\"context\":262144}\n"
            "{\"service\":\"treebeard\",\"state\":\"active\",\"slots\":12,\"context\":262144}\n"
            "{\"service\":\"treebeard\",\"state\":\"active\",\"slots\":12,\"context\":262144}\n"
            "</payload>\n"
        ),
    },
    {
        "id": "code-edit",
        "prompt": (
            "Return only the following function with every occurrence of old_state changed to new_state.\n"
            "def update(old_state):\n"
            "    old_state[\"count\"] = old_state.get(\"count\", 0) + 1\n"
            "    old_state[\"ready\"] = old_state[\"count\"] > 3\n"
            "    old_state[\"label\"] = \"ready\" if old_state[\"ready\"] else \"waiting\"\n"
            "    return old_state\n"
        ),
    },
    {
        "id": "free-prose",
        "prompt": (
            "Explain in one compact paragraph why parallel aggregate throughput does not automatically "
            "reduce the latency of one autoregressive response."
        ),
    },
]


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_widths(value: str) -> list[int]:
    widths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not widths or min(widths) < 0:
        raise argparse.ArgumentTypeError("widths must be non-negative integers")
    if len(widths) != len(set(widths)):
        raise argparse.ArgumentTypeError("widths must not contain duplicates")
    if 0 not in widths:
        raise argparse.ArgumentTypeError("widths must include 0 as the control")
    return widths


def load_cases(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("case file must contain a JSON array")
    cases: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise SystemExit(f"case {index} must be an object with a string prompt")
        cases.append({"id": str(item.get("id") or f"case-{index:03d}"), "prompt": item["prompt"]})
    if not cases:
        raise SystemExit("case file is empty")
    return cases


def run_request(
    port: int,
    case: dict[str, str],
    width: int,
    n_predict: int,
    seed: int,
    timeout: float,
    serial_anchor: bool,
) -> dict[str, Any]:
    payload = {
        "prompt": case["prompt"],
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": False,
        "return_tokens": True,
        "ignore_eos": True,
        "stop": [],
        "speculative.n_max": width,
        "speculative.serial_anchor": serial_anchor,
    }
    started = time.perf_counter()
    response = http_json("POST", f"http://127.0.0.1:{port}/completion", payload, timeout)
    wall_s = time.perf_counter() - started
    timings = response.get("timings") or {}
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens):
        raise RuntimeError("server response did not contain an integer token list")
    anchor_n = int(timings.get("draft_anchor_n") or 0)
    anchor_match_n = int(timings.get("draft_anchor_match_n") or 0)
    anchor_fallback_n = int(timings.get("draft_anchor_fallback_n") or 0)
    anchor_serial = list(timings.get("draft_anchor_serial_tokens") or [])
    anchor_batched = list(timings.get("draft_anchor_batched_tokens") or [])
    audit_tokens = int(timings.get("draft_audit_tokens") or 0)
    audit_matched = int(timings.get("draft_audit_tokens_matched") or 0)
    audit_fallback_n = int(timings.get("draft_audit_fallback_n") or 0)
    audit_first_mismatch = list(timings.get("draft_audit_first_mismatch") or [])
    audit_serial = list(timings.get("draft_audit_serial_tokens") or [])
    audit_batched = list(timings.get("draft_audit_batched_tokens") or [])
    draft_n = int(timings.get("draft_n") or 0)
    if anchor_n != len(anchor_serial) or anchor_n != len(anchor_batched):
        raise RuntimeError("serial anchor token arrays are not aligned")
    if anchor_match_n + anchor_fallback_n != anchor_n:
        raise RuntimeError("serial anchor outcomes do not match the anchor total")
    if sum(a == b for a, b in zip(anchor_serial, anchor_batched)) != anchor_match_n:
        raise RuntimeError("serial anchor token comparisons do not match the reported outcomes")
    audit_values = [
        audit_tokens,
        audit_matched,
        audit_fallback_n,
        *audit_first_mismatch,
        *audit_serial,
        *audit_batched,
    ]
    if not all(type(value) is int for value in audit_values):
        raise RuntimeError("serial audit telemetry contains a non-integer")
    if audit_matched + audit_fallback_n != audit_tokens:
        raise RuntimeError("serial audit comparisons do not match the audited token total")
    if len(audit_first_mismatch) != anchor_n:
        raise RuntimeError("serial audit rounds do not match the anchor total")
    if sum(index >= 0 for index in audit_first_mismatch) != audit_fallback_n:
        raise RuntimeError("serial audit mismatch columns do not match the fallback total")
    if any(index < -1 or index > width for index in audit_first_mismatch):
        raise RuntimeError("serial audit contains an invalid mismatch column")
    if len(audit_serial) != audit_fallback_n or len(audit_batched) != audit_fallback_n:
        raise RuntimeError("serial audit mismatch token arrays are not aligned")
    if any(serial == batched for serial, batched in zip(audit_serial, audit_batched)):
        raise RuntimeError("serial audit mismatch token arrays contain a match")
    if (width == 0 or not serial_anchor) and (anchor_n or audit_tokens or audit_first_mismatch):
        raise RuntimeError("serial anchor telemetry was emitted for an unanchored request")
    if serial_anchor and draft_n > 0 and anchor_n == 0:
        raise RuntimeError("drafted request did not execute a serial anchor")
    return {
        "case_id": case["id"],
        "width": width,
        "wall_s": wall_s,
        "tokens": tokens,
        "predicted_n": int(timings.get("predicted_n") or len(tokens)),
        "predicted_tps": timings.get("predicted_per_second"),
        "draft_n": draft_n,
        "draft_n_accepted": int(timings.get("draft_n_accepted") or 0),
        "draft_n_per_round": list(timings.get("draft_n_per_round") or []),
        "draft_n_accepted_per_round": list(timings.get("draft_n_accepted_per_round") or []),
        "draft_anchor_n": anchor_n,
        "draft_anchor_match_n": anchor_match_n,
        "draft_anchor_fallback_n": anchor_fallback_n,
        "draft_anchor_serial_tokens": anchor_serial,
        "draft_anchor_batched_tokens": anchor_batched,
        "draft_audit_tokens": audit_tokens,
        "draft_audit_tokens_matched": audit_matched,
        "draft_audit_fallback_n": audit_fallback_n,
        "draft_audit_first_mismatch": audit_first_mismatch,
        "draft_audit_serial_tokens": audit_serial,
        "draft_audit_batched_tokens": audit_batched,
    }


def summarize(samples: list[dict[str, Any]], widths: list[int]) -> list[dict[str, Any]]:
    control_by_case: dict[str, list[int]] = {}
    for sample in samples:
        if sample["width"] == 0 and sample["case_id"] not in control_by_case:
            control_by_case[sample["case_id"]] = sample["tokens"]

    rows: list[dict[str, Any]] = []
    for width in widths:
        selected = [sample for sample in samples if sample["width"] == width]
        walls = [float(sample["wall_s"]) for sample in selected]
        drafted = sum(int(sample["draft_n"]) for sample in selected)
        accepted = sum(int(sample["draft_n_accepted"]) for sample in selected)
        anchor_n = sum(int(sample.get("draft_anchor_n") or 0) for sample in selected)
        anchor_match_n = sum(int(sample.get("draft_anchor_match_n") or 0) for sample in selected)
        anchor_fallback_n = sum(int(sample.get("draft_anchor_fallback_n") or 0) for sample in selected)
        audit_tokens = sum(int(sample.get("draft_audit_tokens") or 0) for sample in selected)
        audit_matched = sum(int(sample.get("draft_audit_tokens_matched") or 0) for sample in selected)
        audit_fallback_n = sum(int(sample.get("draft_audit_fallback_n") or 0) for sample in selected)
        mismatch_counts: dict[str, int] = {}
        for sample in selected:
            for column in sample.get("draft_audit_first_mismatch") or []:
                if column >= 0:
                    key = str(column)
                    mismatch_counts[key] = mismatch_counts.get(key, 0) + 1
        predicted = sum(int(sample["predicted_n"]) for sample in selected)
        parity = [sample["tokens"] == control_by_case.get(sample["case_id"]) for sample in selected]
        rows.append(
            {
                "width": width,
                "samples": len(selected),
                "wall_p50_s": statistics.median(walls),
                "wall_mean_s": statistics.mean(walls),
                "aggregate_tps": predicted / sum(walls),
                "draft_n": drafted,
                "draft_n_accepted": accepted,
                "acceptance": accepted / drafted if drafted else None,
                "draft_rounds": sum(len(sample["draft_n_per_round"]) for sample in selected),
                "draft_anchor_n": anchor_n,
                "draft_anchor_match_n": anchor_match_n,
                "draft_anchor_fallback_n": anchor_fallback_n,
                "draft_anchor_match_rate": anchor_match_n / anchor_n if anchor_n else None,
                "draft_audit_tokens": audit_tokens,
                "draft_audit_tokens_matched": audit_matched,
                "draft_audit_fallback_n": audit_fallback_n,
                "draft_audit_match_rate": audit_matched / audit_tokens if audit_tokens else None,
                "draft_audit_first_mismatch_counts": mismatch_counts,
                "greedy_parity": all(parity),
            }
        )

    control = next(row for row in rows if row["width"] == 0)
    for row in rows:
        row["wall_p50_speedup"] = control["wall_p50_s"] / row["wall_p50_s"]
        row["aggregate_tps_gain_pct"] = 100.0 * (row["aggregate_tps"] / control["aggregate_tps"] - 1.0)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep per-request verified-lookahead widths on one warmed server.")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--widths", default=DEFAULT_WIDTHS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cases", help="JSON array of objects with id and prompt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--strict-parity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aba", action=argparse.BooleanOptionalAction, default=True, help="Bracket each rotated width sweep with width-0 controls")
    parser.add_argument("--serial-anchor", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    widths = parse_widths(args.widths)
    if args.repeats < 1 or args.n_predict < 1:
        raise SystemExit("--repeats and --n-predict must be positive")
    cases = load_cases(args.cases)
    props = http_json("GET", f"http://127.0.0.1:{args.port}/props", None, args.timeout)

    samples: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for case_index, case in enumerate(cases):
            nonzero_widths = [width for width in widths if width != 0]
            offset = repeat % max(1, len(nonzero_widths))
            ordered_widths = nonzero_widths[offset:] + nonzero_widths[:offset]
            request_widths = [0, *ordered_widths, 0] if args.aba else widths
            for request_index, width in enumerate(request_widths):
                sample = run_request(
                    args.port,
                    case,
                    width,
                    args.n_predict,
                    args.seed + repeat * 1000 + case_index,
                    args.timeout,
                    args.serial_anchor,
                )
                sample["repeat"] = repeat
                sample["order"] = request_index
                sample["control_phase"] = (
                    "before" if args.aba and request_index == 0
                    else "after" if args.aba and request_index == len(request_widths) - 1
                    else None
                )
                samples.append(sample)

    summary = summarize(samples, widths)
    case_summaries = {
        case["id"]: summarize(
            [sample for sample in samples if sample["case_id"] == case["id"]],
            widths,
        )
        for case in cases
    }
    artifact = {
        "kind": "treebeard-wavefront-width-sweep",
        "server_props": props,
        "config": {
            "port": args.port,
            "widths": widths,
            "repeats": args.repeats,
            "n_predict": args.n_predict,
            "seed": args.seed,
            "aba": args.aba,
            "serial_anchor": args.serial_anchor,
            "cases": [case["id"] for case in cases],
        },
        "summary": summary,
        "case_summaries": case_summaries,
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("width  p50_s  speedup  agg_tps  gain_pct  accept  parity")
    for row in summary:
        acceptance = "n/a" if row["acceptance"] is None else f"{row['acceptance']:.3f}"
        print(
            f"{row['width']:>5}  {row['wall_p50_s']:>6.3f}  {row['wall_p50_speedup']:>7.3f}  "
            f"{row['aggregate_tps']:>7.2f}  {row['aggregate_tps_gain_pct']:>8.2f}  "
            f"{acceptance:>6}  {str(row['greedy_parity']).lower()}"
        )
    print(f"artifact: {out}")

    if args.strict_parity and not all(row["greedy_parity"] for row in summary):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
