#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BIN = "/home/frosty40/builds/turbo-experimental-build/bin/llama-server"
DEFAULT_MODEL = "/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
DEFAULT_OUT_DIR = "/tmp/turbo-speculative-breakout"
DEFAULT_SLOT_SAVE_PATH = "/tmp/turbo-speculative-breakout-slots"
RESULT_SCHEMA_VERSION = 3
DEFAULT_STOPS = [
    "</think>",
    "\n\n<task>",
    "\n<task>",
    "\n\nAnswer:",
    "\n\nCandidate answer:",
    "\n\nFinal answer:",
    "\n\nRepaired final answer:",
    "\n\nAssessment:",
    "\n\nVerification:",
    "\n\n<think>",
    "\n<think>",
    "\n\nWait,",
    "\nWait,",
    "\n\nLet's ",
    "\nLet's ",
    "\n\nI need ",
    "\nI need ",
]
OBJECTIVE_REPAIR_STOPS = [
    "\n\nTask:",
    "\n\nCurrent answer:",
    "\n\nFailed checks:",
    "\n\nHint:",
    "\n\nExplanation:",
    "\n\nNotes:",
    "\n\n<task>",
    "\n<task>",
]

SCORE_RE = re.compile(r"(?:score|rating)\s*[:=]\s*(\d{1,3})(?:\s*/\s*100)?", re.IGNORECASE)
FRACTION_RE = re.compile(r"\b(\d{1,3})\s*/\s*100\b")
LEADING_SCORE_RE = re.compile(r"^\s*(\d{1,3})(?:\s*/\s*100)?\b")
PREFIX_OK_RE = re.compile(r"prefix[_ -]?ok\s*[:=]\s*(yes|no|true|false)", re.IGNORECASE)

BRANCH_STRATEGIES = [
    "exact arithmetic audit, then direct answer",
    "format compliance: validate every requested key and type",
    "independent checksum or consistency check before finalizing",
    "look for hidden constraints and edge cases",
    "optimize for factual precision and avoid speculation",
    "optimize for completeness without padding",
    "make assumptions explicit only when needed",
    "produce the shortest answer that is still complete",
    "stress-test the likely answer before writing it",
    "prefer concrete examples over abstractions",
    "resolve ambiguities in the most useful direction",
    "prioritize operational next steps",
    "prioritize correctness over elegance",
    "write the answer a strict reviewer would keep",
]

DEFAULT_TASK_SUITES = {
    "turbo-smoke": [
        (
            "Draft a concise acceptance rubric and minimal benchmark plan for a speculative branch-and-verify harness. "
            "Include exactly 4 bullets named Baseline, Branch fanout, Verifier quality, and Pass/fail gate. "
            "Each bullet must include one measurable signal."
        ),
        (
            "You are reviewing a benchmark result where baseline score is 88, breakout score is 95, "
            "and one selected branch scored 100 in branch verification but 92 in final scoring. "
            "Give a 5-line diagnosis with one concrete next experiment per line."
        ),
        (
            "Write a minimal JSON schema for recording a speculative breakout run. "
            "It must include task_id, baseline_score, breakout_score, score_delta, selected_branch, "
            "and artifact_path, with types and one required list."
        ),
    ],
}

DEFAULT_BENCHMARK_SUITES = {
    "objective-smoke": [
        {
            "id": "arithmetic-json",
            "validator": "arithmetic_json",
            "repair_hint": (
                "For gamma, compute 2^10 as 1024, then subtract 37 by subtracting 30 and then 7. "
                "Do not keep gamma=995 or checksum=1475; those are the failed values. "
                "Then recompute checksum from alpha+beta+gamma."
            ),
            "task": (
                "Return only a compact JSON object, no markdown. Compute exactly: "
                "alpha = 17*23+19; beta = (144/12)+58; gamma = 2^10-37; "
                "checksum = alpha+beta+gamma. Keys must be exactly alpha, beta, gamma, checksum."
            ),
        },
        {
            "id": "schema-fields",
            "validator": "schema_fields",
            "task": (
                "Return only a minimal JSON schema object, no markdown. It must describe an object with "
                "task_id string, baseline_score number, breakout_score number, score_delta number, "
                "selected_branch object, and artifact_path string. Include a required list containing exactly "
                "those six field names."
            ),
        },
        {
            "id": "five-experiments",
            "validator": "five_experiments",
            "task": (
                "Return exactly five non-empty lines and no other text. The lines must start with "
                "Experiment 1:, Experiment 2:, Experiment 3:, Experiment 4:, Experiment 5:. "
                "In order, the lines must mention baseline_score, breakout_score, score_delta, "
                "branch_verify_score, and final_score."
            ),
        },
        {
            "id": "csv-json-transform",
            "validator": "csv_json_transform",
            "task": (
                "Return only a JSON array, no markdown. Convert these rows to objects sorted by id: "
                "id=3,name=Branch,score=88; id=1,name=Baseline,score=82; id=2,name=Breakout,score=95. "
                "Each object must include id, name, score, and delta_from_baseline where delta_from_baseline = score - 82."
            ),
        },
        {
            "id": "rubric-signals",
            "validator": "rubric_signals",
            "task": (
                "Return exactly four hyphen bullets and no heading. Bullet labels must be exactly Baseline, "
                "Branch fanout, Verifier quality, and Pass/fail gate. Each bullet must include at least one digit."
            ),
        },
    ],
}

DEFAULT_BENCHMARK_SUITES["objective-core"] = [
    *DEFAULT_BENCHMARK_SUITES["objective-smoke"],
    {
        "id": "nested-batch-json",
        "validator": "json_exact",
        "expected": {
            "sums": {"A": 59, "B": 34, "C": 40},
            "grand_total": 133,
            "largest_batch": "A",
        },
        "repair_hint": "Add each batch separately, compare the three batch totals, then sum the batch totals.",
        "task": (
            "Return only a compact JSON object, no markdown. Batch A has 17, 23, 19; "
            "batch B has 8, 5, 21; batch C has 13, 13, 14. Return keys exactly "
            "sums, grand_total, largest_batch. sums maps A, B, C to totals. "
            "largest_batch is the batch with the greatest total."
        ),
    },
    {
        "id": "filter-sort-json",
        "validator": "json_exact",
        "expected": [4, 2, 5],
        "repair_hint": "Keep only active records with score at least 85, then sort by score descending and id ascending for ties.",
        "task": (
            "Return only a JSON array of ids, no markdown. Records: "
            "id=7,status=hold,score=91; id=2,status=active,score=87; "
            "id=5,status=active,score=87; id=4,status=active,score=93; "
            "id=9,status=active,score=72. Keep only status active with score >= 85. "
            "Sort by score descending, breaking ties by id ascending."
        ),
    },
    {
        "id": "matrix-json",
        "validator": "json_exact",
        "expected": {
            "row_sums": [16, 22, 21],
            "col_sums": [20, 16, 23],
            "main_diag": 16,
            "anti_diag": 11,
            "diag_delta": 5,
        },
        "repair_hint": "For columns, add vertically. The main diagonal is top-left to bottom-right. For anti_diag, add the top-right cell 5, center cell 2, and bottom-left cell 4; do not reuse a row or column total.",
        "task": (
            "Return only a compact JSON object, no markdown. Matrix rows are [3,8,5], "
            "[13,2,7], [4,6,11]. Output keys exactly row_sums, col_sums, "
            "main_diag, anti_diag, diag_delta. diag_delta = main_diag - anti_diag."
        ),
    },
    {
        "id": "priority-lines",
        "validator": "lines_exact",
        "expected_lines": [
            "1. A2 - ml - p1",
            "2. B1 - api - p1",
            "3. C4 - db - p2",
            "4. D7 - ui - p2",
            "5. K3 - infra - p3",
            "6. Z9 - ops - p3",
        ],
        "repair_hint": "Sort by priority ascending; when priorities match, sort codes alphabetically.",
        "task": (
            "Return exactly six non-empty lines and no extra text. Reorder codes by "
            "priority ascending, then code alphabetical: code=Z9 priority=3 owner=ops; "
            "code=A2 priority=1 owner=ml; code=C4 priority=2 owner=db; "
            "code=B1 priority=1 owner=api; code=D7 priority=2 owner=ui; "
            "code=K3 priority=3 owner=infra. Each line format must be: "
            "<rank>. <code> - <owner> - p<priority>"
        ),
    },
    {
        "id": "json-merge-patches",
        "validator": "json_exact",
        "expected": {"alpha": True, "delta": "new", "gamma": True},
        "repair_hint": "Start with the base object, apply patches in order, and remove beta entirely rather than setting it to null.",
        "task": (
            "Return only a compact JSON object, no markdown. Base flags are "
            "alpha=false, beta=true, gamma=false. Apply patches in order: set alpha=true; "
            "add delta=\"new\"; set gamma=true; remove beta. Return the final object."
        ),
    },
    {
        "id": "duration-minutes-json",
        "validator": "json_exact",
        "expected": {
            "minutes": {"build": 135, "verify": 47, "deploy": 68},
            "total_minutes": 250,
        },
        "repair_hint": "Convert each hour to 60 minutes, add remaining minutes, then sum all converted durations.",
        "task": (
            "Return only a compact JSON object, no markdown. Convert durations to minutes: "
            "build=2h15m, verify=47m, deploy=1h08m. Return keys exactly minutes "
            "and total_minutes. minutes maps build, verify, deploy to minute counts."
        ),
    },
]


@dataclass
class RequestResult:
    ok: bool
    label: str
    id_slot: int | None
    start: float
    end: float
    wall_s: float
    raw_content: str
    content: str
    prompt_tokens: int
    predicted_tokens: int
    prompt_tps: float | None
    predicted_tps: float | None
    cache_tokens: int | None = None
    tokens: list[int] | None = None
    raw_content_sha256: str | None = None
    content_sha256: str | None = None
    tokens_sha256: str | None = None
    error: str | None = None


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def read_text_arg(value: str | None, file_path: str | None) -> str:
    if value and file_path:
        raise SystemExit("use either --task or --task-file, not both")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    if value:
        return value
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("provide --task, --task-file, or stdin")


def read_task_suite_file(path: str) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        tasks = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
            if isinstance(item, str):
                tasks.append(item)
            elif isinstance(item, dict):
                task = item.get("task") or item.get("prompt")
                if not isinstance(task, str) or not task.strip():
                    raise SystemExit(f"{path}:{lineno}: expected string field 'task' or 'prompt'")
                tasks.append(task)
            else:
                raise SystemExit(f"{path}:{lineno}: expected JSON string or object")
        return tasks
    if "\n---\n" in raw:
        return [part.strip() for part in raw.split("\n---\n") if part.strip()]
    return [line.strip() for line in raw.splitlines() if line.strip()]


def normalize_benchmark_case(item: Any, source: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise SystemExit(f"{source}: expected benchmark case object")
    task = item.get("task") or item.get("prompt")
    validator = item.get("validator")
    if not isinstance(task, str) or not task.strip():
        raise SystemExit(f"{source}: expected string field 'task' or 'prompt'")
    if not isinstance(validator, str) or not validator.strip():
        raise SystemExit(f"{source}: expected string field 'validator'")
    case = dict(item)
    case["task"] = task
    case["validator"] = validator
    digest = hashlib.sha1(task.encode("utf-8")).hexdigest()[:8]
    case["id"] = str(item.get("id") or f"{validator}-{digest}")
    return case


def read_benchmark_suite_file(path: str) -> list[dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            data = data["cases"]
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected JSON list or object with cases list")
        return [normalize_benchmark_case(item, f"{path}:{index + 1}") for index, item in enumerate(data)]

    cases = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
        cases.append(normalize_benchmark_case(item, f"{path}:{lineno}"))
    return cases


def resolve_tasks(args: argparse.Namespace) -> list[str]:
    tasks: list[str] = []
    if args.task_suite != "none":
        tasks.extend(DEFAULT_TASK_SUITES[args.task_suite])
    if args.task_suite_file:
        tasks.extend(read_task_suite_file(args.task_suite_file))
    if tasks:
        if args.task or args.task_file:
            raise SystemExit("use either suite mode or --task/--task-file, not both")
        return tasks
    return [read_text_arg(args.task, args.task_file)]


def resolve_benchmarks(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if args.benchmark_suite != "none":
        cases.extend(dict(case) for case in DEFAULT_BENCHMARK_SUITES[args.benchmark_suite])
    if args.benchmark_suite_file:
        cases.extend(read_benchmark_suite_file(args.benchmark_suite_file))
    if not cases:
        return []
    if args.task or args.task_file or args.task_suite != "none" or args.task_suite_file:
        raise SystemExit("use benchmark mode without --task, --task-file, --task-suite, or --task-suite-file")
    return cases


def parse_slot_list(value: str) -> list[int]:
    slots: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_raw, hi_raw = part.split("-", 1)
            lo = int(lo_raw)
            hi = int(hi_raw)
            if hi < lo:
                raise argparse.ArgumentTypeError(f"invalid descending slot range: {part}")
            slots.extend(range(lo, hi + 1))
        else:
            slots.append(int(part))
    if not slots:
        raise argparse.ArgumentTypeError("expected at least one slot")
    if len(set(slots)) != len(slots):
        raise argparse.ArgumentTypeError("duplicate slot id in --branch-slots")
    if min(slots) < 0:
        raise argparse.ArgumentTypeError("slot ids must be non-negative")
    return slots


def parse_fanout_stages(value: str, n_slots: int) -> list[int]:
    stages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        width = int(part)
        if width <= 0:
            raise argparse.ArgumentTypeError("fanout stages must be positive")
        width = min(width, n_slots)
        if not stages or width > stages[-1]:
            stages.append(width)
    if not stages:
        raise argparse.ArgumentTypeError("expected at least one fanout stage")
    return stages


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def wait_healthy(port: int, proc: subprocess.Popen[Any] | None, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    last_error = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"server did not become healthy on port {port}: {last_error}")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def clean_content(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<think>"):
        end = cleaned.find("</think>")
        if end >= 0:
            cleaned = cleaned[end + len("</think>") :].strip()
        else:
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:]).strip() if lines and lines[0].strip() == "<think>" else cleaned
    for marker in ("</task>", "<task>"):
        if cleaned.startswith(marker):
            cleaned = cleaned[len(marker) :].strip()
    for label in ("Corrected answer:", "Corrected JSON:", "Answer:"):
        if cleaned.lower().startswith(label.lower()):
            cleaned = cleaned[len(label) :].strip()
    cut_points = [cleaned.find(marker) for marker in DEFAULT_STOPS if cleaned.find(marker) > 0]
    if cut_points:
        cleaned = cleaned[: min(cut_points)].rstrip()
    return cleaned


def completion(
    port: int,
    label: str,
    prompt: str,
    n_predict: int,
    id_slot: int | None,
    cache_prompt: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    timeout: float,
    stop: list[str] | None = None,
    ignore_eos: bool = False,
    use_stop_strings: bool = True,
) -> RequestResult:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "cache_prompt": cache_prompt,
        "ignore_eos": ignore_eos,
        "stream": False,
        "return_tokens": True,
    }
    payload["stop"] = (DEFAULT_STOPS if stop is None else stop) if use_stop_strings else []
    if id_slot is not None:
        payload["id_slot"] = id_slot

    start = time.perf_counter()
    try:
        data = http_json("POST", f"http://127.0.0.1:{port}/completion", payload, timeout=timeout)
        end = time.perf_counter()
        if not isinstance(data, dict):
            raise ValueError("completion response must be a JSON object")
        raw_content = data.get("content")
        tokens = data.get("tokens")
        if not isinstance(raw_content, str):
            raise ValueError("completion response content must be a string")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("completion response tokens must be a list of integers")
        if id_slot is not None and data.get("id_slot") != id_slot:
            raise ValueError(
                f"completion response slot mismatch: requested {id_slot}, got {data.get('id_slot')!r}"
            )
        timings = data.get("timings", {})
        if not isinstance(timings, dict):
            raise ValueError("completion response timings must be a JSON object")
        predicted_tokens = int(timings.get("predicted_n") or 0)
        if len(tokens) != predicted_tokens:
            raise ValueError(
                f"completion token count mismatch: got {len(tokens)}, timings predicted_n={predicted_tokens}"
            )
        content = clean_content(raw_content)
        encoded_tokens = json.dumps(tokens, separators=(",", ":")).encode("ascii")
        return RequestResult(
            ok=True,
            label=label,
            id_slot=id_slot,
            start=start,
            end=end,
            wall_s=end - start,
            raw_content=raw_content,
            content=content,
            prompt_tokens=int(timings.get("prompt_n") or 0),
            predicted_tokens=predicted_tokens,
            prompt_tps=float(timings["prompt_per_second"]) if timings.get("prompt_per_second") is not None else None,
            predicted_tps=float(timings["predicted_per_second"]) if timings.get("predicted_per_second") is not None else None,
            cache_tokens=int(timings.get("cache_n") or 0),
            tokens=tokens,
            raw_content_sha256=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            tokens_sha256=hashlib.sha256(encoded_tokens).hexdigest(),
        )
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        return RequestResult(
            ok=False,
            label=label,
            id_slot=id_slot,
            start=start,
            end=end,
            wall_s=end - start,
            raw_content="",
            content="",
            prompt_tokens=0,
            predicted_tokens=0,
            prompt_tps=None,
            predicted_tps=None,
            error=str(exc),
        )


def slot_action(port: int, slot_id: int, action: str, filename: str | None = None, timeout: float = 300.0) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if filename is not None:
        payload["filename"] = filename
    result = http_json("POST", f"http://127.0.0.1:{port}/slots/{slot_id}?action={action}", payload, timeout=timeout)
    return result if isinstance(result, dict) else {}


def slot_fork(
    port: int,
    source_id: int,
    destinations: list[int],
    timeout: float = 300.0,
    fork_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"destinations": destinations}
    if fork_id is not None:
        payload["fork_id"] = fork_id
    result = http_json(
        "POST",
        f"http://127.0.0.1:{port}/slots/{source_id}?action=fork",
        payload,
        timeout=timeout,
    )
    if not isinstance(result, dict):
        raise RuntimeError("slot fork returned a non-object response")
    if result.get("id_slot") != source_id or result.get("destinations") != destinations:
        raise RuntimeError("slot fork response does not match the requested source and destinations")
    return result


def slot_commit(port: int, slot_id: int, fork_id: int, timeout: float = 300.0) -> dict[str, Any]:
    result = http_json(
        "POST",
        f"http://127.0.0.1:{port}/slots/{slot_id}?action=commit",
        {"fork_id": fork_id},
        timeout=timeout,
    )
    if not isinstance(result, dict) or result.get("id_slot") != slot_id:
        raise RuntimeError("slot commit response does not match the requested winner")
    return result


def pcbt_create(
    port: int,
    request_id: str,
    source_node_id: int,
    branches: list[dict[str, Any]],
    budget: dict[str, int],
    timeout: float = 300.0,
) -> dict[str, Any]:
    result = http_json(
        "POST",
        f"http://127.0.0.1:{port}/transactions",
        {
            "request_id": request_id,
            "source": {"node_id": source_node_id},
            "branches": branches,
            "budget": budget,
            "acceptance_contract": {"kind": "external", "name": "objective-validator-v1"},
        },
        timeout=timeout,
    )
    if not isinstance(result, dict) or "transaction_id" not in result:
        raise RuntimeError(f"pcbt create failed: {result}")
    return result


def pcbt_observe(port: int, transaction_id: int, timeout: float = 60.0) -> dict[str, Any]:
    result = http_json("GET", f"http://127.0.0.1:{port}/transactions/{transaction_id}", None, timeout=timeout)
    if not isinstance(result, dict) or "status" not in result:
        raise RuntimeError(f"pcbt observe failed: {result}")
    return result


def pcbt_commit_tx(
    port: int,
    transaction_id: int,
    winner_node_id: int,
    expected_fork_id: int,
    candidate_digest: str,
    evidence_digest: str,
    evidence_summary: dict[str, Any],
    timeout: float = 300.0,
) -> dict[str, Any]:
    result = http_json(
        "POST",
        f"http://127.0.0.1:{port}/transactions/{transaction_id}?action=commit",
        {
            "winner_node_id": winner_node_id,
            "expected_fork_id": expected_fork_id,
            "candidate_digest": candidate_digest,
            "evidence": {
                "kind": "external",
                "digest": evidence_digest,
                "summary": evidence_summary,
            },
        },
        timeout=timeout,
    )
    if not isinstance(result, dict) or "receipt_digest" not in result:
        raise RuntimeError(f"pcbt commit failed: {result}")
    return result


def pcbt_evidence_digest(report: dict[str, Any]) -> str:
    canon = json.dumps(report, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def slot_node_id(port: int, slot_id: int, timeout: float = 30.0) -> int:
    rows = http_json("GET", f"http://127.0.0.1:{port}/slots", None, timeout=timeout)
    for row in rows:
        if row.get("id") == slot_id:
            return int(row.get("node_id", -1))
    return -1


def cleanup_fork_reservations(
    port: int,
    slots: list[int],
    source_id: int,
    timeout: float,
) -> list[str]:
    try:
        slot_rows = http_json("GET", f"http://127.0.0.1:{port}/slots", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return [f"failed to inspect fork reservations: {exc}"]
    if not isinstance(slot_rows, list):
        return ["slots endpoint returned a non-array response during fork cleanup"]

    reserved = {
        int(row["id"])
        for row in slot_rows
        if isinstance(row, dict) and row.get("is_reserved") and isinstance(row.get("id"), int)
    }
    cleanup_order = [slot_id for slot_id in slots if slot_id != source_id] + [source_id]
    errors = []
    for slot_id in cleanup_order:
        if slot_id not in reserved:
            continue
        try:
            slot_action(port, slot_id, "erase", timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"slot {slot_id}: {exc}")
    return errors


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
        raise RuntimeError(f"port {args.port} is already listening; use --attach or a different --port")

    slot_save_path = Path(args.slot_save_path)
    slot_save_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["GGML_SYCL_ENABLE_FUSION"] = "1"

    server_cmd = [
        args.bin,
        "-m", args.model,
        "-ngl", str(args.ngl),
        "-ncmoe", str(args.ncmoe),
        "--no-op-offload",
        "-c", str(args.ctx),
        "-np", str(args.parallel),
        "-fa", args.flash_attn,
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--jinja",
        "--slot-save-path", str(slot_save_path),
        "--no-cache-idle-slots",
        "-a", f"turbo-spec-breakout-np{args.parallel}-c{args.ctx}",
    ]
    if args.kvu:
        server_cmd.insert(server_cmd.index("-fa"), "-kvu")
    if args.prefix_clone and args.prefix_clone_backend == "fork":
        server_cmd.append("--slots")
    if args.extra_server_args:
        server_cmd.extend(shlex.split(args.extra_server_args))

    cmd = server_cmd
    if args.source_oneapi:
        cmd = [
            "bash",
            "-lc",
            "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; "
            "export GGML_SYCL_ENABLE_FUSION=1; exec " + shlex.join(server_cmd),
        ]

    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    proc._turbo_log_file = log_f  # type: ignore[attr-defined]
    return proc


def base_prefix(task: str) -> str:
    return (
        "You are solving one user task. Use rigorous reasoning, but return only useful work.\n"
        "The final answer must be correct, concise, and directly responsive.\n\n"
        "Do not output hidden reasoning, scratch work, or thinking tags such as <think>.\n"
        "Use plain text by default. If bullets are useful, prefer simple hyphen bullets without decorative styling.\n\n"
        "<task>\n"
        f"{task.strip()}\n"
        "</task>\n\n"
    )


def baseline_prompt(task: str) -> str:
    return (
        base_prefix(task)
        + "Produce the best single-pass answer. Fit the answer in the token budget; do not trail off mid-sentence.\n\n"
        "Answer:\n"
    )


def branch_suffix(slot_id: int, branch_index: int, n_branches: int) -> str:
    strategy = BRANCH_STRATEGIES[branch_index % len(BRANCH_STRATEGIES)]
    return (
        f"Branch {branch_index + 1} of {n_branches}, running in slot {slot_id}.\n"
        f"Strategy: {strategy}.\n"
        "Explore an independent solution path. Make concrete assumptions when needed.\n"
        "For exact arithmetic, compute privately at least twice and audit subtraction and sums before output.\n"
        "Do not defer to other branches. Produce a complete candidate answer and do not trail off mid-sentence.\n\n"
        "Candidate answer:\n"
    )


def verifier_prompt(task: str, index: int, candidate: str, prefix_chars: int) -> str:
    prefix = candidate[:prefix_chars]
    return (
        "You are a strict verifier for one speculative branch.\n"
        "Check factual correctness, instruction following, completeness, and whether the early prefix already commits to a bad path.\n"
        "If the task contains arithmetic, recompute every numeric expression independently; do not trust the candidate's arithmetic.\n"
        "Return these fields in plain text: score: <0-100>, prefix_ok: yes|no, verdict: accept|partial|reject, notes: <short reason>.\n\n"
        "<task>\n"
        f"{task.strip()}\n"
        "</task>\n\n"
        f"<candidate_index>{index}</candidate_index>\n"
        "<candidate_prefix>\n"
        f"{prefix}\n"
        "</candidate_prefix>\n\n"
        "<candidate_full>\n"
        f"{candidate}\n"
        "</candidate_full>\n\n"
        "Verification:\n"
        "score: "
    )


def recombine_prompt(
    task: str,
    packets: list[dict[str, Any]],
    accept_score: int,
    candidate_max_chars: int,
    verifier_max_chars: int,
    baseline_answer: str | None = None,
) -> str:
    blocks = []
    for packet in packets:
        candidate = str(packet["candidate"])[:candidate_max_chars]
        verifier = str(packet.get("verifier_report") or "")[:verifier_max_chars]
        objective = packet.get("objective_validation") or {}
        objective_lines = ""
        if objective:
            objective_lines = (
                f"objective_score: {objective.get('score')}\n"
                f"objective_passed: {objective.get('passed')}\n"
                f"objective_failed_checks: {failed_checks(objective)}\n"
            )
        blocks.append(
            "-----\n"
            f"branch: {packet['index']}\n"
            f"slot: {packet['slot']}\n"
            f"score: {packet.get('score')}\n"
            f"prefix_ok: {packet.get('prefix_ok')}\n"
            f"{objective_lines}"
            "<candidate>\n"
            f"{candidate}\n"
            "</candidate>\n"
            "<verifier_report>\n"
            f"{verifier}\n"
            "</verifier_report>\n"
        )
    baseline_block = (
        "<single_pass_baseline>\n"
        f"{baseline_answer[:candidate_max_chars]}\n"
        "</single_pass_baseline>\n\n"
        if baseline_answer
        else ""
    )
    return (
        "You are recombining a speculative branch search into one final answer.\n"
        "Use the single-pass baseline as a floor if it is provided, and improve on it with stronger branch material.\n"
        f"Prefer material from branches with score >= {accept_score} and prefix_ok yes.\n"
        "If no branch meets that bar, repair the answer using the verifier notes instead of blindly accepting a branch.\n"
        "Return only the final answer for the user. Use plain text unless the user requested styling.\n"
        "Fit the answer in the token budget; do not trail off mid-sentence.\n\n"
        "<task>\n"
        f"{task.strip()}\n"
        "</task>\n\n"
        + baseline_block
        + "<branch_packets>\n"
        + "\n".join(blocks)
        + "\n</branch_packets>\n\n"
        "Final answer:\n"
    )


def score_prompt(task: str, answer_label: str, answer: str, answer_max_chars: int) -> str:
    return (
        "You are scoring one answer to one user task.\n"
        "Rubric: correctness 50 points, completeness 25, directness 15, clarity 10.\n"
        "Be strict: reserve 100 for unusually strong answers. A merely compliant but generic answer should score 85-90.\n"
        "Reward concrete thresholds, units, sample sizes, artifact names, and falsifiable pass/fail signals.\n"
        "Penalize metrics that are measurable only in name but lack a target, unit, or collection method.\n"
        "Penalize incomplete trailing text, unsupported claims, missing requested format, and irrelevant padding.\n"
        "Return plain text with exactly these fields: score: <0-100>, notes: <short reason>.\n\n"
        "<task>\n"
        f"{task.strip()}\n"
        "</task>\n\n"
        f"<answer_label>{answer_label}</answer_label>\n"
        "<answer>\n"
        f"{answer[:answer_max_chars]}\n"
        "</answer>\n\n"
        "Assessment:\n"
        "score: "
    )


def repair_prompt(
    task: str,
    baseline_answer: str,
    breakout_answer: str,
    baseline_report: str,
    breakout_report: str,
    packets: list[dict[str, Any]],
    candidate_max_chars: int,
    verifier_max_chars: int,
) -> str:
    branch_blocks = []
    for packet in packets:
        branch_blocks.append(
            "-----\n"
            f"branch: {packet['index']}\n"
            f"score: {packet.get('score')}\n"
            f"prefix_ok: {packet.get('prefix_ok')}\n"
            "<candidate>\n"
            f"{str(packet.get('candidate') or '')[:candidate_max_chars]}\n"
            "</candidate>\n"
            "<verifier_report>\n"
            f"{str(packet.get('verifier_report') or '')[:verifier_max_chars]}\n"
            "</verifier_report>\n"
        )
    return (
        "The breakout answer did not clearly beat the single-pass baseline.\n"
        "Write a repaired final answer that keeps the best correct material and fixes the scorer's complaints.\n"
        "Return only the final answer for the user. Use plain text unless the user requested styling.\n"
        "Fit the answer in the token budget; do not trail off mid-sentence.\n\n"
        "<task>\n"
        f"{task.strip()}\n"
        "</task>\n\n"
        "<baseline_answer>\n"
        f"{baseline_answer}\n"
        "</baseline_answer>\n"
        "<baseline_score_report>\n"
        f"{baseline_report}\n"
        "</baseline_score_report>\n\n"
        "<breakout_answer>\n"
        f"{breakout_answer}\n"
        "</breakout_answer>\n"
        "<breakout_score_report>\n"
        f"{breakout_report}\n"
        "</breakout_score_report>\n\n"
        "<branch_packets>\n"
        + "\n".join(branch_blocks)
        + "\n</branch_packets>\n\n"
        "Repaired final answer:\n"
    )


def failed_checks(validation: dict[str, Any]) -> list[str]:
    return [key for key, ok in validation.get("checks", {}).items() if not ok]


def objective_repair_feedback(case: dict[str, Any], current_answer: str, current_validation: dict[str, Any]) -> str:
    checks = current_validation.get("checks", {})
    if not isinstance(checks, dict):
        return ""

    lines: list[str] = []
    if case.get("validator") == "json_exact" and "expected" in case:
        try:
            actual = parse_json_output(current_answer)
        except Exception:  # noqa: BLE001
            return ""
        expected_paths = json_leaf_paths(case["expected"])
        actual_paths = json_leaf_paths(actual)
        failed_path_lines: list[str] = []
        for path in sorted(expected_paths):
            check_name = f"path_{check_key(path)}"
            if checks.get(check_name) is not False:
                continue
            if path in actual_paths:
                current_value = json.dumps(actual_paths[path], sort_keys=True, separators=(",", ":"))
                failed_path_lines.append(
                    f"- {path}: current value {current_value} failed validation; recompute it and do not repeat this value."
                )
            else:
                failed_path_lines.append(f"- {path}: missing from current answer; add the recomputed value.")
        if failed_path_lines:
            lines.append("Failed exact JSON fields; current values below are wrong, not expected values:")
            lines.extend(failed_path_lines)
        if checks.get("no_extra_paths") is False:
            extra_paths = sorted(set(actual_paths) - set(expected_paths))
            if extra_paths:
                lines.append(f"Remove unexpected JSON paths: {', '.join(extra_paths)}.")

    return "\n".join(lines)


def objective_repair_prompt(
    task: str,
    case: dict[str, Any],
    baseline_answer: str,
    current_answer: str,
    baseline_validation: dict[str, Any],
    current_validation: dict[str, Any],
    extra_context: str = "",
) -> str:
    hint = str(case.get("repair_hint") or "").strip()
    hint_block = f"Hint: {hint}\n" if hint else ""
    feedback = objective_repair_feedback(case, current_answer, current_validation)
    feedback_block = f"Repair feedback:\n{feedback}\n" if feedback else ""
    baseline_block = ""
    if baseline_answer.strip() and int(baseline_validation.get("score") or 0) > int(current_validation.get("score") or 0):
        baseline_block = f"Higher-scoring baseline: {baseline_answer}\n"
    extra_block = f"Additional audit context:\n{extra_context.strip()}\n" if extra_context.strip() else ""
    return (
        "Fix this answer for a deterministic benchmark.\n"
        "Return only the corrected final answer in the exact format requested. No markdown, notes, or thinking tags.\n\n"
        f"Benchmark: {case.get('id')} ({case.get('validator')})\n"
        f"Task: {task.strip()}\n"
        f"{baseline_block}"
        f"Current answer: {current_answer}\n"
        f"Failed checks: {', '.join(failed_checks(current_validation)) or 'none'}\n"
        f"Validation errors: {json.dumps(current_validation.get('errors', []), sort_keys=True)}\n"
        f"{feedback_block}"
        "Do not repeat values or structure associated with failed checks unless the task proves them correct.\n"
        f"{extra_block}"
        f"{hint_block}\n"
        "Corrected answer:\n"
    )


def parse_score(text: str) -> int | None:
    match = SCORE_RE.search(text) or FRACTION_RE.search(text) or LEADING_SCORE_RE.search(text)
    if not match:
        return None
    score = int(match.group(1))
    return max(0, min(100, score))


def parse_prefix_ok(text: str) -> bool | None:
    match = PREFIX_OK_RE.search(text)
    if not match:
        return None
    return match.group(1).lower() in {"yes", "true"}


def extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    starts = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0]
    if not starts:
        return stripped
    start = min(starts)
    opener = stripped[start]
    closer = "}" if opener == "{" else "]"
    end = stripped.rfind(closer)
    if end >= start:
        return stripped[start : end + 1]
    return stripped[start:]


def parse_json_output(text: str) -> Any:
    return json.loads(extract_json_text(text))


def parse_strict_json_output(text: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    return json.loads(
        text.strip(),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonstandard_constant,
    )


def score_from_checks(checks: dict[str, bool]) -> int:
    if not checks:
        return 0
    return round(100 * sum(1 for ok in checks.values() if ok) / len(checks))


def objective_result(score: int, checks: dict[str, bool], errors: list[str] | None = None) -> dict[str, Any]:
    errors = errors or []
    if checks and all(checks.values()) and not errors:
        score = 100
    score = max(0, min(100, int(score)))
    return {
        "score": score,
        "passed": score == 100 and bool(checks) and all(checks.values()) and not errors,
        "checks": checks,
        "errors": errors,
    }


def check_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return cleaned or "root"


def json_leaf_paths(value: Any, prefix: str = "root") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: {}}
        paths: dict[str, Any] = {}
        for key in sorted(value):
            paths.update(json_leaf_paths(value[key], f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        if not value:
            return {prefix: []}
        paths = {}
        for index, item in enumerate(value):
            paths.update(json_leaf_paths(item, f"{prefix}[{index}]"))
        return paths
    return {prefix: value}


def json_exact_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            json_exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            json_exact_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def validate_json_exact(case: dict[str, Any], text: str) -> dict[str, Any]:
    checks = {"has_expected": "expected" in case, "json_value": False, "type_match": False, "exact_match": False}
    if "expected" not in case:
        return objective_result(0, checks, ["case is missing expected"])
    expected = case["expected"]
    try:
        data = parse_strict_json_output(text)
    except Exception as exc:  # noqa: BLE001
        return objective_result(0, checks, [str(exc)])

    checks["json_value"] = True
    checks["type_match"] = type(data) is type(expected)  # noqa: E721
    checks["exact_match"] = json_exact_equal(data, expected)

    expected_paths = json_leaf_paths(expected)
    actual_paths = json_leaf_paths(data)
    checks["no_extra_paths"] = set(actual_paths) == set(expected_paths)
    for path, expected_value in expected_paths.items():
        checks[f"path_{check_key(path)}"] = (
            path in actual_paths and json_exact_equal(actual_paths[path], expected_value)
        )
    return objective_result(score_from_checks(checks), checks)


def validate_lines_exact(case: dict[str, Any], text: str) -> dict[str, Any]:
    expected = case.get("expected_lines")
    checks = {"has_expected_lines": isinstance(expected, list), "line_count": False}
    if not isinstance(expected, list):
        return objective_result(0, checks, ["case is missing expected_lines list"])
    expected_lines = [str(line).strip() for line in expected]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    checks["line_count"] = len(lines) == len(expected_lines)
    for index, expected_line in enumerate(expected_lines):
        actual = lines[index] if index < len(lines) else ""
        checks[f"line_{index + 1}"] = actual == expected_line
    return objective_result(score_from_checks(checks), checks)


def validate_arithmetic_json(text: str) -> dict[str, Any]:
    expected = {"alpha": 410, "beta": 70, "gamma": 987, "checksum": 1467}
    checks = {f"{key}_correct": False for key in expected}
    checks["json_object"] = False
    checks["exact_keys"] = False
    try:
        data = parse_json_output(text)
    except Exception as exc:  # noqa: BLE001
        return objective_result(0, checks, [str(exc)])
    checks["json_object"] = isinstance(data, dict)
    if not isinstance(data, dict):
        return objective_result(0, checks, ["expected JSON object"])
    checks["exact_keys"] = set(data) == set(expected)
    score = 20 if checks["json_object"] else 0
    score += 20 if checks["exact_keys"] else 0
    for key, value in expected.items():
        checks[f"{key}_correct"] = data.get(key) == value
        score += 15 if checks[f"{key}_correct"] else 0
    return objective_result(score, checks)


def validate_schema_fields(text: str) -> dict[str, Any]:
    expected_types = {
        "task_id": "string",
        "baseline_score": "number",
        "breakout_score": "number",
        "score_delta": "number",
        "selected_branch": "object",
        "artifact_path": "string",
    }
    checks = {
        "json_object": False,
        "type_object": False,
        "properties_object": False,
        "required_exact": False,
    }
    checks.update({f"has_{key}": False for key in expected_types})
    checks.update({f"type_{key}": False for key in expected_types})
    try:
        data = parse_json_output(text)
    except Exception as exc:  # noqa: BLE001
        return objective_result(0, checks, [str(exc)])
    checks["json_object"] = isinstance(data, dict)
    if not isinstance(data, dict):
        return objective_result(0, checks, ["expected JSON object"])
    checks["type_object"] = data.get("type") == "object"
    props = data.get("properties")
    checks["properties_object"] = isinstance(props, dict)
    required = data.get("required")
    checks["required_exact"] = isinstance(required, list) and set(required) == set(expected_types) and len(required) == len(expected_types)
    score = 10 if checks["json_object"] else 0
    score += 10 if checks["type_object"] else 0
    score += 10 if checks["properties_object"] else 0
    score += 20 if checks["required_exact"] else 0
    if isinstance(props, dict):
        for key, type_name in expected_types.items():
            checks[f"has_{key}"] = key in props
            prop = props.get(key)
            checks[f"type_{key}"] = isinstance(prop, dict) and prop.get("type") == type_name
            score += 4 if checks[f"has_{key}"] else 0
            score += 4 if checks[f"type_{key}"] else 0
    return objective_result(score, checks)


def validate_five_experiments(text: str) -> dict[str, Any]:
    required_terms = ["baseline_score", "breakout_score", "score_delta", "branch_verify_score", "final_score"]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    checks = {"exactly_five_lines": len(lines) == 5}
    score = 20 if checks["exactly_five_lines"] else max(0, 20 - abs(len(lines) - 5) * 5)
    for index in range(5):
        label = f"Experiment {index + 1}:"
        line = lines[index] if index < len(lines) else ""
        checks[f"line_{index + 1}_prefix"] = line.startswith(label)
        checks[f"line_{index + 1}_term"] = required_terms[index] in line
        score += 5 if checks[f"line_{index + 1}_prefix"] else 0
        score += 11 if checks[f"line_{index + 1}_term"] else 0
    return objective_result(score, checks)


def validate_csv_json_transform(text: str) -> dict[str, Any]:
    expected = [
        {"id": 1, "name": "Baseline", "score": 82, "delta_from_baseline": 0},
        {"id": 2, "name": "Breakout", "score": 95, "delta_from_baseline": 13},
        {"id": 3, "name": "Branch", "score": 88, "delta_from_baseline": 6},
    ]
    checks = {"json_array": False, "length_three": False, "ids_sorted": False}
    try:
        data = parse_json_output(text)
    except Exception as exc:  # noqa: BLE001
        return objective_result(0, checks, [str(exc)])
    checks["json_array"] = isinstance(data, list)
    if not isinstance(data, list):
        return objective_result(0, checks, ["expected JSON array"])
    checks["length_three"] = len(data) == 3
    ids = [row.get("id") for row in data if isinstance(row, dict)]
    checks["ids_sorted"] = ids == [1, 2, 3]
    score = 20 if checks["json_array"] else 0
    score += 10 if checks["length_three"] else 0
    score += 15 if checks["ids_sorted"] else 0
    expected_by_id = {row["id"]: row for row in expected}
    for row in data:
        if not isinstance(row, dict):
            continue
        expected_row = expected_by_id.get(row.get("id"))
        if expected_row is None:
            continue
        for key, value in expected_row.items():
            check_key = f"id_{expected_row['id']}_{key}"
            checks[check_key] = row.get(key) == value
            score += 5 if checks[check_key] else 0
    return objective_result(score, checks)


def validate_rubric_signals(text: str) -> dict[str, Any]:
    labels = ["Baseline", "Branch fanout", "Verifier quality", "Pass/fail gate"]
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]
    checks = {"exactly_four_bullets": len(bullet_lines) == 4}
    score = 20 if checks["exactly_four_bullets"] else max(0, 20 - abs(len(bullet_lines) - 4) * 5)
    for label in labels:
        matching = [line for line in bullet_lines if line.startswith(f"- {label}:")]
        checks[f"has_{label}"] = bool(matching)
        checks[f"digit_{label}"] = bool(matching and re.search(r"\d", matching[0]))
        score += 10 if checks[f"has_{label}"] else 0
        score += 10 if checks[f"digit_{label}"] else 0
    return objective_result(score, checks)


OBJECTIVE_VALIDATORS = {
    "arithmetic_json": validate_arithmetic_json,
    "schema_fields": validate_schema_fields,
    "five_experiments": validate_five_experiments,
    "csv_json_transform": validate_csv_json_transform,
    "rubric_signals": validate_rubric_signals,
}

OBJECTIVE_CASE_VALIDATORS = {
    "json_exact": validate_json_exact,
    "lines_exact": validate_lines_exact,
}

TERMINAL_OBJECTIVE_VALIDATORS = {"json_exact", "lines_exact"}
STRICT_JSON_VALIDATORS = {"json_exact"}


def validate_objective(case: dict[str, Any], text: str) -> dict[str, Any]:
    validator_name = str(case.get("validator") or "")
    validator = OBJECTIVE_VALIDATORS.get(validator_name)
    case_validator = OBJECTIVE_CASE_VALIDATORS.get(validator_name)
    if validator is None and case_validator is None:
        return objective_result(0, {"known_validator": False}, [f"unknown validator: {validator_name}"])
    result = case_validator(case, text) if case_validator is not None else validator(text)
    result["validator"] = validator_name
    return result


def is_terminal_objective_pass(
    case: dict[str, Any],
    text: str,
    validation: dict[str, Any] | None = None,
) -> bool:
    validation = validation or validate_objective(case, text)
    if not validation.get("passed") or int(validation.get("score") or 0) != 100:
        return False

    validator_name = str(case.get("validator") or "")
    if validator_name not in TERMINAL_OBJECTIVE_VALIDATORS:
        return False

    if validator_name in STRICT_JSON_VALIDATORS:
        try:
            parse_strict_json_output(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

    return True


def summarize_requests(results: list[RequestResult]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    pred_tps = [r.predicted_tps for r in ok if r.predicted_tps is not None]
    prompt_tps = [r.prompt_tps for r in ok if r.prompt_tps is not None]
    total_predicted = sum(r.predicted_tokens for r in ok)
    total_prompt = sum(r.prompt_tokens for r in ok)
    if ok:
        start = min(r.start for r in ok)
        end = max(r.end for r in ok)
        elapsed = end - start
    else:
        elapsed = 0.0
    return {
        "requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "total_prompt_tokens": total_prompt,
        "total_predicted_tokens": total_predicted,
        "parallel_wall_s": elapsed,
        "aggregate_predicted_tps_floor": total_predicted / elapsed if elapsed else None,
        "mean_slot_predicted_tps": statistics.mean(pred_tps) if pred_tps else None,
        "mean_prompt_tps": statistics.mean(prompt_tps) if prompt_tps else None,
        "errors": [r.error for r in results if r.error][:5],
    }


def summarize_suite(results: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [int(r["score_delta"]) for r in results if r.get("score_delta") is not None]
    baseline_scores = [
        int(r["final_scores"]["baseline"]["score"])
        for r in results
        if r.get("final_scores", {}).get("baseline", {}).get("score") is not None
    ]
    breakout_scores = [
        int(r["final_scores"]["breakout"]["score"])
        for r in results
        if r.get("final_scores", {}).get("breakout", {}).get("score") is not None
    ]
    wins = sum(1 for delta in deltas if delta > 0)
    ties = sum(1 for delta in deltas if delta == 0)
    losses = sum(1 for delta in deltas if delta < 0)
    return {
        "tasks": len(results),
        "scored_tasks": len(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_score_delta": statistics.mean(deltas) if deltas else None,
        "median_score_delta": statistics.median(deltas) if deltas else None,
        "min_score_delta": min(deltas) if deltas else None,
        "max_score_delta": max(deltas) if deltas else None,
        "mean_baseline_score": statistics.mean(baseline_scores) if baseline_scores else None,
        "mean_breakout_score": statistics.mean(breakout_scores) if breakout_scores else None,
        "all_positive": bool(deltas) and all(delta > 0 for delta in deltas),
        "non_negative": bool(deltas) and all(delta >= 0 for delta in deltas),
    }


def summarize_objective_benchmarks(results: list[dict[str, Any]]) -> dict[str, Any]:
    objective_rows = [r.get("objective_benchmark") or {} for r in results]
    escalated_rows = [
        row
        for result, row in zip(results, objective_rows)
        if result.get("decision", {}).get("short_circuit_phase") is None
    ]
    initial_rows = [row for row in escalated_rows if row.get("initial_breakout")]
    baseline_scores = [int(row["baseline"]["score"]) for row in objective_rows if row.get("baseline")]
    initial_breakout_scores = [
        int(row["initial_breakout"]["score"])
        for row in objective_rows
        if row.get("initial_breakout")
    ]
    breakout_scores = [int(row["breakout"]["score"]) for row in escalated_rows if row.get("breakout")]
    final_answer_scores = [int(row["breakout"]["score"]) for row in objective_rows if row.get("breakout")]
    initial_deltas = [
        int(row["initial_score_delta"])
        for row in objective_rows
        if row.get("initial_score_delta") is not None
    ]
    deltas = [int(row["score_delta"]) for row in objective_rows if row.get("score_delta") is not None]
    baseline_passes = sum(1 for row in objective_rows if row.get("baseline", {}).get("passed"))
    initial_baseline_passes = sum(1 for row in initial_rows if row.get("baseline", {}).get("passed"))
    initial_breakout_passes = sum(1 for row in initial_rows if row.get("initial_breakout", {}).get("passed"))
    breakout_passes = sum(1 for row in escalated_rows if row.get("breakout", {}).get("passed"))
    final_answer_passes = sum(1 for row in objective_rows if row.get("breakout", {}).get("passed"))
    baseline_short_circuits = sum(
        1 for result in results if result.get("decision", {}).get("short_circuit_phase") == "baseline"
    )
    wins = sum(1 for delta in deltas if delta > 0)
    ties = sum(1 for delta in deltas if delta == 0)
    losses = sum(1 for delta in deltas if delta < 0)
    repair_attempts = sum(int(row.get("objective_repairs_attempted") or 0) for row in objective_rows)
    repair_accepts = sum(int(row.get("objective_repairs_accepted") or 0) for row in objective_rows)
    fallback_attempts = sum(int(row.get("objective_fallback_recombine_attempted") or 0) for row in objective_rows)
    fallback_accepts = sum(int(row.get("objective_fallback_recombine_accepted") or 0) for row in objective_rows)
    fallback_repair_attempts = sum(int(row.get("objective_fallback_repair_attempted") or 0) for row in objective_rows)
    fallback_repair_accepts = sum(int(row.get("objective_fallback_repair_accepted") or 0) for row in objective_rows)
    return {
        "tasks": len(results),
        "validated_tasks": len(deltas),
        "baseline_passes": baseline_passes,
        "baseline_short_circuits": baseline_short_circuits,
        "escalated_tasks": len(escalated_rows),
        "initial_breakout_passes": initial_breakout_passes,
        "breakout_passes": breakout_passes,
        "final_answer_passes": final_answer_passes,
        "all_breakout_passed": (
            breakout_passes == len(escalated_rows) if escalated_rows else None
        ),
        "all_final_answers_passed": len(results) > 0 and final_answer_passes == len(results),
        "initial_pass_delta": initial_breakout_passes - initial_baseline_passes,
        "pass_delta": final_answer_passes - baseline_passes,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "no_objective_losses": losses == 0,
        "mean_baseline_score": statistics.mean(baseline_scores) if baseline_scores else None,
        "mean_initial_breakout_score": statistics.mean(initial_breakout_scores) if initial_breakout_scores else None,
        "mean_breakout_score": statistics.mean(breakout_scores) if breakout_scores else None,
        "mean_final_answer_score": statistics.mean(final_answer_scores) if final_answer_scores else None,
        "mean_initial_score_delta": statistics.mean(initial_deltas) if initial_deltas else None,
        "mean_score_delta": statistics.mean(deltas) if deltas else None,
        "median_score_delta": statistics.median(deltas) if deltas else None,
        "min_score_delta": min(deltas) if deltas else None,
        "max_score_delta": max(deltas) if deltas else None,
        "objective_repair_attempts": repair_attempts,
        "objective_repair_accepts": repair_accepts,
        "objective_fallback_recombine_attempts": fallback_attempts,
        "objective_fallback_recombine_accepts": fallback_accepts,
        "objective_fallback_repair_attempts": fallback_repair_attempts,
        "objective_fallback_repair_accepts": fallback_repair_accepts,
        "all_positive": bool(deltas) and all(delta > 0 for delta in deltas),
        "non_negative": bool(deltas) and all(delta >= 0 for delta in deltas),
    }


def phase_summaries(results: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        summary = result.get("summaries", {}).get(phase)
        if isinstance(summary, dict) and summary.get("ok_requests"):
            rows.append(summary)
    return rows


def summarize_phase(results: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    rows = phase_summaries(results, phase)
    total_predicted = sum(int(row.get("total_predicted_tokens") or 0) for row in rows)
    total_prompt = sum(int(row.get("total_prompt_tokens") or 0) for row in rows)
    total_wall = sum(float(row.get("parallel_wall_s") or 0.0) for row in rows)
    walls = [float(row.get("parallel_wall_s") or 0.0) for row in rows]
    requests = sum(int(row.get("requests") or 0) for row in rows)
    ok_requests = sum(int(row.get("ok_requests") or 0) for row in rows)
    return {
        "tasks_with_phase": len(rows),
        "requests": requests,
        "ok_requests": ok_requests,
        "total_prompt_tokens": total_prompt,
        "total_predicted_tokens": total_predicted,
        "total_wall_s": total_wall,
        "mean_task_wall_s": statistics.mean(walls) if walls else None,
        "aggregate_predicted_tps": total_predicted / total_wall if total_wall else None,
        "aggregate_prompt_tps": total_prompt / total_wall if total_wall else None,
    }


def summarize_phase_group(results: list[dict[str, Any]], phases: list[str]) -> dict[str, Any]:
    per_task_walls = []
    escalated_task_walls = []
    all_task_walls = []
    total_predicted = 0
    total_prompt = 0
    total_wall = 0.0
    requests = 0
    ok_requests = 0
    for result in results:
        task_wall = 0.0
        task_has_phase = False
        for phase in phases:
            summary = result.get("summaries", {}).get(phase)
            if not isinstance(summary, dict) or not summary.get("ok_requests"):
                continue
            task_has_phase = True
            wall = float(summary.get("parallel_wall_s") or 0.0)
            task_wall += wall
            total_wall += wall
            total_predicted += int(summary.get("total_predicted_tokens") or 0)
            total_prompt += int(summary.get("total_prompt_tokens") or 0)
            requests += int(summary.get("requests") or 0)
            ok_requests += int(summary.get("ok_requests") or 0)
        if task_has_phase:
            per_task_walls.append(task_wall)
        if result.get("decision", {}).get("short_circuit_phase") is None:
            escalated_task_walls.append(task_wall)
        all_task_walls.append(task_wall)
    return {
        "phases": phases,
        "tasks_with_any_phase": len(per_task_walls),
        "escalated_tasks": len(escalated_task_walls),
        "requests": requests,
        "ok_requests": ok_requests,
        "total_prompt_tokens": total_prompt,
        "total_predicted_tokens": total_predicted,
        "total_wall_s": total_wall,
        "mean_task_wall_s": statistics.mean(per_task_walls) if per_task_walls else None,
        "median_task_wall_s": statistics.median(per_task_walls) if per_task_walls else None,
        "mean_escalated_task_wall_s": statistics.mean(escalated_task_walls) if escalated_task_walls else None,
        "median_escalated_task_wall_s": statistics.median(escalated_task_walls) if escalated_task_walls else None,
        "mean_all_tasks_wall_s": statistics.mean(all_task_walls) if all_task_walls else None,
        "median_all_tasks_wall_s": statistics.median(all_task_walls) if all_task_walls else None,
        "aggregate_predicted_tps": total_predicted / total_wall if total_wall else None,
        "aggregate_prompt_tps": total_prompt / total_wall if total_wall else None,
    }


def summarize_accuracy_and_throughput(results: list[dict[str, Any]]) -> dict[str, Any]:
    objective_summary = summarize_objective_benchmarks(results)
    tasks = max(1, len(results))
    escalated_tasks = int(objective_summary["escalated_tasks"])
    multipass_phases = ["branches", "verifiers", "final", "objective_repairs"]
    multipass_with_baseline_phases = ["baseline", *multipass_phases]
    return {
        "accuracy": {
            "tasks": len(results),
            "baseline_pass_rate": objective_summary["baseline_passes"] / tasks,
            "initial_multipass_pass_rate": (
                objective_summary["initial_breakout_passes"] / escalated_tasks
                if escalated_tasks
                else None
            ),
            "final_multipass_pass_rate": (
                objective_summary["breakout_passes"] / escalated_tasks
                if escalated_tasks
                else None
            ),
            "final_system_pass_rate": objective_summary["final_answer_passes"] / tasks,
            "pass_rate_delta": objective_summary["pass_delta"] / tasks,
            "wins": objective_summary["wins"],
            "ties": objective_summary["ties"],
            "losses": objective_summary["losses"],
            "mean_score_delta": objective_summary["mean_score_delta"],
            "all_multipass_passed": objective_summary["all_breakout_passed"],
            "all_final_passed": objective_summary["all_final_answers_passed"],
            "no_objective_losses": objective_summary["no_objective_losses"],
            "baseline_short_circuits": objective_summary["baseline_short_circuits"],
            "escalated_tasks": escalated_tasks,
        },
        "throughput": {
            "baseline_single_pass": summarize_phase(results, "baseline"),
            "branch_fanout": summarize_phase(results, "branches"),
            "verifier_fanout": summarize_phase(results, "verifiers"),
            "recombine_final": summarize_phase(results, "final"),
            "objective_repair": summarize_phase(results, "objective_repairs"),
            "multipass_core": summarize_phase_group(results, multipass_phases),
            "multipass_with_baseline": summarize_phase_group(results, multipass_with_baseline_phases),
        },
    }


def summarize_decisions(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(results),
        "baseline_short_circuits": sum(
            1 for result in results if result.get("decision", {}).get("short_circuit_phase") == "baseline"
        ),
        "escalated_tasks": sum(
            1 for result in results if result.get("decision", {}).get("short_circuit_phase") is None
        ),
        "branches_launched": sum(
            int(result.get("decision", {}).get("branches_launched") or 0) for result in results
        ),
        "fanout_waves": sum(
            int(result.get("decision", {}).get("fanout_waves") or 0) for result in results
        ),
    }


def breakout_run_config(
    args: argparse.Namespace,
    slots: list[int],
    effective_verifier_mode: str,
) -> dict[str, Any]:
    return {
        "port": args.port,
        "attach": args.attach,
        "prefix_clone": args.prefix_clone,
        "prefix_clone_backend": args.prefix_clone_backend,
        "branch_slots": slots,
        "prefix_slot": args.prefix_slot,
        "final_slot": args.final_slot,
        "score_slot": args.score_slot,
        "branch_tokens": args.branch_tokens,
        "baseline_tokens": args.baseline_tokens,
        "verify_tokens": args.verify_tokens,
        "final_tokens": args.final_tokens,
        "score_tokens": args.score_tokens,
        "ignore_eos": args.ignore_eos,
        "stop_strings": args.stop_strings,
        "repair_rounds": args.repair_rounds,
        "objective_repair_rounds": args.objective_repair_rounds,
        "run_baseline": args.run_baseline,
        "score_final": args.score_final,
        "verifier_mode": args.verifier_mode,
        "effective_verifier_mode": effective_verifier_mode,
        "objective_fast_path": args.objective_fast_path,
        "objective_baseline_short_circuit": args.objective_baseline_short_circuit,
        "objective_fast_fallback_recombine": args.objective_fast_fallback_recombine,
        "objective_fast_fallback_model_verifier": args.objective_fast_fallback_model_verifier,
        "objective_fast_fallback_repair_rounds": args.objective_fast_fallback_repair_rounds,
        "adaptive_fanout": getattr(args, "adaptive_fanout", False),
        "fanout_stages": getattr(args, "fanout_stages", "1,2,4,8"),
        "preserve_prefix_root": getattr(args, "preserve_prefix_root", False),
        "accept_score": args.accept_score,
        "verify_prefix_chars": args.verify_prefix_chars,
    }


def baseline_terminal_result(
    args: argparse.Namespace,
    task: str,
    out_dir: Path,
    log_path: Path,
    props: dict[str, Any],
    slots: list[int],
    objective_case: dict[str, Any],
    baseline_result: RequestResult,
    baseline_validation: dict[str, Any],
) -> dict[str, Any]:
    empty_summary = summarize_requests([])
    objective_version = {
        "label": "baseline-terminal",
        "accepted": True,
        "validation": baseline_validation,
    }
    objective_benchmark = {
        "id": objective_case.get("id"),
        "validator": objective_case.get("validator"),
        "baseline": baseline_validation,
        "initial_breakout": None,
        "breakout": baseline_validation,
        "initial_score_delta": None,
        "score_delta": 0,
        "initial_pass_delta": None,
        "pass_delta": 0,
        "objective_repair_rounds": args.objective_repair_rounds,
        "objective_repairs_attempted": 0,
        "objective_repairs_accepted": 0,
        "objective_fallback_recombine_attempted": 0,
        "objective_fallback_recombine_accepted": 0,
        "objective_fallback_repair_attempted": 0,
        "objective_fallback_repair_accepted": 0,
        "versions": [objective_version],
    }
    request = asdict(baseline_result)

    return {
        "kind": "turbo-speculative-breakout",
        "schema_version": RESULT_SCHEMA_VERSION,
        "started": args.started,
        "task": task,
        "config": breakout_run_config(args, slots, "objective"),
        "server_props": props,
        "decision": {
            "short_circuit_phase": "baseline",
            "decision_reason": "baseline_terminal_objective_pass",
            "terminal_validation_surface": "cleaned_content",
            "branches_launched": 0,
            "fanout_waves": 0,
            "fanout_stages": [],
        },
        "prefix": {
            "backend": None,
            "result": None,
            "clone_wall_s": None,
            "fork": None,
            "save": None,
            "restore": [],
            "clone_filename": None,
            "cleanup_errors": [],
        },
        "summaries": {
            "baseline": summarize_requests([baseline_result]),
            "branches": empty_summary,
            "verifiers": empty_summary,
            "final": None,
            "scores": None,
            "objective_repairs": None,
        },
        "baseline": request,
        "final_scores": {},
        "final_versions": [
            {
                "label": "baseline-terminal",
                "score": None,
                "score_report": "",
                "request": request,
                "objective_validation": baseline_validation,
                "accepted": True,
            }
        ],
        "score_delta": None,
        "objective_benchmark": objective_benchmark,
        "selected": {
            "source": "baseline",
            "index": None,
            "slot": args.final_slot,
            "score": None,
            "prefix_ok": None,
            "objective_score": int(baseline_validation["score"]),
            "objective_passed": True,
            "objective_failed_checks": [],
            "model_accepted": False,
            "accepted": True,
        },
        "packets": [],
        "requests": {
            "baseline": request,
            "branches": [],
            "verifiers": [],
            "final": None,
            "scores": {},
            "objective_repairs": [],
        },
        "final_answer": baseline_result.content,
        "log_path": str(log_path),
        "out_dir": str(out_dir),
    }


def fanout(
    args: argparse.Namespace,
    prompts: list[tuple[str, str, int, bool, float, int, float, int, int]],
) -> list[RequestResult]:
    barrier = threading.Barrier(len(prompts))

    def one(item: tuple[str, str, int, bool, float, int, float, int, int]) -> RequestResult:
        label, prompt, slot_id, cache_prompt, temperature, top_k, top_p, seed, n_predict = item
        barrier.wait()
        return completion(
            args.port,
            label,
            prompt,
            n_predict,
            slot_id,
            cache_prompt,
            temperature,
            top_k,
            top_p,
            seed,
            args.request_timeout,
            ignore_eos=args.ignore_eos,
            use_stop_strings=args.stop_strings,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return list(ex.map(one, prompts))


def run_breakout(
    args: argparse.Namespace,
    task: str,
    out_dir: Path,
    log_path: Path,
    objective_case: dict[str, Any] | None = None,
    suite_index: int = 0,
) -> dict[str, Any]:
    slots = parse_slot_list(args.branch_slots)
    preserve_prefix_root = getattr(args, "preserve_prefix_root", False)
    if preserve_prefix_root:
        pcbt_mode = getattr(args, "pcbt", False)
        if (not getattr(args, "adaptive_fanout", False) and not pcbt_mode) or objective_case is None:
            raise SystemExit("--preserve-prefix-root requires adaptive objective fanout or --pcbt with an objective suite")
        if not pcbt_mode and (not args.prefix_clone or args.prefix_clone_backend != "fork"):
            raise SystemExit("--preserve-prefix-root requires --prefix-clone and --prefix-clone-backend fork")
        if args.prefix_slot in slots:
            raise SystemExit("--prefix-slot must be excluded from --branch-slots with --preserve-prefix-root")
    elif args.prefix_slot not in slots:
        raise SystemExit("--prefix-slot must be one of --branch-slots")

    props = {}
    with contextlib.suppress(Exception):
        props = http_json("GET", f"http://127.0.0.1:{args.port}/props", timeout=10.0)
    total_slots = int(props.get("total_slots") or 0) if isinstance(props, dict) else 0
    max_requested_slot = max(slots + [args.prefix_slot, args.final_slot, args.score_slot])
    if total_slots and max_requested_slot >= total_slots:
        raise RuntimeError(f"requested slot {max_requested_slot}, but server reports {total_slots} slots")

    prefix = base_prefix(task)
    baseline_result: RequestResult | None = None
    if args.run_baseline:
        baseline_result = completion(
            args.port,
            "baseline-single",
            baseline_prompt(task),
            args.baseline_tokens,
            args.final_slot,
            False,
            0.0,
            1,
            1.0,
            args.seed + 500,
            args.request_timeout,
            ignore_eos=args.ignore_eos,
            use_stop_strings=args.stop_strings,
        )
        if not baseline_result.ok:
            raise RuntimeError(f"baseline generation failed: {baseline_result.error}")

    baseline_gate_reason = "baseline_short_circuit_not_applicable"
    if args.objective_fast_path and objective_case is not None and baseline_result is not None:
        if not args.objective_baseline_short_circuit:
            baseline_gate_reason = "baseline_short_circuit_disabled"
        elif args.score_final:
            baseline_gate_reason = "model_scoring_requested"
        else:
            baseline_validation = validate_objective(objective_case, baseline_result.content)
            if is_terminal_objective_pass(objective_case, baseline_result.content, baseline_validation):
                return baseline_terminal_result(
                    args,
                    task,
                    out_dir,
                    log_path,
                    props,
                    slots,
                    objective_case,
                    baseline_result,
                    baseline_validation,
                )
            baseline_gate_reason = (
                "baseline_nonterminal_output"
                if baseline_validation.get("passed")
                else "baseline_objective_not_passed"
            )

    prefix_result: RequestResult | None = None
    save_result: dict[str, Any] | None = None
    restore_results: list[dict[str, Any]] = []
    fork_result: dict[str, Any] | None = None
    fork_waves: list[dict[str, Any]] = []
    clone_wall_s: float | None = None
    clone_filename = f"breakout-prefix-{utc_stamp()}-{os.getpid()}.bin"

    if args.prefix_clone:
        prefix_result = completion(
            args.port,
            "prefix-eval",
            prefix,
            0,
            args.prefix_slot,
            True,
            0.0,
            1,
            1.0,
            args.seed,
            args.request_timeout,
            ignore_eos=args.ignore_eos,
            use_stop_strings=args.stop_strings,
        )
        if not prefix_result.ok:
            raise RuntimeError(f"prefix evaluation failed: {prefix_result.error}")
        destinations = [slot_id for slot_id in slots if slot_id != args.prefix_slot]
        clone_start = time.perf_counter()
        if args.prefix_clone_backend == "fork":
            if destinations and not preserve_prefix_root:
                try:
                    fork_result = slot_fork(
                        args.port,
                        args.prefix_slot,
                        destinations,
                        timeout=args.request_timeout,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "prefix fork failed; use a unified-KV server with slot-fork support, "
                        "select --prefix-clone-backend file, or rerun with --no-prefix-clone"
                    ) from exc
        else:
            try:
                save_result = slot_action(args.port, args.prefix_slot, "save", clone_filename, timeout=args.request_timeout)
                for slot_id in destinations:
                    restore_results.append(slot_action(args.port, slot_id, "restore", clone_filename, timeout=args.request_timeout))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "prefix clone failed; start the server with --slot-save-path or rerun with --no-prefix-clone"
                ) from exc
        clone_wall_s = time.perf_counter() - clone_start

    n_branches = len(slots)
    branch_prompts: list[tuple[str, str, int, bool, float, int, float, int, int]] = []
    for branch_index, slot_id in enumerate(slots):
        prompt = prefix + branch_suffix(slot_id, branch_index, n_branches)
        branch_prompts.append(
            (
                f"branch-{branch_index:02d}-slot-{slot_id}",
                prompt,
                slot_id,
                bool(args.prefix_clone),
                args.temperature,
                args.top_k,
                args.top_p,
                args.seed + 1000 + branch_index,
                args.branch_tokens,
            )
        )

    adaptive_fanout = getattr(args, "adaptive_fanout", False)
    if adaptive_fanout and objective_case is None:
        raise SystemExit("--adaptive-fanout requires --benchmark-suite or --benchmark-suite-file")
    if getattr(args, "pcbt", False) and not preserve_prefix_root:
        raise SystemExit("--pcbt requires --preserve-prefix-root")

    stage_ends = (
        parse_fanout_stages(getattr(args, "fanout_stages", "1,2,4,8"), len(branch_prompts))
        if adaptive_fanout
        else [len(branch_prompts)]
    )
    fanout_waves = 0
    fanout_stage_results: list[dict[str, Any]] = []
    branch_results: list[RequestResult] = []
    current_root_fork_id: int | None = None
    current_source_node: int = -1
    for stage_end in stage_ends:
        stage_start = len(branch_results)
        stage_fork: dict[str, Any] | None = None
        stage_commit: dict[str, Any] | None = None
        stage_tx: dict[str, Any] | None = None
        stage_prompts = branch_prompts[stage_start:stage_end]
        if preserve_prefix_root and getattr(args, "pcbt", False):
            source_node = current_source_node
            if source_node < 0:
                source_node = slot_node_id(args.port, args.prefix_slot, timeout=args.request_timeout)
            if source_node < 0:
                # Mint node identity once: fork + commit the root in place.
                mint_dest = [item[2] for item in stage_prompts][:1]
                minted = slot_fork(args.port, args.prefix_slot, mint_dest, timeout=args.request_timeout)
                slot_commit(args.port, args.prefix_slot, int(minted["fork_id"]), timeout=args.request_timeout)
                source_node = slot_node_id(args.port, args.prefix_slot, timeout=args.request_timeout)
            tx_branches = []
            for item in stage_prompts:
                tx_branches.append({
                    "key": item[0],
                    "request": {"prompt": item[1], "max_tokens": int(item[8]), "seed": int(item[7])},
                })
            budget = {
                "max_slots": max(2, len(tx_branches)),
                "max_predicted_tokens": max(1, sum(int(item[8]) for item in stage_prompts) * 2),
                "deadline_ms": min(600000, max(1000, int(args.request_timeout * 1000))),
                "max_candidate_bytes": 1048576,
            }
            stage_tx = pcbt_create(
                args.port,
                f"breakout-{os.getpid()}-{suite_index}-{fanout_waves}",
                source_node,
                tx_branches,
                budget,
                timeout=args.request_timeout,
            )
            fork_waves.append({"pcbt": True, "transaction_id": stage_tx["transaction_id"],
                               "generation": stage_tx["generation"]})
            # Remap each branch decode onto the transaction's slot assignment.
            key_to_slot = {b["key"]: b["slot_id"] for b in stage_tx["branches"]}
            stage_prompts = [
                (item[0], item[1], key_to_slot[item[0]], item[3], item[4], item[5], item[6], item[7], item[8])
                for item in stage_prompts
            ]
        elif preserve_prefix_root:
            stage_slots = [item[2] for item in stage_prompts]
            stage_fork = slot_fork(
                args.port,
                args.prefix_slot,
                stage_slots,
                timeout=args.request_timeout,
                fork_id=current_root_fork_id,
            )
            fork_waves.append(stage_fork)

        stage_results = fanout(args, stage_prompts)
        branch_results.extend(stage_results)
        fanout_waves += 1

        passing_indices: list[int] = []
        if objective_case is not None:
            for index, branch in enumerate(branch_results):
                if branch.ok and validate_objective(objective_case, branch.content).get("passed"):
                    passing_indices.append(index)

        if preserve_prefix_root and stage_tx is not None:
            view = pcbt_observe(args.port, stage_tx["transaction_id"], timeout=args.request_timeout)
            completed = {b["key"]: b for b in view["branches"] if b["phase"] == "completed"}
            winner_key = None
            if passing_indices:
                idx0 = passing_indices[0]
                if idx0 >= stage_start:
                    winner_key = stage_prompts[idx0 - stage_start][0]
            if winner_key is None or winner_key not in completed:
                root_key = stage_prompts[0][0]
                winner_key = root_key if root_key in completed else (next(iter(completed)) if completed else None)
            if winner_key is None:
                raise RuntimeError("pcbt stage produced no completed candidate to commit")
            wb = completed[winner_key]
            report = {
                "validator": "objective",
                "passing": [stage_prompts[i - stage_start][0] for i in passing_indices if i >= stage_start],
                "winner": winner_key,
            }
            stage_commit = pcbt_commit_tx(
                args.port,
                stage_tx["transaction_id"],
                int(wb["node_id"]),
                int(view["generation"]),
                wb["candidate_digest"],
                pcbt_evidence_digest(report),
                report,
                timeout=args.request_timeout,
            )
            current_root_fork_id = None
            current_source_node = int(stage_commit["winner_node_id"])
        elif preserve_prefix_root:
            winner_slot = (
                branch_results[passing_indices[0]].id_slot
                if passing_indices
                else args.prefix_slot
            )
            stage_commit = slot_commit(
                args.port,
                int(winner_slot),
                int(stage_fork["fork_id"]),
                timeout=args.request_timeout,
            )
            current_root_fork_id = int(stage_commit["fork_id"])

        fanout_stage_results.append({
            "target_width": stage_end,
            "launched": len(stage_results),
            "cumulative_launched": len(branch_results),
            "passing_indices": passing_indices,
            "stopped": bool(adaptive_fanout and passing_indices),
            "fork": stage_fork,
            "commit": stage_commit,
        })
        if adaptive_fanout and passing_indices:
            break

    if not any(r.ok for r in branch_results):
        raise RuntimeError(f"all branch requests failed: {summarize_requests(branch_results)['errors']}")

    effective_verifier_mode = args.verifier_mode
    if args.objective_fast_path and objective_case is not None:
        effective_verifier_mode = "objective"
    if effective_verifier_mode == "objective" and objective_case is None:
        raise SystemExit("--verifier-mode objective requires --benchmark-suite or --benchmark-suite-file")

    verifier_results: list[RequestResult] = []
    if effective_verifier_mode == "model":
        verifier_prompts: list[tuple[str, str, int, bool, float, int, float, int, int]] = []
        for index, (slot_id, branch) in enumerate(zip(slots, branch_results)):
            candidate = branch.content if branch.ok else f"BRANCH FAILED: {branch.error}"
            verifier_prompts.append(
                (
                    f"verify-{index:02d}-slot-{slot_id}",
                    verifier_prompt(task, index, candidate, args.verify_prefix_chars),
                    slot_id,
                    False,
                    0.0,
                    1,
                    1.0,
                    args.seed + 2000 + index,
                    args.verify_tokens,
                )
            )
        verifier_results = fanout(args, verifier_prompts)
        fanout_waves += 1

    packets: list[dict[str, Any]] = []
    for index, (slot_id, branch) in enumerate(zip(slots, branch_results)):
        if effective_verifier_mode == "model":
            verifier = verifier_results[index]
            report = verifier.content if verifier.ok else f"VERIFIER FAILED: {verifier.error}"
            verifier_ok = verifier.ok
            verifier_error = verifier.error
            parsed_score = parse_score(report)
            prefix_ok = parse_prefix_ok(report)
        elif effective_verifier_mode == "objective":
            report = "OBJECTIVE VERIFIER: deterministic validator result is recorded in objective_validation."
            verifier_ok = True
            verifier_error = None
            parsed_score = None
            prefix_ok = None
        else:
            report = "VERIFIER SKIPPED"
            verifier_ok = True
            verifier_error = None
            parsed_score = None
            prefix_ok = None
        packets.append(
            {
                "index": index,
                "slot": slot_id,
                "candidate": branch.content,
                "branch_ok": branch.ok,
                "branch_error": branch.error,
                "verifier_report": report,
                "verifier_ok": verifier_ok,
                "verifier_error": verifier_error,
                "score": parsed_score,
                "prefix_ok": prefix_ok,
            }
        )

    if objective_case is not None:
        for packet in packets:
            objective_validation = validate_objective(objective_case, str(packet.get("candidate") or ""))
            packet["objective_validation"] = objective_validation
            packet["objective_score"] = int(objective_validation["score"])
            packet["objective_passed"] = bool(objective_validation["passed"])
            if effective_verifier_mode == "objective":
                packet["score"] = int(objective_validation["score"])
                packet["prefix_ok"] = bool(objective_validation["passed"])
                packet["verifier_report"] = (
                    f"objective_score: {objective_validation['score']}, "
                    f"objective_passed: {objective_validation['passed']}, "
                    f"failed_checks: {failed_checks(objective_validation)}"
                )

    def packet_key(packet: dict[str, Any]) -> tuple[int, int, int, int, int]:
        if objective_case is not None:
            objective_score = int(packet.get("objective_score") or 0)
            objective_pass_bonus = 1 if packet.get("objective_passed") else 0
        else:
            objective_score = 0
            objective_pass_bonus = 0
        score = packet["score"] if packet["score"] is not None else -1
        prefix_bonus = 1 if packet["prefix_ok"] is True else 0
        branch_bonus = 1 if packet["branch_ok"] else 0
        return (objective_pass_bonus, objective_score, score, prefix_bonus, branch_bonus)

    selected = max(packets, key=packet_key)
    selected_branch_result = branch_results[int(selected["index"])]
    final_request_result: RequestResult | None = None
    final_version_label = "initial"
    if args.objective_fast_path and objective_case is not None and selected_branch_result.ok:
        final_result = selected_branch_result
        final_version_label = "objective-fast-selected" if selected.get("objective_passed") else "objective-fast-repair-seed"
    else:
        final_prompt = recombine_prompt(
            task,
            packets,
            args.accept_score,
            args.candidate_max_chars,
            args.verifier_max_chars,
            baseline_result.content if baseline_result else None,
        )
        final_result = completion(
            args.port,
            "recombine-final",
            final_prompt,
            args.final_tokens,
            args.final_slot,
            False,
            0.0,
            1,
            1.0,
            args.seed + 3000,
            args.request_timeout,
            ignore_eos=args.ignore_eos,
            use_stop_strings=args.stop_strings,
        )
        final_request_result = final_result
        if not final_result.ok:
            raise RuntimeError(f"final recombine failed: {final_result.error}")

    score_results: dict[str, RequestResult] = {}
    parsed_scores: dict[str, int | None] = {}
    score_delta: int | None = None
    final_versions: list[dict[str, Any]] = [
        {
            "label": final_version_label,
            "result": final_result,
            "score": None,
            "score_report": "",
        }
    ]
    objective_benchmark: dict[str, Any] | None = None
    objective_repair_results: list[RequestResult] = []

    def run_score(label: str, answer: str, seed_offset: int) -> tuple[RequestResult, int | None]:
        score_result = completion(
            args.port,
            f"score-{label}",
            score_prompt(task, label, answer, args.score_answer_max_chars),
            args.score_tokens,
            args.score_slot,
            False,
            0.0,
            1,
            1.0,
            args.seed + seed_offset,
            args.request_timeout,
            ignore_eos=args.ignore_eos,
            use_stop_strings=args.stop_strings,
        )
        return score_result, parse_score(score_result.content) if score_result.ok else None

    if args.score_final:
        if baseline_result is not None:
            score_results["baseline"], parsed_scores["baseline"] = run_score("baseline", baseline_result.content, 4000)
        score_results["breakout"], parsed_scores["breakout"] = run_score("breakout", final_result.content, 4001)
        final_versions[0]["score"] = parsed_scores.get("breakout")
        final_versions[0]["score_report"] = score_results["breakout"].content if score_results.get("breakout") else ""

        selected_branch_result = branch_results[int(selected["index"])]
        if selected_branch_result.ok:
            selected_label = "selected-branch"
            score_results[selected_label], parsed_scores[selected_label] = run_score(
                selected_label,
                selected_branch_result.content,
                4050,
            )
            selected_score = parsed_scores.get(selected_label)
            final_versions.append(
                {
                    "label": selected_label,
                    "result": selected_branch_result,
                    "score": selected_score,
                    "score_report": score_results[selected_label].content,
                }
            )
            breakout_score = parsed_scores.get("breakout")
            if selected_score is not None and (breakout_score is None or selected_score > breakout_score):
                final_result = selected_branch_result
                parsed_scores["breakout"] = selected_score
                score_results["breakout"] = score_results[selected_label]

        for repair_index in range(args.repair_rounds):
            baseline_score = parsed_scores.get("baseline")
            breakout_score = parsed_scores.get("breakout")
            if baseline_result is None or baseline_score is None or breakout_score is None or breakout_score > baseline_score:
                break
            repair_result = completion(
                args.port,
                f"repair-{repair_index}",
                repair_prompt(
                    task,
                    baseline_result.content,
                    final_result.content,
                    score_results["baseline"].content,
                    score_results["breakout"].content,
                    packets,
                    args.candidate_max_chars,
                    args.verifier_max_chars,
                ),
                args.final_tokens,
                args.final_slot,
                False,
                0.0,
                1,
                1.0,
                args.seed + 5000 + repair_index,
                args.request_timeout,
                ignore_eos=args.ignore_eos,
                use_stop_strings=args.stop_strings,
            )
            if not repair_result.ok:
                final_versions.append(
                    {
                        "label": f"repair-{repair_index}",
                        "result": repair_result,
                        "score": None,
                        "score_report": "",
                    }
                )
                break
            repair_label = f"breakout-repair-{repair_index}"
            score_results[repair_label], parsed_scores[repair_label] = run_score(
                repair_label,
                repair_result.content,
                4100 + repair_index,
            )
            repair_score = parsed_scores.get(repair_label)
            final_versions.append(
                {
                    "label": f"repair-{repair_index}",
                    "result": repair_result,
                    "score": repair_score,
                    "score_report": score_results[repair_label].content,
                }
            )
            if repair_score is not None and repair_score > breakout_score:
                final_result = repair_result
                parsed_scores["breakout"] = repair_score
                score_results["breakout"] = score_results[repair_label]
            else:
                break

        if "baseline" in parsed_scores and parsed_scores.get("baseline") is not None and parsed_scores.get("breakout") is not None:
            score_delta = int(parsed_scores["breakout"]) - int(parsed_scores["baseline"])

    if objective_case is not None:
        baseline_text = baseline_result.content if baseline_result else ""
        baseline_validation = validate_objective(objective_case, baseline_text)
        initial_breakout_validation = validate_objective(objective_case, final_result.content)
        current_validation = initial_breakout_validation
        objective_versions: list[dict[str, Any]] = [
            {
                "label": "pre-objective-repair",
                "accepted": True,
                "validation": initial_breakout_validation,
            }
        ]
        selected_branch_result = branch_results[int(selected["index"])]
        selected_branch_validation = selected.get("objective_validation")
        if (
            selected_branch_result.ok
            and isinstance(selected_branch_validation, dict)
            and int(selected_branch_validation.get("score") or 0) > int(current_validation["score"])
        ):
            final_result = selected_branch_result
            current_validation = selected_branch_validation
            objective_versions.append(
                {
                    "label": "objective-selected-branch",
                    "accepted": True,
                    "validation": selected_branch_validation,
                }
            )
            if args.score_final and "selected-branch" in parsed_scores:
                selected_score = parsed_scores.get("selected-branch")
                if selected_score is not None:
                    parsed_scores["breakout"] = selected_score
                    score_results["breakout"] = score_results["selected-branch"]
                    baseline_score = parsed_scores.get("baseline")
                    if baseline_score is not None:
                        score_delta = int(selected_score) - int(baseline_score)

        for objective_repair_index in range(args.objective_repair_rounds):
            if current_validation.get("passed"):
                break

            repair_result = completion(
                args.port,
                f"objective-repair-{objective_repair_index}",
                objective_repair_prompt(
                    task,
                    objective_case,
                    baseline_text,
                    final_result.content,
                    baseline_validation,
                    current_validation,
                ),
                args.final_tokens,
                args.final_slot,
                False,
                0.0,
                1,
                1.0,
                args.seed + 6000 + suite_index * 100 + objective_repair_index,
                args.request_timeout,
                OBJECTIVE_REPAIR_STOPS,
                ignore_eos=args.ignore_eos,
                use_stop_strings=args.stop_strings,
            )
            objective_repair_results.append(repair_result)
            repair_label = f"objective-repair-{objective_repair_index}"
            if not repair_result.ok:
                objective_versions.append(
                    {
                        "label": repair_label,
                        "accepted": False,
                        "validation": None,
                        "error": repair_result.error,
                    }
                )
                final_versions.append(
                    {
                        "label": repair_label,
                        "result": repair_result,
                        "score": None,
                        "score_report": "",
                        "objective_validation": None,
                        "accepted": False,
                    }
                )
                break

            repair_validation = validate_objective(objective_case, repair_result.content)
            accepted = int(repair_validation["score"]) > int(current_validation["score"])
            repair_model_score: int | None = None
            repair_score_report = ""

            if accepted:
                final_result = repair_result
                current_validation = repair_validation
                if args.score_final:
                    score_results[repair_label], parsed_scores[repair_label] = run_score(
                        repair_label,
                        final_result.content,
                        4200 + suite_index * 100 + objective_repair_index,
                    )
                    repair_model_score = parsed_scores.get(repair_label)
                    repair_score_report = score_results[repair_label].content
                    if repair_model_score is not None:
                        parsed_scores["breakout"] = repair_model_score
                        score_results["breakout"] = score_results[repair_label]
                        baseline_score = parsed_scores.get("baseline")
                        if baseline_score is not None:
                            score_delta = int(repair_model_score) - int(baseline_score)

            objective_versions.append(
                {
                    "label": repair_label,
                    "accepted": accepted,
                    "validation": repair_validation,
                }
            )
            final_versions.append(
                {
                    "label": repair_label,
                    "result": repair_result,
                    "score": repair_model_score,
                    "score_report": repair_score_report,
                    "objective_validation": repair_validation,
                    "accepted": accepted,
                }
            )
            if not accepted:
                break

        if (
            args.objective_fast_path
            and args.objective_fast_fallback_recombine
            and final_request_result is None
            and not current_validation.get("passed")
        ):
            if args.objective_fast_fallback_model_verifier and not verifier_results:
                fallback_verifier_prompts: list[tuple[str, str, int, bool, float, int, float, int, int]] = []
                for index, (slot_id, branch) in enumerate(zip(slots, branch_results)):
                    candidate = branch.content if branch.ok else f"BRANCH FAILED: {branch.error}"
                    fallback_verifier_prompts.append(
                        (
                            f"fallback-verify-{index:02d}-slot-{slot_id}",
                            verifier_prompt(task, index, candidate, args.verify_prefix_chars),
                            slot_id,
                            False,
                            0.0,
                            1,
                            1.0,
                            args.seed + 8000 + suite_index * 100 + index,
                            args.verify_tokens,
                        )
                    )
                verifier_results = fanout(args, fallback_verifier_prompts)
                fanout_waves += 1
                for packet, verifier in zip(packets, verifier_results):
                    report = verifier.content if verifier.ok else f"VERIFIER FAILED: {verifier.error}"
                    packet["verifier_report"] = report
                    packet["verifier_ok"] = verifier.ok
                    packet["verifier_error"] = verifier.error
                    packet["score"] = parse_score(report)
                    packet["prefix_ok"] = parse_prefix_ok(report)

            fallback_result = completion(
                args.port,
                "objective-fallback-recombine",
                recombine_prompt(
                    task,
                    packets,
                    args.accept_score,
                    args.candidate_max_chars,
                    args.verifier_max_chars,
                    baseline_result.content if baseline_result else None,
                ),
                args.final_tokens,
                args.final_slot,
                False,
                0.0,
                1,
                1.0,
                args.seed + 7000 + suite_index,
                args.request_timeout,
                ignore_eos=args.ignore_eos,
                use_stop_strings=args.stop_strings,
            )
            final_request_result = fallback_result
            fallback_validation = validate_objective(objective_case, fallback_result.content) if fallback_result.ok else None
            fallback_accepted = (
                fallback_validation is not None
                and int(fallback_validation["score"]) > int(current_validation["score"])
            )
            objective_versions.append(
                {
                    "label": "objective-fallback-recombine",
                    "accepted": fallback_accepted,
                    "validation": fallback_validation,
                    "error": fallback_result.error,
                }
            )
            final_versions.append(
                {
                    "label": "objective-fallback-recombine",
                    "result": fallback_result,
                    "score": None,
                    "score_report": "",
                    "objective_validation": fallback_validation,
                    "accepted": fallback_accepted,
                }
            )
            if fallback_accepted and fallback_validation is not None:
                final_result = fallback_result
                current_validation = fallback_validation
                if args.score_final:
                    score_results["objective-fallback-recombine"], parsed_scores["objective-fallback-recombine"] = run_score(
                        "objective-fallback-recombine",
                        final_result.content,
                        4300 + suite_index,
                    )
                    fallback_model_score = parsed_scores.get("objective-fallback-recombine")
                    if fallback_model_score is not None:
                        parsed_scores["breakout"] = fallback_model_score
                        score_results["breakout"] = score_results["objective-fallback-recombine"]
                        baseline_score = parsed_scores.get("baseline")
                        if baseline_score is not None:
                            score_delta = int(fallback_model_score) - int(baseline_score)

            if (
                fallback_validation is not None
                and not current_validation.get("passed")
                and args.objective_fast_fallback_repair_rounds > 0
                and verifier_results
            ):
                fallback_repair_answer = final_result.content if fallback_accepted else fallback_result.content
                fallback_repair_validation_seed = current_validation if fallback_accepted else fallback_validation
                verifier_context = "\n".join(
                    (
                        f"branch {packet['index']} verifier: "
                        f"{str(packet.get('verifier_report') or '')[:args.verifier_max_chars]}"
                    )
                    for packet in packets
                    if packet.get("verifier_report")
                )
                for fallback_repair_index in range(args.objective_fast_fallback_repair_rounds):
                    fallback_repair_result = completion(
                        args.port,
                        f"objective-fallback-repair-{fallback_repair_index}",
                        objective_repair_prompt(
                            task,
                            objective_case,
                            baseline_text,
                            fallback_repair_answer,
                            baseline_validation,
                            fallback_repair_validation_seed,
                            verifier_context,
                        ),
                        args.final_tokens,
                        args.final_slot,
                        False,
                        0.0,
                        1,
                        1.0,
                        args.seed + 9000 + suite_index * 100 + fallback_repair_index,
                        args.request_timeout,
                        OBJECTIVE_REPAIR_STOPS,
                        ignore_eos=args.ignore_eos,
                        use_stop_strings=args.stop_strings,
                    )
                    objective_repair_results.append(fallback_repair_result)
                    fallback_repair_label = f"objective-fallback-repair-{fallback_repair_index}"
                    if not fallback_repair_result.ok:
                        objective_versions.append(
                            {
                                "label": fallback_repair_label,
                                "accepted": False,
                                "validation": None,
                                "error": fallback_repair_result.error,
                            }
                        )
                        final_versions.append(
                            {
                                "label": fallback_repair_label,
                                "result": fallback_repair_result,
                                "score": None,
                                "score_report": "",
                                "objective_validation": None,
                                "accepted": False,
                            }
                        )
                        break
                    fallback_repair_validation = validate_objective(objective_case, fallback_repair_result.content)
                    fallback_repair_accepted = int(fallback_repair_validation["score"]) > int(current_validation["score"])
                    objective_versions.append(
                        {
                            "label": fallback_repair_label,
                            "accepted": fallback_repair_accepted,
                            "validation": fallback_repair_validation,
                        }
                    )
                    final_versions.append(
                        {
                            "label": fallback_repair_label,
                            "result": fallback_repair_result,
                            "score": None,
                            "score_report": "",
                            "objective_validation": fallback_repair_validation,
                            "accepted": fallback_repair_accepted,
                        }
                    )
                    if fallback_repair_accepted:
                        final_result = fallback_repair_result
                        current_validation = fallback_repair_validation
                        if args.score_final:
                            score_results[fallback_repair_label], parsed_scores[fallback_repair_label] = run_score(
                                fallback_repair_label,
                                final_result.content,
                                4400 + suite_index * 100 + fallback_repair_index,
                            )
                            fallback_repair_model_score = parsed_scores.get(fallback_repair_label)
                            if fallback_repair_model_score is not None:
                                parsed_scores["breakout"] = fallback_repair_model_score
                                score_results["breakout"] = score_results[fallback_repair_label]
                                baseline_score = parsed_scores.get("baseline")
                                if baseline_score is not None:
                                    score_delta = int(fallback_repair_model_score) - int(baseline_score)
                        break

        objective_benchmark = {
            "id": objective_case.get("id"),
            "validator": objective_case.get("validator"),
            "baseline": baseline_validation,
            "initial_breakout": initial_breakout_validation,
            "breakout": current_validation,
            "initial_score_delta": int(initial_breakout_validation["score"]) - int(baseline_validation["score"]),
            "score_delta": int(current_validation["score"]) - int(baseline_validation["score"]),
            "initial_pass_delta": int(bool(initial_breakout_validation["passed"])) - int(bool(baseline_validation["passed"])),
            "pass_delta": int(bool(current_validation["passed"])) - int(bool(baseline_validation["passed"])),
            "objective_repair_rounds": args.objective_repair_rounds,
            "objective_repairs_attempted": sum(
                1
                for version in objective_versions
                if str(version.get("label", "")).startswith("objective-repair-")
            ),
            "objective_repairs_accepted": sum(
                1
                for version in objective_versions
                if str(version.get("label", "")).startswith("objective-repair-") and version.get("accepted")
            ),
            "objective_fallback_recombine_attempted": sum(
                1 for version in objective_versions if version.get("label") == "objective-fallback-recombine"
            ),
            "objective_fallback_recombine_accepted": sum(
                1
                for version in objective_versions
                if version.get("label") == "objective-fallback-recombine" and version.get("accepted")
            ),
            "objective_fallback_repair_attempted": sum(
                1
                for version in objective_versions
                if str(version.get("label", "")).startswith("objective-fallback-repair-")
            ),
            "objective_fallback_repair_accepted": sum(
                1
                for version in objective_versions
                if str(version.get("label", "")).startswith("objective-fallback-repair-") and version.get("accepted")
            ),
            "versions": objective_versions,
        }

    return {
        "kind": "turbo-speculative-breakout",
        "schema_version": RESULT_SCHEMA_VERSION,
        "started": args.started,
        "task": task,
        "config": breakout_run_config(args, slots, effective_verifier_mode),
        "server_props": props,
        "decision": {
            "short_circuit_phase": None,
            "decision_reason": baseline_gate_reason,
            "terminal_validation_surface": "cleaned_content",
            "branches_launched": len(branch_results),
            "fanout_waves": fanout_waves,
            "fanout_stages": fanout_stage_results,
        },
        "prefix": {
            "backend": args.prefix_clone_backend if args.prefix_clone else None,
            "result": asdict(prefix_result) if prefix_result else None,
            "clone_wall_s": clone_wall_s,
            "fork": fork_result,
            "fork_waves": fork_waves,
            "save": save_result,
            "restore": restore_results,
            "clone_filename": clone_filename if args.prefix_clone and args.prefix_clone_backend == "file" else None,
            "cleanup_errors": [],
        },
        "summaries": {
            "baseline": summarize_requests([baseline_result]) if baseline_result else None,
            "branches": summarize_requests(branch_results),
            "verifiers": summarize_requests(verifier_results),
            "final": summarize_requests([final_request_result]) if final_request_result else None,
            "scores": summarize_requests(list(score_results.values())) if score_results else None,
            "objective_repairs": summarize_requests(objective_repair_results) if objective_repair_results else None,
        },
        "baseline": asdict(baseline_result) if baseline_result else None,
        "final_scores": {
            label: {
                "score": parsed_scores.get(label),
                "report": result.content,
                "ok": result.ok,
                "error": result.error,
            }
            for label, result in score_results.items()
        },
        "final_versions": [
            {
                "label": version["label"],
                "score": version["score"],
                "score_report": version["score_report"],
                "request": asdict(version["result"]),
                "objective_validation": version.get("objective_validation"),
                "accepted": version.get("accepted"),
            }
            for version in final_versions
        ],
        "score_delta": score_delta,
        "objective_benchmark": objective_benchmark,
        "selected": {
            "source": "branch",
            "index": selected["index"],
            "slot": selected["slot"],
            "score": selected["score"],
            "prefix_ok": selected["prefix_ok"],
            "objective_score": selected.get("objective_score"),
            "objective_passed": selected.get("objective_passed"),
            "objective_failed_checks": failed_checks(selected.get("objective_validation") or {}),
            "model_accepted": selected["score"] is not None
            and selected["score"] >= args.accept_score
            and selected["prefix_ok"] is not False,
            "accepted": bool(selected.get("objective_passed"))
            if objective_case is not None
            else (
                selected["score"] is not None
                and selected["score"] >= args.accept_score
                and selected["prefix_ok"] is not False
            ),
        },
        "packets": packets,
        "requests": {
            "baseline": asdict(baseline_result) if baseline_result else None,
            "branches": [asdict(r) for r in branch_results],
            "verifiers": [asdict(r) for r in verifier_results],
            "final": asdict(final_request_result) if final_request_result else None,
            "scores": {label: asdict(result) for label, result in score_results.items()},
            "objective_repairs": [asdict(result) for result in objective_repair_results],
        },
        "final_answer": final_result.content,
        "log_path": str(log_path),
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prototype single-answer branch, verify, and recombine over llama-server slots.",
    )
    parser.add_argument("--task", help="Task text. If omitted, read --task-file or stdin.")
    parser.add_argument("--task-file", help="Read task text from this file.")
    parser.add_argument("--task-suite", choices=["none", *sorted(DEFAULT_TASK_SUITES)], default="none", help="Run a built-in task suite")
    parser.add_argument("--task-suite-file", help="Read task suite from JSONL, newline, or --- separated text")
    parser.add_argument("--benchmark-suite", choices=["none", *sorted(DEFAULT_BENCHMARK_SUITES)], default="none", help="Run an objective benchmark suite with deterministic validators")
    parser.add_argument("--benchmark-suite-file", help="Read objective benchmark cases from JSON or JSONL")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for JSON result and server log")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--attach", action="store_true", help="Use an already-running llama-server on --port")
    parser.add_argument("--bin", default=DEFAULT_BIN, help="Path to llama-server for launch mode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model for launch mode")
    parser.add_argument("--slot-save-path", default=DEFAULT_SLOT_SAVE_PATH, help="Slot-save directory for launch mode")
    parser.add_argument("--branch-slots", default="0-11", help="Comma list/ranges of slots to use, e.g. 0-11 or 0,2,4")
    parser.add_argument("--prefix-slot", type=int, default=0)
    parser.add_argument("--final-slot", type=int, default=0)
    parser.add_argument("--score-slot", type=int, default=0)
    parser.add_argument("--prefix-clone", action=argparse.BooleanOptionalAction, default=True, help="Save a pure prompt prefix and restore it into all branch slots")
    parser.add_argument("--prefix-clone-backend", choices=("file", "fork"), default="file", help="Clone the shared prefix through slot files or in-memory unified-KV sequence sharing")
    parser.add_argument("--run-baseline", action=argparse.BooleanOptionalAction, default=True, help="Generate a single-pass baseline answer before branch fanout")
    parser.add_argument("--score-final", action=argparse.BooleanOptionalAction, default=True, help="Score baseline and breakout answers with the same rubric")
    parser.add_argument("--verifier-mode", choices=("model", "objective", "none"), default="model", help="Verifier source for branch selection")
    parser.add_argument("--objective-fast-path", action=argparse.BooleanOptionalAction, default=False, help="In objective benchmark mode, use deterministic branch validation and skip recombine when possible")
    parser.add_argument("--objective-baseline-short-circuit", action=argparse.BooleanOptionalAction, default=True, help="In objective fast path without model scoring, return a terminal-valid baseline before branch fanout")
    parser.add_argument("--objective-fast-fallback-recombine", action=argparse.BooleanOptionalAction, default=True, help="In objective fast path, run recombine only if direct branch/repair still fails validation")
    parser.add_argument("--objective-fast-fallback-model-verifier", action=argparse.BooleanOptionalAction, default=True, help="In objective fast path, run model verifier fanout only before fallback recombine")
    parser.add_argument("--objective-fast-fallback-repair-rounds", type=int, default=1, help="Verifier-informed objective repair rounds after fallback recombine fails")
    parser.add_argument("--adaptive-fanout", action=argparse.BooleanOptionalAction, default=False, help="Escalate objective branch width in stages and stop after a validator pass")
    parser.add_argument("--fanout-stages", default="1,2,4,8", help="Cumulative adaptive fanout widths, capped by available branch slots")
    parser.add_argument("--preserve-prefix-root", action=argparse.BooleanOptionalAction, default=False, help="Keep an untouched StateTree root, collapse failed waves, and commit a passing branch")
    parser.add_argument("--pcbt", action=argparse.BooleanOptionalAction, default=False,
                        help="Drive stages through proof-carrying branch transactions (/transactions API; server needs TREEBEARD_PCBT_ENABLE=1)")
    parser.add_argument("--baseline-tokens", type=int, default=512)
    parser.add_argument("--branch-tokens", type=int, default=384)
    parser.add_argument("--verify-tokens", type=int, default=192)
    parser.add_argument("--final-tokens", type=int, default=512)
    parser.add_argument("--score-tokens", type=int, default=160)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=False, help="Forward ignore_eos to every completion request for forced-length throughput measurements")
    parser.add_argument("--stop-strings", action=argparse.BooleanOptionalAction, default=True, help="Use harness stop strings; disable with --no-stop-strings for forced-length throughput measurements")
    parser.add_argument("--repair-rounds", type=int, default=1, help="Repair the final answer if breakout does not beat the baseline score")
    parser.add_argument("--objective-repair-rounds", type=int, default=1, help="Repair benchmark answers using deterministic validator feedback")
    parser.add_argument("--score-answer-max-chars", type=int, default=6000)
    parser.add_argument("--verify-prefix-chars", type=int, default=1600)
    parser.add_argument("--candidate-max-chars", type=int, default=6000)
    parser.add_argument("--verifier-max-chars", type=int, default=1800)
    parser.add_argument("--accept-score", type=int, default=70)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=19_907)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ncmoe", type=int, default=0)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--kvu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source-oneapi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-server-args", default="", help="Extra launch-mode llama-server args, parsed with shlex")
    args = parser.parse_args()
    args.started = utc_stamp()

    benchmark_cases = resolve_benchmarks(args)
    tasks = [str(case["task"]) for case in benchmark_cases] if benchmark_cases else resolve_tasks(args)
    if not tasks or any(not task.strip() for task in tasks):
        raise SystemExit("empty task")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.started}-{os.getpid()}"
    log_path = out_dir / f"{run_id}.server.log"
    json_path = out_dir / f"{run_id}.json"
    suite_json_path = out_dir / f"{run_id}.suite.json"
    jsonl_path = out_dir / "results.jsonl"

    proc = None
    try:
        if args.attach:
            wait_healthy(args.port, None, args.startup_timeout)
        else:
            proc = launch_server(args, log_path)
            wait_healthy(args.port, proc, args.startup_timeout)

        use_fork_backend = args.prefix_clone and args.prefix_clone_backend == "fork"
        cleanup_slots = parse_slot_list(args.branch_slots)
        if args.preserve_prefix_root and args.prefix_slot not in cleanup_slots:
            cleanup_slots.append(args.prefix_slot)
        if use_fork_backend:
            cleanup_errors = cleanup_fork_reservations(
                args.port,
                cleanup_slots,
                args.prefix_slot,
                args.request_timeout,
            )
            if cleanup_errors:
                raise RuntimeError("failed to clear stale fork reservations: " + "; ".join(cleanup_errors))

        results = []
        with jsonl_path.open("a", encoding="utf-8") as out:
            for index, task in enumerate(tasks):
                case = benchmark_cases[index] if benchmark_cases else None
                cleanup_errors = []
                try:
                    result = run_breakout(args, task, out_dir, log_path, case, index)
                finally:
                    if use_fork_backend:
                        cleanup_errors = cleanup_fork_reservations(
                            args.port,
                            cleanup_slots,
                            args.prefix_slot,
                            args.request_timeout,
                        )
                result["prefix"]["cleanup_errors"] = cleanup_errors
                result["suite_index"] = index
                results.append(result)
                task_json_path = out_dir / f"{run_id}.task{index:03d}.json"
                result["result_path"] = str(task_json_path)
                task_json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                out.write(json.dumps(result, sort_keys=True) + "\n")
                out.flush()
                if cleanup_errors:
                    raise RuntimeError("failed to release fork reservations: " + "; ".join(cleanup_errors))

        if len(results) == 1:
            result = results[0]
            json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(result["final_answer"].strip())
            print(f"\nresult: {json_path}")
        else:
            objective_summary = summarize_objective_benchmarks(results) if benchmark_cases else None
            accuracy_throughput_summary = summarize_accuracy_and_throughput(results) if benchmark_cases else None
            model_score_summary = summarize_suite(results)
            if benchmark_cases and model_score_summary.get("scored_tasks") == 0:
                model_score_summary = None
            suite = {
                "kind": "turbo-speculative-breakout-objective-benchmark" if benchmark_cases else "turbo-speculative-breakout-suite",
                "schema_version": RESULT_SCHEMA_VERSION,
                "started": args.started,
                "tasks": len(results),
                "config": results[0].get("config") if results else None,
                "decision_summary": summarize_decisions(results),
                "summary": objective_summary or summarize_suite(results),
                "accuracy_throughput_summary": accuracy_throughput_summary,
                "model_score_summary": model_score_summary,
                "results": [
                    {
                        "suite_index": result["suite_index"],
                        "result_path": result["result_path"],
                        "baseline_score": result.get("final_scores", {}).get("baseline", {}).get("score"),
                        "breakout_score": result.get("final_scores", {}).get("breakout", {}).get("score"),
                        "score_delta": result.get("score_delta"),
                        "objective_benchmark": result.get("objective_benchmark"),
                    }
                    for result in results
                ],
            }
            suite_json_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(suite["summary"], indent=2, sort_keys=True))
            print(f"\nsuite:  {suite_json_path}")
        print(f"jsonl:  {jsonl_path}")
        if not args.attach:
            print(f"log:    {log_path}")
        return 0
    finally:
        if not args.attach:
            stop_process(proc)
            if proc is not None:
                log_file = getattr(proc, "_turbo_log_file", None)
                if log_file is not None:
                    log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
