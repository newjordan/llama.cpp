#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BIN = "/home/frosty40/builds/turbo-experimental-build/bin/llama-server"
DEFAULT_MODEL = "/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
RESULT_SCHEMA_VERSION = 2
VALID_VARIANTS = {
    "attached",
    "baseline",
    "control",
    "probe",
    "compact",
    "compact-probe",
    "indexed",
    "indexed-probe",
    "indexed-force",
    "indexed-force-probe",
}

PROBE_RE = re.compile(
    r"kv-page-probe\[(?P<tag>[^\]]+)\]: "
    r"stream=(?P<stream>\d+) "
    r"n_kv=(?P<n_kv>-?\d+) "
    r"size=(?P<size>\d+) "
    r"used=(?P<used>\d+) "
    r"used_min=(?P<used_min>\d+) "
    r"used_max_p1=(?P<used_max_p1>\d+) "
    r"holes=(?P<holes>\d+) "
    r"live_pages=(?P<live_pages>\d+) "
    r"dense_pages=(?P<dense_pages>\d+) "
    r"largest_free_run=(?P<largest_free_run>\d+) "
    r"head=(?P<head>\d+) "
    r"slot_rows=(?P<slot_rows>\d+) "
    r"slot_pages=(?P<slot_pages>\d+) "
    r"page_size=(?P<page_size>\d+)"
)


@dataclass
class RequestResult:
    ok: bool
    label: str
    wall_s: float
    prompt_tokens: int
    predicted_tokens: int
    prompt_tps: float | None
    predicted_tps: float | None
    content: str | None = None
    tokens: list[int] | None = None
    content_sha256: str | None = None
    tokens_sha256: str | None = None
    error: str | None = None


def parse_csv_ints(value: str) -> list[int]:
    out = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer greater than zero")
    return parsed


def resolve_variants(value: str, attach: bool, fragment_slots: int) -> list[str]:
    variants = [variant.strip() for variant in value.split(",") if variant.strip()]
    if not variants:
        raise ValueError("expected at least one variant")
    if len(set(variants)) != len(variants):
        raise ValueError("duplicate variants are not allowed")

    unknown = sorted(set(variants) - VALID_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")

    if attach:
        if variants != ["attached"]:
            raise ValueError(
                "--attach cannot perform controlled ablations; use --variants attached for one "
                "unlabelled measurement, or let the harness launch managed variant servers"
            )
        if fragment_slots > 0:
            raise ValueError("--attach cannot prepare fragmented slots because that erases live slot state")
        return variants

    if "attached" in variants:
        raise ValueError("the attached variant requires --attach")
    if len(variants) > 1 and "baseline" not in variants:
        raise ValueError("multi-variant ablations require baseline")
    if "baseline" in variants:
        variants.remove("baseline")
        variants.insert(0, "baseline")
    if "control" in variants:
        variants.remove("control")
        variants.insert(1 if variants and variants[0] == "baseline" else 0, "control")
    return variants


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_tokens(tokens: list[int]) -> str:
    encoded = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def compare_request_outputs(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    max_mismatches: int = 20,
) -> dict[str, Any]:
    mismatch_count = 0
    exact_token_matches = 0
    exact_content_matches = 0
    mismatches: list[dict[str, Any]] = []

    for index in range(max(len(reference), len(candidate))):
        ref = reference[index] if index < len(reference) else None
        cur = candidate[index] if index < len(candidate) else None
        reasons: list[str] = []

        if ref is None or cur is None:
            reasons.append("request_count")
        else:
            if ref.get("label") != cur.get("label"):
                reasons.append("label")
            if not ref.get("ok") or not cur.get("ok"):
                reasons.append("request_error")

            ref_tokens = ref.get("tokens")
            cur_tokens = cur.get("tokens")
            ref_content = ref.get("content")
            cur_content = cur.get("content")
            if ref_tokens is None or cur_tokens is None or ref_content is None or cur_content is None:
                reasons.append("missing_output")
            else:
                if ref_tokens == cur_tokens:
                    exact_token_matches += 1
                else:
                    reasons.append("tokens")
                if ref_content == cur_content:
                    exact_content_matches += 1
                else:
                    reasons.append("content")

        if reasons:
            mismatch_count += 1
            if len(mismatches) < max_mismatches:
                mismatches.append(
                    {
                        "request_index": index,
                        "reference_label": ref.get("label") if ref else None,
                        "candidate_label": cur.get("label") if cur else None,
                        "reasons": reasons,
                        "reference_tokens_sha256": ref.get("tokens_sha256") if ref else None,
                        "candidate_tokens_sha256": cur.get("tokens_sha256") if cur else None,
                        "reference_content_sha256": ref.get("content_sha256") if ref else None,
                        "candidate_content_sha256": cur.get("content_sha256") if cur else None,
                    }
                )

    return {
        "reference_requests": len(reference),
        "candidate_requests": len(candidate),
        "exact_token_matches": exact_token_matches,
        "exact_content_matches": exact_content_matches,
        "mismatch_count": mismatch_count,
        "passed": mismatch_count == 0,
        "mismatches": mismatches,
    }


def compare_output_parity(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    phases = {
        "warmup": compare_request_outputs(
            [reference["warmup_request"]] if reference.get("warmup_request") else [],
            [candidate["warmup_request"]] if candidate.get("warmup_request") else [],
        ),
        "fragment_fill": compare_request_outputs(
            reference.get("fragmentation", {}).get("fill_requests", []),
            candidate.get("fragmentation", {}).get("fill_requests", []),
        ),
        "measured": compare_request_outputs(reference.get("requests", []), candidate.get("requests", [])),
    }
    passed = all(phase["passed"] for phase in phases.values())
    return {
        "status": "matched" if passed else "mismatched",
        "scope": "top1_sequence",
        "reference_variant": str(reference.get("variant") or "baseline"),
        "passed": passed,
        "phases": phases,
    }


def reference_output_parity(result: dict[str, Any]) -> dict[str, Any]:
    phases = {
        "warmup": compare_request_outputs(
            [result["warmup_request"]] if result.get("warmup_request") else [],
            [result["warmup_request"]] if result.get("warmup_request") else [],
        ),
        "fragment_fill": compare_request_outputs(
            result.get("fragmentation", {}).get("fill_requests", []),
            result.get("fragmentation", {}).get("fill_requests", []),
        ),
        "measured": compare_request_outputs(result.get("requests", []), result.get("requests", [])),
    }
    return {
        "status": "reference",
        "scope": "top1_sequence",
        "reference_variant": str(result.get("variant") or "baseline"),
        "passed": all(phase["passed"] for phase in phases.values()),
        "phases": phases,
    }


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
        except Exception as exc:  # noqa: BLE001 - keep launch diagnostics simple
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"server did not become healthy on port {port}: {last_error}")


def make_prompt(label: str, approx_tokens: int) -> str:
    stem = (
        f"Scenario {label}. Keep all details internally consistent. "
        "This is a deterministic cache-pressure prompt for llama.cpp KV testing. "
    )
    block = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
        "xray yankee zulu "
    )
    words_needed = max(1, approx_tokens)
    words = (block.split() * ((words_needed // 26) + 1))[:words_needed]
    return stem + " ".join(words) + "\nReturn a compact numbered summary."


def send_completion(
    port: int,
    label: str,
    prompt_tokens: int,
    gen_tokens: int,
    seed: int,
    id_slot: int | None = None,
    cache_prompt: bool = False,
) -> RequestResult:
    payload = {
        "prompt": make_prompt(label, prompt_tokens),
        "n_predict": gen_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": cache_prompt,
        "stream": False,
        "return_tokens": True,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot

    t0 = time.perf_counter()
    try:
        data = http_json("POST", f"http://127.0.0.1:{port}/completion", payload, timeout=900.0)
        wall_s = time.perf_counter() - t0
        if not isinstance(data, dict):
            raise ValueError("completion response is not a JSON object")
        content = data.get("content")
        tokens = data.get("tokens")
        if not isinstance(content, str):
            raise ValueError("completion response is missing string content")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("completion response is missing integer-list tokens")
        timings = data.get("timings", {}) if isinstance(data, dict) else {}
        return RequestResult(
            ok=True,
            label=label,
            wall_s=wall_s,
            prompt_tokens=int(timings.get("prompt_n") or 0),
            predicted_tokens=int(timings.get("predicted_n") or 0),
            prompt_tps=float(timings["prompt_per_second"]) if timings.get("prompt_per_second") is not None else None,
            predicted_tps=float(timings["predicted_per_second"]) if timings.get("predicted_per_second") is not None else None,
            content=content,
            tokens=tokens,
            content_sha256=hash_content(content),
            tokens_sha256=hash_tokens(tokens),
        )
    except Exception as exc:  # noqa: BLE001 - result JSON should capture request failures
        return RequestResult(
            ok=False,
            label=label,
            wall_s=time.perf_counter() - t0,
            prompt_tokens=0,
            predicted_tokens=0,
            prompt_tps=None,
            predicted_tps=None,
            error=str(exc),
        )


def run_wave(port: int, wave: int, concurrency: int, prompt_lens: list[int], gen_tokens: int) -> list[RequestResult]:
    barrier = threading.Barrier(concurrency)

    def one(i: int) -> RequestResult:
        barrier.wait()
        prompt_tokens = prompt_lens[(wave * concurrency + i) % len(prompt_lens)]
        label = f"w{wave:03d}-r{i:03d}-p{prompt_tokens}"
        return send_completion(
            port,
            label,
            prompt_tokens,
            gen_tokens,
            seed=17_000 + wave * 1000 + i,
            id_slot=i,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, range(concurrency)))


def erase_slot(port: int, slot_id: int) -> None:
    http_json("POST", f"http://127.0.0.1:{port}/slots/{slot_id}?action=erase", {}, timeout=120.0)


def prepare_fragmented_slots(args: argparse.Namespace) -> dict[str, Any]:
    n_slots = min(args.fragment_slots, args.parallel)
    if n_slots <= 0:
        return {"enabled": False}
    if n_slots < 3:
        raise RuntimeError("--fragment-slots must be at least 3 to leave holes below live rows")

    fill_specs: dict[int, tuple[str, int]] = {}
    fill_results: list[RequestResult] = []

    for slot_id in range(n_slots):
        prompt_tokens = args.fragment_prompt_lens[slot_id % len(args.fragment_prompt_lens)]
        label = f"frag-fill-s{slot_id:02d}-p{prompt_tokens}"
        fill_specs[slot_id] = (label, prompt_tokens)
        result = send_completion(
            args.port,
            label,
            prompt_tokens,
            args.fragment_gen_tokens,
            seed=31_000 + slot_id,
            id_slot=slot_id,
            cache_prompt=True,
        )
        fill_results.append(result)
        if not result.ok:
            raise RuntimeError(f"fragment fill failed for slot {slot_id}: {result.error}")

    erased_slots = [slot_id for slot_id in range(n_slots) if slot_id % 2 == 1]
    for slot_id in erased_slots:
        erase_slot(args.port, slot_id)

    survivor_slots = [slot_id for slot_id in range(n_slots) if slot_id not in erased_slots]
    survivor_slots = [slot_id for slot_id in survivor_slots if slot_id != 0] or survivor_slots
    fill_cache_tokens = sum(r.prompt_tokens + r.predicted_tokens for r in fill_results if r.ok)

    return {
        "enabled": True,
        "n_slots": n_slots,
        "erased_slots": erased_slots,
        "survivor_slots": survivor_slots,
        "fill_cache_tokens": fill_cache_tokens,
        "fill_context_pressure": fill_cache_tokens / args.ctx if args.ctx else None,
        "fill_exceeds_context": fill_cache_tokens > args.ctx,
        "fill_specs": {str(k): {"label": v[0], "prompt_tokens": v[1]} for k, v in fill_specs.items()},
        "fill_summary": summarize_requests(fill_results),
        "fill_requests": [asdict(result) for result in fill_results],
    }


def run_fragmented_wave(args: argparse.Namespace, wave: int, frag: dict[str, Any]) -> list[RequestResult]:
    survivor_slots = [int(v) for v in frag["survivor_slots"]]
    fill_specs = {int(k): (v["label"], int(v["prompt_tokens"])) for k, v in frag["fill_specs"].items()}
    concurrency = len(survivor_slots)
    barrier = threading.Barrier(concurrency)

    def one(i: int) -> RequestResult:
        barrier.wait()
        slot_id = survivor_slots[i]
        label, prompt_tokens = fill_specs[slot_id]
        return send_completion(
            args.port,
            label,
            prompt_tokens,
            args.gen_tokens,
            seed=41_000 + wave * 1000 + slot_id,
            id_slot=slot_id,
            cache_prompt=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, range(concurrency)))


def summarize_requests(results: list[RequestResult]) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    pred_tps = [r.predicted_tps for r in ok if r.predicted_tps is not None]
    prompt_tps = [r.prompt_tps for r in ok if r.prompt_tps is not None]
    wall_s = [r.wall_s for r in ok]
    total_predicted = sum(r.predicted_tokens for r in ok)
    total_prompt = sum(r.prompt_tokens for r in ok)
    elapsed = max(wall_s) if wall_s else 0.0

    return {
        "requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(results) - len(ok),
        "total_prompt_tokens": total_prompt,
        "total_predicted_tokens": total_predicted,
        "max_request_wall_s": elapsed,
        "aggregate_predicted_tps_floor": total_predicted / elapsed if elapsed else None,
        "mean_slot_predicted_tps": statistics.mean(pred_tps) if pred_tps else None,
        "min_slot_predicted_tps": min(pred_tps) if pred_tps else None,
        "max_slot_predicted_tps": max(pred_tps) if pred_tps else None,
        "mean_prompt_tps": statistics.mean(prompt_tps) if prompt_tps else None,
        "p50_request_wall_s": statistics.median(wall_s) if wall_s else None,
        "errors": [r.error for r in results if r.error][:5],
    }


def parse_probe_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PROBE_RE.search(line)
            if not m:
                continue
            row: dict[str, Any] = m.groupdict()
            row["tag"] = str(row["tag"])
            for key, value in list(row.items()):
                if key != "tag":
                    row[key] = int(value)
            row["hole_ratio"] = row["holes"] / row["used_max_p1"] if row["used_max_p1"] else 0.0
            row["dense_to_live_pages"] = row["dense_pages"] / row["live_pages"] if row["live_pages"] else None
            row["n_kv_to_used"] = row["n_kv"] / row["used"] if row["used"] else None
            rows.append(row)
    return rows


def summarize_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"probe_rows": 0}

    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    max_row = max(rows, key=lambda r: r["hole_ratio"])
    return {
        "probe_rows": len(rows),
        "max_n_kv": max(vals("n_kv")),
        "mean_n_kv": statistics.mean(vals("n_kv")),
        "max_used": max(vals("used")),
        "max_used_max_p1": max(vals("used_max_p1")),
        "max_holes": max(vals("holes")),
        "max_hole_ratio": max(vals("hole_ratio")),
        "max_dense_pages": max(vals("dense_pages")),
        "max_live_pages": max(vals("live_pages")),
        "max_dense_to_live_pages": max(vals("dense_to_live_pages")) if vals("dense_to_live_pages") else None,
        "max_largest_free_run": max(vals("largest_free_run")),
        "worst_fragmentation_row": max_row,
    }


def stop_process(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def launch_server(args: argparse.Namespace, variant: str, log_path: Path, slot_save_path: Path | None) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env["GGML_SYCL_ENABLE_FUSION"] = "1"
    if variant in {"probe", "compact-probe", "indexed-probe", "indexed-force-probe"}:
        env["LLAMA_KV_PAGE_PROBE"] = "1"
    else:
        env.pop("LLAMA_KV_PAGE_PROBE", None)
    if variant in {"compact", "compact-probe", "indexed", "indexed-probe", "indexed-force", "indexed-force-probe"}:
        env["LLAMA_KV_COMPACT_ATTN"] = "1"
    else:
        env.pop("LLAMA_KV_COMPACT_ATTN", None)
    if variant in {"indexed", "indexed-probe", "indexed-force", "indexed-force-probe"}:
        env["LLAMA_KV_INDEXED_FATTN"] = "2" if variant.startswith("indexed-force") else "1"
        env["GGML_SYCL_FA_DEBUG"] = "1"
    else:
        env.pop("LLAMA_KV_INDEXED_FATTN", None)
        env.pop("GGML_SYCL_FA_DEBUG", None)

    server_cmd = [
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
        "--jinja",
        "-a", f"turbo-kv-page-{variant}-np{args.parallel}-c{args.ctx}",
    ]
    if slot_save_path is not None:
        server_cmd += ["--slot-save-path", str(slot_save_path)]
    if args.fragment_slots > 0:
        server_cmd += ["--no-cache-idle-slots"]

    env_exports = "export GGML_SYCL_ENABLE_FUSION=1;"
    if variant in {"probe", "compact-probe", "indexed-probe", "indexed-force-probe"}:
        env_exports += " export LLAMA_KV_PAGE_PROBE=1;"
    else:
        env_exports += " unset LLAMA_KV_PAGE_PROBE;"
    if variant in {"compact", "compact-probe", "indexed", "indexed-probe", "indexed-force", "indexed-force-probe"}:
        env_exports += " export LLAMA_KV_COMPACT_ATTN=1;"
    else:
        env_exports += " unset LLAMA_KV_COMPACT_ATTN;"
    if variant in {"indexed", "indexed-probe", "indexed-force", "indexed-force-probe"}:
        indexed_value = "2" if variant.startswith("indexed-force") else "1"
        env_exports += f" export LLAMA_KV_INDEXED_FATTN={indexed_value}; export GGML_SYCL_FA_DEBUG=1;"
    else:
        env_exports += " unset LLAMA_KV_INDEXED_FATTN; unset GGML_SYCL_FA_DEBUG;"

    cmd = server_cmd
    if args.source_oneapi:
        cmd = [
            "bash",
            "-lc",
            "source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; "
            + env_exports
            + " exec "
            + shlex.join(server_cmd),
        ]

    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    proc._turbo_log_file = log_f  # type: ignore[attr-defined]
    return proc


def run_variant(args: argparse.Namespace, variant: str, run_id: int, out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / f"{variant}-run{run_id}.log"
    slot_save_path = out_dir / f"{variant}-run{run_id}-slots" if args.fragment_slots > 0 and not args.attach else None
    if slot_save_path is not None:
        slot_save_path.mkdir(parents=True, exist_ok=True)

    proc = None
    server_returncode: int | None = None
    all_results: list[RequestResult] = []
    warmup_result: RequestResult | None = None
    fragmentation: dict[str, Any] = {"enabled": False}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        if not args.attach:
            proc = launch_server(args, variant, log_path, slot_save_path)
            wait_healthy(args.port, proc, args.startup_timeout)
        else:
            wait_healthy(args.port, None, args.startup_timeout)

        props = http_json("GET", f"http://127.0.0.1:{args.port}/props", timeout=30.0)
        total_slots = int(props.get("total_slots") or 0) if isinstance(props, dict) else 0
        if total_slots and args.parallel > total_slots:
            raise RuntimeError(
                f"--parallel requests {args.parallel} fixed slots, but the server reports {total_slots}"
            )

        if args.warmup:
            warmup_result = send_completion(
                args.port,
                "warmup",
                min(args.prompt_lens),
                min(16, args.gen_tokens),
                seed=1,
                id_slot=0,
            )

        if args.fragment_slots > 0:
            fragmentation = prepare_fragmented_slots(args)

        for wave in range(args.waves):
            if args.fragment_slots > 0:
                all_results.extend(run_fragmented_wave(args, wave, fragmentation))
            else:
                all_results.extend(run_wave(args.port, wave, args.parallel, args.prompt_lens, args.gen_tokens))
            if args.wave_pause_s > 0:
                time.sleep(args.wave_pause_s)
    finally:
        if not args.attach:
            stop_process(proc)
            if proc is not None:
                server_returncode = proc.returncode
            log_file = getattr(proc, "_turbo_log_file", None)
            if log_file is not None:
                log_file.close()

    probe_rows = parse_probe_log(log_path) if variant in {"probe", "compact-probe", "indexed-probe", "indexed-force-probe"} else []
    result = {
        "kind": "turbo-kv-page-ablation",
        "schema_version": RESULT_SCHEMA_VERSION,
        "started": started,
        "variant": variant,
        "run_id": run_id,
        "config": {
            "bin": args.bin,
            "model": args.model,
            "ctx": args.ctx,
            "parallel": args.parallel,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "threads": args.threads,
            "cache_type_k": args.cache_type_k,
            "cache_type_v": args.cache_type_v,
            "flash_attn": args.flash_attn,
            "waves": args.waves,
            "prompt_lens": args.prompt_lens,
            "gen_tokens": args.gen_tokens,
            "fragment_slots": args.fragment_slots,
            "fragment_prompt_lens": args.fragment_prompt_lens,
            "fragment_gen_tokens": args.fragment_gen_tokens,
            "attach": args.attach,
        },
        "fragmentation": fragmentation,
        "warmup_request": asdict(warmup_result) if warmup_result else None,
        "request_summary": summarize_requests(all_results),
        "probe_summary": summarize_probe(probe_rows),
        "log_path": str(log_path),
        "server_returncode": server_returncode,
        "requests": [asdict(r) for r in all_results],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Turbo unified-KV page-probe ablations against llama-server.",
    )
    parser.add_argument("--bin", default=DEFAULT_BIN, help="Path to llama-server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--out-dir", default="/tmp/turbo-kv-page-ablate", help="Directory for logs and JSONL")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--parallel", type=positive_int, default=12)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--ubatch", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ncmoe", type=int, default=0)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--waves", type=positive_int, default=3)
    parser.add_argument("--prompt-lens", type=parse_csv_ints, default=parse_csv_ints("256,2048,8192"))
    parser.add_argument("--gen-tokens", type=positive_int, default=96)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--variants", default="baseline,probe", help="Comma-separated managed variants (control repeats baseline settings), or attached with --attach")
    parser.add_argument("--fragment-slots", type=int, default=0, help="Fill this many explicit slots, erase odd slots, then measure surviving slots")
    parser.add_argument("--fragment-prompt-lens", type=parse_csv_ints, default=parse_csv_ints("128,256,384,512"), help="Prompt lengths used to prefill fragment slots")
    parser.add_argument("--fragment-gen-tokens", type=positive_int, default=2, help="Generation tokens used during fragment-slot prefill")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wave-pause-s", type=float, default=0.25)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--attach", action="store_true", help="Use an already-running server on --port")
    parser.add_argument("--fail-on-output-mismatch", action=argparse.BooleanOptionalAction, default=True, help="Return nonzero when a managed variant differs from its same-run baseline")
    parser.add_argument("--source-oneapi", action=argparse.BooleanOptionalAction, default=True, help="Source oneAPI before launching llama-server")
    args = parser.parse_args()

    try:
        variants = resolve_variants(args.variants, args.attach, args.fragment_slots)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"

    parity_failed = False
    measurement_failed = False
    with out_path.open("a", encoding="utf-8") as out:
        for run_id in range(args.repeats):
            reference: dict[str, Any] | None = None
            control_stable: bool | None = None
            for variant in variants:
                print(f"run {run_id} variant {variant}", file=sys.stderr, flush=True)
                result = run_variant(args, variant, run_id, out_dir)

                if variant == "baseline":
                    reference = result
                    result["output_parity"] = reference_output_parity(result)
                elif reference is None:
                    result["output_parity"] = {
                        "status": "not_compared",
                        "scope": "top1_sequence",
                        "reference_variant": None,
                        "passed": None,
                        "phases": {},
                    }
                else:
                    result["output_parity"] = compare_output_parity(reference, result)

                if variant == "control":
                    control_stable = bool(result["output_parity"]["passed"])
                elif reference is not None and variant != "baseline":
                    result["output_parity"]["control_status"] = (
                        "stable" if control_stable else "unstable"
                    ) if control_stable is not None else "not_run"
                    result["output_parity"]["variant_attribution_valid"] = control_stable

                if result["output_parity"]["passed"] is False:
                    parity_failed = True
                measurement_failed = measurement_failed or bool(
                    result.get("request_summary", {}).get("failed_requests")
                )
                measurement_failed = measurement_failed or bool(
                    result.get("warmup_request") and not result["warmup_request"].get("ok")
                )

                out.write(json.dumps(result) + "\n")
                out.flush()
                summary = {
                    "variant": result["variant"],
                    "run_id": run_id,
                    "request_summary": result["request_summary"],
                    "probe_summary": result["probe_summary"],
                    "output_parity": result["output_parity"],
                    "log_path": result["log_path"],
                }
                print(json.dumps(summary, indent=2), flush=True)

    print(f"results: {out_path}")
    print(f"logs: {out_dir}")
    return 1 if measurement_failed or (parity_failed and args.fail_on_output_mismatch) else 0


if __name__ == "__main__":
    raise SystemExit(main())
