#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
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


class ManagedServer:
    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.args = args
        self.log_path = log_path
        self.proc: Any = None

    def __enter__(self) -> Any:
        launch = argparse.Namespace(
            bin=self.args.bin,
            model=self.args.model,
            ngl=self.args.ngl,
            ncmoe=self.args.ncmoe,
            ctx=self.args.ctx,
            parallel=self.args.parallel,
            flash_attn=self.args.flash_attn,
            cache_type_k=self.args.cache_type_k,
            cache_type_v=self.args.cache_type_v,
            batch=self.args.batch,
            ubatch=self.args.ubatch,
            threads=self.args.threads,
            port=self.args.port,
            label=self.args.label,
            statetree_lease_ms=self.args.lease_ms,
            statetree_max_state_bytes=0,
            extra_server_args=(
                f"--statetree-max-snapshot-bytes {self.args.snapshot_budget_bytes}"
            ),
            source_oneapi=self.args.source_oneapi,
        )
        self.proc = bench.launch_server(launch, self.log_path)
        bench.wait_healthy(self.args.port, self.proc, self.args.startup_timeout)
        bench.erase_current_slots(
            self.args.port, list(range(self.args.parallel)), self.args.request_timeout
        )
        return self.proc

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.proc is None:
            return
        try:
            bench.erase_current_slots(
                self.args.port, list(range(self.args.parallel)), self.args.request_timeout
            )
        except Exception:
            pass
        bench.stop_process(self.proc)
        log_file = getattr(self.proc, "_turbo_log_file", None)
        if log_file is not None:
            log_file.close()


def snapshot_request(
    args: argparse.Namespace,
    node_id: int,
    *,
    state_id: int | None = None,
    fork_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if state_id is not None:
        payload["state_id"] = state_id
    if fork_id is not None:
        payload["fork_id"] = fork_id
    return bench.timed_json(
        "POST",
        f"http://127.0.0.1:{args.port}/nodes/{node_id}?action=snapshot",
        payload,
        timeout=args.request_timeout,
    )


def materialize_request(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    id_slot: int,
) -> dict[str, Any]:
    return bench.timed_json(
        "POST",
        f"http://127.0.0.1:{args.port}/snapshots/{snapshot['snapshot_id']}?action=materialize",
        {"digest": snapshot["digest"], "id_slot": id_slot},
        timeout=args.request_timeout,
    )


def continuation(
    args: argparse.Namespace,
    prompt: list[int],
    node_id: int,
) -> dict[str, Any]:
    return bench.timed_json(
        "POST",
        f"http://127.0.0.1:{args.port}/completion",
        {
            "prompt": prompt,
            "node_id": node_id,
            "n_predict": args.predict_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "seed": args.seed,
            "cache_prompt": True,
            "ignore_eos": True,
            "stop": [],
            "stream": False,
            "return_tokens": True,
        },
        timeout=args.request_timeout,
    )


def require_snapshot(body: Any, label: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise RuntimeError(f"{label} response is not an object")
    if type(body.get("snapshot_id")) is not int or body["snapshot_id"] < 0:
        raise RuntimeError(f"{label} did not return a snapshot identity")
    digest = body.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError(f"{label} did not return a SHA-256 content digest")
    payload_bytes = body.get("payload_bytes")
    if type(payload_bytes) is not int or payload_bytes <= 0:
        raise RuntimeError(f"{label} did not return positive payload bytes")
    if payload_bytes != body.get("state_bytes", -1) + body.get("token_bytes", -1):
        raise RuntimeError(f"{label} payload accounting is not exact")
    return body


def run_gate(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    with ManagedServer(args, out_dir / f"{args.label}.server.log") as proc:
        props = bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/props", timeout=args.request_timeout
        )
        contract = props.get("statetree", {}) if isinstance(props, dict) else {}
        if not contract.get("snapshot_enabled"):
            raise RuntimeError("server did not advertise enabled immutable snapshots")
        if contract.get("snapshot_storage") != "digest_deduplicated_immutable_payloads":
            raise RuntimeError("server did not advertise digest-deduplicated storage")

        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "snapshot-gate-prefix")
        suffix = bench.exact_tokens(args.port, args.suffix_tokens, "snapshot-gate-suffix")
        seed = bench.completion(args.port, prefix, 0, 0, args.seed, args.request_timeout)
        fork = bench.fork_slot(args.port, 0, [1], args.request_timeout)
        fork_body = fork["response"]
        state_id = fork_body.get("state_id")
        fork_id = fork_body.get("fork_id")
        nodes = {
            row.get("id_slot"): row
            for row in fork_body.get("nodes", [])
            if isinstance(row, dict)
        }
        if type(state_id) is not int or type(fork_id) is not int or set(nodes) != {0, 1}:
            raise RuntimeError("fork did not establish the expected two-head StateTree family")

        first_timed = snapshot_request(
            args, nodes[0]["node_id"], state_id=state_id, fork_id=fork_id
        )
        second_timed = snapshot_request(
            args, nodes[1]["node_id"], state_id=state_id, fork_id=fork_id
        )
        first = require_snapshot(first_timed["response"], "first capture")
        second = require_snapshot(second_timed["response"], "second capture")
        if first["digest"] != second["digest"]:
            raise RuntimeError("identical physical fork heads did not converge to one digest")
        if first.get("deduplicated") is not False or second.get("deduplicated") is not True:
            raise RuntimeError("identical captures did not exercise content-pool deduplication")

        listing = bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/snapshots", timeout=args.request_timeout
        )
        if listing.get("snapshot_bytes") != first["payload_bytes"]:
            raise RuntimeError("snapshot byte accounting counted duplicate content")
        if listing.get("snapshot_content_count") != 1:
            raise RuntimeError("identical snapshots did not share one content object")

        commit = bench.timed_json(
            "POST",
            f"http://127.0.0.1:{args.port}/nodes/{nodes[1]['node_id']}?action=commit",
            {},
            timeout=args.request_timeout,
        )
        changed_completion = continuation(args, prefix + suffix, nodes[1]["node_id"])
        changed_timed = snapshot_request(args, nodes[1]["node_id"])
        changed = require_snapshot(changed_timed["response"], "changed capture")
        if changed["digest"] == first["digest"]:
            raise RuntimeError("source mutation did not change the captured content digest")

        restore_a = materialize_request(args, first, 2)
        restore_b = materialize_request(args, second, 3)
        restored_a = restore_a["response"]
        restored_b = restore_b["response"]
        if restored_a.get("parent_node_id") != first["source_node_id"]:
            raise RuntimeError("materialized node lost its snapshot provenance parent")
        if restored_b.get("parent_node_id") != second["source_node_id"]:
            raise RuntimeError("second materialized node lost its provenance parent")

        roundtrip_timed = snapshot_request(args, restored_a["node_id"])
        roundtrip = require_snapshot(roundtrip_timed["response"], "round-trip capture")
        if roundtrip["digest"] != first["digest"] or roundtrip.get("deduplicated") is not True:
            raise RuntimeError("materialized content did not round-trip to its original object")

        continued_a = continuation(args, prefix + suffix, restored_a["node_id"])
        continued_b = continuation(args, prefix + suffix, restored_b["node_id"])
        body_a = continued_a["response"]
        body_b = continued_b["response"]
        if body_a.get("tokens") != body_b.get("tokens") or body_a.get("content") != body_b.get("content"):
            raise RuntimeError("two materializations produced different deterministic continuations")
        continued_snapshot_a = require_snapshot(
            snapshot_request(args, restored_a["node_id"])["response"], "continued A capture"
        )
        continued_snapshot_b = require_snapshot(
            snapshot_request(args, restored_b["node_id"])["response"], "continued B capture"
        )
        continued_byte_convergence = (
            continued_snapshot_a["digest"] == continued_snapshot_b["digest"]
        )

        mismatch_status = None
        try:
            bench.timed_json(
                "POST",
                f"http://127.0.0.1:{args.port}/snapshots/{first['snapshot_id']}?action=materialize",
                {"digest": "sha256:" + "0" * 64, "id_slot": 0},
                timeout=args.request_timeout,
            )
        except bench.HttpStatusError as exc:
            mismatch_status = exc.status
        if mismatch_status != 503:
            raise RuntimeError(f"digest mismatch returned {mismatch_status!r}, expected 503")

        erase = bench.timed_json(
            "POST",
            f"http://127.0.0.1:{args.port}/snapshots/{first['snapshot_id']}?action=erase",
            {"digest": first["digest"]},
            timeout=args.request_timeout,
        )
        if erase["response"].get("content_reclaimed") is not False:
            raise RuntimeError("erasing one deduplicated handle reclaimed shared content")
        stale_status = None
        try:
            bench.timed_json(
                "POST",
                f"http://127.0.0.1:{args.port}/snapshots/{first['snapshot_id']}?action=materialize",
                {"id_slot": 0},
                timeout=args.request_timeout,
            )
        except bench.HttpStatusError as exc:
            stale_status = exc.status
        if stale_status != 503:
            raise RuntimeError(f"erased snapshot handle returned {stale_status!r}, expected 503")

        final_listing = bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/snapshots", timeout=args.request_timeout
        )
        metrics = bench.parse_prometheus(
            bench.http_text(f"http://127.0.0.1:{args.port}/metrics", args.request_timeout)
        )
        return {
            "contract": contract,
            "state_id": state_id,
            "fork_id": fork_id,
            "seed": seed,
            "fork": fork,
            "captures": {
                "first": {"client_ms": first_timed["client_ms"], "response": first},
                "deduplicated": {"client_ms": second_timed["client_ms"], "response": second},
                "changed": {"client_ms": changed_timed["client_ms"], "response": changed},
                "roundtrip": {"client_ms": roundtrip_timed["client_ms"], "response": roundtrip},
                "continued_a": continued_snapshot_a,
                "continued_b": continued_snapshot_b,
                "continued_byte_convergence": continued_byte_convergence,
            },
            "commit": commit,
            "changed_completion": changed_completion,
            "materialize": {"first": restore_a, "second": restore_b},
            "continuations": {"first": continued_a, "second": continued_b},
            "digest_mismatch_status": mismatch_status,
            "erase": erase,
            "stale_handle_status": stale_status,
            "final_snapshots": final_listing,
            "snapshot_metrics": {
                key: value
                for key, value in metrics.items()
                if key.startswith("statetree_snapshot")
            },
            "process": bench.read_process_memory(proc.pid),
            "gpu": bench.read_drm_memory(proc.pid),
            "passed": True,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate immutable StateTree snapshot semantics.")
    parser.add_argument("--bin", default=bench.DEFAULT_BIN)
    parser.add_argument("--model", default=bench.DEFAULT_MODEL)
    parser.add_argument("--label", default="statetree-snapshot")
    parser.add_argument("--commit", default="dirty")
    parser.add_argument("--out-dir", default="/tmp/turbo-statetree-snapshot-gate")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--ctx", type=bench.positive_int, default=32768)
    parser.add_argument("--parallel", type=bench.positive_int, default=4)
    parser.add_argument("--prefix-tokens", type=bench.positive_int, default=1024)
    parser.add_argument("--suffix-tokens", type=bench.positive_int, default=32)
    parser.add_argument("--predict-tokens", type=bench.positive_int, default=8)
    parser.add_argument("--snapshot-budget-bytes", type=bench.positive_int, default=536870912)
    parser.add_argument("--lease-ms", type=bench.positive_int, default=30000)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--batch", type=bench.positive_int, default=2048)
    parser.add_argument("--ubatch", type=bench.positive_int, default=512)
    parser.add_argument("--threads", type=bench.positive_int, default=16)
    parser.add_argument("--ngl", type=bench.non_negative_int, default=0)
    parser.add_argument("--ncmoe", type=bench.non_negative_int, default=0)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="auto")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--source-oneapi", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.port == 8093:
        raise SystemExit("refusing production port 8093; use an isolated managed server")
    if args.parallel < 4:
        raise SystemExit("snapshot gate requires at least four slots")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{args.label}.result.json"
    result: dict[str, Any] = {
        "kind": "turbo-statetree-snapshot-gate",
        "schema_version": 1,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": args.label,
        "config": {key: value for key, value in vars(args).items() if key not in {"bin", "model"}},
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
        result["gate"] = run_gate(args, out_dir)
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
