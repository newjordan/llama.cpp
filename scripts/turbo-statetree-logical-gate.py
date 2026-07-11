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
            statetree_max_state_bytes=self.args.max_state_bytes,
            extra_server_args="",
            source_oneapi=self.args.source_oneapi,
        )
        self.proc = bench.launch_server(launch, self.log_path)
        bench.wait_healthy(self.args.port, self.proc, self.args.startup_timeout)
        bench.erase_current_slots(
            self.args.port,
            list(range(self.args.parallel)),
            self.args.request_timeout,
        )
        return self.proc

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.proc is None:
            return
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


def completion_payload(
    prompt: list[int],
    *,
    id_slot: int | None = None,
    state_id: int | None = None,
    fork_id: int | None = None,
    node_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": 0,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
        "ignore_eos": True,
        "stop": [],
        "stream": False,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot
    if state_id is not None:
        payload["state_id"] = state_id
    if fork_id is not None:
        payload["fork_id"] = fork_id
    if node_id is not None:
        payload["node_id"] = node_id
    return payload


def timed_completion(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    timed = bench.timed_json(
        "POST",
        f"http://127.0.0.1:{args.port}/completion",
        payload,
        timeout=args.request_timeout,
    )
    body = timed["response"]
    if not isinstance(body, dict) or not isinstance(body.get("timings"), dict):
        raise RuntimeError("completion response is missing timings")
    return timed


def timed_states(args: argparse.Namespace, journal_after: int | None = None) -> dict[str, Any]:
    suffix = "" if journal_after is None else f"?journal_after={journal_after}"
    timed = bench.timed_json(
        "GET",
        f"http://127.0.0.1:{args.port}/states{suffix}",
        timeout=args.request_timeout,
    )
    body = timed["response"]
    if not isinstance(body, dict):
        raise RuntimeError("states response is not an object")
    return timed


def validate_ring(body: dict[str, Any], expected_capacity: int) -> dict[str, int]:
    journal = body.get("journal")
    oldest = body.get("journal_oldest_sequence")
    next_sequence = body.get("journal_next_sequence")
    capacity = body.get("journal_capacity")
    if type(capacity) is not int or capacity != expected_capacity:
        raise RuntimeError(f"unexpected journal capacity: {capacity!r}")
    if not isinstance(journal, list) or len(journal) != capacity:
        raise RuntimeError("journal did not fill to its advertised capacity")
    if type(oldest) is not int or type(next_sequence) is not int:
        raise RuntimeError("journal boundaries are not integers")
    sequences = [entry.get("sequence") for entry in journal if isinstance(entry, dict)]
    if sequences != list(range(oldest, next_sequence)):
        raise RuntimeError("journal ring is not a contiguous retained sequence")
    if next_sequence - oldest != capacity or oldest <= 1:
        raise RuntimeError("journal ring did not report detectable truncation")
    return {"oldest_sequence": oldest, "next_sequence": next_sequence}


def wait_for_expiry(args: argparse.Namespace, state_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(10.0, args.lease_ms / 1000.0 + 5.0)
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = timed_states(args)["response"]
        states = last.get("states")
        if isinstance(states, list) and all(
            not isinstance(state, dict) or state.get("state_id") != state_id
            for state in states
        ):
            return last
        time.sleep(args.poll_ms / 1000.0)
    raise RuntimeError(f"logical state {state_id} did not expire: {last!r}")


def run_gate(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    with ManagedServer(args, out_dir / f"{args.label}.server.log") as proc:
        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "logical-gate-prefix")
        suffix = bench.exact_tokens(args.port, 8, "logical-gate-winner")

        seed = bench.completion(
            args.port,
            prefix,
            0,
            0,
            args.seed,
            args.request_timeout,
        )
        fork = bench.fork_slot(args.port, 0, [1, 2], args.request_timeout)
        state_id = fork["response"].get("state_id")
        fork_id = fork.get("fork_id")
        if type(state_id) is not int or state_id < 0 or type(fork_id) is not int:
            raise RuntimeError("fork did not allocate logical and generation identities")
        nodes = fork["response"].get("nodes")
        has_nodes = isinstance(nodes, list) and bool(nodes)
        if args.require_nodes and not has_nodes:
            raise RuntimeError("fork did not allocate immutable branch nodes")
        winner_node_id: int | None = None
        if has_nodes:
            nodes_by_slot = {
                node.get("id_slot"): node
                for node in nodes
                if isinstance(node, dict)
            }
            if set(nodes_by_slot) != {0, 1, 2}:
                raise RuntimeError("fork node set does not match family slots")
            node_ids = [node.get("node_id") for node in nodes_by_slot.values()]
            if any(type(node_id) is not int or node_id < 0 for node_id in node_ids):
                raise RuntimeError("fork returned an invalid branch node ID")
            if len(set(node_ids)) != 3:
                raise RuntimeError("fork branch node IDs are not unique")
            if {node.get("parent_node_id") for node in nodes_by_slot.values()} != {-1}:
                raise RuntimeError("first-generation branch nodes have an unexpected parent")
            winner_node_id = nodes_by_slot[1]["node_id"]

        open_state = timed_states(args)
        states = open_state["response"].get("states")
        if not isinstance(states, list) or len(states) != 1:
            raise RuntimeError("open logical family was not visible as exactly one state")
        logical = states[0]
        if not isinstance(logical, dict) or logical.get("members") != [0, 1, 2]:
            raise RuntimeError("open logical family members are incorrect")
        if has_nodes and logical.get("heads") != nodes:
            raise RuntimeError("live logical heads do not match the forked nodes")

        ambiguity_status = None
        try:
            timed_completion(
                args,
                completion_payload(prefix, state_id=state_id, fork_id=fork_id),
            )
        except bench.HttpStatusError as exc:
            ambiguity_status = exc.status
        if ambiguity_status != 400:
            raise RuntimeError(f"ambiguous logical lookup returned {ambiguity_status!r}, expected 400")

        winner_prompt = prefix + suffix
        winner = timed_completion(
            args,
            completion_payload(
                winner_prompt,
                id_slot=None if has_nodes else 1,
                state_id=None if has_nodes else state_id,
                fork_id=None if has_nodes else fork_id,
                node_id=winner_node_id,
            ),
        )
        if winner["response"].get("id_slot") != 1:
            raise RuntimeError("physical winner request used the wrong slot")
        node_mutations: dict[str, Any] = {"required": args.require_node_mutations}
        if args.require_node_mutations:
            open_renew = bench.timed_json(
                "POST",
                f"http://127.0.0.1:{args.port}/nodes/{winner_node_id}?action=renew",
                {"state_id": state_id},
                timeout=args.request_timeout,
            )
            if open_renew["response"].get("id_slot") != 1:
                raise RuntimeError("node-addressed renew resolved the wrong branch")
            commit_timed = bench.timed_json(
                "POST",
                f"http://127.0.0.1:{args.port}/nodes/{winner_node_id}?action=commit",
                {"state_id": state_id},
                timeout=args.request_timeout,
            )
            commit_body = commit_timed["response"]
            if not isinstance(commit_body, dict) or commit_body.get("node_id") != winner_node_id:
                raise RuntimeError("node-addressed commit did not preserve the winner node")
            commit = {
                "client_ms": commit_timed["client_ms"],
                "server_ms": bench.required_nonnegative_number(
                    commit_body.get("timings", {}), "commit_ms", "node commit server timing"
                ),
                "response": commit_body,
            }
            node_mutations["open_renew"] = open_renew
            node_mutations["commit"] = commit
        else:
            commit = bench.commit_slot(args.port, 1, fork_id, args.request_timeout)
        if commit["response"].get("state_id") != state_id:
            raise RuntimeError("winner migration changed the logical identity")

        physical_ms: list[float] = []
        logical_ms: list[float] = []
        node_ms: list[float] = []
        for repeat in range(args.lookup_repeats):
            order = ["physical", "logical"]
            if has_nodes:
                order.append("node")
            if repeat % 2:
                order.reverse()
            for mode in order:
                payload = completion_payload(
                    winner_prompt,
                    id_slot=1 if mode == "physical" else None,
                    state_id=state_id if mode == "logical" else None,
                    fork_id=fork_id if mode != "node" else None,
                    node_id=winner_node_id if mode == "node" else None,
                )
                timed = timed_completion(args, payload)
                if timed["response"].get("id_slot") != 1:
                    raise RuntimeError(f"{mode} continuation did not resolve the committed winner")
                if int(timed["response"]["timings"].get("cache_n", 0)) <= 0:
                    raise RuntimeError(f"{mode} continuation did not reuse cached state")
                if mode == "logical":
                    logical_ms.append(timed["client_ms"])
                elif mode == "node":
                    node_ms.append(timed["client_ms"])
                else:
                    physical_ms.append(timed["client_ms"])

        physical = bench.summarize_values(physical_ms)
        logical_lookup = bench.summarize_values(logical_ms)
        lookup_limit_ms = max(float(physical["p50"]) * 1.25, float(physical["p50"]) + 1.0)
        if float(logical_lookup["p50"]) > lookup_limit_ms:
            raise RuntimeError(
                f"logical lookup p50 {logical_lookup['p50']:.3f} ms exceeds {lookup_limit_ms:.3f} ms"
            )
        node_lookup = bench.summarize_values(node_ms)
        if node_ms and float(node_lookup["p50"]) > lookup_limit_ms:
            raise RuntimeError(
                f"node lookup p50 {node_lookup['p50']:.3f} ms exceeds {lookup_limit_ms:.3f} ms"
            )

        renew_client_ms: list[float] = []
        renew_server_ms: list[float] = []
        for _ in range(args.journal_events):
            renew_url = (
                f"http://127.0.0.1:{args.port}/nodes/{winner_node_id}?action=renew"
                if args.require_node_mutations
                else f"http://127.0.0.1:{args.port}/slots/1?action=renew"
            )
            timed = bench.timed_json(
                "POST",
                renew_url,
                {} if args.require_node_mutations else {"fork_id": fork_id},
                timeout=args.request_timeout,
            )
            body = timed["response"]
            if not isinstance(body, dict) or body.get("state_id") != state_id:
                raise RuntimeError("renew did not preserve the logical identity")
            timing = body.get("timings")
            if not isinstance(timing, dict) or not isinstance(timing.get("renew_ms"), (int, float)):
                raise RuntimeError("renew response is missing server timing")
            renew_client_ms.append(timed["client_ms"])
            renew_server_ms.append(float(timing["renew_ms"]))

        full_reads = [timed_states(args) for _ in range(args.read_repeats)]
        ring = validate_ring(full_reads[-1]["response"], args.journal_capacity)
        incremental_reads = [
            timed_states(args, ring["next_sequence"] - 5)
            for _ in range(args.read_repeats)
        ]
        for timed in incremental_reads:
            journal = timed["response"].get("journal")
            if not isinstance(journal, list) or len(journal) != 4:
                raise RuntimeError("incremental journal read returned the wrong suffix")

        read_full_ms = bench.summarize_values([float(item["client_ms"]) for item in full_reads])
        read_incremental_ms = bench.summarize_values(
            [float(item["client_ms"]) for item in incremental_reads]
        )
        if float(read_full_ms["p95"]) > args.max_full_read_p95_ms:
            raise RuntimeError(
                f"full journal read p95 {read_full_ms['p95']:.3f} ms exceeds "
                f"{args.max_full_read_p95_ms:.3f} ms"
            )

        if args.require_node_refork:
            refork_timed = bench.timed_json(
                "POST",
                f"http://127.0.0.1:{args.port}/nodes/{winner_node_id}?action=fork",
                {"state_id": state_id, "destinations": [0, 2]},
                timeout=args.request_timeout,
            )
            refork_body = refork_timed["response"]
            if not isinstance(refork_body, dict) or refork_body.get("id_slot") != 1:
                raise RuntimeError("node-addressed re-fork resolved the wrong source")
            refork_timings = refork_body.get("timings")
            if not isinstance(refork_timings, dict):
                raise RuntimeError("node-addressed re-fork response is missing timings")
            refork = {
                "client_ms": refork_timed["client_ms"],
                "server_ms": bench.required_nonnegative_number(
                    refork_timings, "fork_ms", "node re-fork server timing"
                ),
                "fork_id": refork_body.get("fork_id"),
                "n_tokens": bench.required_nonnegative_int(
                    refork_body, "n_tokens", "node re-fork n_tokens"
                ),
                "response": refork_body,
            }
            node_mutations["refork"] = refork
        else:
            refork = bench.fork_slot(args.port, 1, [0, 2], args.request_timeout, fork_id=fork_id)
        refork_id = refork.get("fork_id")
        if type(refork_id) is not int or refork_id == fork_id:
            raise RuntimeError("refork did not advance the generation")
        if refork["response"].get("state_id") != state_id:
            raise RuntimeError("refork changed the logical identity")
        refork_nodes = refork["response"].get("nodes")
        if has_nodes:
            if not isinstance(refork_nodes, list) or len(refork_nodes) != 3:
                raise RuntimeError("refork did not return three branch nodes")
            if {node.get("parent_node_id") for node in refork_nodes} != {winner_node_id}:
                raise RuntimeError("refork branch nodes do not descend from the committed winner")
            if {node.get("node_id") for node in refork_nodes} & {
                node.get("node_id") for node in nodes
            }:
                raise RuntimeError("refork recycled a branch node ID")
            stale_node_status = None
            try:
                timed_completion(args, completion_payload(winner_prompt, node_id=winner_node_id))
            except bench.HttpStatusError as exc:
                stale_node_status = exc.status
            if stale_node_status != 503:
                raise RuntimeError(f"stale parent node returned {stale_node_status!r}, expected 503")
            if args.require_node_mutations:
                refork_nodes_by_slot = {
                    node.get("id_slot"): node
                    for node in refork_nodes
                    if isinstance(node, dict)
                }
                erased_node_id = refork_nodes_by_slot[0].get("node_id")
                erased = bench.timed_json(
                    "POST",
                    f"http://127.0.0.1:{args.port}/nodes/{erased_node_id}?action=erase",
                    {"state_id": state_id, "fork_id": refork_id},
                    timeout=args.request_timeout,
                )
                erased_body = erased["response"]
                if not isinstance(erased_body, dict) or erased_body.get("node_id") != erased_node_id:
                    raise RuntimeError("node-addressed erase did not report the destroyed node")
                stale_erase_status = None
                try:
                    bench.timed_json(
                        "POST",
                        f"http://127.0.0.1:{args.port}/nodes/{erased_node_id}?action=erase",
                        {},
                        timeout=args.request_timeout,
                    )
                except bench.HttpStatusError as exc:
                    stale_erase_status = exc.status
                if stale_erase_status != 503:
                    raise RuntimeError(
                        f"stale erased node returned {stale_erase_status!r}, expected 503"
                    )
                node_mutations["erase"] = erased
                node_mutations["stale_erase_status"] = stale_erase_status

        expired = wait_for_expiry(args, state_id)
        journal = expired.get("journal")
        if not isinstance(journal, list) or not journal or journal[-1].get("event") != "expire":
            raise RuntimeError("expiry was not retained as the terminal journal event")
        if journal[-1].get("state_id") != state_id or journal[-1].get("fork_id") != refork_id:
            raise RuntimeError("expiry tombstone does not identify the destroyed generation")

        unavailable_status = None
        try:
            timed_completion(
                args,
                completion_payload(winner_prompt, state_id=state_id, fork_id=refork_id),
            )
        except bench.HttpStatusError as exc:
            unavailable_status = exc.status
        if unavailable_status != 503:
            raise RuntimeError(f"expired logical lookup returned {unavailable_status!r}, expected 503")

        final_slots = bench.http_json(
            "GET",
            f"http://127.0.0.1:{args.port}/slots",
            timeout=args.request_timeout,
        )
        if not isinstance(final_slots, list) or any(row.get("is_reserved") for row in final_slots):
            raise RuntimeError("logical gate cleanup left a reserved slot")

        return {
            "state_id": state_id,
            "fork_id": fork_id,
            "refork_id": refork_id,
            "ambiguity_status": ambiguity_status,
            "expired_lookup_status": unavailable_status,
            "seed": seed,
            "fork": fork,
            "commit": commit,
            "lookup": {
                "physical_client_ms": physical,
                "logical_client_ms": logical_lookup,
                "node_client_ms": node_lookup,
                "logical_p50_limit_ms": lookup_limit_ms,
            },
            "nodes": {
                "supported": has_nodes,
                "first_generation": nodes if has_nodes else [],
                "second_generation": refork_nodes if has_nodes else [],
            },
            "node_mutations": node_mutations,
            "journal": {
                **ring,
                "renew_events": args.journal_events,
                "renew_client_ms": bench.summarize_values(renew_client_ms),
                "renew_server_ms": bench.summarize_values(renew_server_ms),
                "full_read_client_ms": read_full_ms,
                "incremental_read_client_ms": read_incremental_ms,
            },
            "process": bench.read_process_memory(proc.pid),
            "gpu": bench.read_drm_memory(proc.pid),
            "passed": True,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate StateTree logical lookup and bounded journal behavior.")
    parser.add_argument("--bin", default=bench.DEFAULT_BIN)
    parser.add_argument("--model", default=bench.DEFAULT_MODEL)
    parser.add_argument("--label", default="statetree-logical")
    parser.add_argument("--commit", default="dirty")
    parser.add_argument("--out-dir", default="/tmp/turbo-statetree-logical-gate")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--ctx", type=bench.positive_int, default=32768)
    parser.add_argument("--parallel", type=bench.positive_int, default=4)
    parser.add_argument("--prefix-tokens", type=bench.positive_int, default=1024)
    parser.add_argument("--lookup-repeats", type=bench.positive_int, default=20)
    parser.add_argument("--journal-events", type=bench.positive_int, default=1100)
    parser.add_argument("--journal-capacity", type=bench.positive_int, default=1024)
    parser.add_argument("--read-repeats", type=bench.positive_int, default=10)
    parser.add_argument("--max-full-read-p95-ms", type=float, default=50.0)
    parser.add_argument("--require-nodes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-node-mutations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-node-refork", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lease-ms", type=bench.positive_int, default=1000)
    parser.add_argument("--max-state-bytes", type=bench.non_negative_int, default=0)
    parser.add_argument("--poll-ms", type=float, default=5.0)
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
    if args.parallel < 3:
        raise SystemExit("logical gate requires at least three slots")
    if args.require_node_mutations and not args.require_nodes:
        raise SystemExit("node mutation gate requires --require-nodes")
    if args.require_node_refork and not args.require_node_mutations:
        raise SystemExit("node re-fork gate requires --require-node-mutations")
    if args.journal_events <= args.journal_capacity:
        raise SystemExit("journal events must exceed capacity to prove truncation")
    if args.poll_ms <= 0 or args.max_full_read_p95_ms <= 0:
        raise SystemExit("poll and latency thresholds must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{args.label}.result.json"
    result: dict[str, Any] = {
        "kind": "turbo-statetree-logical-gate",
        "schema_version": 1,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
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
