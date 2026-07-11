#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, args: argparse.Namespace, store_path: Path, log_path: Path):
        self.args = args
        self.store_path = store_path
        self.log_path = log_path
        self.proc: Any = None

    def __enter__(self) -> Any:
        extra = shlex.join([
            "--statetree-max-snapshot-bytes", str(self.args.snapshot_budget_bytes),
            "--statetree-snapshot-store", str(self.store_path),
            "--statetree-snapshot-compat-id", self.args.compat_id,
            "--statetree-max-snapshot-disk-bytes", str(self.args.disk_budget_bytes),
            "--statetree-max-snapshot-load-bytes", str(self.args.load_budget_bytes),
            "--statetree-max-snapshot-manifest-bytes", str(self.args.manifest_budget_bytes),
        ])
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
            extra_server_args=extra,
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


def request(args: argparse.Namespace, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return bench.timed_json(
        method,
        f"http://127.0.0.1:{args.port}{path}",
        payload,
        timeout=args.request_timeout,
    )


def continuation(args: argparse.Namespace, prompt: list[int], node_id: int) -> dict[str, Any]:
    return request(args, "POST", "/completion", {
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
    })


def first_phase(args: argparse.Namespace, store_path: Path, out_dir: Path) -> dict[str, Any]:
    with ManagedServer(args, store_path, out_dir / f"{args.label}-spill.server.log") as proc:
        props = bench.http_json("GET", f"http://127.0.0.1:{args.port}/props", timeout=args.request_timeout)
        contract = props.get("statetree", {}) if isinstance(props, dict) else {}
        if not contract.get("durable_snapshot_enabled"):
            raise RuntimeError("server did not advertise durable snapshot content")
        if contract.get("durable_snapshot_compatibility_id") != args.compat_id:
            raise RuntimeError("server compatibility fence does not match the gate")
        if contract.get("durable_snapshot_io") != "single_ordered_worker_two_phase":
            raise RuntimeError("server did not advertise the ordered two-phase durable I/O path")
        if contract.get("snapshot_manifest_format") != "turbo-statetree-manifest-v1":
            raise RuntimeError("server did not advertise the checksummed ownership manifest")
        if contract.get("max_snapshot_manifest_bytes") != args.manifest_budget_bytes:
            raise RuntimeError("server manifest budget does not match the gate")
        if contract.get("snapshot_retention_classes") != ["pinned", "cache"]:
            raise RuntimeError("server did not advertise the expected retention classes")
        if args.exercise_cache_pressure and contract.get("durable_cache_reclamation") != \
                "managed_cache_only_oldest_revision":
            raise RuntimeError("server did not advertise managed cache reconciliation")
        head_mode = args.exercise_logical_head or args.exercise_atomic_publish_advance
        if head_mode:
            if contract.get("durable_logical_heads") != "generation_digest_cas":
                raise RuntimeError("server did not advertise durable logical-head CAS")
            if contract.get("durable_logical_head_delete") != \
                    "retry_safe_retired_name_tombstone":
                raise RuntimeError("server did not advertise retry-safe retired head names")
        if args.exercise_atomic_publish_advance:
            if contract.get("durable_publish_advance") != \
                    "intent_object_atomic_owner_head_commit":
                raise RuntimeError("server did not advertise atomic publish-and-advance")
            if contract.get("snapshot_manifest_checkpoint_schema") != "atomic-publish-advance-v3":
                raise RuntimeError("server did not advertise the atomic transaction checkpoint schema")
        elif args.exercise_logical_head and contract.get("snapshot_manifest_checkpoint_schema") not in {
                "logical-heads-v2", "atomic-publish-advance-v3"}:
            raise RuntimeError("server did not advertise a logical-head checkpoint schema")
        before = bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/snapshot-contents", timeout=args.request_timeout
        )
        if before.get("contents"):
            raise RuntimeError("cold gate requires an initially empty compatibility namespace")

        prefix = bench.exact_tokens(args.port, args.prefix_tokens, "cold-gate-prefix")
        suffix = bench.exact_tokens(args.port, args.suffix_tokens, "cold-gate-suffix")
        seed = bench.completion(args.port, prefix, 0, 0, args.seed, args.request_timeout)
        fork = bench.fork_slot(args.port, 0, [1], args.request_timeout)
        nodes = {
            row.get("id_slot"): row
            for row in fork["response"].get("nodes", [])
            if isinstance(row, dict)
        }
        if set(nodes) != {0, 1}:
            raise RuntimeError("fork did not return the expected node set")
        captured = request(
            args, "POST", f"/nodes/{nodes[0]['node_id']}?action=snapshot", {}
        )
        snapshot = captured["response"]
        digest = snapshot.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise RuntimeError("capture did not return a content digest")
        def spill_request() -> dict[str, Any]:
            return request(
                args,
                "POST",
                f"/snapshots/{snapshot['snapshot_id']}?action=publish",
                {
                    "digest": digest,
                    "owner": args.owner,
                    "retention_class": args.retention_class,
                },
            )

        probe_samples_ms: list[float] = []
        pending_samples: list[dict[str, Any]] = []
        scheduler_probe: dict[str, Any] | None = None
        scheduler_probe_finished_before_spills = False
        with ThreadPoolExecutor(max_workers=2) as executor:
            spill_futures = [executor.submit(spill_request) for _ in range(2)]
            deadline = time.monotonic() + args.request_timeout
            while time.monotonic() < deadline and not all(future.done() for future in spill_futures):
                started = time.perf_counter()
                state_probe = bench.http_json(
                    "GET", f"http://127.0.0.1:{args.port}/states", timeout=args.request_timeout
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                if state_probe.get("durable_io_pending", 0) > 0:
                    probe_samples_ms.append(elapsed_ms)
                    pending_samples.append({
                        "pending": state_probe.get("durable_io_pending"),
                        "reserved_disk_bytes": state_probe.get("durable_io_reserved_disk_bytes"),
                    })
                    if scheduler_probe is None and args.parallel >= 3:
                        scheduler_probe = bench.completion(
                            args.port, suffix, 2, 1, args.seed, args.request_timeout
                        )
                        scheduler_probe_finished_before_spills = not all(
                            future.done() for future in spill_futures
                        )
                time.sleep(0.001)
            spill_results = [future.result() for future in spill_futures]
        if args.require_io_probes and not probe_samples_ms:
            raise RuntimeError("durable spill completed without an observable nonblocking state probe")
        if args.require_io_probes and scheduler_probe is None:
            raise RuntimeError("scheduler probe was not issued while durable spill work remained")
        if args.require_scheduler_overlap and not scheduler_probe_finished_before_spills:
            raise RuntimeError("scheduler probe did not complete while durable spill work remained")
        spilled = next(
            item for item in spill_results
            if item["response"].get("object_deduplicated") is False
        )
        duplicate = next(
            item for item in spill_results
            if item["response"].get("object_deduplicated") is True
        )
        if spilled["response"].get("ownership_deduplicated") is not False:
            raise RuntimeError("first managed publish did not commit a fresh owner")
        if duplicate["response"].get("ownership_deduplicated") is not True:
            raise RuntimeError("duplicate managed publish did not reuse the owner transaction")
        if spilled["response"].get("ref_count") != 1:
            raise RuntimeError("managed publish did not establish one durable owner reference")

        head_create = None
        head_compact = None
        if head_mode:
            head_create = request(
                args,
                "POST",
                f"/snapshot-heads/{args.head_name}?action=create",
                {"digest": digest},
            )
            if head_create["response"].get("generation") != 1 or \
                    head_create["response"].get("digest") != digest:
                raise RuntimeError("logical head create did not establish generation 1")
            head_compact = request(args, "POST", "/snapshot-manifest?action=compact", {})
            if head_compact["response"].get("records_after") != 1:
                raise RuntimeError("logical head was not compacted into one checkpoint")

        commit = request(
            args, "POST", f"/nodes/{nodes[1]['node_id']}?action=commit", {}
        )
        hot = request(
            args,
            "POST",
            f"/snapshots/{snapshot['snapshot_id']}?action=materialize",
            {"digest": digest, "id_slot": 0},
        )
        hot_continuation = continuation(args, prefix + suffix, hot["response"]["node_id"])
        listing = bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/snapshot-contents", timeout=args.request_timeout
        )
        if len(listing.get("contents", [])) != 1:
            raise RuntimeError("spill phase did not retain exactly one durable object")
        if len(listing.get("refs", [])) != 1 or listing["refs"][0].get("owner") != args.owner:
            raise RuntimeError("spill phase ownership manifest does not match the gate owner")
        return {
            "contract": contract,
            "prefix": prefix,
            "suffix": suffix,
            "seed": seed,
            "fork": fork,
            "capture": captured,
            "spill": spilled,
            "duplicate_spill": duplicate,
            "publish": spilled,
            "head_create": head_create,
            "head_compact": head_compact,
            "io_probe": {
                "state_probe_samples_ms": probe_samples_ms,
                "state_probe_min_ms": min(probe_samples_ms) if probe_samples_ms else None,
                "state_probe_max_ms": max(probe_samples_ms) if probe_samples_ms else None,
                "pending_samples": pending_samples,
                "scheduler_probe": scheduler_probe,
                "scheduler_probe_finished_before_spills": scheduler_probe_finished_before_spills,
            },
            "commit": commit,
            "hot_materialize": hot,
            "hot_continuation": hot_continuation,
            "listing": listing,
            "process": bench.read_process_memory(proc.pid),
            "gpu": bench.read_drm_memory(proc.pid),
        }


def restart_phase(
    args: argparse.Namespace,
    store_path: Path,
    out_dir: Path,
    phase_one: dict[str, Any],
) -> dict[str, Any]:
    namespaces = [path for path in store_path.iterdir() if path.is_dir()]
    if len(namespaces) != 1:
        raise RuntimeError("spill phase did not create exactly one compatibility namespace")
    crash_artifact = namespaces[0] / ".tmp-gate-interrupted"
    crash_artifact.write_bytes(b"incomplete")

    with ManagedServer(args, store_path, out_dir / f"{args.label}-restart.server.log") as proc:
        listing = bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/snapshot-contents", timeout=args.request_timeout
        )
        if listing.get("recovered_temp_files") != 1 or len(listing.get("contents", [])) != 1:
            raise RuntimeError("restart did not recover the temp artifact and rediscover one object")
        if len(listing.get("refs", [])) != 1 or listing["refs"][0].get("owner") != args.owner:
            raise RuntimeError("restart did not replay the durable owner reference")
        if bench.http_json(
            "GET", f"http://127.0.0.1:{args.port}/snapshots", timeout=args.request_timeout
        ).get("snapshots"):
            raise RuntimeError("process-local snapshot handles survived restart")

        snapshot = phase_one["capture"]["response"]
        digest = snapshot["digest"]
        digest_hex = digest.removeprefix("sha256:")
        head_mode = args.exercise_logical_head or args.exercise_atomic_publish_advance
        replayed_head = None
        stale_head_status = None
        if head_mode:
            replayed_head = request(args, "GET", "/snapshot-heads")
            heads = replayed_head["response"].get("heads", [])
            if len(heads) != 1 or heads[0].get("name") != args.head_name or \
                    heads[0].get("generation") != 1 or heads[0].get("digest") != digest:
                raise RuntimeError("restart did not replay the compacted logical head")
            try:
                request(
                    args,
                    "POST",
                    f"/snapshot-heads/{args.head_name}?action=materialize",
                    {"expected_generation": 2, "expected_digest": digest},
                )
            except bench.HttpStatusError as exc:
                stale_head_status = exc.status
            if stale_head_status != 503:
                raise RuntimeError("stale logical-head materialization did not fail with HTTP 503")
        erase_fence_status = None
        try:
            request(args, "POST", f"/snapshot-contents/{digest_hex}?action=erase", {})
        except bench.HttpStatusError as exc:
            erase_fence_status = exc.status
        if erase_fence_status != 503:
            raise RuntimeError(
                f"retained content erase returned {erase_fence_status!r}, expected 503"
            )
        cold_probe_samples_ms: list[float] = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            cold_path = f"/snapshot-contents/{digest_hex}?action=materialize"
            cold_payload = {"id_slot": 0}
            if head_mode:
                cold_path = f"/snapshot-heads/{args.head_name}?action=materialize"
                cold_payload = {
                    "expected_generation": 1,
                    "expected_digest": digest,
                    "id_slot": 0,
                }
            cold_future = executor.submit(
                request,
                args,
                "POST",
                cold_path,
                cold_payload,
            )
            deadline = time.monotonic() + args.request_timeout
            while time.monotonic() < deadline and not cold_future.done():
                started = time.perf_counter()
                state_probe = bench.http_json(
                    "GET", f"http://127.0.0.1:{args.port}/states", timeout=args.request_timeout
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                if state_probe.get("durable_io_pending", 0) > 0:
                    cold_probe_samples_ms.append(elapsed_ms)
                time.sleep(0.001)
            cold = cold_future.result()
        if args.require_io_probes and not cold_probe_samples_ms:
            raise RuntimeError("cold load completed without an observable nonblocking state probe")
        cold_body = cold["response"]
        if cold_body.get("cold") is not True or cold_body.get("snapshot_id") is not None:
            raise RuntimeError("restart materialization did not identify itself as durable content")
        if cold_body.get("parent_node_id") != -1 or cold_body.get("digest") != digest:
            raise RuntimeError("restart materialization returned incorrect provenance")
        if head_mode and (
                cold_body.get("action") != "materialize_head" or
                cold_body.get("head", {}).get("generation") != 1):
            raise RuntimeError("restart materialization did not preserve logical-head provenance")

        stale_handle_status = None
        try:
            request(args, "POST", f"/snapshots/{snapshot['snapshot_id']}?action=materialize", {"id_slot": 1})
        except bench.HttpStatusError as exc:
            stale_handle_status = exc.status
        if stale_handle_status != 503:
            raise RuntimeError(f"pre-restart snapshot handle returned {stale_handle_status!r}, expected 503")

        roundtrip = request(
            args, "POST", f"/nodes/{cold_body['node_id']}?action=snapshot", {}
        )
        if roundtrip["response"].get("digest") != digest:
            raise RuntimeError("cold materialization did not round-trip to the durable digest")
        cold_continuation = continuation(
            args,
            phase_one["prefix"] + phase_one["suffix"],
            cold_body["node_id"],
        )
        hot_body = phase_one["hot_continuation"]["response"]
        restarted_body = cold_continuation["response"]
        if hot_body.get("tokens") != restarted_body.get("tokens") or hot_body.get("content") != restarted_body.get("content"):
            raise RuntimeError("cold restart continuation differs from hot materialization")

        states = bench.http_json("GET", f"http://127.0.0.1:{args.port}/states", timeout=args.request_timeout)
        heads = [
            head
            for state in states.get("states", [])
            for head in state.get("heads", [])
            if head.get("node_id") == cold_body["node_id"]
        ]
        if len(heads) != 1 or heads[0].get("materialized_content_digest") != digest:
            raise RuntimeError("cold node did not expose durable content provenance")

        pressure_publish = None
        pressure_capture = None
        head_advance = None
        head_advance_retry = None
        head_delete = None
        head_delete_retry = None
        head_aba_create_status = None
        old_release = None
        old_erase = None
        final_digest = digest
        final_owner = args.owner
        if args.exercise_cache_pressure or head_mode:
            if args.retention_class != "cache":
                raise RuntimeError("replacement-content gate requires cache retention_class")
            pressure_capture = request(
                args, "POST", f"/nodes/{cold_body['node_id']}?action=snapshot", {}
            )
            final_digest = pressure_capture["response"]["digest"]
            if final_digest == digest:
                raise RuntimeError("post-continuation pressure snapshot did not change content digest")
            final_owner = args.owner + "/replacement"
            if args.exercise_atomic_publish_advance:
                pressure_publish = request(
                    args,
                    "POST",
                    f"/snapshots/{pressure_capture['response']['snapshot_id']}?action=publish-advance",
                    {
                        "digest": final_digest,
                        "owner": final_owner,
                        "retention_class": "pinned",
                        "head_name": args.head_name,
                        "expected_generation": 1,
                        "expected_digest": digest,
                    },
                )
            else:
                pressure_publish = request(
                    args,
                    "POST",
                    f"/snapshots/{pressure_capture['response']['snapshot_id']}?action=publish",
                    {
                        "digest": final_digest,
                        "owner": final_owner,
                        "retention_class": "pinned",
                    },
                )
            evicted = pressure_publish["response"].get("cache_evictions", [])
            if args.exercise_cache_pressure and [row.get("digest") for row in evicted] != [digest]:
                raise RuntimeError("pressure publish did not evict the prior cache digest exactly")
            if head_mode and evicted:
                raise RuntimeError("logical-head target publish unexpectedly evicted fenced content")
            pressure_listing = bench.http_json(
                "GET", f"http://127.0.0.1:{args.port}/snapshot-contents", timeout=args.request_timeout
            )
            expected_digests = [final_digest] if args.exercise_cache_pressure else sorted([digest, final_digest])
            if sorted(row.get("digest") for row in pressure_listing.get("contents", [])) != expected_digests:
                raise RuntimeError("pressure publish did not leave exactly the replacement object")

        if head_mode:
            if args.exercise_atomic_publish_advance:
                head_advance = pressure_publish
                atomic_head = head_advance["response"].get("head", {})
                if atomic_head.get("generation") != 2 or atomic_head.get("parent_digest") != digest or \
                        head_advance["response"].get("ref_revision") != atomic_head.get("revision"):
                    raise RuntimeError("atomic publish-and-advance did not commit owner and generation 2 together")
                head_advance_retry = request(
                    args,
                    "POST",
                    f"/snapshots/{pressure_capture['response']['snapshot_id']}?action=publish-advance",
                    {
                        "digest": final_digest,
                        "owner": final_owner,
                        "retention_class": "pinned",
                        "head_name": args.head_name,
                        "expected_generation": 1,
                        "expected_digest": digest,
                    },
                )
                if head_advance_retry["response"].get("transaction_deduplicated") is not True:
                    raise RuntimeError("atomic publish-and-advance retry was not deduplicated")
            else:
                head_advance = request(
                    args,
                    "POST",
                    f"/snapshot-heads/{args.head_name}?action=advance",
                    {"digest": final_digest, "expected_generation": 1, "expected_digest": digest},
                )
                if head_advance["response"].get("generation") != 2 or \
                        head_advance["response"].get("parent_digest") != digest:
                    raise RuntimeError("logical head advance did not establish generation 2 and its parent edge")
                head_advance_retry = request(
                    args,
                    "POST",
                    f"/snapshot-heads/{args.head_name}?action=advance",
                    {"digest": final_digest, "expected_generation": 1, "expected_digest": digest},
                )
                if head_advance_retry["response"].get("deduplicated") is not True:
                    raise RuntimeError("logical head advance retry was not deduplicated")
            old_release = request(
                args, "POST", f"/snapshot-contents/{digest_hex}?action=release", {"owner": args.owner}
            )
            old_erase = request(args, "POST", f"/snapshot-contents/{digest_hex}?action=erase", {})
            head_delete = request(
                args,
                "POST",
                f"/snapshot-heads/{args.head_name}?action=delete",
                {"expected_generation": 2, "expected_digest": final_digest},
            )
            head_delete_retry = request(
                args,
                "POST",
                f"/snapshot-heads/{args.head_name}?action=delete",
                {"expected_generation": 2, "expected_digest": final_digest},
            )
            if head_delete_retry["response"].get("deduplicated") is not True:
                raise RuntimeError("logical head delete retry was not deduplicated")
            try:
                request(
                    args,
                    "POST",
                    f"/snapshot-heads/{args.head_name}?action=create",
                    {"digest": final_digest},
                )
            except bench.HttpStatusError as exc:
                head_aba_create_status = exc.status
            if head_aba_create_status != 503:
                raise RuntimeError("retired logical head name did not fence generation-reset ABA")

        compact = request(args, "POST", "/snapshot-manifest?action=compact", {})
        if compact["response"].get("records_after") != 1:
            raise RuntimeError("manifest compaction did not publish one checkpoint record")
        released = request(
            args,
            "POST",
            f"/snapshot-contents/{final_digest.removeprefix('sha256:')}?action=release",
            {"owner": final_owner},
        )
        if released["response"].get("ref_count") != 0:
            raise RuntimeError("manifest release did not remove the final owner reference")
        erased = request(
            args,
            "POST",
            f"/snapshot-contents/{final_digest.removeprefix('sha256:')}?action=erase",
            {},
        )
        metrics = bench.parse_prometheus(
            bench.http_text(f"http://127.0.0.1:{args.port}/metrics", args.request_timeout)
        )
        return {
            "listing": listing,
            "replayed_head": replayed_head,
            "stale_head_status": stale_head_status,
            "erase_fence_status": erase_fence_status,
            "cold_materialize": cold,
            "io_probe": {
                "state_probe_samples_ms": cold_probe_samples_ms,
                "state_probe_min_ms": min(cold_probe_samples_ms) if cold_probe_samples_ms else None,
                "state_probe_max_ms": max(cold_probe_samples_ms) if cold_probe_samples_ms else None,
            },
            "roundtrip": roundtrip,
            "cold_continuation": cold_continuation,
            "stale_handle_status": stale_handle_status,
            "states": states,
            "durable_metrics": {
                key: value
                for key, value in metrics.items()
                if key.startswith("statetree_durable")
            },
            "manifest_compact": compact,
            "pressure_capture": pressure_capture,
            "pressure_publish": pressure_publish,
            "head_advance": head_advance,
            "head_advance_retry": head_advance_retry,
            "head_delete": head_delete,
            "head_delete_retry": head_delete_retry,
            "head_aba_create_status": head_aba_create_status,
            "old_release": old_release,
            "old_erase": old_erase,
            "release": released,
            "erase": erased,
            "process": bench.read_process_memory(proc.pid),
            "gpu": bench.read_drm_memory(proc.pid),
        }


def logical_head_replay_phase(
    args: argparse.Namespace,
    store_path: Path,
    out_dir: Path,
    restart: dict[str, Any],
) -> dict[str, Any]:
    final_digest = restart["pressure_capture"]["response"]["digest"]
    with ManagedServer(args, store_path, out_dir / f"{args.label}-head-replay.server.log") as proc:
        heads = request(args, "GET", "/snapshot-heads")
        if heads["response"].get("heads") != []:
            raise RuntimeError("deleted logical head became active after restart")
        retry = request(
            args,
            "POST",
            f"/snapshot-heads/{args.head_name}?action=delete",
            {"expected_generation": 2, "expected_digest": final_digest},
        )
        if retry["response"].get("deduplicated") is not True:
            raise RuntimeError("logical head tombstone did not deduplicate delete after restart")
        aba_status = None
        try:
            request(
                args,
                "POST",
                f"/snapshot-heads/{args.head_name}?action=create",
                {"digest": final_digest},
            )
        except bench.HttpStatusError as exc:
            aba_status = exc.status
        if aba_status != 503:
            raise RuntimeError("logical head tombstone did not fence ABA after restart")
        return {
            "heads": heads,
            "delete_retry": retry,
            "aba_create_status": aba_status,
            "process": bench.read_process_memory(proc.pid),
            "gpu": bench.read_drm_memory(proc.pid),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate durable StateTree cold content across restart.")
    parser.add_argument("--bin", default=bench.DEFAULT_BIN)
    parser.add_argument("--model", default=bench.DEFAULT_MODEL)
    parser.add_argument("--label", default="statetree-cold")
    parser.add_argument("--commit", default="dirty")
    parser.add_argument("--compat-id", required=True)
    parser.add_argument("--out-dir", default="/tmp/turbo-statetree-cold-gate")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--ctx", type=bench.positive_int, default=32768)
    parser.add_argument("--parallel", type=bench.positive_int, default=4)
    parser.add_argument("--prefix-tokens", type=bench.positive_int, default=1024)
    parser.add_argument("--suffix-tokens", type=bench.positive_int, default=32)
    parser.add_argument("--predict-tokens", type=bench.positive_int, default=8)
    parser.add_argument("--snapshot-budget-bytes", type=bench.positive_int, default=536870912)
    parser.add_argument("--disk-budget-bytes", type=bench.positive_int, default=1073741824)
    parser.add_argument("--load-budget-bytes", type=bench.positive_int, default=536870912)
    parser.add_argument("--manifest-budget-bytes", type=bench.positive_int, default=67108864)
    parser.add_argument("--owner", default="gate/b70-lifecycle")
    parser.add_argument("--retention-class", choices=("pinned", "cache"), default="pinned")
    parser.add_argument("--exercise-cache-pressure", action="store_true")
    parser.add_argument("--exercise-logical-head", action="store_true")
    parser.add_argument("--exercise-atomic-publish-advance", action="store_true")
    parser.add_argument("--head-name", default="gate-b70-main")
    parser.add_argument("--lease-ms", type=bench.positive_int, default=60000)
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
    parser.add_argument("--require-scheduler-overlap", action="store_true")
    parser.add_argument("--require-io-probes", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.port == 8093:
        raise SystemExit("refusing production port 8093; use an isolated managed server")
    if args.parallel < 2:
        raise SystemExit("cold gate requires at least two slots")
    selected_modes = sum((
        bool(args.exercise_cache_pressure),
        bool(args.exercise_logical_head),
        bool(args.exercise_atomic_publish_advance),
    ))
    if selected_modes > 1:
        raise SystemExit("run cache pressure, logical-head, and atomic acceptance as separate gates")
    if (args.exercise_logical_head or args.exercise_atomic_publish_advance) and \
            args.retention_class != "cache":
        raise SystemExit("logical-head transaction acceptance requires --retention-class cache")
    if not (1 <= len(args.compat_id.encode("utf-8")) <= 256):
        raise SystemExit("compatibility ID must contain 1 to 256 UTF-8 bytes")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "store"
    store_path.mkdir(exist_ok=True)
    result_path = out_dir / f"{args.label}.result.json"
    result: dict[str, Any] = {
        "kind": "turbo-statetree-cold-gate",
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
        phase_one = first_phase(args, store_path, out_dir)
        result["spill_phase"] = phase_one
        restart = restart_phase(args, store_path, out_dir, phase_one)
        result["restart_phase"] = restart
        if args.exercise_logical_head or args.exercise_atomic_publish_advance:
            result["logical_head_replay_phase"] = logical_head_replay_phase(
                args, store_path, out_dir, restart
            )
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
