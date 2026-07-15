#!/usr/bin/env python3

import argparse
import json
import urllib.error
import urllib.request


def request(port: int, method: str, path: str, body=None, timeout: float = 300.0):
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc


def slots(port: int):
    rows = request(port, "GET", "/slots")
    if not isinstance(rows, list):
        raise RuntimeError("GET /slots did not return an array")
    return {int(row["id"]): row for row in rows}


def require_scale(rows, slot_id: int, expected: float, label: str):
    actual = rows[slot_id].get("jspace_control_scale")
    if actual != expected:
        raise RuntimeError(f"{label}: slot {slot_id} scale {actual!r}, expected {expected!r}")


def completion(port: int, slot_id: int, prompt: str, scale=None):
    body = {
        "prompt": prompt,
        "id_slot": slot_id,
        "n_predict": 2,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1,
        "cache_prompt": True,
        "ignore_eos": True,
        "return_tokens": True,
    }
    if scale is not None:
        body["jspace_control_scale"] = scale
    result = request(port, "POST", "/completion", body)
    if result.get("id_slot") != slot_id or result.get("timings", {}).get("predicted_n") != 2:
        raise RuntimeError(f"completion did not finish on slot {slot_id}: {result!r}")
    return result


def node_for_slot(fork, slot_id: int) -> int:
    for node in fork.get("nodes", []):
        if node.get("id_slot") == slot_id:
            return int(node["node_id"])
    raise RuntimeError(f"fork response has no node for slot {slot_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    port = args.port

    evidence = {"schema": "treebeard.jspace.server_lifecycle.v1", "events": []}
    prompt = "The operator checks the signed manifest before routing the request."

    active = completion(port, 0, prompt, 1.0)
    rows = slots(port)
    require_scale(rows, 0, 1.0, "initial active request")
    evidence["events"].append({"event": "active_request", "response": active})

    fork = request(port, "POST", "/slots/0?action=fork", {"destinations": [2]}, args.timeout)
    fork_id = int(fork["fork_id"])
    source_node = node_for_slot(fork, 0)
    rows = slots(port)
    require_scale(rows, 0, 1.0, "fork source")
    require_scale(rows, 2, 1.0, "fork destination")
    evidence["events"].append({"event": "fork", "response": fork})

    protected = completion(port, 1, prompt, 0.0)
    rows = slots(port)
    require_scale(rows, 0, 1.0, "reserved active request")
    require_scale(rows, 1, 0.0, "protected request")
    require_scale(rows, 2, 1.0, "reserved fork destination")
    evidence["events"].append({"event": "protected_request", "response": protected})

    snapshot = request(
        port,
        "POST",
        f"/nodes/{source_node}?action=snapshot",
        {"state_id": fork["state_id"], "fork_id": fork_id},
        args.timeout,
    )
    if int(snapshot.get("payload_bytes", 0)) <= 12:
        raise RuntimeError("snapshot payload did not include serialized state")
    evidence["events"].append({"event": "snapshot", "response": snapshot})

    commit = request(port, "POST", "/slots/2?action=commit", {"fork_id": fork_id}, args.timeout)
    rows = slots(port)
    require_scale(rows, 0, 0.0, "commit loser")
    require_scale(rows, 2, 1.0, "commit winner")
    evidence["events"].append({"event": "commit", "response": commit})

    materialized = request(
        port,
        "POST",
        f"/snapshots/{snapshot['snapshot_id']}?action=materialize",
        {"digest": snapshot["digest"], "id_slot": 3},
        args.timeout,
    )
    rows = slots(port)
    require_scale(rows, 3, 1.0, "materialized snapshot")
    evidence["events"].append({"event": "materialize", "response": materialized})

    request(port, "POST", "/slots/2?action=erase", {"fork_id": commit["fork_id"]}, args.timeout)
    request(port, "POST", "/slots/3?action=erase", {"fork_id": materialized["fork_id"]}, args.timeout)
    request(port, "POST", "/slots/1?action=erase", {}, args.timeout)

    reused = completion(port, 0, "Return only the word protected.", None)
    rows = slots(port)
    for slot_id in range(4):
        require_scale(rows, slot_id, 0.0, "final reset")
        if rows[slot_id].get("is_reserved") or rows[slot_id].get("is_processing"):
            raise RuntimeError(f"slot {slot_id} remained active after cleanup")
    evidence["events"].append({"event": "slot_reuse_default_zero", "response": reused})
    evidence["status"] = "pass"
    evidence["checks"] = {
        "request_scope": "pass",
        "fork_copy": "pass",
        "commit_loser_clear": "pass",
        "commit_winner_keep": "pass",
        "snapshot_scale_restore": "pass",
        "slot_reuse_default_zero": "pass",
    }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "pass", "out": args.out}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
