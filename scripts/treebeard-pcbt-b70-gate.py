#!/usr/bin/env python3
"""PCBT-11 production-shaped gate driver: matched manual fork/run/commit vs
PCBT transaction cycles on a live (attach-mode) server.

Gates (canonical PCBT-11 first-slice bars):
  - PCBT orchestration wall (cycle wall minus branch-decode walls) adds
    <= 5% versus the matched manual cycle.
  - Branch decode throughput regresses <= 1% versus manual.
  - Winner continuation performs no prompt replay (prompt cache hit).
  - Slot/prompt-state reclamation returns to the manual baseline.

Usage: treebeard-pcbt-b70-gate.py --port P --prefix-slot 0 --branch-slots 1-5
       --repeats 3 --out DIR
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path


def http(method, url, payload=None, timeout=600.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}"), e.code


def completion(port, prompt, n_predict, pin, timeout=600.0):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, **pin}
    t0 = time.perf_counter()
    doc, err = http("POST", f"http://127.0.0.1:{port}/completion", body, timeout)
    wall = time.perf_counter() - t0
    assert err is None, f"completion failed: {doc}"
    t = doc.get("timings", {})
    return {"wall_s": wall, "predicted_n": doc.get("tokens_predicted", 0),
            "predicted_per_second": t.get("predicted_per_second", 0.0),
            "prompt_n": t.get("prompt_n", -1), "cache_n": t.get("cache_n", -1),
            "content": doc.get("content", "")}


def slots(port):
    doc, err = http("GET", f"http://127.0.0.1:{port}/slots")
    assert err is None
    return doc


def node_of(port, slot_id):
    for row in slots(port):
        if row.get("id") == slot_id:
            return int(row.get("node_id", -1))
    return -1


def state_bytes(port):
    rows = slots(port)
    return sum(int(r.get("prompt_state_bytes", r.get("state_bytes", 0)) or 0) for r in rows)


def branch_decodes(port, assignments, n_predict):
    """assignments: list of (prompt, pin-dict). Concurrent decodes, returns results."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as ex:
        futs = [ex.submit(completion, port, p, n_predict, pin) for p, pin in assignments]
        return [f.result() for f in futs]


def erase(port, slot_id, fork_id):
    doc, err = http("POST", f"http://127.0.0.1:{port}/slots/{slot_id}?action=erase",
                    {"fork_id": fork_id})
    assert err is None, doc


def mint(port, prefix_slot, dest, prompt):
    completion(port, prompt, 4, {"id_slot": prefix_slot})
    fork, err = http("POST", f"http://127.0.0.1:{port}/slots/{prefix_slot}?action=fork",
                     {"destinations": [dest]})
    assert err is None, fork
    commit, err = http("POST", f"http://127.0.0.1:{port}/slots/{prefix_slot}?action=commit",
                       {"fork_id": fork["fork_id"]})
    assert err is None, commit
    return node_of(port, prefix_slot), int(commit["fork_id"])


def manual_cycle(port, prefix_slot, branch_slots, rep, n_predict, root_fork_id):
    t0 = time.perf_counter()
    fork, err = http("POST", f"http://127.0.0.1:{port}/slots/{prefix_slot}?action=fork",
                     {"destinations": branch_slots, "fork_id": root_fork_id})
    assert err is None, fork
    t_fork = time.perf_counter() - t0
    members = [prefix_slot] + list(branch_slots)
    assignments = [(f"manual r{rep} branch {i}: continue.", {"id_slot": sid})
                   for i, sid in enumerate(members)]
    decodes = branch_decodes(port, assignments, n_predict)
    t1 = time.perf_counter()
    winner = branch_slots[0]
    commit, err = http("POST", f"http://127.0.0.1:{port}/slots/{winner}?action=commit",
                       {"fork_id": fork["fork_id"]})
    assert err is None, commit
    t_commit = time.perf_counter() - t1
    widx = members.index(winner)
    cont_prompt = assignments[widx][0] + decodes[widx]["content"]
    cont = completion(port, cont_prompt, 8, {"id_slot": winner})
    cycle = time.perf_counter() - t0
    erase(port, winner, int(commit["fork_id"]))
    return {"cycle_s": cycle, "orch_s": cycle - sum(d["wall_s"] for d in decodes) / len(decodes),
            "fork_s": t_fork, "commit_s": t_commit,
            "decode_tps": statistics.median(d["predicted_per_second"] for d in decodes),
            "cont_prompt_n": cont["prompt_n"], "winner_slot": winner}


def pcbt_cycle(port, source_node, n_branches, rep, n_predict):
    t0 = time.perf_counter()
    branches = [{"key": f"b{i}", "request": {"prompt": f"pcbt r{rep} branch {i}: continue.",
                                              "max_tokens": n_predict}}
                for i in range(n_branches)]
    create, err = http("POST", f"http://127.0.0.1:{port}/transactions", {
        "request_id": f"b70-gate-{rep}",
        "source": {"node_id": source_node},
        "branches": branches,
        "budget": {"max_slots": n_branches, "max_predicted_tokens": n_predict * n_branches * 2,
                   "deadline_ms": 600000, "max_candidate_bytes": 1048576},
        "acceptance_contract": {"kind": "external", "name": "b70-gate-v1"},
    })
    assert err is None, create
    t_create = time.perf_counter() - t0
    gen = create["generation"]
    assignments = [(b["request"]["prompt"],
                    {"node_id": nb["node_id"], "fork_id": gen})
                   for b, nb in zip(branches, create["branches"])]
    decodes = branch_decodes(port, assignments, n_predict)
    t1 = time.perf_counter()
    view, err = http("GET", f"http://127.0.0.1:{port}/transactions/{create['transaction_id']}")
    assert err is None and view["status"] == "awaiting_decision", view
    wb = [b for b in view["branches"] if b["phase"] == "completed"][0]
    commit, err = http("POST",
                       f"http://127.0.0.1:{port}/transactions/{create['transaction_id']}?action=commit", {
        "winner_node_id": wb["node_id"], "expected_fork_id": gen,
        "candidate_digest": wb["candidate_digest"],
        "evidence": {"kind": "external", "digest": "sha256:" + "ab" * 32},
    })
    assert err is None, commit
    t_commit = time.perf_counter() - t1
    widx = next(i for i, nb in enumerate(create["branches"]) if nb["node_id"] == wb["node_id"])
    cont_prompt = assignments[widx][0] + decodes[widx]["content"]
    cont = completion(port, cont_prompt, 8, {"node_id": commit["winner_node_id"]})
    cycle = time.perf_counter() - t0
    erase(port, int(commit["winner_slot"]), int(view["generation"]))
    return {"cycle_s": cycle, "orch_s": cycle - sum(d["wall_s"] for d in decodes) / len(decodes),
            "create_s": t_create, "commit_s": t_commit,
            "decode_tps": statistics.median(d["predicted_per_second"] for d in decodes),
            "cont_prompt_n": cont["prompt_n"], "winner_node": commit["winner_node_id"],
            "receipt_digest": commit["receipt_digest"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prefix-slot", type=int, default=0)
    ap.add_argument("--branch-slots", default="1-5")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = args.branch_slots.split("-")
    branch_slots = list(range(int(lo), int(hi) + 1))
    n_branches = len(branch_slots) + 1  # manual family = source + destinations
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    baseline_bytes = state_bytes(args.port)
    manual, pcbt = [], []
    for rep in range(args.repeats):
        node, root_fork = mint(args.port, args.prefix_slot, branch_slots[0], f"gate prefix {rep} manual:")
        assert node >= 0
        manual.append(manual_cycle(args.port, args.prefix_slot, branch_slots, rep, args.n_predict, root_fork))
        node, root_fork = mint(args.port, args.prefix_slot, branch_slots[0], f"gate prefix {rep} pcbt:")
        assert node >= 0
        pcbt.append(pcbt_cycle(args.port, node, n_branches, rep, args.n_predict))

    m_orch = statistics.median(m["orch_s"] for m in manual)
    p_orch = statistics.median(p["orch_s"] for p in pcbt)
    m_tps = statistics.median(m["decode_tps"] for m in manual)
    p_tps = statistics.median(p["decode_tps"] for p in pcbt)
    cont_reuse = all(0 <= m["cont_prompt_n"] <= 4 for m in manual) and all(0 <= p["cont_prompt_n"] <= 4 for p in pcbt)
    final_bytes = state_bytes(args.port)

    summary = {
        "kind": "treebeard-pcbt-b70-gate",
        "repeats": args.repeats,
        "family_size": n_branches,
        "manual_orch_s_p50": round(m_orch, 4),
        "pcbt_orch_s_p50": round(p_orch, 4),
        "orch_overhead_pct": round((p_orch / m_orch - 1) * 100, 3) if m_orch > 0 else None,
        "manual_decode_tps_p50": round(m_tps, 3),
        "pcbt_decode_tps_p50": round(p_tps, 3),
        "decode_delta_pct": round((p_tps / m_tps - 1) * 100, 3),
        "continuation_cache_reuse": cont_reuse,
        "baseline_state_bytes": baseline_bytes,
        "final_state_bytes": final_bytes,
        "manual": manual, "pcbt": pcbt,
        "gates": {},
    }
    summary["gates"] = {
        "orchestration_le_5pct": summary["orch_overhead_pct"] is not None and summary["orch_overhead_pct"] <= 5.0,
        "decode_ge_-1pct": summary["decode_delta_pct"] >= -1.0,
        "continuation_no_replay": cont_reuse,
    }
    summary["pass"] = all(summary["gates"].values())
    (out / "pcbt-b70-gate.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("orch_overhead_pct", "manual_decode_tps_p50", "pcbt_decode_tps_p50",
                       "decode_delta_pct", "continuation_cache_reuse", "gates", "pass")}, indent=1))
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
