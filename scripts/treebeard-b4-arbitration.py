#!/usr/bin/env python3
"""B4 slot-arbitration benchmark (frozen protocol:
reports/treebeard-b4-arbitration-protocol-20260716.md).

For each load point L in {4, 8, 11}: run L closed-loop real agents for
--duration seconds, once with the thinking client OFF and once ON. The
thinking client is idle-slot-only: it polls /slots and runs one 2-branch
PCBT wave only when >= 3 slots are idle and none are queued, else fanout 0.

Gates:
  G-B4a  real-agent request-wall p50 regression <= 1% (ON vs OFF) per L
  G-B4b  thinking wave wall >= 15% better than the matched sequential
         single-pass pair at L <= 8
  G-B4c  zero waves at L = 11 (fanout degrades to 0) and no starved agents
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import threading
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


def completion(port, prompt, n_predict, pin=None, timeout=600.0):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, **(pin or {})}
    t0 = time.perf_counter()
    doc, err = http("POST", f"http://127.0.0.1:{port}/completion", body, timeout)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "ok": err is None,
            "tps": doc.get("timings", {}).get("predicted_per_second", 0.0) if err is None else 0.0}


def idle_slots(port):
    doc, err = http("GET", f"http://127.0.0.1:{port}/slots", None, 10.0)
    if err is not None:
        return []
    return [r["id"] for r in doc if not r.get("is_processing")]


def agent_loop(port, agent_id, stop, records):
    i = 0
    while not stop.is_set():
        r = completion(port, f"agent {agent_id} request {i}: summarize the state of play.", 48)
        records.append(r)
        i += 1


def thinking_wave(port, wave_id):
    """One idle-slot-only PCBT wave: mint on an idle slot, 2 branches, commit,
    erase. Returns wall seconds or None if skipped."""
    idles = idle_slots(port)
    if len(idles) < 3:
        return None
    t0 = time.perf_counter()
    root = idles[0]
    completion(port, f"thinking {wave_id} prefix:", 4, {"id_slot": root})
    fork, err = http("POST", f"http://127.0.0.1:{port}/slots/{root}?action=fork",
                     {"destinations": [idles[1]]})
    if err is not None:
        return None
    _, err = http("POST", f"http://127.0.0.1:{port}/slots/{root}?action=commit",
                  {"fork_id": fork["fork_id"]})
    if err is not None:
        return None
    doc, err = http("GET", f"http://127.0.0.1:{port}/slots", None, 10.0)
    node = next((int(r.get("node_id", -1)) for r in doc if r.get("id") == root), -1)
    create, err = http("POST", f"http://127.0.0.1:{port}/transactions", {
        "request_id": f"b4-wave-{wave_id}",
        "source": {"node_id": node},
        "branches": [{"key": "a", "request": {"prompt": f"thinking {wave_id} option A:", "max_tokens": 48}},
                     {"key": "b", "request": {"prompt": f"thinking {wave_id} option B:", "max_tokens": 48}}],
        "budget": {"max_slots": 2, "max_predicted_tokens": 256,
                   "deadline_ms": 120000, "max_candidate_bytes": 65536},
        "acceptance_contract": {"kind": "external", "name": "b4-v1"},
    })
    if err is not None:
        return None
    gen = create["generation"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(lambda nb: completion(port, f"thinking {wave_id} branch:", 48,
                                          {"node_id": nb["node_id"], "fork_id": gen}),
                    create["branches"]))
    view, err = http("GET", f"http://127.0.0.1:{port}/transactions/{create['transaction_id']}")
    if err is not None or view.get("status") != "awaiting_decision":
        return None
    wb = next(b for b in view["branches"] if b["phase"] == "completed")
    commit, err = http("POST",
                       f"http://127.0.0.1:{port}/transactions/{create['transaction_id']}?action=commit", {
        "winner_node_id": wb["node_id"], "expected_fork_id": gen,
        "candidate_digest": wb["candidate_digest"],
        "evidence": {"kind": "external", "digest": "sha256:" + "b4" * 32},
    })
    wall = time.perf_counter() - t0
    if err is None:
        http("POST", f"http://127.0.0.1:{port}/slots/{commit['winner_slot']}?action=erase",
             {"fork_id": view["generation"]})
    return wall


def thinking_loop(port, stop, waves, skips):
    wave_id = 0
    while not stop.is_set():
        wall = thinking_wave(port, wave_id)
        if wall is None:
            skips.append(1)
            time.sleep(0.5)
        else:
            waves.append(wall)
        wave_id += 1


def run_point(port, load, duration, thinking):
    stop = threading.Event()
    records, waves, skips = [], [], []
    threads = [threading.Thread(target=agent_loop, args=(port, i, stop, records))
               for i in range(load)]
    if thinking:
        threads.append(threading.Thread(target=thinking_loop, args=(port, stop, waves, skips)))
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=300)
    ok = [r for r in records if r["ok"]]
    return {
        "load": load, "thinking": thinking,
        "agent_requests": len(records), "agent_failures": len(records) - len(ok),
        "agent_wall_p50": round(statistics.median(r["wall_s"] for r in ok), 4) if ok else None,
        "agent_tps_p50": round(statistics.median(r["tps"] for r in ok), 3) if ok else None,
        "waves": len(waves), "wave_wall_p50": round(statistics.median(waves), 3) if waves else None,
        "wave_skips": len(skips),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--loads", default="4,8,11")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # matched sequential single-pass thinking reference (2 candidates, no load)
    seq0 = time.perf_counter()
    completion(args.port, "thinking ref option A:", 48)
    completion(args.port, "thinking ref option B:", 48)
    seq_ref_idle = time.perf_counter() - seq0

    points = []
    for load in [int(x) for x in args.loads.split(",")]:
        for thinking in (False, True):
            print(f"-- load={load} thinking={thinking}", flush=True)
            points.append(run_point(args.port, load, args.duration, thinking))

    def pick(load, thinking):
        return next(p for p in points if p["load"] == load and p["thinking"] == thinking)

    gates = {}
    per_load = {}
    for load in {p["load"] for p in points}:
        off, on = pick(load, False), pick(load, True)
        reg = (on["agent_wall_p50"] / off["agent_wall_p50"] - 1) * 100 if off["agent_wall_p50"] else None
        per_load[load] = {"off": off, "on": on, "agent_wall_regression_pct": round(reg, 2) if reg is not None else None}
    gates["G-B4a_regression_le_1pct"] = all(
        v["agent_wall_regression_pct"] is not None and v["agent_wall_regression_pct"] <= 1.0
        for v in per_load.values())
    low_loads = [l for l in per_load if l <= 8]
    wave_walls = [per_load[l]["on"]["wave_wall_p50"] for l in low_loads
                  if per_load[l]["on"]["wave_wall_p50"] is not None]
    # sequential reference under load ~= 2 * single request wall at that load
    seq_under_load = [2 * per_load[l]["on"]["agent_wall_p50"] for l in low_loads]
    gates["G-B4b_wave_ge_15pct_faster"] = bool(wave_walls) and all(
        w <= s * 0.85 for w, s in zip(wave_walls, sorted(seq_under_load)[:len(wave_walls)]))
    high = max(per_load)
    gates["G-B4c_fanout_zero_at_high_load"] = (per_load[high]["on"]["waves"] == 0 and
                                               per_load[high]["on"]["agent_failures"] == 0)

    summary = {"kind": "treebeard-b4-arbitration", "duration_s": args.duration,
               "seq_ref_idle_s": round(seq_ref_idle, 3),
               "per_load": {str(k): v for k, v in sorted(per_load.items())},
               "gates": gates, "pass": all(gates.values())}
    (out / "b4-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"per_load": {k: {"regression_pct": v["agent_wall_regression_pct"],
                                       "waves": v["on"]["waves"]}
                                   for k, v in summary["per_load"].items()},
                      "gates": gates, "pass": summary["pass"]}, indent=1))
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
