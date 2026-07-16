#!/usr/bin/env python3
"""B2 branch-cost curve c(N) evaluation.

Reads <out>/run/{ragged,dense}-n<N>.result.json (turbo-statetree-bench output)
and writes <out>/branch-cost-summary.json.

Per point: per-stream tok/s = median request predicted_tps in the branch wave
(trunk and branches are symmetric 64-token greedy decodes); aggregate =
branch_wave.aggregate_predicted_tps; fork/commit client latencies recorded.

Preregistered kill gate G-B2: with ragged on, per-stream loss at N=3 relative
to N=0 > 35% -> optimizer narrows to idle-slot-only filling (G-B3b strict).
Exit 0 always unless arms are missing or samples failed — the kill gate is a
routing decision, not an artifact failure; its verdict is in the JSON.
"""

import json
import statistics
import sys
from pathlib import Path

FANOUTS = [0, 1, 3, 5, 7, 11]


def point(out, label):
    p = out / "run" / f"{label}.result.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    if doc.get("kind") == "solo-probe":
        return {
            "per_stream_tps_p50": doc["timings"]["predicted_per_second"],
            "aggregate_tps_p50": doc["timings"]["predicted_per_second"],
            "fork_client_ms_p50": None,
            "requests": 1,
            "failed_samples": 0,
            "hash_sets": [],
        }
    per_stream, aggregate, fork_ms, hashes = [], [], [], []
    for s in doc["samples"]:
        reqs = s["branch_wave"]["requests"]
        per_stream.append(statistics.median(r["predicted_tps"] for r in reqs))
        aggregate.append(s["branch_wave"]["aggregate_predicted_tps"])
        fork_ms.append(s["fork"]["client_ms"])
        hashes.append(tuple(sorted(r["tokens_sha256"] for r in reqs)))
    return {
        "per_stream_tps_p50": statistics.median(per_stream),
        "aggregate_tps_p50": statistics.median(aggregate),
        "fork_client_ms_p50": statistics.median(fork_ms),
        "requests": len(doc["samples"][0]["branch_wave"]["requests"]),
        "failed_samples": doc["summary"]["failed_samples"],
        "hash_sets": hashes,
    }


def main():
    out = Path(sys.argv[1])
    curve = {"ragged": {}, "dense": {}}
    missing, failures = [], 0
    for arm in ("ragged", "dense"):
        for n in FANOUTS:
            pt = point(out, f"{arm}-n{n}")
            if pt is None:
                missing.append(f"{arm}-n{n}")
                continue
            failures += pt["failed_samples"]
            curve[arm][n] = pt
    if missing:
        print(f"MISSING_POINTS {missing}", file=sys.stderr)
        sys.exit(1)

    # Cross-arm token parity is DIAGNOSTIC, not a gate: the backend has
    # recorded run-to-run scheduling nondeterminism on multi-stream waves
    # (intra-arm repeat flutter observed in dense-only arms; see the
    # 20260716-002318 verdict). Report per-N parity and flutter instead.
    parity_by_n = {n: curve["ragged"][n]["hash_sets"][0] == curve["dense"][n]["hash_sets"][0]
                   for n in FANOUTS if n > 0}
    flutter = [f"{arm}-n{n}" for arm in ("ragged", "dense") for n in FANOUTS
               if n > 0 and len(set(curve[arm][n]["hash_sets"])) > 1]
    parity = all(parity_by_n.values())

    r = curve["ragged"]
    trunk_loss_n3_pct = (1 - r[3]["per_stream_tps_p50"] / r[0]["per_stream_tps_p50"]) * 100
    kill = trunk_loss_n3_pct > 35.0

    summary = {
        "kind": "treebeard-branch-cost-summary",
        "curve": {arm: {n: {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in pt.items() if k != "hash_sets"}
                        for n, pt in pts.items()}
                  for arm, pts in curve.items()},
        "trunk_loss_at_n3_pct_ragged": round(trunk_loss_n3_pct, 4),
        "gate_G-B2_kill": kill,
        "parity_ragged_vs_dense_diagnostic": parity,
        "parity_by_n": {str(k): v for k, v in parity_by_n.items()},
        "intra_arm_flutter": flutter,
        "zero_failed_samples": failures == 0,
    }
    (out / "branch-cost-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
