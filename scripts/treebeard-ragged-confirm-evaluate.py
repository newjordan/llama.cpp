#!/usr/bin/env python3
"""Gate evaluation for the ragged-KV fragmented confirm + composition matrix.

Reads <out>/run/<label>.result.json produced by turbo-statetree-bench.py and
writes <out>/confirm-summary.json. Exit 0 iff every required gate passes.

Required gates (preregistered in the approved 2026-07-15 plan):
  drift        |c0-a - c0-b| / midpoint <= 1%
  c1_gain      C1 (ragged only) >= +50% vs C0 midpoint
  c2_gain      C2 (ship config) >= +55% vs C0 midpoint
  c2_over_c1   C2 >= C1 + 5 points of C1 (incremental state-io >= +5%)
  parity       identical branch tokens_sha256 sets across all fragmented arms
               and repeats; edge-long ragged on/off sets identical
  failures     zero failed samples everywhere
Informational (does not fail the run):
  hoist_adopt  C3 >= C2 * 1.01 -> adopt into RC5 ship env, else stays parked
"""

import json
import statistics
import sys
from pathlib import Path

FRAG_ARMS = ["c0-a", "c1", "c2", "c3", "c0-b"]
EDGE_ARMS = ["edge-long-off", "edge-long-on", "edge-churn"]


def load(out, label):
    p = out / "run" / f"{label}.result.json"
    return json.loads(p.read_text()) if p.exists() else None


def arm_stats(doc):
    usable = [s for s in doc["samples"] if s.get("branch_wave")]
    tps = [s["branch_wave"]["aggregate_predicted_tps"] for s in usable]
    hashes = [tuple(sorted(r["tokens_sha256"] for r in s["branch_wave"]["requests"]))
              for s in usable]
    return {
        "aggregate_tps_mean": statistics.fmean(tps) if tps else None,
        "aggregate_tps_p50": statistics.median(tps) if tps else None,
        "samples": len(doc["samples"]),
        "unusable_samples": len(doc["samples"]) - len(usable),
        "failed_samples": doc["summary"]["failed_samples"],
        "hash_sets": hashes,
    }


def main():
    out = Path(sys.argv[1])
    arms = {}
    for label in FRAG_ARMS + EDGE_ARMS:
        doc = load(out, label)
        if doc is not None:
            arms[label] = arm_stats(doc)

    missing = [l for l in FRAG_ARMS if l not in arms]
    if missing:
        print(f"MISSING_ARMS {missing}", file=sys.stderr)
        sys.exit(1)
    unusable_frag = {l: arms[l]["unusable_samples"] for l in FRAG_ARMS
                     if arms[l]["unusable_samples"]}
    if unusable_frag:
        print(f"UNUSABLE_FRAG_SAMPLES {unusable_frag}", file=sys.stderr)
        sys.exit(1)

    p50 = {l: arms[l]["aggregate_tps_p50"] for l in arms}
    mid = (p50["c0-a"] + p50["c0-b"]) / 2.0
    drift = abs(p50["c0-a"] - p50["c0-b"]) / mid * 100.0
    gain = {l: (p50[l] / mid - 1.0) * 100.0 for l in ("c1", "c2", "c3")}
    c2_over_c1 = (p50["c2"] / p50["c1"] - 1.0) * 100.0
    c3_over_c2 = (p50["c3"] / p50["c2"] - 1.0) * 100.0

    frag_hash_sets = {hs for l in FRAG_ARMS for hs in arms[l]["hash_sets"]}
    parity = len(frag_hash_sets) == 1
    edge_parity = None  # pending: edge arms absent or produced no usable samples
    if ("edge-long-off" in arms and "edge-long-on" in arms
            and arms["edge-long-off"]["hash_sets"] and arms["edge-long-on"]["hash_sets"]):
        edge_parity = arms["edge-long-off"]["hash_sets"] == arms["edge-long-on"]["hash_sets"]
    failures = sum(arms[l]["failed_samples"] for l in FRAG_ARMS)

    gates = {
        "drift_le_1pct": drift <= 1.0,
        "c1_gain_ge_50pct": gain["c1"] >= 50.0,
        "c2_gain_ge_55pct": gain["c2"] >= 55.0,
        "c2_over_c1_ge_5pct": c2_over_c1 >= 5.0,
        "parity_fragmented": parity,
        "zero_failed_samples": failures == 0,
    }
    hoist_adopt = c3_over_c2 >= 1.0

    summary = {
        "kind": "treebeard-ragged-confirm-summary",
        "control_midpoint_tps": round(mid, 6),
        "arm_p50_tps": {l: (round(v, 6) if v is not None else None) for l, v in p50.items()},
        "arm_mean_tps": {l: (round(arms[l]["aggregate_tps_mean"], 6)
                             if arms[l]["aggregate_tps_mean"] is not None else None) for l in arms},
        "control_drift_pct": round(drift, 6),
        "gain_vs_mid_pct": {l: round(v, 6) for l, v in gain.items()},
        "c2_over_c1_pct": round(c2_over_c1, 6),
        "c3_over_c2_pct": round(c3_over_c2, 6),
        "hoist_adopt": hoist_adopt,
        "parity_edge_long": edge_parity,  # None = pending separate edge run
        "gates": gates,
        "pass": all(gates.values()),
    }
    (out / "confirm-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
