#!/usr/bin/env python3
"""Gate evaluation for the ragged-KV dense golden-shape A/B/A.

Reads <out>/{control-a,candidate,control-b}/pareto/<label>.result.json from
treebeard-ragged-golden-aba-guarded.sh and writes <out>/golden-summary.json.
Exit 0 iff every gate passes.

Gates (preregistered):
  no_regression  candidate >= -1.0% vs control midpoint, p50 AND mean,
                 at every agents level (1, 8, 12)
  drift          |control-a - control-b| / midpoint <= 1% at every level (p50)
  failures       zero failed samples in every arm
"""

import json
import sys
from pathlib import Path

ARMS = ["control-a", "candidate", "control-b"]


def rows(out, label):
    doc = json.loads((out / label / "pareto" / f"{label}.result.json").read_text())
    failed = doc["summary"]["failed_samples"]
    return {r["active_agents"]: r["aggregate_wall_tps"] for r in doc["summary"]["rows"]}, failed


def main():
    out = Path(sys.argv[1])
    data, failed = {}, {}
    for a in ARMS:
        data[a], failed[a] = rows(out, a)

    levels = sorted(data["candidate"].keys())
    per_level = {}
    ok_regression, ok_drift = True, True
    for lv in levels:
        mid_p50 = (data["control-a"][lv]["p50"] + data["control-b"][lv]["p50"]) / 2
        mid_mean = (data["control-a"][lv]["mean"] + data["control-b"][lv]["mean"]) / 2
        d_p50 = (data["candidate"][lv]["p50"] / mid_p50 - 1) * 100
        d_mean = (data["candidate"][lv]["mean"] / mid_mean - 1) * 100
        drift = abs(data["control-a"][lv]["p50"] - data["control-b"][lv]["p50"]) / mid_p50 * 100
        per_level[lv] = {
            "control_mid_p50": round(mid_p50, 4), "candidate_p50": round(data["candidate"][lv]["p50"], 4),
            "delta_p50_pct": round(d_p50, 4), "delta_mean_pct": round(d_mean, 4),
            "control_drift_pct": round(drift, 4),
        }
        ok_regression &= d_p50 >= -1.0 and d_mean >= -1.0
        ok_drift &= drift <= 1.0

    gates = {
        "no_regression_ge_-1pct_all_levels": ok_regression,
        "control_drift_le_1pct_all_levels": ok_drift,
        "zero_failed_samples": all(v == 0 for v in failed.values()),
    }
    summary = {"kind": "treebeard-ragged-golden-summary",
               "levels": per_level, "failed_samples": failed,
               "gates": gates, "pass": all(gates.values())}
    (out / "golden-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
