#!/usr/bin/env python3
"""Offline replay of the frozen N(X)+Y fanout controller on recorded traces.

Preregistration: reports/treebeard-nxy-optimizer-preregistration-20260715.md
(B1, including the pre-run method amendment). Zero rig cost: inputs are the
already-recorded CPU wavefront width-sweep artifacts. This validates the
controller LOGIC only; it makes no claim about token-channel viability on B70
(three park verdicts stand).

Controller (frozen):
  ladder {0,2,4,8}; EWMA alpha 0.3; promote one step after 2 consecutive
  rounds with acceptance >= 0.75; demote to 0 on any round < 0.25; while at 0,
  probe one round at fanout 2 every 32 serial tokens; initial state is the
  coverage-first probe (first round at fanout 2).

Cost model (amended, pre-run):
  c_tok0        = wall_s(w=0) / predicted_n                (same case+repeat)
  n_serial(w)   = predicted_n - (rounds(w) + accepted(w))
  c_round(w)    = (wall_s(w) - n_serial(w) * c_tok0) / rounds(w)
  Empty trace at width w (no rounds recorded): a chosen round degenerates to
  one serial token at c_tok0 with zero proposals (coverage miss).
  Crude preregistered model (wall_s / rounds) is reported as sensitivity.
"""

import json
import statistics
import sys
from pathlib import Path

RESULTS = Path("/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront")
INPUTS = [
    RESULTS / "cpu-width-sweep-clean-r2.json",
    RESULTS / "serial-anchor-cpu-proposal-smoke-20260714.json",
]
LADDER = [0, 2, 4, 8]
ALPHA = 0.3
PROMOTE_ACC = 0.75
PROMOTE_CONSEC = 2
DEMOTE_ACC = 0.25
PROBE_EVERY = 32
PROBE_WIDTH = 2

GATE_CAPTURE_MIN = 0.70   # G-B1a on structured-copy
GATE_ZERO_FRAC = 0.95     # G-B1b on free-prose and code-edit


def candidate_samples(data):
    """Candidate (non-control) samples grouped by (case, repeat) -> {width: sample}."""
    groups = {}
    for s in data["samples"]:
        if s.get("control_phase"):
            continue
        groups.setdefault((s["case_id"], s.get("repeat", 0)), {})[s["width"]] = s
    return groups


def round_cost(sample, c_tok0, crude):
    rounds = len(sample["draft_n_per_round"])
    if rounds == 0:
        return c_tok0
    if crude:
        return sample["wall_s"] / rounds
    n_serial = sample["predicted_n"] - (rounds + sample["draft_n_accepted"])
    return max((sample["wall_s"] - n_serial * c_tok0) / rounds, 1e-9)


def replay(widths, c_tok0, n_target, crude):
    """Run the frozen controller until n_target tokens commit.

    widths: {width: sample} for one case+repeat. Returns totals and the
    per-round width choices (serial tokens count as width-0 rounds).
    """
    width = 0
    pending_probe = True          # coverage-first: first action is a probe
    ewma = 0.0
    consec = 0
    since_probe = 0
    committed = 0
    wall = 0.0
    choices = []
    cursor = {w: 0 for w in widths}

    while committed < n_target:
        if width == 0 and not pending_probe:
            committed += 1
            wall += c_tok0
            since_probe += 1
            choices.append(0)
            if since_probe >= PROBE_EVERY:
                pending_probe = True
                since_probe = 0
            continue

        w = PROBE_WIDTH if width == 0 else width
        pending_probe = False
        sample = widths.get(w)
        trace = sample["draft_n_per_round"] if sample else []
        acc_trace = sample["draft_n_accepted_per_round"] if sample else []
        if trace:
            i = min(cursor[w], len(trace) - 1)
            proposed, accepted = trace[i], acc_trace[i]
            cursor[w] = cursor.get(w, 0) + 1
            cost = round_cost(sample, c_tok0, crude)
        else:
            proposed, accepted, cost = 0, 0, c_tok0
        committed += 1 + accepted
        wall += cost
        choices.append(w)
        acc = accepted / proposed if proposed else 0.0
        ewma = ALPHA * acc + (1 - ALPHA) * ewma
        if acc >= PROMOTE_ACC:
            consec += 1
            if consec >= PROMOTE_CONSEC:
                width = LADDER[min(LADDER.index(width) + 1, len(LADDER) - 1)] if width else PROBE_WIDTH
                consec = 0
        else:
            consec = 0
        if acc < DEMOTE_ACC:
            width = 0
            since_probe = 0
    return committed, wall, choices


def main():
    out = {"kind": "treebeard-nxy-controller-replay",
           "preregistration": "reports/treebeard-nxy-optimizer-preregistration-20260715.md",
           "controller": {"ladder": LADDER, "alpha": ALPHA, "promote_acc": PROMOTE_ACC,
                          "promote_consec": PROMOTE_CONSEC, "demote_acc": DEMOTE_ACC,
                          "probe_every": PROBE_EVERY, "probe_width": PROBE_WIDTH},
           "inputs": [str(p) for p in INPUTS], "traces": []}

    per_case = {}
    for path in INPUTS:
        data = json.loads(path.read_text())
        best_fixed = {c: max(r["aggregate_tps_gain_pct"] for r in rows)
                      for c, rows in data["case_summaries"].items()}
        for (case, rep), widths in candidate_samples(data).items():
            base = next((s for s in data["samples"]
                         if s.get("control_phase") and s["case_id"] == case
                         and s.get("repeat", 0) == rep and s["width"] == 0), None)
            if base is None:
                base = next(s for s in data["samples"]
                            if s.get("control_phase") and s["case_id"] == case and s["width"] == 0)
            n_target = base["predicted_n"]
            c_tok0 = base["wall_s"] / n_target
            row = {"file": path.name, "case": case, "repeat": rep,
                   "best_fixed_gain_pct": best_fixed[case]}
            for crude in (False, True):
                committed, wall, choices = replay(widths, c_tok0, n_target, crude)
                tps = committed / wall
                gain = (tps / (1.0 / c_tok0) - 1.0) * 100.0
                key = "crude" if crude else "amended"
                row[key] = {"gain_pct": round(gain, 3),
                            "rounds": len(choices),
                            "zero_width_frac": round(sum(1 for c in choices if c == 0) / len(choices), 4)}
            per_case.setdefault(case, []).append(row)
            out["traces"].append(row)

    summary = {}
    for case, rows in per_case.items():
        gains = [r["amended"]["gain_pct"] for r in rows]
        zfrac = [r["amended"]["zero_width_frac"] for r in rows]
        best = statistics.median(r["best_fixed_gain_pct"] for r in rows)
        med_gain = statistics.median(gains)
        summary[case] = {"median_adaptive_gain_pct": round(med_gain, 3),
                         "median_best_fixed_gain_pct": round(best, 3),
                         "capture_ratio": round(med_gain / best, 4) if best > 0 else None,
                         "median_zero_width_frac": round(statistics.median(zfrac), 4)}
    out["summary"] = summary

    g_b1a = (summary.get("structured-copy", {}).get("capture_ratio") or 0) >= GATE_CAPTURE_MIN
    g_b1b = all(summary[c]["median_zero_width_frac"] >= GATE_ZERO_FRAC
                for c in ("free-prose", "code-edit") if c in summary)
    out["gates"] = {"G-B1a_capture_ge_0.70_structured": g_b1a,
                    "G-B1b_zero_width_ge_0.95_prose_code": g_b1b,
                    "pass": g_b1a and g_b1b}

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("nxy-controller-replay.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({"summary": summary, "gates": out["gates"]}, indent=2))


if __name__ == "__main__":
    main()
