# Prompt: Treebeard quality-gate loop (gated)

Copy everything below the line into a new agent session (or a scheduled
foreground task). Do not soften the gates. Production pin stays until a human
writes ACCEPT.

---

## Role

You are continuing Treebeard B70 quality-gate work on turbo. Your job is to run
**one gated round** of the quality gate (or a short multi-round loop if the
contract allows), write durable receipts, and **always restore production**.

You are **not** authorized to:

- ship / unpin `TREEBEARD_GDN_OUT_FLAT` without a human `ACCEPT` file
- rebuild `build-treebeard-single-wavefront` unless the round explicitly requires
  it (that tree **is** production on next restart)
- use spark-bench (retired; not the B70 gate)
- invent statistical significance theatre when same-arm is exact or floor-dominated

## Read first (in order)

1. `worktrees/treebeard-moe-down-reduce/reports/treebeard-quality-gate-handoff-20260726.md`
2. `treebeard-work/results/treebeard-quality-gate-20260726/decision.md`
3. This prompt’s **Gate contract** (below)

Prior facts you must not re-discover from scratch (verify if you change binary/env):

| Fact | Status |
|---|---|
| Multi-seq `llama-perplexity` exercises GDN graph via `n_seq = n_batch/n_ctx` | true; **`-np` is ignored** on PPL/KL |
| Full-vocab PPL is **noise-blind** for GDN (32ch: 0.000750 ≈ 0.000743) | use PPL for **byte levers only** |
| Sequential logprob same-arm is clean (~4e-7 KL, 0 flips) | **GDN-blind** (n_seqs=1) |
| Concurrent logprob is noisy (np4 ~1–2e-4; np12 ~9e-4 same-arm) | only path that **sees** GDN |
| Deferred-reduce is **not** the concurrency noise source | `DEFERRED_REDUCE=0` as bad or worse |
| GDN still parked | production pin `TREEBEARD_GDN_OUT_FLAT=0` |
| Production unit | `treebeard-b70-ship` :8093; restore with `results/treebeard-b70-gdn-out-flat-20260725/restore-and-verify.sh 0` |

## Gate contract (frozen for this loop)

### Strategies (pick exactly one per loop invocation)

Declare in the first line of the round receipt: `STRATEGY=A|B|C`.

| ID | Name | Goal |
|---|---|---|
| **A** | Determinism first | Drive concurrent same-arm toward KL ≪ 1e-5 / 0 flips at np4 (or prove why impossible) |
| **B** | Noise-model GDN gate | N≥5 concurrent same-arm pairs → floor μ; GDN must clear written multiple + downstream later |
| **C** | Byte levers | Surgery GGUF + PPL/KL vs stock; GDN out of scope |

**Default if owner did not specify: STRATEGY=A** (gated approach).  
Do not mix A+B metrics in one ACCEPT decision.

### ACCEPT (human only)

You may create `…/round-N/PROPOSE_ACCEPT.md` with evidence.  
You may **not** create `…/ACCEPT` or change the systemd drop-in to unset flat=0.

Human ACCEPT requires, for GDN (strategies A/B after success):

1. Instrument floor stated and measured this round  
2. GDN (or other treatment) measured on the **same** instrument  
3. Explicit judgment paragraph (not p-values cosplay)  
4. Optional but preferred: downstream score with its own same-arm/rerun spread  
5. Speed claim only after quality — ABA p50 with per-round arms checked (not aggregate alone)

### KILL (auto-stop the loop)

Stop the loop and write `…/round-N/KILLED.md` if any:

1. Ship restore fails health after a window  
2. Same-arm floor **worsens >2×** vs previous round without a deliberate env change  
3. Strategy A: after **2** consecutive rounds concurrent same-arm still has ≥1/4 greedy flips **and** mean KL > 5e-5 with no new hypothesis  
4. Strategy B: GDN mean KL / same-arm μ < **3** after N≥5 (current reval was ~2× — not enough)  
5. Strategy C: quality regression on PPL/KL exceeds a pre-declared bound, or surgery confounds (non-down tensors changed)  
6. Wall time for one window > **45 minutes** without a partial receipt  
7. You are about to rebuild the ship binary without a round hypothesis that names that risk  

### Shell / GPU traps (mandatory)

```bash
# Bench window (symlink-mask; stop alone is insufficient)
U=~/.config/systemd/user/treebeard-b70-ship.service
systemctl --user stop treebeard-b70-ship
[ -L "$U" ] || { mv "$U" "$U.bench-window"; ln -s /dev/null "$U"; }
systemctl --user daemon-reload

# Always trap restore
trap 'bash /home/frosty40/turbo/treebeard-work/results/treebeard-b70-gdn-out-flat-20260725/restore-and-verify.sh 0' EXIT INT TERM

# setvars under set -u
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u

# wait only on collected curl PIDs — never bare wait with background llama-server
# never pkill; never pgrep -f that matches your wrapper
# result JSON is empty until process exit — do not poll mid-run as success
```

Ship env for arms:

`ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0`
`GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ENABLE_MOE_PIPELINE=0 GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0`
`LD_LIBRARY_PATH=…/build-treebeard-single-wavefront/bin:$LD_LIBRARY_PATH`

Binary: `/home/frosty40/turbo/treebeard-work/build-treebeard-single-wavefront/bin/llama-server`  
Model: `/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`

## One round procedure

### Round layout

```
treebeard-work/results/treebeard-quality-gate-20260726/rounds/YYYYMMDD-HHMM-roundN/
  ROUND.md          # hypothesis, STRATEGY, kill checks
  env.txt           # uname, git SHA if any, binary mtime, env dump
  metrics.json      # machine-readable summary
  KILLED.md         # only if killed
  PROPOSE_ACCEPT.md # only if evidence supports human accept
  raw/              # logs, lp-*.json, ppl logs
```

### STRATEGY=A (default — determinism first)

**Hypothesis template:** “Concurrent multi-slot decode can be made same-arm clean by &lt;one change&gt;.”

Allowed experiments (one primary knob per round):

- concurrency structure (np4 vs fewer simultaneous curls; batch vs staggered)  
- env knobs that might affect reduction order / graphs (document each)  
- confirm sequential still clean as control every round  

**Pass for this strategy (round success, not ship):**  
concurrent same-arm (np4, 4 prompts, 96 tok, temp0, top_k1, n_probs8, seed1234, cache_prompt false):  

- greedy flips **0/4**  
- mean top-8 KL **&lt; 1e-5** (prefer &lt; 1e-6)  
- sequential control still 0 flips  

If pass: next round may open STRATEGY=B measurement of GDN under the now-clean instrument.  
If fail: write what you tried, residual floor, and whether to pivot to B or C.

**Do not** claim GDN ship-ready from A alone.

### STRATEGY=B (noise-model GDN)

Only after owner chooses B, or A is declared impossible in a KILLED/PROPOSE note.

1. Run **N≥5** independent concurrent same-arm pairs (fresh process each arm).  
2. Record μ_floor, max_floor, flip rate.  
3. Run GDN flat0 vs flat1 concurrent same harness (prefer np4; np12 only if floor is re-characterised there).  
4. Require **mean_KL_GDN ≥ 3 × μ_floor** **and** flip pattern not explained by same-arm alone — else FAIL.  
5. Even if PASS, only write `PROPOSE_ACCEPT.md`; still need downstream score for ship.

### STRATEGY=C (byte levers)

Independent of GDN.

1. Verify free disk (≥30 GB).  
2. Build surgery model with  
   `llama-quantize --allow-requantize --tensor-type ffn_down_exps=q5_K in.gguf out.gguf COPY`  
   (expect 40 tensors; note blk.1 is q8_0).  
3. PPL + `--kl-divergence` vs stock on wikitext-2 (single-stream OK).  
4. Predict speed from **2.445 us/MB**; then measure; then 12-agent ABA if quality OK.  
5. State double-quant pessimism and Unsloth mix caveats in the receipt.

## Metrics schema (`metrics.json`)

```json
{
  "strategy": "A",
  "round": 1,
  "git_or_binary": "...",
  "same_arm": {"mode": "concurrent_np4", "mean_kl": 0.0, "flips": "0/4", "n_pairs": 1},
  "treatment": {"name": "none|gdn|q5k_down", "mean_kl": null, "flips": null},
  "ratio_treatment_over_floor": null,
  "sequential_control": {"mean_kl": null, "flips": null},
  "ship_restored": true,
  "decision": "continue|killed|propose_accept"
}
```

## Output to the user

After the round:

1. STRATEGY and decision (`continue` / `killed` / `propose_accept`)  
2. Table: same-arm floor vs treatment  
3. Path to `ROUND.md` + `metrics.json`  
4. Production health proof (`is-active` + `/health`)  
5. **One** recommended next round hypothesis (single knob)

## Anti-goals

- Re-running full multi-seq PPL “just to check GDN” after decision.md already showed blindness  
- Quoting aggregate ABA without per-round arms  
- Reading profiler’s first cumulative window  
- Bare `wait` on background server  
- Leaving production down  

## Start now

1. Confirm production is up (or restore first if mid-failure).  
2. Declare `STRATEGY=A` unless the user overrode.  
3. Write `rounds/…/ROUND.md` with hypothesis **before** taking the GPU.  
4. Take one bench window with restore trap.  
5. Run the strategy procedure.  
6. Exit with restore verified.

---

## Owner one-liner (optional override)

Paste above the agent prompt if needed:

```
STRATEGY=A
Max rounds this session: 1
Do not propose GDN ship.
Budget: one GPU window ≤ 45 min.
```

For byte levers:

```
STRATEGY=C
Target: ffn_down_exps q5_K surgery only.
Max rounds: 1
```

For noise-model GDN (only if concurrent determinism abandoned):

```
STRATEGY=B
N_same_arm_pairs: 5
Require ratio >= 3
Still no ship without human ACCEPT
```
