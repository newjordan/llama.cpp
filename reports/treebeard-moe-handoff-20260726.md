# Handoff: MoE-down is finished. One unshipped win, two open levers.

Status: ACTIVE. Written 2026-07-26 (end of the 2026-07-25 kernel sessions).
Branch `agent/treebeard-single-wavefront` @ `a7dc94ce0`, pushed to `turbo-private`.
Supersedes `reports/treebeard-moe-kernel-handoff-20260725.md` (marked RESOLVED).

Read sections 0 and 1 before touching anything. Section 1 exists because this
session published a conclusion that was an artifact of how a number was read.

---

## 0. State of the box

Production is UP and untouched by any of this work: `treebeard-b70-ship`
(:8093, Angel X), 12 slots, ctx 262144, `TREEBEARD_GDN_OUT_FLAT=0` pinned via
drop-in. Every experiment below took a bench window and restored it with
`results/treebeard-b70-gdn-out-flat-20260725/restore-and-verify.sh 0`.

Nothing was flipped into production this session. Three commits, all pushed:

| commit | what |
|---|---|
| `78284f41f` | `TREEBEARD_MOE_ROUTE_HIST` diagnostic; parked expert reuse |
| `753b4f9cc` | `TREEBEARD_MOE_DOWN_PHASE_PROF`; corrected the cost model |
| `a7dc94ce0` | deferred-reduce kernel; `TREEBEARD_MOE_ROUTE_FORCE`; quantize surgery |

---

## 1. Two methodology rules, both learned the hard way

**Rule 1 - read the per-op profiler's LAST cumulative window, never the first.**
`TREEBEARD_SYCL_PROF` reports cumulatively every 50 graph evals, so the first
report is warmup: JIT, cold caches, first-touch of ~30 GB of weights. It
overstated MoE-down by 55% (371.93 vs 239.86 us/op at np12, from the same file;
`serialized-total` falls 83990 -> 46902 us/eval across one run). A `grep -m1`
on that output produced a confident, wrong conclusion - a "~188 us/op fixed
cost" named as the top optimisation target - which survived a commit before a
second, independent instrument caught it. Correction notice:
`results/treebeard-moe-route-hist-20260725/SUPERSEDED.md`.

**Rule 2 - "batched-bench overstates production by 3.5x" is NOT universal.**
That rule came from the GDN fix (+54% screen -> +15% p50) and is real *for
changes whose benefit competes with context-scaling work*: at ctx 262144 the KV
and attention traffic dominates and dilutes them. The deferred-reduce change
removes a **context-independent per-op cost** and carried over intact
(+0.80% screen -> +0.67% p50). Ask which kind of change you have before
discounting a screen.

Corollary to both: when a lever measures flat, prove the code ran AND prove your
instrument is measuring what you think. Two instruments that agree
(phase-profiler 243.58 vs profiler 245.76 us/op, 0.9% apart) is what finally
made these numbers trustworthy.

---

## 2. What is now settled about MoE-down

**Cost model** (`results/treebeard-moe-down-fixed-cost-20260725/decision.md`):

    GEMV only  us/op =  21.2 + 2.4370*MB   marginal 410 GB/s   fixed  9.5% @np12
    whole op   us/op =  43.5 + 2.4449*MB   marginal 409 GB/s   fixed 17.7% @np12

Residuals <1.6 us over npl 2/4/8/12. `reorder` is flat at 14.3 us, `quantize`
flat at ~8.4 us. **MoE-down runs at ~82% of the ~500 GB/s roofline.** It is a
well-optimised memory-bound kernel; there is no overhead bug in it.

**Expert reuse is closed - with a measured ceiling, not a null.**
`TREEBEARD_MOE_ROUTE_FORCE=N` forces N distinct experts while holding the
(token,slot) pair count at 96. At distinct=1 the kernel reads 0.9 MB and still
takes 192.35 us, against 222.24 us at natural routing (39.54 distinct). So:

    ceiling on ANY reuse kernel = 222.24 - 192.35 = 29.9 us = 13.5% of the GEMV

~2.5% of decode for a *perfect, zero-overhead* kernel. The existing grouped
kernel spends more than that on its own overheads. It does beat per-pair at
distinct=8 (164.54 vs 193.67) and loses at distinct=96 (299.23 vs 260.88) -
crossover around 16-24 distinct, and this model sits at 39.5. **Do not revisit
expert reuse, grouped/par8/TCHUNK included.** Same reasoning kills dual reuse:
`FUSED_MOE_DUAL` at 207.22 us/op would need >roofline bandwidth to be reading
all its pair-bytes, so cache already serves it.

Family shares at np12 (last window, profiler-only run): MUL_MAT 28.4%,
FUSED_MOE_DOWN 20.4%, FUSED_MOE_DUAL 17.6%, FUSED_STATE_IO 8.4%. MoE = 38.0% of
serialized decode - the handoff's original premise stands, but that 38% is now
known to be near-roofline streaming.

---

## 3. The unshipped win - decide this first

`GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE=1` (default OFF), `mmvq.cpp`. Computes all
eight experts' per-lane partial sums, then the eight reductions, then the ordered
volatile sum. **Bit-identical by construction**: same per-expert dot order, same
`reduce_over_group`, same ordered accumulation - only the interleaving changes.

| gate | result |
|---|---|
| GEMV | **-4.9%** (222.95 -> 212.02 us, two reps each) |
| screen, ctx 12288 np12 | **+0.80%** end-to-end, control reps 0.04% apart |
| **12-agent ABA, ctx 262144** | **+0.67% p50/agent, +0.89% aggregate** |
| A-drift | **-0.35%** (gate <1%) |
| greedy parity, n_seqs>1 | **4/4 IDENTICAL** |
| `test-backend-ops -o MUL_MAT_ID` | NOT RUN - see below |

**Honest read:** +0.67% is only ~2x the measured drift, so a single ABA cannot
call it decisively. What makes it credible is that three independent
measurements agree in direction and magnitude, and parity is exact, so the
downside of shipping is essentially zero.

**To ship it** you still need: `test-backend-ops test -o MUL_MAT_ID -b SYCL0`
and `-o MUL_MAT` (not run this session - the change is bit-exact by construction
and passed 4/4 greedy parity, but the ladder asks for it); optionally a second
ABA to tighten the number; then packaging by pin-by-hash + copy-install
(`scripts/treebeard-rc5-package.sh`) and the unit flip with the rollback chain
intact. **This is a user decision, not an agent one** - it is a production
change worth <1%.

---

## 4. Open levers, ranked

### 4a. Byte levers - now unblocked, blocked only on a quality gate

`a7dc94ce0` patches `src/llama-quant.cpp` so an explicit `--tensor-type`
override survives `ftype COPY`. Previously `tensor_allows_quantization` bailed on
`only_copy` before consulting overrides, and COPY maps to `ALL_F32` so the normal
override path was unreachable anyway. Now:

    llama-quantize --allow-requantize --tensor-type ffn_down_exps=q5_K in.gguf out.gguf COPY

Dry-run validated: control converts **0** tensors, surgery converts exactly **40**.
It immediately proved its own necessity - **`blk.1.ffn_down_exps.weight` is q8_0
while the other 39 are q6_K**. Unsloth's UD mix is not uniform, so a whole-model
requantize would have silently moved that tensor too.

Targets: sub-Q6_K down tensor (bytes scale at 2.445 us/MB, so ~-15% on a family
worth 20.4%) and the Q6_K `result_output` head (540 MB Q8_0, 1477-1481 us/op).

**No model has been built yet, deliberately.** Two things to settle first:
1. Q6_K -> F32 -> Q5_K is *double* quantization, strictly worse than Q5_K built
   from source. The speed delta is honest; the quality delta is pessimistic.
2. Unsloth chose those types on purpose and CLAUDE.md makes quality the product.
   This needs a real capability gate, not a greedy diff - and **spark-bench is
   still not on this box**. Get a source or a substitute before spending GPU time.

### 4b. The ~192 us pair-count floor

Still 86% of the GEMV after deferred-reduce took ~10 us off it. Not bytes, not
occupancy, not reuse - all three are measured. It is 96 pairs x 2048 rows of a
**2-K-block** dot (ncols=512, QK_K=256), so the per-row subgroup reduction is
large relative to the arithmetic. Two questions worth asking:
- should one subgroup own more than one row, amortising the reduction?
- is the `volatile` ordered-sum contract worth its cost? Removing it is NOT
  bit-exact and needs the logprob gate
  (`results/treebeard-b70-gdn-out-flat-20260725/analyze-logprob.py`, calibration:
  shipped subgroup knob = 5.21e-05 nats, GDN fix = 7.03e-04).

### 4c. Do not reopen

Expert reuse (ceiling measured at 13.5%, grouped/par8/TCHUNK all inside it),
MMID workgroup sizing (flat over 8x), MoE pipeline (-13.6%), dual reuse
(cache already serves it).

---

## 5. Tools added this session

| env | what | cost |
|---|---|---|
| `TREEBEARD_MOE_ROUTE_HIST=1` | per-layer draws/distinct/top_expert_count | host-syncs ids; measurement build only |
| `TREEBEARD_MOE_DOWN_PHASE_PROF=1` | splits the op into reorder/quantize/GEMV | 3 stream waits per op; attribution only |
| `TREEBEARD_MOE_ROUTE_FORCE=N` | forces N distinct experts at fixed pair count | **DESTROYS OUTPUT** - perf probe only |
| `GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE=1` | the win in section 3 | none; bit-exact |

All default OFF. The phase profiler reports **window deltas**, not a running
mean, specifically so warmup cannot leak into later reports (Rule 1).

Harnesses: `results/treebeard-moe-reuse-probe-20260725/run-force-sweep.sh`;
`results/treebeard-moe-deferred-reduce-20260725/{run-ab.sh, parity-deferred.sh,
run-aba-deferred.sh}`. **`run-aba-deferred.sh`'s inline summary printer dies with
a SyntaxError** (escaping lost when adapting it from `run-aba.sh`) - the
`aba-*.json` files it writes are complete and are the actual evidence; parse
those and ignore the console summary.

---

## 6. The one-paragraph version

MoE-down is done: it runs at ~82% of roofline, its cost is `43.5 us + 2.445 us/MB`,
and expert reuse has a measured ceiling of 13.5% of the GEMV that no kernel in
that family can reach - stop trying. One bit-exact win is sitting unshipped
(`GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE=1`, +0.67% p50 on a real 12-agent ABA,
4/4 parity) and needs a `test-backend-ops` run plus a human decision to package
and flip. The remaining levers are fewer bytes - now unblocked by single-tensor
GGUF surgery, but gated on a capability eval this box does not have - and the
~192 us pair-count floor, which is a subgroup-reduction question, not a
bandwidth one. And whatever you measure, read the profiler's last window and
check whether your change competes with context-scaling work before you trust a
short-ctx screen.
