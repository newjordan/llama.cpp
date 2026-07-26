# Handoff: MoE-down is closed. One win shipped, two levers open.

Status: ACTIVE. Written 2026-07-26, covering the 2026-07-25/26 kernel sessions.
Branch `agent/treebeard-single-wavefront` @ `056a62a09`, pushed to `turbo-private`.
Supersedes `reports/treebeard-moe-kernel-handoff-20260725.md` (marked RESOLVED;
its absolute us/op figures are warmup-inflated - see Rule 1).

Read sections 0 and 1 first. Section 1 exists because these sessions published a
confident conclusion that was an artifact of how a number was read.

---

## 0. State of the box

Production is UP and healthy: `treebeard-b70-ship` (:8093, Angel X), 12 slots,
ctx 262144, `TREEBEARD_GDN_OUT_FLAT=0` pinned via drop-in. Every experiment took
a bench window (symlink-mask; a plain stop is not enough, the fleet babysitter
force-starts it) and restored with
`results/treebeard-b70-gdn-out-flat-20260725/restore-and-verify.sh 0`.

**One change is now live in production**: the deferred-reduce MoE-down kernel is
the default as of 2026-07-26 (section 3). It is bit-exact, so production *output*
is unchanged; only speed moved.

**`treebeard-b70-ship.service` execs `build-treebeard-single-wavefront/bin/llama-server`
directly.** There is no copy-install between the build tree and production for
this unit, so **rebuilding that tree changes production on the next restart.**
Know that before you build. (The unit's alias still reads `...-7fffe5cb2-...`, a
build hash stale since 2026-07-21 - cosmetic, deliberately left alone because
renaming churns the surface Angel X registers. Worth a future packaging pass.)

Commits, all pushed:

| commit | what |
|---|---|
| `78284f41f` | `TREEBEARD_MOE_ROUTE_HIST`; parked expert reuse |
| `753b4f9cc` | `TREEBEARD_MOE_DOWN_PHASE_PROF`; corrected the cost model |
| `a7dc94ce0` | deferred-reduce kernel; `TREEBEARD_MOE_ROUTE_FORCE`; quantize surgery |
| `fda833103` | first version of this handoff |
| `056a62a09` | **defaulted deferred-reduce ON** + ship gates |

---

## 1. Three rules, all learned the expensive way

**Rule 1 - read the per-op profiler's LAST cumulative window, never the first.**
`TREEBEARD_SYCL_PROF` reports cumulatively every 50 graph evals, so the first
report is warmup: JIT, cold caches, first-touch of ~30 GB of weights. It
overstated MoE-down by 55% (371.93 vs 239.86 us/op at np12 *from the same file*;
`serialized-total` falls 83990 -> 46902 us/eval over one run). A `grep -m1` on
that output produced a wrong conclusion - a "~188 us/op fixed cost" named as the
top optimisation target - that survived a commit before a second, independent
instrument caught it. Notice:
`results/treebeard-moe-route-hist-20260725/SUPERSEDED.md`.

**Rule 2 - "batched-bench overstates production by 3.5x" is NOT universal.**
That came from the GDN fix (+54% screen -> +15% p50) and holds *for changes whose
benefit competes with context-scaling work*: at ctx 262144 KV/attention traffic
dominates and dilutes them. Deferred-reduce removes a **context-independent
per-op cost** and carried over intact (+0.80% screen -> +0.68% p50). Ask which
kind of change you have before discounting a screen.

**Rule 3 - check an ABA's per-round arms before quoting a delta.** The `rounds`
array is in the JSON. The aggregate metric is far noisier than p50: on the
shipped change, aggregate's within-arm A-to-A spread was 1.8%, twice the effect,
and an apparent "+0.89% aggregate" was entirely A2 running low. That figure was
published once and retracted. p50 is the metric that separates on this box.

Corollary: when a lever measures flat, prove the code ran AND prove your
instrument measures what you think. Two instruments agreeing (phase profiler
243.58 vs profiler 245.76 us/op, 0.9% apart) is what finally made these numbers
trustworthy.

---

## 2. What is settled about MoE-down

**Cost model** (`results/treebeard-moe-down-fixed-cost-20260725/decision.md`):

    GEMV only  us/op =  21.2 + 2.4370*MB   marginal 410 GB/s   fixed  9.5% @np12
    whole op   us/op =  43.5 + 2.4449*MB   marginal 409 GB/s   fixed 17.7% @np12

Residuals <1.6 us over npl 2/4/8/12. `reorder` flat at 14.3 us, `quantize` flat
at ~8.4 us. **MoE-down runs at ~82% of the ~500 GB/s roofline** - a well-optimised
memory-bound kernel with no overhead bug in it.

**Expert reuse is closed, with a measured ceiling rather than a null.**
`TREEBEARD_MOE_ROUTE_FORCE=N` forces N distinct experts while holding the
(token,slot) pair count at 96. At distinct=1 the kernel reads 0.9 MB and *still*
takes 192.35 us, against 222.24 us at natural routing (39.54 distinct):

    ceiling on ANY reuse kernel = 222.24 - 192.35 = 29.9 us = 13.5% of the GEMV

~2.5% of decode for a *perfect, zero-overhead* kernel; the existing grouped
kernel spends more than that on its own overheads. Grouped does beat per-pair at
distinct=8 (164.54 vs 193.67) and loses at distinct=96 (299.23 vs 260.88) -
crossover around 16-24 distinct, and this model sits at 39.5. It is not a wrong
kernel, it is aimed at a redundancy regime this model does not occupy.

Same reasoning kills dual reuse: `FUSED_MOE_DUAL` at 207.22 us/op would need
>roofline bandwidth to be reading all its pair-bytes, so cache already serves it.

Family shares at np12 (last window, profiler-only run): MUL_MAT 28.4%,
FUSED_MOE_DOWN 20.4%, FUSED_MOE_DUAL 17.6%, FUSED_STATE_IO 8.4%. MoE = 38.0% of
serialized decode - the original premise stands, but that 38% is now known to be
near-roofline streaming, not fixable overhead.

---

## 3. Shipped: deferred reduction is the MoE-down default

`mmvq.cpp`, `mul_mat_vec_q_moe_weighted_reorder`. Computes all eight experts'
per-lane partial sums, *then* the eight subgroup reductions, *then* the ordered
volatile sum. The old path put a dependent `reduce_over_group` + leader-only
volatile accumulate between each expert and the next, so the eight experts' loads
could not all be in flight.

**Bit-identical by construction**: same per-expert dot order, same
`reduce_over_group` per expert, same ordered accumulation. Only interleaving
changed. That is why it could ship without a numerical gate.

| gate | result |
|---|---|
| GEMV | **-4.9%** (222.95 -> 212.02 us, two reps each, non-overlapping) |
| screen, ctx 12288 np12 | **+0.80%** end-to-end, control reps 0.04% apart |
| 12-agent ABA, ctx 262144 | **+0.68% p50/agent**, A-drift -0.35% |
| `test-backend-ops -o MUL_MAT_ID -b SYCL0` | **721/721**, exit 0, 2/2 backends |
| `test-backend-ops -o MUL_MAT -b SYCL0` | **921/921**, exit 0, 2/2 backends |
| greedy parity, n_seqs>1, post-rebuild | **4/4 IDENTICAL** |
| production smoke | health ok, 12 slots, ctx 262144, sane completion |

The p50 result separated cleanly: every B round (24.715-24.809) above every A
round (24.508-24.673), 9 rounds, no overlap (p ~ 0.012 one-sided), and above the
~24.587 linear-drift expectation that the A/B/A design exists to remove. Do NOT
quote the aggregate number - see Rule 3.

**Rollback** - `GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE=0` restores the serialized
chain, and is also the control arm for any future A/B:

    mkdir -p ~/.config/systemd/user/treebeard-b70-ship.service.d
    printf '[Service]\nEnvironment=GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE=0\n' \
      > ~/.config/systemd/user/treebeard-b70-ship.service.d/20-deferred-reduce.conf
    systemctl --user daemon-reload && systemctl --user restart treebeard-b70-ship

Because both paths are bit-identical, rollback is a pure performance revert with
no output change.

**Do not headline this.** It is ~0.7%, perceptible to nobody. It was taken
because it is free and risk-free, on the reasoning that small bit-exact wins
compound. A second ABA was deliberately not run: the p50 separation was already
clean and the change bit-exact, so it would have cost a GPU window without
changing the decision. Record: `SHIPPED.md` in the deferred-reduce results dir.

---

## 4. Open levers, ranked

### 4a. Byte levers - tooling unblocked, gated on a quality eval

`a7dc94ce0` patches `src/llama-quant.cpp` so an explicit `--tensor-type` override
survives `ftype COPY`. Previously `tensor_allows_quantization` bailed on
`only_copy` before consulting overrides, and COPY maps to `ALL_F32` so the normal
override path was unreachable anyway (hence the direct resolve in
`llama_tensor_get_type`). Now:

    llama-quantize --allow-requantize --tensor-type ffn_down_exps=q5_K in.gguf out.gguf COPY

Dry-run validated: control converts **0** tensors, surgery converts exactly **40**.
It immediately proved its own necessity - **`blk.1.ffn_down_exps.weight` is q8_0
while the other 39 are q6_K.** Unsloth's UD mix is not uniform, so a whole-model
requantize would have silently moved that tensor too and confounded the arm.

Targets: sub-Q6_K down tensor (bytes scale at 2.445 us/MB, so ~-15% on a family
worth 20.4%) and the Q6_K `result_output` head (540 MB Q8_0, 1477-1481 us/op).

**No model built yet, deliberately.** Settle two things first:
1. Q6_K -> F32 -> Q5_K is *double* quantization, strictly worse than Q5_K from
   source. The speed delta measured that way is honest; the quality delta is
   pessimistic, so a null result would not be conclusive.
2. Unsloth chose those types on purpose and CLAUDE.md makes quality the product,
   so this needs a real quality gate, not a greedy diff.
   **The gate is local - build it here, do not go looking for spark-bench.**
   spark-bench is a DGX Spark bench and is not the B70 quality gate; earlier
   handoffs wrongly cited "spark-bench is not on this box" as a blocker and that
   framing should be dropped. What this box actually has:
   - `llama-perplexity` (in the build) over a held-out corpus - the direct,
     sensitive measure of a quantization quality change, and the right primary
     gate for anything that alters weights.
   - `results/treebeard-b70-gdn-out-flat-20260725/analyze-logprob.py` - top-8
     logprob KL per position, with calibration already established (shipped
     subgroup knob 5.21e-05 nats, GDN fix 7.03e-04). Right for kernel changes
     that only perturb FP reduction order.
   - `held-out-probe/` (scenarios + runner.py) for behavioural spot checks.
   Perplexity plus the KL harness is a stronger gate for a weight change than a
   capability leaderboard anyway, because it is sensitive to small distributional
   shifts rather than to task-score flutter.

### 4b. The ~192 us pair-count floor

Still ~86% of the GEMV after deferred-reduce took ~10 us off it. Not bytes, not
occupancy, not reuse - all three measured. It is 96 pairs x 2048 rows of a
**2-K-block** dot (ncols=512, QK_K=256), so the per-row subgroup reduction is
large relative to the arithmetic. Two questions worth asking:
- should one subgroup own more than one row, amortising the reduction?
- is the `volatile` ordered-sum contract worth its cost? Removing it is NOT
  bit-exact and needs the logprob gate
  (`results/treebeard-b70-gdn-out-flat-20260725/analyze-logprob.py`; calibration:
  shipped subgroup knob = 5.21e-05 nats/position, GDN fix = 7.03e-04).

### 4c. Do not reopen

Expert reuse (ceiling measured at 13.5%; grouped, par8 and TCHUNK all sit inside
it), MMID workgroup sizing (flat over an 8x sweep), MoE pipeline (-13.6%), dual
reuse (cache already serves it).

Also still unshipped from the prior session and unrelated to MoE: the **GDN 2D
output projection** (+15% p50, real) remains unshipped because it is *not*
bit-exact - 1 greedy flip, mean KL 7.03e-04 nats, ~13.5x the most recent shipped
tuning. Production stays pinned `TREEBEARD_GDN_OUT_FLAT=0`. It needs the same
local gate as 4a (perplexity + the KL harness), NOT spark-bench - it is worth
+15% p50, far more than anything else open, so building that gate is the
highest-value unblock available.

---

## 5. Tools and harnesses

| env | what | cost |
|---|---|---|
| `TREEBEARD_MOE_ROUTE_HIST=1` | per-layer draws/distinct/top_expert_count | host-syncs ids; measurement build only |
| `TREEBEARD_MOE_DOWN_PHASE_PROF=1` | splits the op into reorder/quantize/GEMV | 3 stream waits per op; attribution shares only |
| `TREEBEARD_MOE_ROUTE_FORCE=N` | forces N distinct experts at fixed pair count | **DESTROYS OUTPUT** - perf probe only |
| `GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE` | section 3; **default ON**, `=0` to revert | none; bit-exact |

The first three default OFF. The phase profiler reports **window deltas**, not a
running mean, specifically so warmup cannot leak into later reports (Rule 1).

Harnesses: `results/treebeard-moe-reuse-probe-20260725/run-force-sweep.sh`;
`results/treebeard-moe-deferred-reduce-20260725/{run-ab.sh, parity-deferred.sh,
run-aba-deferred.sh}`; earlier `run-route-hist.sh` + `analyze-route-hist.py`,
`run-npl-scaling.sh` + `fit-cost-model.py`.

**`run-aba-deferred.sh`'s inline summary printer dies with a SyntaxError**
(escaping lost when adapting it from `run-aba.sh`) - the `aba-*.json` files it
writes are complete and ARE the evidence. Parse those; ignore the console output.

Shell traps that cost real time here (still true): wrap `source setvars.sh` in
`set +u`/`set -u`; a bare `wait` also waits on a backgrounded llama-server that
never exits, so collect curl PIDs; `pgrep -f llama-batched-bench` matches your own
`bash -c` wrapper, so use `ps aux | grep "[b]in/llama-batched-bench"`; result
JSONs are written at process exit, so mid-run polling shows 0 bytes; never
`pkill`; and `llama-cli` rejects `-no-cnv` (use `llama-completion`) - it spins a
progress animation into stdout and can produce a multi-GB junk file if redirected.

---

## 6. The one-paragraph version

MoE-down is closed. It runs at ~82% of roofline, its cost is
`43.5 us + 2.445 us/MB`, and expert reuse has a measured ceiling of 13.5% of the
GEMV that no kernel in that family can reach - stop trying. The one win in it
shipped on 2026-07-26 (deferred reduction, default ON, bit-exact, +0.68% p50,
rollback `GGML_SYCL_MOE_DOWN_DEFERRED_REDUCE=0`) and is worth ~0.7%, so take it
as compounding rather than headline. What is left is fewer bytes - now unblocked
by single-tensor GGUF surgery but gated on a capability eval this box does not
have, the same blocker holding the GDN 2D fix - and the ~192 us pair-count floor,
which is a subgroup-reduction question, not a bandwidth one. And whatever you
measure: read the profiler's last window, check the ABA's per-round arms, and ask
whether your change competes with context-scaling work before you trust a
short-ctx screen.
