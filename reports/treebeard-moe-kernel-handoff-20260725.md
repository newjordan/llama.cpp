# Handoff: the MoE expert kernel. Explicit instructions.

Status: RESOLVED 2026-07-25. The section-2 number was measured (distinct = 39.54
of 96 draws, upside 2.57x -> GO), the section-3 investigation was carried out,
and the premise was falsified: the existing grouped reuse kernel provably
executes at np12 and ties, an 8x workgroup sweep is flat, and MoE-down's cost
scales with (token,slot) PAIR count, not with distinct-expert weight bytes.
Project parked with a mechanism. Verdict, evidence and the re-ranked residual:
`results/treebeard-moe-route-hist-20260725/decision.md`.
The new top item is the ~188 us/op FIXED cost in MoE-down (50% of the op at
np12) - an attribution problem, not a bandwidth one. Sections 5 (operating the
box) and 6 (residual) below remain accurate; section 3's build plan does not.

Original text follows.

Written 2026-07-25 after the GDN shape fix landed.
Branch `agent/treebeard-single-wavefront` @ `00dd22275`, pushed to `turbo-private`.
Evidence: `results/treebeard-b70-gdn-out-flat-20260725/` (119 files).

Read this whole file before touching a kernel. Sections 0-2 exist because this
session burned a full GPU window proving a plausible hypothesis was wrong, twice.

---

## 0. Why the target moved here

The gated-DeltaNet output projection was built 3D (`ne2 = n_seqs`), so the SYCL
mul_mat batch loop issued one dispatch per sequence and re-read the whole `ssm_out`
weight n_seqs times per layer. Fixed in `00dd22275`. That single change took
`linear_attn_out` from 652.64 -> 46.22 us/op and moved the wall.

np12 decode profile, before vs after (`TREEBEARD_SYCL_PROF=1`, `-npp 32 -ntg 512`,
500 evals; serialized total 66364 -> 46786 us/eval):

| family | before | after | us/op | n/eval |
|--------|-------:|------:|------:|-------:|
| MUL_MAT | 48.4% | 28.4% | 42.70 | 311.2 |
| **FUSED_MOE_DOWN** | 14.7% | **20.5%** | **239.77** | 39.9 |
| **FUSED_MOE_DUAL** | 12.8% | **17.7%** | **207.01** | 39.9 |
| FUSED_STATE_IO | 6.0% | 8.4% | 32.77 | 119.5 |

**MoE = 38.2% of np12 decode. That is the target.**

---

## 1. Two traps that already ate this session. Do not repeat them.

**Trap A - a null result can mean "the code never ran".**
The multi-column Q8 MMVQ subgroup knob measured dead flat across a 16x sweep
(sg=1 vs sg=16 -> +0.09%). The knob was fine; the path was starved because the GDN
dispatch storm dominated. The null was the *diagnostic*, not the answer.

**Trap B - `par8` is gated off at exactly the shape you care about.**
`ggml/src/ggml-sycl/mmvq.cpp:3269`:

```c
if (n_experts_used == 8 && n_tokens <= 2 && ggml_sycl_moe_down_par8_enabled()) {
```

`n_tokens <= 2`. At np12, `n_tokens = 12`, so **par8 never fires in production.**
The 2026-07-21 park ("flat") and this session's re-adjudication (-0.40%) both measured
a kernel that was structurally inert at the tested shape. **The par8 kernel has never
been benchmarked at the serving shape.** Its park verdict is not evidence.

Before you conclude anything about a lever, prove it executed. Add a one-shot
`fprintf(stderr, ...)` on first entry, or perturb a constant 16x and confirm the number
moves. If a 16x perturbation does nothing, the code is not running.

---

## 2. Measure this ONE number before writing any kernel

The default MoE-down path (`launch_mul_mat_vec_q_moe_weighted_reorder`,
`mmvq.cpp:3256`, non-par8 branch at ~3290) launches:

```c
block_nums((unsigned) n_tokens, 1, (unsigned) block_num_y);
```

One workgroup per (token, row-block). Each workgroup **serially loops the 8 routed
experts for its token**. There is no cross-token expert reuse: if tokens 3 and 7 both
route to expert 42, expert 42's weights are read twice.

At np12 there are `12 tokens x top-8 = 96` expert draws per layer from `n_expert=256`.
The entire upside of a reuse kernel is:

```
upside = 96 / (distinct experts actually touched per layer per step)
```

- If routing is near-uniform, distinct ~ 80-90 -> redundancy ~1.1x -> **no lever, stop.**
- If routing is skewed (popular experts), distinct ~ 40-50 -> redundancy ~2x ->
  up to ~half of 20.5% of decode is recoverable. **That is worth a kernel.**

**Nobody has measured this.** The roofline arithmetic in
`results/.../ADDENDUM-numerical-gate-and-next-wall.md` section 4 (DOWN ~344 GB/s,
DUAL ~487 GB/s) *assumed* a distinct count. Do not build on it.

### How to measure it

Instrument the ids buffer, not the kernel. In `ggml-sycl.cpp` where MUL_MAT_ID /
the fused MoE down path reads `ids_dev`, add an env-gated diagnostic
(`TREEBEARD_MOE_ROUTE_HIST=1`) that, every N-th layer-step, copies the ids tensor to
host and prints:

```
[treebeard-moe-route] layer=%d n_tokens=%d draws=%d distinct=%d top_expert_count=%d
```

Run at the serving shape and histogram it:

```bash
# GPU window first - see section 5
TREEBEARD_MOE_ROUTE_HIST=1 llama-batched-bench ... -npl 12 -npp 32 -ntg 512
```

Report mean/p90 `distinct` across layers and steps. **That number decides whether
this project happens.** If distinct/draws > 0.85, write it up as a park and go to
section 6 instead.

---

## 3. If the number says go: what to build

A reuse kernel for MoE down (and the same shape for dual). The pre-pass already
exists and is the right primitive - `k_mmid_group_pairs` in `mmvq.cpp` does
SLM histogram -> prefix sum -> active-expert compaction -> pair scatter, landed in
`cd395a152`. `mul_mat_vec_q_moe_grouped_reorder` consumes it and "reads each distinct
expert once, dots all its routed tokens per weight pass" (TCHUNK=4).

That is **exactly the kernel this section would ask you to write.** It exists.
It is gated `GGML_SYCL_ENABLE_MOE_DOWN_GROUPED` and measured -0.02% this session.

So the real work is not "write a reuse kernel". It is:

1. **Find out why the existing grouped path does not win.** Confirm it executes at
   np12 (Trap A discipline). If it executes and still ties, the weight traffic is not
   the bottleneck and the whole premise is wrong - go to section 6.
2. If it does not execute at np12, find the gate (there is likely an `n_tokens` or
   `ne12` condition like par8's) and remove it.
3. If it executes but TCHUNK=4 is wrong for 12 tokens, sweep TCHUNK (4/8/12/16) -
   at np12 you want one pass to cover all routed tokens for an expert.

Only after 1-3 fail should you write something new. Candidate if you get there:
one workgroup per **distinct expert**, holding the expert's weight block in registers/SLM
while iterating that expert's routed (token, row) pairs from the compacted pair list.

Product shape to design against: down `nrows=2048, ncols=512` (2 QK_K blocks) Q6_K,
`top-k=8`, `n_as=256`; dual gate/up `[2048, 512, 256]` Q5_K. WARP_SIZE=**16** on this
build (`GGML_SYCL_WARP_SIZE=16`, Intel target). B70: 256 CUs, max WG 1024, so
num_subgroups*16 <= 1024.

---

## 4. Gates it must pass

1. `test-backend-ops test -o MUL_MAT_ID -b SYCL0` and `-o MUL_MAT` - both OK.
2. Activation proof: a first-entry trace showing the new path ran at `n_tokens=12`.
3. **12-agent ABA at production shape** - `scripts`-adjacent harness lives at
   `results/treebeard-b70-gdn-out-flat-20260725/aba-12agent.py` + `run-aba.sh`.
   A/B/A, require A-drift < 1%. **This is the number that counts.**
4. Numerical gate if the kernel changes accumulation order:
   `results/.../analyze-logprob.py`. Compare mean KL/position against the calibration
   reference in that dir (shipped subgroup knob = 5.21e-05 nats; GDN fix = 7.03e-04).
5. Record a decision.md in a new `results/` dir. Never edit existing evidence.

### The methodology rule this session established

**`llama-batched-bench` at short ctx overstates the production win by ~3.5x.**
The GDN fix measured **+54%** at ctx 12288 and **+15.0% p50 / +10.3% aggregate** at
the real shape (ctx 262144, 12 concurrent clients, A-drift -0.37%). Use batched-bench
to *screen* cheaply (~28 s/run); use the 12-agent ABA to *claim*. Never quote a
batched-bench number as a product number.

---

## 5. Operating the box

Angel X (`treebeard-b70-ship.service`, :8093) is the live surface. To take a window:

```bash
U=~/.config/systemd/user/treebeard-b70-ship.service
systemctl --user stop treebeard-b70-ship
mv "$U" "$U.bench-window"; ln -s /dev/null "$U"
systemctl --user daemon-reload          # symlink-mask; plain stop is not enough
```

Restore with `results/treebeard-b70-gdn-out-flat-20260725/restore-and-verify.sh`
(arg `0` pins `TREEBEARD_GDN_OUT_FLAT=0`, no arg = built-in default).
Only `treebeard-b70-ship` is an enabled GPU unit; every other model unit is disabled
and no model healthcheck timers are armed. `mempalace-embed` on :8091 is CPU-only
(`-ngl 0`) - leave it.

Build (incremental, ~30 s for one TU; cwd must NOT be the worktree):

```bash
source /opt/intel/oneapi/setvars.sh --force
cmake --build /home/frosty40/turbo/treebeard-work/build-treebeard-single-wavefront -j 16
```

Ship env for any bench arm:
`GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ENABLE_MOE_PIPELINE=0
GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
ZE_AFFINITY_MASK=0`, flags `-ngl 99 -ncmoe 0 --no-op-offload -fa on -ctk f16 -ctv f16
-b 8192 -ub 1024 -t 15`.

### Shell traps that cost real time here

- `set -u` + `source setvars.sh` kills the script. Wrap: `set +u; source ...; set -u`.
- A bare `wait` also waits on the llama-server you backgrounded, which never exits.
  Collect curl PIDs and `wait "$p"` each.
- `pgrep -f llama-batched-bench` matches your own `bash -c` wrapper, so an
  `until ! pgrep -f ...` loop never exits. Use `ps aux | grep "[l]lama-batched-bench"`.
- A result JSON is written at process exit; polling mid-run shows 0 bytes. Not a bug.
- Kill a stuck bench server by PID. Never `pkill` - it can hit managed units.

---

## 6. If the MoE premise collapses

Ranked residual, honest about what is measured vs assumed:

1. **Ship the GDN fix.** +15% p50 at production shape, ABA-clean. Blocked only on a
   capability eval. **spark-bench is NOT on this box** (`find /` finds nothing) and
   CLAUDE.md bars external network - get a source or a substitute quality gate.
2. **`result_output` Q6_K head.** 540 MB Q8_0 head, 1478.9 us/op, ~365 GB/s vs a
   ~500 GB/s roofline; ~19% of per-token weight traffic at np1. Q6_K is -23% on that
   tensor. Blocked on tooling, not permission: no f16/bf16 source locally (only
   Q5_K_XL and Q8_K_XL GGUFs), and `llama-quantize` would requantize Unsloth's whole
   per-tensor mix and confound the arm. Needs GGUF surgery on one tensor.
3. **FUSED_STATE_IO**, 8.4% over 119.5 ops/eval (~4 per linear-attn layer):
   gather/scatter for `SSM_CONV` + `GATED_DELTA_NET` recurrent state. Est. ~1.5 GB/step
   of state traffic at ~385 GB/s - likely near bound and intrinsic. Rank last.
4. **Re-sweep the multi-col Q8 subgroup knob** (`GGML_SYCL_Q8_MMVQ_NCOLS_SUBGROUPS`)
   *after* any change that alters which kernel serves `ne11=12`. It is plumbed and
   default 16 (behavior-preserving). Currently null, but null-because-starved has
   already happened once.

---

## 7. The one-paragraph version

MoE is 38.2% of np12 decode. Before writing a kernel, measure how many *distinct*
experts a layer actually touches per step at np12 - if it is close to 96 of 96 there
is no reuse to win and this project is dead. If it is closer to 45, the reuse kernel
you want already exists (`mul_mat_vec_q_moe_grouped_reorder` + `k_mmid_group_pairs`)
and the job is to find out why it does not fire or does not pay, starting with whether
it executes at all at `n_tokens=12` - because `par8` right next to it is hard-gated to
`n_tokens <= 2` and its "park" verdict was measured on a kernel that never ran.
Screen with batched-bench, claim only with the 12-agent ABA, and remember that the
short-ctx harness overstated the last win by 3.5x.
