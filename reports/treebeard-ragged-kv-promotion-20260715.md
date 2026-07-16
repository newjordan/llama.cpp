# Treebeard ragged-KV + state-I/O promotion evidence (2026-07-15)

Campaign: promote sequence-ragged StateTree KV attention plus recurrent
state-I/O fusion from screened/confirmed research state to production
(turbo-statetree-0.1.0-rc.5). Ledger entry: T10 in
`docs/treebeard-throughput-rnd.md`.

## Frozen candidate

- Branch `agent/treebeard-single-wavefront`, runtime source frozen at
  `de0834ca0`; later commits are reports/scripts only (runtime dirs
  byte-identical, asserted by the packaging script).
- Accepted runtime chain: `81df9e2ff` ragged plan + indexed FATTN,
  `789dfe8ec` `LLAMA_KV_TREE_RAGGED` gate (default 1), `827a4f007` SIQ
  profiler, `76befe8c8` state-I/O fusion (default on), `a1d92fed7` Q8 ncols
  weight-hoist (default off).
- Clean rebuild in `build-treebeard-single-wavefront`; llama-server sha256
  `393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c`
  (full freeze record:
  `results/treebeard-ragged-promo-b70/build-freeze-20260715.md`).
- One binary served every arm; arms differ only in env:
  control `LLAMA_KV_TREE_RAGGED=0 GGML_SYCL_ENABLE_STATE_IO_FUSION=0`,
  ship `=1/=1`, hoist arm adds `GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=1`.

## Correctness before perf

- `test-tree-ragged-kv`: pass on the CPU build and inside the guarded window
  on the SYCL build.
- `test-backend-ops -b SYCL0 -o MOE_DOWN_REDUCE|MOE_DUAL_SWIGLU` with debug
  envs: pass with integrated-hit events; no perf arm emitted
  `batched-weights-snapshot` or materialized-fallback events.
- Exact branch token sha256 parity held across every fragmented arm and
  repeat (one distinct 6-hash set for c0-a/c1/c2/c3/c0-b x 3 repeats), so the
  ragged, state-io, and hoist paths are output-exact on this workload.

## Dense golden shape (1/8/12 agents, 10 repeats, seed 42)

Artifact: `results/treebeard-ragged-promo-b70/20260715-193609-golden-aba`
(`golden-summary.json`; superseded invalid attempt: `20260715-192041`,
guard-assert bug, arms intact).

| agents | control mid p50 | candidate p50 | delta p50 | delta mean | drift |
|---|---|---|---|---|---|
| 1  | 78.273  | 81.174  | +3.71%  | +3.69%  | 0.03% |
| 8  | 193.126 | 225.345 | +16.68% | +16.72% | 0.20% |
| 12 | 208.116 | 247.399 | +18.88% | +19.16% | 0.15% |

Attribution arm (stio-only, ragged=0/state-io=1): +3.70%/+16.88%/+20.06% —
the dense win is entirely state-I/O fusion, previously measured only as a
fragmented +8.1% increment. The ragged env is inert on dense (no fork
families; plan cannot reduce columns) and its per-ubatch plan scan costs
about 1% at 12 agents, far inside the -1% gate. The control midpoint matches
RC4 production (207.44 shipped) within 0.33%. Kernel fault scan empty;
production restored exactly after every window (`NRestarts=0`, identity and
exe digest compared).

## Fragmented multi-branch shape (screen geometry, 3 repeats, seed 1709)

Artifact: `results/treebeard-ragged-promo-b70/20260715-195819-confirm-comp-aba`
(`confirm-summary.json`). Geometry: ctx 262144, np 12, fanout 5, six 8192-token
persistent fragmentation fillers, 32768-token shared prefix, 64 branch tokens.

| arm | env | p50 tok/s | vs C0 midpoint |
|---|---|---|---|
| C0 midpoint (a/b) | both off | 74.501 | — |
| C1 | ragged only | 118.385 | +58.90% |
| C2 (ship) | ragged + state-io | 127.335 | +70.92% |
| C3 | C2 + q8 hoist | 128.757 | +72.83% |

Control drift 0.055%. State-io incremental over ragged: +7.56% (gate >= 5%).
C2 gate >= +55%: met. C1 gate >= +50%: met. Zero failed samples.

Q8 hoist (C3): +1.12% p50 over C2 — its preregistered >= 1.0% p50 gate
passes — but +0.18% mean over 3 repeats. Because every golden-shape
measurement ran hoist=0, RC5 ships `GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0`
and hoist adoption is deferred pending a hoist-on dense golden arm. The
activation trace `[treebeard-q8-hoist] activated` appeared in C3 only.

## Edge probes

250k long-prefix parity (ragged on/off) and commit-churn probes: first
attempt timed out at the 900s request ceiling (a 250k dense prefill takes
about 15 minutes); rerun in flight with a 3600s ceiling in edge-only mode.
This section is updated by the edge run's `edge-summary.json`.

## Ship decision

ADVANCE to turbo-statetree-0.1.0-rc.5 with ship env
`LLAMA_KV_TREE_RAGGED=1 GGML_SYCL_ENABLE_STATE_IO_FUSION=1` (+ RC4 base env,
hoist=0). Expected production deltas versus RC4: about +19% aggregate at 12
agents and +3.7% single-stream on dense serving, and about +71% aggregate on
fragmented multi-branch StateTree workloads. Rollback: re-enable
`turbo-statetree-rc4.service` (build `b9627-3fcf1c626`, exe sha
`211d4115...`).
