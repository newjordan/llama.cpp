# Treebeard MoE Throughput R&D Ledger

This is the canonical execution ledger for closing the B70/Qwen MoE decode
throughput gap. It separates measured facts from hypotheses and requires every
optimization to pass correctness, production-shape activation, and guarded
same-binary performance gates.

## Fixed target and current evidence

- Model: Qwen3.6-35B-A3B Q5_K_XL, top-8 MoE.
- Device/runtime: Intel Arc Pro B70, SYCL.
- Serving shape: 12 slots, unified KV, 262144-token trained context.
- Current production service: `turbo-statetree-rc4.service` on port 8093.
- Observed generated throughput: about 79 tok/s at one active slot and about
  194 aggregate tok/s at 12 active slots.
- The naive 12x single-slot ceiling is not a performance claim. It only makes
  the large concurrency-efficiency gap visible; memory bandwidth, arithmetic,
  scheduling, and synchronization impose real shared limits.
- `moe-down-reduce` removes seven serial adds and one weighting multiply from a
  top-8 batched tail, but its 12-slot result is statistically flat. The guarded
  30-run confirmation measured -0.06% versus the A/B midpoint at p50 and
  +0.03% at mean. It is parked, not promoted.
- Evidence: `results/treebeard-moe-down-reduce/20260714-004421` and
  `results/treebeard-moe-down-reduce/20260714-010513`.

## Acceptance contract

Each candidate must satisfy all of the following before it can advance:

1. Preserve CPU-reference correctness across single-token and batched top-8
   shapes, including Q6_K and Q8_0 expert outputs.
2. Preserve the separate multiply rounding point unless an explicitly approved
   numerical contract replaces it.
3. Prove the exact production graph activated the intended path with bounded
   telemetry. A compiled kernel or synthetic-only hit is insufficient.
4. Pass overlap, alias, fallback, and disable-control cases without relying on
   allocator luck.
5. Run same-binary A/B/A with identical runtime and model hashes, a healthy
   device, no reset/fault/Level Zero error, and automatic restoration of the
   production service.
6. Advance only on repeatable wall-throughput improvement. Kernel-time wins
   that do not move end-to-end serving throughput are recorded and parked.

## Ordered task list

### T0 - Measurement and guardrails

- [x] Build a same-binary A/B/A serving harness with production identity checks,
  health checks, hardware fault checks, and automatic service restoration.
- [x] Add focused CPU-reference fixtures for interleaved, views-first,
  single-token, and forced routing-weight/output alias graphs.
- [x] Establish one-slot, eight-slot, and twelve-slot baselines.
- [ ] Add per-layer critical-path telemetry for MMID, routing weight/reduce,
  gate/up activation, down projection, queue gaps, and graph submission.
- [ ] Record bytes moved, kernel count, queue idle time, and occupancy alongside
  aggregate tok/s so later candidates can explain wins or losses.

### T1 - Fusion-aware liveness before allocation

State: complete as an enabling mechanism; standalone throughput gate failed and
the tail-only candidate is parked.

Hypothesis: the production allocator currently reuses the tiny routing-weight
buffer for the final ADD output. The fused runtime must snapshot those weights,
which erases part of the intended launch/memory saving. SYCL already has a
backend graph-optimization hook before graph allocation.

- [x] Recognize the production views-first top-8 tail in SYCL graph optimization.
- [x] Add a late, non-compute dependency from the final reduction output to the
  routing weights so the existing allocator keeps them live.
- [x] Share the runtime disable control and make annotation idempotent.
- [x] Retain the runtime overlap check and snapshot as a defensive fallback.
- [x] Prove the production graph emits `liveness-annotated` and an integrated hit
  without `batched-weights-snapshot`.
- [x] Pass focused correctness and a guarded twelve-slot A/B/A.

Decision: retain the liveness mechanism because T2 and T3 require their late
inputs to survive allocation. The standalone ten-repeat twelve-slot result was
+0.87% p50 and +0.74% mean versus the A/B midpoint, below the +1.0% gate.
Evidence: `results/treebeard-moe-down-reduce/20260714-110808`.

Perf gate: at least +1.0% p50 and mean versus the A/B midpoint over a minimum
of ten repeats, followed by the full 1/8/12 gate. A flat result is evidence that
the reduction tail is not material on the decode critical path.

### T2 - Accumulate routing weights inside down-projection MMID

State: implemented, production-activated, and advanced as part of the combined
T2+T3 candidate.

Hypothesis: eliminating the expert-output tensor and its write/read traffic is
materially larger than fusing only the tail reduction.

- [x] Specify an ordered top-8 accumulation contract that preserves the current
  multiply rounding and expert order.
- [x] Add an MMID epilogue that multiplies each selected expert result by its
  routing weight and accumulates directly into the token output.
- [x] Avoid materializing the `[n_embd, n_expert_used, n_token]` intermediate.
- [x] Cover aliasing, production Q6_K/Q8_0 weights, batched token counts through
  64, and the unfused fallback. Single-token decode retains its existing
  ordered two-kernel path.
- [ ] Measure eliminated bytes and launches as well as wall throughput.

Activation evidence: the guarded production graph emitted
`batched-integrated-hit` at 2, 12, and 48 active tokens with no materialized
fallback or routing-weight snapshot. Evidence:
`results/treebeard-moe-down-reduce/diagnostics/20260714-135228`.

Combined performance decision: advance. At 12 agents the T2+T3 candidate
measured +3.73% p50 and +3.80% mean versus the A/B control midpoint. At one
agent it measured +1.73% p50 and +1.79% mean. Evidence:
`results/treebeard-moe-combined/20260714-135527`.

Perf gate: +2.0% twelve-slot aggregate throughput with no one-slot regression
larger than 1.0%.

### T3 - Fused gate/up MMID plus SwiGLU epilogue

State: implemented, production-activated, and advanced with T2.

Hypothesis: the separate gate and up expert matrices share routed activations
and ids, so matching rows can feed a SwiGLU epilogue directly. This avoids both
full MMID outputs and a separate activation launch.

- [x] Confirm exact row pairing and quantized block layout for every production
  expert tensor variant.
- [x] Fuse gate and up dequant/dot results into SwiGLU before global storage.
- [x] Store only the `n_ff` activated result consumed by the down projection.
- [x] Preserve a shape/type fallback, allocator liveness, overlap guards, and
  tail-dimension validation.
- [x] Benchmark combined with T2 under the guarded same-binary gate.

Production coverage is 39 Q5_K gate/up pairs and one Q6_K pair. CPU-reference
tests pass direct single-token and grouped 12-token paths for both types. The
guarded real graph emitted `dual-integrated-hit` for Q5_K and Q6_K at 2, 12,
and 48 active tokens with no decode-shape fallback or overlap. Evidence:
`results/treebeard-moe-down-reduce/diagnostics/20260714-135228`.

Combined performance decision: advance. Ten repeats per arm at one and twelve
agents passed correctness, activation, identity, restoration, and hardware
fault gates. The 12-agent gain was +3.73% p50 and +3.80% mean; the one-agent
gain was +1.73% p50 and +1.79% mean. Evidence:
`results/treebeard-moe-combined/20260714-135527`.

Perf gate: +2.0% twelve-slot throughput alone, or a repeatable incremental win
of at least +1.0% on top of T2.

### T4 - Twelve-column MMVQ reuse and occupancy

State: current reuse variants implemented and parked; no shape sweep follows.

- [x] Retain the prior exact-12-column geometry and Q8 reuse evidence; explicit
  Q8 weight/scale reuse regressed the model-backed dense shapes.
- [x] Implement grouped ordered down projection with cross-token expert reuse,
  workgroup-local contributions, volatile multiply rounding, and slot-ordered
  accumulation.
- [x] Pass 7/7 CPU-reference cases and activate the grouped Q6 path on the real
  graph at 12 and 48 tokens.
- [x] Run an isolated same-binary A/B/A without disabling grouped gate/up.

Decision: park. The grouped ordered-down candidate measured +0.06% p50 at one
agent and -0.26% p50 at twelve agents; means were +0.01% and -0.10%. Direct
ordered down remains default and the grouped kernel is retained opt-in for use
by a larger composite pipeline. Evidence:
`results/treebeard-moe-down-grouped/20260714-144032`.

Perf gate: reduce production MMID critical-path time enough to yield at least
+2.0% twelve-slot end-to-end throughput.

### T5 - Whole-decode submission amortization

State: implemented, production-activated, and parked.

- [x] Re-evaluate SYCL command-graph capture for stable decode shapes.
- [x] Define invalidation for scheduler graph UID, split identity, and stale
  cache lifetime.
- [x] Add side-effect-free lazy-reorder preparation so capture never executes
  aliased or stateful graph outputs twice.
- [x] Prove record/replay of nonzero UIDs on the production 3,726-node graph.
- [x] Compare capture/replay with ordinary in-order submission under the exact
  service lifecycle.

Decision: park. Cached replay regressed one-agent p50 by 4.57% and twelve-agent
p50 by 4.24%; means regressed 4.59% and 4.30%. The cache implementation remains
available behind the existing graph opt-in, but production stays disabled by
default. Evidence: `results/treebeard-sycl-graph/20260714-142617`.

### T6 - Composite MoE activation/down pipeline

State: implemented, production-activated, and parked.

- [x] Match the complete separate gate/up MMID -> SwiGLU -> down MMID ->
  ordered top-eight reduction subgraph.
- [x] Quantize SwiGLU results directly into the reordered Q8_1 layout consumed
  by down projection instead of materializing an F32 GLU tensor.
- [x] Reuse one device routing pre-pass across gate/up and down kernels.
- [x] Preserve CPU-reference numerics, allocator liveness, overlap guards, and
  Q8/non-production fallbacks.
- [x] Gate only the complete pipeline on the production 12-token graph.

Decision: park. The full graph passed its production-shaped CPU reference and
activated on the real 12- and 48-token graphs for reordered K-quant down
layers; Q8_0 down layers retained the accepted fallback. In guarded
same-binary A/B/A, one-agent p50/mean changed -0.29%/-0.31%, while twelve-agent
p50/mean regressed 6.65%/6.83%. The eliminated global F32 activation and
quantization submission did not compensate for reduced row-level parallelism
in the grouped composite topology. Keep it opt-in and do not sweep geometry.
Evidence: `results/treebeard-moe-pipeline/activation-20260714-151357` and
`results/treebeard-moe-pipeline/20260714-151507`.

Perf gate: at least +2.0% twelve-agent aggregate throughput without a one-agent
regression larger than 1.0%.

### T7 - Continuous-batch shape control

State: queued after the composite MoE pipeline.

- [ ] Measure active-column distribution and shape churn under realistic request
  arrivals rather than forced steady 12-way decode alone.
- [ ] Test short admission/coalescing windows and decode-lane bucketing.
- [ ] Optimize aggregate throughput subject to explicit first-token and per-token
  latency ceilings; do not hide batching delay inside tok/s.

Perf gate: improve workload-level completed tokens per second without worsening
p95 time-to-first-token or inter-token latency beyond the declared budget.

### T8 - Single-token ordered expert-down epilogue

State: production-activated and parked at the major-play entry gate.

- [x] Split opaque fused profile time into named fusion buckets.
- [x] Attribute the steady one-token graph and identify residual expert-down
  MMID as 16.9% of serialized time.
- [x] Preserve ordered routing multiply/reduction semantics across Q5_K, Q6_K,
  and Q8_0; pass 11/11 SYCL-vs-CPU fixtures.
- [x] Handle allocator reuse by snapshotting the eight routing weights and
  eight expert IDs before direct output.
- [x] Prove live production activation across about 40 MoE layers.

Decision: park and remove the experimental kernel/recognizer. Activation
collapsed residual `MUL_MAT_ID` from 38.2 to 1.2 calls/eval and moved
`FUSED_MOE_DOWN` from 2.6 to 39.6 calls/eval, but the identical 128-token
screen improved only from 36.7296 to 37.9101 tok/s (+3.21%). This misses the
10% entry gate; a concurrency sweep is not warranted. Retain only the named
profiling instrumentation. Evidence:
`reports/treebeard-single-token-moe-down-fusion-20260715.md` and
`results/treebeard-single-wavefront-b70/20260715-003634`.

## Decision order

Execute T1 first because it is the smallest architecture-enabling slice and
tests whether the current tail fusion was neutralized by allocation. If T1 is
flat, retain its graph/liveness mechanism only if it is required by T2 or T3;
otherwise park it. T2 and T3 are the first candidates expected to remove
material global-memory traffic. T4 through T7 are selected by measured
critical-path attribution, not by generic accelerator folklore.

Update this ledger after every gate with the exact result directory, measured
delta, activation evidence, and advance/park/reject decision.
