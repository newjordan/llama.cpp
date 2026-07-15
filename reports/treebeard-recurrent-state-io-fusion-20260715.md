# Recurrent state-I/O fusion on Intel B70 (2026-07-15)

## Decision

Keep the implementation as an opt-in, correctness-preserving fusion with
`GGML_SYCL_ENABLE_STATE_IO_FUSION=1`. It clears the bounded-entry gate and is a
useful building block, but it does not yet clear the active goal's major-play
acceptance threshold of 20% at 12 agents.

The accepted result is:

- exact serial top-1 token parity across 12 heterogeneous prompts;
- exact natural-admission 12-agent token parity across all 12 streams;
- exact fixed-admission 12-agent token parity across all 12 streams;
- +11.66% natural-admission aggregate decode throughput;
- +10.58% fixed-admission aggregate decode throughput;
- +3.11% serial aggregate throughput.

## Profile and graph evidence

The no-speculation serialized SIQ profile is in
`results/treebeard-single-wavefront-b70/20260715-051036`.

Across 50 graph evaluations, the removable recurrent state movement was:

- Gated Delta Net state GET_ROWS plus copy-back: approximately 6.6% of the
  serialized profile;
- convolution-state GET_ROWS, CONCAT, and copy-back: approximately 5.0%;
- combined ideal state-I/O bound: approximately 11.6% while retaining the
  GDN and SSM math.

The exact graph topology is captured in
`results/treebeard-single-wavefront-b70/20260715-051445`. The pattern repeats
for all 30 recurrent layers: a cache-row gather feeds each math kernel, CONCAT
materializes convolution history plus the current token, and separate CPY
nodes write the next recurrent states back to cache.

## Implementation

Graph optimization recognizes only the exact autoregressive pattern (`K=1`,
one token per sequence, F32 recurrent state, one state writeback). It extends
allocator liveness through spare source edges and dispatches direct-cache SSM
and GDN kernels.

The kernels:

- consume recurrent cache rows through the device copy-map;
- write next convolution and delta states directly to the destination cache;
- eliminate the convolution CONCAT and both main state copy-back operations;
- replace each full state gather with a conflict-only snapshot prepass.

The safety invariant is conservative: only a true in-place mapping
`source_row == destination_row_for_this_sequence` bypasses staging. Every
other mapping snapshots the complete source row before any cache writes. This
preserves gather semantics for new slots, reordered slots, shared sequences,
and sources outside the local destination window while making steady decode a
zero-state-copy path.

Controls:

- `GGML_SYCL_ENABLE_STATE_IO_FUSION=1` enables the fusion;
- `GGML_SYCL_STATE_IO_MODE=all|ssm|gdn` permits diagnostic isolation;
- `GGML_SYCL_STATE_IO_DEBUG=1` emits bounded graph/activation traces;
- `TREEBEARD_SERVER_ADMISSION_HOLD_MS` is a diagnostic-only server queue hold
  used to compare identical admission batches.

## Correctness sequence

The first host-side copy-map gate remained inactive because the backend
scheduler correctly places the indices on the device. The first device-safe
prototype then exposed two issues under the guarded rig:

- `20260715-053119`: a transposed-token stride assertion, fixed before any
  result was accepted;
- `20260715-053331` and `20260715-053537`: repeatable divergence in late slots
  10 and 11 despite a +10.9% result.

Serial A/B proved the math was exact, and SSM/GDN isolation traced the batched
failure to an insufficient conflict window. The initial prepass staged only
sources inside the current reorder destination interval. Late slots can source
a row outside that interval. Changing the condition to stage every non-in-place
mapping restored exact parity without affecting steady decode.

## Final A/B results

All runs used the same exact model, no speculative decoding, 52 generated
tokens per prompt, temperature 0, top-k 1, and the guarded service restoration
path.

### Serial top-1 anchor

- candidate: `20260715-053754`, 65.2022 aggregate tok/s;
- baseline: `20260715-053928`, 63.2382 aggregate tok/s;
- lift: +3.1056%;
- parity: 12/12 token hashes identical.

### Fixed-admission batched verification

- baseline: `20260715-054358`, 119.0731 aggregate tok/s;
- final candidate: `20260715-055137`, 131.6681 aggregate tok/s;
- lift: +10.5775%;
- parity: 12/12 token hashes identical.

### Natural-admission production-shaped verification

- baseline: `20260715-052501`, 124.9098 aggregate tok/s;
- final candidate: `20260715-055319`, 139.4712 aggregate tok/s;
- lift: +11.6575%;
- parity: 12/12 token hashes identical.

The targeted existing SYCL SSM_CONV and GATED_DELTA_NET suite passed 81/81.
Every guarded run restored `turbo-statetree-rc4.service`; the final service
state was active/running with `NRestarts=0` and the expected production binary,
alias, context, and idle slot/state checks.

## Next gate

Do not default-enable this lane yet. Combine it with the next profile-proven
bandwidth/graph play and require the aggregate to reach at least +20% at 12
agents with the same serial-first and exact fixed/natural batched parity gates.
