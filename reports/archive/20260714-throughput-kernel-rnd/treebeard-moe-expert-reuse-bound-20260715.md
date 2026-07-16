# Treebeard heterogeneous MoE expert-reuse bound - 2026-07-15

## Outcome

Park a new cross-token expert-as-lanes or workgroup-reuse down-projection
kernel for the current B70/Qwen3.6 production shape. Heterogeneous twelve-agent
routing leaves useful collisions, but even a deliberately impossible bound
places the end-to-end saving at only **8.43%**, below the active 10% major-play
entry gate.

Keep the opt-in occupancy sensor and heterogeneous probe. They add only a
disabled static branch in normal execution; `GGML_SYCL_MOE_REUSE_PROFILE=1`
deliberately serializes each grouped routing pre-pass and reads back its one
`n_active` integer. This is diagnostic instrumentation, not a performance
mode.

## Why the benchmark-clone shape was rejected

The existing throughput harness gives every concurrent slot the same prompt
and greedy continuation. That shape creates extreme cross-token expert reuse,
but it is not representative of independent production agents. An initial
diagnostic confirmed the distortion and was discarded: its speculative audit
also hit the already parked serial-verifier mismatch path and the server later
aborted on an invalid logits index. Production restored successfully, but no
correctness or performance claim uses that run.

The replacement probe sends twelve simultaneous, domain-diverse prompts with
speculation disabled. It requires all twelve prompt hashes and all twelve
64-token output hashes to be distinct.

## Clean guarded capture

- Artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260715-045647`
- Model: Qwen3.6-35B-A3B Q5_K_XL
- Device/runtime: Intel Arc Pro B70, SYCL
- Serving shape: 262144 context, 12 slots, 64 generated tokens per agent
- Prompt hashes: 12/12 distinct
- Output token hashes: 12/12 distinct
- Full 12-token layer samples: 2,520
- Probe result SHA-256:
  `755f215c1b93bc5a31898d85176f89955cd6ae4f9a4efee9383b0347b7a2383e`
- Server log SHA-256:
  `5e000c3781afb821584686e961911c6f0d8f84e5336d1b20a8b167f968ebf5cc`

The guard found no kernel reset, hang, fault, OOM, or panic signature. It
restored `turbo-statetree-rc4.service` as active/running at build
`b9627-3fcf1c626`, with the expected model alias, twelve slots, 262144 context,
and zero service restarts.

The reported 133.27 aggregate tokens/s is intentionally not a benchmark:
every grouped routing pre-pass blocks for a one-integer device readback.

## Occupancy result

Each full decode layer has 96 routes: twelve tokens times eight selected
experts out of 256. The clean full-shape distribution was:

| Metric | Distinct active experts | Ideal weight-read reduction |
| --- | ---: | ---: |
| Minimum | 15 | 84.38% |
| Median | 66 | 31.25% |
| p95 | 76 | 20.83% |
| Maximum | 86 | 10.42% |
| Mean | **60.1730** | **37.3198%** |

The reduction is `1 - active_experts / 96`. It is an ideal weight-read bound,
not an expected kernel speedup: real execution still pays routing, activation
reads, dot products, ordered accumulation, and launch overhead.

## End-to-end falsification bound

The named serialized profile at
`results/treebeard-single-wavefront-b70/20260715-001620` attributed 16.9% of
the 100-evaluation total to remaining `MUL_MAT_ID` work and 5.7% to
`FUSED_MOE_DOWN`. With the accepted paired gate/up path active, these are the
down-projection pool that a new cross-token down topology could address:

```text
down-projection share              = 16.9% + 5.7% = 22.6%
mean ideal expert weight reuse     = 37.3198%
impossible end-to-end upper bound  = 22.6% * 37.3198% = 8.4343%
```

This assumes all measured down-projection time is expert-weight bandwidth and
that every duplicate weight read disappears at zero cost. Both assumptions
favor the candidate. The real ceiling is lower, and the already implemented
grouped ordered-down topology measured -0.26% at twelve agents.

## Re-entry condition

Do not build another expert-reuse-only down kernel for this production shape.
Re-enter only with profile evidence for an independent saving that raises the
combined conservative end-to-end bound above 10%, or with a materially
different workload whose heterogeneous routing occupancy is measured rather
than inferred from cloned prompts.
