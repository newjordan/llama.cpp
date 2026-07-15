# Treebeard serial commit-frontier B70 benchmark - 2026-07-15

## Decision

PARK exact-output speculative verification on the current B70 backend.

The serial commit-frontier oracle is exact only when it reconstructs and
commits target KV state serially after every speculative round. That path
passed the bounded CPU and B70 correctness gates, but regressed shallow B70
throughput by 49.43% to 67.63%. It cannot meet the required 20% same-answer
gain.

Committing wide-batch KV state is not safe. Per-round serial and batched top-1
vectors can match while their hidden state drifts, causing later output
divergence. Disabling all optimized SYCL paths did not restore parity, so the
problem is not isolated to the Treebeard MoE fusions.

The shallow falsification gate is decisive. No 32K, 128K, 256K, or 12-agent
matrix was run for a path already 49% or more below control.

## Implementation

The opt-in `speculative.serial_anchor` request mode now:

1. clones the sampler and evaluates the target transitions serially through
   the first serial draft rejection;
2. runs the normal batched verifier and records the first differing column and
   token pair;
3. restores the target, draft, prompt, and sampler checkpoint;
4. serially reconstructs the committed prefix even when all sampled tokens
   matched, so the next round begins from trusted state.

Response timings add commit-frontier audit totals, matched comparisons,
fallbacks, mismatch columns, and serial/batched mismatch token IDs. Both
wavefront harnesses reject inconsistent telemetry and aggregate mismatch
columns.

## CPU gate

The isolated Qwen3.5 0.8B Q8 CPU A/B/A smoke passed exact parity for all 12
requests at widths 0, 4, and 8. Proposal-bearing requests satisfied every
audit invariant; no-proposal requests emitted zero audit state.

Artifact:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/serial-frontier-cpu-smoke-r2-20260715.json
```

## B70 state-commit gate

Configuration: Qwen3.6-35B-A3B Q5_K_XL, 262K f16 unified KV, shallow 512-token
prompts, 256 generated tokens, `ngram-simple` proposals, widths 0/4/8, three
deterministic workloads, one A/B/A repeat, and no concurrency.

All six speculative waves exactly matched their bracketed width-zero controls.

| Workload | Width | Control tok/s | Candidate tok/s | Paired gain | Audit fallbacks | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured copy | 4 | 80.71 | 35.51 | -56.00% | 0 | yes |
| Structured copy | 8 | 80.71 | 39.48 | -51.09% | 1, column 7 | yes |
| Code edit | 4 | 80.56 | 26.08 | -67.63% | 0 | yes |
| Code edit | 8 | 80.56 | 26.82 | -66.70% | 0 | yes |
| Free prose | 4 | 80.57 | 37.82 | -53.06% | 0 | yes |
| Free prose | 8 | 80.57 | 40.75 | -49.43% | 1, column 4 | yes |

The width-8 structured mismatch was `serial=290`, `batched=271` at column 7.
The width-8 prose mismatch was `serial=279`, `batched=211342` at column 4.

Artifact:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260714-235613
```

## Why token audit alone failed

An earlier version restored state only after a sampled-token mismatch. It
failed exact parity in four of six speculative waves. Most importantly, every
one of 343 audited comparisons in the two prose candidates matched, but both
outputs still diverged. Wide-batch KV state had drifted without immediately
crossing a top-1 boundary; later serial checks began from that contaminated
state and therefore could not detect the original transition.

Diagnostic artifact:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260714-234959
```

## Generic-backend falsification

The deterministic prose probe was repeated with `GGML_SYCL_DISABLE_OPT=1`, no
serial oracle, width 8, and 128 generated tokens. It still failed exact parity.
The generic control ran at 41.54 tok/s and the nonexact candidate at 48.07
tok/s, a raw gain of only 15.72%. This is below the promotion threshold even if
the correctness failure were ignored.

Artifact:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260715-000002
```

## Safety and restoration

Every guarded maintenance window restored the private RC4 production service.
The final service is active with zero restarts, the expected build and model
alias, real inference passed, occupied/reserved slots are zero, live StateTree
families are zero, and no Xe/DRM reset, hang, fault, OOM, or kernel-panic
signature was found.

## Promotion consequence

Keep the serial commit-frontier implementation as an opt-in diagnostic oracle,
not a production speed path. Do not spend deep-context or concurrency rig time
on speculative verification until a backend change can make wide-batch state
serial-equivalent without serial reconstruction. The active optimization lane
now moves to profile-proven operator, graph, memory, and pipeline fusions.
