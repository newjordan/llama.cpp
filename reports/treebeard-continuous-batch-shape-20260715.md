# Treebeard Continuous-Batch Shape Screen - 2026-07-15

## Outcome

Park short admission coalescing for the current B70/12-slot server. The server
already converts synchronized and 8 ms-staggered arrivals into full-width
decode batches for roughly 96-97% of classified steady generation calls. An
8 ms adjacent-request stagger changed workload p50 by only +1.41%, far below
the 10% major-play gate.

Retain the opt-in `TREEBEARD_BATCH_SHAPE_PROF=1` telemetry and the guarded
`TREEBEARD_WAVEFRONT_ARRIVAL_GAP_MS` control. They measure the actual
`llama_decode` submission shape and can qualify a future workload with a real
backlog or ragged-lane problem.

## Identity and guard

Both screens used the same candidate files:

```text
393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c  llama-server
bc96ab95fdf911ca1fb42d2af7e405844807edabb64eccc655b41d697aef1bb3  libggml-sycl.so
ee374529bce89191383247934489c246fdbf9aea840f1fdc3cfde6afe7f9d790  libllama-server-impl.so
```

The workload used the exact Qwen3.6-35B-A3B Q5_K_XL model, 262144 context,
12 unified-KV slots, a reused 512-token prompt, 32 generated tokens per agent,
and two repeated control waves. Production was stopped and restored by the
guarded harness for each screen. The completed staggered artifact has an empty
kernel-fault signature file; the user service restored active/running with
zero restarts and a healthy inference endpoint.

## Results

| Adjacent arrival gap | Workload p50 tok/s | Workload mean tok/s | Server per-slot p50 tok/s |
| ---: | ---: | ---: | ---: |
| 0 ms | 46.442 | 46.436 | 18.771 |
| 8 ms | 47.099 | 47.031 | 18.733 |

The 8 ms result is +1.41% at p50 and +1.28% at mean. It is not a meaningful
coalescing opportunity.

At the last cumulative profile point:

| Gap | Full-width `active=12, submitted=12` | 11-column tail | Classified full-width share |
| ---: | ---: | ---: | ---: |
| 0 ms | 106 | 3 | 97.25% |
| 8 ms | 104 | 4 | 96.30% |

The remaining profiler rows are prompt, warmup, single-stream, or mixed
prompt/admission submissions and are not relabeled as steady decode.

## Correctness boundary

Both screens reproduced the pre-existing width-zero backend batch-shape
divergence: two of four repeated concurrent waves changed greedy output relative
to their first control wave. The synchronized run therefore failed strict
parity; the staggered diagnostic recorded the same 50% repeated-wave parity
rate without promotion. This experiment did not introduce or fix that boundary,
and no scheduler policy is eligible for rollout while making it worse.

## Evidence

- Synchronized artifact:
  `results/treebeard-single-wavefront-b70/20260715-010408`
- 8 ms staggered artifact:
  `results/treebeard-single-wavefront-b70/20260715-010643`
