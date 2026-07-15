# Single-token MoE down fusion: B70 rejection report

Date: 2026-07-15

Decision: **park**. The production graph activated the direct ordered down
epilogue across the MoE stack, but the exact 128-token serving screen improved
only 3.21%, below the active goal's 10% candidate-entry gate and far below its
25% single-stream promotion target. The experimental kernel and recognizer
changes were removed; only the named profiling instrumentation is retained.

## Why this was tested

An opt-in serialized SYCL profile separated previously opaque fused work by
fusion kind. At 100 graph evaluations on the production Qwen3.6-35B-A3B model,
the baseline attributed 16.9% to `MUL_MAT_ID` and 5.7% to the existing fused
MoE down/reduce path. This put expert down projection above the 10% evidence
threshold and made a one-token direct weighted epilogue a plausible major play.

Baseline profile and serving screen:

`/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260715-001620`

The named-fusion profile that established attribution is also preserved at:

`/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260715-000850`

## Bounded implementation and correctness

The experiment crossed the existing ordered weighted MMVQ kernel into the
one-token graph. It preserved expert-slot accumulation order and the routing
multiply rounding point. Production allocator reuse required snapshotting the
eight routing weights and eight expert IDs (64 bytes total per MoE layer)
before writing the final output.

The expanded SYCL-vs-CPU oracle passed 11/11 cases across Q5_K, Q6_K, and Q8_0,
including the optional two-multiply scale/route fixture:

`/home/frosty40/turbo/treebeard-work/results/treebeard-moe-down-single/validation-scaled-20260715-002530`

The guard restored `turbo-statetree-rc4.service`, verified build identity,
performed real inference, found no busy/reserved slots or live StateTree state,
and reported zero service restarts.

## Production activation

The final activation artifact is:

`/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260715-003634`

At 100 serialized graph evaluations:

- `MUL_MAT_ID` fell from 38.2 to 1.2 calls/eval.
- `FUSED_MOE_DOWN` rose from 2.6 to 39.6 calls/eval.
- The materialized single-token `CONT` and `SUM_ROWS` tail nearly disappeared.
- Live telemetry emitted `single-input-snapshot detail=0x3` followed by
  `single-integrated-hit` for the allocator-overlap cases.

This proves that the intended production layers, rather than only synthetic
fixtures, used the candidate.

## Performance result

The identical profiler-enabled 128-token screen measured:

| Arm | Mean generated tok/s |
| --- | ---: |
| Baseline (`20260715-001620`) | 36.7296 |
| Activated candidate (`20260715-003634`) | 37.9101 |
| Delta | **+3.21%** |

The two candidate observations were 37.9043 and 37.9159 tok/s with identical
output hashes. The final guard restored production with zero restarts and no
kernel reset, hang, fault, OOM, or panic signatures.

Although the graph eliminated almost all standalone expert-down MMID calls,
the direct ordered kernel absorbed most of their serialized cost. The measured
end-to-end gain is real but too small for the current major-improvement lane.
No A/B/A or concurrency sweep is justified after this decisive entry-gate
failure.

## Re-entry condition

Do not revisit this topology without a new profile-backed mechanism expected
to make the direct down kernel itself substantially faster, such as measured
subgroup/workgroup expert weight reuse or a different expert-as-lanes mapping.
The new mechanism must plausibly add at least another 7% end-to-end before rig
time is allocated.
