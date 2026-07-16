# Treebeard Linear-Attention Projection Fusion Screen - 2026-07-15

## Outcome

Do not implement a fused QKV/z/alpha/beta projection kernel for the current
B70/Qwen3.6 production shape. The complete cluster is large enough to notice,
but almost all of its time is irreducible streaming of four distinct quantized
weight matrices. The bounded removable share is about 3.21% of serialized
decode time, below the 10% major-play entry gate.

Keep the profiler improvement that attributes otherwise generic scaled matrix
multiplies by normalized source-weight family and reports fixed 50-eval
windows. It removes both naming ambiguity and one-time JIT inflation from the
next fusion decision.

## Guarded run

- Artifact: `results/treebeard-single-wavefront-b70/20260715-005316`
- Model: Qwen3.6-35B-A3B Q5_K_XL
- Device/runtime: Intel Arc Pro B70, SYCL
- Shape: 262144 context, 12 slots, strict-parity width-zero control, 128
  generated tokens
- Result: benchmark passed, exact parity passed, kernel-signature file empty
- Restoration: `turbo-statetree-rc4.service` active/running under the user
  systemd manager, zero restarts, health endpoint OK

## Steady-state attribution

The final 50-eval window contained 564.5 ms of `MUL_MAT` time. Its leading
families were:

| Family | Window time | Share of `MUL_MAT` | Calls/eval |
| --- | ---: | ---: | ---: |
| `blk.*.attn_qkv.weight` | 77.7 ms | 13.8% | 30 |
| `ffn_moe_logits-*` | 59.1 ms | 10.5% | 40 |
| `linear_attn_out-*` | 54.8 ms | 9.7% | 30 |
| `result_output` | 53.7 ms | 9.5% | 1 |
| `z-*` | 50.6 ms | 9.0% | 30 |
| `blk.*.ssm_alpha.weight` | 42.0 ms | 7.4% | 30 |
| `blk.*.ssm_beta.weight` | 35.5 ms | 6.3% | 30 |

The four same-input linear-attention projections total 205.8 ms. The global
serialized total increased from 7,969.4 ms after 200 graph evaluations to
9,304.1 ms after 250, so the matching steady window is 1,334.7 ms. The cluster
therefore represents 15.42% of serialized time.

## Falsification bound

Fusion cannot remove any of the four weight reads. It can share the small input
activation read and replace four submissions with one. The surrounding
serialized one-token graph measures about 9.5 us per small glue operation,
which is a deliberately generous proxy for a removable submission. Removing
three submissions across 30 linear-attention layers and 50 graph evaluations
bounds the saving at:

```text
3 * 30 * 50 * 9.5 us = 42.75 ms
42.75 / 1334.7 = 3.21%
```

Even perfect recovery of that bound misses the 10% entry gate, and real fusion
would retain epilogue work and may reduce row-level parallelism. The kernel is
parked without spending a build/validation sweep.
