# Quantized KV bandwidth serial stop on Intel B70 (2026-07-15)

## Decision

Reject quantized KV cache as the next sequence-ragged bandwidth play. Do not
enable q8 indexed attention and do not spend a production-shaped batched run
on this lane.

The required serial top-1 gate failed before any ragged capability change was
made. The most compressed candidate was also substantially slower, so neither
correctness nor the speed premise survived the cheap screen.

## Serial gate

All arms used the same rebuilt binary, 12 heterogeneous prompts run one at a
time, deterministic top-1 sampling, no speculation, and 52 generated tokens
per prompt. The accepted recurrent state-I/O default was active in every arm.

| K cache | V cache | Artifact | Aggregate tok/s | Hash parity vs f16/f16 |
| --- | --- | --- | ---: | ---: |
| f16 | f16 | `20260715-110122` | 63.411579 | 12/12 reference |
| q8_0 | q8_0 | `20260715-110302` | 52.123706 | 9/12 |
| f16 | q8_0 | `20260715-110455` | 58.602611 | 8/12 |

Relative to f16/f16, q8_0/q8_0 regressed 17.800966% and f16/q8_0 regressed
7.583738%. The V-only isolation was chosen as the lower-numerical-risk mixed
format. It failed more output hashes while still losing speed. Quantizing K
alone has greater direct influence on attention logits and cannot supply a
major bandwidth win after saving only the other half of the cache, so the
bounded screen stops here.

## Consequence

The existing f16-only fail-closed gate for Treebeard ragged indexed attention
is unchanged. No quantized ragged code was implemented and no batched q8 run
was started.

Each guarded arm restored the exact production service with `NRestarts=0`.
All three kernel fault-signature files were empty.
