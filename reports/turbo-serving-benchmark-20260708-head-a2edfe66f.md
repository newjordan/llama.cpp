# Turbo Serving Benchmark Report - HEAD a2edfe66f - 2026-07-08

## Executive Summary

Yes: on the rebuilt HEAD binary, exact old-prompt np12 serving preserved
essentially the same aggregate throughput when moving from 32k context to 262k
context with `-kvu`.

| Final HEAD comparison | Aggregate tok/s | Ratio |
| --- | ---: | ---: |
| Short c32k np12 old-prompt | 178.1459 | baseline |
| Long c262k np12 `-kvu` old-prompt | 178.7020 | 1.003x |

The conclusion uses only rebuilt HEAD data from binary `version 61`
(`a2edfe66f`). HEAD also reached 207.8500 tok/s at short-context np32, so the
200-class multi-agent saturation result remains present.

A warmed single active request on the live HEAD c262k np12 `-kvu` server measured
79.6102 tok/s, so single-agent speed remains around the prior 80 tok/s class.

## Final HEAD Artifacts

| Artifact | Path |
| --- | --- |
| Old cd395a baseline sweep | `/tmp/cd395a152-server-sweep.jsonl` |
| HEAD short c32k old-prompt sweep | `/tmp/turbo-serving-20260708-short-c32k-head-a2edfe66f.jsonl` |
| HEAD short c32k logs | `/tmp/turbo-serving-20260708-short-c32k-head-a2edfe66f-logs` |
| HEAD long c262k old-prompt sweep | `/tmp/turbo-serving-20260708-long-c262k-head-a2edfe66f-oldprompt.jsonl` |
| HEAD long c262k old-prompt logs | `/tmp/turbo-serving-20260708-long-c262k-head-a2edfe66f-oldprompt-logs` |
| HEAD long c262k ablation results | `/tmp/turbo-serving-20260708-long-c262k-head-a2edfe66f-ablate/results.jsonl` |
| HEAD long c262k ablation logs | `/tmp/turbo-serving-20260708-long-c262k-head-a2edfe66f-ablate` |
| HEAD live c262k np12 single-request result | `/tmp/turbo-serving-20260708-head-live-8093-single-oldprompt.jsonl` |

## Config Summary

- Binary: `/home/frosty40/builds/turbo-experimental-build/bin/llama-server`
- Verified binary version: `version 61 (a2edfe66f)`
- Repo state at handoff: branch `turbo-experimental`, clean status
- Model: `/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`
- Hardware: Intel Arc Pro B70, SYCL
- Old-prompt workload: 34 prompt tokens, `n_predict=256`, `temperature=0.0`, `top_k=1`
- Main metric: aggregate predicted tokens per second over batch wall time

Short c32k logs show fixed per-slot context:

| np | Effective slot ctx |
| ---: | ---: |
| 12 | 2816, after total ctx rounded to 33792 |
| 16 | 2048 |
| 32 | 1024 |

Long c262k logs show 12 slots with `n_ctx = 262144`, `-kvu`, and `ubatch=1024`.

## Throughput

Old cd395a short c32k baseline:

| np | Aggregate tok/s | Mean slot tok/s |
| ---: | ---: | ---: |
| 16 | 200.1588 | 14.0371 |
| 32 | 213.5611 | 7.3589 |

Final HEAD short c32k old-prompt:

| np | Aggregate tok/s | Mean slot tok/s |
| ---: | ---: | ---: |
| 12 | 178.1459 | 16.7701 |
| 16 | 189.5578 | 13.3002 |
| 32 | 207.8500 | 7.1632 |

Final HEAD long c262k np12:

| Run | Aggregate tok/s | Mean slot tok/s | Notes |
| --- | ---: | ---: | --- |
| Old-prompt `-kvu` | 178.7020 | 16.6758 | Final conclusion row |
| Ablation baseline | 179.7589 | 16.5583 | 12/12 ok, no fragmentation |
| Ablation indexed | 178.9223 | 16.4725 | 12/12 ok, no fragmentation; indexed path not exercised |

Final HEAD live c262k np12 single active request:

| Run | Predicted tok/s | Prompt tok/s | Notes |
| --- | ---: | ---: | --- |
| Old-prompt `-kvu`, one active request | 79.6102 | 327.7961 | Warmed request against live `8093` server |

## Comparisons

| Comparison | Ratio | Delta |
| --- | ---: | ---: |
| HEAD long c262k np12 vs HEAD short c32k np12 | 100.3% | +0.5561 tok/s |
| HEAD short c32k np16 vs old cd395a np16 | 94.7% | -10.6011 tok/s |
| HEAD short c32k np32 vs old cd395a np32 | 97.3% | -5.7111 tok/s |
| HEAD ablation indexed vs baseline | 99.5% | -0.8365 tok/s |

## Superseded Stale Data

The following pre-rebuild artifacts are superseded and should not drive the
final conclusion. They were produced with binary `version 59 (8b785c42a)`, not
rebuilt HEAD `version 61 (a2edfe66f)`:

- `/tmp/turbo-serving-20260708-short-c32k-current.jsonl`
- `/tmp/turbo-serving-20260708-short-c32k-current-np12-rerun.jsonl`
- `/tmp/turbo-serving-20260708-long-c262k-current-oldprompt.jsonl`
- `/tmp/turbo-serving-20260708-long-c262k-current-ablate/results.jsonl`

## Validation And Caveats

- All HEAD benchmark requests completed successfully in the result artifacts.
- Run validation reported no reset, coredump, or segfault matches in `journalctl -k` after `2026-07-08 16:15:00`.
- One GPU engine reset occurred at `2026-07-08 16:05`, before the rebuilt HEAD reruns; it is not part of the final HEAD window.
- Logs contain the expected `common_fit_params` warning from `-ngl 99`; it did not abort the runs.
- The clean 35B ablation had contiguous KV. The indexed log shows `Qrows=12` with `indexed=0`, so this run does not demonstrate indexed-FATTN speedup.
- Indexed-FATTN benefit, if cited, belongs to the separate prior 8B fragmented ablation, not this clean 35B run.
- This is a narrow serving result for exact old-prompt, np12/np16/np32 short generation. It does not prove latency or throughput under real long prompts or fragmented live traffic.
