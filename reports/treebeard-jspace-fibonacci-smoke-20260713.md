# Treebeard J-Space Fibonacci pooling smoke

Date: 2026-07-13

Status: exact-model CPU diagnostic only; not a semantic, B70, or production
gate.

## Identity

- Model: `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`
- Full SHA-256 verified immediately before recording this report:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`
- Reported model metadata: Qwen3.6-35B-A3B, 2048 embedding dimensions,
  llama.cpp architecture tag `qwen35moe`.
- Backend: CPU-only probe pinned to CPUs 16–31. The live B70 server was not
  touched.

## Operator

The opt-in probe constructed the sparse causal suffix matrix at Fibonacci
horizons `1, 2, 3, 5, 8, 13, 21, ...`, capped by both the requested maximum and
the tokenized prompt length. Every row uniformly averages its suffix, so the
matrix is nonnegative and row-stochastic. Fibonacci selects support lengths,
not signal weights. The pooling evaluator is schedule-agnostic; this run passed
the Fibonacci horizon generator into that generic suffix-boxcar primitive.

The full command shape was:

```console
taskset -c 16-31 build-cpu/bin/llama-jspace-probe \
  -m /home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  -ngl 0 --no-op-offload -c 128 -b 8 -ub 4 -t 16 \
  -p 'The operator reviews the queue, checks each record, marks the completed items, and files the report before closing the desk.' \
  --token-ids 13 --fibonacci-pool-max 21
```

The deliberately small batch sizes forced the 24-token prompt across multiple
outer batches and inner ubatches.

## Result

- Prompt tokens expected: 24
- Residual columns observed across callback chunks: 24
- Retained suffix columns: 21
- Reported sparse-matrix shape: `7 x 24` over the full prompt-token domain;
  retained-buffer shape: `21 x 2048`, starting at global prompt position 3
- Emitted complete horizons: `1, 2, 3, 5, 8, 13, 21`
- Residual width: 2048
- Horizon-one L2: `20.436967339862107`
- Existing final-residual L2: `20.436967339862107`
- Horizon-one vector equality with `--include-residual-vector`: exact, all 2048
  components
- A single-batch `-b 64 -ub 64` run and the forced multi-batch `-b 8 -ub 4`
  run produced exactly equal 2048-component pooled vectors at every emitted
  horizon (maximum absolute difference and RMSE both zero).
- Pool L2 by horizon:
  - 1: `20.436967339862107`
  - 2: `14.561710484816942`
  - 3: `14.115564947164323`
  - 5: `14.035611668143414`
  - 8: `14.474230148303372`
  - 13: `15.934605485820269`
  - 21: `16.79586805524528`

The callback's exact column-count and horizon-one checks passed. The model-
independent self-test also passed horizon generation, suffix means, constant
preservation, generic comparator horizons, delayed non-overlapping Fibonacci
recursion, vector serialization gating, invalid-bound rejection, and non-finite-
input rejection.

A separate raw-dual sadness sweep exercised baseline reuse and one nonzero
command. The baseline, reused zero arm, and independently evaluated nonzero arm
all reported exactly 24 columns and the same complete horizon set through 13;
the zero arm retained exact zero divergence while the nonzero arm computed its
full-vocabulary collateral normally.

## Evidence boundary

This proves that the bounded suffix matrix whose supports obey the Fibonacci
recurrence can be collected correctly from the exact Qwen3.6 final residual
across batch boundaries. It does not show that Fibonacci spacing improves
sensing or control. That claim remains gated against mean-age/effective-sample-
size-matched EMA and feature-count/support-matched boxcar controls as specified
in `docs/treebeard-jspace-steering.md`.
