# Turbo Speculative Breakout Objective Core Benchmark - 2026-07-08

## Summary

Final deterministic 11-case run against the existing `:8093` 12-slot Turbo
server. Model final scoring was disabled for this run; acceptance is entirely
from deterministic validators.

```text
suite: /tmp/turbo-speculative-breakout-objective-core/20260709T013917Z-1985364.suite.json
jsonl: /tmp/turbo-speculative-breakout-objective-core/results.jsonl
```

Aggregate deterministic result:

| Metric | Value |
| --- | ---: |
| Tasks | 11 |
| Validated tasks | 11 |
| Baseline passes | 4 |
| Initial breakout passes | 10 |
| Final breakout passes | 11 |
| Pass delta | +7 |
| Wins | 7 |
| Ties | 4 |
| Losses | 0 |
| All breakout passed | true |
| No objective losses | true |
| Mean baseline score | 55.00 |
| Mean initial breakout score | 97.27 |
| Mean final breakout score | 100.00 |
| Mean final score delta | +45.00 |
| Median final score delta | +30 |
| Min final score delta | 0 |
| Max final score delta | +100 |
| Objective repair attempts | 1 |
| Objective repair accepts | 1 |

## Accuracy And Tok/sec

This is the current mandate-level readout for single-agent multipass:

| Metric | Value |
| --- | ---: |
| Baseline single-pass accuracy | 36.36% |
| Initial multipass accuracy | 90.91% |
| Final multipass accuracy | 100.00% |
| Accuracy delta vs baseline | +63.64 points |
| Objective losses | 0 |
| Baseline single-pass predicted tok/s | 41.71 |
| Branch fanout predicted tok/s | 85.88 |
| Verifier fanout predicted tok/s | 82.65 |
| Multipass core predicted tok/s | 68.58 |
| Multipass with baseline predicted tok/s | 67.40 |
| Mean baseline wall time / task | 0.97 s |
| Mean multipass core wall time / task | 21.30 s |
| Mean multipass with baseline wall time / task | 22.28 s |

Definitions:

- `baseline single-pass`: the one-shot baseline request.
- `branch fanout`: 12 parallel candidate branches.
- `verifier fanout`: 12 parallel verifier passes.
- `multipass core`: branch fanout + verifier fanout + recombine + objective
  repair, excluding the separate baseline comparator.
- `multipass with baseline`: baseline + multipass core, matching the current
  harness path when baseline is used as recombination context.

## Interpretation

This is good news for the orchestration harness:

- The benchmark runner, branch validation, objective repair, artifact logging,
  and deterministic summaries are now working end to end.
- The harness can expose model-verifier failures instead of trusting them.
- On this deterministic suite, breakout mode produced valid final answers for
  all cases while the single-pass baseline passed only 4 of 11.
- Multi-slot branch and verifier phases are materially faster than a serial
  single-slot equivalent would be: branch fanout sustained 85.88 predicted tok/s
  and verifier fanout sustained 82.65 predicted tok/s.

This is not yet product-value evidence:

- The cases are controlled transform tasks, not real user workflows.
- The validators score exact outputs, not usefulness, saved labor, or decision
  quality.
- The arithmetic win came from validator-guided repair after all branches made
  the same mistake, so it does not prove branch fanout alone adds intelligence.
- Cost, latency, and operator trust are not yet tied to a target use case.

Read this result as a green light for the evaluation machinery, not as proof
that the product is valuable.

## Per-Task Results

| Task | Validator | Baseline | Final | Delta | Result |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | `arithmetic_json` | 70 | 100 | +30 | repaired win |
| 1 | `schema_fields` | 100 | 100 | 0 | tie pass |
| 2 | `five_experiments` | 100 | 100 | 0 | tie pass |
| 3 | `csv_json_transform` | 0 | 100 | +100 | win |
| 4 | `rubric_signals` | 100 | 100 | 0 | tie pass |
| 5 | `json_exact` / `nested-batch-json` | 100 | 100 | 0 | tie pass |
| 6 | `json_exact` / `filter-sort-json` | 0 | 100 | +100 | win |
| 7 | `json_exact` / `matrix-json` | 79 | 100 | +21 | win |
| 8 | `lines_exact` / `priority-lines` | 12 | 100 | +88 | win |
| 9 | `json_exact` / `json-merge-patches` | 0 | 100 | +100 | win |
| 10 | `json_exact` / `duration-minutes-json` | 44 | 100 | +56 | win |

## What Improved

- The suite expanded from 5 smoke cases to 11 deterministic cases.
- `json_exact` and `lines_exact` validators allow case-defined expected outputs
  without writing a custom Python validator for every transform.
- Each branch is deterministically validated before selection in objective mode.
- Objective repair uses concise failed-check feedback and optional `repair_hint`
  text instead of a verbose validator dump.
- Request artifacts keep both cleaned `content` and raw model `raw_content`.
- Empty model-score summaries are suppressed when `--no-score-final` is used.

The arithmetic case still shows the core acceptance problem: all branches kept
the wrong `gamma=995`, but deterministic repair corrected the final answer to
`gamma=987` and `checksum=1467`. This benchmark result is therefore a valid
measurement of branch-plus-validator-plus-repair, not branch fanout alone.

## Runtime Snapshot

| Task | Wall s | Branch wall s | Branch agg tok/s floor | Pred tok | Prompt tok |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18.82 | 5.29 | 70.31 | 1067 | 6681 |
| 1 | 29.30 | 9.57 | 105.08 | 1966 | 10516 |
| 2 | 20.93 | 6.17 | 85.13 | 1448 | 8651 |
| 3 | 26.47 | 9.31 | 108.00 | 1958 | 10625 |
| 4 | 25.71 | 8.98 | 97.25 | 1724 | 9151 |
| 5 | 23.86 | 8.39 | 84.15 | 1461 | 9668 |
| 6 | 21.96 | 6.79 | 47.84 | 1285 | 9045 |
| 7 | 24.69 | 6.69 | 95.25 | 1911 | 9613 |
| 8 | 21.57 | 5.92 | 66.93 | 1406 | 9558 |
| 9 | 18.23 | 4.29 | 50.99 | 1218 | 7574 |
| 10 | 18.12 | 5.44 | 79.03 | 1109 | 8082 |

Mean multipass-with-baseline wall time was 22.28 seconds per task. Total
predicted tokens across the suite were 16516; total prompt tokens were 98949.

## Command

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --benchmark-suite objective-core \
  --branch-slots 0-11 \
  --baseline-tokens 260 \
  --branch-tokens 220 \
  --verify-tokens 120 \
  --final-tokens 260 \
  --score-tokens 120 \
  --no-score-final \
  --repair-rounds 0 \
  --objective-repair-rounds 1 \
  --out-dir /tmp/turbo-speculative-breakout-objective-core
```

## Acceptance

This is the current solid benchmark artifact:

- No breakout validation failures.
- No validator errors.
- No objective losses.
- Final breakout passes all 11 tasks.
- Baseline passes only 4 of 11 tasks.

Remaining caveat: the benchmark is still small and validator-backed. It proves
the current objective-mode harness can turn deterministic failures into valid
final answers on this suite; it does not prove general free-form answer quality.
