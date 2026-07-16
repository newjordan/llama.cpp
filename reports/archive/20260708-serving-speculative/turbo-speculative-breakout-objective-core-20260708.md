# Turbo Speculative Breakout Objective Core Benchmark - 2026-07-08

## Summary

Current optimized deterministic 11-case run against the existing `:8093`
12-slot Turbo server. Model final scoring was disabled; acceptance is entirely
from deterministic validators.

The optimization target was the previous 21.30s mean multipass-core wall time.
This run enables objective fast path: validate branches deterministically, use
the selected branch directly when it already passes, run validator-feedback
repair only when needed, and escalate to model verifier plus recombine only
after direct branch/repair still fails.

```text
suite: /tmp/turbo-speculative-breakout-objective-core-fast-current2/20260709T024357Z-2058197.suite.json
jsonl: /tmp/turbo-speculative-breakout-objective-core-fast-current2/results.jsonl
```

Aggregate deterministic result:

| Metric | Value |
| --- | ---: |
| Tasks | 11 |
| Validated tasks | 11 |
| Baseline passes | 5 |
| Initial breakout passes | 9 |
| Final breakout passes | 11 |
| Pass delta | +6 |
| Wins | 6 |
| Ties | 5 |
| Losses | 0 |
| All breakout passed | true |
| No objective losses | true |
| Mean baseline score | 64.09 |
| Mean initial breakout score | 95.36 |
| Mean final breakout score | 100.00 |
| Mean final score delta | +35.91 |
| Median final score delta | +21 |
| Min final score delta | 0 |
| Max final score delta | +100 |
| Objective repair attempts | 2 |
| Objective repair accepts | 2 |
| Fallback recombine attempts | 0 |
| Fallback recombine accepts | 0 |
| Fallback repair attempts | 0 |
| Fallback repair accepts | 0 |

## Wall-Time Optimization

Reference full-verifier artifact:

```text
/tmp/turbo-speculative-breakout-objective-core/20260709T013917Z-1985364.suite.json
```

That path ran model verifier fanout and recombination for every task. The
optimized path kept the same final 11/11 deterministic pass result and zero
objective losses while making the expensive verifier/recombine path conditional.

| Metric | Full verifier path | Optimized fast path | Change |
| --- | ---: | ---: | ---: |
| Final multipass accuracy | 100.00% | 100.00% | 0.00 points |
| Objective losses | 0 | 0 | 0 |
| Mean multipass core wall / task | 21.30 s | 7.29 s | -65.77% |
| Mean multipass with baseline wall / task | 22.28 s | 8.91 s | -59.99% |
| Core speedup | 1.00x | 2.92x | +1.92x |
| With-baseline speedup | 1.00x | 2.50x | +1.50x |
| Multipass core predicted tok/s | 68.58 | 82.84 | +20.80% |
| Branch fanout predicted tok/s | 85.88 | 86.48 | +0.71% |
| Verifier fanout task count | 11 | 0 | -11 |
| Recombine task count | 11 | 0 | -11 |

Baseline pass count varied between runs because the baseline is a sampled model
completion. The stable acceptance signal here is final validator pass rate and
loss count: both artifacts finished 11/11 with zero objective losses.

## Accuracy And Tok/sec

This is the current mandate-level readout for single-agent multipass:

| Metric | Value |
| --- | ---: |
| Baseline single-pass accuracy | 45.45% |
| Initial multipass accuracy | 81.82% |
| Final multipass accuracy | 100.00% |
| Accuracy delta vs baseline | +54.55 points |
| Objective losses | 0 |
| Baseline single-pass predicted tok/s | 19.86 |
| Branch fanout predicted tok/s | 86.48 |
| Verifier fanout predicted tok/s | n/a (0 requests) |
| Recombine final predicted tok/s | n/a (0 requests) |
| Objective repair predicted tok/s | 21.53 |
| Multipass core predicted tok/s | 82.84 |
| Multipass with baseline predicted tok/s | 71.38 |
| Mean baseline wall time / task | 1.62 s |
| Mean multipass core wall time / task | 7.29 s |
| Mean multipass with baseline wall time / task | 8.91 s |

Definitions:

- `baseline single-pass`: the one-shot baseline request.
- `branch fanout`: 12 parallel candidate branches.
- `verifier fanout`: 12 parallel verifier passes, now only on fast-path
  fallback.
- `recombine final`: the model recombination request, now only on fast-path
  fallback.
- `objective repair`: validator-feedback repair requests.
- `multipass core`: branch fanout + any verifier fanout + any recombine + any
  objective repair, excluding the separate baseline comparator.
- `multipass with baseline`: baseline + multipass core, matching the harness
  path when baseline is used as recombination context.

## Interpretation

This is good news for the current optimization target:

- The 21.30s multipass-core mean dropped to 7.29s while preserving 11/11 final
  deterministic accuracy and zero losses.
- Branch throughput stayed effectively flat, so the win came from skipping
  unnecessary verifier/recombine phases, not from a lower-quality branch pass.
- Nine tasks passed directly from deterministic branch selection, two were
  fixed by objective repair, and no task needed fallback verifier plus
  recombine.

This is still slower than the single-pass baseline:

- Baseline mean wall time was 1.62s per task; optimized multipass core was
  7.29s.
- The useful reading is "slower, but smarter" on this deterministic suite:
  +54.55 accuracy points and zero losses, paid for with roughly 4.5x baseline
  wall time before counting baseline comparator overhead.

This is not yet product-value evidence:

- The cases are controlled transform tasks, not real user workflows.
- The validators score exact outputs, not usefulness, saved labor, or decision
  quality.
- The matrix win required field-level validator repair after direct branch
  selection repeated the same wrong diagonal fields.
- Cost, latency, and operator trust are not yet tied to a target use case.

Read this result as a stronger green light for multipass mechanics, not as
proof that the product is valuable.

## Per-Task Results

| Task | Validator | Baseline | Initial | Final | Delta | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `arithmetic_json` | 70 | 70 | 100 | +30 | repaired win |
| 1 | `schema_fields` | 100 | 100 | 100 | 0 | tie pass |
| 2 | `five_experiments` | 100 | 100 | 100 | 0 | tie pass |
| 3 | `csv_json_transform` | 100 | 100 | 100 | 0 | tie pass |
| 4 | `rubric_signals` | 100 | 100 | 100 | 0 | tie pass |
| 5 | `json_exact` / `nested-batch-json` | 100 | 100 | 100 | 0 | tie pass |
| 6 | `json_exact` / `filter-sort-json` | 0 | 100 | 100 | +100 | direct win |
| 7 | `json_exact` / `matrix-json` | 79 | 79 | 100 | +21 | repaired win |
| 8 | `lines_exact` / `priority-lines` | 12 | 100 | 100 | +88 | direct win |
| 9 | `json_exact` / `json-merge-patches` | 0 | 100 | 100 | +100 | direct win |
| 10 | `json_exact` / `duration-minutes-json` | 44 | 100 | 100 | +56 | direct win |

## What Improved

- Objective fast path skips model verifier and recombine when deterministic
  branch validation already finds a passing branch.
- Fallback verifier and recombine run only for unresolved fast-path failures.
- Fallback counters are recorded separately from normal objective repair.
- Verifier-informed fallback repair is available if fallback recombine improves
  but still does not pass.
- Objective repair adds field-level feedback for exact JSON failures, naming
  wrong current values without revealing expected values.
- `accuracy_throughput_summary` now reports accuracy and phase throughput in one
  artifact, including multipass core wall time and predicted tok/s.

The matrix case still shows the core acceptance problem: direct branch selection
kept wrong diagonal fields. Field-level repair feedback and the non-oracle
anti-diagonal hint recovered the valid final JSON:

```json
{"row_sums":[16,22,21],"col_sums":[20,16,23],"main_diag":16,"anti_diag":11,"diag_delta":5}
```

This benchmark result is therefore a valid measurement of
branch-plus-validator-plus-repair with conditional fallback available, not
branch fanout alone.

## Runtime Snapshot

| Task | Core wall s | Branch s | Verifier s | Final s | Repair s | Fallback recombine |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8.87 | 6.97 | 0.00 | 0.00 | 1.90 | 0/0 |
| 1 | 10.18 | 10.18 | 0.00 | 0.00 | 0.00 | 0/0 |
| 2 | 6.71 | 6.71 | 0.00 | 0.00 | 0.00 | 0/0 |
| 3 | 9.11 | 9.11 | 0.00 | 0.00 | 0.00 | 0/0 |
| 4 | 6.80 | 6.80 | 0.00 | 0.00 | 0.00 | 0/0 |
| 5 | 8.76 | 8.76 | 0.00 | 0.00 | 0.00 | 0/0 |
| 6 | 4.15 | 4.15 | 0.00 | 0.00 | 0.00 | 0/0 |
| 7 | 9.57 | 6.96 | 0.00 | 0.00 | 2.61 | 0/0 |
| 8 | 6.13 | 6.13 | 0.00 | 0.00 | 0.00 | 0/0 |
| 9 | 4.33 | 4.33 | 0.00 | 0.00 | 0.00 | 0/0 |
| 10 | 5.58 | 5.58 | 0.00 | 0.00 | 0.00 | 0/0 |

Mean multipass-core wall time was 7.29 seconds per task. Mean
multipass-with-baseline wall time was 8.91 seconds per task.

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
  --objective-fast-path \
  --out-dir /tmp/turbo-speculative-breakout-objective-core-fast-current2
```

## Acceptance

This is the current solid benchmark artifact:

- No breakout validation failures.
- No validator errors.
- No objective losses.
- Final breakout passes all 11 tasks.
- Baseline passes 5 of 11 tasks.
- Mean multipass-core wall time is under 8 seconds.

Remaining caveat: the benchmark is still small and validator-backed. It proves
the current objective-mode harness can turn deterministic failures into valid
final answers on this suite with substantially lower wall time; it does not
prove general free-form answer quality.
