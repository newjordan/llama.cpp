# Turbo Speculative Breakout Objective Benchmark - 2026-07-08

Superseded for acceptance by the expanded 11-case core run in
`reports/turbo-speculative-breakout-objective-core-20260708.md`.

## Summary

Live 12-slot run against the existing `:8093` Turbo server using deterministic
validators, not model-only scoring.

```text
suite: /tmp/turbo-speculative-breakout-objective-12/20260708T231010Z-1824994.suite.json
jsonl: /tmp/turbo-speculative-breakout-objective-12/results.jsonl
```

Aggregate deterministic result:

| Metric | Value |
| --- | ---: |
| Tasks | 5 |
| Baseline passes | 4 |
| Breakout passes | 5 |
| Pass delta | +1 |
| Wins | 1 |
| Ties | 4 |
| Losses | 0 |
| Mean baseline score | 94.00 |
| Mean initial breakout score | 94.00 |
| Mean breakout score | 100.00 |
| Mean score delta | +6.00 |
| Min score delta | 0 |
| Max score delta | +30 |
| Objective repair attempts | 1 |
| Objective repair accepts | 1 |

Model-score telemetry on the same final answers is not the acceptance source:

| Metric | Value |
| --- | ---: |
| Wins | 1 |
| Ties | 3 |
| Losses | 1 |
| Mean baseline score | 95.00 |
| Mean breakout score | 96.00 |
| Mean model score delta | +1.00 |

## Per-Task Objective Results

| Task | Validator | Baseline | Breakout | Delta | Result |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | `arithmetic_json` | 70 | 100 | +30 | repaired breakout win |
| 1 | `schema_fields` | 100 | 100 | 0 | tie pass |
| 2 | `five_experiments` | 100 | 100 | 0 | tie pass |
| 3 | `csv_json_transform` | 100 | 100 | 0 | tie pass |
| 4 | `rubric_signals` | 100 | 100 | 0 | tie pass |

The arithmetic failure from the previous run is now converted into an accepted
objective repair. All 12 arithmetic branches still produced `gamma=995` and
`checksum=1475` for `2^10-37`, but the deterministic validator rejected them,
the shorter hint-aware repair prompt corrected the answer to `gamma=987` and
`checksum=1467`, and the artifact records the model verifier miss.

Before this fix, the comparable run
`/tmp/turbo-speculative-breakout-objective-12/20260708T225437Z-1807148.suite.json`
had baseline 3/5 pass, breakout 4/5 pass, and zero accepted objective repairs.
The current run has baseline 4/5 pass, breakout 5/5 pass, and one accepted
objective repair. The baseline pass increase is from cleaner leading-tag
handling, not from branch search.

## Implemented Fixes

- Added deterministic validation to each branch packet in objective benchmark
  mode, and made branch selection prefer validator-passing candidates.
- Added `raw_content` alongside cleaned `content` in request artifacts so tag
  stripping and stop behavior remain auditable.
- Added compact `repair_hint` support for objective benchmark cases.
- Replaced the verbose objective repair prompt with a shorter failed-checks
  prompt and objective-repair-specific stop sequences.
- Cleaned leading `</task>` / `<task>` artifacts before validation.

## Runtime Snapshot

| Task | Wall s | Branch wall s | Branch agg tok/s floor | Pred tok | Prompt tok |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 29.95 | 6.79 | 78.35 | 1530 | 8391 |
| 1 | 40.44 | 9.92 | 109.93 | 2330 | 12086 |
| 2 | 31.52 | 6.39 | 91.36 | 1844 | 10007 |
| 3 | 36.93 | 9.66 | 101.29 | 2136 | 11892 |
| 4 | 28.71 | 8.24 | 91.45 | 1815 | 9569 |

Mean end-to-end task wall time was 33.51 seconds. Total predicted tokens across
the suite were 9655; total prompt tokens were 51945.

## Command

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --benchmark-suite objective-smoke \
  --branch-slots 0-11 \
  --baseline-tokens 260 \
  --branch-tokens 220 \
  --verify-tokens 120 \
  --final-tokens 260 \
  --score-tokens 120 \
  --repair-rounds 1 \
  --objective-repair-rounds 1 \
  --out-dir /tmp/turbo-speculative-breakout-objective-12
```

## Notes

- This is a real deterministic benchmark smoke, not a production benchmark.
  Five tasks is too small to claim general quality.
- The current improvement is validator-assisted repair, not proof that branch
  fanout alone fixes arithmetic. The artifact shows every arithmetic branch
  still made the same mistake.
- Prefix clone was disabled because this run attached to an already-running
  server. Launch mode can use slot save/restore when the server has
  `--slot-save-path`.
- The objective suite now supports JSON/JSONL case files through
  `--benchmark-suite-file`, so larger validator-backed suites can be run without
  changing the harness.
