# Turbo Speculative Breakout Suite - 2026-07-08

## Summary

Live run against the existing `:8093` 12-slot Turbo server showed breakout
answers scoring better than the single-pass baseline on all built-in smoke tasks.

```text
suite: /tmp/turbo-speculative-breakout-suite/20260708T223050Z-1780546.suite.json
jsonl: /tmp/turbo-speculative-breakout-suite/results.jsonl
```

Aggregate result:

| Metric | Value |
| --- | ---: |
| Tasks | 3 |
| Wins | 3 |
| Ties | 0 |
| Losses | 0 |
| Mean baseline score | 86.00 |
| Mean breakout score | 92.67 |
| Mean score delta | +6.67 |
| Min score delta | +6 |
| Max score delta | +7 |

Per-task result:

| Task | Baseline | Breakout | Delta | Artifact |
| ---: | ---: | ---: | ---: | --- |
| 0 | 88 | 95 | +7 | `/tmp/turbo-speculative-breakout-suite/20260708T223050Z-1780546.task000.json` |
| 1 | 82 | 88 | +6 | `/tmp/turbo-speculative-breakout-suite/20260708T223050Z-1780546.task001.json` |
| 2 | 88 | 95 | +7 | `/tmp/turbo-speculative-breakout-suite/20260708T223050Z-1780546.task002.json` |

Command:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --task-suite turbo-smoke \
  --branch-slots 0-3 \
  --baseline-tokens 240 \
  --branch-tokens 180 \
  --verify-tokens 110 \
  --final-tokens 240 \
  --score-tokens 130 \
  --repair-rounds 1 \
  --out-dir /tmp/turbo-speculative-breakout-suite
```

## Notes

- This validates the current orchestration harness on the smoke suite, not the
  hard acceptance problem in general.
- The scoring model is itself a model call. Use the JSON artifacts for audit and
  rerun with task suites closer to the target workload before treating this as a
  production gate.
- The stronger deterministic objective benchmark run is recorded separately in
  `reports/turbo-speculative-breakout-objective-core-20260708.md`.
- Prefix clone was disabled for this run because the live attached server was
  not launched with `--slot-save-path`. Launch mode enables prefix clone.
