# Turbo Speculative Breakout Handoff - 2026-07-09

## Current State

The optimized objective fast path is the current valid benchmark for
single-agent multipass over 12 slots.

Fresh passing artifact:

```text
/tmp/turbo-speculative-breakout-objective-core-fast-current2/20260709T024357Z-2058197.suite.json
```

Result:

- Baseline: 5/11 deterministic passes.
- Final multipass: 11/11 deterministic passes.
- Objective losses: 0.
- Mean multipass-core wall time: 7.29s, down from 21.30s full-verifier path.
- Mean multipass-with-baseline wall time: 8.91s.
- Multipass-core predicted throughput: 82.84 tok/s.
- Branch fanout predicted throughput: 86.48 tok/s.
- Verifier/recombine fallback requests in the passing run: 0.
- Objective repairs accepted: 2.

## What Changed

- `--objective-fast-path` uses deterministic branch validation as the initial
  verifier in objective benchmark mode.
- Passing deterministic branches skip model verifier fanout and recombination.
- Model verifier plus recombine remains available as fallback for unresolved
  validator failures.
- Exact JSON repair now includes field-level feedback that names wrong current
  values without revealing expected values.
- Repair cleanup accepts model outputs that start with labels such as
  `Corrected answer:`.
- Summary artifacts now record fallback recombine and fallback repair counters.
- The report and docs now use `accuracy_throughput_summary` for accuracy and
  tok/sec interpretation.

## Failure Investigated

A fresh fast-path rerun initially failed the `matrix-json` case:

```text
/tmp/turbo-speculative-breakout-objective-core-fast-current/20260709T023855Z-2052283.suite.json
```

The failure repeated wrong `anti_diag` and `diag_delta` values. Model verifier
reports also accepted wrong arithmetic, so the fix was not to trust verifier
scores harder. The fix was to improve validator-guided repair feedback and make
the matrix repair hint identify the anti-diagonal cells without embedding the
expected final JSON.

Focused matrix retest passed:

```text
/tmp/turbo-speculative-breakout-matrix-fast-current/20260709T024331Z-2057692.json
```

## Reproduce

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

## Validation Run

- `python3 -m py_compile scripts/turbo-speculative-breakout.py`
- `python3 scripts/turbo-speculative-breakout.py --help`
- Full 11-case objective-core benchmark above.
- Artifact assertions: 11/11 final passes, zero losses, zero fallback
  recombines, mean core wall under 8s.
- `git diff --check`

## Next Optimization Targets

- Reduce branch fanout wall time, now the dominant cost.
- Add repeated-run variance tracking; baseline and branch outputs are sampled.
- Add harder objective suites that are not just exact transforms.
- Keep product-value evidence separate from harness evidence.
