# Turbo Experiment: Single-Answer Speculative Breakout

## Status

Prototype harness only. This is not server-side speculative decoding and it does
not solve the hard acceptance problem.

The harness is:

```text
scripts/turbo-speculative-breakout.py
```

It targets the Turbo serving shape:

```text
Qwen3.6-35B-A3B Q5_K_XL
Intel Arc Pro B70
SYCL
12-slot unified KV
262144-token context
```

## What It Does

The script uses multi-slot serving throughput to work on one user task:

1. Evaluate a shared prompt prefix with `n_predict=0`.
2. Save that slot state through `/slots/{id}?action=save`.
3. Restore the prefix into the remaining branch slots.
4. Run independent candidate completions across the branch slots.
5. Run one verifier prompt per candidate, also in parallel.
6. Recombine the branch packets into a single final answer.
7. Generate a single-pass baseline answer and score both baseline and breakout
   with the same rubric.
8. Score the selected branch as a final candidate too, so recombination cannot
   degrade a strong branch silently.
9. If breakout does not beat the baseline score, run a repair pass using the
   baseline, branch packets, and score reports, then keep the best-scored final.
10. In objective benchmark mode, validate baseline and breakout answers with a
    deterministic checker and optionally run one validator-feedback repair pass.
11. In objective benchmark mode, validate each branch before recombination and
    prefer validator-passing branches over branches that only have high model
    verifier scores.

The result JSON records branch outputs, verifier reports, parsed scores,
`prefix_ok` decisions, selected branch metadata, timings, the final answer,
baseline answer, final scores, and `score_delta` (`breakout - baseline`).
`final_versions` keeps the initial recombination, selected-branch candidate, and
any repair attempts. Each request stores both cleaned `content` and
`raw_content` so stop/tag cleanup can be audited.

Objective benchmark artifacts also include `objective_benchmark`, with baseline,
pre-repair breakout, and final breakout validator results. This is the preferred
path for acceptance testing because it can expose failures that the model scorer
misses.

## Example

Launch a dedicated 12-slot server and run one task:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --task-file /tmp/task.txt \
  --out-dir /tmp/turbo-speculative-breakout \
  --branch-slots 0-11
```

The run prints the final answer and writes a JSON artifact. In that artifact,
`final_scores.baseline.score`, `final_scores.breakout.score`, and `score_delta`
are the quick check for whether the breakout beat the current single-pass path.
Use `--repair-rounds 0` to disable the feedback repair pass.

Run the built-in comparison suite:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --task-suite turbo-smoke \
  --branch-slots 0-3
```

Suite mode writes one task artifact per task plus a `.suite.json` summary with
win/tie/loss counts and aggregate score deltas.

Run the built-in deterministic objective core suite:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --benchmark-suite objective-core \
  --branch-slots 0-11 \
  --no-score-final
```

Custom objective suites can be supplied as JSON or JSONL:

```json
{"id":"case-001","validator":"csv_json_transform","task":"Return only a JSON array..."}
```

Cases may include `repair_hint` for non-oracle repair guidance. The built-in
arithmetic smoke case uses this to say how to decompose the subtraction without
embedding the final expected JSON as the prompt answer.

Then run:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --benchmark-suite-file /tmp/objective-cases.jsonl \
  --branch-slots 0-11
```

Available validators in this prototype are `arithmetic_json`,
`schema_fields`, `five_experiments`, `csv_json_transform`, `rubric_signals`,
`json_exact`, and `lines_exact`.

Attach to an existing server that was started without `--slot-save-path`:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --task "Write a short plan for measuring branch acceptance."
```

Prefix cloning requires the server to be started with `--slot-save-path`. Launch
mode does this automatically.

## Acceptance Risk

The verifier is still a model call. It can miss subtle errors, over-score a
polished wrong branch, or reject a branch that has useful partial material. Treat
the score and `prefix_ok` as telemetry, not a proof.

The model scorer is also telemetry. The objective benchmark mode is stricter:
it records deterministic pass/fail checks and keeps model scores separate from
validator scores.

The 2026-07-08 objective run exposed a concrete failure: all arithmetic branches
and the model verifier agreed on `1024-37=995`. The deterministic validator
caught it, and the hint-aware objective repair corrected the final answer. This
is an improvement to benchmark-mode acceptance and repair; it is not evidence
that branch fanout alone solves arithmetic errors.

The current stronger artifact is the 11-case `objective-core` run:

```text
/tmp/turbo-speculative-breakout-objective-core/20260709T013917Z-1985364.suite.json
```

It has baseline 4/11 pass, final breakout 11/11 pass, zero objective losses,
mean deterministic score delta +45, branch fanout 85.88 predicted tok/s, and
multipass core 68.58 predicted tok/s.

This is still harness evidence, not product-value evidence. The product-value
benchmark requirements are tracked separately in:

```text
docs/turbo-speculative-breakout-value-benchmark.md
```

The useful acceptance bar for this experiment is:

- `score_delta` is positive against the single-pass baseline on tasks that
  benefit from search.
- Branch fanout plus verification beats one warmed single-slot answer on wall
  time for tasks that benefit from search.
- The recombined final answer is at least as correct as the best branch in human
  review.
- Low-scoring or `prefix_ok: no` branches are not copied into the final answer
  without repair.
- Result JSON is sufficient to audit why a branch was accepted.

Rejection bar:

- Verifier scores do not correlate with human review.
- Prefix save/restore overhead dominates branch work.
- Recombination frequently loses correct details from the best branch.
- The final answer becomes harder to audit than a normal single-slot answer.
