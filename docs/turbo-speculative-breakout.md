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
2. Clone it with either file save/restore or the in-memory unified-KV fork API.
3. Reserve the forked slots until branch work completes, then erase the
   reservations even when a task fails.
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
12. With `--objective-fast-path`, skip model verifier and recombination when a
    deterministic branch passes, then escalate to verifier plus recombine only
    for unresolved validator failures.
13. With objective fast path, `--no-score-final`, and an exact-output validator,
    return a terminal-valid baseline without launching branches.
14. With `--adaptive-fanout`, launch cumulative branch widths such as 1, 2, 4,
    and 8, validate after each wave, and stop at the first passing branch.
15. With the fork backend and `--preserve-prefix-root`, keep one untouched
    StateTree root, collapse failed waves back to it, refork the next wave with
    a generation fence, and commit a passing branch atomically.

The result JSON records branch outputs, verifier reports, parsed scores,
`prefix_ok` decisions, selected branch metadata, timings, the final answer,
baseline answer, final scores, and `score_delta` (`breakout - baseline`).
`final_versions` keeps the initial recombination, selected-branch candidate, and
any repair attempts. Each request stores both cleaned `content` and
`raw_content` so stop/tag cleanup can be audited.

Schema-v3 request telemetry records exact returned token IDs plus raw-content,
cleaned-content, and token-sequence SHA-256 hashes. It validates explicit slot
identity and returned token count, and records `cache_n` separately from the
processed `prompt_n`. Decision telemetry records
whether a baseline short-circuited, the number of branches and actual fanout
waves, and whether timing belongs to all tasks or only escalated tasks.
`final_system_pass_rate` includes terminal
baseline answers; `final_multipass_pass_rate` includes only tasks that launched
branches. An all-short-circuit suite therefore has no multipass pass rate.

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
  --no-score-final \
  --objective-fast-path
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

Only `json_exact` and `lines_exact` can terminate before fanout. `json_exact`
requires one strict JSON document, rejects duplicate keys and wrapper prose,
and compares nested JSON types exactly (`true` is not `1`). The terminal gate
checks cleaned `content`, which is the answer returned by the harness;
`raw_content` remains available to audit whether cleanup removed a label or
model tag.

For equal-length throughput lanes, pass `--ignore-eos --no-stop-strings`. The
first option prevents EOS termination; the second sends an explicit empty stop
list, which disables server command-line reverse prompts for those requests.
Both defaults preserve normal task behavior.

Focused harness regression tests:

```bash
python3 tests/test_turbo_speculative_breakout.py
```

Attach to an existing server that was started without `--slot-save-path`:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --no-prefix-clone \
  --task "Write a short plan for measuring branch acceptance."
```

File-based prefix cloning requires the server to be started with
`--slot-save-path`. Launch mode does this automatically. This remains the
default A/B backend. Managed launch also adds `--no-cache-idle-slots`; an
attached A/B server must use it too, or ordinary file-restored slots can be
moved into prompt RAM cache and cleared while fork-reserved slots survive.

Use the server-native in-memory backend with a Turbo server that supports slot
forking:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8098 \
  --prefix-clone-backend fork \
  --branch-slots 0-11 \
  --task-file /tmp/task.txt
```

The server must use unified KV (`-kvu`) and expose `GET /slots` (`--slots`) so
the harness can audit and release reservations. Managed launch mode adds
`--slots` automatically. Fork mode does not require `--slot-save-path`.
Use a dedicated serving endpoint: the harness explicitly owns every slot in
`--branch-slots` while a task is running.

The task artifact records `prefix.backend`, client-observed `clone_wall_s`, the
fork response and `fork_ms`, exact returned token IDs and hashes, `cache_n`, and
any cleanup errors. Forked slots reject automatic scheduling but remain
available to explicit `id_slot` requests. Context shift and cache-reuse shifts
are disabled while state is shared. Inherited checkpoints are discarded;
checkpoints created after divergence remain sequence-local so hybrid/recurrent
models can retain their own branch prefix without mutating siblings.

Run validator-first escalation while preserving a transactional prefix root:

```bash
python3 scripts/turbo-speculative-breakout.py \
  --attach \
  --port 8093 \
  --prefix-clone-backend fork \
  --preserve-prefix-root \
  --adaptive-fanout \
  --fanout-stages 1,2,4,8 \
  --benchmark-suite objective-core \
  --branch-slots 1-8 \
  --prefix-slot 0 \
  --final-slot 9 \
  --no-score-final \
  --objective-fast-path
```

In this mode the prefix slot is excluded from `--branch-slots`. Each stage
forks only its new branch slots. A failed stage commits the untouched root and
uses its returned `fork_id` to fence the next refork. A passing stage commits
the first validator-passing branch. Final cleanup still erases the committed
singleton so a benchmark task cannot leak a reservation into the next task.

For controlled A/B resets, verify every requested slot reports
`is_processing=false`, `is_reserved=false`, `fork_source_id=-1`,
`n_prompt_tokens=0`, and `n_prompt_checkpoints=0`. The server clears checkpoint
metadata on erase and successful file restore; older builds did not.

The first controlled 35B/B70 gate is recorded in
`reports/archive/20260708-serving-speculative/turbo-slot-fork-20260709.md`. After fixing stale file-lane checkpoint
metadata, the clean rerun made clone wall time 28.9x faster, kept an equal-token
12-way branch wave effectively flat, and improved complete flow wall by 2.17%.
Clone latency, branch throughput, full-flow wall, and output quality remain
separate gates.

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

The current historical artifact is the optimized 11-case `objective-core` run:

```text
/tmp/turbo-speculative-breakout-objective-core-fast-current2/20260709T024357Z-2058197.suite.json
```

It has baseline 5/11 pass, final breakout 11/11 pass, zero objective losses,
mean deterministic score delta +35.91, branch fanout 86.48 predicted tok/s,
multipass core 82.84 predicted tok/s, and mean multipass core wall time 7.29s.
The previous full-verifier path also reached 11/11, but took 21.30s mean
multipass-core wall time; the fast path is a 65.77% wall-time reduction while
preserving final deterministic accuracy on this suite.

That artifact predates schema v2, strict type-sensitive `json_exact`, and the
baseline terminal gate. It remains evidence for the older fast-path mechanics,
not evidence for the new shortcut or reporting fields.

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
