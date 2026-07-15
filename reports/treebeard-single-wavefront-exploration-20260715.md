# Treebeard Single-Answer Wavefront Exploration - 2026-07-15

## Outcome

The local evidence supports a concrete single-agent architecture:

```text
one preserved trunk
  -> fork only the next useful work wave
  -> evaluate the wave in the efficient multi-column shape
  -> validate candidates or future tokens
  -> atomically commit one continuation
  -> collapse everything else
```

This branch implements and exercises both forms of the wave:

1. Target-verified future-token waves with a per-request proposal-width cap.
2. Validator-first semantic waves with adaptive 1/2/4/8 fanout.

The quality path is already positive on the local objective-core harness. An
adaptive A/B/A comparison retained 11/11 final validator passes while reducing
branch work by 45.0% and multipass-core mean wall by 20.46% relative to fixed
eight-wide fanout.

The speed path is functionally proven but not yet measured on the B70 target.
A CPU-only 0.8B gate returned exact greedy token parity in all 54 samples. On a
high-coverage structured-copy case, verified lookahead increased throughput
from 26.10 tok/s at width 0 to 53.85 tok/s at width 48. Cases that generated no
drafts remained approximately flat. These CPU numbers prove the controller and
verification shape, not a production speed claim.

RC4 production remained active throughout. It was not restarted or modified.

## Branch and build identity

- Branch: `agent/treebeard-single-wavefront`
- Base: `3fcf1c626ba290107b6264827b47d5888bbb7e43`
- Candidate build directory:
  `/home/frosty40/turbo/treebeard-work/build-treebeard-single-wavefront`
- Candidate `llama-server` SHA-256:
  `393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c`
- RC4 remained `active`, with `NRestarts=0`, after all live gates.

No commit or push was made.

## What the existing information was saying

The 12-agent result exposes a shape preference, not an agent preference. The
GPU is much more productive when independent columns are visible, while one
autoregressive answer exposes one serial token at a time. The transfer problem
is therefore to manufacture useful independent columns without creating
independent final answers.

The existing system already had the two required halves:

- RC4 makes real multi-token and multi-row MoE shapes efficient.
- StateTree can fork, refork, collapse, and commit sequence state cheaply.

The missing abstraction was a candidate state transition. A candidate can be
either a proposed future token block or a semantic branch. Both are useful only
when independent evidence can decide what to commit.

The rejected RC4 paths also constrain the design. Work must stay wide in both
rows and columns. Removing intermediate data is insufficient when it destroys
parallel structure. That is why the branch does not add a large decode
megakernel or an unconditional 12-way sampler.

## Implemented speed surface

### Per-request width cap

The completion API now accepts:

```json
{"speculative.n_max": 12}
```

The value caps the proposal returned by any configured speculative
implementation:

- `-1`: use the server-configured maximum.
- `0`: disable speculative proposals for this request.
- Positive value: cap each proposal round at that width.

This makes width 0 and widths 1/2/4/8/12/24/48 comparable on one warmed server.
No model reload is needed between the control and candidates.

The response timing object also records:

```text
draft_n
draft_n_accepted
draft_n_per_round
draft_n_accepted_per_round
```

The per-round arrays expose the actual verifier shape and committed work rather
than only a final acceptance ratio.

### Width sweep harness

`scripts/treebeard-wavefront-sweep.py`:

- Brackets a rotated candidate order with width-0 controls.
- Uses greedy sampling and requires exact returned token parity.
- Records proposal coverage, acceptance, round widths, wall time, and throughput.
- Separates summaries by workload so a high-coverage copy case cannot hide a
  no-coverage prose case.

### CPU functional gate

Artifact:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/cpu-width-sweep-clean-r2.json
```

Configuration:

- Qwen3.5 0.8B Q8, CPU-only, 15 threads.
- One slot, 8K context.
- `ngram-simple`, lookup width 4, configured proposal maximum 48.
- 64 generated tokens, two repeats, A/B/A-style width-0 bracketing.
- 54 total measured requests.

All widths preserved exact greedy token IDs. All proposal rounds respected the
request cap, and proposal/acceptance arrays had one-to-one round alignment.
`speculative.n_max=-2` was correctly rejected with HTTP 400.

Structured-copy result:

| Width | Throughput | Gain vs width 0 | Acceptance | Draft rounds |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 26.10 tok/s | control | n/a | 0 |
| 1 | 34.31 tok/s | +31.47% | 100.00% | 40 |
| 2 | 35.91 tok/s | +37.61% | 96.43% | 28 |
| 4 | 42.94 tok/s | +64.56% | 100.00% | 16 |
| 8 | 46.75 tok/s | +79.15% | 97.30% | 10 |
| 12 | 47.75 tok/s | +82.99% | 100.00% | 8 |
| 24 | 52.39 tok/s | +100.75% | 100.00% | 4 |
| 48 | 53.85 tok/s | +106.37% | 100.00% | 2 |

The code-edit and free-prose cases generated no proposals. Their width results
stayed within roughly -1.80% to +0.47% of their controls. This is the most
important policy observation: width selection matters only after a proposal
exists. The first controller decision is coverage, not width.

## Implemented quality surface

### Adaptive fanout

The breakout harness now supports cumulative stages such as:

```text
1 -> 2 -> 4 -> 8
```

Those values are cumulative branch counts. New work per wave is therefore
1, 1, 2, and 4 branches. Each wave is validated immediately. A passing branch
stops escalation and is returned directly. Model recombination and repair are
used only after direct validation fails.

### A/B/A result against fixed eight-wide fanout

Artifacts:

```text
A1  /home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/adaptive-objective-core/20260714T235952Z-3808442.suite.json
B   /home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/fixed8-objective-core/20260715T000137Z-3810901.suite.json
A2  /home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/adaptive-objective-core-a2/20260715T000404Z-3815947.suite.json
```

All three runs returned 11/11 final validator passes with zero objective losses.

| Metric | Adaptive A1 | Fixed 8 | Adaptive A2 | Adaptive midpoint | Midpoint vs fixed 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Branches launched | 39 | 80 | 49 | 44 | -45.00% |
| Branch wall/task | 3.305 s | 5.423 s | 4.720 s | 4.013 s | -26.00% |
| Multipass-core mean | 6.301 s | 7.959 s | 6.360 s | 6.331 s | -20.46% |
| Multipass-core median | 2.344 s | 5.957 s | 5.221 s | 3.782 s | -36.51% |
| Full-system mean | 6.857 s | 8.664 s | 6.982 s | 6.919 s | -20.14% |
| Final passes | 11/11 | 11/11 | 11/11 | 11/11 | equal |

Adaptive scheduling launched more waves: midpoint 27 versus 11 for fixed eight.
That is the intended trade. Easy tasks stop after one or two candidates. Hard
tasks pay sequential wave latency. The aggregate suite still improved because
the easy-task savings were larger than the escalation cost.

This is harness evidence, not a product-value claim. The suite is deterministic
and useful for controller correctness, but it is not the required 30-task real
workflow suite.

## Transactional StateTree wavefront

`--preserve-prefix-root` changes adaptive fork mode from "clone all slots, then
decide" to a state transaction:

1. Evaluate one prefix root and leave it untouched.
2. Fork only the slots needed for the next wave.
3. Run and validate those branches.
4. If none passes, commit the untouched root, clearing the failed branches.
5. Refork the next wave with the returned `fork_id` as a generation fence.
6. If a branch passes, commit that branch as the single surviving node.
7. Erase the benchmark singleton during cleanup.

The objective-smoke gate completed five tasks across 12 fork/commit generations.
Reforks carried the expected `parent_fork_id`, validator-passing branches became
the commit source, all five final answers passed, cleanup reported no errors,
`GET /slots` showed no processing or reserved slots, and `GET /states` showed no
live families afterward.

Artifact:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/preserved-root-objective-smoke/20260715T000800Z-3820420.suite.json
```

The full objective-core transactional run also completed with 11/11 final
passes and zero objective losses:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront/preserved-root-objective-core/20260715T000844Z-3821342.suite.json
```

It launched 55 semantic branches across 30 StateTree fork/commit transactions.
Fork latency was 1.210 ms p50, 2.474 ms mean, and 7.171 ms maximum. Commit
latency was 6.245 ms p50, 9.949 ms mean, and 27.255 ms maximum. The higher
commit cost includes clearing loser state. The run ended with zero reserved or
processing slots, zero live families, and zero cleanup errors.

## What is now clearer

### One answer needs two independent controllers

The speed controller asks:

```text
Is there a proposal, and which width maximizes committed tokens / verifier wall?
```

The quality controller asks:

```text
Has independent evidence already proved a candidate, and if not, which new
failure coordinate deserves the next branch wave?
```

Neither controller should optimize raw generated tokens, branch count, or
acceptance alone.

### Preserve the trunk, spend only reversible work

The StateTree gate demonstrates the stronger internal contract. The user sees
one continuous agent. Internally, speculative work is temporary and
generation-fenced. Only one state transition survives. This is more useful than
letting eight long-lived agents later negotiate a merged answer.

### Proposal coverage is the product boundary

When the n-gram source covered the output, wide verification more than doubled
CPU throughput. When it did not cover, the speculative system had no material
effect. The B70 experiment should therefore stratify by proposal coverage and
not report one blended number.

### The next B70 question is narrow

The remaining speed question is not whether verified lookahead can work. It is:

```text
For B70 RC4 at real context depths, what width maximizes accepted target tokens
per target verification wall on workloads where a draftless proposal exists?
```

That requires an isolated B70 candidate server with n-gram speculation enabled.
RC4 currently owns nearly all device memory, so running that gate requires a
controlled production interruption and exact restore. This branch did not infer
permission to interrupt the deployed service.

## Parked paths

- MTP remains quarantined and was not touched.
- The 0.8B CPU draft model remains rejected as a sequential production drafter.
- GPU draft placement remains outside the current VRAM contract.
- An unconditional 12-way semantic fanout remains rejected.
- The batch-1 expert epilogue remains profile-gated. No fresh RC4 critical-path
  profile was available without disturbing production, so no kernel was added.
- Jacobi/tree-attention work remains unnecessary until the B70 verifier curve
  is measured and proposal coverage, rather than verification cost, is proven
  to be the limit.

## Verification performed

- Candidate `llama-server` release build completed with Intel oneAPI/SYCL.
- Python harness tests: 19 passed.
- Width-sweep tests: 3 passed.
- `git diff --check`: passed.
- CPU width gate: 54/54 exact greedy parity.
- Per-round cap and telemetry alignment: passed for all 54 samples.
- Invalid negative request cap: HTTP 400 with the expected error.
- Adaptive quality A/B/A: 11/11 final passes in all three arms.
- Transactional StateTree smoke: 5/5 final passes and zero leaked families.
- Transactional StateTree core: 11/11 final passes across 30 fork/commit
  transactions and zero leaked families.
- RC4 post-gate health: active, zero restarts, `/health` OK.

## Next controlled gate

With an approved production maintenance window:

1. Stop RC4 through the existing guarded deployment workflow.
2. Launch this candidate from the exact RC4 base with `ngram-simple` configured
   to 48 and the rejected kernel paths still disabled.
3. Sweep request caps 0/1/2/4/8/12/24/48 at shallow, 32K, 128K, and 256K depth.
4. Run low-coverage prose, mixed code/edit, and high-coverage structured-copy
   cases separately.
5. Require exact greedy token parity, no hardware faults, and a same-binary A/B/A
   gain of at least 3% on an identified workload before adding an online width
   chooser.
6. Restore and verify the exact RC4 service whether the candidate passes or
   fails.
