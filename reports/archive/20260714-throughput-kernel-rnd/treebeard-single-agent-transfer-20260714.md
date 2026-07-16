# Treebeard Single-Agent Transfer Exploration - 2026-07-14

## Outcome

The twelve-agent throughput gain cannot be copied into one response by merely
occupying twelve slots. The fixed RC4 result is:

| Shape | Aggregate p50 | Per-stream p50 | Relative to one stream |
| --- | ---: | ---: | ---: |
| One active agent | 78.219 tok/s | 80.649 tok/s | 1.000x aggregate |
| Twelve active agents | 207.441 tok/s | 18.807 tok/s | 2.652x aggregate, 0.233x per stream |

Concurrency exposes a 2.65x aggregate compute envelope, but it makes each
individual stream about 4.3x slower. A single output can use that envelope only
if it creates useful parallel work inside the request:

1. Verify several proposed future tokens in one target-model call.
2. Explore several semantic continuations, validate them, and return one winner.
3. Convert another independent model dimension, such as the top-eight expert
   dimension, into parallel work without serializing the critical path.

The recommended next work is therefore two small, independent lanes:

- P0 speed: **Batch-Shape-Aware Verified Lookahead** using the existing
  draftless n-gram proposal paths and exact target verification.
- P0 quality: **Validator-First Branch Escalation** using StateTree fork/commit,
  deliberate branch diversity, deterministic evidence, and adaptive 1/2/4/8
  fanout.

The first new kernel candidate is a P1 one-token ordered expert epilogue. MTP,
the available CPU draft model, and unconditional twelve-way best-of-N should
remain parked.

## Evidence boundary

This is an architecture and experiment-order report, not a new performance
claim. It uses:

- The accepted RC4 same-binary A/B/A result in
  `reports/archive/20260714-throughput-kernel-rnd/treebeard-moe-rnd-b70-evidence-20260714.md`.
- The implementation ledger in `docs/treebeard-throughput-rnd.md`.
- The historical StateTree and speculative-breakout reports.
- The earlier B70 single-token fusion measurements in the local repository.
- A read-only feasibility measurement of the available Qwen3.5 0.8B Q8 draft
  model on the host CPU.
- Primary research references linked below.

No production configuration was changed and no RC4 service was restarted for
this exploration.

## What RC4 actually transfers

RC4's accepted T2 and T3 paths transfer three principles, not a special
twelve-agent algorithm:

1. **Materialize less.** T2 avoids the full routed expert-output tensor for
   batched down projection. T3 avoids both gate/up MMID outputs and the
   standalone SwiGLU pass.
2. **Keep independent work visible.** T3 retains expert and row parallelism.
   The rejected composite pipeline removed more traffic but lost row-level
   parallelism and regressed twelve-agent p50 by 6.65%.
3. **Specialize by useful shape.** The winning paths cover real token counts
   from 2 through 64. Grouped down projection and graph replay were not useful
   merely because they reduced launches or reused work.

For one active agent, T3 already applies at `n_tokens == 1`. T2 does not: the
integrated weighted down dispatcher explicitly requires at least two tokens,
and single-token decode retains the prior MMID plus ordered reduction path.
That boundary creates a small, concrete kernel opportunity, but it is not the
main route to a 2x single-stream improvement.

Earlier B70 profiling explains why. Large Q5/Q6 GEMVs were measured near the
device bandwidth roofline. The reclaimable work was serial chains of small
operations with no large operation available to hide submission. The already
landed single-token expert reduction, top-k MoE router fusion, and gated-delta
glue fusion came from that model. Generic layer megakernels and isolated glue
fusion are therefore low-priority unless a fresh RC4 profile contradicts the
old attribution.

## Speed lane: Batch-Shape-Aware Verified Lookahead

### Design

Use a nearly free proposal source to draft future tokens, then evaluate the
proposed block with the RC4 target in one causal verification batch. Start with
the n-gram implementations already present in `llama-server`:

- `ngram-mod` for a shared bounded hash pool.
- `ngram-simple` or `ngram-map-k4v` for repetition inside the current context.

The target model remains the authority. Accepted tokens use the existing
speculative verification and rollback machinery. This is the same lossless
principle as speculative decoding: the proposal source affects efficiency, not
the target distribution. The original speculative-decoding work establishes
the exact-distribution contract for a correctly implemented rejection sampler:
https://proceedings.mlr.press/v202/leviathan23a.html

The Treebeard-specific addition should be a small controller that chooses the
next proposal width from the measured RC4 verifier curve rather than using one
fixed maximum. Candidate widths are:

```text
4, 8, 12, 24, 48
```

For every request class and recent acceptance history, choose the width that
maximizes:

```text
expected accepted target tokens
-----------------------------------------------
draft time + target verify time + rollback time
```

Use hysteresis so a single rejection does not make the shape oscillate. Twelve
is important because it maps to the measured serving width. Twenty-four and
forty-eight are worth measuring because RC4 correctness and real-graph
activation already cover the 48-token path. They are not assumed to be faster.

This combines two established ideas in a target-specific way:

- Dynamic lookahead adapts the speculative depth rather than using a fixed
  draft length: https://proceedings.mlr.press/v262/mamou24a.html
- Hardware-aware tree selection chooses proposal topology from measured target
  costs: https://arxiv.org/abs/2402.12374

The novelty for this rig is that the cost model is trained on the exact B70,
Qwen3.6 MoE, RC4 fusion, and context-depth surface. It should include active
token count, context depth, recent acceptance, rollback rate, and verifier
wall time. It must not optimize acceptance rate alone.

### Why n-gram first

The deployed GGUF has no MTP/NextN layer. A previous MTP lane discovered that
constraint too late and is explicitly quarantined.

The available Qwen3.5 0.8B Q8 model was measured on the Ryzen 5950X with the
current RC4 build, 15 threads, CPU-only execution, 256 generated tokens, and
three repeats:

```text
29.601 +/- 0.115 tok/s
```

Reproduction command:

```bash
llama-bench \
  -m /home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf \
  -p 0 -n 256 -r 3 -t 15 -ngl 0 -dev none -fa off -o json
```

RC4 produces 78.219 aggregate wall tok/s at one active agent. Even with 100%
draft acceptance and free target verification, a sequential CPU drafter cannot
exceed its own 29.6 tok/s. Real verification only lowers that ceiling. The CPU
draft is therefore not a speed candidate.

GPU placement is not currently plausible without changing the deployment
contract. The production process reports about 32,297 MiB resident on a
32,656 MiB device, leaving roughly 359 MiB. The draft GGUF alone is about
774 MiB before KV and runtime allocations.

Draftless proposals avoid both failures. They should help most on source edits,
code, repeated structured output, summarization, and reasoning that restates
earlier material. They may provide no value on high-entropy prose. The feature
must remain request-adaptive. RC4 currently reports `speculative.types: none`,
so this is an isolated candidate configuration, not a description of the live
service.

### First gate

Use a guarded same-binary A/B/A isolated run. Do not change RC4 production.

1. Measure target verification wall for widths 1, 2, 4, 8, 12, 24, and 48 at
   shallow, 32K, 128K, and 256K context depths.
2. Confirm T2/T3 activation and collect accepted tokens, drafted tokens,
   rollback rows, graph rebuilds, kernel count, and inter-token latency.
3. Run low-, mixed-, and high-entropy SPEED-Bench categories plus local code
   edit, JSON transform, log summary, and reasoning cases.
4. Require greedy token parity and unchanged task-specific validator results.
   For sampled output, require the exact speculative-sampling contract and
   distribution/quality checks rather than assuming byte parity.
5. Advance only if one-agent p50 and mean improve at least 3% on an identified
   product workload, with no p95 inter-token regression above 2% and no
   validator regression.

Stop after the configuration and controller probe if verifier batching does
not move wall throughput. Do not build tree attention or a new draft model
before the linear, draftless path proves value.

## Quality lane: Validator-First Branch Escalation

### Design

Use the multi-slot envelope to improve one returned answer, but allocate slots
only where independent evidence can decide between candidates.

```text
single baseline
  -> deterministic or task-specific validation
  -> return immediately on pass
  -> otherwise fork 2 diverse branches
  -> validate in parallel
  -> return and commit the first proven winner
  -> otherwise escalate to 4, then at most 8
  -> targeted repair from compact failure packets
  -> commit one continuous StateTree node
```

This differs from unconditional best-of-N in four ways:

1. **Validation precedes fanout.** Easy tasks use one stream.
2. **Fanout is adaptive.** The prior 262K Pareto curve placed the operational
   knee at eight active agents, which retained 96.5% of twelve-agent aggregate
   p50 while improving per-agent decode by 1.452x. Twelve is a ceiling, not a
   default.
3. **Branches are intervention-diverse.** Use an independent decomposition,
   adversarial counterexample search, constraint audit, source-grounded check,
   or alternative algorithm. Temperature and seed changes alone do not create
   reliable independence.
4. **A passing branch is returned directly.** Recombination occurs only when no
   complete branch passes. This prevents a merger from degrading a strong
   answer.

Self-consistency and tree search establish that multiple reasoning paths can
improve difficult answers, but their value depends on selection:

- https://arxiv.org/abs/2203.11171
- https://arxiv.org/abs/2305.10601

Adaptive allocation is important because the best inference-time strategy
changes with task difficulty. Compute-optimal test-time scaling has been
reported to use inference compute more than four times as efficiently as a
fixed best-of-N baseline: https://arxiv.org/abs/2408.03314

### Local evidence

The historical objective-core breakout path improved deterministic passes from
5/11 for the baseline to 11/11 for the final multipass result. Its optimized
fast path reduced mean multipass-core wall from 21.30 seconds to 7.29 seconds
by skipping verifier/recombine work when a deterministic candidate passed.

That same evidence contains the key warning: all twelve arithmetic branches
made the same wrong calculation. A deterministic validator and targeted repair,
not branch count or model consensus, fixed the result. The transferable quality
mechanism is therefore:

```text
diverse search + independent evidence + localized repair
```

not:

```text
more samples + majority vote
```

### Branch packet contract

Every branch should emit a bounded packet rather than prose for the merger:

```json
{
  "candidate": "bounded final artifact or patch",
  "claims": [
    {"id": "c1", "statement": "...", "evidence": ["source:line-or-digest"]}
  ],
  "checks": [
    {"name": "schema", "status": "pass", "evidence_digest": "sha256:..."}
  ],
  "uncertainties": ["..."],
  "deferred_effects": []
}
```

The selector accepts claims supported by independent evidence, not merely the
most fluent packet. Contradictions trigger a narrow validator or repair request.
Speculative branches may describe tool/effect intents but must not execute
external side effects.

The planned Proof-Carrying Branch Transactions design is the right eventual
server boundary for exact branch and evidence digests, atomic winner commit,
and loser cleanup. It should not be implemented before the existing harness
passes the product-value gate.

### First gate

Extend the existing breakout product-value suite, not the objective smoke
suite alone:

- At least 30 real repository, trace, config, review, and structured-output
  tasks.
- Baseline, adaptive branch, and fixed fanout controls.
- Task-specific validation plus human review where validation is incomplete.
- Pass rate, net wins, severity-weighted wins, p50/p95 latency, generated token
  cost, fanout distribution, early-exit rate, correlated-failure rate, repair
  rate, and recombination-loss rate.
- Advance at +15 percentage points task pass rate, no safety-critical loss,
  positive human-reviewed net win rate, and an acceptable declared latency
  budget.

## P1 kernel lane: Expert-as-Lanes Ordered Epilogue

### Opportunity

The accepted T2 weighted down-projection path requires `n_tokens >= 2`. For one
token, the backend still computes eight selected expert rows and then performs
the existing ordered weighted reduction. That is the only obvious part of the
accepted T2 idea that has not crossed the batch-1 boundary.

### Proposed one-token topology

Map one workgroup to one output row and one subgroup to each of the eight routed
experts:

```text
8 subgroups compute 8 expert dots in parallel
  -> each subgroup publishes one float to 8-float local storage
  -> workgroup barrier
  -> subgroup 0 applies routing multiplies in exact slot order
  -> one final global output write
```

This removes the `[n_embd, 8]` global expert-output write/read and the separate
reduction submission while retaining expert parallelism. It explicitly avoids
the rejected composite pipeline's mistake: reducing row-level parallelism in
exchange for a larger fusion.

The correctness contract must preserve the current per-expert dot result, the
separate multiply rounding point, and ordered top-eight accumulation. It needs
Q5_K/Q6_K/Q8_0 coverage, alias guards, an opt-out control, and a fallback.

### Gate and stop condition

First profile the current RC4 batch-1 graph. Implement this kernel only if the
materialized down output and ordered reduction remain visible on the critical
path. Advance only with:

- Focused CPU-reference parity for one-token top-eight production shapes.
- Real Q5/Q6 graph activation.
- At least 5% improvement in the isolated down subgraph.
- At least 1% one-agent p50 and mean end-to-end improvement over A/B midpoint.
- No twelve-agent regression larger than 0.5%.

If the isolated subgraph moves but wall throughput does not, park it immediately.

## P1 policy lane: Evidence-Triggered Reasoning Budget

The MTP postmortem contains a separate useful result: a 64-token adaptive
reasoning budget matched unlimited reasoning on 11/12 panel points while
reducing wall time by 48.6%. That is not a broad quality claim, but it suggests
a cheap policy experiment:

1. Run a bounded first solution.
2. Validate it.
3. Spend additional reasoning tokens or branch fanout only on failed checks,
   high uncertainty, or high-impact tasks.

Saved tokens can fund targeted counterexample branches without increasing the
median task budget. This should be tested as one policy arm in Validator-First
Branch Escalation, not as an independent server subsystem.

## Research-only fallback: Jacobi/lookahead decoding

If draftless n-gram verification proves that widths 8 to 48 are efficient but
coverage is too low, training-free lookahead decoding becomes relevant. It
uses parallel Jacobi iterations to propose and verify future n-grams without a
separate draft model and reports exact greedy decoding:
https://proceedings.mlr.press/v235/fu24a.html

It maps conceptually to the B70's wide-token advantage, but it is a much larger
runtime change than enabling n-gram speculation. It should receive only a
bounded microprobe after these prerequisites are measured:

- The target verifier curve is favorable.
- N-gram acceptance, not verification cost, is the limiting factor.
- A minimal greedy-only prototype has a plausible speed model before core KV
  or attention code changes.

Do not begin a full Jacobi or tree-attention implementation on architectural
appeal alone.

## Parked and rejected directions

| Direction | Decision | Target-specific reason |
| --- | --- | --- |
| Production MTP | Park | The production GGUF has no MTP/NextN layer; the previous proxy lane is quarantined. |
| Qwen3.5 0.8B CPU drafter | Reject for speed | 29.60 tok/s is below the 78.22 tok/s target before verification cost. |
| Qwen3.5 0.8B GPU drafter | Park under current config | About 359 MiB reported headroom versus a 774 MiB GGUF before KV/runtime allocations. |
| Unconditional 12-way best-of-N | Reject as default | It maximizes aggregate tokens, not single-answer latency, and amplifies correlated errors without independent validation. |
| Duplicate or padded virtual agents | Reject | Duplicate rows do not resolve the autoregressive dependency and consume verifier work without proposing useful future tokens. |
| Whole-MoE composite pipeline | Reject | Measured -6.65% twelve-agent p50 and -0.29% one-agent p50. |
| SYCL graph replay | Reject | Measured -4.57% one-agent and -4.24% twelve-agent p50. |
| Grouped ordered down | Reject as default | Measured +0.06% one-agent and -0.26% twelve-agent p50. |
| Generic layer megakernel | Low priority | Prior profiling placed large GEMVs at the execution floor; saved isolated launches often hide behind them. |
| Medusa/EAGLE-style trained heads | Research only | They require target-specific training/artifacts not present for this production model. |

## Ordered execution plan

### ST-0 - Measure the exact transfer surface

- Add diagnostic-only RC4 critical-path telemetry for batch 1 and verifier
  widths 2/4/8/12/24/48.
- Record kernel count, queue gaps, bytes, T2/T3 activation, verifier wall, and
  rollback cost at multiple context depths.
- Time box: 60 minutes. No implementation beyond telemetry.

### ST-1 - Run draftless verified-lookahead A/B/A

- Use existing n-gram implementations with fixed-width controls first.
- Run matched low/mixed/high-entropy and product workflow cases.
- If positive, add the smallest controller that chooses among measured widths.
- Stop if the fixed-width sweep has no workload with at least 3% one-agent gain.

### ST-2 - Run adaptive quality fanout in the existing harness

- Add baseline-first validation and 1/2/4/8 escalation.
- Add intervention-diverse branch prompts and direct return of a valid branch.
- Measure against baseline and fixed 4/8/12 fanout on at least 30 real tasks.
- Do not implement the PCBT server API until this gate passes.

### ST-3 - Profile, then decide on the one-token down epilogue

- Implement only if ST-0 shows a live batch-1 down/reduce opportunity.
- Keep the first slice below 500 changed lines and one production shape.
- Stop at the isolated kernel gate before expanding coverage.

### ST-4 - Reassess

Choose the next step from measured failure mode:

- Good verifier curve, low n-gram coverage: bounded Jacobi/lookahead probe.
- Good quality wins, excessive latency: improve early exit and validator cost.
- Batch-1 kernel still critical: continue only with a new profile-selected
  serial chain or memory handoff.
- No positive lane: preserve RC4 and stop.

## Final recommendation

The highest-value transfer is not to make one answer imitate twelve independent
clients. It is to turn otherwise wasted parallel width into **verified future
tokens** for speed and **evidence-producing alternative continuations** for
quality.

Start with the two mechanisms already closest to proof on this rig:

1. Draftless target verification, because RC4 already accelerates real 2-to-64
   token MoE shapes and it requires no incompatible model artifact.
2. Validator-first adaptive StateTree branching, because local evidence already
   shows that validation and repair can turn parallel attempts into a better
   single answer, while also showing that consensus alone fails.

Keep the experiment budget short, preserve exact controls, and require a
product workload win before expanding either into a new server subsystem.
