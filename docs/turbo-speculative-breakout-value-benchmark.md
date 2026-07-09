# Turbo Speculative Breakout Product-Value Benchmark

## Purpose

The `objective-core` suite validates the harness. It does not measure product
value. Product value requires tasks that represent the work a user would
actually pay for or depend on.

This benchmark layer should answer:

- Does breakout mode produce better usable work than a single-pass answer?
- Is the improvement large enough to justify extra tokens, latency, and
  orchestration complexity?
- Can users audit why the final answer was accepted?

## Required Task Shape

Each task should have:

- A realistic user request from a target workflow.
- A single-pass baseline artifact.
- A breakout artifact.
- A deterministic or semi-deterministic acceptance gate.
- A cost and latency record.
- A short human-review note when deterministic validation is insufficient.

Good product-value tasks are not generic puzzles. Examples:

- Patch a real failing test in a local repo, then run the test.
- Summarize a real benchmark log and extract the next action with cited lines.
- Convert a real config or report into a schema that downstream code accepts.
- Diagnose a real server trace and propose a concrete remediation.
- Generate a PR review where findings are checked against source lines.

## Acceptance Metrics

The minimum report should include:

| Metric | Meaning |
| --- | --- |
| Task pass rate | Final answer passes the task-specific gate. |
| Baseline pass rate | Single-pass answer passes the same gate. |
| Net win rate | Breakout pass and baseline fail, minus breakout fail and baseline pass. |
| Severity-weighted wins | Wins weighted by user impact, not just count. |
| Mean latency | End-to-end wall time per task. |
| Mean token cost | Prompt and predicted tokens per task. |
| Auditability | Whether the artifact explains selected branches and repairs. |
| Human override rate | Cases where deterministic validation passed but reviewer rejected. |

## Acceptance Bar

Treat the feature as product-promising only if a representative suite shows:

- At least 30 real workflow tasks.
- No safety-critical regressions.
- Breakout beats baseline by at least 15 percentage points on task pass rate.
- Net win rate is positive after human review.
- Median added latency is acceptable for the workflow.
- Every accepted final answer has an auditable branch or repair trail.

## Current Status

Current evidence:

- `objective-core` is a valid harness benchmark.
- It is not a product-value benchmark.
- The next meaningful milestone is a workflow suite with real artifacts and
  task-specific gates.

Until that exists, the honest claim is:

```text
The breakout harness can run, validate, repair, and audit deterministic tasks.
It has not yet demonstrated product value on representative user workflows.
```
