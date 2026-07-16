# B4 slot-arbitration protocol (queued; not run overnight by design)

Preregistered gates: `reports/treebeard-nxy-optimizer-preregistration-20260715.md`
section B4. Deferred to a daylight session: it needs a fresh multi-tenant load
driver, and building one at 4am produces sloppy evidence. Everything below is
ready to execute.

## Policy under test (zero server changes)

Idle-slot-only thinking client: poll `GET /slots`; fork branch families only
into non-processing slots; never start a wave while any request is queued;
commit or abort promptly.

## Setup

- Live surface: `turbo-statetree-rc6.service` on :8093 (build b9695-d794fd15d).
- Real-agent load: `scripts/turbo-multiagent-pareto.py` gives the request
  shape; drive a sustained closed-loop variant at 4, 8, and 11 occupied slots
  (extend it with a `--duration` closed loop, or loop repeats).
- Thinking task: `scripts/turbo-speculative-breakout.py --attach --port 8093
  --benchmark-suite objective-smoke --preserve-prefix-root` with
  `--branch-slots` restricted to the idle set discovered via `/slots`.
- Telemetry: `TREEBEARD_BATCH_SHAPE_PROF=1` on a diagnostic pass only.

## Gates (frozen 2026-07-15; B2 kill makes these the whole ballgame)

- G-B4a: real-agent p50 regression <= 1% at every load point (thinking on/off).
- G-B4b: thinking-task wall >= 15% better than single-pass at <= 8 occupied.
- G-B4c: fanout degrades to 0 at 11-12 occupied; no queued-request starvation.

## Context from B2 (results/treebeard-nxy-optimizer/20260715-221845-branch-cost)

Trunk per-stream loses 51% at N=3, so co-resident thinking WILL tax active
agents unless it truly only consumes idle slots; the c(N) table gives the
expected per-stream costs to validate the measurement against.
