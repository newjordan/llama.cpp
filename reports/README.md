# Treebeard reports index

R&D evidence narratives and handoffs. Completed/superseded work is grouped
under `archive/<date>-<theme>/`; documents tied to the current ship or an open
thread stay at top level. Raw run evidence lives in the treebeard-work
snapshot's `results/` tree (append-only, immutable) and is cited by path from
these reports; do not edit it.

Reorganized 2026-07-16. Seven forwarding stubs remain at their old top-level
paths because those paths are cited from the immutable `results/` trail, from
memory, or from a `scripts/` guard wrapper (paths that cannot be edited). Each
stub links to its archived location.

## Active (top level)

- `treebeard-moe-handoff-20260726.md` - **CURRENT handoff; read first.** MoE-down
  closed (cost model, measured 13.5% reuse ceiling), deferred-reduce kernel
  shipped default-ON 2026-07-26, remaining levers + the three measurement rules
  these sessions cost.
- `treebeard-moe-kernel-handoff-20260725.md` - RESOLVED, superseded by the above.
  Its absolute us/op figures are warmup-inflated; its shares are fine.
- `treebeard-fresh-session-handoff-20260716-rc8.md` - prior handoff. Production
  state through RC8, PCBT/StateTree threads, next moves.
- `treebeard-fresh-session-handoff-20260716.md` - prior handoff (stops at RC6);
  superseded, kept for provenance.
- `treebeard-b4-arbitration-protocol-20260716.md` - frozen B4 slot-arbitration
  protocol (open thread: abort-on-arrival successor lever).
- `treebeard-nxy-optimizer-preregistration-20260715.md` - N(X)+Y channel
  program preregistration (open thread: ragged activation-ratio heuristic).
- `treebeard-ragged-kv-promotion-20260715.md` - ragged-KV / state-io promotion
  narrative underpinning current production numerics.
- `treebeard-continuous-batch-shape-20260715.md` - T7 continuous-batch shape
  control (queued ledger lever).

## Archive map

- `archive/20260708-serving-speculative/` - pre-StateTree serving benchmarks,
  slot-fork, and the single-answer speculative-breakout foundation (07-08/09).
- `archive/20260710-statetree-durability/` - StateTree benchmark/acceptance/
  retention work and the RC1/RC2/RC3 releases (07-09..11). Stub at old path:
  `turbo-statetree-0.1.0-rc.3-release-20260711.md`.
- `archive/20260713-jspace-steering/` - J-Space steering G0/G1/G2/G2b
  exploration and the G2 session handoff (07-13..15). Stubs at old paths:
  `treebeard-jspace-g2-preregistration-20260715.md`,
  `treebeard-jspace-g2-development-stop-20260715.md`,
  `treebeard-jspace-g2b-preregistration-20260715.md`,
  `treebeard-jspace-g2b-embedding-resume-20260715.md`.
- `archive/20260714-throughput-kernel-rnd/` - MoE, single-token, single-
  wavefront, linear-attention, and serial-frontier throughput R&D (07-14/15).
  Stub at old path: `treebeard-moe-rnd-b70-evidence-20260714.md`.
- `archive/20260715-ragged-promotion/` - ragged state-io composition and
  recurrent state-io fusion supporting evidence, plus the 07-15 session
  handoff. Stub at old path: `treebeard-fresh-session-handoff-20260715.md`.
- `archive/20260715-parked-plays/` - Q8 split/XMX and quantized-KV serial STOP
  verdicts (parked; do not reopen without new evidence).
- `archive/20260714-pcbt-origin/` - the origin PCBT handoff (superseded by the
  canonical `docs/treebeard-proof-carrying-branch-transactions.md` and the
  RC8 handoff).
