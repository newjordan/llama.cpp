# Treebeard fresh-session handoff (2026-07-16, post-RC8)

Supersedes `reports/treebeard-fresh-session-handoff-20260716.md` (that one
predates RC7/RC8; it stops at RC6). Read this one first.

## Start here

Production advanced to RC8 today (15:52 CDT). RC8 promotes the PCBT-4..8
transaction lifecycle from a bench capability to a live production surface on
top of the already-shipped ragged-KV + state-I/O fusion RC5..RC7 lineage.

- **Production NOW: `turbo-statetree-rc8.service`, :8093, build
  `b9743-c7091b65b`.** Alias
  `turbo-statetree-0.1.0-rc.8-Qwen3.6-35B-A3B-Q5-c262144-np12-ragged-pcbt`.
- **Ship env:** `LLAMA_KV_TREE_RAGGED=1 GGML_SYCL_ENABLE_STATE_IO_FUSION=1
  GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0 TREEBEARD_PCBT_ENABLE=1`.
- **Rollback chain:** rc8 -> rc7 (`b9723-356cefa61`, PCBT present but gated
  off) -> rc6 (`b9695-d794fd15d`) -> rc5 (`b9680-de0834ca0`) -> rc4
  (`b9627-3fcf1c626`). All systemd user units retained.
  `systemctl --user disable --now turbo-statetree-rc8 && systemctl --user
  enable --now turbo-statetree-rc7`.

Read these first:

1. `results/treebeard-rc5-deploy/20260716-155210-rc8-production/deployment-summary.md`
   - the RC8 flip record, gate evidence, live verification at flip.
2. `results/treebeard-ragged-promo-b70/build-freeze-rc8-20260716.md` - the
   pin-by-hash freeze; server-source delta vs RC7 is tools/server only
   (`git diff db705ce67 c7091b65b -- src ggml common include` is EMPTY, so
   decode/attention numerics are the already-gated RC7 bytes).
3. `docs/treebeard-throughput-rnd.md` - canonical T-numbered lever ledger
   (T10 shipped; tail updated through RC8).
4. `docs/treebeard-proof-carrying-branch-transactions.md` +
   `docs/treebeard-pcbt-contract-v1.md` (frozen) - PCBT plan and contract.
5. `reports/treebeard-ragged-kv-promotion-20260715.md` - the ragged/state-io
   promotion narrative that underpins the current production numerics.

## How to check production

- `/props` on :8093 should show build `b9743-c7091b65b`, the rc.8 alias,
  12 slots, ctx 262144, and `pcbt {contract v1, enabled true}`.
- `/metrics` exposes `pcbt_*` counters; `/transactions` is live and issues
  `pcbt.receipt.v1` receipts.
- Runtime binary hashes (pin-by-hash): llama-server `393397fb...`,
  libllama `ec23a7cf...`, libggml-sycl `0a0536fd...` (all unchanged from the
  gated RC7 lineage), libllama-server-impl `5fa98bbd...` (NEW: PCBT-4..8 plus
  the /props fix). Full digests in the freeze record.

## What changed today

1. **RC8 shipped with full same-binary gate evidence on the shipped bytes:**
   - Dense golden 1/8/12 A/B/A: +2.63% / +16.30% / +19.16% p50 vs control
     midpoint, drift <= 0.28%, zero failures, empty kernel scan; the candidate
     arm ran the full ship env and its captured /props proved pcbt enabled
     during the gate
     (`results/treebeard-ragged-promo-b70/20260716-141226-golden-aba`).
   - Fragmented confirm + composition: C1 +57.8%, ship C2 +73.4%, state-io
     incremental +9.87%, drift 0.136%; ship-config token parity EXACT vs both
     A/A controls on all samples
     (`results/treebeard-ragged-promo-b70/20260716-142959-confirm-comp-aba`).
   - Edge probes: long-prefix ON parity true; commit-churn fresh and
     hash-identical over the PCBT-5/6 statetree_commit_family refactor.
2. **/props enable-gate truthfulness fix (`c7091b65b`)** - found via the RC8
   golden gate: the first golden run's candidate /props claimed
   `pcbt.enabled=false` while the route actually read the env gate (hardcoded
   literal vs real state). That run (20260716-134917, PASS) was superseded by
   the re-run on fixed bytes (20260716-141226). This is a JSON-literal /
   comment-text change only; decode numerics untouched.
3. **B4 idle-capacity thinking FALSIFIED** - the B4 slot-arbitration probe
   measured a +26.8% tax at 4 agents, killing the "free idle capacity"
   framing (`results/treebeard-b4/20260716-130842-b4/verdict.md`). The design
   conclusion is to pivot to **preemptive abort-on-arrival via PCBT-7
   fencing** instead of speculative idle fanout. The arbitration protocol is
   frozen (`reports/treebeard-b4-arbitration-protocol-20260716.md`) but the
   abort-on-arrival protocol/prereg is NOT yet written.
4. **PCBT-12 product benchmark machine pre-verification 30/30 PASS**, appended
   to `tests/pcbt/pcbt12-review-notes.md`. That file is **awaiting the user's
   human sign-off line** - do NOT modify it. PCBT-12 measured 93.3% vs 43.3%
   single-pass with 30 evidence-bound receipts; PCBT-11 B70 gate ran orch
   -0.457%, decode +0.001%, no-replay.

## Open threads (carry forward)

1. **User sign-off pending on `tests/pcbt/pcbt12-review-notes.md`** (machine
   pre-verification 30/30 PASS appended; sign-off line open). Do not edit.
2. **Next lever: preemptive abort-on-arrival via PCBT-7 fencing.** B4
   falsified the idle-capacity model (+26.8% tax at 4 agents). Protocol and
   preregistration not yet written; write them before any run. See the B4
   verdict above and `reports/treebeard-b4-arbitration-protocol-20260716.md`.
3. **Evaluator follow-up:** make all-arm token parity diagnostic-only for the
   attribution (non-ship) arms. The RC8 confirm run's gate-as-coded flagged a
   single-request, single-repeat self-inconsistency on the non-ship c1
   attribution arm; analyzed as known SYCL backend flutter (a6ec035bc
   precedent). See
   `results/treebeard-ragged-promo-b70/20260716-142959-confirm-comp-aba/verdict.md`.
4. **Ragged activation-ratio heuristic (RC7-era, still open as a refinement):**
   replace the boolean `plan.reduces_columns` with an env-tunable
   reduction-ratio threshold; gate with the branch-cost + confirm guards.
   See `reports/treebeard-nxy-optimizer-preregistration-20260715.md`.
5. **T7 continuous-batch shape control** (ledger) - unchanged, still queued.
   See `reports/treebeard-continuous-batch-shape-20260715.md`.

## Parked plays (do NOT reopen without genuinely new evidence)

- Token-level speculative verification (three park verdicts stand; ragged-KV
  changes multi-sequence indexing, not wide-batch ubatch numerics).
- Q8 ncols weight-hoist (RESOLVED-PARKED, compiled default-off).
- In-server adaptive fanout controllers (B1 STOP; client-side fixed fanout
  only).
- B2/B4 idle-capacity "free thinking channels" (B2 KILL, B4 falsified). The
  abort-on-arrival lever above is the sanctioned successor, not a reopening.

## Recommended next moves, in priority order

1. Obtain the user's sign-off on `tests/pcbt/pcbt12-review-notes.md`, then
   close PCBT-12.
2. Write the abort-on-arrival (PCBT-7 fencing) protocol + preregistration
   with preregistered gates before touching the rig. This is the sanctioned
   next lever.
3. Land the evaluator change making all-arm parity diagnostic-only for
   attribution arms (removes the c1-flutter false-flag from the promotion
   gate as-coded).
4. If a small measured RC9 is wanted, take the ragged activation-ratio
   heuristic through the branch-cost + confirm guards.

## Operating notes

- All work is on `agent/treebeard-single-wavefront`, pushed only to
  `turbo-private` (never origin; GitHub Actions verified disabled before each
  push). Every assisted commit carries `Assisted-by:`.
- Guards: always pass `TREEBEARD_LIVE_SERVICE` explicitly; never pipe guard
  scripts on invocation (masks exit codes); never assert on INFO-level
  llama-server log lines (INFO internals are suppressed) - use W-level lines,
  raw fprintf traces, or result JSONs.
- Never `pkill`; stop/restore exact systemd user units only. `cmake --install`
  strips RUNPATH, so release units need `LD_LIBRARY_PATH` (the packaging
  script handles it, `TREEBEARD_RC`-parametrized).
- Evidence under `results/` is append-only and immutable; cite it, do not edit
  it. The reports/ tree was reorganized on 2026-07-16 - see
  `reports/README.md` for the active-first map and archive layout.
