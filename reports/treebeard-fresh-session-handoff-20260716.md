> SUPERSEDED by `reports/treebeard-fresh-session-handoff-20260716-rc8.md`
> (this handoff stops at RC6; production is now RC8). Kept for provenance.

# Treebeard fresh-session handoff (2026-07-16, post-RC6)

## Start here

Overnight (2026-07-15 19:00 → 07-16 00:30 CDT) the ragged-KV promotion
campaign ran end-to-end and production advanced two RCs:

- **RC5** (21:52) shipped sequence-ragged StateTree KV attention +
  recurrent state-I/O fusion.
- **RC6** (23:39, current) fixed a fork-surface graph-build crash the B2
  fanout sweep exposed in RC5, re-gated everything, and closed every edge
  probe. **Production: `turbo-statetree-rc6.service`, :8093, build
  `b9695-d794fd15d`, alias `...rc.6-...-ragged`.** Rollback chain rc6 → rc5
  (`b9680-de0834ca0`) → rc4 (`b9627-3fcf1c626`), all units retained.

Read these first:

1. `results/treebeard-ragged-promo-b70/decision.md` — the T10 ADVANCE
   decision with all gate tables.
2. `results/treebeard-rc5-deploy/20260715-233939-rc6-production/deployment-summary.md`
   — why RC6 exists and its re-gate evidence.
3. `reports/treebeard-ragged-kv-promotion-20260715.md` — full evidence
   narrative.
4. `reports/treebeard-nxy-optimizer-preregistration-20260715.md` +
   `results/treebeard-nxy-optimizer/*/verdict.md` — the N(X)+Y channel
   program preregistration and its B1/B2/B3 verdicts.
5. `docs/treebeard-throughput-rnd.md` — ledger T10 (shipped) and the B2
   compact-layout finding.

## Headline numbers (all guarded same-binary A/B/A, exact output parity)

- Dense golden shape vs RC4-equivalent control: **+19.2% aggregate @12
  agents, +16.5% @8, +3.9% @1** — entirely attributable to state-I/O fusion
  (stio-only attribution arm; its dense effect had never been measured).
- Fragmented multi-branch: ship config **+73.2%** vs both-off midpoint
  (ragged alone +59.1%, state-io incremental +8.9%).
- B3 semantic fanout on live RC6: validator passes **5/11 → 10/11**, mean
  score 64 → 95, zero losses, zero slot leaks, ~5 min for the 11-task suite.

## Closed questions (do not reopen without new evidence)

- Q8 ncols weight-hoist: RESOLVED-PARKED. Fixed-build C3 re-read −0.69% p50
  (the +1.12% RC5-build read was repeat noise). Compiled default-off.
- Adaptive fanout controller: preregistered B1 STOP (capture 0.48 < 0.70,
  engagement-latency attribution recorded). Client-side fixed fanout only;
  in-server controllers remain quarantined.
- Free-lunch thinking channels: B2 KILL — trunk −51% per-stream at N=3.
  Channels are idle-capacity-only; aggregate is still 1.6–1.8× solo.
- Token-level speculative verification: three park verdicts stand; ragged-KV
  does not change wide-batch ubatch numerics.
- RC5 fork-surface crash: root-caused (`plan.n_kv × n_streams > kv_size` via
  pre-permute stream-broadcast views), fixed at `d794fd15d`, validated to
  250k×6 streams byte-exact, shipped as RC6.

## Overnight loop progress (post-handoff, 00:30-01:40)

- **Activation-ratio heuristic** (`2fed29794`, `LLAMA_KV_TREE_RAGGED_MIN_REDUCTION`,
  default 10%): evidence-complete RC7 candidate — compact-layout N=7 penalty
  recovered from -11.5% to -3.7% (noise band), fragmented untouched. Unit
  case added. Promotion flip NOT taken (user's call).
  Evidence: `results/treebeard-nxy-optimizer/20260716-002318-branch-cost`.
- **PCBT-0 complete**: contract v1 frozen (`docs/treebeard-pcbt-contract-v1.md`),
  11 fixtures + Python reference lint with committed golden digest.
- **PCBT-1 complete**: `tools/server/server-pcbt.h` records + pure state
  machine; isolated tests cover the full transition matrix, event-ring
  bounds, byte accounting, idempotent cleanup, registry idempotency.
- **PCBT-3 complete (03:00)**: atomic create through the extracted
  `statetree_fork_family` helper (SLOT_FORK refactored behavior-identical);
  source by immutable node id, idle-destination auto-selection,
  all-or-nothing capacity, registry mutation only after fork success,
  create events. Gated `TREEBEARD_PCBT_ENABLE=1` (default off until
  PCBT-4/7 lifecycle). Full route smoke green incl. fork-backed
  create/retry/conflict/observe/events/503-capacity
  (`scripts/treebeard-pcbt-route-smoke.sh`, 0.8B CPU server).
- **PCBT-4 scoped, decision pending**: completion requests already carry
  node/fork assertions (the B3 pattern) — see the server-scheduled vs
  client-driven-attribution decision in `docs/treebeard-pcbt-wiring-plan.md`;
  recommendation is client-driven for slice 1, owner's call requested.
- **PCBT-2 schema slice complete**: `tools/server/server-pcbt-parse.h` C++
  parser with contract error classes; `test-pcbt-parse` proves fixture and
  golden-digest parity with the Python reference (cross-language
  canonicalization lock). REMAINING: task type + route handlers +
  /props capability + inference-disabled route tests (PCBT-2), then
  PCBT-3..8.
- Branch-cost evaluator: cross-arm token parity downgraded to diagnostic
  (intra-arm repeat flutter exists in dense-only arms — recorded backend
  multi-stream nondeterminism).

## Open levers, in recommended order

1. **Ragged activation-ratio heuristic** — B2 measured ragged costing 6–11%
   aggregate at fanout ≥7 on compact layouts (indexed gather overhead, ~zero
   column savings). Replace the boolean `plan.reduces_columns`
   (src/llama-kv-cache.cpp:180) with a reduction-ratio threshold (env-tunable,
   e.g. activate only when `n_kv <= ratio × dense_n_kv`). Gate with the
   existing branch-cost + confirm guards. Small, measured, RC7 material.
2. **PCBT-0…8** (`docs/treebeard-proof-carrying-branch-transactions.md`) —
   the canonical heavy-architecture slice: bounded async branch transactions
   over StateTree. CPU-testable; B3's clean fixed-8 run is its consumer
   evidence.
3. **B4 slot arbitration** — protocol frozen in
   `reports/treebeard-b4-arbitration-protocol-20260716.md`; needs a
   closed-loop load driver; gates preregistered.
4. T7 continuous-batch shape control (ledger) — unchanged.

## Operational notes

- Guards now default to rc6 identity; always pass `TREEBEARD_LIVE_SERVICE`
  explicitly anyway (a mis-defaulted window crash-looped rc5 against rc6
  once tonight; cleaned, production unaffected).
- Several long background windows were externally stopped mid-run tonight
  (cause unknown — possibly a harness limit); guards restored production
  exactly every time. Rerun-with-reuse patterns exist
  (`TREEBEARD_EDGE_OFF_RESULT`, `TREEBEARD_SKIP_FRAG`).
- llama-server suppresses INFO-level llama internals: never assert guards on
  INFO lines; use W-level lines, raw fprintf traces, or result JSONs.
- `cmake --install` strips RUNPATH: release units need `LD_LIBRARY_PATH`
  (packaging script handles it; `TREEBEARD_RC`-parametrized).
- All work is on `agent/treebeard-single-wavefront`, pushed to
  `turbo-private` (Actions verified disabled before each push). Every
  assisted commit carries `Assisted-by:` per house rules.
