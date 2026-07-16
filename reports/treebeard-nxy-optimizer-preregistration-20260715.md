# Preregistration: semantic N(X)+Y channel optimizer (2026-07-15)

Frozen before any replay or rig run. No gate below may be retuned after seeing
results. No sample-size extension beyond the stated repeats is admitted.

## Scope and authority

- Approved scope: user-approved plan of 2026-07-15 (ragged-KV promotion RC5 +
  semantic channel program). The adaptive controller is **client-side only**
  (breakout harness / future PCBT client). No in-server speculation controller
  is authorized this campaign (quarantine:
  `/home/frosty40/turbo/DO NOT REVIVE/README.md`).
- Token-channel (in-sequence draft column) rig work is excluded. The three
  recorded B70 park verdicts stand (wide n-gram verification, block-head
  anchor, serial frontier). Ragged-KV changes sequence indexing, not wide-batch
  ubatch numerics, so it does not lift the parity block. Recorded token-channel
  traces are reused below **only** as offline inputs to validate controller
  logic; no claim about token-channel viability is made or implied.
- Source binding: worktree
  `/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce`, branch
  `agent/treebeard-single-wavefront`, frozen at commit `de0834ca0` (q8-hoist
  experiment isolated at `a1d92fed7`; ragged chain `81df9e2ff..76befe8c8`
  included). Model `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`, SHA-256
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`.

## Frozen controller specification

Per-stream state and rule, identical wherever the controller is evaluated:

- Fanout ladder: `{0, 2, 4, 8}`.
- Acceptance signal X: per-round accepted/proposed ratio (B1 replay) or
  validator outcome (B3 semantic fanout: pass=1, fail=0).
- EWMA of X with alpha = 0.3, initialized at 0.
- Promote one ladder step after 2 consecutive rounds with round acceptance
  >= 0.75.
- Demote to fanout 0 immediately on any round with acceptance < 0.25.
- Coverage probe: while at fanout 0, one probe round at fanout 2 every 32
  serial tokens; probe cost is bounded by design (<= 1 extra column per 32
  tokens).
- Floor: fanout 0 is byte-identical to baseline decode.

## B1 — offline controller replay (zero rig)

- Inputs (read-only, already recorded):
  `results/treebeard-single-wavefront/cpu-width-sweep-clean-r2.json`,
  `results/treebeard-single-wavefront/serial-anchor-cpu-proposal-smoke-20260714.json`.
- Method: replay the frozen controller over recorded per-round arrays
  (`draft_n_per_round`, `draft_n_accepted_per_round`) per case and repeat.
  When the controller chooses width w in round i, the round outcome is taken
  from the recorded fixed-width-w sample at round index i (same case, same
  repeat ordering); if that width's trace is exhausted at i, the last observed
  round at that width is held. Committed tokens per round = 1 + accepted
  drafts at the chosen width (0 drafts at width 0). Round cost = recorded
  per-round wall at the chosen width (sample `wall_s` / observed rounds).
- **Pre-run method amendment (2026-07-15, before any replay executed; gates
  unchanged):** the `wall_s / rounds` round cost above is wrong on inspection —
  recorded samples interleave rounds with serial tokens (e.g. structured-copy
  w=4: 8 rounds commit 40 of 64 tokens), so `wall_s / rounds` folds serial
  segments into round cost. Amended round cost:
  `c_round(w) = (wall_s(w) - n_serial(w) * c_tok0) / rounds(w)` where
  `n_serial(w) = predicted_n - (rounds + accepted)` and `c_tok0` is the same
  case/repeat width-0 per-token wall. Controller initial state is the
  coverage-first probe: first round at fanout 2 (unpinned in the original
  text). The crude model is still computed and reported as a sensitivity
  check.
- Reference: best fixed width per case from the same file's
  `case_summaries[*].aggregate_tps_gain_pct`.
- Gates (both must hold):
  - G-B1a: adaptive committed-token throughput gain on `structured-copy`
    >= 70% of the best fixed-width gain on the same traces.
  - G-B1b: fanout 0 chosen on >= 95% of `free-prose` AND `code-edit` rounds
    (probe rounds count against the budget; the 1-in-32 probe cadence keeps
    the ceiling at ~97%).
- On failure: adaptive controller is dropped; B3 runs fixed-fanout arms only.

## B2 — branch-cost curve c(N) (guarded, ~45 min)

- Build: frozen `de0834ca0` binaries from
  `build-treebeard-single-wavefront` (same binary all arms, env toggles only).
- Geometry: ragged-kv screen shape (ctx 262144, np 12, 32K shared prefix),
  1 trunk + N forked branches, N in {0, 1, 3, 5, 7, 11}, greedy, 64 branch
  tokens, seed 1709, 2 repeats per point.
- **Pre-data method amendment (2026-07-15, before any c(N) point was
  measured; gate unchanged):** `turbo-statetree-bench.py` rejects
  `--fanout 0`, so the N=0 solo-trunk point is measured instead by a direct
  `/completion` against the same server env with a 32768-token token-array
  prompt and `n_predict 64`, reading the server's `timings.predicted_per_second`
  (decode throughput depends on context depth, not token content). N >= 1
  points come from the harness fanout sweep as originally specified.
- Arms: `LLAMA_KV_TREE_RAGGED=1` vs `=0`, both with
  `GGML_SYCL_ENABLE_STATE_IO_FUSION=1`.
- Metrics: trunk per-stream tok/s vs N, aggregate tok/s, fork and commit
  latency p50, token hash parity per branch.
- Kill gate G-B2: with ragged=1, trunk per-stream loss at N=3 relative to N=0
  > 35% -> the optimizer target narrows to idle-slot-only filling and G-B3b
  becomes strict (>= 25%).

## B3 — adaptive semantic fanout A/B/A on B70 (guarded, ~2-3 h)

- Driver: `scripts/turbo-speculative-breakout.py` with
  `--preserve-prefix-root` StateTree transactions; objective-core task set
  identical to the CPU A/B/A
  (`results/treebeard-single-wavefront/adaptive-objective-core*`,
  `fixed8-objective-core`).
- Arms: single-pass baseline; fixed-8 fanout; adaptive ladder (frozen spec
  above, X = validator outcome, stop-on-pass, escalate on fail).
- Gates (all must hold):
  - G-B3a: final validator pass count in every arm >= single-pass baseline.
  - G-B3b: adaptive multipass wall >= 15% better than fixed-8 (>= 25% if
    G-B2 killed; CPU precedent: -20.5%).
  - G-B3c: zero leaked StateTree families or slots at run end.

## B4 — slot arbitration under multi-tenant load (guarded, ~2-3 h)

- Policy under test (zero server changes): idle-slot-only client — poll
  `GET /slots`, fork only into non-processing slots, never start a wave while
  any request is queued, commit or abort promptly.
- Load points: background real-agent load occupying 4, 8, 11 of 12 slots;
  one thinking task on the remainder; batch shapes verified with
  `TREEBEARD_BATCH_SHAPE_PROF=1`.
- Gates (all must hold):
  - G-B4a: real-agent p50 regression <= 1% at every load point
    (optimizer on vs off).
  - G-B4b: thinking-task wall >= 15% better than single-pass at <= 8
    occupied slots.
  - G-B4c: fanout degrades to 0 at 11-12 occupied with no queued-request
    starvation event.

## Stop rules

- Any gate failure stops the stage; no threshold retuning, no repeats beyond
  those stated, no post-hoc metric substitution.
- Whole-lever stop: G-B2 kill AND G-B3b(strict) failure, or G-B3a/G-B3b
  failure at standard thresholds -> record the stop in
  `docs/treebeard-throughput-rnd.md` and end the program.
- Every guarded run must restore production RC4/RC5 exactly (identity, exe
  digest, `NRestarts=0`) with empty kernel fault signatures, or its artifact
  is invalid regardless of numbers.
