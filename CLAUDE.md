# Treebeard - agent onboarding

IMPORTANT: this is a PRIVATE downstream fork of ggml-org/llama.cpp on branch
`agent/treebeard-single-wavefront`, pushed only to the `turbo-private` remote.
Review [AGENTS.md](AGENTS.md) for the upstream llama.cpp contributor rules
(they govern any upstream-facing work); the boundaries below govern this fork.
Write all new prose and code in ASCII (no em-dash, no unicode arrows/quotes;
use `-`, `->`, `x`, `...`).

## What this project is

Treebeard is branch-native receipted reasoning built on a llama.cpp SYCL fork
for the Intel Arc Pro B70. The serving substrate (StateTree) makes inference
state transactional and forkable; PCBT (Proof-Carrying Branch Transactions)
layers bounded, evidence-bound branch transactions on top, now a live
production surface. The product direction is quality over speed: the receipted,
branch-native reasoning surface is the product, not raw tok/s. Every
optimization must pass correctness, production-shape activation, and guarded
same-binary performance gates before it can ship.

## Workspace map

- Live source worktree (this git repo):
  `/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce`, branch
  `agent/treebeard-single-wavefront`. Your edit scope. Do NOT reorganize the
  upstream llama.cpp layout (`src/`, `ggml/`, `common/`, `tools/`, `include/`,
  C++ `tests/`, CMake). Treebeard-specific additions live in `reports/`,
  `docs/treebeard-*` / `docs/turbo-*`, `scripts/` (index only, never move),
  and `tests/pcbt/`.
- Evidence snapshot (NOT a git repo):
  `/home/frosty40/turbo/treebeard-work`. Its `results/` and `build-*` dirs are
  live. Everything under `results/` is an append-only, immutable evidence
  trail cited by decision records, the ledger, and memory. Never modify it.
- Builds: `build-treebeard-single-wavefront` (the RC8 SYCL Release build) and
  other `build-*` dirs live in the snapshot. Never run `cmake` here.
- Release install: `/home/frosty40/turbo/turbo-combined/release/` (per-RC
  copy-installs, e.g. `turbo-statetree-0.1.0-rc.8`). Never modify.
- Quarantine: `/home/frosty40/turbo/DO NOT REVIVE/` is off-limits - do not
  read, reorganize, or index it beyond noting it exists.

## Current production state (as of 2026-07-16)

- Service: `turbo-statetree-rc8.service` on port 8093, build `b9743-c7091b65b`.
- Alias `turbo-statetree-0.1.0-rc.8-Qwen3.6-35B-A3B-Q5-c262144-np12-ragged-pcbt`.
- Ship env: `LLAMA_KV_TREE_RAGGED=1 GGML_SYCL_ENABLE_STATE_IO_FUSION=1
  GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0 TREEBEARD_PCBT_ENABLE=1`.
  PCBT-4..8 receipted transactions are a live surface (/transactions,
  /metrics `pcbt_*`, `pcbt.receipt.v1`).
- Rollback chain (all systemd user units retained): rc8 -> rc7
  (`b9723-356cefa61`) -> rc6 (`b9695-d794fd15d`) -> rc5 (`b9680-de0834ca0`)
  -> rc4 (`b9627-3fcf1c626`). Rollback: `systemctl --user disable --now
  turbo-statetree-rc8 && systemctl --user enable --now turbo-statetree-rc7`.
- Check it: `/props` on :8093 shows the build, rc.8 alias, 12 slots,
  ctx 262144, and `pcbt {contract v1, enabled true}`.
- Records: deployment
  `results/treebeard-rc5-deploy/20260716-155210-rc8-production/deployment-summary.md`;
  freeze `results/treebeard-ragged-promo-b70/build-freeze-rc8-20260716.md`.

## Promotion-ladder methodology

1. Preregister the hypothesis, arms, and gates (a report under `reports/`)
   before running.
2. Run a guarded, same-binary A/B/A with explicit env arms (control vs ship),
   identical runtime + model hashes, a healthy device, and automatic
   restoration of the production service. Guard wrappers live in `scripts/`.
3. Require exact output parity for the ship arm and no-regression on the perf
   gate (e.g. dense golden >= -1.0% at 1/8/12 agents; fragmented ship-config
   aggregate above the preregistered floor).
4. Record the decision (`decision.md` / `verdict.md`) in `results/` with the
   exact result directory, measured deltas, and activation evidence.
5. Package by pin-by-hash + copy-install (`scripts/treebeard-rc5-package.sh`,
   `TREEBEARD_RC`-parametrized), then flip the systemd unit with the rollback
   chain intact.

## Hard operating boundaries

- Push only to `turbo-private`; NEVER to origin. Verify GitHub Actions are
  disabled before any push. The main agent reviews and pushes - do not push.
- Every assisted commit ends with the trailer `Assisted-by: Claude Fable 5`
  (no `Co-authored-by`). Commit in small logical chunks.
- Never `pkill`; stop/restore exact systemd user units only. Never run `cmake`
  (especially not with cwd in the worktree), `git clean`, or network calls.
- Never touch `results/`, `build-*`, `turbo-combined/release/`, systemd units,
  or the `DO NOT REVIVE/` quarantine.
- Never pipe a guard script on invocation (a pipe masks its exit code). Never
  assert a guard on INFO-level llama-server log lines (INFO internals are
  suppressed) - assert on W-level lines, raw fprintf traces, or result JSONs.
- `tests/pcbt/` is gitignored; already-tracked files are fine, but any NEW
  file there needs `git add -f`. Do NOT modify
  `tests/pcbt/pcbt12-review-notes.md` (awaiting human sign-off).
- Do not reopen parked plays (token-level speculation, Q8 ncols hoist,
  in-server adaptive fanout controllers, B2/B4 idle-capacity thinking) without
  genuinely new evidence.

## Key docs and indexes (pointers, not duplication)

- `docs/treebeard-throughput-rnd.md` - canonical T-numbered lever ledger.
- `docs/turbo-statetree.md` - StateTree design.
- `docs/treebeard-proof-carrying-branch-transactions.md` - PCBT plan;
  `docs/treebeard-pcbt-contract-v1.md` - frozen contract v1.
- `reports/README.md` - active-first reports index + archive map.
- `reports/treebeard-fresh-session-handoff-20260716-rc8.md` - latest handoff.
- `scripts/README.md` - categorized R&D script index (load-bearing vs
  historical).

## Open threads a new agent must know

1. User sign-off pending on `tests/pcbt/pcbt12-review-notes.md` (machine
   pre-verification 30/30 PASS appended; sign-off line open). Do not modify it.
2. Next lever: preemptive abort-on-arrival via PCBT-7 fencing. B4 falsified
   the idle-capacity model (+26.8% tax at 4 agents,
   `results/treebeard-b4/20260716-130842-b4/verdict.md`); the abort-on-arrival
   protocol/preregistration is not yet written.
3. Evaluator follow-up: make all-arm token parity diagnostic-only for the
   attribution (non-ship) arms (c1 flutter, a6ec035bc precedent;
   `results/treebeard-ragged-promo-b70/20260716-142959-confirm-comp-aba/verdict.md`).
4. Ragged activation-ratio heuristic (open RC9-material refinement;
   `reports/treebeard-nxy-optimizer-preregistration-20260715.md`).

Evidence under `results/` is append-only and immutable: cite it, never edit it.
