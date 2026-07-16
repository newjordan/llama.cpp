# scripts/ index (Treebeard R&D)

Categorized index of the Treebeard-specific R&D scripts in this directory.
These files are referenced by exact path from `results/` evidence records and
prior transcripts, so they are NOT renamed or moved - this README only indexes
them. "LOAD-BEARING" marks scripts on the current RC promotion ladder;
"historical" marks completed or parked lines kept for provenance.

Upstream llama.cpp tooling (build-info.sh, get-*.sh, sync-ggml*, hf.sh,
server-*.py, compare-*, snapdragon/, apple/, hip/, jinja/, etc.) is unchanged
and out of scope here.

House invariants for every guard wrapper: pass `TREEBEARD_LIVE_SERVICE`
explicitly; never pipe the guard on invocation (a pipe masks its exit code);
never assert on INFO-level llama-server log lines (INFO internals are
suppressed) - assert on W-level lines, raw fprintf traces, or result JSONs;
stop/restore exact systemd units only (never pkill).

## Guarded B70 gate wrappers (same-binary A/B/A, production stop/assert/restore)

- `treebeard-ragged-golden-aba-guarded.sh` - LOAD-BEARING. Dense golden
  1/8/12-agent A/B/A gate (the RC5..RC8 promotion gate).
- `treebeard-ragged-confirm-guarded.sh` - LOAD-BEARING. Fragmented confirm +
  state-io composition + edge probes.
- `treebeard-branch-cost-guarded.sh` - LOAD-BEARING. Ragged activation-ratio /
  N(X)+Y branch-cost gate (open RC9 lever).
- `treebeard-pcbt-gate-guarded.sh` - LOAD-BEARING. PCBT B70 acceptance gate.
- `treebeard-pcbt12-guarded.sh` - LOAD-BEARING. PCBT-12 product benchmark.
- `treebeard-b4-guarded.sh` - historical/next-lever. B4 slot-arbitration probe
  (idle-capacity falsified; abort-on-arrival is the successor).
- `treebeard-moe-b70-aba-guarded.sh` - historical. moe-down-reduce A/B/A
  (parked, statistically flat).
- `treebeard-q8-hoist-profile-b70-guarded.sh` - historical. Q8 ncols
  weight-hoist profile (parked, default-off).
- `treebeard-wavefront-b70-guarded.sh` - historical. Single-wavefront
  exploration guard.
- `treebeard-jspace-b70-guarded.sh`, `treebeard-jspace-g1-b70-guarded.sh`,
  `treebeard-jspace-g2-development-b70-guarded.sh`,
  `treebeard-jspace-g2b-embedding-resume-b70-guarded.sh` - historical. J-Space
  steering guards.

## Evaluators (gate scoring / attribution)

- `treebeard-ragged-golden-evaluate.py` - LOAD-BEARING. Scores the dense
  golden A/B/A gate.
- `treebeard-ragged-confirm-evaluate.py` - LOAD-BEARING. Scores fragmented
  confirm/composition (all-arm parity currently being downgraded to
  diagnostic for attribution arms).
- `treebeard-branch-cost-evaluate.py` - LOAD-BEARING. Branch-cost / N(X)+Y
  evaluator.
- `treebeard-pcbt-b70-gate.py` - LOAD-BEARING. PCBT B70 gate scorer.
- `treebeard-b4-arbitration.py` - historical/next-lever. B4 arbitration
  evaluator (see the frozen protocol report).
- `treebeard-nxy-controller-replay.py` - historical. N(X)+Y controller replay
  (B1 controller STOP).
- `treebeard-moe-reuse-probe.py` - historical. MoE expert-reuse bound probe.

## PCBT smoke + contract

- `treebeard-pcbt-route-smoke.sh` - LOAD-BEARING. CPU route smoke over the full
  transaction lifecycle (create/retry/conflict/observe/events/503).
- `treebeard-pcbt-contract-lint.py` - LOAD-BEARING. Lints fixtures against the
  frozen contract v1 golden digest (`docs/treebeard-pcbt-contract-v1.md`).

## Packaging + deploy

- `treebeard-rc5-package.sh` - LOAD-BEARING. Pin-by-hash package + copy-install
  of the release binaries; `TREEBEARD_RC`-parametrized (used through RC8);
  handles the RUNPATH/`LD_LIBRARY_PATH` fixup.

## Harnesses / lifecycle

- `treebeard-wavefront-b70.py`, `treebeard-wavefront-sweep.py` - historical.
  Single-wavefront benchmark harness + sweep.
- `treebeard-jspace-server-lifecycle.py` - historical. J-Space server
  lifecycle helper.

## J-Space research (historical)

- `treebeard-jspace-g1-*.py` (controls, evaluate-controls, fit, manifest,
  v2/v3/v4/v5 evaluate + manifest, v3-meld-manifest) - G1 sensor/actuator
  program.
- `treebeard-jspace-g2-*.py` (embed, generate, route, judge, evaluate,
  manifest, self-test), `treebeard-jspace-g2b-manifest.py` - G2/G2b routing
  program.

## Comparison / analysis utilities

- `turbo-host-baseline.py`, `turbo-host-compare.py` - historical. Host
  baseline capture/compare.
- `turbo-multiagent-pareto.py`, `turbo-pareto-compare.py` - historical.
  Multi-agent Pareto (RC1 era).
- `turbo-kv-page-ablate.py` - historical. Unified-KV paged-attn ablation.
- `turbo-production-preflight.py` - production preflight checks (still usable).
- `turbo-speculative-breakout.py` - historical/parked. Speculative-breakout
  harness.
- `turbo-statetree-bench.py`, `turbo-statetree-cold-gate.py`,
  `turbo-statetree-logical-gate.py`, `turbo-statetree-retention-gate.py`,
  `turbo-statetree-snapshot-gate.py` - historical. StateTree benchmark +
  acceptance gates (RC1-3 era).
- `cd395a152-turbo-sweep.sh` - historical. One-off commit sweep.
- Upstream `compare-commits.sh`, `compare-llama-bench.py`,
  `compare-logprobs.py` - unchanged llama.cpp comparison tooling, used
  during R&D.
