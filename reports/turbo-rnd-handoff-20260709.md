# Turbo R&D Handoff - 2026-07-09

## Read This First

This is the minimal handoff for a fresh primary agent. Do not reconstruct the
prior chat. Start with these files:

1. `AGENTS.md`
2. `TURBO_RND.md`
3. `reports/turbo-slot-fork-20260709.md`
4. `reports/turbo-serving-benchmark-20260708-head-a2edfe66f.md`
5. This handoff

The current milestone is a validated R&D checkpoint, not a production rollout.
Do not replace the live `:8093` server from this branch without a new maintenance
plan, exact rollback capture, and explicit approval.

## Repository State

| Item | Value |
| --- | --- |
| Local worktree | `/home/frosty40/turbo/turbo-combined` |
| Local branch | `turbo-combined` |
| Local checkpoint | `8acb67d399318ee0525b1385625c1101090dd1d3` |
| Private repository | `https://github.com/newjordan/turbo_RND` |
| Private default branch | `main` |
| Private snapshot commit | `bb158776aa49a4e0df4714bb0d95f0f123921fca` |
| Public base | `f3a302b47` |
| Snapshot tree | `de59749a7d0b9bb0d289c433fd5d72e0b477060d` |

The private repository is verified `PRIVATE`. Its root snapshot has the exact
same tree as local checkpoint `8acb67d39`, without importing the full shallow
llama.cpp history. Public `origin` and `fork` were not pushed.

This handoff file was created after that snapshot and is intentionally
uncommitted until the user reviews it.

## Hardware And Model

| Item | Value |
| --- | --- |
| GPU | Intel Arc Pro B70, Xe2/Battlemage |
| GPU memory | 30.3 GiB |
| Model | Qwen3.6-35B-A3B Q5_K_XL |
| Model path | `/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` |
| Backend | SYCL, IntelLLVM 2026 |
| Serving shape | `-kvu -np 12 -c 262144 -ub 1024 -fa on -ctk f16 -ctv f16` |

Validated standalone B70 ceilings from the private kernel workspace:

| Statistic | Value |
| --- | ---: |
| XMX FP16 matrix | 183.5 TFLOPS |
| Vector FP16 | 44.8 TFLOPS |
| Read bandwidth | 769 GB/s |
| Write bandwidth | 349 GB/s |
| STREAM Triad | 530 GB/s |

## Live Production State

The live service was restored after the controlled B70 gate.

| Item | Value |
| --- | --- |
| Endpoint | `http://127.0.0.1:8093` |
| Health | `{"status":"ok"}` |
| Build | `b61-a2edfe66f` |
| Alias | `turbo-head-a2edfe66f-Qwen3.6-35B-A3B-Q5-np12-kvu-c262144-ub1024` |
| Slots | 12 |
| Context pool | 262144 |
| Active unit | `turbo-head-a2edfe66f-rollback-8093.service` |
| Hydra surface | `turbo:8093`, verified / health_verified |

Operational debt:

- The active rollback unit lives under `/run/user/1000/systemd/user/`; it is
  runtime-only and will not survive a reboot.
- `turbo-cd395a152.service` is enabled but inactive. It points at an older
  production shape and may become relevant on restart.
- Hydra was restored to its pre-maintenance manifest. Its `model_id`,
  architecture, and notes still name `cd395a152`, while live `/props` proves
  `b61-a2edfe66f` is serving. Treat endpoint health as valid but metadata as
  stale. Fix this only in a separate operational change.

Quick checks:

```bash
curl -fsS http://127.0.0.1:8093/health
curl -fsS http://127.0.0.1:8093/props | jq '{build_info,model_alias,total_slots,n_ctx:.default_generation_settings.n_ctx}'
systemctl --user status turbo-head-a2edfe66f-rollback-8093.service --no-pager
```

## What Landed

### Server-Native Unified-KV Fork

The candidate implements:

```text
POST /slots/{source}?action=fork
{"destinations":[1,2,...]}
```

Key properties:

- Atomic source/destination validation with canonical slot IDs.
- Unified-KV sequence sharing through `common_context_seq_cp`.
- Fork reservations keep owned slots out of automatic scheduling.
- Explicit `id_slot` requests can run each branch.
- Erase and restore release reservations correctly.
- Deferred requests wake when a reservation is released.
- Idle prompt-cache and idle-sleep paths respect reservations.
- Context/cache shifts are prohibited while sequence state is shared.
- Inherited hybrid/recurrent checkpoints are cleared at fork.
- Post-divergence checkpoints remain sequence-local.
- Non-unified KV returns HTTP 501.
- Fork is rejected for unsupported mtmd, draft/speculative, and LoRA shapes.

Relevant code:

- `tools/server/server-context.cpp`
- `tools/server/server-context.h`
- `tools/server/server-task.cpp`
- `tools/server/server-task.h`
- `tools/server/server-queue.cpp`
- `tools/server/tests/unit/test_slot_fork.py`

### Stale Checkpoint Fix

The first 35B gate found that slot erase and successful file restore cleared
tokens but retained `server_prompt.checkpoints`. File controls therefore reused
hidden 62.8 MiB recurrent checkpoints and made fork look artificially slower.

`server_slot::prompt_metadata_clear()` now clears:

- Prompt tokens.
- Main and draft serialized prompt data.
- Prompt checkpoints.
- Fork ownership.

`/slots` now reports `n_prompt_checkpoints` so resets can prove this state is
zero. The clean rerun falsified the suspected recurrent copy-on-write slowdown.

### Measurement Harnesses

`scripts/turbo-speculative-breakout.py` now provides:

- Result schema version 3.
- Strict response, slot ID, token count, and token type validation.
- Full returned token IDs plus content/token SHA-256 hashes.
- `timings.cache_n` capture.
- File and native-fork prefix clone backends.
- Clone wall and server-internal timing.
- Reservation cleanup.
- `--ignore-eos --no-stop-strings` equal-token lanes.
- Explicit empty `stop: []` overrides server reverse-prompt defaults.

`scripts/turbo-kv-page-ablate.py` now provides strict warmup, fill, and measured
parity plus reference/control attribution.

## Current Serving Benchmarks

Production binary `b61-a2edfe66f`:

| Workload | Result | Sample |
| --- | ---: | ---: |
| One active request, c262k/np12 | 79.610 tok/s | n=1 |
| Production shape, c262k/np12 | 178.702 tok/s | 1 batch, 12 requests |
| Short c32k/np12 | 178.146 tok/s | 1 batch, 12 requests |
| Short c32k/np32 saturation | 207.850 tok/s | 1 batch, 32 requests |
| Fragmented baseline | 87.425 +/- 0.097 tok/s | n=2 |
| Fragmented indexed FATTN | 108.375 +/- 0.029 tok/s | n=2 |
| Fragmented indexed gain | +23.96% | 2 paired runs |

Do not describe `200 tok/s` as the exact production-shape rate. The exact
c262k/np12 measurement is 178.702 tok/s. The 207.850 tok/s result is the
short-context np32 saturation row.

Durable evidence:

- `reports/turbo-serving-benchmark-20260708-head-a2edfe66f.md`
- `reports/turbo-serving-benchmark-20260708-head-a2edfe66f-summary.json`

## Slot-Fork Candidate Gate

Candidate build at measurement time: version 66, base `f3a302b47`, same
35B/B70 serving shape, 124-token shared prefix.

| Metric | File save/restore | Native fork | Result |
| --- | ---: | ---: | ---: |
| Client clone wall | 200.398 +/- 5.678 ms | 6.926 +/- 0.294 ms | 28.9346x |
| Server clone time | 193.880 +/- 6.243 ms | 6.384 +/- 0.269 ms | 30.3704x |
| Short branch wall | 3.0139 +/- 0.0484 s | 3.0371 +/- 0.0278 s | flat |
| Short full flow | 4.9072 +/- 0.0606 s | 4.7065 +/- 0.0275 s | +4.26% |
| Forced aggregate rate | 158.482 +/- 1.220 tok/s | 158.048 +/- 0.513 tok/s | -0.27% |
| Forced full flow | 12.8265 +/- 0.0142 s | 12.5537 +/- 0.0297 s | +2.17% |
| Per-clone file I/O | 68.4 MB write + 752.5 MB read | 0 | eliminated |

Contracts:

- Reset contracts: 18/18 passed.
- Clone contracts: 18/18 passed.
- All 48 forced branch requests generated exactly 128 tokens.
- Five of five cross-backend pairs returned identical final tokens.

Caveat: intermediate branch arrays were not bit-stable. File/file and fork/fork
self-controls also differed, so cross-backend branch differences are not
attributable to fork. Use sequential top-1/logit comparison if bit-exact
equivalence becomes a release gate.

Artifacts:

- `reports/turbo-slot-fork-20260709.md`
- `/home/frosty40/turbo/results/slot-fork-35b-gate/20260709T201929Z-checkpoint-fix/rerun-summary.json`

## Combined-Answer Harness Evidence

The optimized objective-core run used the production `:8093` surface:

| Metric | Value |
| --- | ---: |
| Tasks | 11 |
| Baseline pass rate | 5/11, 45.45% |
| Final multipass pass rate | 11/11, 100% |
| Objective losses | 0 |
| Mean deterministic score delta | +35.91 |
| Branch fanout | 86.48 predicted tok/s |
| Multipass core | 82.84 predicted tok/s |
| Mean multipass core wall | 7.29 s/task |
| Full-verifier reference wall | 21.30 s/task |

This proves harness mechanics on exact-transform tasks. It does not prove
product value or quality on representative user workflows.

## Validation At Checkpoint

Passed:

```text
cmake --build build --target llama-server -j16
tools/server/tests/unit/test_slot_fork.py: 9 passed
tools/server/tests/unit/test_ignore_eos.py: 3 passed
tests/test_turbo_speculative_breakout.py: 16 passed
tests/test_turbo_kv_page_ablate.py: 16 passed
python3 -m py_compile: passed
git diff --check: passed
```

The standard repository pytest session fixture could not bootstrap because the
local build lacks HTTPS model-download support. Focused tests ran against:

```text
/home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf
```

## Remaining Gates

Do these before any canary rollout:

1. Run forced-token file/fork pairs at 1k, 8k, and 32k-plus shared prefixes.
2. Use at least five order-balanced pairs per prefix length.
3. Require fork/file branch throughput within 2% and retain a clone win.
4. Test sparse and fragmented slot layouts while instrumenting shared tails,
   allocated rows, `n_rs`, `n_seqs`, and fork time.
5. Test a fork after bounded recurrent rollback/checkpoint restore. Recurrent
   `seq_cp` aliases the tail but does not copy `rs_idx`; compare source and
   destination top-1 logits before optimizing anything.
6. Add explicit busy-slot, context-limit, LoRA, speculative/draft, and mtmd
   rejection regressions.
7. Build at least 30 representative workflow tasks with deterministic or
   semi-deterministic acceptance gates.
8. Measure net win rate, human override rate, latency, and generated-token cost.
9. Add adaptive fanout and early branch pruning only after the fixed 12-way
   quality baseline is established.
10. Use a dedicated canary endpoint before changing `:8093`.

Do not implement eager recurrent copying now. The clean forced lane shows no
performance reason for it.

## Recommended Ownership

Keep one primary agent responsible for architecture, integration, production
maintenance, and final benchmark claims. Delegate bounded lanes:

| Lane | Deliverable |
| --- | --- |
| Prefix scaling | Exact tasks, paired artifacts, summary statistics |
| Recurrent correctness | Rollback test plus source/destination logit evidence |
| Sparse KV | Instrumentation and fragmented-slot measurements |
| Quality | Representative task pack and acceptance results |
| Review | Independent correctness and measurement audit |

Every worker should return patch paths, exact commands, raw artifact paths,
measurements, caveats, and blockers. Agents share a filesystem, so assign
non-overlapping files or separate worktrees.

## Fresh-Session Prompt

```text
Work in /home/frosty40/turbo/turbo-combined on branch turbo-combined.
Read AGENTS.md, TURBO_RND.md, reports/turbo-rnd-handoff-20260709.md,
and reports/turbo-slot-fork-20260709.md. Treat commit 8acb67d39 as the
validated local checkpoint and private newjordan/turbo_RND main snapshot
bb158776aa49 as the off-machine backup. Do not touch production :8093 until
the next gate is implemented and reviewed. First execute the 1k/8k/32k
prefix-scaling and recurrent-rollback correctness gates, using delegated
read-only analysis/review lanes where useful. Preserve raw artifacts and
restore production exactly after any approved B70 maintenance window.
```
