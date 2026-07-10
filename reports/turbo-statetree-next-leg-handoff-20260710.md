# Turbo StateTree Next-Leg Handoff - 2026-07-10

## Start Here

This is the current fresh-session entry point. Read these files in order:

1. `AGENTS.md`
2. `TURBO_RND.md`
3. `reports/turbo-statetree-b70-benchmark-20260709.md`
4. `docs/turbo-statetree.md`
5. `docs/turbo-statetree-benchmark.md`
6. This handoff

Do not reconstruct the previous conversation. The first StateTree transaction
slice is implemented, correctness-tested, production-size benchmarked, and
accepted as the R&D baseline. It has not been deployed to production.

## Exact Repository State

| Item | Value |
| --- | --- |
| Worktree | `/home/frosty40/turbo/turbo-combined` |
| Branch | `turbo-combined` |
| Accepted benchmark checkpoint | `6051ddf31740cfefe085e80dd65d78dadd870f8e` |
| StateTree implementation | `9a37cb8fd88a737a34d03aebc0d5575976805c59` |
| Private repository | `https://github.com/newjordan/turbo_RND` |
| Pre-handoff private checkpoint | `9e7a82cf091e1f7d8a4d10189a452628a991c246` |
| Accepted baseline tree | `2338e96cfb70f25ebf9b582d85cd6652ebbe9e13` |
| Public pushes | None |

The private repository is verified private. Local commit `6051ddf31` and
private snapshot `9e7a82cf0` have the same tree through the completed B70
report. The handoff commit advances both histories with documentation only; it
does not change the implementation or its accepted benchmark evidence.

## What The Kernel Is Now

This is a server-runtime transaction layer over llama.cpp unified KV and
hybrid/recurrent prompt state. It is not a GPU kernel, LoRA, model adapter, or
weight change.

The implemented physical-slot contract is:

```text
prefill source
  -> fork into a generation-fenced family
  -> run divergent branches
  -> atomically commit one winner
  -> erase exact-family losers
  -> continue or refork the protected winner under a new generation
```

Endpoints:

```text
POST /slots/{source}?action=fork
POST /slots/{winner}?action=commit
```

Fork returns an opaque `fork_id`. Commit requires that generation, rejects
stale or busy mutations atomically, preserves the winner in place, and releases
the losers. Protected roots remain unavailable to automatic slot scheduling and
idle sleep.

This remains a physical-slot StateTree slice. It does not yet provide durable
logical DAG handles, cold spill, persistence across process restart, or a
distributed state namespace.

## Accepted B70 Result

The corrected matched Qwen3.6-35B/B70 gate passes:

| Signal | Result |
| --- | ---: |
| Accepted transaction samples | 96 |
| Contract failures | 0 |
| Dense and fragmented checks | 24/24 passed |
| Candidate branch delta | -0.29% to +0.62% |
| Candidate fork delta | -8.26% to -0.02% |
| Maximum matched RSS delta | +24.254 MiB |
| Matched VRAM delta | -0.008 MiB |
| Twelve-slot loser state reclaimed | 724,508,708 bytes |
| Full-width pre-cleanup RSS | 2.190 GiB |
| Full-width DRM VRAM | 30.896 GiB |

One live Qwen3.6 recurrent checkpoint is exactly 65,864,428 bytes
(62.813 MiB). A six-slot family holds 395,186,568 bytes. A 12-slot family holds
790,373,136 bytes. Full-width commit reduced the latter to one 65,864,428-byte
winner, reclaiming 91.67% of family checkpoint state.

The persistent fragmented fixture also proved that a slot can retain multiple
checkpoints. Each 4K survivor held two checkpoints and 131,728,856 bytes. Six
survivors plus the active family exposed 1,185,559,704 bytes of prompt state.
This is the concrete reason the next leg must impose leases and a hard byte
budget before logical handles make state easier to retain.

Raw evidence:

```text
/home/frosty40/turbo/results/statetree-b70-gate/20260709T230139-0500
```

Durable summaries:

- `reports/turbo-statetree-b70-benchmark-20260709.md`
- `reports/turbo-statetree-b70-benchmark-20260709-summary.json`

## Benchmark Audit Lesson

The first fresh parent build used `GGML_SYCL_F16=OFF`; candidate and production
used `ON`. That unmatched parent produced a false apparent decode regression.
Its 30 samples were retained but excluded. The parent was rebuilt with matching
SYCL flags, the matched parent lanes were rerun, and all 24 checks passed.

Every future parent/candidate gate must capture and compare relevant CMake cache
values before production stops. At minimum verify:

```text
GGML_SYCL
GGML_SYCL_DNN
GGML_SYCL_F16
GGML_SYCL_GRAPH
GGML_SYCL_HOST_MEM_FALLBACK
GGML_SYCL_TARGET
CMAKE_BUILD_TYPE
CMAKE_C_COMPILER
CMAKE_CXX_COMPILER
```

Do not infer an implementation regression from binaries with unmatched backend
flags.

## Live Production State

Production was restored exactly after both approved maintenance intervals.

| Item | Value |
| --- | --- |
| Endpoint | `http://127.0.0.1:8093` |
| Unit | `turbo-head-a2edfe66f-rollback-8093.service` |
| Build | `b61-a2edfe66f` |
| Alias | `turbo-head-a2edfe66f-Qwen3.6-35B-A3B-Q5-np12-kvu-c262144-ub1024` |
| Slots | 12 |
| Context | 262144 |
| Hydra surface | `turbo:8093`, verified / health_verified |

The active unit is runtime-only under `/run/user/1000/systemd/user/` and will
not survive a reboot. `turbo-cd395a152.service` remains enabled but inactive.
Hydra descriptive metadata still names the older `cd395a152` deployment even
though health and `/props` prove `b61-a2edfe66f` is live. Do not mix that
separate operational cleanup into the next StateTree code leg.

Quick checks:

```bash
curl -fsS http://127.0.0.1:8093/health
curl -fsS http://127.0.0.1:8093/props | \
  jq '{build_info,model_alias,total_slots,n_ctx:.default_generation_settings.n_ctx}'
systemctl --user status turbo-head-a2edfe66f-rollback-8093.service --no-pager
```

Do not stop or replace production without a fresh maintenance plan, exact
rollback capture, an automatic rollback timer, and explicit approval.

## Next Architecture Leg

Build bounded StateTree retention: leases plus an authoritative byte budget.

The goal is not merely an expiration timer. The server must make retained state
bounded, observable, race-safe, and useful under pressure.

Minimum contract questions to settle before code:

1. What receives a lease: protected root, whole fork generation, or logical
   state handle?
2. Is lease time measured from fork, commit, last continuation, or last access?
3. Which operation renews a lease, and can stale generations renew anything?
4. Is the byte ceiling global, per client, per family, or both global and local?
5. Which exact bytes count: prompt data, all recurrent checkpoints, auxiliary
   metadata, and any future serialized state?
6. What happens at admission pressure: reject, expire, or evict?
7. What victim policy is deterministic and explainable?
8. How are an in-flight request and an expiring protected root fenced?
9. What response and telemetry expose expiry, rejection, and reclamation?
10. How does shutdown or erase cancel pending expiry work without stale reuse?

The measured `/slots` fields are the current byte authority:

```text
n_prompt_data_bytes
n_prompt_checkpoint_bytes
n_prompt_state_bytes
```

Do not budget from RSS. The host allocator can retain freed capacity, while the
exact slot counters proved deterministic reclamation.

## Required Benchmarks For The Next Leg

Every major implementation leg must carry parent/candidate evidence.

Fast isolated gate first:

- Existing dense and fragmented StateTree regression suite.
- Expiry latency and jitter.
- Zero-use and renewal boundary behavior.
- Exact enforced byte ceiling.
- Deterministic victim selection.
- Rejection behavior when no state is evictable.
- Concurrent continuation versus expiry fencing.
- Throughput and fork/commit/refork overhead with leases enabled and disabled.
- Long churn run proving no state or reservation leak.

B70 gate only after the fast gate passes:

- Use current checkpoint `6051ddf31` as the parent baseline.
- Match all compiler and SYCL flags before downtime.
- Repeat 1K/8K/32K dense and persistent fragmented lanes.
- Add byte-pressure shapes around one, six, and 12 live checkpoints.
- Measure p50/p95 expiry and admission latency, victim correctness, RSS, exact
  state bytes, DRM VRAM, and physical KV rows.
- Preserve raw JSON and page-probe logs.
- Restore production exactly and inspect kernel/server logs.

## Do Not Redo

- Do not rerun the first StateTree CPU or B70 gate unless validating a new
  implementation leg.
- Do not treat the discarded `SYCL_F16=OFF` parent data as acceptance evidence.
- Do not claim a large commit speedup. At 35B, commit was 0.991x to 1.135x
  manual cleanup; its current value is atomicity and generation fencing.
- Do not implement logical DAG handles before retention is bounded.
- Do not use GPU allocation shrinkage as commit evidence. The unified KV pool
  is preallocated; exact state bytes and physical-cell reuse are authoritative.
- Do not modify stale Hydra metadata during StateTree implementation work.
- Do not push to public `origin` or `fork` from this R&D lane.

## Fresh-Session Prompt

```text
Work in /home/frosty40/turbo/turbo-combined on branch turbo-combined.
Read AGENTS.md, TURBO_RND.md,
reports/turbo-statetree-next-leg-handoff-20260710.md,
reports/turbo-statetree-b70-benchmark-20260709.md,
docs/turbo-statetree.md, and docs/turbo-statetree-benchmark.md.

Treat benchmark checkpoint 6051ddf31, its private turbo_RND snapshot
9e7a82cf0, and StateTree implementation 9a37cb8fd as the accepted baseline.
The current HEAD adds handoff documentation only. Production :8093 is live on
b61-a2edfe66f and must not be touched without a new approved maintenance plan
and exact rollback capture.

Design the next StateTree leg: bounded leases plus an authoritative retained
state byte budget. Resolve lease scope, renewal, byte accounting, admission,
victim policy, and concurrency fencing before implementation. Use exact
n_prompt_*_bytes telemetry rather than RSS. Add focused correctness tests and
mandatory parent/candidate benchmarks for every major step. Run the isolated
CPU gate first; do not schedule B70 maintenance until it passes and all build
flags are audited as matched.
```
