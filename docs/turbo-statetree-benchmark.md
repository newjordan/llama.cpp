# Turbo StateTree Benchmark Gate

Latest completed production-size gate:
`reports/turbo-statetree-b70-benchmark-20260709.md` (passed 24/24 matched
dense/fragmented checks; see its `GGML_SYCL_F16` build-audit note).

## Policy

Every major StateTree implementation leg requires a parent/candidate benchmark
before it can become the new baseline. Correctness tests remain mandatory, but
they do not substitute for latency, throughput, and memory measurements.

The gate has two tiers:

1. A fast isolated hybrid-model gate for development and regression checks.
2. A controlled Qwen3.6-35B-A3B/B70 gate before any canary or production claim.

The fast gate may reject a change. It cannot prove production performance.

## Measured Surfaces

`scripts/turbo-statetree-bench.py` measures the complete transaction cycle:

1. Exact-token prefix prefill.
2. Unified-KV fork.
3. Concurrent branch divergence and forced-length decode.
4. Manual loser erasure or atomic winner commit.
5. Protected-root refork into a new generation.
6. Final state cleanup.

Each result contains:

- Client and server fork, commit, and refork latency.
- Concurrent branch wall time and aggregate predicted tokens per second.
- Cache reuse for every branch.
- Process RSS and high-water memory.
- Per-process DRM total and resident VRAM when Linux `fdinfo` exposes it.
- Exact recurrent prompt-data, checkpoint, and combined state bytes per slot.
- Expected shared-KV cells for the known benchmark topology.
- Reservation, generation, winner-preservation, and loser-reclamation checks.

Physical KV cells, holes, and page fragmentation on B70 remain the job of
`scripts/turbo-kv-page-ablate.py` and its server-side page probe. The StateTree
harness does not relabel a topology estimate as physical allocator evidence.
DRM clients are deduplicated by `drm-client-id`; unavailable VRAM telemetry is
reported as unknown rather than zero.

## Safety

The harness launches and stops its own server by default. It refuses port
`8093`. Attach mode is disabled unless `--allow-destructive-attach` is supplied
because every sample erases explicit slot state.

Do not use attach mode against a shared or production server.

## Fast Parent/Candidate Gate

Build parent and candidate with the same CMake options and run them one at a
time on the same host. The development model is the local Qwen3.5 0.8B Q8
hybrid model so recurrent checkpoint behavior is exercised.

Parent example:

```bash
python3 scripts/turbo-statetree-bench.py run \
  --bin /path/to/parent/bin/llama-server \
  --label parent \
  --commit PARENT_SHA \
  --out-dir /tmp/turbo-statetree-bench \
  --port 8098 \
  --ctx 32768 \
  --parallel 4 \
  --fanout 3 \
  --prefix-tokens 128,1024,8192 \
  --branch-suffix-tokens 8 \
  --branch-tokens 8 \
  --repeats 5 \
  --modes manual \
  --ngl 0
```

Candidate example:

```bash
python3 scripts/turbo-statetree-bench.py run \
  --bin /path/to/candidate/bin/llama-server \
  --label candidate \
  --commit CANDIDATE_SHA \
  --out-dir /tmp/turbo-statetree-bench \
  --port 8098 \
  --ctx 32768 \
  --parallel 4 \
  --fanout 3 \
  --prefix-tokens 128,1024,8192 \
  --branch-suffix-tokens 8 \
  --branch-tokens 8 \
  --repeats 5 \
  --modes manual,commit \
  --require-commit \
  --ngl 0
```

Compare the immutable result files:

```bash
python3 scripts/turbo-statetree-bench.py compare \
  /tmp/turbo-statetree-bench/parent.result.json \
  /tmp/turbo-statetree-bench/candidate.result.json \
  --out /tmp/turbo-statetree-bench/comparison.json
```

The default fast regression thresholds are:

- Branch aggregate throughput at least 95% of parent.
- Fork server p50 no greater than `1.10 * parent + 0.25 ms`.
- Process RSS before cleanup no greater than parent plus 64 MiB.
- DRM total VRAM before cleanup no greater than parent plus 64 MiB, when
  available on both servers.
- Zero benchmark contract failures.

These tolerances catch large regressions on a noisy development host. They are
not product SLOs.

## Fragmented Layout Gate

Use alternating occupied survivor slots to measure the same transaction amid
unrelated live sequences:

```bash
python3 scripts/turbo-statetree-bench.py run \
  --bin /path/to/candidate/bin/llama-server \
  --label candidate-fragmented \
  --commit CANDIDATE_SHA \
  --out-dir /tmp/turbo-statetree-bench \
  --port 8098 \
  --ctx 32768 \
  --parallel 8 \
  --fanout 3 \
  --layout fragmented \
  --fragment-fill-tokens 1024 \
  --persistent-fragmentation \
  --prefix-tokens 1024,8192 \
  --repeats 5 \
  --modes manual,commit \
  --require-commit \
  --ngl 0
```

For this shape, slots `0/2/4/6` retain independent prompt state while the fork
family uses `1/3/5/7`. Persistent fragmentation fills the survivor topology
once per managed-server run, preserves it between samples, and erases every
slot before shutdown. This makes a deep fragmented lane practical without
rebuilding unrelated KV state for every sample.

## B70 Gate

Before a major leg is accepted for canary work, rerun with:

- Qwen3.6-35B-A3B Q5_K_XL.
- Intel Arc Pro B70.
- `-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -ub 1024 -fa on -ctk f16 -ctv f16`.
- Shared prefixes of 1K, 8K, and at least 32K tokens.
- Dense and fragmented layouts.
- At least five order-balanced parent/candidate pairs.
- Forced equal branch token counts.
- Physical KV page-probe evidence.
- GPU memory sampling before fork, before commit, after commit, and after erase.
- Kernel reset and crash-log inspection.

This gate requires a maintenance plan because the production model occupies
the B70. The benchmark harness must use an isolated candidate port and must not
replace `:8093` without explicit approval and rollback capture.

## Required Evidence By Architecture Leg

| Leg | Additional required benchmark evidence |
| --- | --- |
| Generation and commit | Fork regression, commit versus manual erase, exact reclamation, refork latency |
| Durable identity and mutation fencing | Same performance gate plus stale-operation rejection under concurrent reuse |
| Lease and byte budget | Expiry latency, enforced byte ceiling, victim selection, throughput under pressure |
| Logical DAG handles | Lookup latency by depth, branch-width scaling, retained bytes per node |
| Cold spill and restore | Hot/cold transition latency, serialized bytes, storage bandwidth, continuation parity |
| Adaptive scheduler | Throughput, p50/p95 latency, deferred time, fairness, and state-hit rate |
| Breakout integration | End-to-end wall time, final-answer quality, accepted-branch rate, and fallback rate |

No leg advances on a speed claim without a raw JSON artifact, a compact report,
the exact parent and candidate identities, and a stated workload.
