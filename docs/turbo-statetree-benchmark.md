# Turbo StateTree Benchmark Gate

Latest completed production-size gate:
`reports/archive/20260710-statetree-durability/turbo-statetree-b70-benchmark-20260709.md` (passed 24/24 matched
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
- Exact StateTree current, retained, active, budget, and high-water byte
  telemetry when the candidate exposes bounded retention.
- The executable and bundled runtime-library hashes, live `/props` identity,
  model path/size, effective workload controls, and the relevant CMake cache
  values, including the SYCL flags that invalidated the first B70 parent run.

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

For the bounded-retention candidate lane, add explicit controls to the same
managed-server command:

```text
--statetree-lease-ms 30000
--statetree-max-state-bytes 1073741824
```

Generation-capable branch completions and cleanup are fenced automatically by
the harness. Setup and teardown inspect the current slot generation before
erase, so a stale physical-slot cleanup cannot touch a newer family.
Across repeats, the harness rotates the fork root and uses that same physical
slot as the winner in paired manual and commit lanes. This avoids winner-slot
confounding while keeping manual cleanup legally reforkable.

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

Comparison is coverage-strict. It fails on result-schema, workload, build-cache,
or required-group mismatches; missing timings cannot be interpreted as zero.
Contract-failed samples are excluded from performance summaries, and every
requested commit sample must support and pass the transaction contract.

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

## Bounded-Retention Pressure Gate

After the matched dense and fragmented matrix passes, run the isolated
retention gate against the same candidate build:

```bash
python3 scripts/turbo-statetree-retention-gate.py \
  --bin /path/to/candidate/bin/llama-server \
  --model /path/to/hybrid-model.gguf \
  --label candidate-retention \
  --commit CANDIDATE_SHA \
  --out-dir /tmp/turbo-statetree-retention \
  --port 8098 \
  --no-source-oneapi
```

The gate probes the model's exact recurrent checkpoint bytes before deriving
its ceilings. It then requires:

- deterministic oldest-family and oldest-slot victims under exact pressure;
- exact reclaimed-byte counters and a post-enforcement high-water no greater
  than the configured ceiling;
- clean rejection when an optional checkpoint cannot fit and no victim is
  safe;
- measured autonomous expiry jitter;
- both legal outcomes at the continuation-versus-expiry boundary, with the
  generation invariant preserved in either case;
- repeated commit/refork and autonomous-expiry churn with zero final state or
  reservation leak.

The result includes executable, runtime-library, model, and CMake-cache
identities. The development defaults are deliberately short; production-size
checkpoint pressure still belongs in the controlled B70 gate.

For the 12-slot B70 pressure matrix, use the same production model and server
shape with:

```text
--pressure-widths 1,6,12 --only-pressure-widths
```

This validates ordinary one-checkpoint replacement, atomic six-member family
reclamation, and the no-safe-victim behavior at full width. Run it only inside
an approved maintenance window with automatic rollback.

## Logical And Node Control Gate

Run the identity-specific stress gate after the normal transaction and pressure
matrices:

```bash
python3 scripts/turbo-statetree-logical-gate.py \
  --bin /path/to/candidate/bin/llama-server \
  --model /path/to/hybrid-model.gguf \
  --label candidate-logical \
  --commit CANDIDATE_SHA \
  --out-dir /tmp/turbo-statetree-logical \
  --port 8098 \
  --journal-events 1100 \
  --require-nodes \
  --require-node-mutations \
  --require-node-refork
```

The gate alternates physical, lineage, and node lookup; checks winner migration
and re-fork parent edges; floods the 1024-entry journal past truncation; measures
full and incremental reads; and proves expiry tombstones plus stale-identity
rejection. With `--require-node-mutations`, it also drives open-family renew,
winner commit, 1,100 journal-producing renewals, exact erase, and stale erase
through `/nodes/{node_id}`. Omit `--require-nodes` only for a logical-state
parent that predates branch-node identity; node mutation mode requires it.
With `--require-node-refork`, the committed winner is re-forked through its
node, and the gate verifies fresh child IDs, parent edges, stale-parent
rejection, and lineage preservation.

## Immutable Snapshot Control Gate

Run the content-specific gate with a positive, isolated host-memory budget:

```bash
python3 scripts/turbo-statetree-snapshot-gate.py \
  --bin /path/to/candidate/bin/llama-server \
  --model /path/to/hybrid-model.gguf \
  --label candidate-snapshot \
  --commit CANDIDATE_SHA \
  --out-dir /tmp/turbo-statetree-snapshot \
  --port 8098 \
  --snapshot-budget-bytes 536870912
```

The gate requires physical-slot-independent digest convergence for shared fork
heads, one unique payload behind multiple provenance handles, source-mutation
independence, exact byte accounting, full materialization round-trip, fresh
node/fork identities and provenance edges, deterministic continuation parity,
digest mismatch fencing, and stale-handle rejection. Post-materialization
independent computation is compared for generated output; its serialized KV
digest is recorded but is not required to converge bitwise.

The result records server/runtime/model identities, capture and materialization
timings, unique payload growth, Prometheus snapshot counters and gauges, host
RSS, and DRM memory. On B70 use the production 12-slot shape and at least a 1K
prefix. Snapshot budget pressure rejection is separately covered by the server
unit suite because the control gate intentionally retains several unique
payloads for growth telemetry.

## Durable Cold-Content Restart Gate

Run the managed two-process gate against an isolated, initially empty store:

```bash
python3 scripts/turbo-statetree-cold-gate.py \
  --bin /path/to/candidate/bin/llama-server \
  --model /path/to/hybrid-model.gguf \
  --label candidate-cold \
  --commit CANDIDATE_SHA \
  --compat-id 'model=...;llama=...;server=...;format=cold-v1' \
  --out-dir /tmp/turbo-statetree-cold \
  --port 8098 \
  --snapshot-budget-bytes 536870912 \
  --disk-budget-bytes 1073741824 \
  --load-budget-bytes 536870912
```

Phase one captures and spills exact content, verifies duplicate publication,
materializes from the hot handle, and records deterministic continuation.
After a full server stop, phase two proves temporary-file recovery, startup
discovery, cold digest round-trip, hot/cold output parity, fresh provenance,
and rejection of the stale pre-restart handle. The server unit suite adds
malformed-object accounting, compatibility isolation, pre-allocation load
ceiling rejection, runtime corruption failure, exact erase, and disk-pressure
admission with no stray published file.

The gate also launches duplicate spills concurrently, samples `/states` while
I/O is pending, and runs an independent inference request on another slot. Use
`--require-scheduler-overlap` on B70 to require that inference to finish before
the ordered spill queue drains. Restart cold load is likewise probed through
`/states`. Acceptance requires sub-I/O-latency control-plane responses, exact
reservation release, and unchanged hot/cold continuation parity.

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
| Logical and branch-node handles | Physical/lineage/node lookup, parent-edge correctness, ring truncation, retained bytes per branch |
| Frozen immutable content | Capture/materialization cost, digest convergence, deduplication, exact growth, continuation parity |
| Durable content | Restart recovery, compatibility isolation, integrity, crash cleanup, exact disk/load ceilings |
| Durable DAG | Restart-stable graph identity, merge semantics, and graph compaction |
| Cold spill and restore | Hot/cold transition latency, serialized bytes, storage bandwidth, continuation parity |
| Adaptive scheduler | Throughput, p50/p95 latency, deferred time, fairness, and state-hit rate |
| Breakout integration | End-to-end wall time, final-answer quality, accepted-branch rate, and fallback rate |

No leg advances on a speed claim without a raw JSON artifact, a compact report,
the exact parent and candidate identities, and a stated workload.
