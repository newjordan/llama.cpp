# Turbo StateTree Durable Content B70 Acceptance — 2026-07-10

## Decision

Accept the explicit durable-content/cold-restore slice as the next R&D
baseline. It passed the complete server regression suite, focused adversarial
restart tests, a two-process CPU control, and a fresh production-shape B70
control. It is not deployed and does not make StateTree graph identities
durable.

The production unit `turbo-head-a2edfe66f-rollback-8093.service` remained
inactive. Ports 8093 and 8098 were clear after the gate.

## Architecture Accepted

- Hot immutable content can be explicitly spilled with
  `POST /snapshots/{id}?action=spill` and addressed thereafter by its canonical
  SHA-256 digest.
- A versioned little-endian envelope binds format, explicit compatibility ID,
  digest, token count, and serialized-state size.
- The operator-supplied compatibility ID hashes to an isolated namespace. The
  accepted gate binds model SHA, `libllama.so`, `libllama-server-impl.so`, and
  `cold-v1`.
- Linux publication uses a unique temporary file, complete write, file sync,
  atomic rename, directory sync, then full read-back and digest verification
  before acknowledging success.
- Startup removes interrupted temporary objects, discovers valid content, and
  counts malformed `.tss` bytes against the namespace budget without trusting
  or indexing them.
- Runtime materialization validates envelope, exact file size, and canonical
  SHA-256 before touching a slot. Indexed payload bytes are checked against an
  explicit cold-load ceiling before token/state buffer allocation.
- Disk admission is exact and reject-only. There is no silent eviction.
- Cold restore creates fresh process-local state/fork/node identities and
  records `materialized_content_digest`; old snapshot IDs and graph parentage
  are intentionally not durable.
- Explicit durable erase, `/props`, `/states`, `/snapshot-contents`, and
  Prometheus telemetry cover bytes, high water, recovery, malformed objects,
  runtime integrity, spills, restores, erases, and rejections.

## Verification

| Check | Result |
| --- | ---: |
| Argument parser, including CLI and environment load ceiling | passed |
| Cold/snapshot gate Python compilation | passed |
| Focused durable server tests | 2/2 passed |
| Complete StateTree server suite | 26/26 passed |
| Harness and host-safety tests | 45/45 plus 5 subtests passed |
| CPU two-process cold gate | passed, zero failures |
| B70 two-process cold gate | passed, zero failures |

The adversarial tests cover duplicate spill, startup discovery, crash-temp
cleanup, malformed-file accounting, explicit compatibility isolation,
pre-allocation load-ceiling rejection, bit corruption after discovery,
untouched destination slots on failure, durable erase, stale content, and
disk-pressure rejection without a stray object.

## Final B70 Control

Workload: Qwen3.6-35B-A3B Q5_K_XL, Intel Arc Pro B70,
`-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -b 8192 -ub 1024 -fa on`, f16 KV,
1,024-token captured prefix, 32-token suffix, and deterministic 8-token
continuation. Snapshot and cold-load ceilings were 512 MiB; the namespace disk
ceiling was 1 GiB.

| Signal | Result |
| --- | ---: |
| Canonical payload | 86,860,780 bytes |
| Durable object | 86,861,119 bytes |
| Hot capture | 93.541 ms |
| First spill, including sync and read-back | 163.439 ms |
| Duplicate object verification | 98.105 ms |
| Hot materialization | 18.947 ms |
| Restart cold load and verification | 88.036 ms |
| Cold sequence materialization | 19.031 ms |
| Total cold materialization endpoint | 107.093 ms |
| Crash temporary objects recovered | 1 |
| Pre-restart hot handle after restart | HTTP 503 |
| Hot/cold continuation | exact token and content parity |
| Round-trip capture digest | exact match |

Content digest:
`sha256:3ac301c2b3a19bc617402de027210007fdfb0803b78d5a5514d07c19b6eb2e8f`.

## Runtime Identity

- `llama-server`:
  `e76ba8aa5fab1df8020efd725b329a493d3e7bb30064802c4ed05b9c9a113d37`
- `libllama-server-impl.so`:
  `cd1b07267c563a498cd90381982380e9baf8dfc3adc76b80fff4811f76672504`
- `libllama.so`:
  `9bd50e48ab0acce1551025b229d0ef8e8ae514514b0551765f4d4f8f8f529b73`
- Model:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`

The raw B70 result is
`/home/frosty40/turbo/results/statetree-cold-b70/20260710-accepted-v2/b70/b70-cold-accepted-v2.result.json`.
The CPU control is
`/home/frosty40/turbo/results/statetree-cold-b70/20260710-accepted-v2/cpu/cpu-cold-accepted-v2.result.json`.
The accepted build is frozen at
`/home/frosty40/turbo/build-durable-snapshot-accepted-b70`.

## Boundary and Next Lever

Superseded for serving-thread behavior by
`reports/turbo-statetree-async-io-b70-acceptance-20260710.md`, which moves all
durable file I/O to an ordered worker while preserving this content contract.

This accepts durable immutable content, not a persistent DAG. Lineage IDs,
node IDs, hot snapshot handles, journal sequences, and parent edges remain
process-local. The next architectural lever should be policy above this exact
content substrate: an explicit durable manifest/index with ownership,
retention, and compaction semantics, or an automatic hot/cold tiering policy
with bounded asynchronous I/O. Neither should weaken content verification or
invent restart-stable graph identity implicitly.
