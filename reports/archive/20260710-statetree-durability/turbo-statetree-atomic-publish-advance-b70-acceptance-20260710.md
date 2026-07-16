# Turbo StateTree Atomic Publish-And-Advance B70 Acceptance - 2026-07-10

## Decision

Accept atomic publish-and-advance as the next Turbo StateTree R&D release
candidate. The transaction atomically creates managed owner reachability and
advances a durable logical head after a verified immutable object is present.

This is not a production deployment. The production service on port 8093 stayed
stopped throughout the gate, and the gate used an isolated server on port 8098.

## Transaction Contract

- A combined intent fences the owner, target digest, and logical head before
  immutable-object publication begins.
- A single terminal manifest record creates the managed owner reference and
  advances the generation-and-digest CAS head together.
- Exact retries preserve the terminal revision and report a deduplicated
  transaction. Changed transaction fields conflict.
- Restart commits a verified pending target and aborts an absent or corrupt
  target. Stale object, head, erase, and retired-name operations remain fenced.

## CPU Evidence

- Focused atomic server tests: 4 passed, 40 deselected.
- Complete StateTree server suite with a local Qwen3.5 0.8B model: 44 passed
  in 159.952 seconds. The JUnit result is
  `/tmp/turbo-rc-statetree-pytest-20260710.xml`.
- Isolated CPU atomic cold gate passed with no failures:
  `/tmp/turbo-rc-atomic-cold-gate-20260710/rc-atomic-cpu.result.json`.
- Host/preflight/benchmark harness tests: 45 passed in 0.11 seconds.
- Portable SHA-256 fallback vectors and segmented unaligned updates pass in
  `test-sha256`. This covers empty input, `abc`, and one million `a` bytes with
  1, 7, 63, 64, 65, 511, and 4096 byte update chunks.

## B70 Gate

The isolated gate used Qwen3.6-35B-A3B Q5_K_XL on Intel Arc Pro B70 with
`-ngl 99 -ncmoe 0 -c 262144 -np 12 -kvu -fa on -ctk f16 -ctv f16 -b 8192
-ub 1024 -t 16`. It used a 1,024-token captured prefix, a 32-token suffix,
and 512 MiB content, disk, and load ceilings with a 64 MiB manifest ceiling.

| Signal | Result |
| --- | ---: |
| Gate result | passed, zero failures |
| Captured payload | 86,860,780 bytes |
| Initial capture | 124.103 ms |
| Initial managed publish worker I/O | 225.936 ms |
| Spill-time `/states` probe range | 0.417-0.719 ms |
| Independent request during spill | 101.102 ms, completed before spill drain |
| Verified cold load and materialization | 137.687 ms total (119.579 + 18.023 ms) |
| Cold continuation | 1,024-token cache hit; 8-token decode at 83.57 tok/s |
| Atomic replacement publish-and-advance | 215.996 ms total worker transaction |
| Atomic retry | deduplicated at the same revision 5 |
| Restart-time `/states` probe range | 0.359-0.647 ms |
| Stale handle/head, erase fence, retired-name ABA | HTTP 503, all fenced |
| Terminal durable integrity failures | 0 |
| Final cleanup | durable content count 0; manifest revision 10 |

The complete machine-readable artifact, three server logs, and isolated store
are under:

```text
/home/frosty40/turbo/results/statetree-atomic-publish-advance/20260710-rc/b70
```

Runtime identities:

```text
llama-server:            5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687
libllama-server-impl.so: a5de1e59f477c9f48d0c23b1564b82f4d47b2f6224305a19b1995a197abb30ec
libllama.so:             873e763483225f4df1d952e18e55644b13a5a2d637e5985267eeb9af2d578485
libggml-sycl.so:         0315247a415714ed402996931a6aaf0103f1625ec6ba45458ec7a5878eef7aa3
model:                   25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506
```

## Release Scope

This candidate is qualified for the StateTree durable-content API on the tested
Turbo workload. It is not a general llama.cpp release: the generic B70 CTest
matrix still has non-StateTree failures in `test-llama-archs` (the existing
DeepSeek32 meta-backend path) and `test-backend-ops`. No `ggml/` source file is
modified by this candidate. Those SYCL failures require a separate kernel lane
before any broader framework release claim.
