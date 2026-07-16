# Turbo StateTree Managed Publish B70 Acceptance — 2026-07-10

## Decision

Accept WAL-backed managed publish as the next StateTree R&D baseline. Object
publication and first-owner retention are now one recoverable lifecycle
transaction rather than two client-visible durability boundaries. The complete
CPU suite, synthesized crash cuts, terminal-space pressure test, matched CPU
gate, and production-shape B70 gate passed with zero failures.

This is not deployed. `turbo-head-a2edfe66f-rollback-8093.service` remained
inactive, and ports 8093 and 8098 were clear after acceptance.

## Transaction Contract

`POST /snapshots/{snapshot_id}?action=publish` accepts the snapshot digest,
owner, and `pinned` or `cache` retention class.

The ordered durable worker executes three stages:

1. Append and sync a publish intent containing owner, digest, class, and
   revision.
2. Publish the immutable object through temp write, file sync, atomic rename,
   directory sync, and full canonical read-back verification.
3. Append and sync the owner commit, atomically moving the intent into the live
   reference graph.

Before stage one, admission proves that a compacted checkpoint containing the
pending intent plus a terminal commit or abort record fits the manifest byte
ceiling. A transaction that starts cannot become unfinishable solely because
the WAL reaches its configured limit.

On startup, each pending intent is reconciled before ordinary owner/content
agreement is checked:

- If its indexed object exists and fits the recovery load ceiling, the server
  fully loads and verifies it, then commits the owner.
- If no object exists, the intent is durably aborted.
- If the pending object fails verification, the intent is aborted and the
  corrupt object is erased.
- If the configured load ceiling cannot safely verify the pending payload,
  startup fails closed instead of deleting a potentially valid object.

Duplicate publish is idempotent at both layers: the existing object is verified
and reused, while the owner/class transaction is reused without a new revision.
Erase also fences pending publish intents, not only committed references.

## Correctness Evidence

| Check | Result |
| --- | ---: |
| Managed publish focused tests | 3/3 passed |
| Durable server tests including prior manifest coverage | 8/8 passed |
| Complete StateTree server suite | 32/32 passed |
| Harness and host-safety suite | 45/45 passed |
| CPU managed-publish lifecycle gate | passed, zero failures |
| B70 managed-publish lifecycle gate | passed, zero failures |

The crash tests synthesize exact checksummed WAL cuts. Intent plus verified
object is committed at restart; intent without object is aborted at restart.
The 512-byte pressure test admits two complete publish transactions, rejects
the third before writing an intent, remains within the ceiling, and replays the
four committed revisions after restart.

## Final B70 Gate

Workload: Qwen3.6-35B-A3B Q5_K_XL on Intel Arc Pro B70,
`-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -b 8192 -ub 1024 -fa on`, f16 KV,
1,024 captured prefix tokens, 32 suffix tokens, and deterministic continuation.
Two concurrent clients published the same snapshot and owner through the new
transactional route.

| Signal | Result |
| --- | ---: |
| Durable payload | 86,860,780 bytes |
| Fresh intent + object + owner transaction | 170.156 ms worker I/O |
| Fresh manifest revision / bytes | 2 / 590 |
| Duplicate object verification | 100.891 ms |
| Duplicate queue wait | 170.064 ms |
| Duplicate owner revision growth | zero |
| Spill-time `/states` latency | 0.313–0.613 ms |
| Independent inference client time | 102.754 ms |
| Inference finished before publish queue drained | yes |
| Restart replayed revision / refs | 2 / 1 |
| Retained erase attempt | HTTP 503 |
| Restart cold load and verification | 91.063 ms |
| Sequence materialization | 18.411 ms |
| Total cold endpoint | 109.541 ms |
| Cold-load `/states` latency | 0.263–0.390 ms |
| Hot/cold deterministic continuation | exact parity |
| Checkpoint compaction | 590 to 449 bytes; 2 records to 1 |
| Final release fsync | 5.244 ms |
| Final object erase | 15.345 ms; disk bytes returned to zero |

The added WAL boundaries do not reintroduce state-thread blocking. The 170 ms
publish transaction and the queued duplicate remained observable while an
independent inference completed and state inspection stayed sub-millisecond.

## Runtime Identity

- `llama-server`:
  `4fe7d2b986db0038e5eb008dbaeffd632b2d483174469d4251269b52bacf1dba`
- `libllama-server-impl.so`:
  `66916fe6377efbcbce28420d9fbe0c619af7545f48173758803eedb495417f4f`
- `libllama.so`:
  `f99006e9bba072e39db39d461a7ea9e5381f673e2a9c3422fdc1dd24aebbbee9`
- Model:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`

Frozen build:
`/home/frosty40/turbo/build-managed-publish-accepted-b70`.

Raw evidence:

- `/home/frosty40/turbo/results/statetree-publish/20260710-final/cpu/cpu-publish-final.result.json`
- `/home/frosty40/turbo/results/statetree-publish/20260710-final/b70/b70-publish-final.result.json`

## Boundary and Next Lever

Managed publish closes the first-owner crash gap, but `cache` remains metadata:
disk pressure does not yet reclaim cache-only managed objects. The next lever is
deterministic cache reconciliation with a durable eviction transition, explicit
managed-object reachability, oldest-revision victim ordering, pinned-owner
fencing, exact target-byte reclamation, and restart completion of erase/manifest
disagreement. Raw unmanaged spill must remain outside that automatic policy.
