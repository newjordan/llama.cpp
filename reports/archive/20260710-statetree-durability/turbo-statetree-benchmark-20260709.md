# Turbo StateTree Benchmark Gate - 2026-07-09

> Follow-up: the matched Qwen3.6-35B/B70 gate subsequently passed all 24
> dense/fragmented regression checks with zero failures. See
> `reports/archive/20260710-statetree-durability/turbo-statetree-b70-benchmark-20260709.md`. An initially mismatched
> `GGML_SYCL_F16=OFF` parent run was discarded before acceptance.

## Status

The first StateTree parent/candidate microbenchmark gate passes.

- 96 measured transaction samples completed with zero contract failures.
- All 15 dense and fragmented parent/candidate regression checks passed.
- Atomic commit reclaimed the exact loser checkpoint bytes in every candidate
  sample and preserved the winner byte-for-byte at the exposed state boundary.
- At the time of this microbenchmark the 35B/B70 gate had not run. The later
  matched production-size result is recorded in the follow-up report above.

The benchmark harness, policy, and prompt-state byte telemetry are new local
changes on top of StateTree commit `9a37cb8fd88a737a34d03aebc0d5575976805c59`.
The transaction implementation itself is unchanged from that commit.

## Compared Revisions

| Role | Revision | Behavior |
| --- | --- | --- |
| Parent | `8acb67d399318ee0525b1385625c1101090dd1d3` | Unified-KV fork plus manual loser erase |
| Candidate | `9a37cb8fd88a737a34d03aebc0d5575976805c59` | Generation-fenced fork, commit, and refork |

Both revisions were built CPU-only from detached worktrees with:

```text
CMAKE_BUILD_TYPE=Release
GGML_SYCL=OFF
GGML_CCACHE=OFF
LLAMA_CURL=OFF
GNU 13.3.0
```

This isolated CPU gate did not touch the B70 or production `:8093` service.

## Workload

Model:

```text
/home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf
```

This is a hybrid/recurrent Qwen model, so the checkpoint path matches the
deployed architecture family.

Common configuration:

```text
-kvu -np 4 -c 32768 -ngl 0
fanout=3
branch_suffix_tokens=8
branch_tokens=8
repeats=5
```

The dense gate used exact 128, 1,024, and 8,192-token prefixes. The fragmented
gate used `-np 8`, retained independent prompt state in slots `0/2/4/6`, and
ran the fork family in slots `1/3/5/7` at 1,024 and 8,192 tokens.

Sample inventory:

| Lane | Parent | Candidate | Total |
| --- | ---: | ---: | ---: |
| Dense | 15 manual | 15 manual + 15 commit | 45 |
| Fragmented | 10 manual | 10 manual + 10 commit | 30 |
| Fragmented 1K commit tail | 0 | 21 commit | 21 |
| Total | 25 | 71 | 96 |

## Parent/Candidate Regression

The comparison gate requires branch aggregate throughput of at least 95% of
parent, fork p50 no greater than `1.10 * parent + 0.25 ms`, RSS no greater than
parent plus 64 MiB, and zero contract failures.

Dense layout:

| Prefix | Parent fork p50 | Candidate fork p50 | Fork delta | Parent branch tok/s | Candidate branch tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.183 ms | 0.159 ms | -13.11% | 57.581 | 61.081 | +6.08% |
| 1,024 | 0.304 ms | 0.262 ms | -13.82% | 52.495 | 53.613 | +2.13% |
| 8,192 | 1.244 ms | 1.310 ms | +5.31% | 39.697 | 43.880 | +10.54% |

Fragmented layout:

| Prefix | Parent fork p50 | Candidate fork p50 | Fork delta | Parent branch tok/s | Candidate branch tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.271 ms | 0.255 ms | -5.90% | 57.902 | 57.835 | -0.12% |
| 8,192 | 1.204 ms | 1.179 ms | -2.08% | 41.969 | 42.876 | +2.16% |

All nine dense checks and all six fragmented checks passed. The positive
throughput deltas are not optimization claims; the gate establishes that the
new transaction bookkeeping did not produce a measured regression.

## Commit And Refork Latency

Dense candidate results:

| Prefix | Commit client p50 | Commit server p50 | Commit server p95 | Manual/commit speedup | Refork server p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.510 ms | 0.104 ms | 0.163 ms | 2.52x | 0.137 ms |
| 1,024 | 0.627 ms | 0.249 ms | 0.319 ms | 2.01x | 0.257 ms |
| 8,192 | 1.547 ms | 1.172 ms | 1.178 ms | 1.38x | 1.068 ms |

Fragmented candidate results:

| Prefix | Commit client p50 | Commit server p50 | Commit server p95 | Manual/commit speedup | Refork server p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.713 ms | 0.279 ms | 2.127 ms | 1.89x | 0.234 ms |
| 8,192 | 1.519 ms | 1.166 ms | 1.182 ms | 1.43x | 1.015 ms |

The five-sample fragmented 1K group contained one 2.579 ms server-side commit;
the other four were 0.250-0.318 ms. A separate 21-sample run on a fresh server
did not reproduce the long tail:

| Signal | p50 | p95 | Max |
| --- | ---: | ---: | ---: |
| Commit server | 0.248 ms | 0.311 ms | 0.320 ms |
| Commit client | 0.657 ms | 0.744 ms | 0.870 ms |

The original outlier remains in the raw artifact and in the five-sample table.

## Memory And Reclamation

The new `/slots` fields separate recurrent prompt-state memory from the
preallocated device KV pool:

```text
n_prompt_data_bytes
n_prompt_checkpoint_bytes
n_prompt_state_bytes
```

For this model and workload, every branch held one 20,201,932-byte checkpoint:

| State | Bytes | MiB |
| --- | ---: | ---: |
| Four-branch family before commit | 80,807,728 | 77.064 |
| Protected winner after commit | 20,201,932 | 19.266 |
| Reclaimed loser state | 60,605,796 | 57.798 |

Commit reclaimed exactly 75% of family checkpoint bytes, matching three losers
out of four members. This result was identical at every prefix and in dense and
fragmented layouts. The checkpoint payload is fixed-size recurrent state for
this model; prefix length primarily affects token-list and KV operations.

Expected shared-KV topology also contracted exactly:

| Prefix | Before commit | After commit | Reusable loser suffix cells |
| ---: | ---: | ---: | ---: |
| 128 | 188 | 143 | 45 |
| 1,024 | 1,084 | 1,039 | 45 |
| 8,192 | 8,252 | 8,207 | 45 |

These are topology-derived cell counts, not physical allocator evidence.
Physical cells and holes require the B70 KV page probe.

Process RSS did not fall by the full 57.798 MiB after every commit because the
host allocator may retain freed capacity. This proves why RSS alone is not an
adequate StateTree budget signal. The exact per-slot byte counters are the
authoritative reclamation measurement.

## Artifacts

Raw local artifacts:

```text
/tmp/turbo-statetree-bench-20260709/parent-8acb67d39.result.json
/tmp/turbo-statetree-bench-20260709/candidate-statetree.result.json
/tmp/turbo-statetree-bench-20260709/comparison.json
/tmp/turbo-statetree-bench-20260709/parent-fragmented-8acb67d39.result.json
/tmp/turbo-statetree-bench-20260709/candidate-fragmented-statetree.result.json
/tmp/turbo-statetree-bench-20260709/comparison-fragmented.json
/tmp/turbo-statetree-bench-20260709/candidate-fragmented-1k-tail.result.json
```

The compact durable data is in
`reports/archive/20260710-statetree-durability/turbo-statetree-benchmark-20260709-summary.json`.

## Harness Validation

- `scripts/turbo-statetree-bench.py` Python compilation passed.
- `tests/test_turbo_statetree_bench.py`: 15 tests passed.
- `tools/server/tests/unit/test_slot_fork.py`: 15 tests passed in 144.75s.
- `tools/server/tests/unit/test_ignore_eos.py`: 3 tests passed.
- `tests/test_turbo_speculative_breakout.py`: 16 tests passed.
- `tests/test_turbo_kv_page_ablate.py`: 16 tests passed.
- `cmake --build build --target llama-server -j 16`: passed under oneAPI.
- Dense managed-server smoke: 2 transaction modes passed.
- Fragmented managed-server smoke: commit and refork passed.
- Matched parent and candidate CPU builds passed.
- All 96 measured samples reported zero contract failures.
- Production `:8093` was untouched.

## B70 Gate Follow-Up

The required maintenance run is complete. It used the production model and
262K/12-slot shape, exact 1K/8K/32K prefixes, five-repeat dense and persistent
fragmented parent/candidate lanes, Xe DRM memory telemetry, physical page-probe
rows, a full-width 12-slot stress lane, and kernel-reset inspection. The
matched gate passed; exact results and the build-confounder audit are in
`reports/archive/20260710-statetree-durability/turbo-statetree-b70-benchmark-20260709.md`.
