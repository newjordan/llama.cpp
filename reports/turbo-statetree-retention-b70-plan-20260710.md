# Turbo StateTree Retention B70 Maintenance Plan - 2026-07-10

## Status

Executed by explicit user request on 2026-07-10. Both matched transaction
windows and the production-scale pressure gate passed. Production `:8093` was
left inactive for continued R&D rather than automatically restored. Results
are in `reports/turbo-statetree-retention-b70-acceptance-20260710.md`.

The material below is preserved as the pre-execution safety and acceptance
plan.

The recommended execution is two maintenance windows so the matched
performance matrix and production-scale pressure gate each have independent
rollback coverage. Neither window should begin without explicit approval.

## Live Rollback Identity

| Item | Value |
| --- | --- |
| Unit | `turbo-head-a2edfe66f-rollback-8093.service` |
| Unit path | `/run/user/1000/systemd/user/turbo-head-a2edfe66f-rollback-8093.service` |
| Unit SHA256 | `3d935f4a0ab6839ef80aebae3d9fed70d670d3b769242c0a85c534d497e7e407` |
| Binary | `/home/frosty40/builds/turbo-experimental-build/bin/llama-server` |
| Binary SHA256 | `7cdac806e66937bc956a84a7ccf9a9ff35313f5ece86d1212e3fbb11019dd9fb` |
| Server library SHA256 | `fefc441977337e0b76726751ce11c8b5f3ec1c4db06f8324c7bf708a3fa6aef6` |
| Model SHA256 | `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506` |
| Live build | `b61-a2edfe66f` |
| Live shape | 12 slots, 262,144 context |
| Live alias | `turbo-head-a2edfe66f-Qwen3.6-35B-A3B-Q5-np12-kvu-c262144-ub1024` |

The unit is runtime-only and static. A fresh copy of the unit, command line,
hashes, `/health`, `/props`, `/slots`, process FDs, and Hydra health must be
captured under the new artifact root before stopping it.

## Matched Benchmark Builds

| Role | Version | Binary | Binary SHA256 | Server library SHA256 |
| --- | --- | --- | --- | --- |
| Parent | `71 (6051ddf31)` | `/home/frosty40/turbo/build-retention-parent-b70/bin/llama-server` | `463e2d7941d7330a1d836bab39f8428cf4dd1da7627f12ece71557cc49d753ab` | `079812927bbc9916ac3a87fef8bfd4d63c60fc2d1c0650ea31ef398c8fc63559` |
| Candidate | `72 (70acde5e6)` dirty | `/home/frosty40/turbo/turbo-combined/build/bin/llama-server` | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `6622138a3975aafb8cbe7000ab7ed34fb7a56cd2c519c3b1ddb5e13876ecbd28` |

Both builds use:

```text
IntelLLVM 2026.0.0
CMAKE_BUILD_TYPE=Release
GGML_CCACHE=ON (ccache absent for both)
GGML_NATIVE=ON
GGML_OPENMP=ON
GGML_SYCL=ON
GGML_SYCL_DNN=ON
GGML_SYCL_F16=ON
GGML_SYCL_GRAPH=ON
GGML_SYCL_HOST_MEM_FALLBACK=ON
GGML_SYCL_TARGET=INTEL
```

Both binaries resolve all runtime dependencies after sourcing oneAPI. The
candidate build is up to date with every changed C++ server source.

## Window A: Matched Performance Matrix

Use a fresh artifact root and arm an automatic rollback before the production
stop. The rollback must terminate only `llama-server` processes on port 8098,
then start the exact captured production unit. Allow 50 minutes for the matrix
and arm rollback no later than 60 minutes after the stop.

Common workload:

```text
model: Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
-ngl 99 -ncmoe 0 --no-op-offload
-kvu -np 12 -c 262144
-fa on -ctk f16 -ctv f16
-b 8192 -ub 1024 -t 16
prefixes: 1024,8192,32768
family: 6 slots
branch suffix: 8 tokens
forced branch decode: 32 tokens
repeats: 5
GGML_SYCL_ENABLE_FUSION=1
LLAMA_KV_PAGE_PROBE=1
```

Run sequentially:

1. Parent dense, manual cleanup.
2. Candidate dense, order-balanced manual and commit, 30,000 ms lease, exact
   20-checkpoint ceiling `1317288560`.
3. Compare and stop immediately on any compatibility, contract, throughput,
   fork, RSS, or VRAM failure.
4. Parent persistent-fragmented, with 4K survivors in slots `0/2/4/6/8/10`.
5. Candidate persistent-fragmented with the same controls and ceiling.
6. Compare and stop immediately on any failure.

The 20-checkpoint ceiling is above the previously measured fragmented peak of
18 checkpoints (1,185,559,704 bytes) while still making the active policy
observable. It must not evict the benchmark fixture; explicit pressure belongs
in Window B.

Acceptance remains the documented gate: zero contract failures, all required
groups and metrics present, branch throughput at least 95% of parent, fork p50
no greater than `1.10 * parent + 0.25 ms`, and RSS/DRM total no greater than
parent plus 64 MiB.

## Window B: Production-Scale Pressure

Arm a new independent rollback timer before this window. Run only the candidate
with the exact production model and 12-slot/262K shape:

```bash
LLAMA_KV_PAGE_PROBE=1 \
python3 scripts/turbo-statetree-retention-gate.py \
  --bin /home/frosty40/turbo/turbo-combined/build/bin/llama-server \
  --model /home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  --label b70-retention-pressure \
  --commit 70acde5e61b92a44884fb45f056f591fb4ab5390-dirty \
  --out-dir ARTIFACT_ROOT/pressure \
  --port 8098 --ctx 262144 --parallel 12 --prefix-tokens 1024 \
  --pressure-widths 1,6,12 --only-pressure-widths \
  --batch 8192 --ubatch 1024 --threads 16 \
  --ngl 99 --ncmoe 0 --flash-attn on
```

The width shapes require:

- one checkpoint: admit a replacement and reclaim the oldest ordinary state;
- six checkpoints: reclaim the older six-member family atomically;
- twelve slots: enforce an 11-checkpoint ceiling by skipping exactly one
  optional checkpoint, without partially evicting the active family.

The `1,2,4` CPU analogue passed before this plan was written.

## Mandatory Restoration

For either window, restoration is part of the gate:

1. Stop every managed server on 8098.
2. Start the exact captured production unit.
3. Require `/health` success and exact `/props` build, alias, slot, and context
   values.
4. Verify the executable, unit, and command line against the preflight hashes.
5. Verify Hydra health and surface routing.
6. Capture DRM state plus server and kernel logs; reject any Xe/DRM hang,
   reset, fault, OOM, assertion, or device-loss signature.
7. Cancel the automatic rollback only after all restoration checks pass.

No canary, rollout, or accepted-baseline claim follows from a partial window.
