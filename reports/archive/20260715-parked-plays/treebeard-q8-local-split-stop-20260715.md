# Treebeard Q8 local-split stop report (2026-07-15)

## Decision

Park the cooperative local-memory split for the production 12-column reordered
Q8_0 decode kernel. Do not retain or enable the candidate implementation.

The exact candidate accelerated a representative isolated kernel by 2.21x, but
the guarded full-model A/B/A measured only +1.03% single-stream throughput and
-9.05% 12-agent throughput relative to the control midpoint. It fails both the
25% single-stream and 20% 12-agent major-fusion gates.

## Candidate

- Device: Intel Arc Pro B70, driver `1.15.38308+1`.
- Production decode geometry: 12 columns, reordered Q8_0 weights, and Q8_1
  activations.
- The reference kernel keeps 12 column accumulators per lane. The candidate
  staged each weight row in workgroup-local memory and split the outputs across
  three 16-lane subgroups with four columns per subgroup.
- It retained the reference subgroup size, Q8 half-DP4A operations, scale and
  block accumulation order, and subgroup reduction topology.

## Isolated and correctness results

- Representative microbenchmark baseline: `0.169421 ms`, `111.405 GB/s`.
- Four-column local split: `0.0765302 ms`, `2.21378x` baseline throughput.
- The 98,304 checked outputs were bitwise exact: zero mismatches and zero
  maximum absolute difference.
- Microbenchmark artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-q8-local-split-micro/20260715-074831/result.txt`
- Microbenchmark result SHA-256:
  `3f276f13b492557922f6777b2b1a046e2fcaaf969439ef670bb87125f3e0e2a3`
- Focused backend test `MUL_MAT(q8_0,f32,m=512,n=12,k=256)` passed its CPU
  comparison and logged candidate dispatch, 1/1 tests passed.
- Backend-test result SHA-256:
  `3b04e3a70ed770a0953c475a37f29e614f92ef75eae584c4cda1ca63ce2c9854`

## Guarded full-model A/B/A

The same candidate binary ran control A, candidate, and control B. Each arm
completed three single-stream and three 12-agent samples with zero failures.
State-IO fusion and the accepted MoE defaults were identical across arms;
speculation was disabled.

| Active agents | Control A mean TPS | Candidate mean TPS | Control B mean TPS | Candidate vs control midpoint |
|---:|---:|---:|---:|---:|
| 1 | 78.9069 | 80.0310 | 79.5236 | +1.03% |
| 12 | 234.9788 | 213.3307 | 234.1280 | -9.05% |

- Artifact root:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-q8-local-split-aba/20260715-075820`
- Result SHA-256 values:
  - control A: `432e9d54ec7edc8427b6e0b26e5d01aaaf0a58ef6176d60ff311aa3418101716`
  - candidate: `c3baa407cf326b22d97c1c90dd7ae873e6f73685329b86af423e33dcd807b254`
  - control B: `3d4f5922377e13f973ba759c10ab91715573f5ece9290c9e3c3e6d55e23fcf54`

## Interpretation

The isolated kernel benefits from sharing the global weight-row load and
reducing per-lane accumulators. At full-model scale, the larger workgroups,
local-memory allocation, staging barrier, and reduced scheduling flexibility
overwhelm that saving under 12 concurrent slots. This is an occupancy and
system-contention failure, not a correctness failure.

Splits of six and three columns were already inferior in the representative
microbenchmark (`0.991x` and `1.982x`). The best four-column geometry has no
remaining tuning margin capable of crossing the system gate, so this family is
closed.

## Safety and repository state

- No candidate kernel or harness enablement is retained.
- The A/B/A kernel-signature scan was empty.
- The guard restored `turbo-statetree-rc4.service`, verified the production
  executable identity, completed an inference, and reported `NRestarts=0`.
