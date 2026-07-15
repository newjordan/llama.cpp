# Treebeard Q8 independent-split stop report (2026-07-15)

## Decision

Park the balanced independent-subgroup split for the production 12-column
reordered Q8_0 decode kernel. Do not retain or enable the implementation.

The exact 6+6 candidate accelerated the isolated representative kernel by
5.92x, but the guarded full-model A/B/A measured only +0.74% single-stream and
-3.00% 12-agent throughput relative to the control midpoint. It fails both the
25% single-stream and 20% 12-agent major-fusion gates.

Together with the local-memory split and XMX stop results, this closes the
current Q8 12-column kernel-geometry family. A new Q8 candidate must change the
memory/computation mechanism, not repartition the same DP4A work.

## Candidate and isolation result

- Device: Intel Arc Pro B70, driver `1.15.38308+1`.
- Production decode geometry: 12 columns, reordered Q8_0 weights, Q8_1
  activations, subgroup size 16.
- The candidate used one launch with two independent 6-column subgroups per
  output row. It avoided workgroup-local memory and reduced live accumulators
  from 12 to six per lane while rereading the weight row for the second split.
- Representative baseline: `0.165989 ms`, `113.709 GB/s` counted as one weight
  pass.
- Independent 6+6 split: `0.0280562 ms`, `5.91629x` baseline throughput.
- Alternate independent splits were slower: 4+4+4 `5.7359x`, 3+3+3+3
  `5.34083x`.
- The 98,304 checked outputs were bitwise exact: zero mismatches and zero
  maximum absolute difference.
- Microbenchmark artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-q8-independent-split-micro/20260715-080617/result.txt`
- Microbenchmark result SHA-256:
  `fc1b46a05193226765b74a63c559bf4cebad307ba310fca3fa12b4731dec418c`

The large isolated ratio includes cache reuse between adjacent subgroup tasks
and repeated benchmark iterations. It was used only as a candidate-entry
signal, not as a system-performance claim.

## Integrated correctness

The production-layout implementation retained the reference half-DP4A, scale
and block accumulation order, and subgroup reduction topology.

- Focused backend test: `MUL_MAT(q8_0,f32,m=512,n=12,k=256)`.
- Candidate activation trace present.
- CPU-oracle comparison: 1/1 passed.
- Backend-test result SHA-256:
  `7baef4298c8e4c7d0562dbb1a1b364e65db3ae298061b46983775b05d9294315`

## Guarded full-model A/B/A

The same candidate binary ran control A, candidate, and control B. Each arm
completed three single-stream and three 12-agent samples with zero failures.
State-I/O fusion and accepted MoE paths were identical across arms;
speculation was disabled.

| Active agents | Control A mean TPS | Candidate mean TPS | Control B mean TPS | Candidate vs control midpoint |
|---:|---:|---:|---:|---:|
| 1 | 79.0389 | 79.5793 | 78.9547 | +0.74% |
| 12 | 235.5833 | 228.3535 | 235.2712 | -3.00% |

- Artifact root:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-q8-independent-split-aba/20260715-081100`
- Result SHA-256 values:
  - control A: `d176afe30139bafca3b7990e85f3c0834fa650dd1610a8f73c827618205a984a`
  - candidate: `68c97d5981654868bf99db5f7ac96e8145a5ec5c47023809ad2e0855052d4a4e`
  - control B: `c6d83db057bd652bbfd6079126ffe1a630e518365b77cf8cc244c2a6441e1f10`

## Interpretation

The independent split removes the local-memory/barrier penalty of the prior
candidate and exploits cache residency in isolation. At model scale, however,
the repeated weight traffic competes across many distinct matrices and 12
slots. Cache reuse is not dependable, so extra weight requests reduce
aggregate goodput even though lower register pressure helps slightly at one
agent.

The existing direct 12-column kernel remains the correct production geometry.
The historical 8+4 two-launch fallback was already slower, the cooperative
local-memory split regressed 12-agent throughput by 9.05%, and this balanced
one-launch split regresses it by 3.00%.

## Safety and repository state

- No candidate kernel or harness enablement is retained.
- The A/B/A kernel-signature scan was empty.
- The guard restored `turbo-statetree-rc4.service`, verified the production
  executable identity, completed an inference, and reported `NRestarts=0`.
