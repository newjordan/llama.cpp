# Treebeard Q8 XMX 12-column stop report (2026-07-15)

## Decision

Park the direct reordered-Q8_0 XMX decode path.  Do not retain or enable the
candidate implementation.

The full 8x16x32 integer tile was fast enough in isolation, but it changed
production-model top-1 results.  Emulating the reference kernel's two scaled
16-element DP4A halves still changed top-1 and reduced the isolated speedup to
1.90x, which caps the profile-weighted system opportunity below the 20% major
fusion gate.

## Candidate and prior

- Device: Intel Arc Pro B70, driver `1.15.38308+1`.
- Production decode shape: 12 columns; representative dense QKV weight shape
  `8192x2048`, reordered Q8_0 weights and Q8_1 activations.
- Native matrix query: signed int8 x signed int8 to signed int32 supports
  `M=1..8`, `N=16`, `K=32`, subgroup 16.
- Serialized post-state-IO profile attributed about 30% of total decode time to
  dense reordered-Q8 matrix paths.  The candidate therefore required both exact
  token parity and enough kernel gain to plausibly produce at least 20% at the
  12-agent system level.

## Results

### Full 32-element XMX dot

- Representative microbenchmark: `0.296585 ms` DP4A versus `0.0710958 ms`
  XMX, `4.17163x`; all 12 microbenchmark top-1 rows matched.
- Artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-xmx-q8-micro/20260715-071407/candidate/result.txt`
- Result SHA-256:
  `4440b17defbc941dd281ec9e8260cac73001a47ce6e8699f23166ccb1865bc1e`
- A focused `MUL_MAT(q8_0,f32,m=512,n=12,k=256)` backend test passed its CPU
  NMSE comparison and confirmed the XMX dispatch launched.
- Backend-test artifact SHA-256:
  `f297d60c99d096da7d47bcdec7bc4776653e54139fb7a6b3fb2ab425de28433c`
- Strict full-model smoke failed serial top-1 audit.  Observed mismatches included
  `serial=2374, batched=2932` and, after matching the scale multiplication order,
  `serial=552, batched=18922`.
- Full-model result SHA-256 values:
  - `89d22f593da4272d90f988240c2caa53490fca505617066ebf38f9db5b6564cc`
  - `1df24e45ac3212ebc3123af57558e7affc5bada050a0a4a8b0267120964a81b2`

### Two 16-element scaled halves

- Representative microbenchmark: `0.297229 ms` DP4A versus `0.156454 ms`
  half-masked XMX, `1.89978x`; all 12 microbenchmark top-1 rows matched.
- Artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-xmx-q8-micro-half/20260715-073602/result.txt`
- Result SHA-256:
  `ec546d4e41c68cb6244d0c856333ec3b93414bd37c6670d3bf101a345eaca6cd`
- Strict full-model smoke still failed serial top-1 audit:
  `serial=271, batched=552`.
- Full-model result SHA-256:
  `16e6c79773ac802946b77f25613c12845328ac8d250ce3fd5025548c83f8dc1d`
- With roughly 30% of decode in eligible dense Q8 paths, a 1.90x kernel gain has
  an optimistic Amdahl ceiling near 14% end to end, before padding and dispatch
  overhead.  It cannot clear the 20% major-play gate.

### Checked matrix load

Intel's checked joint-matrix load was tested as a way to zero-fill columns
12..15 without a padding kernel.  On this driver it caused a Level Zero
`UR_RESULT_ERROR_DEVICE_LOST`; this instruction form must not be used on the
current rig.  The guard restored and verified the exact production service.

## Interpretation

The Q8 reference MMVQ result is sensitive not only to the integer dot product
but to floating-point accumulation topology across subgroup lanes and Q8
blocks.  Reproducing that topology on XMX would require enough extra partial
state, shuffles, or materialization to erase the remaining performance
headroom.  CPU-reference NMSE and synthetic top-1 were insufficient acceptance
tests; the serial production-model top-1 audit was the decisive gate.

## Safety and repository state

- No XMX candidate code or benchmark-harness enablement is retained.
- Every guarded run restored `turbo-statetree-rc4.service`, verified build
  `b9627-3fcf1c626`, verified the production executable SHA-256, completed an
  inference, and reported `NRestarts=0`.
- Kernel logs were clean for the ordinary padded-load candidates.  The checked
  load failure is isolated above and was not retried.
