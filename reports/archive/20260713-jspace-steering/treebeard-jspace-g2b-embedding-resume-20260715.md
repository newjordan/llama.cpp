# Treebeard J-Space G2b embedding-stage resume

Date: 2026-07-15

Status: frozen after the embedding server startup-check failure and before any
embedding request, embedding output, evaluation, selection, or holdout output.

## Failure and permitted correction

The G2b run completed all base-model generation and all six blind judge
batches. The Nomic model then loaded successfully with aggregate context 8192,
16 slots, and 512 tokens per slot. The guard incorrectly required `/props` to
report the aggregate 8192 value; `/props` correctly reports the per-slot value
512. That local assertion exited before the first embedding request.

This resume changes only that instrumentation assertion. It requires 16 slots
and per-slot context 512 while retaining the command-line aggregate context
8192. It reuses the exact frozen response and judgment files and does not
regenerate them. No response preference, candidate metric, embedding,
evaluation, or selection was inspected before freezing this correction.

The resume may run only the six frozen Nomic embedding captures, the six
frozen evaluations, the deterministic selector, the kernel audit, and exact
production restoration. Every original G2/G2b gate remains unchanged. The
holdout remains unavailable unless the selector returns one eligible policy.

## Bound resume inputs

- G2b preregistration SHA-256:
  `fec0fecd84096720f2c46f4d4e9f5e0264b2c3ceb8f74f8fa63df7063cce2079`;
- original run directory:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g2/development-g2b/20260715-163546`;
- original run attestation SHA-256:
  `0bad7e740d632e2a1c6e8d3ebfdb321519a32749140152528a354f546fb9305d`;
- frozen G2b routes SHA-256:
  `89eef9dd91c5c847c5828a3a72e46824b571d8169218ea3a73140b2352d2c54d`;
- sorted SHA-256 list for all 32 response and six judgment files:
  `39f8b5d95349ff5e4e9059c7e98f7465dc04bf5064813bd18dfbfdd06647ca67`;
- Nomic Embed Text v1.5 Q8 model SHA-256:
  `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`;
- embedding client SHA-256:
  `869a9e758e321ebb52fc613b3db7c1e46bc5668e985b77db16411a55b00471c4`;
- evaluator/selector SHA-256:
  `9d9ab88f1a6615b043e1d5d079ae8e9fc8b2b9de0948fd17c3f31a64cb676337`;
- anchor manifest SHA-256:
  `cab229a60bbcf1cd641070227bd4fd4c5bc8345605c36afe5c81850ff15ebb71`;
- server binary SHA-256:
  `393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c`;
- SYCL runtime SHA-256:
  `0a0536fd5eba5caf6c8c4b00febe4320e868fe19bc99bd42767eb77d6c872e61`.

The resume guard must verify the sorted input list with `sha256sum -c`, rehash
this amendment and every bound runtime/model/client, verify the exact live
production service before stopping it, record its own digest, and restore the
production build, alias, executable digest, slots, context, restart count,
idle state, and inference on every exit path.
