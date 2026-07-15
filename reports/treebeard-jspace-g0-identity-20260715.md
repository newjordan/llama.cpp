# Treebeard J-Space G0 identity fence

Date: 2026-07-15

Status: identity-admission slice accepted. Disabled-path bit identity was
accepted subsequently; G0 remains open for per-sequence lifecycle isolation.

## Frozen identity

- Implementation commit: `74f7b77b126878117ac1ee8b9235fc573688b960`
- Model: `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`
- Runner-verified full model SHA-256:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`
- Accepted artifact:
  `qwen36_q5_phase0_joy_rawdual_l39.gguf`
- Accepted artifact schema: `treebeard.jspace.phase0-cvec.v0`
- Accepted axis: `joy`
- Accepted direction normalization: `raw_regularized_dual`
- Backend: CPU-only probe pinned to CPUs 16–31. The live B70 service remained
  active and healthy with `NRestarts=0`.

## Contract

Every diagnostic `--probe-vector NAME=PATH` now requires
`--verified-model-sha256`. Before allocating the model or loading vector tensor
data, the probe reads GGUF metadata only and requires:

1. the exact supported Phase-0 schema;
2. a well-formed base-model SHA-256 equal to the runner attestation;
3. artifact axis metadata exactly equal to `NAME`;
4. one of `unit_l2_regularized_dual`, `raw_regularized_dual`, or
   `direct_pre_rms_contrast`.

The accepted model attestation and artifact identity are copied into the JSON
result. Missing or mismatched values terminate the command instead of silently
loading an unbound legacy control vector.

## Verification

The exact committed source was rebuilt with oneAPI 2026.0 and reported build
commit `74f7b77b1`. Its model-independent self-test passed, including digest
normalization and accepted/rejected schema, digest, axis, and normalization
cases.

Fast command-level negatives rejected:

- a vector with no `--verified-model-sha256`;
- the accepted artifact paired with a different 64-hex model digest;
- the `joy` artifact requested under the forged axis name `sadness`.

The positive exact-model run succeeded with a zero-dose `joy` raw-dual vector.
It reported architecture `qwen35moe`, 40 layers, 2048 embedding dimensions,
the exact runner-attested model digest, and the exact artifact identity above.
The zero arm reused the baseline and reported exactly zero KL and
Jensen-Shannon collateral, as required.

## Evidence boundary

This prevents accidental cross-model, cross-axis, unknown-schema, and
unknown-normalization admission in the diagnostic vector path. The digest is a
runner attestation; the probe does not rehash the multi-gigabyte model on every
invocation. Production harnesses must compute the full hash before supplying
it.

This result did not satisfy all of G0 by itself. Disabled-path logits, sampled
tokens, and logical state were accepted later in
`reports/treebeard-jspace-g0-disabled-invariance-20260715.md`. Controller and
observer state must still prove isolation across reset, cancellation, fork,
commit, slot reuse, and snapshot behavior.
