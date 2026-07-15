# Treebeard J-Space G0 sequence lifecycle

Date: 2026-07-15

Status: request-scoped actuator lifecycle accepted. Together with the artifact
identity and disabled-path invariance slices, this closes G0. It does not accept
a semantic sensor, pooling schedule, feedback controller, or production rollout.

## Frozen candidate

- Implementation commit: `a5f709b2da4340497aa353198d9d3392ac95960c`
- Model: `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`
- Full model SHA-256:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`
- Probe runtime SHA-256:
  `578628d0c351e2920194d357160ea370a5e9841c01c2454c2c0490ee315dbda9`
- Server runtime SHA-256:
  `393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c`
- Server implementation SHA-256:
  `d100d2a8b99733a7640342a0ba538551b8c390240ba7b621edceec47d433e8e5`
- SYCL runtime SHA-256:
  `3a4f7403d075d7f4c9dfeee43e0a0f104fef3dd322d99896e9bc863b0c63d8da`
- Backend: Intel B70, full model offload, batch and microbatch 64, fusion
  enabled, SYCL graph capture disabled for the diagnostic.
- Exact committed acceptance artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-b70/20260715-040813`

The guarded artifact recorded the implementation commit above and an empty
source patch. The rebuilt binaries reported that commit, so the measurements
are attributable to committed source rather than a dirty worktree.

## Runtime contract

An installed control vector keeps its legacy context-wide behavior until the
first `llama_adapter_cvec_seq_set`. That transition is one-way for the installed
vector and fail-closed: every unspecified sequence receives scale zero. A token
belonging to multiple logical sequences with different scales is rejected before
memory mutation.

Sequence mode adds one F32 scale input per token. Each active control-vector
layer constructs the direction-by-token outer product with a K=1 matrix
multiplication and adds it to the residual. Graph reuse includes the mode in its
compatibility key and fills scales from the current microbatch sequence IDs.

Whole-sequence remove, copy, and keep operations remove, copy, and retain the
controller scale. The server exposes an optional finite
`jspace_control_scale`; after sequence mode begins, omission means protected
scale zero. Cached prompt state is discarded on scale change, and token-only
prompt-cache reuse is disabled in this mode because it cannot identify the
control state used to create KV/GDN tensors.

StateTree snapshots append a versioned 12-byte `JSPCVEC1` plus F32 scale
trailer. It participates in the digest and memory budget. Materialization
validates and strips the trailer before restoring llama state, then reinstalls
the scale. Legacy snapshots without the trailer remain accepted as unsteered.

## Exact mixed-sequence gate

`--verify-sequence-lifecycle 4` evaluated three same-shape, two-sequence
controls for every prompt: all-off `(0,0)`, all-on `(1,1)`, and mixed `(1,0)`.
For each greedy step it retained both complete 248,320-value logit vectors,
sampled deterministic top-1, and serialized each logical sequence.

Across code, neutral prose, factual completion, JSON/tool-shaped text, and a
109-token multi-decode boundary prompt:

- all five cases passed;
- prompt token counts were `13, 5, 13, 109, 8` in artifact glob order;
- the mixed active sequence bit-matched all-on logits, tokens, and state;
- the mixed protected sequence bit-matched all-off logits, tokens, and state;
- the vector caused an observable logit change in every case;
- full-vocabulary bytes compared were `39,731,200`;
- compared serialized state sizes were `66,274,688`, `66,110,688`,
  `66,274,688`, `68,242,688`, and `66,172,188` bytes;
- reset, cancellation, fork, commit, slot reuse, and snapshot restore hooks all
  passed for every case;
- the earlier disabled-path corpus also passed in the same run, comparing
  another `39,731,200` full-vocabulary bytes exactly.

## Live server and StateTree gate

A four-slot candidate server exercised request ownership through the public
server transaction paths. It verified:

- an active scale-one request and an interleaved protected scale-zero request;
- exact scale inheritance by fork source and destination;
- scale-zero cleanup of the commit loser and scale-one retention by the winner;
- snapshot capture and materialization into a fresh physical slot;
- erasure followed by slot-zero reuse with the request field omitted, leaving
  all four slots at protected scale zero and none active or reserved.

The snapshot stored 12 tokens, `66,110,696` state bytes and `66,110,744` total
payload bytes. The 12-byte difference from the corresponding unextended state
is the controller trailer. Its digest was
`sha256:17152ed271853a4bca0b13181ce4eaef6c5a2cfbcf50c46e036b197e01b8a0c2`.
Capture took 142.156 ms, materialization 14.781 ms, and commit 0.012 ms in this
correctness gate; these single samples are not throughput claims.

## Production guard

The kernel journal contained zero reset, hang, fault, OOM, or panic signatures.
The exit trap restored the exact production server SHA and build
`b9627-3fcf1c626`, model alias, 262,144-token context, and all 12 slots. A live
inference completed, all slots were idle, no StateTree families remained, the
service was active, and `NRestarts=0`.

## Evidence boundary and next gate

G0 now proves fail-closed artifact admission, exact disabled behavior, and
request-scoped actuator ownership for the tested Qwen3.6/B70 runtime. It does
not prove that the current anchor-logit coordinates represent semantic state,
that Fibonacci pooling beats matched alternatives, or that feedback is safe or
fast enough for production.

G1 must fit and freeze an anchor-free semantic sensor using disjoint training,
calibration, and held-out sets. It advances only on predefined held-out
accuracy, calibration, invariance, abstention, and protected-task gates; the G0
actuator stays diagnostic-only until those gates pass.
