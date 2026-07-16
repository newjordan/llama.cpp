# Treebeard J-Space G0 disabled-path invariance

Date: 2026-07-15

Status: disabled-path bit-identity slice accepted. G0 remains open for
per-sequence lifecycle isolation across reset, cancellation, fork, commit, slot
reuse, and snapshot restore.

## Frozen candidate

- Implementation commit: `2b837755de16ad7f16b5181499f1bb306ecf14ca`
- Probe version: `9643 (2b837755d)`
- Model: `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`
- Full model SHA-256:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`
- Probe runtime SHA-256:
  `85a682db4344529352a76ed8193fc2b96d057521f4f4e7f69e21b3d50dc7370b`
- SYCL runtime SHA-256:
  `3a4f7403d075d7f4c9dfeee43e0a0f104fef3dd322d99896e9bc863b0c63d8da`
- Backend: Intel B70, full model offload, batch and microbatch 64, context 128,
  fusion enabled, SYCL graph capture disabled for the diagnostic.
- Exact committed acceptance artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-b70/20260715-032538`

The acceptance harness recorded the exact source head above and an empty source
patch, so the measured runtime corresponds to the committed revision.

## Contract

`--verify-disabled-invariance N` requires exactly one identity-bound probe
vector, no base control vectors, and 1 through 32 greedy tokens. It compares:

1. the untouched baseline;
2. an untouched no-API replay;
3. a graph rebuild with no artifact installed;
4. an artifact enable followed by explicit
   `llama_set_adapter_cvec(data=null)` disable.

Every evaluation clears memory metadata and data. Each trace retains the full
248,320-value vocabulary logits at every greedy step, uses deterministic top-1
sampling with lowest-token-ID tie break, and serializes the logical sequence
state. The gate requires bit-identical logits, identical sampled tokens, and
byte-identical sequence state. It fails closed on the first mismatch.

## Failure isolated

The first fully offloaded corpus passed four shallow prompt strata but failed a
109-token prompt when batch 64 forced two `llama_decode` calls. CPU and a B70
batch-128 single-call run passed. Fresh B70 batch-64 processes produced different
baseline logits, while fresh batch-128 processes were identical.

The bounded falsification sequence established:

- explicit synchronization, graph-reuse disable, host KV/recurrent storage, and
  general SYCL optimization disable did not close the failure;
- fused and decomposed GDN, convolution and SSM state writes, SSM convolution,
  concat, and L2 normalization did not discriminate it;
- `-ngl 2` passed and `-ngl 3` failed, identifying the first additional B70
  transformer layer;
- moving all layer-38 ordinary matrix multiplies to CPU passed;
- moving only `blk.38.ffn_gate_inp.weight` to CPU passed;
- the router tensor is FP32 `[2048, 256]`, and the failing large projection used
  the oneDNN FP32 GEMM path;
- globally disabling oneDNN passed.

The apparent recurrent-state boundary bug was therefore an FP32 MoE-router GEMM
reproducibility failure amplified by expert selection, not a state handoff bug.

## Fix

FP32 tensors whose semantic weight name contains `.ffn_gate_inp.weight` now use
the existing oneMKL FP32 GEMM path instead of oneDNN. The router remains resident
and executes on B70; the change does not offload the operation or disable oneDNN
for unrelated GEMMs.

## Exact committed acceptance

The guarded corpus covered code, neutral prose, factual completion,
JSON/tool-shaped text, and the 109-token boundary prompt. All five cases passed:

- prompt token counts: `13, 8, 5, 13, 109`;
- greedy steps per arm: `8`;
- full-vocabulary reference bytes compared against each control: `39,731,200`
  (`119,193,600` across the three controls);
- serialized reference-state bytes compared against each control: `333,074,920`
  (`999,224,760` across the three controls);
- no-API replay: exact;
- no-artifact graph rebuild: exact;
- enable then explicit disable: exact;
- sampled token sequences: exact;
- kernel fault signatures: `0`.

A second clean-process long-prompt run at
`/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-b70/20260715-032848`
produced an identical baseline JSON to the committed corpus. Its long-prompt
fingerprint was log-sum-exp `20.725883719057972`, residual L2
`15.106915077214175`, and token-13 logit `3.5476174354553223`.

## Performance and production guard

The guarded fresh long-prompt wall samples, including model load, service stop,
and exact service restoration, were 52 and 51 seconds before the fix and 51
seconds on the clean committed fix. This coarse end-to-end gate detects no
regression; it is not a microbenchmark claim for the router alone.

Every guarded run restored production build `b9627-3fcf1c626`, verified the exact
server SHA-256 and model alias, completed an inference probe, found all 12 slots
idle and no StateTree families, and ended with the service active and
`NRestarts=0`.

## Evidence boundary

This closes G0 artifact admission plus disabled-path bit identity for the fixed
single-sequence corpus and its multi-decode boundary. It does not yet prove that
observer/controller state follows production sequence ownership. The next G0
slice must exercise reset, cancellation, fork, commit, slot reuse, and snapshot
restore with at least two interleaved logical sequences before any controller is
allowed into production serving.
