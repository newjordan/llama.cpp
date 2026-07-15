# J-Space causal logit probe

`llama-jspace-probe` evaluates a prompt and writes a single JSON document to
stdout. For each requested vocabulary token it reports the raw logit and the
full-vocabulary log-probability at the last prompt position. It also reports the
L2 norm and RMS of the final prompt column from the last post-block residual
(`l_out-39` for the 40-layer Qwen3.6 model). llama.cpp logs stay on stderr, so
stdout can be redirected directly to a JSON file.

The probe uses the common llama.cpp model/context arguments. It also uses the
normal control-vector loader, including scaled vectors and layer ranges:

```console
./build/bin/llama-jspace-probe \
  -m /home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  -c 4096 -b 512 -ub 512 -ngl 0 \
  -p "Write one sentence about a long-awaited reunion." \
  --token-ids 15420,6051,22292,17434 \
  --probe-vector joy=/path/to/joy.gguf \
  --probe-vector sadness=/path/to/sadness.gguf \
  --verified-model-sha256 25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506 \
  --probe-strengths=-2,-1,0,1,2 \
  --fibonacci-pool-max 144 \
  --include-residual-vector \
  --control-vector-layer-range 39 39 \
  > mood-dose-sweep.json
```

`--token-ids` accepts a comma-separated list of integer vocabulary IDs and can
be repeated. The JSON `logprob` values are a temperature-1 softmax over every
model vocabulary logit, before any sampler transforms.

Repeated `--probe-vector NAME=PATH` options share `--probe-strengths`. The model
is loaded once, a baseline is evaluated once, and each nonzero vector/dose pair
replays the identical prompt. A requested zero dose reuses the exact baseline
result. Before every real evaluation the probe clears both memory metadata and
data, resetting Qwen3.6 KV and recurrent/GDN state. Standard
`--control-vector-scaled FILE:SCALE,...` vectors remain supported and form the
base intervention for the baseline and every sweep run.

Every `--probe-vector` also requires `--verified-model-sha256`. The runner must
compute the full model-file SHA-256 before invoking the probe; the option is an
attestation, not a fast in-tool hash. Before model allocation or vector-tensor
loading, the probe reads only GGUF metadata and requires the Phase-0 schema, an
exact base-model digest match, an exact `NAME`/axis match, and one of the three
defined direction normalizations (`unit_l2_regularized_dual`,
`raw_regularized_dual`, or `direct_pre_rms_contrast`). Missing, malformed, or
unknown identity metadata fails closed. The attested digest and accepted
artifact identity are copied into the output JSON.

Residual collection uses a selective evaluation callback. By default it
transfers only the final column of the last `l_out` tensor. Fibonacci pooling
retains only its largest complete suffix, capped at 144 columns (1.125 MiB for
Qwen3.6), rather than retaining complete layer activations. Baseline and every
nonzero dose include `residual.l2`, `residual.rms`, `residual.dimension`, and
the exact tensor name in JSON.
`--include-residual-vector` additionally emits that exact column under
`residual.values`; it is omitted by default to keep ordinary sweep JSON compact.

`--fibonacci-pool-max N` adds an opt-in offline causal suffix extractor over the
post-final-block residual columns. `N` is capped at 144 tokens. The probe emits
complete uniform suffix means at the unique horizons
`1, 2, 3, 5, 8, 13, ... <= min(N, prompt_tokens)`. The Fibonacci values choose
only the horizon schedule; they are not used as mystical coordinate weights.
Each row is nonnegative, causal, and sums to one, so constants and linear
coordinate closure are preserved. The evaluator itself accepts any strictly
increasing suffix-horizon schedule; Fibonacci is a separate generator, so
logarithmic and linear controls can reuse identical pooling arithmetic.

The sparse matrix row for horizon `h` has weight `1/h` on the final `h` prompt
columns and zero elsewhere. The JSON matrix shape covers the full tokenized
prompt domain; `retained_suffix` separately records the bounded residual buffer
and its global prompt-position offset. Writing `u_k` for the unnormalized
suffix sum (the reported mean is `u_k/F_k`), its non-overlapping Fibonacci
factorization is:

```text
u_k(t) = u_(k-1)(t) + u_(k-2)(t - F_(k-1)),
F_k    = F_(k-1) + F_(k-2).
```

The delay in the second term is required; combining two overlapping suffix
means at the same position would double-count recent tokens. A streaming
Phase-1 controller can later use the equivalent sliding update
`u_h(t) = u_h(t-1) + x_t - x_(t-h)` with one ring buffer per logical sequence.
That per-sequence streaming state is not implemented by this diagnostic probe.

Pooling JSON records its sparse support, prompt positions, L2/RMS, token-domain
semantics, and exact column accounting. With `--include-residual-vector`, every
pooled row also includes its 2048 values; otherwise only compact diagnostics are
serialized. Horizon one is checked for exact equality with the existing final
residual. Multi-batch or future graph changes that expose a different number of
columns fail closed instead of silently returning a misaligned pool.

This is an observer candidate, not evidence that Fibonacci spacing is better.
The required falsification compares equal feature counts and tuning budgets;
boxcar controls also match maximum support, while EMA controls match mean age
and effective sample size. If those tie, the honest result is only that causal
multiscale pooling helps.

The broader sensor/actuator/controller design and frozen evidence boundary are
documented in [`docs/treebeard-jspace-steering.md`](../../docs/treebeard-jspace-steering.md).
The exact-Qwen3.6 multi-batch pooling smoke is recorded in
[`reports/treebeard-jspace-fibonacci-smoke-20260713.md`](../../reports/treebeard-jspace-fibonacci-smoke-20260713.md).
The fail-closed artifact admission check is recorded in
[`reports/treebeard-jspace-g0-identity-20260715.md`](../../reports/treebeard-jspace-g0-identity-20260715.md).

The baseline full-vocabulary logits stay in-process (about 1 MiB for Qwen3.6)
and are not serialized. Each sweep run reports full-vocabulary
`collateral.kl_base_to_run`, `collateral.kl_run_to_base`, and
`collateral.jensen_shannon` in nats. A zero-dose run reports exact zeros.

A failed or incompatible control vector makes the command fail instead of
returning an unsteered result.

Run the model-independent deterministic smoke test with:

```console
./build/bin/llama-jspace-probe --self-test
```
