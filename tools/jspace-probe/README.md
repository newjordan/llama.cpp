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
  --probe-strengths=-2,-1,0,1,2 \
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

Residual collection uses a selective evaluation callback: it transfers only
the final column of the last `l_out` tensor, rather than retaining complete
layer activations. Baseline and every nonzero dose include `residual.l2`,
`residual.rms`, `residual.dimension`, and the exact tensor name in JSON.
`--include-residual-vector` additionally emits that exact column under
`residual.values`; it is omitted by default to keep ordinary sweep JSON compact.

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
