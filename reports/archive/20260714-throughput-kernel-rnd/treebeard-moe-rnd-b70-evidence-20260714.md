# Treebeard MoE B70 control evidence - 2026-07-14

## Outcome

The only end-to-end improvement accepted from this campaign is the combined
T2 + T3 default path:

- T2 accumulates routing-weighted expert results inside down-projection MMID.
- T3 pairs gate/up MMID rows and applies SwiGLU before global storage.

Against the midpoint of same-binary control A and control B, the combined path
improved aggregate wall throughput by 3.73% p50 and 3.80% mean at twelve active
agents. It improved one-agent throughput by 1.73% p50 and 1.79% mean.

The later whole-MoE composite pipeline is not an improvement. It regressed the
twelve-agent p50 by 6.65% and remains opt-in. Grouped down projection and SYCL
graph replay also remain parked behind their controls.

## Primary metric

The promotion metric is aggregate generated wall tokens per second. It is
measured from the client-visible wall interval for 256 generated tokens per
request. The primary serving shape is twelve simultaneous agents; one agent is
the regression guard.

The benchmark does not claim the naive 12x single-agent ceiling is attainable.
It measures the candidate against the same build with only the candidate paths
disabled.

## Fixed benchmark identity

| Item | Value |
| --- | --- |
| Accelerator | Intel Arc Pro B70 Graphics |
| Driver | 1.15.38308+1 |
| Model | Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf |
| Context and slots | 262144 context, 12 slots, unified f16 KV |
| Server batch | `b8192`, `ub1024` |
| Generation | 256 tokens per request, seed 42 |
| Candidate order | control A, candidate, control B |
| Combined repeats | 10 per arm and active-agent shape |
| Composite repeats | 4 per arm and active-agent shape |
| Build | Release, SYCL, F16, DNNL, Level Zero, VMM, graph support compiled |

Every arm within a comparison used byte-identical `llama-server`,
`libggml-sycl.so`, and `libllama-server-impl.so` files. Controls changed only
environment gates in the same binary.

## Accepted T2 + T3 control comparison

Control A and B set `GGML_SYCL_DISABLE_MOE_DOWN_REDUCE=1` and
`GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`. The candidate leaves both accepted paths
enabled. Graph execution is disabled for all arms.

| Agents | Metric | Control A | Candidate | Control B | Control midpoint | Candidate delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | aggregate p50 tok/s | 76.969300 | 78.218926 | 76.812328 | 76.890814 | +1.7273% |
| 1 | aggregate mean tok/s | 76.988378 | 78.182813 | 76.620856 | 76.804617 | +1.7944% |
| 12 | aggregate p50 tok/s | 199.683226 | 207.440849 | 200.274572 | 199.978899 | +3.7314% |
| 12 | aggregate mean tok/s | 199.538266 | 207.447134 | 200.163711 | 199.850988 | +3.8009% |

There were 60 measured samples across the three arms and two active-agent
shapes, with zero failed samples. The candidate gain is larger than the
control-A/control-B drift at the primary twelve-agent shape.

Runtime SHA-256 values for all three arms:

```text
llama-server              f869b2fe53a7338681c5043c2271b9dfec6dd19d7a335737f5d6fd940b775626
libggml-sycl.so           7ea9b70faddce0e6ec65d9896a9fe13d6dcb83e37be5ea31d8d8891a12f7969c
libllama-server-impl.so   5871921d931e28e03989b8f310dd71904d36385bbf91b4a3b2a25562b339061e
```

Correctness and activation evidence:

- Ordered down reduction: 7/7 focused SYCL-versus-CPU comparisons passed.
- Dual gate/up SwiGLU: 3/3 focused SYCL-versus-CPU comparisons passed.
- Production logs recorded Q5_K and Q6_K hits at the twelve-token shape.
- No decode-shape overlap, materialized fallback, failed request, device reset,
  Level Zero fault, or OOM signature was recorded.

Raw rig artifact:

`/home/frosty40/turbo/treebeard-work/results/treebeard-moe-combined/20260714-135527`

Raw result JSON SHA-256 values:

```text
control A   384c0acdc225d0d38cb48475773f629dc1af5072e13acbb713cfc3611d5a9698
candidate   5eb1fe34ef0b5eea4bacc6f9a029be90676c7ceb0adc0f55339303f5ebf289ed
control B   b81b246747dc1ef0fd724cbe06892cfce39fd43dd3888b3b0afc553b07bf5a8e
```

## Rejected whole-MoE composite comparison

The control retains the accepted T2 + T3 paths. The candidate additionally
sets `GGML_SYCL_ENABLE_MOE_PIPELINE=1`. This isolates the incremental value of
the composite topology rather than comparing it with the older baseline.

| Agents | Metric | Control midpoint | Composite candidate | Candidate delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | aggregate p50 tok/s | 78.393622 | 78.168357 | -0.2874% |
| 1 | aggregate mean tok/s | 78.409919 | 78.166647 | -0.3103% |
| 12 | aggregate p50 tok/s | 207.491939 | 193.697510 | -6.6482% |
| 12 | aggregate mean tok/s | 207.630707 | 193.448864 | -6.8303% |

There were 24 measured samples with zero failed samples. The production-shaped
CPU comparison passed, the real 12- and 48-token graphs activated, and the
hardware fault scan was empty. The regression is therefore an end-to-end
performance rejection, not a correctness failure. Reduced row-level
parallelism outweighed the saved activation traffic and submission.

Raw rig artifact:

`/home/frosty40/turbo/treebeard-work/results/treebeard-moe-pipeline/20260714-151507`

Raw result JSON SHA-256 values:

```text
control A   81e6dcdd4ff1a53e48ffdf0c052ea44fe9ed311276e751f413821ec129e0df17
candidate   c0db23bc074b9038cd587e3cb2e2d75a378a00930f10d81e9f769b6e16f2c45c
control B   004ff6ab14ce17d4162192adf5a48cd024401cf86484f934d1947e2cd068c785
```

## Other candidate decisions

| Candidate | One-agent p50 delta | Twelve-agent p50 delta | Decision |
| --- | ---: | ---: | --- |
| Tail reduction confirmation | effectively flat | -0.06% versus midpoint | Park |
| Pre-allocation liveness | not promoted alone | +0.87% versus midpoint | Retain only as T2/T3 enabler |
| Grouped ordered down | +0.06% | -0.26% | Park; opt-in only |
| SYCL executable graph replay | -4.57% | -4.24% | Park; disabled by default |
| Whole-MoE composite pipeline | -0.29% | -6.65% | Park; opt-in only |

The detailed task and artifact ledger is in
`docs/treebeard-throughput-rnd.md`.

## Reproduction

The exact guarded harness used for these comparisons is preserved as
`scripts/treebeard-moe-b70-aba-guarded.sh`. It verifies the live RC2 service
identity, stops it only inside a restoration trap, runs correctness before the
performance arms, enforces activation and control assertions, hashes every
runtime arm, scans kernel faults, and restores the original service.

Accepted combined gate:

```bash
TREEBEARD_COMBINED=1 \
TREEBEARD_BENCH_AGENTS='1 12' \
TREEBEARD_BENCH_REPEATS=10 \
scripts/treebeard-moe-b70-aba-guarded.sh
```

Composite rejection gate:

```bash
TREEBEARD_PIPELINE_ABA=1 \
TREEBEARD_BENCH_AGENTS='1 12' \
TREEBEARD_BENCH_REPEATS=4 \
scripts/treebeard-moe-b70-aba-guarded.sh
```

## Interpretation boundary

The accepted numbers establish a repeatable same-rig, same-binary improvement
for the fixed B70/Qwen steady one- and twelve-agent shapes. They are not a
cross-device claim, a confidence interval, or a continuous-arrival serving
claim. The next product-value gate must measure completed requests per second,
p95 time to first token, and p95 inter-token latency under arrival-driven load.

No candidate from this campaign has been deployed to the live RC2 service.
The restored service remained build `b72-70acde5e6`, 12 slots, and context
262144.
