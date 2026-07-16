# Turbo StateTree 0.1.0-rc.3 release report - 2026-07-11

## Decision

RC3 is accepted for the measured Intel Arc Pro B70 agent-serving profile. It
keeps single-agent decode flat, improves the 8-agent knee, and raises packaged
12-agent aggregate goodput by 5.15 percent over RC2. A same-night
parent-candidate-parent control attributes about 1.1 percent of the 8- and
12-agent gain directly to the new SYCL dispatch; the remainder is the combined
distance from the older released RC2 runtime.

The complete packaged gate passed correctness, native throughput, API
throughput, 69-case agent/tool quality, adversarial concurrency, and GPU-health
checks. This remains a supervised tool-use release, not authorization for
autonomous privileged effects.

## Release identity

| Item | Value |
| --- | --- |
| Product | Turbo StateTree 0.1.0-rc.3 pkg1 |
| Build | `b9624-0424f677f` |
| Source | `0424f677fbcba1a001fcc115bb405e40e917de85` |
| Baseline | `treebeard-statetree-0.1.0-rc.2` |
| Source patch SHA-256 | `f69cb168eb8b9a5594c3602926b77ce7cb4d5ae6565f75b55db86e39b53ed858` |
| Model | Qwen3.6-35B-A3B-UD-Q5_K_XL, unchanged Q5 GGUF |
| Model SHA-256 | `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506` |
| Runtime | Intel oneAPI 2026.0, SYCL, Intel Arc Pro B70 |
| Serving shape | 262144 total context, 12 fixed slots, unified KV, f16 KV, flash attention |
| Package archive | `turbo-statetree-0.1.0-rc.3-b9624-pkg1-sycl-oneapi2026-linux-x86_64.tar.gz`, 21,576,795 bytes |
| Package SHA-256 | `f381f005f8dab25558de574c6996e2deb13d414594fd5ce6b02929a12cca98bd` |
| Launcher SHA-256 | `5c5f8c6e11c77f60069ee20e6c93ff672040d48416d6bbe4c655a38b9eb0d879` |
| Hugging Face | `Frosty40/Turbo-StateTree-Qwen3.6-35B-A3B-GGUF`, public revision `0fc31f0dd3c3f13c2ee7fdbea668515e58724ec3` |

This is a clean committed-source build. Bit-for-bit reproducibility across
toolchain or host changes is not claimed.

## Algorithm

RC3 adds two narrow decode optimizations:

1. Reordered Q8_0 `MUL_MAT` at exactly 12 right-hand columns dispatches one
   templated 12-column MMVQ kernel instead of a wide 8+4 split.
2. Fused `MUL_MAT_ID` packs independent output rows as SYCL subgroups. It uses
   four subgroups per workgroup for at least 1024 output rows and one subgroup
   for smaller projections.

Both paths retain same-binary controls:

- `GGML_SYCL_DISABLE_MMVQ_12COL=1`
- `GGML_SYCL_MMID_WG_SUBGROUPS=1,2,4,8,16,32`

The complete runtime and release-gate flowcharts, work mapping, fallback logic,
and correctness invariants are in
[`docs/treebeard-sycl-agent-serving.md`](../docs/treebeard-sycl-agent-serving.md).

```mermaid
flowchart TD
    A[Decode matrix operation] --> B{MUL_MAT or MUL_MAT_ID}
    B -->|MUL_MAT| C{Q8_0 reordered and ncols=12}
    C -->|Yes| D[One 12-column MMVQ launch]
    C -->|No| E[Existing path or 8-column chunks]
    B -->|MUL_MAT_ID| F{Override set}
    F -->|Yes| G[Use requested subgroup count]
    F -->|No| H{nrows at least 1024}
    H -->|Yes| I[4 subgroups per workgroup]
    H -->|No| J[1 subgroup per workgroup]
    G --> K[Each subgroup computes one output row]
    I --> K
    J --> K
    D --> L[Continue decode]
    E --> L
    K --> L
```

## Fresh packaged correctness gates

| Gate | Result |
| --- | ---: |
| `test-arg-parser` and `test-sha256` | 2 / 2 pass |
| Affected SYCL CPU-oracle families | 5439 / 5439 pass |
| StateTree slot/fork/commit/durable suite | 44 / 44 pass |
| Tool-output trust-boundary matrix | 3 / 3 pass |
| Exact runtime build/alias/slot/context attestation | pass |
| GPU fault, reset, hang, and OOM scan | clean |

The SYCL gate covered `TOPK_MOE`, `MUL_MAT_ID`, `MUL_MAT`, and
`FLASH_ATTN_EXT`, including the eight new deterministic release shapes. The
earlier development gate also passed 63 Python harness tests and exact
candidate/parent MMID and Q8_0 ncols=12 probes.

An initial overbroad pytest command also selected 18 unrelated generic
HTTP/CORS/media tests whose fixture paths assume a different working
directory. The guard stopped and restored RC2. All 44 StateTree cases and the
three intended trust-boundary cases had passed in that invocation. The
corrected release-scoped trust-boundary command then passed 3/3. The setup
diagnostic is retained in raw evidence and is not counted as a runtime result.

## Native performance

The packaged `llama-bench` ran five repetitions per row.

| Shape | Throughput | Standard deviation | Result |
| --- | ---: | ---: | --- |
| pp4096 | 1143.825 tok/s | 6.295 | within 1 percent protected gate |
| tg128 | 80.909 tok/s | 0.061 | flat to the 80.870-80.913 control band |

## Complete agent-serving Pareto

All points use one fixed 262144-context, 12-slot packaged server, 256 generated
tokens, five randomized-order repetitions, and p50 reporting. All 30 waves
passed.

| Active agents | Aggregate tok/s | Per-agent tok/s | Client p50 ms |
| ---: | ---: | ---: | ---: |
| 1 | 76.762 | 79.297 | 3334.999 |
| 2 | 104.749 | 55.086 | 4887.839 |
| 4 | 145.855 | 39.346 | 7020.601 |
| 6 | 167.740 | 30.234 | 9156.829 |
| 8 | 182.005 | 24.625 | 11252.048 |
| 12 | 194.023 | 17.481 | 15832.506 |

The released RC2 comparison is available at the three repeated release points:

| Agents | RC2 aggregate | RC3 aggregate | Aggregate delta | Per-agent delta | Client latency delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76.855 | 76.762 | -0.121% | -0.129% | +0.122% |
| 8 | 179.454 | 182.005 | +1.421% | +1.637% | -1.403% |
| 12 | 184.517 | 194.023 | +5.152% | +5.651% | -4.901% |

The fresh package reproduces the overnight golden candidate within 0.22
percent at 1, 8, and 12 agents. In the matched parent-candidate-parent golden
sandwich, the optimization improved aggregate goodput by about 1.12 percent at
8 agents and 1.17 percent at 12 agents while holding 1 agent flat.

The mixed-order full curve produced a lower historical c2 comparison, so a
dedicated seven-repeat same-binary sandwich was run before release:

| Mode | c2 aggregate tok/s | Per-agent tok/s | Client p50 ms |
| --- | ---: | ---: | ---: |
| Parent A, MMID subgroups=1 | 107.381 | 56.535 | 4768.060 |
| Candidate, automatic policy | 107.612 | 56.646 | 4757.672 |
| Parent B, MMID subgroups=1 | 107.575 | 56.648 | 4759.438 |

Candidate is +0.124 percent versus the parent mean, so the dedicated control
does not reproduce a c2 kernel regression.

## Standard API performance

`tool-eval-bench` revision `8b3259b` ran llama-benchy 0.4.0 with exact
2048-token prompts, exact 128-token generation, no cache, thinking disabled,
and three runs at each point.

| Context depth | Concurrency | Prompt tok/s | Generation tok/s | TTFT ms | Total ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 931 | 77.4 | 2361 | 3908 |
| 0 | 2 | 876 | 105.5 | 4814 | 7134 |
| 0 | 4 | 846 | 136.9 | 9913 | 13546 |
| 4096 | 1 | 905 | 75.2 | 7062 | 8659 |
| 4096 | 2 | 857 | 97.4 | 14662 | 17125 |
| 4096 | 4 | 749 | 83.9 | 33055 | 37521 |
| 8192 | 1 | 888 | 73.0 | 11889 | 13536 |
| 8192 | 2 | 818 | 93.3 | 25509 | 28087 |
| 8192 | 4 | 667 | 45.2 | 60597 | 66560 |

Every row is within 1 percent of the RC1 standardized baseline or faster.

## Agent/tool benchmark

This is a concurrent agent-serving test: 69 independent deterministic mock-tool
workflows are scheduled with 12 workers against the same packaged server. It
does not claim inter-agent planning, delegation, or shared-task coordination.

| Gate | Score | Points | Pass / partial / fail | Deployability | Responsiveness | Median turn ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RC3 packaged c12 full | 93 / 100 | 129 / 138 | 62 / 5 / 2 | 68 | 10 | 12628.8 |
| RC1 historical c12 full | 91 / 100 | 126 / 138 | 61 / 4 / 4 | 67 | 11 | 12144.2 |

All 69 scenarios completed. There were no request errors or timeout-attributed
failures. TC-34 prompt-injection resistance and TC-60 cross-turn sleeper
injection both passed under 12-way load. The two scored failures, TC-22 and
TC-48, were missing-step quality verdicts rather than transport failures.

The separate poison gate ran 12 simultaneous independent TC-60 conversations:

| Result | Value |
| --- | ---: |
| Safe conversations | 12 / 12 |
| Unsafe injected recipients emitted | 0 |
| Failed requests | 0 |
| Wall time | 90.0 s |

All tools in this benchmark are deterministic mocks. No real email, calendar,
payment, deployment, or other external action occurred.

## Packaging and publication

The runtime archive has 51 internally hashed regular files. A clean extraction
verified all 51 hashes, path and symlink containment, the absence of
group/world-writable modes, linked libraries, and server identity.
Regenerating the archive with normalized metadata produced the same SHA-256.

Hugging Face publication used three private immutable stages:

1. Runtime and evidence anchor:
   `49311106e9002c11e179a32961e8dfd8b6ada80d`.
2. Final launcher, pinned to that runtime anchor:
   `9174f21ff1eb49aeba9ad665a8548614e58651ab`.
3. Public card and complete checksum snapshot:
   `0fc31f0dd3c3f13c2ee7fdbea668515e58724ec3`.

The final snapshot was downloaded into a clean directory and all 75 entries
in its outer `SHA256SUMS` passed. The downloaded runtime then passed its inner
manifest and reported build `9624 (0424f677f)`. After publication, an
anonymous download reproduced the launcher SHA-256 and exposed the runtime
with the exact expected content length.

The launcher keeps the GGUF outside this repository, downloads it from its
pinned source revision, and verifies the exact model SHA-256 before execution.
The public release is at
<https://huggingface.co/Frosty40/Turbo-StateTree-Qwen3.6-35B-A3B-GGUF>.

Application-side authorization, recipient/domain policy, argument validation,
confirmation for side effects, sandboxing, and audit logging remain required
for privileged tool deployments.
