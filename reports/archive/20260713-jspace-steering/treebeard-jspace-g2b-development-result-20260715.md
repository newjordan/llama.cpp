# Treebeard J-Space G2b development result

Date: 2026-07-15

Status: negative development result. No actuator policy is eligible and the
untouched holdout remains untouched.

## Outcome

The original 50-row G2 set stopped at its frozen routing gate with 16 active
rows. The separately preregistered, routing-blinded G2b extension used 98
train-only rows and produced 30 active rows, clearing the unchanged minimum of
20. All six preregistered policy/amplitude candidates were then generated,
judged, embedded, and evaluated.

No candidate reached the frozen active-preference minimum of 0.52:

| Policy | Dose | Changed active tokens | Active preference | Candidate / control / tie | Reference cosine delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| engage curiosity | 1/8 | 16.7% | 0.5000 | 0 / 0 / 30 | -0.000305 |
| engage curiosity | 1/4 | 30.0% | 0.5000 | 0 / 0 / 30 | +0.000071 |
| engage curiosity | 1/2 | 23.3% | 0.5000 | 0 / 0 / 30 | -0.000402 |
| match route | 1/8 | 16.7% | 0.4833 | 0 / 1 / 29 | +0.000107 |
| match route | 1/4 | 26.7% | 0.5000 | 0 / 0 / 30 | -0.000741 |
| match route | 1/2 | 33.3% | 0.5000 | 0 / 0 / 30 | -0.000284 |

The actuator was behaviorally nontrivial: every candidate changed at least 5
of the 30 active token sequences, and the largest candidate changed 10 of 30.
The blind, order-swapped judge nevertheless found zero order-consistent
candidate wins. Matched-route 1/8 produced the only order-consistent non-tie,
and it favored the scale-zero control.

The deterministic selector returned `stop_no_eligible_candidate`. No selected
policy exists, so generating holdout routing logits, responses, judgments, or
embeddings is prohibited under G2/G2b.

## Gates and safety diagnostics

Every candidate failed only `active_preference_min_0_52`. Every other frozen
development gate passed:

- 30 active rows across all six routed axes;
- changed-active rate at least 10%;
- each source preference at least 0.45;
- mean reference-cosine delta at least -0.005;
- anchor-emission, repeated-4-gram, and malformed-output deltas within limits;
- mean candidate/control generated-token length ratio exactly 1.0;
- zero order-consistent candidate-only factual, format, task-completion,
  refusal, repetition, or malformed flags;
- every inactive scale-zero candidate token sequence exactly identical to its
  paired control.

Across candidates, repeated-4-gram rate rose by 1.02 percentage points, below
the frozen 2-point development ceiling. Malformed-output delta was zero.
Unsolicited-refusal delta was zero. Anchor emission was unchanged or fell by
1.02 points.

## Instrumentation-only resume

After all response generation and all six blind-judge files completed, the
first embedding-server startup check compared `/props` per-slot context 512
against command-line aggregate context 8192 and exited. The server itself had
loaded normally, and zero embedding requests were sent. Production restored
successfully.

Before resuming, a separate amendment froze the exact SHA-256 list for all 32
response and six judgment files. The guarded resume verified that list and
changed only the context assertion to require 16 slots with 512 tokens per
slot. It then captured the six frozen embedding diagnostics, ran the six
evaluations and selector, audited the kernel journal, and restored production.
No response or judgment was regenerated.

## Bound evidence

- run directory:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g2/development-g2b/20260715-163546`;
- G2b routes SHA-256:
  `89eef9dd91c5c847c5828a3a72e46824b571d8169218ea3a73140b2352d2c54d`;
- frozen response/judgment input-list SHA-256:
  `39f8b5d95349ff5e4e9059c7e98f7465dc04bf5064813bd18dfbfdd06647ca67`;
- selector SHA-256:
  `0eb6890ca17fd5696ecf67583d69f96803abdbc522e275732fb9e9ae80815bee`;
- resume attestation SHA-256:
  `1cb30f2a2a275f492179e13a422b2a2d600a04460ce9ac7429cf48a78c426d29`;
- resume run-status SHA-256:
  `0765558dd9f41753ec989ab81f1752f6f38579f7600b670d480c8cbad4fe7d3e`.

Evaluation SHA-256 values:

| Candidate | SHA-256 |
| --- | --- |
| engage curiosity 1/8 | `7a46b9796e4a94a3e9d24aa4323d5497c6bc00c1466296b6621b39283d09ce9c` |
| engage curiosity 1/4 | `0e7fc091d5dcee026b0ad0c8a111a5e7e506508ec20cba28d54baf1fa2c7702a` |
| engage curiosity 1/2 | `43b43246efe094a1b30449d30d58e75acbc19d8d132562d694afca9913c26e29` |
| match route 1/8 | `d87d950935cf021965f466661c594d3c6f29814d2ea81b3335dd787f4a692266` |
| match route 1/4 | `8b6410e789f5c4dfe606e1f87e332548a96b74ef65ec751ee6592e1058a06156` |
| match route 1/2 | `9ac0b31f112ecbda0d9b3ca856f5af3a26550aa9784f8973a21a403e72df1934` |

## Production restoration

Both guarded exits and the embedding/evaluation resume restored the exact
production service. The final restoration verified build `b9627-3fcf1c626`,
the expected model alias and server executable digest, 12 slots, context
262144, `NRestarts=0`, idle slots, empty StateTree families, and successful
post-restore inference. The resume kernel audit found zero Xe/DRM reset, hang,
or fault signatures and zero OOM or kernel-panic signatures.

## Conclusion

At the bounded sub-inner doses tested here, a request-scoped final-head
raw-dual actuator can change sampled outputs while preserving exact no-op
identity and the measured safety/quality diagnostics. It did not improve blind
response quality. G2/G2b therefore stops without a holdout claim or production
rollout. A future experiment should introduce a separately justified actuator
family or evaluation hypothesis rather than tune these amplitudes further on
the observed development result.
