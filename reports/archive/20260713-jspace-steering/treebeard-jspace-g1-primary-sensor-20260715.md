# Treebeard J-Space G1 primary semantic sensor

Date: 2026-07-15

Status: the frozen primary held-out sensor slice passed. G1 remains open for
the predeclared lexical echo, negation, quotation, sarcasm, third-person,
mixed-mood, neutral-flat, and narrated-tedium controls. No actuator or
production steering claim follows from this result.

Subsequent result: the lexical-echo control failed its frozen gate, so G1 is
parked. See
[`treebeard-jspace-g1-control-stop-20260715.md`](treebeard-jspace-g1-control-stop-20260715.md).

## Dataset and evidence boundary

The source is the official agreement-filtered
[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
train/dev/test split, a human-annotated English Reddit corpus described by
[Demszky et al.](https://aclanthology.org/2020.acl-main.372/). The experiment
uses only the eight exact J-Space labels: sadness, surprise, joy, disgust,
fear, anger, curiosity, and neutral.

The deterministic freezer accepted only single-label rows, case-folded and
morphologically scrubbed all 102 anchor terms plus axis names, rejected
normalized duplicate text across splits, and selected rows by fixed SHA-256
rank. Each axis contributes 224 train, 32 calibration, and 32 untouched test
rows: 2,304 rows total.

- Frozen manifest SHA-256:
  `eee5eac3e2b0d9cd440a890af5504c5403ac52570c2db63e3bc19912c7e2e718`
- Upstream train SHA-256:
  `1c254a142be5c00e80d819b9ae1bbd36d94b2eeb8f4b1271846508d57e57d9c5`
- Upstream dev SHA-256:
  `575489c079c9de1097062a01738f998590d6b7ead66dd1c9fd1d2ba01fd8bc62`
- Upstream test SHA-256:
  `0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4`
- Label manifest SHA-256:
  `45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89`

The fitter was split into separate `fit` and `test` commands. Layer, ridge,
calibration, deadband, and abstention choices were frozen into an immutable
artifact before the test command first indexed any test row.

## Exact-runtime activation capture

Clean committed extraction ran on Intel B70 from commit
`cb8457f2ec62c61acfcb7e4ea2b2931fe3392dc4`. It captured the last raw-comment
token from paired DeltaNet/full-attention post-block residuals at layers
`2,3,10,11,18,19,26,27,34,35,38,39`. No chat template, control vector, special
token parsing, or cross-sample state reuse was permitted. Every row fit in one
microbatch and began after a full data-and-metadata memory clear.

- Model SHA-256:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`
- Extractor SHA-256:
  `d321484bfe561a41b370d668d03dbcb773560fda3a8666d9db6b1bc733ac031b`
- Metadata SHA-256:
  `a6211220f988cc020cf92141ad72fe937594b50509c6c820269dc9bce114cafb`
- Raw `[2304,12,2048]` F32 SHA-256:
  `ce1bdaa74498de02d4c918ccd3459953fe2a27aa8de70244ac1b46c90bb4d6bd`
- Rows/tokens/raw bytes: `2,304` / `35,835` / `226,492,416`
- Committed artifact directory:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-042455`

The guarded run recorded an empty source patch, zero kernel-fault signatures,
and restored the exact production build, model alias, 262,144-token context,
and 12 idle slots. Production completed a live inference and ended active with
`NRestarts=0` and no StateTree families.

## Frozen method and calibration gate

The sensor is deliberately small: a diagonal-variance-regularized one-vs-rest
mean discriminant in each 2,048-dimensional residual. The fixed ridge grid was
`0.01, 0.1, 1, 10, 100`; calibration macro AUROC selected the layer and ridge.
Each axis then received scalar Platt calibration. ECE is the macro of eight
one-vs-rest, ten-bin equal-mass estimates.

Layer 18 DeltaNet won the calibration-only selection at ridge 1:

- macro AUROC: `0.819685`;
- macro ECE: `0.034064`;
- neutral deadband calibration FPR: `3.125%` (one of 32, the conservative
  finite-sample realization of a 5% ceiling);
- abstention calibration coverage/precision: `26.172%` / `80.597%`.

The independently selected layer-19 full-attention companion reached macro
AUROC `0.816616`. Across 32 stratified bootstrap fits, DeltaNet and attention
median direction cosines were `0.900434` and `0.894508`; their fifth
percentiles were `0.828582` and `0.833100`. Median paired calibration-score
correlation was `0.991255`. Eight shuffled-training-label fits averaged macro
AUROC `0.513883`.

The frozen sensor artifact SHA-256 is
`61ff680d36517309cdab3081445651b79328499d770ced49aedff234749c59fd`.
The fitting protocol was committed as `007838172` before test access.

## One-shot held-out result

The untouched balanced test split passed every declared primary gate:

| Measurement | Gate | Result |
| --- | ---: | ---: |
| Macro one-vs-rest AUROC | at least 0.80 | **0.842791** |
| 95% bootstrap interval | reported | **0.819199–0.867411** |
| Macro one-vs-rest ECE | at most 0.05 | **0.044479** |
| Neutral deadband FPR | at most 0.10 | **0.0625** |
| Abstention coverage | at least 0.15 | **0.289062** |
| Retained precision | at least 0.70 | **0.756757** |
| Shuffled-test-label macro AUROC | near chance | **0.495121** |

Per-axis AUROC was sadness `0.838449`, surprise `0.764369`, joy `0.928292`,
disgust `0.787528`, fear `0.866211`, anger `0.858817`, curiosity `0.913644`,
and neutral `0.785017`. The aggregate passes, but surprise, disgust, and
neutral remain weaker coordinates and must be visible in downstream confidence
and abstention logic.

The exact test report SHA-256 is
`7b227c582e565fbdcce434a6082a847492985ac441b6d9986fa5829a79776529`.

## Evidence boundary and next falsification

This is the first evidence that a Qwen3.6 internal residual contains a compact,
held-out, anchor-free semantic signal for the declared label set. It is not a
Jacobian lens and does not show that writing along these directions changes
meaning, preserves tasks, or improves generated text. It also inherits the
domain and annotation limitations of English Reddit comments.

G1 remains open until the control matrix is captured and reported. Anchor
leave-out and split-anchor controls are structurally invariant because no
anchor is an input to this fitter, but lexical echo versus non-echo still needs
an explicit measured comparison. Negation, quotation, sarcasm, third-person,
mixed-mood, neutral-flat, and narrated-tedium cases must be frozen before
inspection. If those expose shortcut behavior, the sensor is parked despite
the primary aggregate pass.
