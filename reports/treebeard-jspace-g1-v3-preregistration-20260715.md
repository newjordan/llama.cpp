# Treebeard J-Space G1 v3 preregistration

Date: 2026-07-15

Status: **frozen before source-calibration activation capture and before any
v3 test activation is captured or indexed.** G1 v1 and v2 remain failed and
parked. This is a source-disjoint seven-axis replacement attempt; curiosity is
explicitly unsupported and cannot be proxied or relabeled.

## Scope and source

V3 evaluates whether the already observed mean-pooled residual signal transfers
from Reddit comments to independently annotated conversational utterances. The
source is the official MELD repository at revision
`e8cedf27b5d2877e198332c957127e16eb214afe`.

MELD directly labels sadness, surprise, joy, disgust, fear, anger, and neutral.
It does not label curiosity. V3 therefore emits no curiosity coordinate and
cannot advance curiosity to pooling, actuation, or runtime work.

Exact source files:

- `train_sent_emo.csv` SHA-256:
  `d2fa2d6529cf03cac2989efec05c9b27d8fd2f4c8fc5974c7ae88aa537fa02db`;
- `dev_sent_emo.csv` SHA-256:
  `2e89c6f8aa182d6f62f8c6331aece905ac7273ca4999660bfb5213e1d0370c1c`;
- `test_sent_emo.csv` SHA-256:
  `8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888`.

The source repository declares GPL-3.0. The selected manifests and all result
artifacts remain in this private research repository.

## Frozen selection

All MELD rows are globally case-folded/whitespace-normalized and only texts
that occur exactly once across train, dev, and test remain eligible. Every text
from the consumed G1 v1 primary/control and v2 primary/control manifests is
also excluded. Selection uses a fixed SHA-256 rank within label, scope, and
whitespace-word length band:

- short: 1–5 words;
- medium: 6–12 words;
- long: 13 or more words.

Source calibration uses 32 MELD-train rows per label per length band: 672 rows
total. The untouched MELD-test gate contains:

- 10 anchor-free rows per label per length band: 210 primary rows;
- four disjoint natural lexical-echo rows per label: 28 control rows.

Artifacts:

- calibration manifest:
  `research/jspace-g1-v3/g1-v3-meld-calibration-manifest.json`;
- calibration manifest SHA-256:
  `b66dd8fa848f4481a7db045c317a440c32c776fc56ef9ba599f60df618f580a6`;
- test manifest:
  `research/jspace-g1-v3/g1-v3-meld-test-manifest.json`;
- test manifest SHA-256:
  `2fdf45ab0af896ca6ea90bdab4daa4b18702228af04f242e19265bd034c13b91`.

The test manifest may be committed and audited now, but it is not passed to
the fit command and must not be captured until the fitted artifact and its
calibration result are committed.

## Frozen representation and fit

- Retain the arithmetic mean over every literal prompt-token `l_out` residual.
- Use layer 35 and ridge `1.0`; these are inherited unchanged from v2.
- Fit seven diagonal-variance one-vs-rest mean-discriminant directions on only
  the 1,568 non-curiosity v1 GoEmotions training rows.
- Standardize direction scores only from those training rows.
- Fit one source calibrator per axis on the 672 MELD-train calibration rows.
  Each fixed logistic design contains the standardized sensor score,
  standardized `log1p` exact Qwen token count, and an intercept. The token-count
  coefficient has fixed L2 penalty `0.1`; there is no penalty sweep.
- Choose the deadband from the source-calibration neutral rows at at most 5%
  realized false positives.
- Choose the maximum-coverage abstention threshold whose source-calibration
  top-1 precision is at least `0.75`.

The evaluator is `scripts/treebeard-jspace-g1-v3-evaluate.py`. Its `fit`
command has no test-manifest argument.

## Source-calibration gate

The fit artifact may be frozen only if all of these pass on the 672 declared
source-calibration rows:

- seven-axis macro one-vs-rest AUROC at least `0.80`;
- macro ten-bin equal-mass ECE at most `0.05`;
- neutral deadband FPR at most `0.05`;
- abstention coverage at least `0.15` and retained precision at least `0.75`;
- macro AUROC at least `0.70` independently in short, medium, and long bands;
- mean shuffled-label macro AUROC between `0.40` and `0.60`.

Failure stops v3 before MELD-test capture.

## Untouched test gate

After a passing fit artifact is committed, v3 passes G1 only if all of these
hold without threshold changes:

- 210-row anchor-free primary macro AUROC at least `0.80`;
- primary macro equal-mass ECE at most `0.05`;
- primary neutral deadband FPR at most `0.10`;
- primary abstention coverage at least `0.15` and retained precision at least
  `0.70`;
- primary macro AUROC at least `0.70` separately in every length band;
- 28-row balanced lexical-echo macro AUROC at least `0.70`;
- mean shuffled-label primary macro AUROC between `0.40` and `0.60`.

The 1,000-replicate stratified bootstrap interval, top-1 accuracy, per-axis
metrics, and post-calibration score/token-count correlations are report-only.
They cannot trigger a v3 feature or threshold change.

## Advancement boundary

A full pass admits an offline seven-axis G1 sensor candidate only. It does not
admit curiosity, prove causal steering, or authorize production feedback. G2
may then compare frozen multiscale pooling schedules for these seven axes.
Any failed calibration, primary, length-band, lexical, or shuffle gate parks v3
without opening G2.
