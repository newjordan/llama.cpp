# Treebeard J-Space G1 v4 instruction-routing preregistration

Date: 2026-07-15

Status: **frozen before v4 source-calibration capture and before any v4 test
row is captured or indexed.** V1 through v3 remain failed and parked. V4
changes both the state representation and the supervised objective; it is not
another layer, pooling, ridge, or threshold retry.

## Representation

V4 appends one fixed classification suffix to each literal text and reads the
raw next-token logits for seven ordered, single-token verbalizers:

`sadness, surprise, joy, disgust, fear, anger, neutral`

The exact token IDs are `49166,12395,15420,64797,8415,18654,20002`. The fixed
prompt is recorded byte-for-byte in both manifests and every exact-runtime
capture. There is no chat template, generated classification token, control
vector, or residual layer selection. Every row starts after a full model-memory
clear and fits in one decode.

This is an instruction-probed output-head routing state, not a linear readout
of a selected hidden residual. The only fitted parameters are seven independent
positive-slope Platt calibrators. A non-positive slope fails closed rather than
silently reversing an axis.

## Frozen sources and selection

Calibration uses deterministic balanced subsets from EmpatheticDialogues train
and DailyDialog train. The untouched gate uses only their declared test splits.
Texts are normalized and deduplicated across both sources and every split,
limited to 64 whitespace words, and selected by a fixed SHA-256 rank within
source and mapped axis. Natural anchor echoes are retained and marked for a
separate holdout stratum.

- calibration manifest:
  `research/jspace-g1-v4/g1-v4-routing-calibration-manifest.json`;
- calibration SHA-256:
  `0a2fccf25f6e1c43d59ba839a123b4a89253e30d98358426083289f07f0412ae`;
- calibration rows: 832 (64 per source/axis, with no EmpatheticDialogues
  neutral label);
- test manifest:
  `research/jspace-g1-v4/g1-v4-routing-test-manifest.json`;
- test SHA-256:
  `ffcf0ac11e343a9674984a2e497ed1a91cbb5fb3e4f3babbc7407d8f4c19165b`;
- test rows: 248 (32 per EmpatheticDialogues affect axis and eight per
  DailyDialog axis).

Source file SHA-256 values are embedded in both manifests. The freezer makes
no redistribution-rights claim; these local sources are used only for private
research.

## Calibration gate

V4 stops before test capture unless the 832-row calibration result passes all
of the following:

- seven-axis macro one-vs-rest AUROC at least `0.80`;
- macro AUROC at least `0.75` independently in each source;
- macro equal-mass ECE at most `0.08`;
- top-1 accuracy at least `0.55`;
- neutral deadband FPR at most `0.10`;
- abstention coverage at least `0.20` at retained precision at least `0.75`;
- every fitted verbalizer slope remains positive.

The deadband is the most permissive calibration threshold meeting its FPR
ceiling. The abstention threshold is the maximum-coverage threshold meeting the
precision floor. Neither may change after calibration.

## Untouched test gate

After a passing artifact is committed, v4 passes G1 only if all of these hold
without refitting:

- overall seven-axis macro AUROC at least `0.80`;
- EmpatheticDialogues six-axis macro AUROC at least `0.80`;
- DailyDialog seven-axis macro AUROC at least `0.75`;
- macro ECE at most `0.10` and top-1 accuracy at least `0.55`;
- neutral deadband FPR at most `0.15`;
- abstention coverage at least `0.15` and retained precision at least `0.70`;
- EmpatheticDialogues anchor-echo and non-echo macro AUROC each at least `0.75`;
- mean shuffled-label macro AUROC between `0.40` and `0.60`.

## Pre-capture diagnostics and boundary

On the already consumed MELD calibration capture, a coupled multi-class ridge
over joint last-token and mean-token residuals reached only `0.768725` macro
AUROC. RBF and polynomial kernels did not improve that bound. Those diagnostics
do not admit a residual v4.

A one-row guarded B70 smoke of the new representation selected the correct
`fear` verbalizer with a 1.235468-logit margin over the runner-up. The artifact
is `results/treebeard-jspace-b70/20260715-111250`; production restored with
`NRestarts=0` and no kernel fault signatures. This smoke motivated the frozen
protocol but is not included in either v4 source and is not a pass claim.

The extractor and evaluator self-tests pass, and regenerating both manifests
byte-for-byte reproduces their hashes. No v4 test logit has been captured or
read. Failure of the source-calibration gate parks this representation before
test access, Fibonacci pooling, actuation, feedback, or production routing.
