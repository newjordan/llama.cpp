# Treebeard J-Space G1 v5 hierarchical-routing preregistration

Date: 2026-07-15

Status: **frozen after bounded calibration-only method selection and before any
fresh-validation row is captured or indexed by the model.** V4 remains a failed
and consumed holdout. V5 changes the decision rule and uses a disjoint source
split; it does not retune the v4 deadband or reuse the two failed v4 rows.

## Decision rule

V5 preserves the byte-identical v4 prompt and seven raw next-token verbalizer
logits. It replaces seven independent operational probabilities with a fixed
hierarchy:

1. form a neutral contrast as the raw neutral logit minus the log-sum-exp of
   the six raw affect logits;
2. fit one positive-slope Platt calibrator on DailyDialog calibration rows,
   with neutral and aggregate-affect classes weighted equally;
3. fit six positive-slope one-vs-rest Platt calibrators on affect calibration
   rows only, then normalize their outputs with a six-way softmax;
4. assign the neutral probability from step 2 and distribute the remaining
   probability mass through the conditional affect softmax;
5. choose the most permissive neutral gate meeting a 0.10 calibration neutral
   false-positive ceiling and independently choose the maximum-coverage
   abstention threshold meeting 0.75 calibration precision.

The gate reports both neutral false-positive rate and affect recall. Affect
recall is a required gate so the method cannot pass by suppressing nearly all
affect activations.

## Bounded calibration-only selection

The already-open v4 calibration capture was used for one eight-fold,
source-and-label-stratified comparison of six fixed variants: neutral contrast
in {log-sum-exp, max-affect, raw-neutral} crossed with conditional affect logits
in {positive-slope Platt, raw}. The selected log-sum-exp plus Platt variant had:

- macro AUROC `0.895563`;
- EmpatheticDialogues macro AUROC `0.925741`;
- DailyDialog macro AUROC `0.868118`;
- macro ECE `0.067042`;
- top-1 accuracy `0.574519`;
- neutral false-positive rate `0.109375`;
- affect recall `0.644531`;
- abstention coverage/precision `0.478365/0.746231`.

No fresh-validation model output existed during this selection.

## Bound calibration inputs

- calibration manifest:
  `research/jspace-g1-v4/g1-v4-routing-calibration-manifest.json`;
- calibration manifest SHA-256:
  `0a2fccf25f6e1c43d59ba839a123b4a89253e30d98358426083289f07f0412ae`;
- calibration routing metadata SHA-256:
  `ee2d69e763d8f62f9c2c9865d271398e149a48f6a83cbbe91c78fe46865b32f3`;
- calibration routing logits SHA-256:
  `3acb615451b00019ee03710255c450485b6106169fd843e960d534472e66865a`;
- v5 evaluator SHA-256:
  `ecf9ba704842a74a3fb360cbd2d12a46d1efa216729ca849b339b440fd8f9b07`.

The v5 fit stops unless eight-fold out-of-fold calibration passes every gate:

- macro AUROC at least `0.85`;
- macro AUROC at least `0.82` independently in each source;
- macro equal-mass ECE at most `0.08`;
- top-1 accuracy at least `0.55`;
- neutral false-positive rate at most `0.15`;
- affect recall at least `0.60`;
- abstention coverage at least `0.40` and precision at least `0.72`;
- all seven fitted slopes remain positive.

## Fresh holdout

- freezer SHA-256:
  `36c711564a9087995dad82be52464fade8e0a5a4ce052a0e420ccf0315811441`;
- manifest:
  `research/jspace-g1-v5/g1-v5-routing-holdout-manifest.json`;
- manifest SHA-256:
  `9c94e76883359a7aa685185148ae17ee49f918b029ca41a0daec84fce73822c7`;
- rows: `649`;
- EmpatheticDialogues valid: 64 rows for each of six affect axes;
- DailyDialog validation: 32 sadness, 32 surprise, 32 joy, 1 disgust,
  8 fear, 32 anger, and 128 neutral rows.

The freezer globally normalizes and deduplicates text across all source splits,
then explicitly proves zero selected-text overlap with both consumed v4
manifests. DailyDialog validation contains only one eligible disgust row. Its
disgust metric is report-only; the DailyDialog pass metric covers the other six
axes, including neutral.

After a passing calibration artifact is frozen, v5 passes G1 only if the fresh
holdout passes all of these without refitting:

- overall seven-axis macro AUROC at least `0.82`;
- EmpatheticDialogues six-axis macro AUROC at least `0.82`;
- DailyDialog six-axis macro AUROC, excluding the single disgust positive, at
  least `0.75`;
- macro ECE at most `0.10` and top-1 accuracy at least `0.55`;
- neutral false-positive rate at most `0.15` across 128 neutral rows;
- affect recall at least `0.55`;
- abstention coverage at least `0.30` and retained precision at least `0.70`;
- EmpatheticDialogues anchor-echo and non-echo macro AUROC each at least `0.78`;
- mean shuffled-label macro AUROC between `0.40` and `0.60`.

## Bound on the claim

A v5 pass would validate a deployable state router and authorize G2 actuator
work. It would not itself establish an improvement in generated response
quality. No holdout capture may begin until the passing calibration artifact,
this preregistration, the manifest, and the exact evaluator are committed with
explicit human approval.
