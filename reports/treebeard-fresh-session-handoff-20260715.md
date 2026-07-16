# Treebeard fresh-session handoff (2026-07-15, post-G1-v5)

## Start here

The active quality result is now G1 v5, not the earlier v4 stop. A fresh,
hierarchical JSpace router passed every frozen calibration and untouched-holdout
gate and is promoted through G1. The next major research target is G2: prove
that conservative state-conditioned actuation improves generated responses
without breaking factual, format, or task-completion constraints.

The Q8 source-hoist experiment is also no longer an unfinished profile. Its
valid cold-model profile was a small positive, about +0.60%, so it is retained
default-off and deprioritized rather than removed.

Read these first:

1. `reports/archive/20260713-jspace-steering/treebeard-jspace-g1-v5-holdout-pass-20260715.md`
2. `reports/archive/20260713-jspace-steering/treebeard-jspace-g1-v5-preregistration-20260715.md`
3. `reports/archive/20260713-jspace-steering/treebeard-jspace-g1-v5-calibration-20260715.md`
4. `/home/frosty40/turbo/treebeard-work/results/treebeard-q8-hoist-profile-b70/20260715-133357/profile-decision.md`
5. `reports/treebeard-ragged-state-io-composition-20260715.md`

## Hard operating boundary

This is private hard-R&D work based on llama.cpp.

- Never create, edit, enable, disable, schedule, dispatch, rerun, or otherwise
  operate a GitHub Actions workflow unless the user explicitly authorizes that
  exact workflow action. GitHub workflows are the prohibited automation surface.
- Do not treat a push as permission to start or modify a workflow. Before a
  private push, a read-only Actions check is appropriate when needed to ensure
  the push cannot implicitly start automation. Do not change the Actions setting
  without explicit instruction.
- Pushes to the user private repository are standing-approved. The active private
  remote is `turbo-private`, repository `newjordan/turbo`.
- Do not push this branch to `origin`; `origin` is upstream
  `ggml-org/llama.cpp`. Upstream llama.cpp contribution and PR guardrails remain
  fully in force.
- Local builds, tests, guarded B70 runs, research reports, and explicitly
  approved local commits are allowed. Every assisted commit must contain
  `Assisted-by: Codex` and must not use `Co-authored-by`.
- The G1 v5 commits and B70 runs described below were explicitly approved.
- Never use GitHub as a benchmark or validation surface unless the user gives a
  new exact instruction.

At the last private push, `newjordan/turbo` reported repository Actions disabled
and all 12 workflows `disabled_manually`. No workflow action was taken. The
branch push completed without a workflow surface.

## Repository state

- Worktree:
  `/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce`
- Branch: `agent/treebeard-single-wavefront`
- HEAD: `823e95314b9bcb151d4f40e32de3954d375f84d6`
- Private tracking branch:
  `turbo-private/agent/treebeard-single-wavefront`
- Local and private tracking branch were synchronized after the G1 promotion
  push.

Newest commits:

- `823e95314`: promote G1 v5 hierarchical router;
- `936861b7f`: accept v5 routing manifests in the local extractor;
- `22b1ec0fa`: preregister and fit G1 v5;
- `d4bd83387`: record the earlier G1 v4 holdout stop.

All three new v5 commits contain `Assisted-by: Codex`.

The worktree is intentionally dirty only for the retained Q8 experiment and
this handoff:

- modified `ggml/src/ggml-sycl/mmvq.cpp`;
- modified `tests/test-backend-ops.cpp`;
- untracked `scripts/treebeard-q8-hoist-profile-b70-guarded.sh`;
- untracked `reports/treebeard-fresh-session-handoff-20260715.md`.

Do not fold those Q8 files into unrelated JSpace commits.

## Production and rig state

Production was restored after every guarded attempt and was rechecked after the
valid G1 v5 capture:

- unit: `turbo-statetree-rc4.service`;
- port: `8093`;
- state: active/running;
- `NRestarts=0`;
- executable SHA-256:
  `211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff`;
- build: `b9627-3fcf1c626`;
- alias:
  `turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3`;
- slots/context: 12 slots, 262144 tokens.

The post-capture inference probe produced all eight requested tokens. Use
`systemctl --user`. During guarded work, stop only the exact service or exact
benchmark process group. Never use `pkill -f`.

Build and data locations:

- build:
  `/home/frosty40/turbo/treebeard-work/build-treebeard-single-wavefront`;
- model:
  `/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`;
- model SHA-256:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`;
- Python used for v5:
  `/home/frosty40/nx2-venv/bin/python`;
- oneAPI:
  source `/opt/intel/oneapi/setvars.sh --force` before rebuilding SYCL targets.

## Main research result: JSpace G1 v5 passed

### Why v5 existed

G1 v4 had strong untouched generalization but failed its neutral operational
gate because two of eight DailyDialog neutral rows crossed the frozen deadband:

- overall macro AUROC `0.917102`;
- top-1 `0.754032`;
- ECE `0.046316`;
- neutral FPR `0.25`, above the `0.15` gate.

V4 was correctly parked without retuning its consumed holdout. V5 used a new
decision rule and fresh source validation splits.

### Frozen v5 rule

V5 preserves the exact v4 prompt and seven next-token verbalizer logits. It
replaces seven independent operational probabilities with a hierarchy:

1. neutral contrast = raw neutral logit minus log-sum-exp of the six affect
   logits;
2. one positive-slope, class-balanced Platt fit for neutral versus affect using
   DailyDialog calibration rows;
3. six positive-slope conditional affect Platt fits on affect rows;
4. six-way softmax for conditional affect mass;
5. frozen neutral gate plus independent abstention threshold.

Affect recall is a required metric, so low neutral FPR cannot be bought by
suppressing nearly every affect activation.

### Calibration pass

Eight-fold, source-and-label-stratified out-of-fold calibration:

| Metric | Gate | Result |
| --- | ---: | ---: |
| Macro AUROC | >= 0.85 | 0.897494 |
| Minimum source AUROC | >= 0.82 | 0.869879 |
| Macro ECE | <= 0.08 | 0.066733 |
| Top-1 | >= 0.55 | 0.570913 |
| Neutral FPR | <= 0.15 | 0.093750 |
| Affect recall | >= 0.60 | 0.651042 |
| Coverage | >= 0.40 | 0.483173 |
| Precision | >= 0.72 | 0.746269 |

Frozen artifact:

`research/jspace-g1-v5/g1-sensor-v5.npz`

Artifact SHA-256:

`6f50e2ea6c7cca8c6ee361d6611f0e764c1539a8a05d59f0c50f553fe0ace7d4`

### Fresh holdout construction

The holdout used 649 rows and had zero selected-text overlap with either
consumed v4 manifest:

- EmpatheticDialogues valid: 64 rows for each of six affect axes;
- DailyDialog validation: 32 sadness, 32 surprise, 32 joy, 1 disgust, 8 fear,
  32 anger, and 128 neutral.

DailyDialog validation contains only one eligible disgust row. That axis was
report-only as preregistered; the gated DailyDialog macro covers sadness,
surprise, joy, fear, anger, and neutral.

Manifest:

`research/jspace-g1-v5/g1-v5-routing-holdout-manifest.json`

Manifest SHA-256:

`9c94e76883359a7aa685185148ae17ee49f918b029ca41a0daec84fce73822c7`

The manifest regenerated byte-for-byte to the same hash before capture.

### Valid B70 holdout pass

Valid guarded artifact:

`/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-144532`

Bound inputs and outputs:

- source HEAD:
  `936861b7f03e8f4a8a00a63ed80c3a536fd962fe`;
- metadata SHA-256:
  `949f85562a6997c0bedea19fc2251f429c980abeecbd2bd6ef4e690a0cb4c205`;
- raw logits SHA-256:
  `d0297892c5af45a795d6b78d4110e23d0258c706f6c550dea48fb4c1696a95ec`;
- raw shape: `649 x 7`, little-endian float32;
- exact test report SHA-256:
  `9e668b18cbe8dbc27f89b59d642c25f1111f14f79902fc73aa1b44f5a5b59e20`.

Untouched results:

| Metric | Gate | Result |
| --- | ---: | ---: |
| Overall macro AUROC | >= 0.82 | 0.915539 |
| EmpatheticDialogues AUROC | >= 0.82 | 0.933415 |
| DailyDialog gated AUROC | >= 0.75 | 0.849792 |
| Macro ECE | <= 0.10 | 0.042931 |
| Top-1 | >= 0.55 | 0.661017 |
| Neutral FPR | <= 0.15 | 0.046875 |
| Affect recall | >= 0.55 | 0.727447 |
| Coverage | >= 0.30 | 0.539291 |
| Precision | >= 0.70 | 0.825714 |
| Minimum ED echo/non-echo AUROC | >= 0.78 | 0.906360 |
| Shuffle control | 0.40-0.60 | 0.500885 |

Operational routed-label accuracy was `0.642527`. DailyDialog neutral FPR was
six of 128. DailyDialog affect recall was `0.583942`. DailyDialog fear was the
weakest individual diagnostic axis at `0.696012` AUROC; do not hide this when
designing G2 confidence or abstention handling.

The valid capture source patch and kernel fault signature files were empty.
Production restored exactly.

Machine-readable result:

`research/jspace-g1-v5/g1-sensor-v5.test.json`

Promotion report:

`reports/archive/20260713-jspace-steering/treebeard-jspace-g1-v5-holdout-pass-20260715.md`

### Invalid pre-capture attempt

Artifact `20260715-144024` is invalid. The extractor rejected dataset schema v5
before model load because its allowlist stopped at v4. It produced no logits.
Production restored successfully. Commit `936861b7f` added v5 to the same frozen
seven-verbalizer path, the extractor self-test passed, and only then was the one
valid capture run.

## Immediate next research target: G2 actuator quality

G1 is a real unlock, but it validates sensing, not response quality. Do not
claim better generated responses from G1 metrics alone.

The next session should:

1. Audit the existing phase-0 actuator and state-prediction evidence under:
   `/home/frosty40/turbo/treebeard-work/research/jspace-20260713/treebeard-jspace/results/phase0_cpu_state_prediction/`.
2. Freeze a conservative G2 actuator family before looking at a fresh response
   holdout. Reuse the G1 v5 artifact exactly; do not refit it or tune against the
   v5 holdout.
3. Gate actuation on the frozen neutral/affect decision and abstention state.
   Treat DailyDialog fear confidence as a known weak diagnostic, not as a reason
   to retune G1.
4. Compare same-prompt, same-seed no-actuation control against actuation. Use a
   fresh development split for amplitude selection and a separate untouched
   response-quality holdout.
5. Require improvement on generated-response quality while preserving factual,
   format, protected-content, and task-completion constraints. Stop on any
   protected-constraint regression.
6. Keep G2 bounded. Do not open an unlimited layer, pooling, vector, or amplitude
   sweep. The old residual-layer search families are already falsified.
7. Use guarded local B70 runs, raw artifacts, exact source binding, zero kernel
   fault signatures, and exact production restoration.

A good G2 preregistration must state the actuator insertion point, direction,
amplitude candidates, G1 confidence/abstention policy, response set, quality
metric, protected constraints, and stop gates before the untouched response
holdout is generated.

## Q8 source-hoist result: retain, default-off, deprioritize

Dirty candidate files:

- `ggml/src/ggml-sycl/mmvq.cpp`;
- `tests/test-backend-ops.cpp`;
- `scripts/treebeard-q8-hoist-profile-b70-guarded.sh`.

Gate:

`GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=1`

The change hoists invariant reordered-Q8 loads across destination columns for
widths 2-8. It passed 11 of 11 selected CPU-reference backend cases and emitted
the expected activation trace.

Valid cold-model profile:

`/home/frosty40/turbo/treebeard-work/results/treebeard-q8-hoist-profile-b70/20260715-133357`

Results:

- aggregate decode `107.484266 -> 108.125612 tok/s`, `+0.596688%`;
- wall time `14290.464 -> 14205.700 ms`, `-0.593149%`;
- evals 51-250 `9106.9 -> 9050.9 ms`, `-0.614918%`;
- cumulative MUL_MAT `6406.1 -> 6369.2 ms`, `-0.576013%`;
- final named 50-eval MUL_MAT window `851.2 -> 842.8 ms`, `-0.986842%`;
- `linear_attn_out` `204.8 -> 204.3 ms`, `-0.244141%`;
- QKV `91.3 -> 90.6 ms`, `-0.766703%`;
- all six control/candidate token hashes matched;
- zero failed samples and zero kernel fault signatures;
- production restored exactly.

The +0.60 figure means +0.60 percent, not 0.60 on a 0-to-1 scale and not 60
percent. It is a small positive. Retain the default-off experiment and evidence;
do not remove it merely because it is below the major-play threshold. Do not
spend parity/A-B-A promotion time unless later composition evidence raises its
expected impact.

Invalid interrupted Q8 artifacts:

- `20260715-115242`;
- `20260715-123257`.

Do not use either as evidence.

## Accepted throughput baseline

The accepted ragged-attention plus state-I/O A/B/A artifact is:

`/home/frosty40/turbo/treebeard-work/results/treebeard-ragged-state-io-b70/20260715-104450-screen-r1-aba`

Results:

| Arm | Aggregate decode tok/s | Failed samples |
| --- | ---: | ---: |
| Control A | 117.007345 | 0 |
| Candidate | 127.099375 | 0 |
| Control B | 118.118014 | 0 |

- control midpoint `117.562679 tok/s`;
- candidate gain `8.112010%`;
- control drift `0.949230%`;
- all six branch token hashes matched;
- zero kernel fault signatures;
- production restored exactly.

This `117.56 -> 127.10 tok/s` figure is aggregate multi-request throughput under
the retained 12-slot, six-family StateTree workload. It is not single-stream
throughput.

Accepted commits in that chain:

- `81df9e2ff`: sequence-ragged StateTree KV attention;
- `789dfe8ec`: gate the ragged path;
- `827a4f007`: late activation for the internal SIQ profiler;
- `76befe8c8`: enable validated recurrent state-I/O fusion by default.

## Parked plays

Do not reopen without genuinely new evidence:

- G1 v4 threshold retuning on its consumed holdout;
- v1 through v3 residual layer, pooling, ridge, RBF, or polynomial sensor sweeps;
- quantized KV after the serial stop;
- native Q8 XMX after exact top-1 mismatch;
- the earlier width-12 Q8 geometry, XMX, local-split, and independent-split
  families.

The objective is not to accumulate speculative kernels or sensors. Preserve
exact behavior, falsify cheaply, and compound only measured wins. The current
priority is the first controlled generated-quality improvement through G2.
