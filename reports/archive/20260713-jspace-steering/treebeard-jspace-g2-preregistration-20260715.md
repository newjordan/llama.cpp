# Treebeard J-Space G2 generated-quality preregistration

Date: 2026-07-15

Status: frozen after the phase-0/G0/G1 audit and before any G2 routing capture,
development response, judge response, or holdout response is generated.

## Claim and stop boundary

G2 asks one bounded question: can a small, request-scoped final-head actuator,
enabled only by the frozen G1 v5 affect decision and abstention state, improve
generated conversational responses against a same-prompt, same-seed scale-zero
control without regressing protected tasks or runtime safety?

A pass is evidence only for the selected policy, amplitudes, response corpus,
model, and runtime recorded here. It is not evidence for a hidden-layer
Jacobian lens, closed-loop feedback, all affect policies, or production rollout.
G1 v5 is an immutable input. No G1 probability, Platt fit, neutral threshold,
abstention threshold, label, or holdout row may be refit or retuned in G2.

The experiment stops before the untouched holdout if no development candidate
passes its entry gates. It stops on any protected-task output mismatch,
candidate-only factual/format/task failure, kernel fault signature, or failed
production restoration.

## Evidence audit and selected insertion contract

The phase-0 CPU evidence established a real local final-head geometry but did
not pass its full aggregate gate. All 168 prompt-axis cells had the correct
sign and bidirectional order, minimum intended-response R2 was `0.997355`, and
median/max independent off-axis ratios were `2.280%/5.814%`. Three cells
narrowly exceeded the frozen 10% even/odd ceiling; the maximum was `10.3504%`.
That failure is retained and is the reason G2 uses amplitudes strictly below
the smallest measured phase-0 nonzero dose.

The accepted G0 server contract already supplies the required runtime path:

- raw regularized-dual control vector;
- decoder block 39 zero-based, after the final block residual and before output
  RMSNorm;
- one request-scoped finite `jspace_control_scale`;
- fail-closed zero scale on omission after sequence mode begins;
- tested reset, cancellation, fork, commit, slot reuse, and snapshot ownership;
- exact disabled-path identity.

No new C++ controller or kernel is admitted for G2. Although the final-block
addition is evaluated on prompt positions too, it is after the final block and
does not feed a later layer or stored KV/recurrent state. The generation-time
effect is a per-token output-logit intervention.

## Frozen G1 gating

For each response input, capture the byte-identical seven verbalizer logits and
apply `research/jspace-g1-v5/g1-sensor-v5.npz` with the existing v5 evaluator.

- neutral threshold: `0.2532228901819629`;
- abstention threshold: `0.7519634658067124`;
- active dialogue condition: G1 neutral probability is below the neutral
  threshold and maximum seven-way probability is at least the abstention
  threshold;
- routed affect axis: argmax across sadness, surprise, joy, disgust, fear, and
  anger;
- neutral or abstained rows: scale zero;
- every `protected_task` row: scale zero regardless of the G1 result.

DailyDialog fear remains a known weak diagnostic axis. It receives no new
threshold or special-case retuning. The one eligible held-out DailyDialog fear
response pair is report-only by source-axis; aggregate gates still include it.

## Bounded actuator family

Only two positive-direction policies enter development:

1. `engage_curiosity`: apply the positive raw regularized-dual curiosity vector
   to every active dialogue row;
2. `match_route`: apply the positive raw regularized-dual vector matching the
   frozen routed affect axis.

Only three raw-dual scales enter development:

| Scale ID | Raw `jspace_control_scale` | Fraction of phase-0 inner dose |
| --- | ---: | ---: |
| `eighth` | `0.008641079027104324` | 1/8 |
| `quarter` | `0.017282158054208648` | 1/4 |
| `half` | `0.034564316108417296` | 1/2 |

The phase-0 inner nonzero dose was `0.06912863221683459`. G2 does not test it
or any larger amplitude. It does not search a layer, band, pooling rule,
negative direction, mixed vector, per-axis amplitude, or additional actuator
family.

The scale-zero control runs through the same binary, model, loaded-vector path,
prompt, sampler, and seed. Only `jspace_control_scale` differs on active rows.
For `match_route`, controls are therefore partitioned by routed axis and
generated while the same axis vector as their candidate is loaded. Inactive
rows use the curiosity loaded-vector path at scale zero in both arms. The
evaluator rejects a pair whose recorded vector axis or vector SHA-256 differs
between arms.

## Frozen response sets

The freezer excludes the entire source conversation for every row consumed by
the G1 v4 calibration, G1 v4 test, or G1 v5 holdout manifest. It also rejects
duplicate inputs, exact input/reference echoes, inputs outside 3-64 words, and
references outside 2-80 words. Selection is the lowest fixed-seed SHA-256 rank
within source and label.

Development contains 50 rows:

- EmpatheticDialogues train: 4 per affect axis, 24 total;
- DailyDialog train: 3 per affect axis plus 8 neutral, 26 total.

Holdout contains 100 rows:

- EmpatheticDialogues test: 8 per affect axis, 48 total;
- DailyDialog test: 4 per affect axis except one eligible fear row, plus 19
  neutral, 40 total;
- 12 frozen protected factual, math, JSON, exact-format, extraction,
  instruction, and safety tasks.

Development and holdout have zero selected-input overlap and use disjoint
source splits. The holdout manifest may be inspected for construction audit,
but no holdout routing logits, responses, embeddings, or judge outputs may be
generated until one development policy is frozen.

## Generation protocol

Use the exact Qwen3.6 Q5 model with the accepted local SYCL runtime and no draft
model. Requests use the server chat endpoint with this fixed system message:

`Respond naturally and directly to the user. Be helpful, concise, and honest.`

Sampler and length are frozen:

- development seed: `1709`;
- holdout seeds: `1709`, `2718`, and `31415`;
- temperature `0.7`, top-k `40`, top-p `0.9`, min-p `0.05`;
- repeat penalty `1.05`;
- maximum 96 generated tokens;
- model end-of-generation stopping, no custom stop string;
- no prompt-cache reuse across arms.

Every control/candidate pair uses the same prompt and seed. Response text,
token IDs, finish reason, timing, route, G1 probabilities, actuation decision,
vector hash, scale, source HEAD, source patch, model hash, and runtime hashes
are retained as raw artifacts.

## Development selection

All six policy/scale combinations are evaluated once on the 50-row development
set. Quality is judged blind to arm identity with the frozen Qwen3-8B Q8 model.
Each active pair is judged twice with A/B order reversed. A preference counts
only when the two orientations agree after undoing the swap; disagreement is a
tie. The judge sees the user input, human reference, and two candidate
responses. It scores contextual appropriateness, empathy/engagement,
helpfulness, factual restraint, instruction completion, formatting, repetition,
and refusal pathology in a fixed JSON schema.

Nomic Embed Text v1.5 Q8 provides a secondary reference-similarity diagnostic.
It is never used to change the judge prompt or response set.

A development candidate is eligible only if:

- at least 20 dialogue rows are active under frozen G1 gating;
- at least 10% of active candidate token sequences differ from control;
- blind preference score `(wins + 0.5 * ties) / comparisons` is at least
  `0.52` on active pairs;
- neither source has a preference score below `0.45`;
- mean candidate-minus-control reference cosine is at least `-0.005`;
- candidate anchor-word emission rate rises by no more than 2 percentage
  points;
- repeated 4-gram and malformed-output rates each rise by no more than 2
  percentage points;
- mean candidate/control length ratio is in `[0.75, 1.33]`;
- no candidate-only factual, format, task-completion, or refusal-pathology flag
  is emitted by an order-consistent judge result.
- every inactive scale-zero candidate has token IDs exactly identical to its
  control.

Choose the eligible candidate with the highest blind preference score. Exact
ties are broken by lower amplitude, then by `engage_curiosity`. Freeze the
selected policy, scale, development metrics, and all input/output hashes before
any holdout route is captured. If no candidate is eligible, G2 stops.

## Untouched holdout gates

Generate only the frozen selected candidate and its scale-zero control for all
three seeds. Primary quality is the order-consistent blind preference score on
active dialogue pairs. G2 passes only if every gate below passes:

- active-pair preference score at least `0.55`;
- two-sided 95% Wilson lower bound for the active-pair score, treating a tie as
  half a success, strictly above `0.50`;
- each source preference score at least `0.47`;
- no routed-axis preference score below `0.40` and no source-label preference
  score below `0.40`; DailyDialog fear is excluded only from its one-row
  source-label gate and remains in pooled label, source, routed-axis, and
  aggregate results;
- mean candidate-minus-control reference cosine at least `-0.005`;
- neutral-dialogue candidate preference score at least `0.50` and active rate
  at most `0.15`;
- candidate anchor-word emission rises by no more than 1 percentage point;
- repeated 4-gram, malformed-output, and unsolicited-refusal rates each rise by
  no more than 1 percentage point;
- mean candidate/control length ratio in `[0.80, 1.25]`;
- zero order-consistent candidate-only factual, format, task-completion, or
  refusal-pathology flags;
- all protected rows use scale zero and have token IDs exactly identical to
  control for every seed;
- every other inactive scale-zero candidate has token IDs exactly identical to
  its control;
- at least 10 of 12 protected controls pass their frozen absolute validator,
  and the candidate has exactly the same validator outcomes;
- zero failed requests and zero kernel reset, hang, fault, OOM, or panic
  signatures;
- exact production service identity, executable digest, build, alias, slots,
  context, restart count, idle StateTree state, and post-restore inference.

All metrics, including failures, are reported. No row, seed, axis, source,
judge disagreement, truncated response, or malformed judge record may be
dropped after generation.

## Bound artifacts

- base model SHA-256:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`;
- frozen G1 v5 artifact SHA-256:
  `6f50e2ea6c7cca8c6ee361d6611f0e764c1539a8a05d59f0c50f553fe0ace7d4`;
- frozen G1 v5 evaluator SHA-256:
  `ecf9ba704842a74a3fb360cbd2d12a46d1efa216729ca849b339b440fd8f9b07`;
- phase-0 build report SHA-256:
  `0b1f161edec3ed0916045c95da233e70466e233938c184d903b6853d257bb7a8`;
- G2 freezer SHA-256:
  `1bb21cfead3aae015997ccfe55b6d31ed10cf3e976e31873fd84b0c5988ab30d`;
- G2 frozen-router wrapper SHA-256:
  `8610029f1261275084fafb50080e9151b60e51836b5275ea4c6ef6959005b401`;
- G2 response generator SHA-256:
  `adf3278dc4a715dfc5678a4e71cdb82bf3da0757b96a159ae6ce29d17e62ee6b`;
- G2 blind judge client SHA-256:
  `c8ca52bc365f81041150e72ba5f6908d100d21966bfb188d55e9eca68fa0e79d`;
- G2 embedding client SHA-256:
  `869a9e758e321ebb52fc613b3db7c1e46bc5668e985b77db16411a55b00471c4`;
- G2 evaluator/selector SHA-256:
  `9d9ab88f1a6615b043e1d5d079ae8e9fc8b2b9de0948fd17c3f31a64cb676337`;
- model-independent G2 pipeline self-test SHA-256:
  `3d45f6804cad20496c071ee1fd64b3f47562389203581a251bea4bd995e3de0c`;
- Q5 anchor manifest SHA-256:
  `cab229a60bbcf1cd641070227bd4fd4c5bc8345605c36afe5c81850ff15ebb71`;
- development response manifest SHA-256:
  `a39b8d041e51133a359298b5e99bf42d014aee6eb6ef5884b5056cb5dc355d4d`;
- development routing manifest SHA-256:
  `32953d19c9716568a3b55b79e604df3a8648c0284173943101e429edd46f07e4`;
- holdout response manifest SHA-256:
  `24191deb47cc3ec897e2833a98ca333d1369227efcbbb3355a6f9015326cf9cd`;
- holdout routing manifest SHA-256:
  `d58fcb5b3404649d6f2d0cb0d9c405fcbc30dac617f2243e640a8c32663f53ae`;
- Qwen3-8B Q8 judge model SHA-256:
  `0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e`;
- Nomic Embed Text v1.5 Q8 SHA-256:
  `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`.

Raw-dual vector SHA-256 values:

| Axis | SHA-256 |
| --- | --- |
| sadness | `2d7f1d254bdbcbcea1350d028e2d5b4f5aabb381b1887f041d0bc1f6ca9fa8c7` |
| surprise | `191534954903c474e80512b877b5ec2ef26ed8304416247cddaaab18d8d57830` |
| joy | `a9af4c6b8dfb189e1a9f93917b2ea2d1154a1f6fb7987da2459a176c5a1cf8c0` |
| disgust | `6e584cc9450b1734f18bf8be10595afa27ed754cd2f22d2d2d47f42d2af9cf28` |
| fear | `0b577aca9899a62d49981f10ccd148a48073bc6b92c587aa3ca0aaf2fdf39ab5` |
| anger | `338aaf760651ed491408d327b1b7879d06d0e5a0f9090dfa9087f18659f02108` |
| curiosity | `9cd787c2bbde921a516afb7ba885db267cd575a021abba855da49804a096f0f9` |

The guarded runner must rehash every bound input and record the exact server,
extractor, embedding runtime, SYCL runtime, source HEAD, and source patch before
stopping production. A mismatch is a pre-load stop.
