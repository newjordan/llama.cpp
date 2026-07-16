# Treebeard G2 session handoff

Date: 2026-07-15

## Current state

G2/G2b completed as a negative bounded actuator-quality result. No candidate is
eligible, no holdout output was generated, and no production rollout is
authorized.

The complete result is in
`reports/archive/20260713-jspace-steering/treebeard-jspace-g2b-development-result-20260715.md`.

The development selector is
`/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g2/development-g2b/20260715-163546/g2-selected-policy.json`
with status `stop_no_eligible_candidate`.

Development routing used 98 train-only rows and produced 30 active rows across
all six frozen routed axes. All six candidate policies changed 16.7% to 33.3%
of active token sequences, but no candidate won a blind order-swapped judgment.
Five scored 0.5000 and match-route at 1/8 scored 0.4833. Each candidate failed
only the active-preference minimum of 0.52. All other development gates passed,
including exact inactive no-op token identity and zero candidate-only factual,
format, task-completion, refusal, repetition, or malformed flags.

Do not generate the G2 holdout. The preregistered stop rule prohibits it because
there is no selected development policy.

## Best configuration currently loaded

The running production configuration is the established scale-zero baseline,
not a J-Space actuator candidate:

- service: `turbo-statetree-rc4.service` on port 8093;
- build: `b9627-3fcf1c626`;
- alias:
  `turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3`;
- model: Qwen3.6-35B-A3B UD Q5_K_XL;
- B70 configuration: `-ngl 99 -ncmoe 0 --no-op-offload -c 262144 -np 12`
  with unified KV, flash attention, F16 KV, batch 8192, ubatch 1024, and 15
  CPU threads;
- environment: SYCL fusion enabled, graph disabled, MoE pipeline disabled, and
  grouped MoE-down disabled;
- expected server SHA-256:
  `211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff`.

Production is active, has `NRestarts=0`, and was verified after both guarded
G2 exits. Kernel audits found no Xe/DRM reset, hang, or fault signatures and no
OOM or kernel-panic signatures.

## Git state

Branch: `agent/treebeard-single-wavefront`

Committed G2/G2b work:

`feb24d5cd jspace: record bounded G2 actuator-quality stop`

The commit contains the G2/G2b manifests, preregistrations, guarded runners,
route/generation/judge/embedding/evaluation scripts, self-test, and result
reports. It deliberately excludes unrelated existing Q8 work.

The branch is one commit ahead of
`turbo-private/agent/treebeard-single-wavefront` at this handoff. Push target
is `turbo-private`, not upstream `origin`. Do not open a PR unless the human
explicitly requests one.

## Preserve these unrelated working-tree changes

- `ggml/src/ggml-sycl/mmvq.cpp`;
- `tests/test-backend-ops.cpp`;
- `scripts/treebeard-q8-hoist-profile-b70-guarded.sh`;
- `reports/treebeard-fresh-session-handoff-20260715.md`.

## Guardrails

- G1 v5 remains immutable.
- Do not tune the tested G2 scales further based on this development result.
- Any next actuator study needs a separate preregistered family or evaluation
  hypothesis.
- Do not operate GitHub Actions.
- Do not stop production except through an exact-service guarded runner with
  verified restoration.
