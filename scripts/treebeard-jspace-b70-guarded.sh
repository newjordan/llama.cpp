#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
VECTOR="$ROOT/research/jspace-20260713/treebeard-jspace/artifacts/phase0/qwen36_q5_phase0_joy_rawdual_l39.gguf"
MODEL_SHA=25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
EXPECTED_SERVER_SHA=211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff
CASE_FILTER=${TREEBEARD_JSPACE_CASE:-all}
ENABLE_FUSION=${TREEBEARD_JSPACE_ENABLE_FUSION:-1}
DISABLE_MOE_FUSIONS=${TREEBEARD_JSPACE_DISABLE_MOE_FUSIONS:-0}
DISABLE_MMID_GROUPED=${TREEBEARD_JSPACE_DISABLE_MMID_GROUPED:-0}
DISABLE_MMID_FUSED_BATCH=${TREEBEARD_JSPACE_DISABLE_MMID_FUSED_BATCH:-0}
DISABLE_SYCL_OPT=${TREEBEARD_JSPACE_DISABLE_SYCL_OPT:-0}
BATCH=${TREEBEARD_JSPACE_BATCH:-64}
UBATCH=${TREEBEARD_JSPACE_UBATCH:-$BATCH}
NO_KV_OFFLOAD=${TREEBEARD_JSPACE_NO_KV_OFFLOAD:-0}
NGL=${TREEBEARD_JSPACE_NGL:-99}
VERIFY_TOKENS=${TREEBEARD_JSPACE_VERIFY_TOKENS:-8}
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/treebeard-jspace-b70/$RUN_ID"
CASE_IDS=(code neutral factual json long)
PROMPTS=(
    'Complete this function: int add(int a, int b) {'
    'The operator reviews the system before deployment.'
    'The capital of France is'
    '{"task":"sum","arguments":[2,3],"result":'
    'Before the overnight maintenance window, the operator checks the queue, verifies every identifier against the signed manifest, reviews the capacity chart, confirms that no request is still active, records the current service revision, tests the rollback command, asks a second engineer to inspect the checklist, closes the stale dashboards, and writes a short handoff note. After those checks are complete, the operator starts the maintenance job, watches each stage report its status, compares the new measurements with the frozen baseline, and keeps the previous release ready until the final health probe succeeds.'
)

mkdir -p "$OUT/maintenance" "$OUT/candidate"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-jspace-b70/latest-run.txt"
for i in "${!CASE_IDS[@]}"; do
    printf '%s\t%s\n' "${CASE_IDS[$i]}" "${PROMPTS[$i]}" >> "$OUT/candidate/prompt-corpus.tsv"
done

restore_service() {
    local rc=${1:-0}
    local ok=1
    trap - EXIT INT TERM HUP
    set +e
    systemctl --user start "$SERVICE" || ok=0
    for _ in {1..240}; do
        if curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" \
            > "$OUT/maintenance/restore-health.json" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    systemctl --user show "$SERVICE" \
        -p ActiveState -p SubState -p MainPID -p NRestarts --no-pager \
        > "$OUT/maintenance/restore-unit.txt" || ok=0
    systemctl --user is-active --quiet "$SERVICE" || ok=0
    curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" \
        > "$OUT/maintenance/restore-props.json" || ok=0
    jq -e --arg build "$EXPECTED_BUILD" --arg alias "$EXPECTED_ALIAS" \
        '.build_info == $build and .model_alias == $alias and
         .total_slots == 12 and .default_generation_settings.n_ctx == 262144' \
        "$OUT/maintenance/restore-props.json" >/dev/null || ok=0
    local restored_pid restored_exe
    restored_pid=$(systemctl --user show "$SERVICE" -p MainPID --value)
    restored_exe=$(readlink -f "/proc/$restored_pid/exe" 2>/dev/null)
    if [[ -z "$restored_exe" ]]; then
        ok=0
    else
        sha256sum "$restored_exe" > "$OUT/maintenance/restore-server.sha256" || ok=0
        [[ $(awk '{print $1}' "$OUT/maintenance/restore-server.sha256") == "$EXPECTED_SERVER_SHA" ]] || ok=0
    fi
    curl -fsS --max-time 120 "http://127.0.0.1:$LIVE_PORT/completion" \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Return only the word healthy.","n_predict":8,"temperature":0,"top_k":1}' \
        > "$OUT/maintenance/restore-inference.json" || ok=0
    jq -e '.timings.predicted_n == 8' \
        "$OUT/maintenance/restore-inference.json" >/dev/null || ok=0
    curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/slots" \
        > "$OUT/maintenance/restore-slots.json" || ok=0
    curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/states" \
        > "$OUT/maintenance/restore-states.json" || ok=0
    jq -e '[.[] | select(.is_processing == true or .is_reserved == true)] | length == 0' \
        "$OUT/maintenance/restore-slots.json" >/dev/null || ok=0
    jq -e '(.families | length) == 0' \
        "$OUT/maintenance/restore-states.json" >/dev/null || ok=0
    date --iso-8601=seconds > "$OUT/maintenance/restore-date.txt"
    if (( ok == 0 )); then
        rc=1
        printf 'RESTORE_FAILED out=%s\n' "$OUT" >&2
    else
        printf 'RESTORE_OK out=%s\n' "$OUT"
    fi
    exit "$rc"
}

trap 'restore_service $?' EXIT INT TERM HUP

systemctl --user is-active --quiet "$SERVICE"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/health" \
    > "$OUT/maintenance/before-health.json"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" \
    > "$OUT/maintenance/before-props.json"
jq -e --arg build "$EXPECTED_BUILD" --arg alias "$EXPECTED_ALIAS" \
    '.build_info == $build and .model_alias == $alias and
     .total_slots == 12 and .default_generation_settings.n_ctx == 262144' \
    "$OUT/maintenance/before-props.json" >/dev/null
systemctl --user show "$SERVICE" \
    -p ActiveState -p SubState -p MainPID -p NRestarts --no-pager \
    > "$OUT/maintenance/before-unit.txt"
before_pid=$(systemctl --user show "$SERVICE" -p MainPID --value)
before_exe=$(readlink -f "/proc/$before_pid/exe")
sha256sum "$before_exe" > "$OUT/maintenance/before-server.sha256"
[[ $(awk '{print $1}' "$OUT/maintenance/before-server.sha256") == "$EXPECTED_SERVER_SHA" ]]
sha256sum "$BUILD/bin/llama-jspace-probe" "$BUILD/bin/libggml-sycl.so" \
    > "$OUT/candidate/runtime.sha256"
sha256sum "$MODEL" > "$OUT/candidate/model.sha256"
[[ $(awk '{print $1}' "$OUT/candidate/model.sha256") == "$MODEL_SHA" ]]
git -C "$WORKTREE" rev-parse HEAD > "$OUT/candidate/source-head.txt"
git -C "$WORKTREE" diff -- ggml/src/ggml-sycl/ggml-sycl.cpp tools/jspace-probe scripts/treebeard-jspace-b70-guarded.sh \
    > "$OUT/candidate/source.patch"
date --iso-8601=seconds > "$OUT/maintenance/start-date.txt"

systemctl --user stop "$SERVICE"
for _ in {1..60}; do
    systemctl --user is-active --quiet "$SERVICE" || break
    sleep 1
done
if systemctl --user is-active --quiet "$SERVICE"; then
    printf 'SERVICE_STOP_FAILED\n' >&2
    exit 1
fi
printf 'SERVICE_STOPPED_GUARDED\n'

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

mode_env=()
if [[ "$DISABLE_MOE_FUSIONS" == 1 ]]; then
    mode_env+=(GGML_SYCL_DISABLE_MOE_DOWN_REDUCE=1)
    mode_env+=(GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1)
fi
if [[ "$DISABLE_MMID_GROUPED" == 1 ]]; then
    mode_env+=(GGML_SYCL_DISABLE_MMID_GROUPED=1)
fi
if [[ "$DISABLE_MMID_FUSED_BATCH" == 1 ]]; then
    mode_env+=(GGML_SYCL_DISABLE_MMID_FUSED_BATCH=1)
fi
if [[ "$DISABLE_SYCL_OPT" == 1 ]]; then
    mode_env+=(GGML_SYCL_DISABLE_OPT=1)
fi

kv_offload_arg=-kvo
if [[ "$NO_KV_OFFLOAD" == 1 ]]; then
    kv_offload_arg=-nkvo
fi

invariance_args=()
if [[ "$VERIFY_TOKENS" != 0 ]]; then
    invariance_args+=(--verify-disabled-invariance "$VERIFY_TOKENS")
fi

for i in "${!CASE_IDS[@]}"; do
    case_id=${CASE_IDS[$i]}
    if [[ "$CASE_FILTER" != all && "$CASE_FILTER" != "$case_id" ]]; then
        continue
    fi
    env \
        GGML_SYCL_ENABLE_FUSION="$ENABLE_FUSION" \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
        GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
        "${mode_env[@]}" \
        taskset -c 0-10,12-15 "$BUILD/bin/llama-jspace-probe" \
        -m "$MODEL" -ngl "$NGL" -ncmoe 0 --no-op-offload "$kv_offload_arg" \
        -c 128 -b "$BATCH" -ub "$UBATCH" -t 15 \
        -p "${PROMPTS[$i]}" \
        --token-ids 13 \
        --probe-vector "joy=$VECTOR" \
        --probe-strengths=0 \
        --verified-model-sha256 "$MODEL_SHA" \
        --control-vector-layer-range 39 39 \
        "${invariance_args[@]}" \
        > "$OUT/jspace-disabled-$case_id.json" \
        2> "$OUT/candidate/probe-$case_id.log"

    if [[ "$VERIFY_TOKENS" == 0 ]]; then
        jq -e '
            .model.runner_verified_sha256 == "25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506"
        ' "$OUT/jspace-disabled-$case_id.json" >/dev/null
    else
        jq -e --argjson steps "$VERIFY_TOKENS" '
            .model.runner_verified_sha256 == "25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506" and
            .disabled_invariance.status == "pass" and
            .disabled_invariance.steps == $steps and
            .disabled_invariance.logits_bit_identical == true and
            .disabled_invariance.sampled_tokens_identical == true and
            .disabled_invariance.sequence_state_bit_identical == true
        ' "$OUT/jspace-disabled-$case_id.json" >/dev/null
    fi
done

if [[ "$VERIFY_TOKENS" == 0 ]]; then
    jq -s '{
        schema: "treebeard.jspace.fresh_baseline.corpus.v1",
        cases: length,
        prompt_token_counts: map(.probe.prompt_token_count),
        baselines: map(.baseline)
    }' "$OUT"/jspace-disabled-*.json > "$OUT/jspace-disabled-summary.json"
else
    jq -s '{
        schema: "treebeard.jspace.disabled_invariance.corpus.v1",
        cases: length,
        all_pass: all(.disabled_invariance.status == "pass"),
        total_logit_bytes_compared: (map(.disabled_invariance.logit_bytes_compared) | add),
        sequence_state_bytes_per_case: map(.disabled_invariance.sequence_state_bytes_compared),
        prompt_token_counts: map(.probe.prompt_token_count)
    }' "$OUT"/jspace-disabled-*.json > "$OUT/jspace-disabled-summary.json"
fi

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/kernel-journal.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true
if [[ -s "$OUT/kernel-signatures.txt" ]]; then
    printf 'HARDWARE_FAULT_SIGNATURES_DETECTED\n' >&2
    exit 1
fi

printf 'JSPACE_B70_COMPLETE out=%s\n' "$OUT"
