#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
MODEL_SHA=25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506
MANIFEST=${TREEBEARD_JSPACE_G1_MANIFEST:-$ROOT/research/jspace-g1-20260715/g1-goemotions-manifest.json}
MANIFEST_SHA=${TREEBEARD_JSPACE_G1_MANIFEST_SHA:-eee5eac3e2b0d9cd440a890af5504c5403ac52570c2db63e3bc19912c7e2e718}
LAYERS=${TREEBEARD_JSPACE_G1_LAYERS:-2,3,10,11,18,19,26,27,34,35,38,39}
POOLING=${TREEBEARD_JSPACE_G1_POOLING:-last}
EXPECTED_ROWS=${TREEBEARD_JSPACE_G1_EXPECTED_ROWS:-2304}
EXPECTED_SCHEMA=${TREEBEARD_JSPACE_G1_EXPECTED_SCHEMA:-treebeard.jspace.g1.dataset.v1}
PREFIX_NAME=${TREEBEARD_JSPACE_G1_PREFIX_NAME:-qwen36-g1-12layer}
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
EXPECTED_SERVER_SHA=211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/treebeard-jspace-g1-b70/$RUN_ID"
PREFIX="$OUT/$PREFIX_NAME"

case "$POOLING" in
    last)
        POOLING_WIDTH=1
        EXPECTED_STATUS=exact_runtime_last_token_residuals
        ;;
    last-mean)
        POOLING_WIDTH=2
        EXPECTED_STATUS=exact_runtime_last_mean_token_residuals
        ;;
    *)
        printf 'INVALID_POOLING pooling=%s\n' "$POOLING" >&2
        exit 1
        ;;
esac

mkdir -p "$OUT/maintenance" "$OUT/candidate"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-jspace-g1-b70/latest-run.txt"

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

sha256sum "$MODEL" > "$OUT/candidate/model.sha256"
sha256sum "$MANIFEST" > "$OUT/candidate/manifest.sha256"
[[ $(awk '{print $1}' "$OUT/candidate/model.sha256") == "$MODEL_SHA" ]]
[[ $(awk '{print $1}' "$OUT/candidate/manifest.sha256") == "$MANIFEST_SHA" ]]
sha256sum "$BUILD/bin/llama-jspace-sensor-extract" "$BUILD/bin/libggml-sycl.so" \
    > "$OUT/candidate/runtime.sha256"
git -C "$WORKTREE" rev-parse HEAD > "$OUT/candidate/source-head.txt"
git -C "$WORKTREE" diff -- \
    scripts/treebeard-jspace-g1-manifest.py scripts/treebeard-jspace-g1-controls.py \
    scripts/treebeard-jspace-g1-v2-evaluate.py \
    scripts/treebeard-jspace-g1-v3-meld-manifest.py \
    scripts/treebeard-jspace-g1-v3-evaluate.py \
    scripts/treebeard-jspace-g1-b70-guarded.sh \
    tools/jspace-probe/CMakeLists.txt tools/jspace-probe/README.md \
    tools/jspace-probe/jspace-sensor-extract.cpp \
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

env \
    GGML_SYCL_ENABLE_FUSION=1 \
    GGML_SYCL_DISABLE_GRAPH=1 \
    GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
    GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
    taskset -c 0-10,12-15 "$BUILD/bin/llama-jspace-sensor-extract" \
    -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload -kvo \
    -c 256 -b 256 -ub 256 -t 15 \
    --manifest "$MANIFEST" \
    --verified-manifest-sha256 "$MANIFEST_SHA" \
    --verified-model-sha256 "$MODEL_SHA" \
    --layers "$LAYERS" \
    --pooling "$POOLING" \
    --out-prefix "$PREFIX" \
    > "$OUT/candidate/extractor-result.json" \
    2> "$OUT/candidate/extractor.log"

jq -e --arg manifest_sha "$MANIFEST_SHA" --arg model_sha "$MODEL_SHA" \
    --arg dataset_schema "$EXPECTED_SCHEMA" --arg status "$EXPECTED_STATUS" \
    --argjson rows "$EXPECTED_ROWS" --argjson pooling_width "$POOLING_WIDTH" \
    '.schema == "treebeard.jspace.g1.activations.v1" and
     .status == $status and
     .dataset.runner_verified_sha256 == $manifest_sha and
     .dataset.schema == $dataset_schema and
     .model.runner_verified_sha256 == $model_sha and
     .raw.dtype == "little_endian_float32" and
     .raw.shape == (if $pooling_width == 1 then [$rows,12,2048]
                    else [$rows,12,$pooling_width,2048] end) and
     (.rows | length) == $rows and
     .capture.layers == [2,3,10,11,18,19,26,27,34,35,38,39] and
     .capture.chat_template == false and
     .capture.parse_special == false and
     .capture.one_decode_per_sample == true' \
    "$PREFIX.json" >/dev/null
expected_bytes=$(jq -r '.raw.bytes' "$PREFIX.json")
actual_bytes=$(stat -c '%s' "$PREFIX.f32")
[[ "$expected_bytes" == "$actual_bytes" ]]
sha256sum "$PREFIX.json" "$PREFIX.f32" > "$OUT/candidate/output.sha256"

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/kernel-journal.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true
if [[ -s "$OUT/kernel-signatures.txt" ]]; then
    printf 'HARDWARE_FAULT_SIGNATURES_DETECTED\n' >&2
    exit 1
fi
date --iso-8601=seconds > "$OUT/maintenance/complete-date.txt"
printf 'JSPACE_G1_B70_COMPLETE out=%s\n' "$OUT"
