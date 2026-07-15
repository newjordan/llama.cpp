#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
HARNESS="$WORKTREE/scripts/treebeard-wavefront-b70.py"
REUSE_HARNESS="$WORKTREE/scripts/treebeard-moe-reuse-probe.py"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
PORT=${TREEBEARD_WAVEFRONT_PORT:-8098}
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
EXPECTED_SERVER_SHA=211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff
NGRAM_N=${TREEBEARD_NGRAM_N:-4}
REPEATS=${TREEBEARD_WAVEFRONT_REPEATS:-5}
CONCURRENCY_REPEATS=${TREEBEARD_WAVEFRONT_CONCURRENCY_REPEATS:-5}
N_PREDICT=${TREEBEARD_WAVEFRONT_N_PREDICT:-256}
DEPTHS=${TREEBEARD_WAVEFRONT_DEPTHS:-512,32768,131072,256000}
WIDTHS=${TREEBEARD_WAVEFRONT_WIDTHS:-0,1,2,4,8,12,24,48}
CASES=${TREEBEARD_WAVEFRONT_CASES:-structured-copy,code-edit,free-prose}
RUN_CONCURRENCY=${TREEBEARD_WAVEFRONT_RUN_CONCURRENCY:-1}
DISABLE_MOE_FUSIONS=${TREEBEARD_WAVEFRONT_DISABLE_MOE_FUSIONS:-0}
DISABLE_SYCL_OPT=${TREEBEARD_WAVEFRONT_DISABLE_SYCL_OPT:-0}
SIQ_PROF=${TREEBEARD_WAVEFRONT_SIQ_PROF:-0}
SERIAL_ANCHOR=${TREEBEARD_WAVEFRONT_SERIAL_ANCHOR:-1}
STRICT_PARITY=${TREEBEARD_WAVEFRONT_STRICT_PARITY:-1}
ARRIVAL_GAP_MS=${TREEBEARD_WAVEFRONT_ARRIVAL_GAP_MS:-0}
BATCH_SHAPE_PROF=${TREEBEARD_WAVEFRONT_BATCH_SHAPE_PROF:-0}
REUSE_PROBE=${TREEBEARD_WAVEFRONT_REUSE_PROBE:-0}
REUSE_N_PREDICT=${TREEBEARD_WAVEFRONT_REUSE_N_PREDICT:-64}
REUSE_SEQUENTIAL=${TREEBEARD_WAVEFRONT_REUSE_SEQUENTIAL:-0}
MOE_REUSE_PROFILE=${TREEBEARD_WAVEFRONT_MOE_REUSE_PROFILE:-$REUSE_PROBE}
NO_SPEC=${TREEBEARD_WAVEFRONT_NO_SPEC:-0}
STATE_IO_FUSION=${TREEBEARD_WAVEFRONT_STATE_IO_FUSION:-1}
STATE_IO_DEBUG=${TREEBEARD_WAVEFRONT_STATE_IO_DEBUG:-0}
STATE_IO_MODE=${TREEBEARD_WAVEFRONT_STATE_IO_MODE:-all}
ADMISSION_HOLD_MS=${TREEBEARD_WAVEFRONT_ADMISSION_HOLD_MS:-0}
CACHE_TYPE_K=${TREEBEARD_WAVEFRONT_CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${TREEBEARD_WAVEFRONT_CACHE_TYPE_V:-f16}
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/treebeard-single-wavefront-b70/$RUN_ID"
CANDIDATE_PID=

mkdir -p "$OUT/maintenance" "$OUT/candidate"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-single-wavefront-b70/latest-run.txt"

stop_candidate() {
    if [[ -n "${CANDIDATE_PID:-}" ]] && kill -0 "$CANDIDATE_PID" 2>/dev/null; then
        kill "$CANDIDATE_PID" 2>/dev/null || true
        for _ in {1..60}; do
            kill -0 "$CANDIDATE_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$CANDIDATE_PID" 2>/dev/null; then
            kill -9 "$CANDIDATE_PID" 2>/dev/null || true
        fi
        wait "$CANDIDATE_PID" 2>/dev/null || true
    fi
    CANDIDATE_PID=
}

restore_service() {
    local rc=${1:-0}
    local ok=1
    trap - EXIT INT TERM HUP
    set +e
    stop_candidate
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
sha256sum "$BUILD/bin/llama-server" "$BUILD/bin/libggml-sycl.so" \
    "$BUILD/bin/libllama-server-impl.so" > "$OUT/candidate/runtime.sha256"
sha256sum "$MODEL" > "$OUT/candidate/model.sha256"
date --iso-8601=seconds > "$OUT/maintenance/start-date.txt"

if ss -ltn "( sport = :$PORT )" | rg -q LISTEN; then
    printf 'BENCH_PORT_BUSY port=%s\n' "$PORT" >&2
    exit 1
fi

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
mode_env+=(GGML_SYCL_ENABLE_STATE_IO_FUSION="$STATE_IO_FUSION")
if [[ "$DISABLE_MOE_FUSIONS" == 1 ]]; then
    mode_env+=(GGML_SYCL_DISABLE_MOE_DOWN_REDUCE=1)
    mode_env+=(GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1)
fi
if [[ "$DISABLE_SYCL_OPT" == 1 ]]; then
    mode_env+=(GGML_SYCL_DISABLE_OPT=1)
fi
if [[ "$SIQ_PROF" == 1 ]]; then
    mode_env+=(SIQ_PROF=1)
fi
if [[ "$BATCH_SHAPE_PROF" == 1 ]]; then
    mode_env+=(TREEBEARD_BATCH_SHAPE_PROF=1)
fi
if [[ "$MOE_REUSE_PROFILE" == 1 ]]; then
    mode_env+=(GGML_SYCL_MOE_REUSE_PROFILE=1)
fi
if [[ "$STATE_IO_FUSION" == 1 ]]; then
    mode_env+=(GGML_SYCL_STATE_IO_MODE="$STATE_IO_MODE")
fi
if [[ "$STATE_IO_DEBUG" == 1 ]]; then
    mode_env+=(GGML_SYCL_STATE_IO_DEBUG=1)
fi
if (( ADMISSION_HOLD_MS > 0 )); then
    mode_env+=(TREEBEARD_SERVER_ADMISSION_HOLD_MS="$ADMISSION_HOLD_MS")
fi

spec_args=()
if [[ "$NO_SPEC" != 1 ]]; then
    spec_args=(
        --spec-type ngram-simple --spec-draft-n-max 48
        --spec-ngram-simple-size-n "$NGRAM_N"
        --spec-ngram-simple-size-m 48 --spec-ngram-simple-min-hits 1
    )
fi

env \
    GGML_SYCL_ENABLE_FUSION=1 \
    GGML_SYCL_DISABLE_GRAPH=1 \
    GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
    GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
    "${mode_env[@]}" \
    taskset -c 0-10,12-15 "$BUILD/bin/llama-server" \
    -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload \
    -c 262144 -np 12 -kvu -fa on -ctk "$CACHE_TYPE_K" -ctv "$CACHE_TYPE_V" \
    -b 8192 -ub 1024 -t 15 \
    --host 127.0.0.1 --port "$PORT" --jinja --metrics \
    "${spec_args[@]}" \
    -a "treebeard-single-wavefront-b70-ngram${NGRAM_N}" \
    > "$OUT/candidate/server.log" 2>&1 &
CANDIDATE_PID=$!
printf '%s\n' "$CANDIDATE_PID" > "$OUT/candidate/server.pid"

for _ in {1..300}; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" \
        > "$OUT/candidate/health.json" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$CANDIDATE_PID" 2>/dev/null; then
        tail -200 "$OUT/candidate/server.log" >&2 || true
        exit 1
    fi
    sleep 1
done
curl -fsS --max-time 5 "http://127.0.0.1:$PORT/props" > "$OUT/candidate/props.json"
jq -e '.total_slots == 12 and .default_generation_settings.n_ctx == 262144' \
    "$OUT/candidate/props.json" >/dev/null
printf 'CANDIDATE_READY pid=%s out=%s\n' "$CANDIDATE_PID" "$OUT"

concurrency_arg=--run-concurrency
if [[ "$RUN_CONCURRENCY" != 1 ]]; then
    concurrency_arg=--no-run-concurrency
fi

anchor_arg=--serial-anchor
if [[ "$SERIAL_ANCHOR" != 1 ]]; then
    anchor_arg=--no-serial-anchor
fi

parity_arg=--strict-parity
if [[ "$STRICT_PARITY" != 1 ]]; then
    parity_arg=--no-strict-parity
fi

if [[ "$REUSE_PROBE" == 1 ]]; then
    reuse_mode_args=()
    if [[ "$REUSE_SEQUENTIAL" == 1 ]]; then
        reuse_mode_args+=(--sequential)
    fi
    python3 "$REUSE_HARNESS" \
        --port "$PORT" --n-predict "$REUSE_N_PREDICT" --timeout 1800 \
        "${reuse_mode_args[@]}" \
        --out "$OUT/moe-reuse-probe.json" \
        2>&1 | tee "$OUT/moe-reuse-probe-console.log"
else
    python3 "$HARNESS" \
        --port "$PORT" --ctx 262144 --parallel 12 \
        --depths "$DEPTHS" --widths "$WIDTHS" --concurrency-widths "$WIDTHS" \
        --cases "$CASES" --repeats "$REPEATS" --concurrency-repeats "$CONCURRENCY_REPEATS" \
        --n-predict "$N_PREDICT" --timeout 1800 \
        --arrival-gap-ms "$ARRIVAL_GAP_MS" \
        "$concurrency_arg" "$anchor_arg" "$parity_arg" --reuse-case-prefix \
        --out "$OUT/wavefront-b70.json" \
        2>&1 | tee "$OUT/wavefront-b70-console.log"
fi

curl -fsS --max-time 5 "http://127.0.0.1:$PORT/slots" > "$OUT/candidate/slots-final.json"
curl -fsS --max-time 5 "http://127.0.0.1:$PORT/states" > "$OUT/candidate/states-final.json"
jq -e '[.[] | select(.is_processing == true or .is_reserved == true)] | length == 0' \
    "$OUT/candidate/slots-final.json" >/dev/null
jq -e '(.families | length) == 0' "$OUT/candidate/states-final.json" >/dev/null

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/kernel-journal.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true
if [[ -s "$OUT/kernel-signatures.txt" ]]; then
    printf 'HARDWARE_FAULT_SIGNATURES_DETECTED\n' >&2
    exit 1
fi

printf 'BENCH_COMPLETE out=%s\n' "$OUT"
