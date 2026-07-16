#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
HARNESS="$WORKTREE/scripts/turbo-statetree-bench.py"
PYTHON=/home/frosty40/turbo/.venv-server-tests/bin/python
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
BENCH_PORT=8098
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
EXPECTED_SERVER_SHA=211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/treebeard-q8-hoist-profile-b70/$RUN_ID"
BENCH_PID=

mkdir -p "$OUT/maintenance"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-q8-hoist-profile-b70/latest-run.txt"

stop_benchmark() {
    if [[ -n "${BENCH_PID:-}" ]] && kill -0 "$BENCH_PID" 2>/dev/null; then
        kill -- "-$BENCH_PID" 2>/dev/null || true
        for _ in {1..60}; do
            kill -0 "$BENCH_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$BENCH_PID" 2>/dev/null; then
            kill -9 -- "-$BENCH_PID" 2>/dev/null || true
        fi
        wait "$BENCH_PID" 2>/dev/null || true
    fi
    BENCH_PID=
}

restore_service() {
    local rc=${1:-0}
    local ok=1
    trap - EXIT INT TERM HUP
    set +e
    stop_benchmark
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

run_leg() {
    local label=$1
    local hoist=$2
    local leg_out="$OUT/$label"
    local trigger="$leg_out/siq-profile.trigger"
    local console="$leg_out/console.log"

    mkdir -p "$leg_out/run"
    rm -f "$trigger"
    setsid env \
        GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST="$hoist" \
        GGML_SYCL_ENABLE_STATE_IO_FUSION=1 \
        GGML_SYCL_STATE_IO_MODE=all \
        GGML_SYCL_DISABLE_GRAPH=1 \
        SIQ_PROF=1 \
        SIQ_PROF_TRIGGER_FILE="$trigger" \
        "$PYTHON" "$HARNESS" run \
        --bin "$BUILD/bin/llama-server" \
        --model "$MODEL" \
        --label "$label" \
        --commit "$(git -C "$WORKTREE" rev-parse HEAD)" \
        --out-dir "$leg_out/run" \
        --port "$BENCH_PORT" \
        --ctx 262144 \
        --parallel 12 \
        --fanout 5 \
        --layout fragmented \
        --fragment-fill-tokens 8192 \
        --persistent-fragmentation \
        --prefix-tokens 32768 \
        --branch-suffix-tokens 8 \
        --branch-tokens 256 \
        --repeats 1 \
        --modes manual \
        --seed 1709 \
        --batch 8192 \
        --ubatch 1024 \
        --threads 15 \
        --ngl 99 \
        --ncmoe 0 \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --flash-attn on \
        --request-timeout 1800 \
        --startup-timeout 300 \
        --source-oneapi \
        --tree-ragged \
        --kv-page-probe \
        --no-require-commit \
        --fail-on-error \
        > "$console" 2>&1 &
    BENCH_PID=$!

    for _ in {1..900}; do
        if rg -q '^repeat=0 prefix=32768 mode=manual$' "$console" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$BENCH_PID" 2>/dev/null; then
            wait "$BENCH_PID"
        fi
        sleep 1
    done
    rg -q '^repeat=0 prefix=32768 mode=manual$' "$console"
    touch "$trigger"
    printf 'PROFILE_TRIGGERED leg=%s pid=%s\n' "$label" "$BENCH_PID"

    wait "$BENCH_PID"
    BENCH_PID=
    rg '^\[treebeard-q8-hoist\]|^\[siq-prof' \
        "$leg_out/run/$label.server.log" > "$leg_out/profile-summary.txt"
    printf 'LEG_COMPLETE leg=%s\n' "$label"
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
    "$BUILD/bin/libllama-server-impl.so" > "$OUT/maintenance/candidate-runtime.sha256"
sha256sum "$MODEL" > "$OUT/maintenance/model.sha256"
git -C "$WORKTREE" status --short > "$OUT/maintenance/source-status.txt"
git -C "$WORKTREE" diff --binary > "$OUT/maintenance/source.patch"
date --iso-8601=seconds > "$OUT/maintenance/start-date.txt"

if ss -ltn "( sport = :$BENCH_PORT )" | rg -q LISTEN; then
    printf 'BENCH_PORT_BUSY port=%s\n' "$BENCH_PORT" >&2
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
printf 'SERVICE_STOPPED_GUARDED out=%s\n' "$OUT"

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

run_leg control 0
run_leg candidate 1

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/maintenance/kernel.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/maintenance/kernel.log" > "$OUT/maintenance/kernel-signatures.txt" || true
if [[ -s "$OUT/maintenance/kernel-signatures.txt" ]]; then
    printf 'HARDWARE_FAULT_SIGNATURES_DETECTED\n' >&2
    exit 1
fi

printf 'PROFILE_COMPLETE out=%s\n' "$OUT"
