#!/usr/bin/env bash
# Dense golden-shape 1/8/12-agent no-regression A/B/A for the ragged-KV +
# state-I/O ship config. Clone of treebeard-moe-b70-aba-guarded.sh updated to
# RC4 production identity and the LLAMA_KV_TREE_RAGGED /
# GGML_SYCL_ENABLE_STATE_IO_FUSION arm contract. Same binary all arms.
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
BUILD="$ROOT/build-treebeard-single-wavefront"
SRC=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
HARNESS="$SRC/scripts/turbo-multiagent-pareto.py"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
PORT=8098
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
SKIP_CORRECTNESS="${TREEBEARD_SKIP_CORRECTNESS:-0}"
read -r -a BENCH_AGENTS <<< "${TREEBEARD_BENCH_AGENTS:-1 8 12}"
BENCH_REPEATS="${TREEBEARD_BENCH_REPEATS:-10}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$ROOT/results/treebeard-ragged-promo-b70"
OUT="$OUT_ROOT/$RUN_ID-golden-aba"
CANDIDATE_PID=
CURRENT_LABEL=none

mkdir -p "$OUT/maintenance"
printf '%s\n' "$OUT" > "$OUT_ROOT/latest-run.txt"

# Explicit arm contract: every experiment toggle pinned, no ambient defaults.
BASE_ENV=(
    GGML_SYCL_ENABLE_FUSION=1
    GGML_SYCL_DISABLE_GRAPH=1
    GGML_SYCL_ENABLE_MOE_PIPELINE=0
    GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0
    GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0
    GGML_SYCL_MOE_DOWN_REDUCE_DEBUG=1
    GGML_SYCL_MOE_DUAL_SWIGLU_DEBUG=1
)
CONTROL_ENV=(LLAMA_KV_TREE_RAGGED=0 GGML_SYCL_ENABLE_STATE_IO_FUSION=0)
CANDIDATE_ENV=(LLAMA_KV_TREE_RAGGED=1 GGML_SYCL_ENABLE_STATE_IO_FUSION=1)
STIO_ONLY_ENV=(LLAMA_KV_TREE_RAGGED=0 GGML_SYCL_ENABLE_STATE_IO_FUSION=1)
STIO_DIAG="${TREEBEARD_STIO_DIAG:-1}"

stop_candidate() {
    local label="${CURRENT_LABEL:-none}"
    local dir="$OUT/$label"
    if [[ -n "${CANDIDATE_PID:-}" ]] && kill -0 "$CANDIDATE_PID" 2>/dev/null; then
        mkdir -p "$dir"
        for fd in /proc/"$CANDIDATE_PID"/fd/*; do
            local target
            target=$(readlink "$fd" 2>/dev/null || true)
            if [[ "$target" == /dev/dri/* ]]; then
                cp "/proc/$CANDIDATE_PID/fdinfo/${fd##*/}" \
                    "$dir/drm-fdinfo-end-${fd##*/}.txt" 2>/dev/null || true
            fi
        done
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
    CURRENT_LABEL=none
    for _ in {1..30}; do
        if ! ss -ltn "( sport = :$PORT )" | rg -q LISTEN; then
            return 0
        fi
        sleep 1
    done
    printf 'BENCH_PORT_STILL_LISTENING port=%s\n' "$PORT" >&2
    return 1
}

restore_service() {
    local rc="${1:-0}"
    local ok=1
    trap - EXIT INT TERM HUP
    set +e
    stop_candidate || ok=0
    systemctl --user start "$SERVICE" || ok=0
    for _ in {1..180}; do
        if curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" \
            > "$OUT/maintenance/restore-health.json" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    systemctl --user show "$SERVICE" -p ActiveState -p SubState -p MainPID -p NRestarts --no-pager \
        > "$OUT/maintenance/restore-unit.txt" || ok=0
    systemctl --user is-active --quiet "$SERVICE" || ok=0
    curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/props" \
        > "$OUT/maintenance/restore-props.json" || ok=0
    jq -e --arg build "$EXPECTED_BUILD" --arg alias "$EXPECTED_ALIAS" \
        '.build_info == $build and .model_alias == $alias and
         .total_slots == 12 and .default_generation_settings.n_ctx == 262144' \
        "$OUT/maintenance/restore-props.json" >/dev/null || ok=0
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

start_server() {
    local label="$1"
    local mode="$2"
    shift 2
    local -a extra_env=("$@")
    local dir="$OUT/$label"
    mkdir -p "$dir/pareto"
    CURRENT_LABEL="$label"

    local -a mode_env
    if [[ "$mode" == control ]]; then
        mode_env=("${CONTROL_ENV[@]}")
    elif [[ "$mode" == stio-only ]]; then
        mode_env=("${STIO_ONLY_ENV[@]}")
    else
        mode_env=("${CANDIDATE_ENV[@]}")
    fi

    env -u GGML_SYCL_DISABLE_MMID_GROUPED \
        -u GGML_SYCL_DISABLE_MMVQ_12COL \
        -u GGML_SYCL_MMID_WG_SUBGROUPS \
        -u GGML_SYCL_DISABLE_MOE_DOWN_REDUCE \
        -u GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU \
        -u GGML_SYCL_DISABLE_MOE_DOWN_GROUPED \
        -u GGML_SYCL_STATE_IO_MODE \
        -u LLAMA_KV_INDEXED_FATTN \
        "${BASE_ENV[@]}" "${mode_env[@]}" "${extra_env[@]}" \
        taskset -c 0-10,12-15 "$BUILD/bin/llama-server" \
        -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload \
        -c 262144 -np 12 -kvu -fa on -ctk f16 -ctv f16 \
        -b 8192 -ub 1024 -t 15 \
        --host 127.0.0.1 --port "$PORT" --jinja --metrics -a "$label" \
        > "$dir/server.log" 2>&1 &
    CANDIDATE_PID=$!
    printf '%s\n' "$CANDIDATE_PID" > "$dir/server.pid"

    for _ in {1..240}; do
        if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" \
            > "$dir/health.json" 2>/dev/null; then
            curl -fsS --max-time 3 "http://127.0.0.1:$PORT/props" > "$dir/props.json"
            sha256sum "$BUILD/bin/llama-server" "$BUILD/bin/libggml-sycl.so" \
                "$BUILD/bin/libllama-server-impl.so" > "$dir/runtime.sha256"
            printf 'SERVER_READY label=%s mode=%s pid=%s\n' "$label" "$mode" "$CANDIDATE_PID"
            return 0
        fi
        if ! kill -0 "$CANDIDATE_PID" 2>/dev/null; then
            tail -100 "$dir/server.log" >&2 || true
            return 1
        fi
        sleep 1
    done
    printf 'SERVER_START_TIMEOUT label=%s\n' "$label" >&2
    return 1
}

run_screen() {
    local label="$1"
    local dir="$OUT/$label"
    python3 "$HARNESS" \
        --port "$PORT" --parallel 12 --ctx 262144 \
        --agents "${BENCH_AGENTS[@]}" --repeats "${BENCH_REPEATS}" --n-predict 256 --seed 42 \
        --request-timeout 900 --out-dir "$dir/pareto" --label "$label" \
        2>&1 | tee "$dir/pareto-console.log"
}

check_common_arm() {
    # Both arms carry the RC4 fusions; both must hit and neither may fall back.
    local label="$1"
    rg -q 'event=batched-integrated-hit' "$OUT/$label/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/$label/server.log"
    if rg -q 'event=batched-weights-snapshot' "$OUT/$label/server.log"; then
        printf '%s_UNEXPECTED_WEIGHTS_SNAPSHOT\n' "${label^^}" >&2
        exit 1
    fi
    if rg -q 'event=batched-hit' "$OUT/$label/server.log"; then
        printf '%s_UNEXPECTED_MATERIALIZED_FALLBACK\n' "${label^^}" >&2
        exit 1
    fi
}

# NOTE: llama-server suppresses INFO-level llama internals in this
# configuration, so the `sequence-ragged indexed attention available` INFO
# line NEVER appears in server logs (only W/E lines print — verified against
# the accepted 20260715 screen artifacts and the 20260715-192041 attempt).
# Capability/activation evidence for the ragged path lives in the fragmented
# confirm run (kv-page-probe n_kv reduction + tps gates), not here. Here we
# assert only the env contract via the W-level disable line.
check_control_arm() {
    local label="$1"
    check_common_arm "$label"
    rg -q 'sequence-ragged attention disabled' "$OUT/$label/server.log"
}

check_candidate_arm() {
    local label="$1"
    check_common_arm "$label"
    if rg -q 'sequence-ragged attention disabled' "$OUT/$label/server.log"; then
        printf '%s_UNEXPECTED_RAGGED_DISABLED\n' "${label^^}" >&2
        exit 1
    fi
}

# --- Preflight: live production identity ---
systemctl --user is-active --quiet "$SERVICE"
curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" \
    > "$OUT/maintenance/before-health.json"
curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/props" \
    > "$OUT/maintenance/before-props.json"
jq -e --arg build "$EXPECTED_BUILD" --arg alias "$EXPECTED_ALIAS" \
    '.build_info == $build and .model_alias == $alias and
     .total_slots == 12 and .default_generation_settings.n_ctx == 262144' \
    "$OUT/maintenance/before-props.json" >/dev/null
systemctl --user show "$SERVICE" -p ActiveState -p SubState -p MainPID --no-pager \
    > "$OUT/maintenance/before-unit.txt"
sha256sum "$BUILD/bin/llama-server" "$BUILD/bin/libggml-sycl.so" \
    "$BUILD/bin/libllama-server-impl.so" > "$OUT/candidate-runtime.sha256"
rg -n 'CMAKE_BUILD_TYPE:|GGML_SYCL|IntelSYCL_DIR|DNNL_DIR|MKL_DIR|TBB_DIR' \
    "$BUILD/CMakeCache.txt" > "$OUT/candidate-cmake.txt"
git -C "$SRC" rev-parse HEAD > "$OUT/maintenance/source-rev.txt"
git -C "$SRC" status --short > "$OUT/maintenance/source-status.txt"
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

# --- Correctness before any perf run ---
if [[ "$SKIP_CORRECTNESS" != 1 ]]; then
    "$BUILD/bin/test-tree-ragged-kv" 2>&1 | tee "$OUT/ragged-unit.log"
    rg -q 'Treebeard ragged KV plan tests passed' "$OUT/ragged-unit.log"
    env GGML_SYCL_MOE_DOWN_REDUCE_DEBUG=1 \
        "$BUILD/bin/test-backend-ops" test -b SYCL0 -o MOE_DOWN_REDUCE \
        2>&1 | tee "$OUT/correctness.log"
    rg -q '\[treebeard-moe-down-reduce\]' "$OUT/correctness.log"
    env GGML_SYCL_MOE_DUAL_SWIGLU_DEBUG=1 \
        "$BUILD/bin/test-backend-ops" test -b SYCL0 -o MOE_DUAL_SWIGLU \
        2>&1 | tee "$OUT/dual-swiglu-correctness.log"
    rg -q 'event=dual-integrated-hit' "$OUT/dual-swiglu-correctness.log"
fi

# --- A/B/A perf arms ---
start_server control-a control
run_screen control-a
check_control_arm control-a
stop_candidate
sleep 5

start_server candidate candidate
run_screen candidate
check_candidate_arm candidate
stop_candidate
sleep 5

start_server control-b control
run_screen control-b
check_control_arm control-b
stop_candidate
sleep 5

# Attribution arm (diagnostic, not a gate arm): state-io fusion alone on the
# dense shape, to separate its contribution from the ragged env.
if [[ "$STIO_DIAG" == 1 ]]; then
    start_server stio-only stio-only
    run_screen stio-only
    check_control_arm stio-only
    stop_candidate
    sleep 5
fi

# --- Activation diagnostic (not a perf arm): candidate env + state-io debug,
# one short completion; assert the state-io runtime plan forms. ---
start_server diag-activation candidate GGML_SYCL_STATE_IO_DEBUG=1
curl -fsS --max-time 120 "http://127.0.0.1:$PORT/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Treebeard activation probe:","n_predict":16,"temperature":0}' \
    > "$OUT/diag-activation/completion.json"
jq -e '.content | length > 0' "$OUT/diag-activation/completion.json" >/dev/null
rg -q '\[treebeard-state-io\] event=runtime-plan direct=[1-9]' \
    "$OUT/diag-activation/server.log"
stop_candidate

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/kernel-journal.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true

printf 'BENCH_COMPLETE out=%s\n' "$OUT"
