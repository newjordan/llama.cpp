#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
BUILD="$ROOT/build-treebeard-moe-down-reduce"
HARNESS="$ROOT/worktrees/treebeard-moe-down-reduce/scripts/turbo-multiagent-pareto.py"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE=turbo-statetree-rc2.service
LIVE_PORT=8093
PORT=8098
EXPECTED_BUILD=b72-70acde5e6
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.2-Qwen3.6-35B-A3B-Q5-c262144-np12
COMBINED="${TREEBEARD_COMBINED:-0}"
GRAPH_ABA="${TREEBEARD_GRAPH_ABA:-0}"
DOWN_GROUPED_ABA="${TREEBEARD_DOWN_GROUPED_ABA:-0}"
PIPELINE_ABA="${TREEBEARD_PIPELINE_ABA:-0}"
SKIP_CORRECTNESS="${TREEBEARD_SKIP_CORRECTNESS:-0}"
read -r -a BENCH_AGENTS <<< "${TREEBEARD_BENCH_AGENTS:-1 8 12}"
BENCH_REPEATS="${TREEBEARD_BENCH_REPEATS:-5}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
if [[ "$PIPELINE_ABA" == 1 ]]; then
    OUT_ROOT="$ROOT/results/treebeard-moe-pipeline"
elif [[ "$DOWN_GROUPED_ABA" == 1 ]]; then
    OUT_ROOT="$ROOT/results/treebeard-moe-down-grouped"
elif [[ "$GRAPH_ABA" == 1 ]]; then
    OUT_ROOT="$ROOT/results/treebeard-sycl-graph"
elif [[ "$COMBINED" == 1 ]]; then
    OUT_ROOT="$ROOT/results/treebeard-moe-combined"
else
    OUT_ROOT="$ROOT/results/treebeard-moe-down-reduce"
fi
OUT="$OUT_ROOT/$RUN_ID"
CANDIDATE_PID=
CURRENT_LABEL=none
SERVICE_STOPPED=0

mkdir -p "$OUT/maintenance"
printf '%s\n' "$OUT" > "$OUT_ROOT/latest-run.txt"

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
    systemctl --user show "$SERVICE" -p ActiveState -p SubState -p MainPID --no-pager \
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
    local dir="$OUT/$label"
    mkdir -p "$dir/pareto"
    CURRENT_LABEL="$label"

    local -a mode_env
    if [[ "$PIPELINE_ABA" == 1 && "$mode" == control ]]; then
        mode_env=(GGML_SYCL_DISABLE_GRAPH=1)
    elif [[ "$PIPELINE_ABA" == 1 ]]; then
        mode_env=(GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ENABLE_MOE_PIPELINE=1)
    elif [[ "$DOWN_GROUPED_ABA" == 1 && "$mode" == control ]]; then
        mode_env=(GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_MOE_DOWN_GROUPED=1)
    elif [[ "$DOWN_GROUPED_ABA" == 1 ]]; then
        mode_env=(GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=1)
    elif [[ "$GRAPH_ABA" == 1 && "$mode" == control ]]; then
        mode_env=(GGML_SYCL_DISABLE_GRAPH=1)
    elif [[ "$GRAPH_ABA" == 1 ]]; then
        mode_env=(GGML_SYCL_DISABLE_GRAPH=0)
    elif [[ "$mode" == control ]]; then
        mode_env=(GGML_SYCL_DISABLE_MOE_DOWN_REDUCE=1)
        if [[ "$COMBINED" == 1 ]]; then
            mode_env+=(GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1)
        fi
    else
        mode_env=()
    fi

    env -u GGML_SYCL_DISABLE_MMID_GROUPED \
        -u GGML_SYCL_DISABLE_MMVQ_12COL \
        -u GGML_SYCL_MMID_WG_SUBGROUPS \
        -u GGML_SYCL_DISABLE_MOE_DOWN_REDUCE \
        -u GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU \
        -u GGML_SYCL_DISABLE_MOE_DOWN_GROUPED \
        -u GGML_SYCL_ENABLE_MOE_DOWN_GROUPED \
        -u GGML_SYCL_ENABLE_MOE_PIPELINE \
        "${mode_env[@]}" \
        GGML_SYCL_MOE_DOWN_REDUCE_DEBUG=1 \
        GGML_SYCL_MOE_DUAL_SWIGLU_DEBUG="$(( COMBINED || GRAPH_ABA || DOWN_GROUPED_ABA || PIPELINE_ABA ))" \
        GGML_SYCL_MOE_PIPELINE_DEBUG="$PIPELINE_ABA" \
        GGML_SYCL_GRAPH_DEBUG="$GRAPH_ABA" \
        GGML_SYCL_ENABLE_FUSION=1 \
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
            for fd in /proc/"$CANDIDATE_PID"/fd/*; do
                local target
                target=$(readlink "$fd" 2>/dev/null || true)
                if [[ "$target" == /dev/dri/* ]]; then
                    cp "/proc/$CANDIDATE_PID/fdinfo/${fd##*/}" \
                        "$dir/drm-fdinfo-start-${fd##*/}.txt" 2>/dev/null || true
                fi
            done
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
date --iso-8601=seconds > "$OUT/maintenance/start-date.txt"

if ss -ltn "( sport = :$PORT )" | rg -q LISTEN; then
    printf 'BENCH_PORT_BUSY port=%s\n' "$PORT" >&2
    exit 1
fi

systemctl --user stop "$SERVICE"
SERVICE_STOPPED=1
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

if [[ "$GRAPH_ABA" == 1 ]]; then
    correctness_env=(GGML_SYCL_DISABLE_GRAPH=0 GGML_SYCL_GRAPH_DEBUG=1)
else
    correctness_env=()
fi

if [[ "$SKIP_CORRECTNESS" != 1 ]]; then
    if [[ "$PIPELINE_ABA" == 1 ]]; then
        env "${correctness_env[@]}" GGML_SYCL_ENABLE_MOE_PIPELINE=1 \
            GGML_SYCL_MOE_PIPELINE_DEBUG=1 \
            "$BUILD/bin/test-backend-ops" test -b SYCL0 -o MOE_PIPELINE \
            2>&1 | tee "$OUT/correctness.log"
        rg -q 'event=integrated-hit-views-first' "$OUT/correctness.log"
    else
        env "${correctness_env[@]}" GGML_SYCL_MOE_DOWN_REDUCE_DEBUG=1 \
            "$BUILD/bin/test-backend-ops" test -b SYCL0 -o MOE_DOWN_REDUCE \
            2>&1 | tee "$OUT/correctness.log"
        rg -q '\[treebeard-moe-down-reduce\]' "$OUT/correctness.log"
    fi
    if [[ "$COMBINED" == 1 || "$GRAPH_ABA" == 1 || "$DOWN_GROUPED_ABA" == 1 ]]; then
        env "${correctness_env[@]}" GGML_SYCL_MOE_DUAL_SWIGLU_DEBUG=1 \
            "$BUILD/bin/test-backend-ops" test -b SYCL0 -o MOE_DUAL_SWIGLU \
            2>&1 | tee "$OUT/dual-swiglu-correctness.log"
        rg -q 'event=dual-integrated-hit' "$OUT/dual-swiglu-correctness.log"
    fi
fi

start_server control-a control
run_screen control-a
if [[ "$PIPELINE_ABA" == 1 ]]; then
    rg -q 'event=batched-integrated-hit' "$OUT/control-a/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/control-a/server.log"
    if rg -q '\[treebeard-moe-pipeline\].*event=integrated-hit' "$OUT/control-a/server.log"; then
        printf 'CONTROL_A_UNEXPECTED_PIPELINE_HIT\n' >&2
        exit 1
    fi
elif [[ "$DOWN_GROUPED_ABA" == 1 ]]; then
    rg -q 'event=batched-integrated-hit' "$OUT/control-a/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/control-a/server.log"
    if rg -q 'event=batched-grouped-hit' "$OUT/control-a/server.log"; then
        printf 'CONTROL_A_UNEXPECTED_DOWN_GROUPED_HIT\n' >&2
        exit 1
    fi
elif [[ "$GRAPH_ABA" == 1 ]]; then
    rg -q 'event=batched-integrated-hit' "$OUT/control-a/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/control-a/server.log"
    if rg -q '\[SYCL-GRAPH\] (record|replay)' "$OUT/control-a/server.log"; then
        printf 'CONTROL_A_UNEXPECTED_GRAPH_EXECUTION\n' >&2
        exit 1
    fi
elif rg -q '\[treebeard-moe-down-reduce\]' "$OUT/control-a/server.log"; then
    printf 'CONTROL_A_UNEXPECTED_FUSION_HIT\n' >&2
    exit 1
fi
if [[ "$GRAPH_ABA" != 1 && "$DOWN_GROUPED_ABA" != 1 && "$PIPELINE_ABA" != 1 && "$COMBINED" == 1 ]] && rg -q 'event=dual-integrated-hit' "$OUT/control-a/server.log"; then
    printf 'CONTROL_A_UNEXPECTED_DUAL_SWIGLU_HIT\n' >&2
    exit 1
fi
stop_candidate
sleep 5

start_server candidate candidate
run_screen candidate
if [[ "$GRAPH_ABA" == 1 ]]; then
    rg -q '\[SYCL-GRAPH\] record uid=[1-9][0-9]*' "$OUT/candidate/server.log"
    rg -q '\[SYCL-GRAPH\] replay uid=[1-9][0-9]*' "$OUT/candidate/server.log"
fi
if [[ "$DOWN_GROUPED_ABA" == 1 ]]; then
    rg -q 'event=batched-grouped-hit.*tokens=12' "$OUT/candidate/server.log"
fi
if [[ "$PIPELINE_ABA" == 1 ]]; then
    rg -q 'treebeard-moe-pipeline.*event=liveness-annotated.*tokens=12' \
        "$OUT/candidate/server.log"
    rg -q 'treebeard-moe-pipeline.*event=integrated-hit-views-first.*gate_type=q5_K.*tokens=12' \
        "$OUT/candidate/server.log"
else
    rg -q 'event=batched-liveness-annotated' "$OUT/candidate/server.log"
    rg -q 'event=batched-integrated-hit' "$OUT/candidate/server.log"
fi
if rg -q 'event=batched-weights-snapshot' "$OUT/candidate/server.log"; then
    printf 'CANDIDATE_UNEXPECTED_WEIGHTS_SNAPSHOT\n' >&2
    exit 1
fi
if rg -q 'event=batched-hit' "$OUT/candidate/server.log"; then
    printf 'CANDIDATE_UNEXPECTED_MATERIALIZED_FALLBACK\n' >&2
    exit 1
fi
if [[ "$COMBINED" == 1 || "$GRAPH_ABA" == 1 || "$DOWN_GROUPED_ABA" == 1 ]]; then
    rg -q 'treebeard-moe-dual-swiglu.*event=dual-integrated-hit.*type=q5_K.*tokens=12' \
        "$OUT/candidate/server.log"
    rg -q 'treebeard-moe-dual-swiglu.*event=dual-integrated-hit.*type=q6_K.*tokens=12' \
        "$OUT/candidate/server.log"
    if rg -q 'treebeard-moe-dual-swiglu.*event=(dispatch-reject|activations-overlap|ids-overlap).*tokens=([1-9]|[1-5][0-9]|6[0-4])([^0-9]|$)' \
            "$OUT/candidate/server.log"; then
        printf 'CANDIDATE_UNEXPECTED_DUAL_SWIGLU_FALLBACK\n' >&2
        exit 1
    fi
fi
stop_candidate
sleep 5

start_server control-b control
run_screen control-b
if [[ "$PIPELINE_ABA" == 1 ]]; then
    rg -q 'event=batched-integrated-hit' "$OUT/control-b/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/control-b/server.log"
    if rg -q '\[treebeard-moe-pipeline\].*event=integrated-hit' "$OUT/control-b/server.log"; then
        printf 'CONTROL_B_UNEXPECTED_PIPELINE_HIT\n' >&2
        exit 1
    fi
elif [[ "$DOWN_GROUPED_ABA" == 1 ]]; then
    rg -q 'event=batched-integrated-hit' "$OUT/control-b/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/control-b/server.log"
    if rg -q 'event=batched-grouped-hit' "$OUT/control-b/server.log"; then
        printf 'CONTROL_B_UNEXPECTED_DOWN_GROUPED_HIT\n' >&2
        exit 1
    fi
elif [[ "$GRAPH_ABA" == 1 ]]; then
    rg -q 'event=batched-integrated-hit' "$OUT/control-b/server.log"
    rg -q 'event=dual-integrated-hit' "$OUT/control-b/server.log"
    if rg -q '\[SYCL-GRAPH\] (record|replay)' "$OUT/control-b/server.log"; then
        printf 'CONTROL_B_UNEXPECTED_GRAPH_EXECUTION\n' >&2
        exit 1
    fi
elif rg -q '\[treebeard-moe-down-reduce\]' "$OUT/control-b/server.log"; then
    printf 'CONTROL_B_UNEXPECTED_FUSION_HIT\n' >&2
    exit 1
fi
if [[ "$GRAPH_ABA" != 1 && "$DOWN_GROUPED_ABA" != 1 && "$PIPELINE_ABA" != 1 && "$COMBINED" == 1 ]] && rg -q 'event=dual-integrated-hit' "$OUT/control-b/server.log"; then
    printf 'CONTROL_B_UNEXPECTED_DUAL_SWIGLU_HIT\n' >&2
    exit 1
fi
stop_candidate

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/kernel-journal.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true

printf 'BENCH_COMPLETE out=%s\n' "$OUT"
