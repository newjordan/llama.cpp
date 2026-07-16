#!/usr/bin/env bash
# PCBT-12 three-arm product-value benchmark: single-pass baseline + manual
# breakout (arm A) and single-pass baseline + PCBT transactions (arm B) over
# the 30-task real-artifact suite, on a guarded B70 bench server.
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
SUITE="$WORKTREE/tests/pcbt/pcbt12-value-suite.json"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE="${TREEBEARD_LIVE_SERVICE:-turbo-statetree-rc7.service}"
LIVE_PORT=8093
BENCH_PORT=8098
RUN_ID="$(date +%Y%m%d-%H%M%S)-pcbt12"
OUT="$ROOT/results/treebeard-pcbt/$RUN_ID"
STARTED="$(date --iso-8601=seconds)"
BENCH_PID=

mkdir -p "$OUT/maintenance" "$OUT/manual" "$OUT/pcbt"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-pcbt/latest-run.txt"

restore_production() {
    local rc="${1:-0}" ok=1
    trap - EXIT INT TERM HUP
    set +e
    if [[ -n "${BENCH_PID:-}" ]] && kill -0 "$BENCH_PID" 2>/dev/null; then
        kill "$BENCH_PID"
        for _ in {1..60}; do kill -0 "$BENCH_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$BENCH_PID" 2>/dev/null
        wait "$BENCH_PID" 2>/dev/null
    fi
    systemctl --user start "$SERVICE" || ok=0
    for _ in {1..240}; do
        curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" >/dev/null 2>&1 && break
        sleep 1
    done
    curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" > "$OUT/maintenance/restore-props.json" || ok=0
    cmp -s <(jq -S '{build_info,model_alias}' "$OUT/maintenance/before-props.json") \
           <(jq -S '{build_info,model_alias}' "$OUT/maintenance/restore-props.json") || ok=0
    journalctl -k --since "$STARTED" --no-pager > "$OUT/maintenance/kernel.log" 2>/dev/null || true
    rg -i 'xe.*(hang|reset|fault)' "$OUT/maintenance/kernel.log" > "$OUT/maintenance/kernel-signatures.txt" || true
    (( ok )) && printf 'RESTORE_OK out=%s\n' "$OUT" || { printf 'RESTORE_FAILED out=%s\n' "$OUT" >&2; exit 1; }
    exit "$rc"
}
trap 'restore_production $?' EXIT INT TERM HUP

systemctl --user is-active --quiet "$SERVICE"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" > "$OUT/maintenance/before-props.json"
git -C "$WORKTREE" rev-parse HEAD > "$OUT/maintenance/source-head.txt"
sha256sum "$BUILD/bin/llama-server" "$BUILD/bin/libllama-server-impl.so" > "$OUT/candidate-runtime.sha256"

systemctl --user stop "$SERVICE"
for _ in {1..60}; do systemctl --user is-active --quiet "$SERVICE" || break; sleep 1; done
! systemctl --user is-active --quiet "$SERVICE"
printf 'SERVICE_STOPPED_GUARDED\n'

set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u

env TREEBEARD_PCBT_ENABLE=1 \
    LLAMA_KV_TREE_RAGGED=1 GGML_SYCL_ENABLE_STATE_IO_FUSION=1 \
    GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0 GGML_SYCL_ENABLE_FUSION=1 \
    GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
    GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
    taskset -c 0-10,12-15 "$BUILD/bin/llama-server" \
    -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload \
    -c 262144 -np 12 -kvu -fa on -ctk f16 -ctv f16 -b 8192 -ub 1024 -t 15 \
    --host 127.0.0.1 --port "$BENCH_PORT" --jinja --metrics -a pcbt12 \
    > "$OUT/bench-server.log" 2>&1 &
BENCH_PID=$!
for _ in {1..240}; do
    curl -fsS --max-time 3 "http://127.0.0.1:$BENCH_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$BENCH_PID" 2>/dev/null || { tail -50 "$OUT/bench-server.log" >&2; exit 1; }
    sleep 1
done

# Arm A: single-pass baseline + manual fixed-fanout breakout
python3 "$WORKTREE/scripts/turbo-speculative-breakout.py" \
    --benchmark-suite-file "$SUITE" --attach --port "$BENCH_PORT" \
    --branch-slots 0-7 --no-adaptive-fanout --no-prefix-clone \
    --out-dir "$OUT/manual" 2>&1 | tail -3

# Arm B: single-pass baseline + PCBT transactions
python3 "$WORKTREE/scripts/turbo-speculative-breakout.py" \
    --benchmark-suite-file "$SUITE" --attach --port "$BENCH_PORT" \
    --branch-slots 1-5 --prefix-slot 0 --preserve-prefix-root --pcbt \
    --no-prefix-clone --out-dir "$OUT/pcbt" 2>&1 | tail -3

curl -s "http://127.0.0.1:$BENCH_PORT/metrics" | rg "pcbt_" > "$OUT/pcbt-metrics.txt" || true
printf 'BENCH_COMPLETE out=%s\n' "$OUT"
