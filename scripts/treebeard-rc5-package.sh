#!/usr/bin/env bash
# Package turbo-statetree-0.1.0-rc.5 from the frozen ragged-KV promotion build
# and smoke it on the isolated bench port. Does NOT touch production; the
# rc4->rc5 promotion flip is a separate explicit step.
#
# Usage: treebeard-rc5-package.sh [hoist 0|1]   (default 0; per A4/C3 verdict)
set -Eeuo pipefail

HOIST="${1:-0}"
ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
NAME=turbo-statetree-0.1.0-rc.5
RELEASE=/home/frosty40/turbo/turbo-combined/release/$NAME
PREFIX=/opt/$NAME
ALIAS=$NAME-Qwen3.6-35B-A3B-Q5-c262144-np12-ragged
UNIT="$HOME/.config/systemd/user/turbo-statetree-rc5.service"
BENCH_PORT=8098
FROZEN_COMMIT=de0834ca0
OUT="$ROOT/results/treebeard-rc5-deploy/$(date +%Y%m%d-%H%M%S)-package-smoke"
mkdir -p "$OUT"

# Frozen-source assertion: runtime source must be byte-identical to the
# promotion evidence build (later commits are reports/scripts only).
if ! git -C "$WORKTREE" diff --quiet "$FROZEN_COMMIT" HEAD -- src ggml tools common include; then
    printf 'RUNTIME_SOURCE_DIVERGED_FROM_FROZEN_COMMIT\n' >&2
    exit 1
fi
sha256sum "$BUILD/bin/llama-server" | rg -q \
    '^393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c ' || {
    printf 'BUILD_BINARY_HASH_MISMATCH_VS_FREEZE_RECORD\n' >&2
    exit 1
}

rm -rf "$RELEASE"
mkdir -p "$RELEASE/metadata"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
DESTDIR="$RELEASE/root" cmake --install "$BUILD" --prefix "$PREFIX" >/dev/null

git -C "$WORKTREE" rev-parse "$FROZEN_COMMIT" > "$RELEASE/metadata/commit.txt"
printf 'clean committed source build from private branch agent/treebeard-single-wavefront\n' \
    >> "$RELEASE/metadata/commit.txt"
cp "$BUILD/CMakeCache.txt" "$RELEASE/metadata/CMakeCache.txt"
sha256sum "$MODEL" > "$RELEASE/metadata/model.sha256"
( cd "$RELEASE/root$PREFIX" && find bin lib -type f -exec sha256sum {} + | sort -k2 ) \
    > "$RELEASE/metadata/runtime-libraries.sha256"
cat > "$RELEASE/metadata/provenance.txt" <<EOF
product=Turbo StateTree
version=0.1.0-rc.5
packaging_revision=pkg1
source_commit=$(git -C "$WORKTREE" rev-parse "$FROZEN_COMMIT")
baseline_tag=treebeard-statetree-0.1.0-rc.4
lineage=private downstream fork of ggml-org/llama.cpp
branch=agent/treebeard-single-wavefront
optimization=sequence-ragged StateTree KV attention (81df9e2ff, 789dfe8ec, 827a4f007) plus recurrent state-I/O fusion (76befe8c8) on the accepted T2+T3 base; Q8 ncols weight-hoist compiled default-off (a1d92fed7), ship env GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=$HOIST
rejected_paths=ragged union variant, quantized KV, token-level speculative verification (3 recorded verdicts)
control_evidence=results/treebeard-ragged-promo-b70 (golden 20260715-193609, confirm per latest-run.txt)
model=Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
model_sha256=25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506
context=262144
note=installed binaries differ from build hashes only by cmake install-time RPATH stripping; the unit supplies LD_LIBRARY_PATH; build-binary hashes are in results/treebeard-ragged-promo-b70/build-freeze-20260715.md
EOF

cat > "$UNIT" <<EOF
[Unit]
Description=Turbo StateTree 0.1.0-rc.5 ragged+state-io B70 262K / 12-slot serving surface on :8093
Wants=network-online.target
After=network-online.target

[Service]
Type=exec
WorkingDirectory=$RELEASE
Environment=GGML_SYCL_ENABLE_FUSION=1
Environment=GGML_SYCL_DISABLE_GRAPH=1
Environment=GGML_SYCL_ENABLE_MOE_PIPELINE=0
Environment=GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0
Environment=LLAMA_KV_TREE_RAGGED=1
Environment=GGML_SYCL_ENABLE_STATE_IO_FUSION=1
Environment=GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=$HOIST
ExecStart=/usr/bin/bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=$RELEASE/root$PREFIX/lib:\$LD_LIBRARY_PATH; exec /usr/bin/taskset -c 0-10,12-15 $RELEASE/root$PREFIX/bin/llama-server -m $MODEL -ngl 99 -ncmoe 0 --no-op-offload -c 262144 -np 12 -kvu -fa on -ctk f16 -ctv f16 -b 8192 -ub 1024 -t 15 --host 0.0.0.0 --port 8093 --jinja --metrics -a $ALIAS'
Restart=on-failure
RestartSec=10
TimeoutStartSec=180
TimeoutStopSec=60
OOMPolicy=stop
LimitNOFILE=65536

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload

# --- Isolated smoke on the bench port. The B70 cannot hold two full
# instances, so production is stopped for the smoke and restored after. ---
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
if ss -ltn "( sport = :$BENCH_PORT )" | rg -q LISTEN; then
    printf 'BENCH_PORT_BUSY\n' >&2
    exit 1
fi
systemctl --user is-active --quiet "$SERVICE"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" > "$OUT/before-props.json"
restore_rc4() {
    local rc="${1:-0}"
    trap - EXIT INT TERM HUP
    set +e
    if [[ -n "${SMOKE_PID:-}" ]]; then kill "$SMOKE_PID" 2>/dev/null; wait "$SMOKE_PID" 2>/dev/null; fi
    systemctl --user start "$SERVICE"
    for _ in {1..240}; do
        curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" >/dev/null 2>&1 && break
        sleep 1
    done
    curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" > "$OUT/restore-props.json"
    if ! cmp -s <(jq -S '{build_info,model_alias}' "$OUT/before-props.json") \
                <(jq -S '{build_info,model_alias}' "$OUT/restore-props.json"); then
        printf 'RESTORE_IDENTITY_MISMATCH\n' >&2
        exit 1
    fi
    printf 'RESTORE_OK\n'
    exit "$rc"
}
trap 'restore_rc4 $?' EXIT INT TERM HUP
systemctl --user stop "$SERVICE"
for _ in {1..60}; do
    systemctl --user is-active --quiet "$SERVICE" || break
    sleep 1
done
! systemctl --user is-active --quiet "$SERVICE"
env GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_DISABLE_GRAPH=1 \
    GGML_SYCL_ENABLE_MOE_PIPELINE=0 GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
    LLAMA_KV_TREE_RAGGED=1 GGML_SYCL_ENABLE_STATE_IO_FUSION=1 \
    GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST="$HOIST" \
    LD_LIBRARY_PATH="$RELEASE/root$PREFIX/lib:${LD_LIBRARY_PATH:-}" \
    taskset -c 0-10,12-15 "$RELEASE/root$PREFIX/bin/llama-server" \
    -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload \
    -c 262144 -np 12 -kvu -fa on -ctk f16 -ctv f16 -b 8192 -ub 1024 -t 15 \
    --host 127.0.0.1 --port "$BENCH_PORT" --jinja --metrics -a "$ALIAS" \
    > "$OUT/smoke-server.log" 2>&1 &
SMOKE_PID=$!
for _ in {1..240}; do
    curl -fsS --max-time 3 "http://127.0.0.1:$BENCH_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$SMOKE_PID" 2>/dev/null || { tail -50 "$OUT/smoke-server.log" >&2; exit 1; }
    sleep 1
done
curl -fsS "http://127.0.0.1:$BENCH_PORT/props" > "$OUT/smoke-props.json"
jq -e --arg alias "$ALIAS" \
    '.model_alias == $alias and .total_slots == 12 and
     .default_generation_settings.n_ctx == 262144' "$OUT/smoke-props.json" >/dev/null
jq -r '.build_info' "$OUT/smoke-props.json" > "$OUT/smoke-build-info.txt"
curl -fsS --max-time 120 "http://127.0.0.1:$BENCH_PORT/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Deterministic smoke: 2+2=","n_predict":8,"temperature":0,"seed":42}' \
    > "$OUT/smoke-completion.json"
jq -e '.content | length > 0' "$OUT/smoke-completion.json" >/dev/null
kill "$SMOKE_PID"; wait "$SMOKE_PID" 2>/dev/null || true
SMOKE_PID=

sha256sum "$RELEASE/root$PREFIX/bin/llama-server" > "$OUT/release-exe.sha256"
printf 'PACKAGE_SMOKE_OK release=%s unit=%s out=%s build=%s\n' \
    "$RELEASE" "$UNIT" "$OUT" "$(cat "$OUT/smoke-build-info.txt")"
# EXIT trap restores RC4 with rc=0.
