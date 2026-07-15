#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen G2 development run. This script never contacts GitHub and never
# changes the production unit definition. It stops only the exact user unit
# below and restores/verifies it from an EXIT trap.

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
PYTHON=/home/frosty40/nx2-venv/bin/python
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
JUDGE_MODEL=/home/frosty40/models/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf
EMBED_MODEL=/home/frosty40/models/embeddings/nomic-embed-text-v1.5.Q8_0.gguf
PHASE0="$ROOT/research/jspace-20260713/treebeard-jspace/artifacts/phase0"
ANCHORS="$ROOT/research/jspace-20260713/treebeard-jspace/anchors_qwen36_q5.json"
EXPERIMENT=${TREEBEARD_JSPACE_G2_EXPERIMENT:-development}
MANIFEST=${TREEBEARD_JSPACE_G2_MANIFEST:-$WORKTREE/research/jspace-g2/g2-development-response-manifest.json}
ROUTING_MANIFEST=${TREEBEARD_JSPACE_G2_ROUTING_MANIFEST:-$WORKTREE/research/jspace-g2/g2-development-routing-manifest.json}
G1_ARTIFACT="$WORKTREE/research/jspace-g1-v5/g1-sensor-v5.npz"
G1_EVALUATOR="$WORKTREE/scripts/treebeard-jspace-g1-v5-evaluate.py"
PHASE0_BUILD="$PHASE0/qwen36_q5_phase0_build.json"
PREREG=${TREEBEARD_JSPACE_G2_PREREG:-$WORKTREE/reports/treebeard-jspace-g2-preregistration-20260715.md}
EXTENSION_FREEZER=${TREEBEARD_JSPACE_G2_EXTENSION_FREEZER:-}
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
CANDIDATE_PORT=${TREEBEARD_JSPACE_G2_PORT:-18093}
PREFLIGHT_ONLY=${TREEBEARD_JSPACE_G2_PREFLIGHT_ONLY:-0}
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
EXPECTED_SERVER_SHA=211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff

MODEL_SHA=25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506
JUDGE_MODEL_SHA=0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e
EMBED_MODEL_SHA=3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7
MANIFEST_SHA=${TREEBEARD_JSPACE_G2_MANIFEST_SHA:-a39b8d041e51133a359298b5e99bf42d014aee6eb6ef5884b5056cb5dc355d4d}
ROUTING_MANIFEST_SHA=${TREEBEARD_JSPACE_G2_ROUTING_MANIFEST_SHA:-32953d19c9716568a3b55b79e604df3a8648c0284173943101e429edd46f07e4}
EXPECTED_ROWS=${TREEBEARD_JSPACE_G2_EXPECTED_ROWS:-50}
G1_ARTIFACT_SHA=6f50e2ea6c7cca8c6ee361d6611f0e764c1539a8a05d59f0c50f553fe0ace7d4
G1_EVALUATOR_SHA=ecf9ba704842a74a3fb360cbd2d12a46d1efa216729ca849b339b440fd8f9b07
PHASE0_BUILD_SHA=0b1f161edec3ed0916045c95da233e70466e233938c184d903b6853d257bb7a8
ANCHORS_SHA=cab229a60bbcf1cd641070227bd4fd4c5bc8345605c36afe5c81850ff15ebb71
PREREG_SHA=${TREEBEARD_JSPACE_G2_PREREG_SHA:-edbb09347348f467898f0ad0c6078f4e8128c2c01c9691f1361ae60c867a43bd}
EXTENSION_FREEZER_SHA=${TREEBEARD_JSPACE_G2_EXTENSION_FREEZER_SHA:-}
ROUTE_SHA=8610029f1261275084fafb50080e9151b60e51836b5275ea4c6ef6959005b401
GENERATE_SHA=adf3278dc4a715dfc5678a4e71cdb82bf3da0757b96a159ae6ce29d17e62ee6b
JUDGE_SHA=c8ca52bc365f81041150e72ba5f6908d100d21966bfb188d55e9eca68fa0e79d
EMBED_SHA=869a9e758e321ebb52fc613b3db7c1e46bc5668e985b77db16411a55b00471c4
EVALUATE_SHA=9d9ab88f1a6615b043e1d5d079ae8e9fc8b2b9de0948fd17c3f31a64cb676337
SELF_TEST_SHA=3d45f6804cad20496c071ee1fd64b3f47562389203581a251bea4bd995e3de0c
EXTRACTOR_SHA=e6d349ac9546a6e2086999c112f03d5c75b320c481ffffde8442cff85bcf9acb
SERVER_SHA=393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c
EMBED_BINARY_SHA=9b1bec4f30b6393b74511ea810f981258d93c87096e54949c73e6b67b9b86b60
SYCL_SHA=0a0536fd5eba5caf6c8c4b00febe4320e868fe19bc99bd42767eb77d6c872e61
LIBLLAMA_SHA=c71031ef41abeb15a6b291c7abdd060a8452928e76ebfb3e4d83dc065cf7ba22
VERBALIZERS=49166,12395,15420,64797,8415,18654,20002

declare -A VECTOR_PATH VECTOR_SHA
for axis in sadness surprise joy disgust fear anger curiosity; do
    VECTOR_PATH[$axis]="$PHASE0/qwen36_q5_phase0_${axis}_rawdual_l39.gguf"
done
VECTOR_SHA[sadness]=2d7f1d254bdbcbcea1350d028e2d5b4f5aabb381b1887f041d0bc1f6ca9fa8c7
VECTOR_SHA[surprise]=191534954903c474e80512b877b5ec2ef26ed8304416247cddaaab18d8d57830
VECTOR_SHA[joy]=a9af4c6b8dfb189e1a9f93917b2ea2d1154a1f6fb7987da2459a176c5a1cf8c0
VECTOR_SHA[disgust]=6e584cc9450b1734f18bf8be10595afa27ed754cd2f22d2d2d47f42d2af9cf28
VECTOR_SHA[fear]=0b577aca9899a62d49981f10ccd148a48073bc6b92c587aa3ca0aaf2fdf39ab5
VECTOR_SHA[anger]=338aaf760651ed491408d327b1b7879d06d0e5a0f9090dfa9087f18659f02108
VECTOR_SHA[curiosity]=9cd787c2bbde921a516afb7ba885db267cd575a021abba855da49804a096f0f9

SCALE_NAMES=(eighth quarter half)
SCALES=(0.008641079027104324 0.017282158054208648 0.034564316108417296)
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT="$ROOT/results/treebeard-jspace-g2/$EXPERIMENT/$RUN_ID"
ROUTES="$OUT/routing/g2-$EXPERIMENT-routes.json"
ATTESTATION="$OUT/attestation/run-attestation.json"
SERVER_PID=
SERVICE_STOPPED=0
BEFORE_NRESTARTS=

mkdir -p "$OUT"/{attestation,maintenance,routing,responses,judgments,embeddings,evaluations,logs}
mkdir -p "$ROOT/results/treebeard-jspace-g2/$EXPERIMENT"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-jspace-g2/$EXPERIMENT/latest-run.txt"

sha() {
    sha256sum "$1" | awk '{print $1}'
}

require_sha() {
    local path=$1 expected=$2 label=$3 actual
    actual=$(sha "$path")
    if [[ "$actual" != "$expected" ]]; then
        printf 'HASH_MISMATCH label=%s actual=%s expected=%s path=%s\n' \
            "$label" "$actual" "$expected" "$path" >&2
        exit 1
    fi
}

stop_candidate() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

restore_service() {
    local incoming_rc=$?
    local rc=$incoming_rc ok=1 restored_pid restored_exe restored_restarts
    trap - EXIT INT TERM HUP
    set +e
    stop_candidate
    if [[ -f "$OUT/maintenance/start-date.txt" ]]; then
        journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
            > "$OUT/kernel-journal.log" 2>&1 || true
        rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
            "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true
        [[ ! -s "$OUT/kernel-signatures.txt" ]] || ok=0
    fi
    if [[ "$SERVICE_STOPPED" == 1 ]]; then
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
        restored_pid=$(systemctl --user show "$SERVICE" -p MainPID --value)
        restored_exe=$(readlink -f "/proc/$restored_pid/exe" 2>/dev/null)
        if [[ -z "$restored_exe" ]]; then
            ok=0
        else
            sha256sum "$restored_exe" > "$OUT/maintenance/restore-server.sha256" || ok=0
            [[ $(awk '{print $1}' "$OUT/maintenance/restore-server.sha256") == \
                "$EXPECTED_SERVER_SHA" ]] || ok=0
        fi
        restored_restarts=$(systemctl --user show "$SERVICE" -p NRestarts --value)
        [[ "$restored_restarts" == "$BEFORE_NRESTARTS" ]] || ok=0
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
    fi
    date --iso-8601=seconds > "$OUT/maintenance/restore-date.txt"
    jq -n --argjson incoming_rc "$incoming_rc" --argjson restoration_ok "$ok" \
        --arg out "$OUT" '{schema:"treebeard.jspace.g2.run-status.v1",
        incoming_rc:$incoming_rc,restoration_ok:($restoration_ok == 1),out:$out}' \
        > "$OUT/run-status.json"
    if (( ok == 0 )); then
        printf 'RESTORE_FAILED out=%s\n' "$OUT" >&2
        rc=1
    else
        printf 'RESTORE_OK out=%s\n' "$OUT"
    fi
    exit "$rc"
}

trap restore_service EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

start_server() {
    local label=$1 model=$2 alias=$3 slots=$4 context=$5 vector=$6
    shift 6
    local vector_args=()
    stop_candidate
    if ss -ltn "( sport = :$CANDIDATE_PORT )" | rg -q LISTEN; then
        printf 'CANDIDATE_PORT_BUSY port=%s\n' "$CANDIDATE_PORT" >&2
        exit 1
    fi
    if [[ -n "$vector" ]]; then
        vector_args+=(--control-vector "$vector" --control-vector-layer-range 39 39)
    fi
    printf 'START_SERVER label=%s model=%s\n' "$label" "$model"
    env \
        GGML_SYCL_ENABLE_FUSION=1 \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
        GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
        taskset -c 0-10,12-15 "$BUILD/bin/llama-server" \
        -m "$model" -ngl 99 --no-op-offload -kvo -t 15 \
        -c "$context" -np "$slots" -b 256 -ub 256 \
        --host 127.0.0.1 --port "$CANDIDATE_PORT" --spec-type none \
        -a "$alias" "${vector_args[@]}" "$@" \
        > "$OUT/logs/server-$label.log" 2>&1 &
    SERVER_PID=$!
    printf '%s\n' "$SERVER_PID" > "$OUT/logs/server-$label.pid"
    for _ in {1..300}; do
        if curl -fsS --max-time 3 "http://127.0.0.1:$CANDIDATE_PORT/health" \
                > "$OUT/logs/server-$label-health.json" 2>/dev/null; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -200 "$OUT/logs/server-$label.log" >&2 || true
            exit 1
        fi
        sleep 1
    done
    curl -fsS --max-time 5 "http://127.0.0.1:$CANDIDATE_PORT/props" \
        > "$OUT/logs/server-$label-props.json"
    jq -e --arg alias "$alias" --argjson slots "$slots" --argjson context "$context" \
        '.model_alias == $alias and .total_slots == $slots and
         .default_generation_settings.n_ctx == $context' \
        "$OUT/logs/server-$label-props.json" >/dev/null
    tr '\0' '\n' < "/proc/$SERVER_PID/cmdline" > "$OUT/logs/server-$label-cmdline.txt"
    if [[ -n "$vector" ]]; then
        rg -F -x -- "$vector" "$OUT/logs/server-$label-cmdline.txt" >/dev/null
    fi
}

generate() {
    local out=$1 arm=$2 policy=$3 scale=$4 axis=$5 scope=$6
    printf 'GENERATE arm=%s policy=%s scale=%s axis=%s scope=%s\n' \
        "$arm" "$policy" "$scale" "$axis" "$scope"
    "$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-generate.py" \
        --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
        --routes "$ROUTES" --routes-sha256 "$ROUTES_SHA" \
        --port "$CANDIDATE_PORT" --arm "$arm" --policy "$policy" \
        --scale "$scale" --vector-axis "$axis" --row-scope "$scope" \
        --vector "${VECTOR_PATH[$axis]}" --vector-sha256 "${VECTOR_SHA[$axis]}" \
        --attestation "$ATTESTATION" --attestation-sha256 "$ATTESTATION_SHA" \
        --seed 1709 --out "$out"
    jq -e '.status == "complete" and .audit.failures == 0' "$out" >/dev/null
}

add_response() {
    local -n target=$1
    local path=$2
    target+=(--responses "$path" --responses-sha256 "$(sha "$path")")
}

build_response_args() {
    local policy=$1 scale_name=$2
    local target_name=$3
    local -n target=$target_name
    target=()
    if [[ "$policy" == engage_curiosity ]]; then
        add_response "$target_name" "$OUT/responses/engage-control.json"
        add_response "$target_name" "$OUT/responses/engage-$scale_name-candidate.json"
    else
        add_response "$target_name" "$OUT/responses/match-inactive-control.json"
        add_response "$target_name" "$OUT/responses/match-inactive-$scale_name-candidate.json"
        local axis
        for axis in "${ACTIVE_AXES[@]}"; do
            add_response "$target_name" "$OUT/responses/match-$axis-control.json"
            add_response "$target_name" "$OUT/responses/match-$axis-$scale_name-candidate.json"
        done
    fi
}

allow_gate_fail() {
    set +e
    "$@"
    local rc=$?
    set -e
    if [[ "$rc" != 0 && "$rc" != 2 ]]; then
        return "$rc"
    fi
    return 0
}

printf 'G2_PREFLIGHT out=%s\n' "$OUT"
require_sha "$MODEL" "$MODEL_SHA" base-model
require_sha "$JUDGE_MODEL" "$JUDGE_MODEL_SHA" judge-model
require_sha "$EMBED_MODEL" "$EMBED_MODEL_SHA" embedding-model
require_sha "$MANIFEST" "$MANIFEST_SHA" development-response-manifest
require_sha "$ROUTING_MANIFEST" "$ROUTING_MANIFEST_SHA" development-routing-manifest
require_sha "$G1_ARTIFACT" "$G1_ARTIFACT_SHA" g1-v5-artifact
require_sha "$G1_EVALUATOR" "$G1_EVALUATOR_SHA" g1-v5-evaluator
require_sha "$PHASE0_BUILD" "$PHASE0_BUILD_SHA" phase0-build-report
require_sha "$ANCHORS" "$ANCHORS_SHA" anchor-manifest
require_sha "$PREREG" "$PREREG_SHA" preregistration
if [[ -n "$EXTENSION_FREEZER" ]]; then
    require_sha "$EXTENSION_FREEZER" "$EXTENSION_FREEZER_SHA" extension-freezer
fi
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-route.py" "$ROUTE_SHA" route-client
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-generate.py" "$GENERATE_SHA" generator
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-judge.py" "$JUDGE_SHA" judge-client
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-embed.py" "$EMBED_SHA" embedding-client
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-evaluate.py" "$EVALUATE_SHA" evaluator
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-self-test.py" "$SELF_TEST_SHA" self-test
require_sha "$BUILD/bin/llama-jspace-sensor-extract" "$EXTRACTOR_SHA" sensor-extractor
require_sha "$BUILD/bin/llama-server" "$SERVER_SHA" server
require_sha "$BUILD/bin/llama-embedding" "$EMBED_BINARY_SHA" embedding-binary
require_sha "$BUILD/bin/libggml-sycl.so" "$SYCL_SHA" sycl-runtime
require_sha "$BUILD/bin/libllama.so" "$LIBLLAMA_SHA" llama-runtime
for axis in sadness surprise joy disgust fear anger curiosity; do
    require_sha "${VECTOR_PATH[$axis]}" "${VECTOR_SHA[$axis]}" "vector-$axis"
done
"$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-route.py" --self-test
"$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-self-test.py"

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
BEFORE_NRESTARTS=$(systemctl --user show "$SERVICE" -p NRestarts --value)
before_pid=$(systemctl --user show "$SERVICE" -p MainPID --value)
before_exe=$(readlink -f "/proc/$before_pid/exe")
require_sha "$before_exe" "$EXPECTED_SERVER_SHA" production-server
sha256sum "$before_exe" > "$OUT/maintenance/before-server.sha256"

git -C "$WORKTREE" rev-parse HEAD > "$OUT/attestation/source-head.txt"
git -C "$WORKTREE" status --porcelain=v2 --untracked-files=all \
    > "$OUT/attestation/source-status.txt"
git -C "$WORKTREE" diff --binary -- \
    scripts/treebeard-jspace-g2-manifest.py \
    scripts/treebeard-jspace-g2-route.py \
    scripts/treebeard-jspace-g2-generate.py \
    scripts/treebeard-jspace-g2-judge.py \
    scripts/treebeard-jspace-g2-embed.py \
    scripts/treebeard-jspace-g2-evaluate.py \
    scripts/treebeard-jspace-g2-self-test.py \
    scripts/treebeard-jspace-g2-development-b70-guarded.sh \
    reports/treebeard-jspace-g2-preregistration-20260715.md \
    research/jspace-g2 \
    > "$OUT/attestation/source.patch"
mkdir -p "$OUT/attestation/source-snapshot"
cp \
    "$WORKTREE/scripts/treebeard-jspace-g2-manifest.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-route.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-generate.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-judge.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-embed.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-evaluate.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-self-test.py" \
    "$WORKTREE/scripts/treebeard-jspace-g2-development-b70-guarded.sh" \
    "$PREREG" \
    "$OUT/attestation/source-snapshot/"
if [[ -n "$EXTENSION_FREEZER" ]]; then
    cp "$EXTENSION_FREEZER" "$OUT/attestation/source-snapshot/"
fi
sha256sum "$OUT"/attestation/source-snapshot/* \
    > "$OUT/attestation/source-snapshot.sha256"
sha256sum "$0" > "$OUT/attestation/runner.sha256"
sha256sum \
    "$MODEL" "$JUDGE_MODEL" "$EMBED_MODEL" "$MANIFEST" "$ROUTING_MANIFEST" \
    "$G1_ARTIFACT" "$G1_EVALUATOR" "$PHASE0_BUILD" "$ANCHORS" "$PREREG" \
    "$WORKTREE"/scripts/treebeard-jspace-g2-{manifest,route,generate,judge,embed,evaluate,self-test}.py \
    "$BUILD/bin/llama-jspace-sensor-extract" "$BUILD/bin/llama-server" \
    "$BUILD/bin/llama-embedding" "$BUILD/bin/libggml-sycl.so" \
    "$BUILD/bin/libllama.so" "${VECTOR_PATH[@]}" \
    > "$OUT/attestation/inputs.sha256"
if [[ -n "$EXTENSION_FREEZER" ]]; then
    sha256sum "$EXTENSION_FREEZER" >> "$OUT/attestation/inputs.sha256"
fi
SOURCE_HEAD=$(cat "$OUT/attestation/source-head.txt")
PATCH_SHA=$(sha "$OUT/attestation/source.patch")
STATUS_SHA=$(sha "$OUT/attestation/source-status.txt")
INPUTS_SHA=$(sha "$OUT/attestation/inputs.sha256")
SNAPSHOT_SHA=$(sha "$OUT/attestation/source-snapshot.sha256")
RUNNER_SHA=$(awk '{print $1}' "$OUT/attestation/runner.sha256")
jq -n --arg started "$(date --iso-8601=seconds)" --arg mode "$EXPERIMENT" \
    --arg head "$SOURCE_HEAD" \
    --arg patch_sha "$PATCH_SHA" --arg status_sha "$STATUS_SHA" \
    --arg inputs_sha "$INPUTS_SHA" --arg snapshot_sha "$SNAPSHOT_SHA" \
    --arg runner_sha "$RUNNER_SHA" \
    --arg prereg_sha "$PREREG_SHA" --arg server_sha "$SERVER_SHA" \
    --arg extractor_sha "$EXTRACTOR_SHA" --arg sycl_sha "$SYCL_SHA" \
    --arg model_sha "$MODEL_SHA" \
    '{schema:"treebeard.jspace.g2.run-attestation.v1",mode:$mode,
      started:$started,source:{head:$head,patch_sha256:$patch_sha,
      status_sha256:$status_sha,snapshot_sha256_file:$snapshot_sha},
      inputs_sha256_file:$inputs_sha,
      runner_sha256:$runner_sha,preregistration_sha256:$prereg_sha,
      runtime:{server_sha256:$server_sha,extractor_sha256:$extractor_sha,
      sycl_sha256:$sycl_sha},base_model_sha256:$model_sha}' \
    > "$ATTESTATION"
ATTESTATION_SHA=$(sha "$ATTESTATION")
date --iso-8601=seconds > "$OUT/maintenance/start-date.txt"

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    printf 'G2_PREFLIGHT_COMPLETE out=%s\n' "$OUT"
    exit 0
fi

SERVICE_STOPPED=1
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

PREFIX="$OUT/routing/g2-$EXPERIMENT-routing"
env \
    GGML_SYCL_ENABLE_FUSION=1 \
    GGML_SYCL_DISABLE_GRAPH=1 \
    GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
    GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
    taskset -c 0-10,12-15 "$BUILD/bin/llama-jspace-sensor-extract" \
    -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload -kvo \
    -c 256 -b 256 -ub 256 -t 15 \
    --manifest "$ROUTING_MANIFEST" \
    --verified-manifest-sha256 "$ROUTING_MANIFEST_SHA" \
    --verified-model-sha256 "$MODEL_SHA" \
    --routing-verbalizers "$VERBALIZERS" \
    --out-prefix "$PREFIX" \
    > "$OUT/logs/extractor-result.json" 2> "$OUT/logs/extractor.log"
jq -e --arg manifest_sha "$ROUTING_MANIFEST_SHA" --arg model_sha "$MODEL_SHA" \
    --argjson rows "$EXPECTED_ROWS" \
    '.schema == "treebeard.jspace.g1.routing-logits.v1" and
     .status == "exact_runtime_routing_verbalizer_logits" and
     .dataset.runner_verified_sha256 == $manifest_sha and
     .model.runner_verified_sha256 == $model_sha and
     .raw.dtype == "little_endian_float32" and .raw.shape == [$rows,7] and
     (.rows | length) == $rows and (.capture.verbalizers | length) == 7' \
    "$PREFIX.json" >/dev/null
[[ $(jq -r '.raw.bytes' "$PREFIX.json") == $(stat -c '%s' "$PREFIX.f32") ]]
METADATA_SHA=$(sha "$PREFIX.json")
RAW_SHA=$(sha "$PREFIX.f32")
"$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-route.py" \
    --response-manifest "$MANIFEST" --response-manifest-sha256 "$MANIFEST_SHA" \
    --routing-manifest "$ROUTING_MANIFEST" \
    --routing-manifest-sha256 "$ROUTING_MANIFEST_SHA" \
    --metadata "$PREFIX.json" --metadata-sha256 "$METADATA_SHA" \
    --raw "$PREFIX.f32" --raw-sha256 "$RAW_SHA" \
    --artifact "$G1_ARTIFACT" --artifact-sha256 "$G1_ARTIFACT_SHA" \
    --out "$ROUTES"
ROUTES_SHA=$(sha "$ROUTES")
ACTIVE_ROWS=$(jq -r '.audit.active_rows' "$ROUTES")
if (( ACTIVE_ROWS < 20 )); then
    printf 'G2_STOP active_rows=%s minimum=20\n' "$ACTIVE_ROWS" >&2
    exit 2
fi
mapfile -t ACTIVE_AXES < <(
    jq -r '.rows[] | select(.active == true) | .routed_axis' "$ROUTES" | sort -u)
printf 'ROUTING_COMPLETE active_rows=%s active_axes=%s\n' \
    "$ACTIVE_ROWS" "${ACTIVE_AXES[*]}"

start_server base-curiosity "$MODEL" treebeard-g2-base-curiosity 1 512 \
    "${VECTOR_PATH[curiosity]}" -ncmoe 0 -fa on -ctk f16 -ctv f16
generate "$OUT/responses/engage-control.json" control engage_curiosity 0 curiosity all
generate "$OUT/responses/match-inactive-control.json" control match_route 0 curiosity inactive
for index in "${!SCALES[@]}"; do
    name=${SCALE_NAMES[$index]}
    scale=${SCALES[$index]}
    generate "$OUT/responses/engage-$name-candidate.json" candidate \
        engage_curiosity "$scale" curiosity all
    generate "$OUT/responses/match-inactive-$name-candidate.json" candidate \
        match_route "$scale" curiosity inactive
done
stop_candidate

for axis in "${ACTIVE_AXES[@]}"; do
    start_server "base-$axis" "$MODEL" "treebeard-g2-base-$axis" 1 512 \
        "${VECTOR_PATH[$axis]}" -ncmoe 0 -fa on -ctk f16 -ctv f16
    generate "$OUT/responses/match-$axis-control.json" control match_route 0 \
        "$axis" active-axis
    for index in "${!SCALES[@]}"; do
        name=${SCALE_NAMES[$index]}
        scale=${SCALES[$index]}
        generate "$OUT/responses/match-$axis-$name-candidate.json" candidate \
            match_route "$scale" "$axis" active-axis
    done
    stop_candidate
done

start_server judge "$JUDGE_MODEL" treebeard-g2-judge-qwen3-8b-q8 1 4096 "" \
    -fa on -ctk f16 -ctv f16
for policy in engage_curiosity match_route; do
    for name in "${SCALE_NAMES[@]}"; do
        response_args=()
        build_response_args "$policy" "$name" response_args
        printf 'JUDGE policy=%s scale=%s\n' "$policy" "$name"
        "$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-judge.py" \
            --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
            --routes "$ROUTES" --routes-sha256 "$ROUTES_SHA" \
            "${response_args[@]}" --port "$CANDIDATE_PORT" \
            --out "$OUT/judgments/$policy-$name.json"
    done
done
stop_candidate

start_server embedding "$EMBED_MODEL" treebeard-g2-nomic-v1.5-q8 16 8192 "" \
    --embedding --pooling mean
for policy in engage_curiosity match_route; do
    for name in "${SCALE_NAMES[@]}"; do
        response_args=()
        build_response_args "$policy" "$name" response_args
        printf 'EMBED policy=%s scale=%s\n' "$policy" "$name"
        "$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-embed.py" \
            --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
            --routes "$ROUTES" --routes-sha256 "$ROUTES_SHA" \
            "${response_args[@]}" --port "$CANDIDATE_PORT" --batch-size 16 \
            --raw-out "$OUT/embeddings/$policy-$name.npz" \
            --out "$OUT/embeddings/$policy-$name.json"
    done
done
stop_candidate

selection_args=()
for policy in engage_curiosity match_route; do
    for name in "${SCALE_NAMES[@]}"; do
        response_args=()
        build_response_args "$policy" "$name" response_args
        judgment="$OUT/judgments/$policy-$name.json"
        embedding="$OUT/embeddings/$policy-$name.json"
        embedding_raw="$OUT/embeddings/$policy-$name.npz"
        report="$OUT/evaluations/$policy-$name.json"
        printf 'EVALUATE policy=%s scale=%s\n' "$policy" "$name"
        allow_gate_fail "$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-evaluate.py" \
            evaluate --mode development \
            --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
            --routes "$ROUTES" --routes-sha256 "$ROUTES_SHA" \
            "${response_args[@]}" \
            --judgments "$judgment" --judgments-sha256 "$(sha "$judgment")" \
            --embeddings "$embedding" --embeddings-sha256 "$(sha "$embedding")" \
            --embedding-raw "$embedding_raw" \
            --embedding-raw-sha256 "$(sha "$embedding_raw")" \
            --anchors "$ANCHORS" --anchors-sha256 "$ANCHORS_SHA" \
            --out "$report"
        selection_args+=(--report "$report" --report-sha256 "$(sha "$report")")
    done
done

set +e
"$PYTHON" "$WORKTREE/scripts/treebeard-jspace-g2-evaluate.py" select \
    "${selection_args[@]}" --out "$OUT/g2-selected-policy.json"
SELECTION_RC=$?
set -e
if [[ "$SELECTION_RC" != 0 && "$SELECTION_RC" != 2 ]]; then
    exit "$SELECTION_RC"
fi
sha256sum "$OUT/g2-selected-policy.json" > "$OUT/g2-selected-policy.sha256"

journalctl -k --since "$(cat "$OUT/maintenance/start-date.txt")" --no-pager \
    > "$OUT/kernel-journal.log" 2>&1 || true
rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
    "$OUT/kernel-journal.log" > "$OUT/kernel-signatures.txt" || true
if [[ -s "$OUT/kernel-signatures.txt" ]]; then
    printf 'HARDWARE_FAULT_SIGNATURES_DETECTED\n' >&2
    exit 1
fi
date --iso-8601=seconds > "$OUT/maintenance/complete-date.txt"
printf 'G2_DEVELOPMENT_COMPLETE selection_rc=%s out=%s\n' "$SELECTION_RC" "$OUT"
exit "$SELECTION_RC"
