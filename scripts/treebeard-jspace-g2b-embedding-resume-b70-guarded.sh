#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
PYTHON=/home/frosty40/nx2-venv/bin/python
OUT="$ROOT/results/treebeard-jspace-g2/development-g2b/20260715-163546"
MANIFEST="$WORKTREE/research/jspace-g2/g2b-development-response-manifest.json"
ROUTES="$OUT/routing/g2-development-g2b-routes.json"
ANCHORS="$ROOT/research/jspace-20260713/treebeard-jspace/anchors_qwen36_q5.json"
EMBED_MODEL=/home/frosty40/models/embeddings/nomic-embed-text-v1.5.Q8_0.gguf
PREREG="$WORKTREE/reports/treebeard-jspace-g2b-embedding-resume-20260715.md"
FROZEN_INPUTS="$OUT/resume-frozen-inputs.sha256"
ORIGINAL_ATTESTATION="$OUT/attestation/run-attestation.json"
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
CANDIDATE_PORT=${TREEBEARD_JSPACE_G2_PORT:-18093}
PREFLIGHT_ONLY=${TREEBEARD_JSPACE_G2_PREFLIGHT_ONLY:-0}
EXPECTED_BUILD=b9627-3fcf1c626
EXPECTED_ALIAS=turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3
EXPECTED_SERVER_SHA=211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff

MANIFEST_SHA=a0be16b107ae29579054146be6c57edc1bfd26b88f9f307276d2d0d1c520cace
ROUTES_SHA=89eef9dd91c5c847c5828a3a72e46824b571d8169218ea3a73140b2352d2c54d
ANCHORS_SHA=cab229a60bbcf1cd641070227bd4fd4c5bc8345605c36afe5c81850ff15ebb71
EMBED_MODEL_SHA=3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7
PREREG_SHA=a0a66a804e704b856613b49d9feb3a0525d40982628b4fac597d74d912d60976
FROZEN_INPUTS_SHA=39f8b5d95349ff5e4e9059c7e98f7465dc04bf5064813bd18dfbfdd06647ca67
ORIGINAL_ATTESTATION_SHA=0bad7e740d632e2a1c6e8d3ebfdb321519a32749140152528a354f546fb9305d
EMBED_SHA=869a9e758e321ebb52fc613b3db7c1e46bc5668e985b77db16411a55b00471c4
EVALUATE_SHA=9d9ab88f1a6615b043e1d5d079ae8e9fc8b2b9de0948fd17c3f31a64cb676337
SERVER_SHA=393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c
SYCL_SHA=0a0536fd5eba5caf6c8c4b00febe4320e868fe19bc99bd42767eb77d6c872e61
LIBLLAMA_SHA=c71031ef41abeb15a6b291c7abdd060a8452928e76ebfb3e4d83dc065cf7ba22

SCALE_NAMES=(eighth quarter half)
SERVER_PID=
SERVICE_STOPPED=0
BEFORE_NRESTARTS=
MAINT="$OUT/resume-maintenance"
LOGS="$OUT/resume-logs"
mkdir -p "$MAINT" "$LOGS" "$OUT/embeddings" "$OUT/evaluations"

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
    if [[ -f "$MAINT/start-date.txt" ]]; then
        journalctl -k --since "$(cat "$MAINT/start-date.txt")" --no-pager \
            > "$OUT/resume-kernel-journal.log" 2>&1 || true
        rg -n -i 'xe.*(reset|hang|fault)|drm.*(reset|hang|fault)|oom|kernel panic' \
            "$OUT/resume-kernel-journal.log" \
            > "$OUT/resume-kernel-signatures.txt" || true
        [[ ! -s "$OUT/resume-kernel-signatures.txt" ]] || ok=0
    fi
    if [[ "$SERVICE_STOPPED" == 1 ]]; then
        systemctl --user start "$SERVICE" || ok=0
        for _ in {1..240}; do
            if curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" \
                    > "$MAINT/restore-health.json" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        systemctl --user show "$SERVICE" \
            -p ActiveState -p SubState -p MainPID -p NRestarts --no-pager \
            > "$MAINT/restore-unit.txt" || ok=0
        systemctl --user is-active --quiet "$SERVICE" || ok=0
        curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" \
            > "$MAINT/restore-props.json" || ok=0
        jq -e --arg build "$EXPECTED_BUILD" --arg alias "$EXPECTED_ALIAS" \
            '.build_info == $build and .model_alias == $alias and
             .total_slots == 12 and .default_generation_settings.n_ctx == 262144' \
            "$MAINT/restore-props.json" >/dev/null || ok=0
        restored_pid=$(systemctl --user show "$SERVICE" -p MainPID --value)
        restored_exe=$(readlink -f "/proc/$restored_pid/exe" 2>/dev/null)
        if [[ -z "$restored_exe" ]]; then
            ok=0
        else
            sha256sum "$restored_exe" > "$MAINT/restore-server.sha256" || ok=0
            [[ $(awk '{print $1}' "$MAINT/restore-server.sha256") == \
                "$EXPECTED_SERVER_SHA" ]] || ok=0
        fi
        restored_restarts=$(systemctl --user show "$SERVICE" -p NRestarts --value)
        [[ "$restored_restarts" == "$BEFORE_NRESTARTS" ]] || ok=0
        curl -fsS --max-time 120 "http://127.0.0.1:$LIVE_PORT/completion" \
            -H 'Content-Type: application/json' \
            -d '{"prompt":"Return only the word healthy.","n_predict":8,"temperature":0,"top_k":1}' \
            > "$MAINT/restore-inference.json" || ok=0
        jq -e '.timings.predicted_n == 8' "$MAINT/restore-inference.json" \
            >/dev/null || ok=0
        curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/slots" \
            > "$MAINT/restore-slots.json" || ok=0
        curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/states" \
            > "$MAINT/restore-states.json" || ok=0
        jq -e '[.[] | select(.is_processing == true or .is_reserved == true)] | length == 0' \
            "$MAINT/restore-slots.json" >/dev/null || ok=0
        jq -e '(.families | length) == 0' "$MAINT/restore-states.json" \
            >/dev/null || ok=0
    fi
    date --iso-8601=seconds > "$MAINT/restore-date.txt"
    jq -n --argjson incoming_rc "$incoming_rc" --argjson restoration_ok "$ok" \
        --arg out "$OUT" '{schema:"treebeard.jspace.g2.resume-status.v1",
        incoming_rc:$incoming_rc,restoration_ok:($restoration_ok == 1),out:$out}' \
        > "$OUT/resume-run-status.json"
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

add_response() {
    local -n target=$1
    local path=$2
    target+=(--responses "$path" --responses-sha256 "$(sha "$path")")
}

build_response_args() {
    local policy=$1 scale_name=$2 target_name=$3
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
    [[ "$rc" == 0 || "$rc" == 2 ]]
}

printf 'G2_RESUME_PREFLIGHT out=%s\n' "$OUT"
require_sha "$MANIFEST" "$MANIFEST_SHA" manifest
require_sha "$ROUTES" "$ROUTES_SHA" routes
require_sha "$ANCHORS" "$ANCHORS_SHA" anchors
require_sha "$EMBED_MODEL" "$EMBED_MODEL_SHA" embedding-model
require_sha "$PREREG" "$PREREG_SHA" resume-preregistration
require_sha "$FROZEN_INPUTS" "$FROZEN_INPUTS_SHA" frozen-input-list
require_sha "$ORIGINAL_ATTESTATION" "$ORIGINAL_ATTESTATION_SHA" original-attestation
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-embed.py" "$EMBED_SHA" embedding-client
require_sha "$WORKTREE/scripts/treebeard-jspace-g2-evaluate.py" "$EVALUATE_SHA" evaluator
require_sha "$BUILD/bin/llama-server" "$SERVER_SHA" server
require_sha "$BUILD/bin/libggml-sycl.so" "$SYCL_SHA" sycl-runtime
require_sha "$BUILD/bin/libllama.so" "$LIBLLAMA_SHA" llama-runtime
sha256sum -c "$FROZEN_INPUTS" >/dev/null
[[ $(wc -l < "$FROZEN_INPUTS") == 38 ]]
jq -e -s 'length == 38 and all(.status == "complete") and
    (map(.audit.failures // 0) | add) == 0' \
    "$OUT"/responses/*.json "$OUT"/judgments/*.json >/dev/null
mapfile -t ACTIVE_AXES < <(
    jq -r '.rows[] | select(.active == true) | .routed_axis' "$ROUTES" | sort -u)
[[ "${ACTIVE_AXES[*]}" == "anger disgust fear joy sadness surprise" ]]

systemctl --user is-active --quiet "$SERVICE"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/health" \
    > "$MAINT/before-health.json"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" \
    > "$MAINT/before-props.json"
jq -e --arg build "$EXPECTED_BUILD" --arg alias "$EXPECTED_ALIAS" \
    '.build_info == $build and .model_alias == $alias and .total_slots == 12 and
     .default_generation_settings.n_ctx == 262144' "$MAINT/before-props.json" \
    >/dev/null
systemctl --user show "$SERVICE" -p ActiveState -p SubState -p MainPID -p NRestarts \
    --no-pager > "$MAINT/before-unit.txt"
BEFORE_NRESTARTS=$(systemctl --user show "$SERVICE" -p NRestarts --value)
before_pid=$(systemctl --user show "$SERVICE" -p MainPID --value)
before_exe=$(readlink -f "/proc/$before_pid/exe")
require_sha "$before_exe" "$EXPECTED_SERVER_SHA" production-server

sha256sum "$0" > "$OUT/resume-runner.sha256"
jq -n --arg started "$(date --iso-8601=seconds)" \
    --arg prereg "$PREREG_SHA" --arg frozen "$FROZEN_INPUTS_SHA" \
    --arg original "$ORIGINAL_ATTESTATION_SHA" \
    --arg runner "$(awk '{print $1}' "$OUT/resume-runner.sha256")" \
    --arg head "$(git -C "$WORKTREE" rev-parse HEAD)" \
    '{schema:"treebeard.jspace.g2.resume-attestation.v1",started:$started,
      preregistration_sha256:$prereg,frozen_inputs_sha256:$frozen,
      original_attestation_sha256:$original,runner_sha256:$runner,source_head:$head}' \
    > "$OUT/resume-attestation.json"
date --iso-8601=seconds > "$MAINT/start-date.txt"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    printf 'G2_RESUME_PREFLIGHT_COMPLETE out=%s\n' "$OUT"
    exit 0
fi

SERVICE_STOPPED=1
systemctl --user stop "$SERVICE"
for _ in {1..60}; do
    systemctl --user is-active --quiet "$SERVICE" || break
    sleep 1
done
systemctl --user is-active --quiet "$SERVICE" && exit 1
printf 'SERVICE_STOPPED_GUARDED\n'

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

if ss -ltn "( sport = :$CANDIDATE_PORT )" | rg -q LISTEN; then
    printf 'CANDIDATE_PORT_BUSY port=%s\n' "$CANDIDATE_PORT" >&2
    exit 1
fi
env \
    GGML_SYCL_ENABLE_FUSION=1 \
    GGML_SYCL_DISABLE_GRAPH=1 \
    GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
    GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
    taskset -c 0-10,12-15 "$BUILD/bin/llama-server" \
    -m "$EMBED_MODEL" -ngl 99 --no-op-offload -kvo -t 15 \
    -c 8192 -np 16 -b 256 -ub 256 \
    --host 127.0.0.1 --port "$CANDIDATE_PORT" --spec-type none \
    -a treebeard-g2-nomic-v1.5-q8 --embedding --pooling mean \
    > "$LOGS/server-embedding.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$LOGS/server-embedding.pid"
for _ in {1..300}; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$CANDIDATE_PORT/health" \
            > "$LOGS/server-embedding-health.json" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -200 "$LOGS/server-embedding.log" >&2 || true
        exit 1
    fi
    sleep 1
done
curl -fsS --max-time 5 "http://127.0.0.1:$CANDIDATE_PORT/props" \
    > "$LOGS/server-embedding-props.json"
jq -e '.model_alias == "treebeard-g2-nomic-v1.5-q8" and .total_slots == 16 and
    .default_generation_settings.n_ctx == 512' "$LOGS/server-embedding-props.json" \
    >/dev/null

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
[[ "$SELECTION_RC" == 0 || "$SELECTION_RC" == 2 ]]
sha256sum "$OUT/g2-selected-policy.json" > "$OUT/g2-selected-policy.sha256"
date --iso-8601=seconds > "$MAINT/complete-date.txt"
printf 'G2_RESUME_COMPLETE selection_rc=%s out=%s\n' "$SELECTION_RC" "$OUT"
exit "$SELECTION_RC"
