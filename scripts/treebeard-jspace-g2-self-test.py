#!/usr/bin/env python3

"""Model-independent end-to-end self-test for the J-Space G2 pipeline."""

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass

    def send(self, value, status=200):
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send({"status": "ok"})
        elif self.path == "/props":
            self.send({
                "build_info": "g2-self-test",
                "model_alias": "g2-self-test-model",
                "total_slots": 2,
                "default_generation_settings": {"n_ctx": 1024},
            })
        else:
            self.send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/apply-template":
            user = body["messages"][-1]["content"]
            self.send({"prompt": "<chat>" + user + "</chat>"})
        elif self.path == "/tokenize":
            self.send({"tokens": [1, 2, 3]})
        elif self.path == "/completion":
            candidate = float(body["jspace_control_scale"]) > 0.0
            self.send({
                "content": "candidate helpful response" if candidate else "control generic response",
                "tokens": [20, 21] if candidate else [10, 11],
                "stop": True,
                "stop_type": "eos",
                "stopping_word": "",
                "truncated": False,
                "timings": {"predicted_n": 2},
            })
        elif self.path == "/v1/chat/completions":
            prompt = body["messages"][-1]["content"]
            response_a = prompt.split(
                "<response_a>\n", 1)[1].split("\n</response_a>", 1)[0]
            a_candidate = response_a == "candidate helpful response"
            flags = {
                "factual_error": False,
                "format_failure": False,
                "task_failure": False,
                "refusal_pathology": False,
                "repetition_pathology": False,
                "malformed": False,
            }
            judgment = {
                "winner": "A" if a_candidate else "B",
                "A": flags,
                "B": flags,
                "reason_code": "helpfulness",
            }
            self.send({
                "choices": [{
                    "message": {"role": "assistant", "content": json.dumps(judgment)},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 1},
            })
        elif self.path == "/v1/embeddings":
            rows = []
            for index, text in enumerate(body["input"]):
                if "control generic" in text:
                    vector = [0.0, 1.0]
                else:
                    vector = [1.0, 0.0]
                rows.append({"index": index, "embedding": vector})
            self.send({"data": rows})
        else:
            self.send({"error": "not found"}, 404)


def run(command, expected=(0,)):
    result = subprocess.run(
        [sys.executable, *command], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode not in expected:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(map(str, command))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with tempfile.TemporaryDirectory(prefix="treebeard-g2-self-test-") as directory:
            temp = Path(directory)
            rows = []
            route_rows = []
            for index in range(2):
                sample_id = f"self-test-{index}"
                text = f"User input {index} needs a helpful response."
                rows.append({
                    "sample_id": sample_id,
                    "split": "development",
                    "scope": "dialogue_response",
                    "source": "empatheticdialogues",
                    "source_split": "train",
                    "source_id": sample_id,
                    "conversation_id": sample_id,
                    "source_label": "joyful",
                    "label": "joy",
                    "anchor_echo": False,
                    "anchor_echo_tokens": [],
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "reference_sha256": hashlib.sha256(b"candidate helpful response").hexdigest(),
                    "text": text,
                    "reference": "candidate helpful response",
                })
                route_rows.append({
                    "sample_id": sample_id,
                    "source": "empatheticdialogues",
                    "label": "joy",
                    "scope": "dialogue_response",
                    "probabilities": {
                        "sadness": 0.01, "surprise": 0.01, "joy": 0.9,
                        "disgust": 0.01, "fear": 0.01, "anger": 0.01,
                        "neutral": 0.04,
                    },
                    "neutral_probability": 0.04,
                    "maximum_probability": 0.9,
                    "top_label": "joy",
                    "affect_gate": True,
                    "retained": True,
                    "routed_axis": "joy",
                    "active": True,
                    "decision": "actuate",
                })
            manifest_path = temp / "manifest.json"
            manifest_sha = write_json(manifest_path, {
                "schema": "treebeard.jspace.g2.response-set.v1",
                "status": "frozen_before_response_generation",
                "policy": {"kind": "development"},
                "rows": rows,
            })
            routes_path = temp / "routes.json"
            routes_sha = write_json(routes_path, {
                "schema": "treebeard.jspace.g2.routes.v1",
                "status": "frozen_g1_v5_applied",
                "inputs": {"response_manifest_sha256": manifest_sha},
                "rows": route_rows,
            })
            vector_path = temp / "vector.gguf"
            vector_path.write_bytes(b"self-test-vector")
            vector_sha = sha256_file(vector_path)
            attestation_path = temp / "attestation.json"
            attestation_sha = write_json(attestation_path, {
                "schema": "treebeard.jspace.g2.run-attestation.v1",
                "self_test": True,
            })

            common = [
                str(SCRIPTS / "treebeard-jspace-g2-generate.py"),
                "--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                "--routes", str(routes_path), "--routes-sha256", routes_sha,
                "--port", str(port), "--policy", "engage_curiosity",
                "--vector-axis", "curiosity", "--seed", "1709",
                "--vector", str(vector_path), "--vector-sha256", vector_sha,
                "--attestation", str(attestation_path),
                "--attestation-sha256", attestation_sha,
            ]
            control_path = temp / "control.json"
            run(common + [
                "--arm", "control", "--scale", "0", "--out", str(control_path)])
            control_sha = sha256_file(control_path)
            candidate_path = temp / "candidate.json"
            run(common + [
                "--arm", "candidate", "--scale", "0.008641079027104324",
                "--include-noops", "--out", str(candidate_path)])
            candidate_sha = sha256_file(candidate_path)

            response_args = [
                "--responses", str(control_path), "--responses-sha256", control_sha,
                "--responses", str(candidate_path), "--responses-sha256", candidate_sha,
            ]
            judgments_path = temp / "judgments.json"
            run([
                str(SCRIPTS / "treebeard-jspace-g2-judge.py"),
                "--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                "--routes", str(routes_path), "--routes-sha256", routes_sha,
                *response_args, "--port", str(port), "--out", str(judgments_path),
            ])
            judgments_sha = sha256_file(judgments_path)
            judgments = json.loads(judgments_path.read_text())
            if any(row["preference"] != "candidate" for row in judgments["records"]):
                raise AssertionError("order-swapped judge mapping failed")

            embeddings_path = temp / "embeddings.json"
            embedding_raw_path = temp / "embeddings.npz"
            run([
                str(SCRIPTS / "treebeard-jspace-g2-embed.py"),
                "--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                "--routes", str(routes_path), "--routes-sha256", routes_sha,
                *response_args, "--port", str(port),
                "--raw-out", str(embedding_raw_path), "--out", str(embeddings_path),
            ])
            embeddings_sha = sha256_file(embeddings_path)
            embedding_raw_sha = sha256_file(embedding_raw_path)

            anchors_path = temp / "anchors.json"
            anchors_sha = write_json(anchors_path, {
                "schema": "treebeard.jspace.anchors.v0",
                "axes": {"joy": [{"word": "happy"}]},
            })
            evaluation_path = temp / "evaluation.json"
            run([
                str(SCRIPTS / "treebeard-jspace-g2-evaluate.py"), "evaluate",
                "--mode", "development",
                "--manifest", str(manifest_path), "--manifest-sha256", manifest_sha,
                "--routes", str(routes_path), "--routes-sha256", routes_sha,
                *response_args,
                "--judgments", str(judgments_path), "--judgments-sha256", judgments_sha,
                "--embeddings", str(embeddings_path), "--embeddings-sha256", embeddings_sha,
                "--embedding-raw", str(embedding_raw_path),
                "--embedding-raw-sha256", embedding_raw_sha,
                "--anchors", str(anchors_path), "--anchors-sha256", anchors_sha,
                "--out", str(evaluation_path),
            ], expected=(2,))
            evaluation = json.loads(evaluation_path.read_text())
            if evaluation["metrics"]["active_preference"]["score"] != 1.0 or \
                    evaluation["metrics"]["changed_active_token_rate"] != 1.0 or \
                    evaluation["metrics"]["mean_reference_cosine_delta"] <= 0.0:
                raise AssertionError("evaluation metrics self-test failed")
            if evaluation["gates"]["active_rows_min_20"] is not False:
                raise AssertionError("development minimum-row gate self-test failed")

            report_paths = []
            report_shas = []
            scales = (0.008641079027104324, 0.017282158054208648, 0.034564316108417296)
            for policy_index, policy in enumerate(("engage_curiosity", "match_route")):
                for scale_index, scale in enumerate(scales):
                    synthetic = json.loads(json.dumps(evaluation))
                    synthetic["status"] = "pass"
                    synthetic["all_gates_pass"] = True
                    synthetic["candidate"] = {
                        "policy": policy,
                        "scale": scale,
                        "id": f"{policy}:{scale:.18g}",
                    }
                    synthetic["metrics"]["active_preference"]["score"] = (
                        0.60 + 0.01 * policy_index + 0.001 * scale_index)
                    path = temp / f"report-{policy_index}-{scale_index}.json"
                    report_paths.append(path)
                    report_shas.append(write_json(path, synthetic))
            selected_path = temp / "selected.json"
            select_args = [str(SCRIPTS / "treebeard-jspace-g2-evaluate.py"), "select"]
            for path, digest in zip(report_paths, report_shas):
                select_args.extend(["--report", str(path), "--report-sha256", digest])
            select_args.extend(["--out", str(selected_path)])
            run(select_args)
            selected = json.loads(selected_path.read_text())
            if selected["selected"]["policy"] != "match_route" or \
                    not math_isclose(selected["selected"]["scale"], scales[-1]):
                raise AssertionError("six-candidate selection self-test failed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(json.dumps({"ok": True, "schema": "treebeard.jspace.g2.pipeline-self-test.v1"}))
    return 0


def math_isclose(left, right):
    return abs(left - right) <= 1e-15


if __name__ == "__main__":
    raise SystemExit(main())
