#!/usr/bin/env python3

"""Capture frozen reference/response embeddings for G2 diagnostics."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def request(port, method, path, body=None, timeout=300.0):
    data = None if body is None else json.dumps(
        body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc


def load_documents(args):
    require_sha(args.manifest, args.manifest_sha256, "G2 response manifest")
    require_sha(args.routes, args.routes_sha256, "G2 routes")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    if manifest.get("schema") != "treebeard.jspace.g2.response-set.v1" or \
            routes.get("schema") != "treebeard.jspace.g2.routes.v1":
        raise ValueError("unexpected manifest or routes schema")
    if routes.get("inputs", {}).get("response_manifest_sha256") != args.manifest_sha256:
        raise ValueError("routes do not attest the response manifest")
    records = {}
    response_inputs = []
    for index, path in enumerate(args.responses):
        expected = args.responses_sha256[index]
        require_sha(path, expected, f"generated response file {index}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != "treebeard.jspace.g2.generated-responses.v1" or \
                document.get("status") != "complete" or \
                document.get("inputs", {}).get("manifest_sha256") != args.manifest_sha256 or \
                document.get("inputs", {}).get("routes_sha256") != args.routes_sha256:
            raise ValueError(f"unexpected or unbound response document: {path}")
        for record in document["records"]:
            key = (record["sample_id"], int(record["seed"]), record["arm"])
            if key in records:
                raise ValueError(f"duplicate generated record: {key}")
            records[key] = record
        response_inputs.append({"path": str(path), "sha256": expected})
    return manifest, routes, records, response_inputs


def collect_texts(manifest, records):
    rows = {row["sample_id"]: row for row in manifest["rows"]}
    pairs = sorted({
        (sample_id, seed)
        for sample_id, seed, arm in records
        if arm == "control" and rows[sample_id]["scope"] == "dialogue_response" and
        (sample_id, seed, "candidate") in records
    })
    entries = []
    text_to_index = {}

    def add(text):
        if text not in text_to_index:
            text_to_index[text] = len(entries)
            entries.append(text)
        return text_to_index[text]

    mappings = []
    for sample_id, seed in pairs:
        row = rows[sample_id]
        control = records[(sample_id, seed, "control")]["response"]["content"]
        candidate = records[(sample_id, seed, "candidate")]["response"]["content"]
        mappings.append({
            "sample_id": sample_id,
            "seed": seed,
            "reference_index": add(row["reference"]),
            "control_index": add(control),
            "candidate_index": add(candidate),
        })
    return entries, mappings


def embed_batches(port, texts, batch_size, timeout):
    vectors = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = request(port, "POST", "/v1/embeddings", {
            "input": batch,
            "encoding_format": "float",
        }, timeout)
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError("embedding endpoint returned the wrong batch size")
        for offset, row in enumerate(sorted(data, key=lambda value: value["index"])):
            vector = row.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("embedding endpoint returned an invalid vector")
            vectors[start + offset] = vector
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise RuntimeError("embedding matrix is invalid")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise RuntimeError("embedding matrix contains a zero vector")
    return matrix / norms[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--routes-sha256", required=True)
    parser.add_argument("--responses", action="append", type=Path, required=True)
    parser.add_argument("--responses-sha256", action="append", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--raw-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.responses) != len(args.responses_sha256):
        parser.error("--responses and --responses-sha256 counts differ")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    manifest, routes, records, response_inputs = load_documents(args)
    texts, mappings = collect_texts(manifest, records)
    if not mappings:
        raise ValueError("no dialogue response pairs are available for embedding")
    health = request(args.port, "GET", "/health", timeout=args.timeout)
    props = request(args.port, "GET", "/props", timeout=args.timeout)
    matrix = embed_batches(args.port, texts, args.batch_size, args.timeout)
    results = []
    for mapping in mappings:
        reference = matrix[mapping["reference_index"]]
        control = float(reference @ matrix[mapping["control_index"]])
        candidate = float(reference @ matrix[mapping["candidate_index"]])
        results.append({
            "sample_id": mapping["sample_id"],
            "seed": mapping["seed"],
            "reference_cosine_control": control,
            "reference_cosine_candidate": candidate,
            "candidate_minus_control": candidate - control,
        })

    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.raw_out,
        schema=np.array("treebeard.jspace.g2.embeddings.v1"),
        texts_sha256=np.asarray([
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ]),
        vectors=matrix.astype("<f4"),
    )
    raw_sha = sha256_file(args.raw_out)
    output = {
        "schema": "treebeard.jspace.g2.embedding-scores.v1",
        "status": "complete",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "routes_sha256": args.routes_sha256,
            "response_files": response_inputs,
        },
        "embedding": {
            "model_alias": props.get("model_alias"),
            "build_info": props.get("build_info"),
            "health": health,
            "normalization": "client_l2",
            "unique_texts": len(texts),
            "dimension": int(matrix.shape[1]),
            "raw_path": str(args.raw_out),
            "raw_sha256": raw_sha,
            "raw_shape": list(matrix.shape),
        },
        "audit": {"pairs": len(results)},
        "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "pairs": len(results),
        "raw_sha256": raw_sha,
        "out": str(args.out),
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
