import hashlib
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from utils import *


server = ServerPreset.tinyllama2()

PREFIX = "What is the capital of France?"


def append_manifest_mutation(path, revision, record_type, owner, digest, retention_class):
    payload = owner.encode() + digest.encode() + retention_class.encode()
    total = 40 + len(payload) + hashlib.sha256().digest_size
    record = struct.pack(
        "<IIQB7xIIII",
        total,
        0x31464E4D,
        revision,
        record_type,
        len(owner),
        len(digest),
        len(retention_class),
        0,
    ) + payload
    checksum = hashlib.sha256(
        b"turbo-statetree-manifest-record-v1\0\0" + record
    ).digest()
    with path.open("ab") as manifest:
        manifest.write(record + checksum)
        manifest.flush()
        os.fsync(manifest.fileno())


def append_manifest_publish_intent(path, revision, owner, digest, retention_class):
    append_manifest_mutation(
        path, revision, 4, owner, digest, retention_class
    )


def append_manifest_publish_advance_intent(
    path,
    revision,
    owner,
    digest,
    retention_class,
    head_name,
    expected_generation,
    expected_digest,
    record_type=12,
):
    payload = (
        owner.encode()
        + digest.encode()
        + retention_class.encode()
        + head_name.encode()
        + expected_digest.encode()
    )
    total = 40 + 4 + 8 + len(payload) + hashlib.sha256().digest_size
    record = struct.pack(
        "<IIQB7xIIII",
        total,
        0x31464E4D,
        revision,
        record_type,
        len(owner),
        len(digest),
        len(retention_class),
        len(head_name),
    ) + struct.pack("<IQ", len(expected_digest), expected_generation) + payload
    checksum = hashlib.sha256(
        b"turbo-statetree-manifest-record-v1\0\0" + record
    ).digest()
    with path.open("ab") as manifest:
        manifest.write(record + checksum)
        manifest.flush()
        os.fsync(manifest.fileno())


def append_legacy_empty_manifest_checkpoint(path, revision):
    record = struct.pack(
        "<IIQB7xIIII",
        40 + hashlib.sha256().digest_size,
        0x31464E4D,
        revision,
        3,
        0,
        0,
        0,
        0,
    )
    checksum = hashlib.sha256(
        b"turbo-statetree-manifest-record-v1\0\0" + record
    ).digest()
    with path.open("ab") as manifest:
        manifest.write(record + checksum)
        manifest.flush()
        os.fsync(manifest.fileno())


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.kv_unified = True
    server.n_slots = 4
    server.server_slots = True
    server.no_cache_idle_slots = True
    server.temperature = 0.0
    if model_file := os.environ.get("LLAMA_SERVER_TEST_MODEL"):
        server.model_hf_repo = None
        server.model_file = model_file
        server.offline = True
    yield
    server.stop()


def completion(
    prompt: str | list[int],
    id_slot: int | None = None,
    timeout: float | None = None,
    *,
    n_predict: int = 1,
    return_tokens: bool = False,
    n_probs: int = 0,
    cache_prompt: bool = True,
    fork_id: int | None = None,
    state_id: int | None = None,
    node_id: int | None = None,
):
    data = {
        "prompt": prompt,
        "cache_prompt": cache_prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "return_tokens": return_tokens,
        "n_probs": n_probs,
    }
    if id_slot is not None:
        data["id_slot"] = id_slot
    if fork_id is not None:
        data["fork_id"] = fork_id
    if state_id is not None:
        data["state_id"] = state_id
    if node_id is not None:
        data["node_id"] = node_id
    return server.make_request("POST", "/completion", data=data, timeout=timeout)


def metric_value(body: str, name: str) -> float:
    prefix = f"llamacpp:{name} "
    for line in body.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix):])
    raise AssertionError(f"missing metric {name}")


def wait_for_metric(name: str, expected: float, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metrics = server.make_request("GET", "/metrics")
        assert metrics.status_code == 200
        if metric_value(metrics.body, name) == expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"metric {name} did not reach {expected}")


def wait_for_slot_processing(id_slot: int, expected: bool, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        slots = server.make_request("GET", "/slots")
        assert slots.status_code == 200
        if slots.body[id_slot]["is_processing"] is expected:
            return
        time.sleep(0.05)
    raise AssertionError(f"slot {id_slot} processing state did not reach {expected}")


def test_slot_fork_reserves_and_retains_branches():
    global server
    server.start()

    prefix = completion(PREFIX, 0)
    assert prefix.status_code == 200

    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    assert set(fork.body) == {
        "id_slot",
        "state_id",
        "node_id",
        "parent_node_id",
        "fork_id",
        "nodes",
        "destinations",
        "n_destinations",
        "n_tokens",
        "timings",
    }
    assert fork.body["id_slot"] == 0
    assert fork.body["fork_id"] >= 0
    assert fork.body["destinations"] == [1, 2]
    assert fork.body["n_destinations"] == 2
    assert fork.body["n_tokens"] > 0
    assert fork.body["timings"]["fork_ms"] >= 0

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    for id_slot in (0, 1, 2):
        assert slots.body[id_slot]["is_reserved"] is True
        assert slots.body[id_slot]["fork_source_id"] == 0
    assert slots.body[3]["is_reserved"] is False

    for invalid_id in (99, -4):
        invalid_erase = server.make_request("POST", f"/slots/{invalid_id}?action=erase")
        assert invalid_erase.status_code == 400
    slots_after_invalid_erase = server.make_request("GET", "/slots")
    assert slots_after_invalid_erase.status_code == 200
    for id_slot in (0, 1, 2):
        assert slots_after_invalid_erase.body[id_slot]["is_reserved"] is True

    automatic = completion("An unrelated prompt")
    assert automatic.status_code == 200
    assert automatic.body["id_slot"] == 3

    branch_one_prompt = PREFIX + " Give a concise answer."
    branch_one = completion(branch_one_prompt, 1)
    assert branch_one.status_code == 200
    assert branch_one.body["timings"]["cache_n"] > 0

    branch_one_reuse = completion(branch_one_prompt, 1)
    assert branch_one_reuse.status_code == 200
    assert branch_one_reuse.body["timings"]["cache_n"] > 0

    source_branch = completion(PREFIX + " Answer as a geographer.", 0)
    assert source_branch.status_code == 200
    assert source_branch.body["timings"]["cache_n"] > 0

    branch_two = completion(PREFIX + " Answer in one sentence.", 2)
    assert branch_two.status_code == 200
    assert branch_two.body["timings"]["cache_n"] > 0


def test_slot_fork_validation_is_atomic():
    global server
    server.start()

    source = completion(PREFIX, 0)
    destination_prompt = "Write one word about the moon."
    destination = completion(destination_prompt, 1)
    assert source.status_code == 200
    assert destination.status_code == 200

    invalid = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 99]},
    )
    assert invalid.status_code == 400

    destination_reuse = completion(destination_prompt + " Be literal.", 1)
    assert destination_reuse.status_code == 200
    assert destination_reuse.body["timings"]["cache_n"] > 0

    invalid_requests = [
        ({"destinations": []}, 400),
        ({"destinations": [0]}, 400),
        ({"destinations": [4]}, 400),
        ({"destinations": [-1]}, 400),
        ({"destinations": [1, 5]}, 400),
        ({"destinations": [2, 2]}, 400),
        ({"destinations": ["2"]}, 400),
        ({}, 400),
    ]
    for data, status_code in invalid_requests:
        response = server.make_request(
            "POST",
            "/slots/0?action=fork",
            data=data,
        )
        assert response.status_code == status_code

    empty_source = server.make_request(
        "POST",
        "/slots/2?action=fork",
        data={"destinations": [3]},
    )
    assert empty_source.status_code == 400

    invalid_source = server.make_request(
        "POST",
        "/slots/99?action=fork",
        data={"destinations": [2]},
    )
    assert invalid_source.status_code == 400

    invalid_source_text = server.make_request(
        "POST",
        "/slots/0junk?action=fork",
        data={"destinations": [2]},
    )
    assert invalid_source_text.status_code == 400


def test_slot_fork_erase_releases_one_branch_without_slot_save_path():
    global server
    server.n_slots = 3
    server.server_metrics = True
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200

    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    erased_node = next(node for node in fork.body["nodes"] if node["id_slot"] == 1)

    with ThreadPoolExecutor(max_workers=1) as executor:
        deferred = executor.submit(completion, "This request must use the released branch.", None, 10)
        wait_for_metric("requests_deferred", 1)
        assert not deferred.done()

        erase = server.make_request(
            "POST",
            f"/nodes/{erased_node['node_id']}?action=erase",
            data={},
        )
        assert erase.status_code == 200
        assert erase.body["id_slot"] == 1
        assert erase.body["node_id"] == erased_node["node_id"]
        assert erase.body["n_erased"] > 0

        automatic = deferred.result(timeout=10)
        assert automatic.status_code == 200
        assert automatic.body["id_slot"] == 1
        wait_for_metric("requests_deferred", 0)

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[0]["is_reserved"] is True
    assert slots.body[1]["is_reserved"] is False
    assert slots.body[1]["fork_source_id"] == -1
    assert slots.body[2]["is_reserved"] is True

    sibling_prompt = PREFIX + " Answer with only the city name."
    sibling_reuse = completion(sibling_prompt, 2)
    assert sibling_reuse.status_code == 200
    assert sibling_reuse.body["timings"]["cache_n"] > 0


def test_slot_erase_and_restore_clear_checkpoint_metadata(tmp_path):
    global server
    server.slot_save_path = str(tmp_path)
    server.start()

    source = completion(PREFIX, 0)
    destination = completion("Give one fact about Saturn.", 1)
    assert source.status_code == 200
    assert destination.status_code == 200

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    if os.environ.get("LLAMA_SERVER_TEST_MODEL"):
        assert slots.body[1]["n_prompt_checkpoints"] > 0

    saved = server.make_request(
        "POST",
        "/slots/0?action=save",
        data={"filename": "checkpoint-clear.bin"},
    )
    assert saved.status_code == 200

    restored = server.make_request(
        "POST",
        "/slots/1?action=restore",
        data={"filename": "checkpoint-clear.bin"},
    )
    assert restored.status_code == 200
    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[1]["n_prompt_tokens"] > 0
    assert slots.body[1]["n_prompt_checkpoints"] == 0

    erased = server.make_request("POST", "/slots/0?action=erase")
    assert erased.status_code == 200
    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[0]["n_prompt_tokens"] == 0
    assert slots.body[0]["n_prompt_checkpoints"] == 0


def test_slot_fork_failed_restore_releases_destination(tmp_path):
    global server
    server.n_slots = 3
    server.slot_save_path = f"{tmp_path}/"
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200

    restore = server.make_request(
        "POST",
        "/slots/1?action=restore",
        data={"filename": "missing-slot.bin"},
    )
    assert restore.status_code == 400

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[0]["is_reserved"] is True
    assert slots.body[1]["is_reserved"] is False
    assert slots.body[1]["fork_source_id"] == -1
    assert slots.body[1].get("n_prompt_tokens", 0) == 0
    assert slots.body[2]["is_reserved"] is True


def test_deferred_slot_actions_keep_draining_after_release(tmp_path):
    global server
    server.n_slots = 1
    server.server_metrics = True
    server.slot_save_path = f"{tmp_path}/"
    server.start()

    seeded = completion(PREFIX, 0)
    assert seeded.status_code == 200

    def run_long_completion():
        return server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": PREFIX + " Give a detailed answer.",
                "cache_prompt": True,
                "id_slot": 0,
                "n_predict": 256,
                "temperature": 0.0,
                "ignore_eos": True,
                "stop": [],
            },
            timeout=30,
        )

    def save(filename: str):
        return server.make_request(
            "POST",
            "/slots/0?action=save",
            data={"filename": filename},
            timeout=30,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        branch = executor.submit(run_long_completion)
        wait_for_slot_processing(0, True)
        first = executor.submit(save, "deferred-first.bin")
        second = executor.submit(save, "deferred-second.bin")
        wait_for_metric("requests_deferred", 2)

        assert branch.result(timeout=30).status_code == 200
        first_result = first.result(timeout=10)
        second_result = second.result(timeout=10)
        assert first_result.status_code == 200
        assert second_result.status_code == 200
        assert first_result.body["n_written"] > 0
        assert second_result.body["n_written"] == first_result.body["n_written"]
        wait_for_metric("requests_deferred", 0)


def test_slot_fork_metrics_report_reserved_slots():
    global server
    server.n_slots = 4
    server.server_metrics = True
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert "llamacpp:requests_reserved 3" in metrics.body
    assert "llamacpp:requests_idle 1" in metrics.body


def test_statetree_zero_use_lease_expires_autonomously():
    global server
    server.n_slots = 2
    server.server_metrics = True
    server.statetree_lease_ms = 250
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200
    fork_id = fork.body["fork_id"]
    assert fork.body["retention"]["lease_remaining_ms"] > 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        deferred = executor.submit(completion, "Run after autonomous expiry.", None, 10)
        wait_for_metric("requests_deferred", 1)
        result = deferred.result(timeout=5)
        assert result.status_code == 200

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert not any(row["is_reserved"] for row in slots.body)
    stale = server.make_request(
        "POST",
        "/slots/0?action=commit",
        data={"fork_id": fork_id},
    )
    assert stale.status_code == 503

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert metric_value(metrics.body, "statetree_expired_total") == 1

    states = server.make_request("GET", "/states")
    assert states.status_code == 200
    assert states.body["states"] == []
    assert [entry["event"] for entry in states.body["journal"]] == ["fork", "expire"]


def test_statetree_generation_fenced_access_renews_family():
    global server
    server.n_slots = 3
    server.statetree_lease_ms = 500
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    fork_id = fork.body["fork_id"]

    missing_slot = completion(PREFIX + " Missing slot.", None, fork_id=fork_id)
    assert missing_slot.status_code == 400
    aliased_slot = completion(PREFIX + " Aliased slot.", 4, fork_id=fork_id)
    assert aliased_slot.status_code == 400
    missing = completion(PREFIX + " Missing fence.", 1)
    assert missing.status_code == 400
    stale = completion(PREFIX + " Stale fence.", 1, fork_id=fork_id + 1)
    assert stale.status_code == 503

    max_int64 = completion(PREFIX + " Wide stale fence.", 1, fork_id=(1 << 63) - 1)
    assert max_int64.status_code == 503

    time.sleep(0.3)
    renewed = server.make_request(
        "POST",
        "/slots/1?action=renew",
        data={"fork_id": fork_id},
    )
    assert renewed.status_code == 200
    assert renewed.body["members"] == [0, 1, 2]
    assert renewed.body["retention"]["lease_remaining_ms"] > 0

    time.sleep(0.3)
    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert all(slots.body[id_slot]["is_reserved"] for id_slot in (0, 1, 2))

    continued = completion(PREFIX + " Safely continue.", 1, fork_id=fork_id)
    assert continued.status_code == 200
    assert continued.body["timings"]["cache_n"] > 0

    missing_erase = server.make_request("POST", "/slots/1?action=erase", data={})
    assert missing_erase.status_code == 400
    stale_erase = server.make_request(
        "POST",
        "/slots/1?action=erase",
        data={"fork_id": fork_id + 1},
    )
    assert stale_erase.status_code == 503
    erased = server.make_request(
        "POST",
        "/slots/1?action=erase",
        data={"fork_id": fork_id},
    )
    assert erased.status_code == 200


def test_statetree_logical_state_survives_winner_migration_and_refork():
    global server
    server.n_slots = 3
    server.statetree_lease_ms = 5_000
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    state_id = fork.body["state_id"]
    fork_id = fork.body["fork_id"]
    first_nodes = fork.body["nodes"]
    first_nodes_by_slot = {node["id_slot"]: node for node in first_nodes}
    winner_node_id = first_nodes_by_slot[1]["node_id"]
    assert state_id >= 0
    assert fork.body["node_id"] == first_nodes_by_slot[0]["node_id"]
    assert fork.body["parent_node_id"] == -1
    assert set(first_nodes_by_slot) == {0, 1, 2}
    assert len({node["node_id"] for node in first_nodes}) == 3
    assert {node["parent_node_id"] for node in first_nodes} == {-1}

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert {slots.body[id_slot]["state_id"] for id_slot in (0, 1, 2)} == {state_id}
    assert {
        slots.body[id_slot]["node_id"]
        for id_slot in (0, 1, 2)
    } == {node["node_id"] for node in first_nodes}

    states = server.make_request("GET", "/states")
    assert states.status_code == 200
    assert states.body["journal_capacity"] == 1024
    assert len(states.body["states"]) == 1
    logical = states.body["states"][0]
    assert logical["state_id"] == state_id
    assert logical["fork_id"] == fork_id
    assert logical["source_slot"] == 0
    assert logical["canonical_slot"] is None
    assert logical["canonical_node_id"] is None
    assert logical["status"] == "forked"
    assert logical["members"] == [0, 1, 2]
    assert logical["heads"] == first_nodes
    assert logical["n_members"] == 3
    assert logical["n_tokens"] > 0
    assert logical["active"] is False
    assert logical["retention_touch"] > 0
    assert 0 < logical["lease_remaining_ms"] <= 5_000
    assert [entry["event"] for entry in states.body["journal"]] == ["fork"]
    assert states.body["journal"][0]["nodes"] == first_nodes

    ambiguous = completion(
        PREFIX + " Ambiguous logical head.",
        fork_id=fork_id,
        state_id=state_id,
    )
    assert ambiguous.status_code == 400

    winner = completion(
        PREFIX + " Logical winner.",
        node_id=winner_node_id,
    )
    assert winner.status_code == 200
    assert winner.body["id_slot"] == 1
    commit = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": fork_id},
    )
    assert commit.status_code == 200
    assert commit.body["state_id"] == state_id
    assert commit.body["id_slot"] == 1
    assert commit.body["node_id"] == winner_node_id
    assert commit.body["parent_node_id"] == -1

    continued = completion(
        PREFIX + " Logical continuation.",
        fork_id=fork_id,
        state_id=state_id,
    )
    assert continued.status_code == 200
    assert continued.body["timings"]["cache_n"] > 0

    refork = server.make_request(
        "POST",
        "/slots/1?action=fork",
        data={"fork_id": fork_id, "destinations": [0, 2]},
    )
    assert refork.status_code == 200
    assert refork.body["state_id"] == state_id
    assert refork.body["fork_id"] != fork_id
    refork_nodes = refork.body["nodes"]
    assert len({node["node_id"] for node in refork_nodes}) == 3
    assert {node["node_id"] for node in refork_nodes}.isdisjoint(
        {node["node_id"] for node in first_nodes}
    )
    assert {node["parent_node_id"] for node in refork_nodes} == {winner_node_id}

    stale_node = completion(PREFIX, node_id=winner_node_id)
    assert stale_node.status_code == 503
    selected_refork_node = next(node for node in refork_nodes if node["id_slot"] == 2)
    selected = completion(
        PREFIX + " Select a reforked branch by immutable node.",
        node_id=selected_refork_node["node_id"],
    )
    assert selected.status_code == 200
    assert selected.body["id_slot"] == 2

    states = server.make_request("GET", "/states")
    assert states.status_code == 200
    assert states.body["states"][0]["state_id"] == state_id
    assert states.body["states"][0]["fork_id"] == refork.body["fork_id"]
    assert states.body["states"][0]["members"] == [0, 1, 2]
    assert states.body["states"][0]["heads"] == refork_nodes
    events = states.body["journal"]
    assert [entry["event"] for entry in events] == ["fork", "commit", "fork"]
    assert [entry["sequence"] for entry in events] == sorted(
        entry["sequence"] for entry in events
    )
    assert events[-1]["parent_fork_id"] == fork_id
    assert events[-1]["nodes"] == refork_nodes

    incremental = server.make_request(
        "GET",
        f"/states?journal_after={events[1]['sequence']}",
    )
    assert incremental.status_code == 200
    assert [entry["sequence"] for entry in incremental.body["journal"]] == [
        events[2]["sequence"]
    ]


def test_statetree_node_addressed_mutations_are_exact_and_stale_safe():
    global server
    server.n_slots = 3
    server.statetree_lease_ms = 5_000
    server.start()

    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["statetree"]["node_mutations"] == ["fork", "snapshot", "commit", "renew", "erase"]

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    state_id = fork.body["state_id"]
    fork_id = fork.body["fork_id"]
    first_nodes = {node["id_slot"]: node for node in fork.body["nodes"]}
    winner_node_id = first_nodes[1]["node_id"]

    malformed = server.make_request("POST", "/nodes/not-a-node?action=renew", data={})
    assert malformed.status_code == 400
    open_refork = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=fork",
        data={"destinations": [0]},
    )
    assert open_refork.status_code == 503
    max_node = server.make_request(
        "POST",
        f"/nodes/{(1 << 63) - 1}?action=fork",
        data={"destinations": [0]},
    )
    assert max_node.status_code == 503
    overflow_node = server.make_request(
        "POST",
        f"/nodes/{1 << 63}?action=fork",
        data={"destinations": [0]},
    )
    assert overflow_node.status_code == 400
    unavailable = server.make_request(
        "POST",
        f"/nodes/{winner_node_id + 1000}?action=renew",
        data={},
    )
    assert unavailable.status_code == 503
    mismatched_state = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=renew",
        data={"state_id": state_id + 1},
    )
    assert mismatched_state.status_code == 503
    mismatched_generation = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=commit",
        data={"fork_id": fork_id + 1},
    )
    assert mismatched_generation.status_code == 503

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert all(slots.body[id_slot]["is_reserved"] for id_slot in (0, 1, 2))

    renewed = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=renew",
        data={},
    )
    assert renewed.status_code == 200
    assert renewed.body["id_slot"] == 1
    assert renewed.body["state_id"] == state_id
    assert renewed.body["node_id"] == winner_node_id
    assert renewed.body["fork_id"] == fork_id

    committed = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=commit",
        data={"state_id": state_id},
    )
    assert committed.status_code == 200
    assert committed.body["id_slot"] == 1
    assert committed.body["node_id"] == winner_node_id
    assert committed.body["released"] == [0, 2]

    stale_loser = server.make_request(
        "POST",
        f"/nodes/{first_nodes[0]['node_id']}?action=renew",
        data={},
    )
    assert stale_loser.status_code == 503
    renewed_root = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=renew",
        data={"fork_id": fork_id},
    )
    assert renewed_root.status_code == 200

    def run_long_root():
        return completion(
            PREFIX + " Extend the committed root while re-fork is attempted.",
            timeout=30,
            n_predict=128,
            node_id=winner_node_id,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        root_request = executor.submit(run_long_root)
        wait_for_slot_processing(1, True)
        busy_refork = server.make_request(
            "POST",
            f"/nodes/{winner_node_id}?action=fork",
            data={"state_id": state_id, "destinations": [0, 2]},
        )
        assert busy_refork.status_code == 503
        assert root_request.result(timeout=30).status_code == 200

    mismatched_refork = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=fork",
        data={"fork_id": fork_id + 1, "destinations": [0, 2]},
    )
    assert mismatched_refork.status_code == 503

    refork = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=fork",
        data={"state_id": state_id, "destinations": [0, 2]},
    )
    assert refork.status_code == 200
    assert refork.body["id_slot"] == 1
    assert refork.body["state_id"] == state_id
    refork_id = refork.body["fork_id"]
    refork_nodes = {node["id_slot"]: node for node in refork.body["nodes"]}
    assert {node["parent_node_id"] for node in refork_nodes.values()} == {winner_node_id}

    stale_parent_refork = server.make_request(
        "POST",
        f"/nodes/{winner_node_id}?action=fork",
        data={"destinations": [0, 2]},
    )
    assert stale_parent_refork.status_code == 503

    erased_node = refork_nodes[0]
    erased = server.make_request(
        "POST",
        f"/nodes/{erased_node['node_id']}?action=erase",
        data={"state_id": state_id, "fork_id": refork_id},
    )
    assert erased.status_code == 200
    assert erased.body["id_slot"] == 0
    assert erased.body["state_id"] == state_id
    assert erased.body["node_id"] == erased_node["node_id"]
    assert erased.body["parent_node_id"] == winner_node_id
    assert erased.body["fork_id"] == refork_id
    assert erased.body["n_erased"] > 0

    stale_erase = server.make_request(
        "POST",
        f"/nodes/{erased_node['node_id']}?action=erase",
        data={},
    )
    assert stale_erase.status_code == 503

    final_node = refork_nodes[2]
    final_commit = server.make_request(
        "POST",
        f"/nodes/{final_node['node_id']}?action=commit",
        data={},
    )
    assert final_commit.status_code == 200
    assert final_commit.body["id_slot"] == 2
    assert final_commit.body["released"] == [1]

    final_erase = server.make_request(
        "POST",
        f"/nodes/{final_node['node_id']}?action=erase",
        data={},
    )
    assert final_erase.status_code == 200
    assert final_erase.body["node_id"] == final_node["node_id"]
    states = server.make_request("GET", "/states")
    assert states.status_code == 200
    assert states.body["states"] == []


def test_statetree_snapshots_are_content_addressed_immutable_and_materializable():
    global server
    server.n_slots = 4
    server.server_metrics = True
    server.statetree_lease_ms = 5_000
    server.statetree_max_snapshot_bytes = 1 << 30
    server.start()

    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["statetree"]["snapshot_enabled"] is True
    assert props.body["statetree"]["snapshot_admission"] == "reject"

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200
    nodes = {node["id_slot"]: node for node in fork.body["nodes"]}

    captures = []
    for id_slot in (0, 1):
        node_id = nodes[id_slot]["node_id"]
        captured = server.make_request(
            "POST",
            f"/nodes/{node_id}?action=snapshot",
            data={"state_id": fork.body["state_id"], "fork_id": fork.body["fork_id"]},
        )
        assert captured.status_code == 200
        assert captured.body["source_node_id"] == node_id
        assert captured.body["source_slot"] == id_slot
        assert captured.body["digest"].startswith("sha256:")
        assert captured.body["payload_bytes"] == (
            captured.body["state_bytes"] + captured.body["token_bytes"]
        )
        captures.append(captured.body)

    # The physical sequence ID is not content: identical fork heads converge.
    assert captures[0]["digest"] == captures[1]["digest"]
    assert captures[0]["payload_bytes"] == captures[1]["payload_bytes"]
    assert captures[0]["deduplicated"] is False
    assert captures[1]["deduplicated"] is True

    listed = server.make_request("GET", "/snapshots")
    assert listed.status_code == 200
    assert [row["snapshot_id"] for row in listed.body["snapshots"]] == [0, 1]
    assert listed.body["snapshot_bytes"] == captures[0]["payload_bytes"]
    assert listed.body["snapshot_content_count"] == 1
    assert listed.body["snapshot_high_water_bytes"] == listed.body["snapshot_bytes"]
    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert metric_value(metrics.body, "statetree_snapshot_count") == 2
    assert metric_value(metrics.body, "statetree_snapshot_content_count") == 1
    assert metric_value(metrics.body, "statetree_snapshots_captured_total") == 2

    committed = server.make_request(
        "POST",
        f"/nodes/{nodes[1]['node_id']}?action=commit",
        data={},
    )
    assert committed.status_code == 200
    extended = completion(
        PREFIX + " This text changes the live branch after capture.",
        1,
        node_id=nodes[1]["node_id"],
    )
    assert extended.status_code == 200
    changed = server.make_request(
        "POST",
        f"/nodes/{nodes[1]['node_id']}?action=snapshot",
        data={},
    )
    assert changed.status_code == 200
    assert changed.body["digest"] != captures[0]["digest"]

    # The original object survives source mutation and can recreate exact content.
    materialized = server.make_request(
        "POST",
        f"/snapshots/{captures[0]['snapshot_id']}?action=materialize",
        data={"digest": captures[0]["digest"], "id_slot": 2},
    )
    assert materialized.status_code == 200
    assert materialized.body["id_slot"] == 2
    assert materialized.body["parent_node_id"] == captures[0]["source_node_id"]
    materialized_node = materialized.body["node_id"]

    roundtrip = server.make_request(
        "POST",
        f"/nodes/{materialized_node}?action=snapshot",
        data={},
    )
    assert roundtrip.status_code == 200
    assert roundtrip.body["digest"] == captures[0]["digest"]
    assert roundtrip.body["deduplicated"] is True

    states = server.make_request("GET", "/states")
    assert states.status_code == 200
    restored_head = next(
        head
        for state in states.body["states"]
        for head in state["heads"]
        if head["node_id"] == materialized_node
    )
    assert restored_head["materialized_snapshot_id"] == captures[0]["snapshot_id"]
    unique_payloads = {
        row["digest"]: row["payload_bytes"] for row in states.body["snapshots"]
    }
    assert states.body["snapshot_bytes"] == sum(unique_payloads.values())
    assert states.body["snapshot_content_count"] == len(unique_payloads)

    mismatch = server.make_request(
        "POST",
        f"/snapshots/{captures[0]['snapshot_id']}?action=materialize",
        data={"digest": "sha256:" + "0" * 64, "id_slot": 3},
    )
    assert mismatch.status_code == 503

    erased = server.make_request(
        "POST",
        f"/snapshots/{captures[0]['snapshot_id']}?action=erase",
        data={"digest": captures[0]["digest"]},
    )
    assert erased.status_code == 200
    assert erased.body["content_reclaimed"] is False
    stale = server.make_request(
        "POST",
        f"/snapshots/{captures[0]['snapshot_id']}?action=materialize",
        data={"id_slot": 3},
    )
    assert stale.status_code == 503

    shared_erase = server.make_request(
        "POST",
        f"/snapshots/{captures[1]['snapshot_id']}?action=erase",
        data={"digest": captures[1]["digest"]},
    )
    assert shared_erase.status_code == 200
    assert shared_erase.body["content_reclaimed"] is False
    final_shared_erase = server.make_request(
        "POST",
        f"/snapshots/{roundtrip.body['snapshot_id']}?action=erase",
        data={"digest": roundtrip.body["digest"]},
    )
    assert final_shared_erase.status_code == 200
    assert final_shared_erase.body["content_reclaimed"] is True
    after_reclaim = server.make_request("GET", "/snapshots")
    assert after_reclaim.status_code == 200
    assert after_reclaim.body["snapshot_content_count"] == 1
    assert after_reclaim.body["snapshot_bytes"] == changed.body["payload_bytes"]


def test_statetree_snapshot_budget_rejects_without_eviction():
    global server
    server.n_slots = 2
    server.server_metrics = True
    server.statetree_max_snapshot_bytes = 1
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200
    capture = server.make_request(
        "POST",
        f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot",
        data={},
    )
    assert capture.status_code == 503
    listed = server.make_request("GET", "/snapshots")
    assert listed.status_code == 200
    assert listed.body["snapshots"] == []
    assert listed.body["snapshot_bytes"] == 0
    assert listed.body["snapshot_high_water_bytes"] == 0
    assert listed.body["snapshot_content_count"] == 0
    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert metric_value(metrics.body, "statetree_snapshot_rejected_total") == 1


def test_statetree_durable_snapshot_survives_restart_and_fails_closed(tmp_path):
    global server
    server.n_slots = 2
    server.server_metrics = True
    server.statetree_lease_ms = 30_000
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-model-runtime-A"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200
    captured = server.make_request(
        "POST",
        f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot",
        data={},
    )
    assert captured.status_code == 200
    digest = captured.body["digest"]
    digest_hex = digest.removeprefix("sha256:")

    def spill_request():
        return server.make_request(
            "POST",
            f"/snapshots/{captured.body['snapshot_id']}?action=spill",
            data={"digest": digest},
        )

    # Concurrent duplicates share one disk reservation. State inspection must
    # keep advancing while fsync and full read-back verification are in flight.
    saw_pending_io = False
    pending_probe_ms = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        spill_futures = [executor.submit(spill_request) for _ in range(2)]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not all(future.done() for future in spill_futures):
            probe_start = time.monotonic()
            live = server.make_request("GET", "/states")
            probe_ms = (time.monotonic() - probe_start) * 1000
            assert live.status_code == 200
            if live.body["durable_io_pending"] > 0:
                saw_pending_io = True
                pending_probe_ms.append(probe_ms)
            time.sleep(0.001)
        spill_results = [future.result() for future in spill_futures]
    assert saw_pending_io
    spilled = next(result for result in spill_results if result.body["deduplicated"] is False)
    duplicate = next(result for result in spill_results if result.body["deduplicated"] is True)
    assert spilled.status_code == 200
    assert spilled.body["file_bytes"] > spilled.body["payload_bytes"]
    assert duplicate.status_code == 200
    assert duplicate.body["disk_bytes"] == spilled.body["disk_bytes"]
    assert spilled.body["timings"]["io_ms"] > 0
    assert min(pending_probe_ms) < spilled.body["timings"]["io_ms"] / 2
    io_state = server.make_request("GET", "/states")
    assert io_state.body["durable_io_pending"] == 0
    assert io_state.body["durable_io_queue_high_water"] >= 2
    assert io_state.body["durable_io_completed_total"] >= 2
    assert io_state.body["durable_io_reserved_disk_bytes"] == 0
    assert io_state.body["durable_io_reserved_disk_high_water"] == spilled.body["file_bytes"]

    namespace_a = next(path for path in tmp_path.iterdir() if path.is_dir())
    object_path = namespace_a / f"{digest_hex}.tss"
    assert object_path.stat().st_size == spilled.body["file_bytes"]
    server.stop()

    # A crash artifact is removed, while the complete object is rediscovered.
    (namespace_a / ".tmp-interrupted-object").write_bytes(b"partial")
    malformed_path = namespace_a / ("0" * 64 + ".tss")
    malformed_path.write_bytes(b"bad")
    server.statetree_max_snapshot_load_bytes = captured.body["payload_bytes"]
    server.start()
    contents = server.make_request("GET", "/snapshot-contents")
    assert contents.status_code == 200
    assert contents.body["recovered_temp_files"] == 1
    assert contents.body["ignored_corrupt_files"] == 1
    assert contents.body["orphaned_disk_bytes"] == 3
    assert [row["digest"] for row in contents.body["contents"]] == [digest]
    assert contents.body["disk_bytes"] == spilled.body["file_bytes"] + 3
    assert server.make_request("GET", "/snapshots").body["snapshots"] == []

    # A disconnected owner is noticed on the durable route's short poll and
    # its verified payload is discarded before any slot mutation.
    with pytest.raises(requests.exceptions.Timeout):
        requests.post(
            f"http://{server.server_host}:{server.server_port}/snapshot-contents/{digest_hex}?action=materialize",
            json={"id_slot": 0},
            timeout=0.001,
        )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        cancelled = server.make_request("GET", "/states")
        if cancelled.body["durable_io_pending"] == 0:
            break
        time.sleep(0.01)
    assert cancelled.body["durable_io_cancelled_loads_total"] == 1
    assert all(not row["is_reserved"] for row in server.make_request("GET", "/slots").body)

    # The per-object load ceiling is also the aggregate reservation ceiling.
    # Two simultaneous full-size loads cannot overcommit transient host memory.
    barrier = threading.Barrier(3)

    def cold_request(id_slot):
        barrier.wait()
        return server.make_request(
            "POST",
            f"/snapshot-contents/{digest_hex}?action=materialize",
            data={"id_slot": id_slot},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        cold_futures = [executor.submit(cold_request, id_slot) for id_slot in (0, 1)]
        barrier.wait()
        cold_results = [future.result() for future in cold_futures]
    cold = next(result for result in cold_results if result.status_code == 200)
    pressure = next(result for result in cold_results if result.status_code == 503)
    assert "reservation budget" in str(pressure.body)
    assert cold.status_code == 200
    assert cold.body["cold"] is True
    assert cold.body["snapshot_id"] is None
    assert cold.body["parent_node_id"] == -1
    assert cold.body["digest"] == digest
    roundtrip = server.make_request(
        "POST",
        f"/nodes/{cold.body['node_id']}?action=snapshot",
        data={},
    )
    assert roundtrip.status_code == 200
    assert roundtrip.body["digest"] == digest
    states = server.make_request("GET", "/states")
    head = next(
        head
        for state in states.body["states"]
        for head in state["heads"]
        if head["node_id"] == cold.body["node_id"]
    )
    assert head["materialized_snapshot_id"] is None
    assert head["materialized_content_digest"] == digest
    states = server.make_request("GET", "/states")
    assert states.body["durable_io_reserved_load_bytes"] == 0
    assert states.body["durable_io_reserved_load_high_water"] == captured.body["payload_bytes"]
    server.stop()

    # A different explicit compatibility ID sees an isolated empty namespace.
    server.statetree_snapshot_compat_id = "unit-model-runtime-B"
    server.start()
    incompatible = server.make_request("GET", "/snapshot-contents")
    assert incompatible.status_code == 200
    assert incompatible.body["contents"] == []
    unavailable = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=materialize",
        data={"id_slot": 0},
    )
    assert unavailable.status_code == 503
    server.stop()

    # The cold-load ceiling is enforced from indexed envelope metadata before
    # allocating payload buffers or reserving a destination slot.
    server.statetree_snapshot_compat_id = "unit-model-runtime-A"
    server.statetree_max_snapshot_load_bytes = 1
    server.start()
    oversized = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=materialize",
        data={"id_slot": 0},
    )
    assert oversized.status_code == 503
    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert all(not row["is_reserved"] for row in slots.body)
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["statetree"]["max_snapshot_load_bytes"] == 1
    server.stop()

    # Replacing a discovered file with a larger, structurally consistent
    # envelope cannot bypass the ceiling using stale index metadata.
    server.statetree_max_snapshot_load_bytes = captured.body["payload_bytes"]
    server.start()
    with object_path.open("r+b") as content:
        content.seek(32)
        original_state_bytes = int.from_bytes(content.read(8), "little")
        content.seek(32)
        content.write((original_state_bytes + 1).to_bytes(8, "little"))
        content.seek(0, os.SEEK_END)
        content.write(b"\0")
    replaced = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=materialize",
        data={"id_slot": 0},
    )
    assert replaced.status_code == 503
    assert "cold-load byte ceiling" in str(replaced.body)
    assert all(not row["is_reserved"] for row in server.make_request("GET", "/slots").body)
    server.stop()
    with object_path.open("r+b") as content:
        content.truncate(spilled.body["file_bytes"])
        content.seek(32)
        content.write(original_state_bytes.to_bytes(8, "little"))

    # Corrupt payload remains discoverable by its valid envelope but cannot load.
    server.statetree_snapshot_compat_id = "unit-model-runtime-A"
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    with object_path.open("r+b") as content:
        content.seek(-1, os.SEEK_END)
        final_byte = content.read(1)
        content.seek(-1, os.SEEK_END)
        content.write(bytes([final_byte[0] ^ 0xFF]))
    server.start()
    corrupt = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=materialize",
        data={"id_slot": 0},
    )
    assert corrupt.status_code == 503
    after_corruption = server.make_request("GET", "/snapshot-contents")
    assert after_corruption.status_code == 200
    assert after_corruption.body["runtime_integrity_failures"] == 1
    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert all(not row["is_reserved"] for row in slots.body)
    durable_erase = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=erase",
        data={},
    )
    assert durable_erase.status_code == 200
    assert durable_erase.body["disk_bytes"] == 3
    assert not object_path.exists()
    stale_content = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=materialize",
        data={"id_slot": 0},
    )
    assert stale_content.status_code == 503


def test_statetree_durable_snapshot_disk_budget_rejects_without_files(tmp_path):
    global server
    server.n_slots = 2
    server.server_metrics = True
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-disk-pressure"
    server.statetree_max_snapshot_disk_bytes = 1
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200
    captured = server.make_request(
        "POST",
        f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot",
        data={},
    )
    assert captured.status_code == 200
    rejected = server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=spill",
        data={"digest": captured.body["digest"]},
    )
    assert rejected.status_code == 503
    contents = server.make_request("GET", "/snapshot-contents")
    assert contents.status_code == 200
    assert contents.body["contents"] == []
    assert contents.body["disk_bytes"] == 0
    assert not list(tmp_path.rglob("*.tss"))
    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert metric_value(metrics.body, "statetree_durable_rejected_total") == 1


def test_statetree_managed_publish_commits_object_and_first_owner_together(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-managed-publish"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    )
    digest = captured.body["digest"]
    digest_hex = digest.removeprefix("sha256:")
    payload = {
        "digest": digest,
        "owner": "campaign/atomic",
        "retention_class": "pinned",
    }
    published = server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=publish",
        data=payload,
    )
    assert published.status_code == 200
    assert published.body["action"] == "publish"
    assert published.body["object_deduplicated"] is False
    assert published.body["ownership_deduplicated"] is False
    assert published.body["manifest_revision"] == 2
    assert published.body["ref_count"] == 1

    duplicate = server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=publish",
        data=payload,
    )
    assert duplicate.status_code == 200
    assert duplicate.body["object_deduplicated"] is True
    assert duplicate.body["ownership_deduplicated"] is True
    assert duplicate.body["manifest_revision"] == 2
    server.stop()

    server.start()
    replayed = server.make_request("GET", "/snapshot-contents")
    assert replayed.status_code == 200
    assert replayed.body["manifest_revision"] == 2
    assert replayed.body["contents"][0]["ref_count"] == 1
    assert replayed.body["refs"][0]["owner"] == "campaign/atomic"
    assert server.make_request(
        "POST", f"/snapshot-contents/{digest_hex}?action=erase", data={}
    ).status_code == 503


def test_statetree_managed_publish_reconciles_committed_and_abandoned_intents(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-publish-recovery"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    )
    digest = captured.body["digest"]
    digest_hex = digest.removeprefix("sha256:")
    assert server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=spill",
        data={"digest": digest},
    ).status_code == 200
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = namespace / "manifest.v1"
    server.stop()

    # Simulate a crash after verified object publication but before owner commit.
    append_manifest_publish_intent(
        manifest_path, 1, "campaign/recovered", digest, "pinned"
    )
    server.start()
    committed = server.make_request("GET", "/snapshot-contents")
    assert committed.status_code == 200
    assert committed.body["manifest_revision"] == 2
    assert committed.body["manifest_recovered_publish_commits"] == 1
    assert committed.body["manifest_recovered_publish_aborts"] == 0
    assert committed.body["refs"][0]["owner"] == "campaign/recovered"
    assert server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=release",
        data={"owner": "campaign/recovered"},
    ).status_code == 200
    assert server.make_request(
        "POST", f"/snapshot-contents/{digest_hex}?action=erase", data={}
    ).status_code == 200
    server.stop()

    # Simulate a crash after the intent fsync but before object publication.
    append_manifest_publish_intent(
        manifest_path, 5, "campaign/abandoned", digest, "cache"
    )
    server.start()
    aborted = server.make_request("GET", "/snapshot-contents")
    assert aborted.status_code == 200
    assert aborted.body["manifest_revision"] == 6
    assert aborted.body["manifest_recovered_publish_commits"] == 0
    assert aborted.body["manifest_recovered_publish_aborts"] == 1
    assert aborted.body["refs"] == []
    assert aborted.body["contents"] == []


def test_statetree_managed_publish_reserves_terminal_manifest_space(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-publish-terminal-budget"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 600
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    )
    digest = captured.body["digest"]
    statuses = []
    for index in range(3):
        response = server.make_request(
            "POST",
            f"/snapshots/{captured.body['snapshot_id']}?action=publish",
            data={
                "digest": digest,
                "owner": f"owner/{index}",
                "retention_class": "cache",
            },
        )
        statuses.append(response.status_code)
    assert statuses == [200, 200, 503]
    contents = server.make_request("GET", "/snapshot-contents")
    assert contents.status_code == 200
    assert contents.body["manifest_revision"] == 4
    assert contents.body["manifest_file_bytes"] <= 600
    assert len(contents.body["refs"]) == 2
    server.stop()

    server.start()
    replayed = server.make_request("GET", "/snapshot-contents")
    assert replayed.status_code == 200
    assert replayed.body["manifest_revision"] == 4
    assert len(replayed.body["refs"]) == 2


def test_statetree_logical_head_cas_restart_materialize_and_cache_fences(tmp_path):
    global server
    server.n_slots = 4
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-logical-head-cas-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    snapshots = []
    for index, source in enumerate((0, 2)):
        assert completion(PREFIX + f" logical head {index}", source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [source + 1]}
        )
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        snapshots.append(server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        ).body)

    for index, snapshot in enumerate(snapshots):
        published = server.make_request(
            "POST",
            f"/snapshots/{snapshot['snapshot_id']}?action=publish",
            data={
                "digest": snapshot["digest"],
                "owner": f"cache/head-{index}",
                "retention_class": "cache",
            },
        )
        assert published.status_code == 200

    digest_a, digest_b = (snapshot["digest"] for snapshot in snapshots)
    digest_a_hex = digest_a.removeprefix("sha256:")

    invalid = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=create",
        data={"digest": digest_a, "unexpected": True},
    )
    assert invalid.status_code == 400

    created = server.make_request(
        "POST", "/snapshot-heads/campaign-main?action=create", data={"digest": digest_a}
    )
    assert created.status_code == 200
    assert created.body["generation"] == 1
    assert created.body["digest"] == digest_a
    assert created.body["parent_digest"] is None
    assert created.body["deduplicated"] is False
    created_revision = created.body["manifest_revision"]

    duplicate_create = server.make_request(
        "POST", "/snapshot-heads/campaign-main?action=create", data={"digest": digest_a}
    )
    assert duplicate_create.status_code == 200
    assert duplicate_create.body["deduplicated"] is True
    assert duplicate_create.body["manifest_revision"] == created_revision

    released_a = server.make_request(
        "POST",
        f"/snapshot-contents/{digest_a_hex}?action=release",
        data={"owner": "cache/head-0"},
    )
    assert released_a.status_code == 200
    fenced_a = server.make_request(
        "POST", f"/snapshot-contents/{digest_a_hex}?action=erase", data={}
    )
    assert fenced_a.status_code == 503
    content_a = next(
        row for row in server.make_request("GET", "/snapshot-contents").body["contents"]
        if row["digest"] == digest_a
    )
    assert content_a["owner_ref_count"] == 0
    assert content_a["head_ref_count"] == 1
    assert content_a["ref_count"] == 1

    compacted = server.make_request("POST", "/snapshot-manifest?action=compact", data={})
    assert compacted.status_code == 200
    checkpoint_revision = compacted.body["manifest_revision"]
    server.stop()

    server.start()
    replayed = server.make_request("GET", "/snapshot-heads")
    assert replayed.status_code == 200
    assert replayed.body["manifest_revision"] == checkpoint_revision
    assert replayed.body["heads"] == [{
        "name": "campaign-main",
        "digest": digest_a,
        "parent_digest": None,
        "generation": 1,
        "revision": created.body["head_revision"],
    }]

    materialized = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=materialize",
        data={
            "expected_generation": 1,
            "expected_digest": digest_a,
            "id_slot": 0,
        },
    )
    assert materialized.status_code == 200
    assert materialized.body["action"] == "materialize_head"
    assert materialized.body["digest"] == digest_a
    assert materialized.body["head"]["generation"] == 1
    assert materialized.body["head"]["digest"] == digest_a

    stale_materialize = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=materialize",
        data={"expected_generation": 2, "expected_digest": digest_a},
    )
    assert stale_materialize.status_code == 503

    stale_advance = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=advance",
        data={
            "digest": digest_b,
            "expected_generation": 2,
            "expected_digest": digest_a,
        },
    )
    assert stale_advance.status_code == 503
    assert server.make_request("GET", "/snapshot-heads").body["manifest_revision"] == checkpoint_revision

    advanced = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=advance",
        data={
            "digest": digest_b,
            "expected_generation": 1,
            "expected_digest": digest_a,
        },
    )
    assert advanced.status_code == 200
    assert advanced.body["generation"] == 2
    assert advanced.body["digest"] == digest_b
    assert advanced.body["parent_digest"] == digest_a
    assert advanced.body["deduplicated"] is False
    advance_revision = advanced.body["manifest_revision"]

    retry_advance = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=advance",
        data={
            "digest": digest_b,
            "expected_generation": 1,
            "expected_digest": digest_a,
        },
    )
    assert retry_advance.status_code == 200
    assert retry_advance.body["deduplicated"] is True
    assert retry_advance.body["manifest_revision"] == advance_revision

    erased_a = server.make_request(
        "POST", f"/snapshot-contents/{digest_a_hex}?action=erase", data={}
    )
    assert erased_a.status_code == 200
    prune_fenced = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": 0}
    )
    assert prune_fenced.status_code == 503
    content_b = server.make_request("GET", "/snapshot-contents").body["contents"]
    assert len(content_b) == 1
    assert content_b[0]["digest"] == digest_b
    assert content_b[0]["head_ref_count"] == 1

    deleted = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=delete",
        data={"expected_generation": 2, "expected_digest": digest_b},
    )
    assert deleted.status_code == 200
    delete_revision = deleted.body["manifest_revision"]
    retry_delete = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=delete",
        data={"expected_generation": 2, "expected_digest": digest_b},
    )
    assert retry_delete.status_code == 200
    assert retry_delete.body["deduplicated"] is True
    assert retry_delete.body["manifest_revision"] == delete_revision
    aba_create = server.make_request(
        "POST", "/snapshot-heads/campaign-main?action=create", data={"digest": digest_b}
    )
    assert aba_create.status_code == 503
    assert server.make_request("GET", "/snapshot-heads").body["heads"] == []
    assert server.make_request(
        "POST", "/snapshot-manifest?action=compact", data={}
    ).status_code == 200
    pruned = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": 0}
    )
    assert pruned.status_code == 200
    assert pruned.body["reclaimed_bytes"] > 0
    assert server.make_request("GET", "/snapshot-contents").body["contents"] == []
    server.stop()

    server.start()
    assert server.make_request("GET", "/snapshot-heads").body["heads"] == []
    replayed_delete = server.make_request(
        "POST",
        "/snapshot-heads/campaign-main?action=delete",
        data={"expected_generation": 2, "expected_digest": digest_b},
    )
    assert replayed_delete.status_code == 200
    assert replayed_delete.body["deduplicated"] is True
    assert server.make_request(
        "POST", "/snapshot-heads/campaign-main?action=create", data={"digest": digest_b}
    ).status_code == 503


def test_statetree_logical_head_missing_object_fails_startup(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-logical-head-missing-object-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    )
    digest = captured.body["digest"]
    spilled = server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=spill",
        data={"digest": digest},
    )
    assert spilled.status_code == 200
    assert server.make_request(
        "POST", "/snapshot-heads/main?action=create", data={"digest": digest}
    ).status_code == 200
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    object_path = namespace / f"{digest.removeprefix('sha256:')}.tss"
    server.stop()
    object_path.unlink()

    with pytest.raises(RuntimeError, match="Server process died"):
        server.start()


def test_statetree_atomic_publish_advance_has_one_owner_head_commit(tmp_path):
    global server
    server.n_slots = 4
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-atomic-publish-advance-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    snapshots = []
    for index, source in enumerate((0, 2)):
        assert completion(PREFIX + f" atomic publish advance {index}", source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [source + 1]}
        )
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        snapshots.append(server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        ).body)

    digest_a, digest_b = (snapshot["digest"] for snapshot in snapshots)
    published_a = server.make_request(
        "POST",
        f"/snapshots/{snapshots[0]['snapshot_id']}?action=publish",
        data={
            "digest": digest_a,
            "owner": "cache/atomic-old",
            "retention_class": "cache",
        },
    )
    assert published_a.status_code == 200
    created = server.make_request(
        "POST", "/snapshot-heads/atomic-main?action=create", data={"digest": digest_a}
    )
    assert created.status_code == 200
    revision_before = created.body["manifest_revision"]

    payload = {
        "digest": digest_b,
        "owner": "campaign/atomic-new",
        "retention_class": "pinned",
        "head_name": "atomic-main",
        "expected_generation": 1,
        "expected_digest": digest_a,
    }
    stale = dict(payload, expected_generation=2)
    stale_result = server.make_request(
        "POST",
        f"/snapshots/{snapshots[1]['snapshot_id']}?action=publish-advance",
        data=stale,
    )
    assert stale_result.status_code == 503
    assert server.make_request("GET", "/snapshot-heads").body["manifest_revision"] == revision_before
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    object_b = namespace / f"{digest_b.removeprefix('sha256:')}.tss"
    assert not object_b.exists()

    invalid = server.make_request(
        "POST",
        f"/snapshots/{snapshots[1]['snapshot_id']}?action=publish-advance",
        data=dict(payload, unexpected=True),
    )
    assert invalid.status_code == 400
    assert not object_b.exists()

    committed = server.make_request(
        "POST",
        f"/snapshots/{snapshots[1]['snapshot_id']}?action=publish-advance",
        data=payload,
    )
    assert committed.status_code == 200
    assert committed.body["action"] == "publish_advance"
    assert committed.body["object_deduplicated"] is False
    assert committed.body["transaction_deduplicated"] is False
    assert committed.body["manifest_revision"] == revision_before + 2
    assert committed.body["ref_revision"] == committed.body["manifest_revision"]
    assert committed.body["head"] == {
        "name": "atomic-main",
        "generation": 2,
        "digest": digest_b,
        "parent_digest": digest_a,
        "revision": committed.body["manifest_revision"],
    }
    assert object_b.exists()

    retried = server.make_request(
        "POST",
        f"/snapshots/{snapshots[1]['snapshot_id']}?action=publish-advance",
        data=payload,
    )
    assert retried.status_code == 200
    assert retried.body["object_deduplicated"] is True
    assert retried.body["transaction_deduplicated"] is True
    assert retried.body["manifest_revision"] == committed.body["manifest_revision"]

    contents = server.make_request("GET", "/snapshot-contents").body
    ref_b = next(row for row in contents["refs"] if row["owner"] == "campaign/atomic-new")
    assert ref_b["digest"] == digest_b
    assert ref_b["revision"] == committed.body["manifest_revision"]
    head_b = server.make_request("GET", "/snapshot-heads").body["heads"][0]
    assert head_b["digest"] == digest_b
    assert head_b["revision"] == ref_b["revision"]
    server.stop()

    server.start()
    replayed_head = server.make_request("GET", "/snapshot-heads").body["heads"][0]
    replayed_refs = server.make_request("GET", "/snapshot-contents").body["refs"]
    replayed_ref = next(row for row in replayed_refs if row["owner"] == "campaign/atomic-new")
    assert replayed_head["generation"] == 2
    assert replayed_head["digest"] == digest_b
    assert replayed_head["revision"] == replayed_ref["revision"]


def test_statetree_atomic_publish_advance_recovers_absent_corrupt_and_verified_object(tmp_path):
    global server
    server.n_slots = 4
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-atomic-publish-advance-recovery-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    snapshots = []
    for index, source in enumerate((0, 2)):
        assert completion(PREFIX + f" atomic recovery {index}", source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [source + 1]}
        )
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        snapshots.append(server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        ).body)

    digest_a, digest_b = (snapshot["digest"] for snapshot in snapshots)
    assert server.make_request(
        "POST",
        f"/snapshots/{snapshots[0]['snapshot_id']}?action=publish",
        data={"digest": digest_a, "owner": "cache/recovery-old", "retention_class": "cache"},
    ).status_code == 200
    assert server.make_request(
        "POST", "/snapshot-heads/recovery-main?action=create", data={"digest": digest_a}
    ).status_code == 200
    assert server.make_request(
        "POST",
        f"/snapshots/{snapshots[1]['snapshot_id']}?action=spill",
        data={"digest": digest_b},
    ).status_code == 200
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = namespace / "manifest.v1"
    object_path = namespace / f"{digest_b.removeprefix('sha256:')}.tss"
    valid_object = object_path.read_bytes()
    server.stop()
    object_path.unlink()

    fields = (
        "campaign/recovery-new",
        digest_b,
        "pinned",
        "recovery-main",
        1,
        digest_a,
    )
    append_manifest_publish_advance_intent(manifest_path, 4, *fields)
    server.start()
    absent = server.make_request("GET", "/snapshot-contents").body
    assert absent["manifest_revision"] == 5
    assert all(row["owner"] != fields[0] for row in absent["refs"])
    assert server.make_request("GET", "/snapshot-heads").body["heads"][0]["digest"] == digest_a
    server.stop()

    corrupt_object = bytearray(valid_object)
    corrupt_object[-1] ^= 0xFF
    object_path.write_bytes(corrupt_object)
    append_manifest_publish_advance_intent(manifest_path, 6, *fields)
    server.start()
    corrupt = server.make_request("GET", "/snapshot-contents").body
    assert corrupt["manifest_revision"] == 7
    assert all(row["owner"] != fields[0] for row in corrupt["refs"])
    assert not object_path.exists()
    assert server.make_request("GET", "/snapshot-heads").body["heads"][0]["digest"] == digest_a
    server.stop()

    object_path.write_bytes(valid_object)
    append_manifest_publish_advance_intent(manifest_path, 8, *fields)
    server.start()
    committed = server.make_request("GET", "/snapshot-contents").body
    assert committed["manifest_revision"] == 9
    recovered_ref = next(row for row in committed["refs"] if row["owner"] == fields[0])
    recovered_head = server.make_request("GET", "/snapshot-heads").body["heads"][0]
    assert recovered_ref["digest"] == digest_b
    assert recovered_ref["revision"] == 9
    assert recovered_head["generation"] == 2
    assert recovered_head["digest"] == digest_b
    assert recovered_head["parent_digest"] == digest_a
    assert recovered_head["revision"] == recovered_ref["revision"]


def test_statetree_atomic_publish_advance_reserves_terminal_manifest_space(tmp_path):
    global server
    server.n_slots = 4
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-atomic-terminal-space-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 600
    server.start()

    snapshots = []
    for index, source in enumerate((0, 2)):
        assert completion(PREFIX + f" atomic terminal {index}", source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [source + 1]}
        )
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        snapshots.append(server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        ).body)
    digest_a, digest_b = (snapshot["digest"] for snapshot in snapshots)
    assert server.make_request(
        "POST",
        f"/snapshots/{snapshots[0]['snapshot_id']}?action=spill",
        data={"digest": digest_a},
    ).status_code == 200
    created = server.make_request(
        "POST", "/snapshot-heads/terminal?action=create", data={"digest": digest_a}
    )
    assert created.status_code == 200

    rejected = server.make_request(
        "POST",
        f"/snapshots/{snapshots[1]['snapshot_id']}?action=publish-advance",
        data={
            "digest": digest_b,
            "owner": "campaign/terminal",
            "retention_class": "pinned",
            "head_name": "terminal",
            "expected_generation": 1,
            "expected_digest": digest_a,
        },
    )
    assert rejected.status_code == 503
    assert server.make_request("GET", "/snapshot-heads").body["manifest_revision"] == 1
    contents = server.make_request("GET", "/snapshot-contents").body
    assert all(row["owner"] != "campaign/terminal" for row in contents["refs"])
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not (namespace / f"{digest_b.removeprefix('sha256:')}.tss").exists()


def test_statetree_atomic_publish_advance_commit_checksum_fails_closed(tmp_path):
    global server
    server.n_slots = 4
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-atomic-commit-checksum-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    snapshots = []
    for index, source in enumerate((0, 2)):
        assert completion(PREFIX + f" atomic checksum {index}", source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [source + 1]}
        )
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        snapshots.append(server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        ).body)
    digest_a, digest_b = (snapshot["digest"] for snapshot in snapshots)
    for snapshot in snapshots:
        assert server.make_request(
            "POST",
            f"/snapshots/{snapshot['snapshot_id']}?action=spill",
            data={"digest": snapshot["digest"]},
        ).status_code == 200
    assert server.make_request(
        "POST", "/snapshot-heads/checksum?action=create", data={"digest": digest_a}
    ).status_code == 200
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = namespace / "manifest.v1"
    server.stop()

    fields = (
        "campaign/checksum",
        digest_b,
        "pinned",
        "checksum",
        1,
        digest_a,
    )
    append_manifest_publish_advance_intent(manifest_path, 2, *fields)
    append_manifest_publish_advance_intent(manifest_path, 3, *fields, record_type=13)
    valid_manifest = manifest_path.read_bytes()
    corrupt_manifest = bytearray(valid_manifest)
    corrupt_manifest[-1] ^= 0xFF
    manifest_path.write_bytes(corrupt_manifest)
    with pytest.raises(RuntimeError, match="Server process died"):
        server.start()

    manifest_path.write_bytes(valid_manifest)
    server.start()
    committed_ref = next(
        row for row in server.make_request("GET", "/snapshot-contents").body["refs"]
        if row["owner"] == fields[0]
    )
    committed_head = server.make_request("GET", "/snapshot-heads").body["heads"][0]
    assert committed_ref["revision"] == 3
    assert committed_head["revision"] == 3
    assert committed_head["digest"] == digest_b


def test_statetree_logical_head_manifest_pressure_checkpoints_post_state(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-logical-head-pressure-v1"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 400
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    ).body
    digest = captured["digest"]
    assert server.make_request(
        "POST",
        f"/snapshots/{captured['snapshot_id']}?action=spill",
        data={"digest": digest},
    ).status_code == 200
    created = server.make_request(
        "POST", "/snapshot-heads/pressure?action=create", data={"digest": digest}
    )
    assert created.status_code == 200

    generation = 1
    for _ in range(2):
        advanced = server.make_request(
            "POST",
            "/snapshot-heads/pressure?action=advance",
            data={
                "digest": digest,
                "expected_generation": generation,
                "expected_digest": digest,
            },
        )
        assert advanced.status_code == 200
        generation += 1
        assert advanced.body["generation"] == generation
        assert advanced.body["manifest_file_bytes"] <= 400

    deleted = server.make_request(
        "POST",
        "/snapshot-heads/pressure?action=delete",
        data={"expected_generation": generation, "expected_digest": digest},
    )
    assert deleted.status_code == 200
    assert deleted.body["manifest_file_bytes"] <= 400
    manifest = server.make_request("GET", "/snapshot-contents").body
    assert manifest["manifest_revision"] == 4
    assert manifest["manifest_compactions"] >= 2
    server.stop()

    server.start()
    assert server.make_request("GET", "/snapshot-heads").body["heads"] == []
    retried = server.make_request(
        "POST",
        "/snapshot-heads/pressure?action=delete",
        data={"expected_generation": generation, "expected_digest": digest},
    )
    assert retried.status_code == 200
    assert retried.body["deduplicated"] is True
    assert retried.body["manifest_revision"] == 4


def test_statetree_managed_cache_prune_is_oldest_first_and_respects_fences(tmp_path):
    global server
    server.n_slots = 8
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-managed-prune"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    def make_snapshot(prompt, source, destination):
        assert completion(prompt, source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [destination]}
        )
        assert fork.status_code == 200
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        captured = server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        )
        assert captured.status_code == 200
        return captured.body

    def publish(snapshot, owner, retention_class):
        response = server.make_request(
            "POST",
            f"/snapshots/{snapshot['snapshot_id']}?action=publish",
            data={
                "digest": snapshot["digest"],
                "owner": owner,
                "retention_class": retention_class,
            },
        )
        assert response.status_code == 200
        return response.body

    snapshot_a = make_snapshot(PREFIX + " oldest cache", 0, 1)
    snapshot_b = make_snapshot(PREFIX + " newest cache", 2, 3)
    snapshot_c = make_snapshot(PREFIX + " pinned", 4, 5)
    published_a = publish(snapshot_a, "cache/oldest", "cache")
    published_b = publish(snapshot_b, "cache/newest", "cache")
    published_c = publish(snapshot_c, "campaign/pinned", "pinned")

    initial = server.make_request("GET", "/snapshot-contents").body
    unattainable = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": 0}
    )
    assert unattainable.status_code == 503
    unchanged = server.make_request("GET", "/snapshot-contents").body
    assert unchanged["disk_bytes"] == initial["disk_bytes"]
    assert len(unchanged["contents"]) == 3
    assert len(unchanged["refs"]) == 3

    target_after_a = initial["disk_bytes"] - published_a["file_bytes"]
    first = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": target_after_a}
    )
    assert first.status_code == 200
    assert [row["digest"] for row in first.body["evictions"]] == [snapshot_a["digest"]]
    assert first.body["disk_bytes_after"] == target_after_a

    second = server.make_request(
        "POST",
        "/snapshot-manifest?action=prune",
        data={"target_disk_bytes": published_c["file_bytes"]},
    )
    assert second.status_code == 200
    assert [row["digest"] for row in second.body["evictions"]] == [snapshot_b["digest"]]
    after_cache = server.make_request("GET", "/snapshot-contents").body
    assert [row["digest"] for row in after_cache["contents"]] == [snapshot_c["digest"]]
    assert after_cache["managed_digests"] == [snapshot_c["digest"]]

    snapshot_d = make_snapshot(PREFIX + " unmanaged", 6, 7)
    raw = server.make_request(
        "POST",
        f"/snapshots/{snapshot_d['snapshot_id']}?action=spill",
        data={"digest": snapshot_d["digest"]},
    )
    assert raw.status_code == 200
    fenced = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": 0}
    )
    assert fenced.status_code == 503
    final = server.make_request("GET", "/snapshot-contents").body
    assert {row["digest"] for row in final["contents"]} == {
        snapshot_c["digest"], snapshot_d["digest"]
    }
    assert final["managed_digests"] == [snapshot_c["digest"]]
    assert next(row for row in final["contents"] if row["digest"] == snapshot_d["digest"])["managed"] is False


def test_statetree_managed_publish_reclaims_cache_under_disk_pressure(tmp_path):
    global server
    server.n_slots = 6
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-managed-pressure"
    server.statetree_max_snapshot_disk_bytes = 1 << 25
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    snapshots = []
    for index, source in enumerate((0, 2, 4)):
        assert completion(PREFIX + f" pressure {index}", source).status_code == 200
        fork = server.make_request(
            "POST", f"/slots/{source}?action=fork", data={"destinations": [source + 1]}
        )
        node = next(row for row in fork.body["nodes"] if row["id_slot"] == source)
        snapshots.append(server.make_request(
            "POST", f"/nodes/{node['node_id']}?action=snapshot", data={}
        ).body)

    def managed_publish(snapshot, owner, retention_class):
        return server.make_request(
            "POST",
            f"/snapshots/{snapshot['snapshot_id']}?action=publish",
            data={
                "digest": snapshot["digest"],
                "owner": owner,
                "retention_class": retention_class,
            },
        )

    first = managed_publish(snapshots[0], "cache/pressure", "cache")
    assert first.status_code == 200
    second = managed_publish(snapshots[1], "campaign/current", "pinned")
    assert second.status_code == 200
    assert [row["digest"] for row in second.body["cache_evictions"]] == [snapshots[0]["digest"]]
    assert second.body["cache_reclaimed_bytes"] == first.body["file_bytes"]

    blocked = managed_publish(snapshots[2], "campaign/blocked", "pinned")
    assert blocked.status_code == 503
    contents = server.make_request("GET", "/snapshot-contents").body
    assert [row["digest"] for row in contents["contents"]] == [snapshots[1]["digest"]]
    assert [row["owner"] for row in contents["refs"]] == ["campaign/current"]
    assert contents["managed_digests"] == [snapshots[1]["digest"]]


def test_statetree_managed_cache_never_crosses_mixed_pinned_owner(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-managed-mixed-fence"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    ).body
    digest_hex = captured["digest"].removeprefix("sha256:")
    assert server.make_request(
        "POST",
        f"/snapshots/{captured['snapshot_id']}?action=publish",
        data={
            "digest": captured["digest"],
            "owner": "cache/shared",
            "retention_class": "cache",
        },
    ).status_code == 200
    assert server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=retain",
        data={"owner": "campaign/shared", "retention_class": "pinned"},
    ).status_code == 200

    blocked = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": 0}
    )
    assert blocked.status_code == 503
    fenced = server.make_request("GET", "/snapshot-contents").body
    assert len(fenced["refs"]) == 2
    assert len(fenced["contents"]) == 1

    assert server.make_request(
        "POST",
        f"/snapshot-contents/{digest_hex}?action=release",
        data={"owner": "campaign/shared"},
    ).status_code == 200
    pruned = server.make_request(
        "POST", "/snapshot-manifest?action=prune", data={"target_disk_bytes": 0}
    )
    assert pruned.status_code == 200
    assert pruned.body["released_refs"] == 1
    assert server.make_request("GET", "/snapshot-contents").body["contents"] == []


def test_statetree_managed_eviction_completes_object_erase_after_crash(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-managed-eviction-recovery"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST", f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot", data={}
    ).body
    published = server.make_request(
        "POST",
        f"/snapshots/{captured['snapshot_id']}?action=publish",
        data={
            "digest": captured["digest"],
            "owner": "cache/recovery",
            "retention_class": "cache",
        },
    )
    assert published.status_code == 200
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = namespace / "manifest.v1"
    object_path = namespace / f"{captured['digest'].removeprefix('sha256:')}.tss"
    server.stop()

    # Crash after the atomic cache-owner eviction record, before object erase.
    append_manifest_mutation(manifest_path, 3, 7, "", captured["digest"], "")
    server.start()
    recovered = server.make_request("GET", "/snapshot-contents")
    assert recovered.status_code == 200
    assert recovered.body["manifest_revision"] == 4
    assert recovered.body["managed_recovered_erases"] == 1
    assert recovered.body["managed_recovered_bytes"] == published.body["file_bytes"]
    assert recovered.body["contents"] == []
    assert recovered.body["refs"] == []
    assert recovered.body["managed_digests"] == []
    assert not object_path.exists()


def test_statetree_durable_manifest_replays_fences_and_recovers_torn_tail(tmp_path):
    global server
    server.n_slots = 2
    server.server_metrics = True
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-manifest-lifecycle"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    assert fork.status_code == 200
    captured = server.make_request(
        "POST",
        f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot",
        data={},
    )
    assert captured.status_code == 200
    digest = captured.body["digest"]
    digest_hex = digest.removeprefix("sha256:")
    spilled = server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=spill",
        data={"digest": digest},
    )
    assert spilled.status_code == 200

    def ownership(action, owner, retention_class=None):
        body = {"owner": owner}
        if retention_class is not None:
            body["retention_class"] = retention_class
        return server.make_request(
            "POST",
            f"/snapshot-contents/{digest_hex}?action={action}",
            data=body,
        )

    retained = ownership("retain", "campaign/primary", "pinned")
    assert retained.status_code == 200
    assert retained.body["deduplicated"] is False
    duplicate = ownership("retain", "campaign/primary", "pinned")
    assert duplicate.status_code == 200
    assert duplicate.body["deduplicated"] is True
    cached = ownership("retain", "cache/warm", "cache")
    assert cached.status_code == 200
    assert cached.body["ref_count"] == 2

    contents = server.make_request("GET", "/snapshot-contents")
    assert contents.status_code == 200
    assert contents.body["contents"][0]["ref_count"] == 2
    assert {(ref["owner"], ref["retention_class"]) for ref in contents.body["refs"]} == {
        ("campaign/primary", "pinned"),
        ("cache/warm", "cache"),
    }
    fenced = server.make_request(
        "POST", f"/snapshot-contents/{digest_hex}?action=erase", data={}
    )
    assert fenced.status_code == 503
    assert "retained" in str(fenced.body)

    compacted = server.make_request("POST", "/snapshot-manifest?action=compact", data={})
    assert compacted.status_code == 200
    assert compacted.body["refs"] == 2
    assert compacted.body["records_after"] == 1
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = namespace / "manifest.v1"
    object_path = namespace / f"{digest_hex}.tss"
    server.stop()

    # A complete checksummed checkpoint replays identically after restart.
    interrupted_compaction = namespace / ".manifest.tmp-crash"
    interrupted_compaction.write_bytes(b"partial checkpoint")
    server.start()
    replayed = server.make_request("GET", "/snapshot-contents")
    assert replayed.status_code == 200
    assert replayed.body["manifest_revision"] == cached.body["manifest_revision"]
    assert replayed.body["manifest_recovered_temp_files"] == 1
    assert not interrupted_compaction.exists()
    assert len(replayed.body["refs"]) == 2
    server.stop()

    # An interrupted append is truncated to the last complete checksummed record.
    torn_tail = b"partial-record-tail"
    with manifest_path.open("ab") as manifest:
        manifest.write(torn_tail)
        manifest.flush()
        os.fsync(manifest.fileno())
    server.start()
    recovered = server.make_request("GET", "/snapshot-contents")
    assert recovered.status_code == 200
    assert recovered.body["manifest_recovered_tail_bytes"] == len(torn_tail)
    assert len(recovered.body["refs"]) == 2
    server.stop()

    # A complete record with a bad checksum is corruption, not a recoverable tail.
    valid_manifest = manifest_path.read_bytes()
    corrupted_manifest = bytearray(valid_manifest)
    corrupted_manifest[-1] ^= 0xFF
    manifest_path.write_bytes(corrupted_manifest)
    with pytest.raises(RuntimeError):
        server.start()
    server.stop()
    manifest_path.write_bytes(valid_manifest)
    server.start()
    server.stop()

    # Ownership never silently points at absent content after discovery.
    object_bytes = object_path.read_bytes()
    object_path.unlink()
    with pytest.raises(RuntimeError):
        server.start()
    server.stop()
    object_path.write_bytes(object_bytes)
    server.start()

    first_release = ownership("release", "campaign/primary")
    assert first_release.status_code == 200
    assert first_release.body["ref_count"] == 1
    still_fenced = server.make_request(
        "POST", f"/snapshot-contents/{digest_hex}?action=erase", data={}
    )
    assert still_fenced.status_code == 503
    final_release = ownership("release", "cache/warm")
    assert final_release.status_code == 200
    assert final_release.body["ref_count"] == 0
    erased = server.make_request(
        "POST", f"/snapshot-contents/{digest_hex}?action=erase", data={}
    )
    assert erased.status_code == 200
    assert not object_path.exists()


def test_statetree_durable_manifest_budget_compacts_then_rejects_live_set(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-manifest-pressure"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 512
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    captured = server.make_request(
        "POST",
        f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot",
        data={},
    )
    digest = captured.body["digest"]
    digest_hex = digest.removeprefix("sha256:")
    assert server.make_request(
        "POST",
        f"/snapshots/{captured.body['snapshot_id']}?action=spill",
        data={"digest": digest},
    ).status_code == 200

    statuses = []
    for index in range(8):
        response = server.make_request(
            "POST",
            f"/snapshot-contents/{digest_hex}?action=retain",
            data={"owner": f"owner/{index}", "retention_class": "cache"},
        )
        statuses.append(response.status_code)
        if response.status_code == 503:
            break
    assert statuses[-1] == 503
    assert statuses.count(200) >= 1
    contents = server.make_request("GET", "/snapshot-contents")
    assert contents.status_code == 200
    assert contents.body["manifest_file_bytes"] <= 512
    assert contents.body["manifest_compactions"] >= 1
    assert len(contents.body["refs"]) == statuses.count(200)
    assert contents.body["contents"][0]["ref_count"] == statuses.count(200)


def test_statetree_managed_checkpoint_rejects_ambiguous_legacy_reachability(tmp_path):
    global server
    server.statetree_max_snapshot_bytes = 1 << 20
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-managed-schema-fence"
    server.statetree_max_snapshot_disk_bytes = 1 << 20
    server.statetree_max_snapshot_load_bytes = 1 << 20
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()
    namespace = next(path for path in tmp_path.iterdir() if path.is_dir())
    manifest_path = namespace / "manifest.v1"
    server.stop()

    append_legacy_empty_manifest_checkpoint(manifest_path, 1)
    with pytest.raises(RuntimeError):
        server.start()


def test_statetree_durable_io_shutdown_cancels_queue_without_partial_objects(tmp_path):
    global server
    server.n_slots = 2
    server.statetree_max_snapshot_bytes = 1 << 30
    server.statetree_snapshot_store = str(tmp_path)
    server.statetree_snapshot_compat_id = "unit-io-shutdown"
    server.statetree_max_snapshot_disk_bytes = 1 << 30
    server.statetree_max_snapshot_load_bytes = 1 << 30
    server.statetree_max_snapshot_manifest_bytes = 1 << 20
    server.start()

    assert completion(PREFIX, 0).status_code == 200
    fork = server.make_request("POST", "/slots/0?action=fork", data={"destinations": [1]})
    assert fork.status_code == 200
    captured = server.make_request(
        "POST",
        f"/nodes/{fork.body['nodes'][0]['node_id']}?action=snapshot",
        data={},
    )
    assert captured.status_code == 200

    def spill_request():
        return server.make_request(
            "POST",
            f"/snapshots/{captured.body['snapshot_id']}?action=spill",
            data={"digest": captured.body["digest"]},
            timeout=10,
        )

    executor = ThreadPoolExecutor(max_workers=8)
    futures = [executor.submit(spill_request) for _ in range(8)]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        states = server.make_request("GET", "/states")
        if states.body["durable_io_pending"] >= 2:
            break
        time.sleep(0.001)
    assert states.body["durable_io_pending"] >= 2

    t_start = time.monotonic()
    server.stop()
    assert time.monotonic() - t_start < 4
    for future in futures:
        try:
            future.result(timeout=2)
        except Exception:
            pass
    executor.shutdown(wait=True)

    # A running publish may finish, while queued jobs are canceled. Neither
    # outcome may leave a partial object or prevent clean restart discovery.
    assert not list(tmp_path.rglob(".tmp-*"))
    server.start()
    contents = server.make_request("GET", "/snapshot-contents")
    assert contents.status_code == 200
    assert len(contents.body["contents"]) <= 1
    assert not list(tmp_path.rglob(".tmp-*"))


def test_statetree_busy_family_is_pinned_across_deadline():
    global server
    server.n_slots = 3
    server.statetree_lease_ms = 50
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    fork_id = fork.body["fork_id"]

    def run_long_branch():
        return server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": PREFIX + " Give a detailed answer.",
                "cache_prompt": True,
                "id_slot": 1,
                "fork_id": fork_id,
                "n_predict": 256,
                "temperature": 0.0,
                "ignore_eos": True,
                "stop": [],
            },
            timeout=30,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        branch = executor.submit(run_long_branch)
        wait_for_slot_processing(1, True)
        time.sleep(0.075)

        slots = server.make_request("GET", "/slots")
        assert slots.status_code == 200
        for id_slot in (0, 1, 2):
            assert slots.body[id_slot]["is_reserved"] is True
            assert slots.body[id_slot]["lease_pinned"] is True
            assert slots.body[id_slot]["lease_expired"] is False
            assert slots.body[id_slot]["lease_remaining_ms"] == -1

        result = branch.result(timeout=30)
        assert result.status_code == 200

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert all(slots.body[id_slot]["is_reserved"] for id_slot in (0, 1, 2))
    assert all(slots.body[id_slot]["lease_remaining_ms"] > 0 for id_slot in (0, 1, 2))


def test_statetree_state_budget_is_exact_global_ceiling():
    global server
    server.server_metrics = True
    server.statetree_max_state_bytes = 1
    server.start()

    result = completion(PREFIX, 0)
    assert result.status_code == 200
    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    exact_state_bytes = sum(row["n_prompt_state_bytes"] for row in slots.body)

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert metric_value(metrics.body, "statetree_state_bytes") == exact_state_bytes
    assert metric_value(metrics.body, "statetree_state_budget_bytes") == 1
    assert metric_value(metrics.body, "statetree_state_high_water_bytes") <= 1
    assert exact_state_bytes <= 1
    if os.environ.get("LLAMA_SERVER_TEST_MODEL"):
        assert metric_value(metrics.body, "statetree_pressure_rejected_total") >= 1


def test_slot_fork_survives_idle_prompt_cache_sweep():
    global server
    server.n_slots = 4
    server.no_cache_idle_slots = False
    server.cache_ram = 64
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200

    automatic = completion("Trigger an idle prompt-cache sweep.")
    assert automatic.status_code == 200
    assert automatic.body["id_slot"] == 3

    for id_slot, suffix in ((0, " Source branch."), (1, " First branch."), (2, " Second branch.")):
        branch = completion(PREFIX + suffix, id_slot)
        assert branch.status_code == 200
        assert branch.body["timings"]["cache_n"] > 0


def test_slot_fork_deferred_request_suppresses_idle_sleep():
    global server
    server.n_slots = 2
    server.server_metrics = True
    server.sleep_idle_seconds = 1
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200

    with ThreadPoolExecutor(max_workers=1) as executor:
        deferred = executor.submit(completion, "Wait through the idle sleep threshold.", None, 10)
        wait_for_metric("requests_deferred", 1)
        time.sleep(1.5)

        slots = server.make_request("GET", "/slots")
        assert slots.status_code == 200
        assert all(slot["is_reserved"] for slot in slots.body)
        assert not deferred.done()

        erase = server.make_request("POST", "/slots/1?action=erase")
        assert erase.status_code == 200
        automatic = deferred.result(timeout=10)
        assert automatic.status_code == 200
        assert automatic.body["id_slot"] == 1

    time.sleep(2.5)
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["is_sleeping"] is False

    source_reuse = completion(PREFIX + " Continue from the protected root.", 0)
    assert source_reuse.status_code == 200
    assert source_reuse.body["timings"]["cache_n"] > 0


def test_slot_commit_preserves_winner_and_supports_refork(tmp_path):
    global server
    server.slot_save_path = f"{tmp_path}/"
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    fork_id = fork.body["fork_id"]

    source_branch = completion(PREFIX + " Answer as an atlas editor.", 0)
    winner_prompt = PREFIX + " Answer with only the city name."
    winner_branch = completion(winner_prompt, 1, return_tokens=True, n_probs=10)
    loser_branch = completion(PREFIX + " Answer in a complete sentence.", 2)
    assert source_branch.status_code == 200
    assert winner_branch.status_code == 200
    assert loser_branch.status_code == 200

    before_slots = server.make_request("GET", "/slots")
    assert before_slots.status_code == 200
    winner_checkpoint_count = before_slots.body[1]["n_prompt_checkpoints"]
    winner_cached_tokens = before_slots.body[1]["n_prompt_tokens"]
    winner_data_bytes = before_slots.body[1]["n_prompt_data_bytes"]
    winner_checkpoint_bytes = before_slots.body[1]["n_prompt_checkpoint_bytes"]
    winner_state_bytes = before_slots.body[1]["n_prompt_state_bytes"]
    assert winner_state_bytes == winner_data_bytes + winner_checkpoint_bytes
    for id_slot in (0, 1, 2, 3):
        row = before_slots.body[id_slot]
        assert row["n_prompt_data_bytes"] >= 0
        assert row["n_prompt_checkpoint_bytes"] >= 0
        assert row["n_prompt_state_bytes"] == (
            row["n_prompt_data_bytes"] + row["n_prompt_checkpoint_bytes"]
        )
    if os.environ.get("LLAMA_SERVER_TEST_MODEL"):
        for id_slot in (0, 1, 2):
            assert before_slots.body[id_slot]["n_prompt_checkpoints"] > 0
            assert before_slots.body[id_slot]["n_prompt_checkpoint_bytes"] > 0

    before_save = server.make_request(
        "POST",
        "/slots/1?action=save",
        data={"filename": "winner-before.bin"},
    )
    assert before_save.status_code == 200
    control_restore = server.make_request(
        "POST",
        "/slots/3?action=restore",
        data={"filename": "winner-before.bin"},
    )
    assert control_restore.status_code == 200

    committed = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": fork_id},
    )
    assert committed.status_code == 200
    assert set(committed.body) == {
        "id_slot",
        "state_id",
        "node_id",
        "parent_node_id",
        "source_id",
        "fork_id",
        "nodes",
        "released",
        "n_released",
        "n_cached_tokens",
        "timings",
    }
    assert committed.body["id_slot"] == 1
    assert committed.body["source_id"] == 0
    assert committed.body["fork_id"] == fork_id
    assert committed.body["released"] == [0, 2]
    assert committed.body["n_released"] == 2
    assert committed.body["n_cached_tokens"] == winner_cached_tokens
    assert committed.body["timings"]["commit_ms"] >= 0

    after_save = server.make_request(
        "POST",
        "/slots/1?action=save",
        data={"filename": "winner-after.bin"},
    )
    assert after_save.status_code == 200
    assert before_save.body["n_saved"] == after_save.body["n_saved"]
    assert before_save.body["n_written"] == after_save.body["n_written"]
    before_hash = hashlib.sha256((tmp_path / "winner-before.bin").read_bytes()).hexdigest()
    after_hash = hashlib.sha256((tmp_path / "winner-after.bin").read_bytes()).hexdigest()
    assert before_hash == after_hash

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[1]["is_reserved"] is True
    assert slots.body[1]["fork_source_id"] == 1
    assert slots.body[1]["fork_id"] == fork_id
    assert slots.body[1]["n_prompt_checkpoints"] == winner_checkpoint_count
    assert slots.body[1]["n_prompt_data_bytes"] == winner_data_bytes
    assert slots.body[1]["n_prompt_checkpoint_bytes"] == winner_checkpoint_bytes
    assert slots.body[1]["n_prompt_state_bytes"] == winner_state_bytes
    for id_slot in (0, 2):
        assert slots.body[id_slot]["is_reserved"] is False
        assert slots.body[id_slot]["fork_source_id"] == -1
        assert slots.body[id_slot]["fork_id"] == -1
        assert slots.body[id_slot].get("n_prompt_tokens", 0) == 0
        assert slots.body[id_slot]["n_prompt_checkpoints"] == 0
        assert slots.body[id_slot]["n_prompt_data_bytes"] == 0
        assert slots.body[id_slot]["n_prompt_checkpoint_bytes"] == 0
        assert slots.body[id_slot]["n_prompt_state_bytes"] == 0

    tokenized = server.make_request(
        "POST",
        "/tokenize",
        data={"content": winner_prompt, "add_special": True},
    )
    assert tokenized.status_code == 200
    assert len(winner_branch.body["tokens"]) == 1
    history = tokenized.body["tokens"] + winner_branch.body["tokens"]

    control = completion(
        history,
        3,
        n_predict=1,
        return_tokens=True,
        n_probs=10,
        cache_prompt=True,
    )
    continued = completion(
        history,
        1,
        n_predict=1,
        return_tokens=True,
        n_probs=10,
        cache_prompt=True,
    )
    assert control.status_code == 200
    assert continued.status_code == 200
    assert continued.body["tokens"] == control.body["tokens"]
    assert control.body["timings"]["cache_n"] == len(history) - 1
    assert control.body["timings"]["prompt_n"] == 1
    assert continued.body["timings"]["cache_n"] == len(history) - 1
    assert continued.body["timings"]["prompt_n"] == 1

    control_probs = control.body["completion_probabilities"][0]
    continued_probs = continued.body["completion_probabilities"][0]
    assert continued_probs["id"] == control_probs["id"]
    assert continued_probs["logprob"] == pytest.approx(control_probs["logprob"], abs=0.05)
    control_top = {item["id"]: item["logprob"] for item in control_probs["top_logprobs"]}
    continued_top = {item["id"]: item["logprob"] for item in continued_probs["top_logprobs"]}
    assert {item["id"] for item in continued_probs["top_logprobs"][:5]} == {
        item["id"] for item in control_probs["top_logprobs"][:5]
    }
    assert continued_top.keys() == control_top.keys()
    for token_id, logprob in continued_top.items():
        assert logprob == pytest.approx(control_top[token_id], abs=0.25)

    missing_generation = server.make_request(
        "POST",
        "/slots/1?action=fork",
        data={"destinations": [0, 2]},
    )
    assert missing_generation.status_code == 503
    wrong_generation = server.make_request(
        "POST",
        "/slots/1?action=fork",
        data={"destinations": [0, 2], "fork_id": fork_id + 1000},
    )
    assert wrong_generation.status_code == 503

    refork = server.make_request(
        "POST",
        "/slots/1?action=fork",
        data={"destinations": [0, 2], "fork_id": fork_id},
    )
    assert refork.status_code == 200
    assert refork.body["fork_id"] != fork_id

    slots_after_refork = server.make_request("GET", "/slots")
    assert slots_after_refork.status_code == 200
    for id_slot in (0, 1, 2):
        assert slots_after_refork.body[id_slot]["n_prompt_checkpoints"] == 0
        assert slots_after_refork.body[id_slot]["n_prompt_checkpoint_bytes"] == 0
        assert slots_after_refork.body[id_slot]["n_prompt_state_bytes"] == (
            slots_after_refork.body[id_slot]["n_prompt_data_bytes"]
        )

    stale = server.make_request(
        "POST",
        "/slots/0?action=commit",
        data={"fork_id": fork_id},
    )
    assert stale.status_code == 503

    slots_after_stale = server.make_request("GET", "/slots")
    assert slots_after_stale.status_code == 200
    for id_slot in (0, 1, 2):
        assert slots_after_stale.body[id_slot]["is_reserved"] is True
        assert slots_after_stale.body[id_slot]["fork_source_id"] == 1
        assert slots_after_stale.body[id_slot]["fork_id"] == refork.body["fork_id"]

    suffix = server.make_request(
        "POST",
        "/tokenize",
        data={"content": " Continue.", "add_special": False},
    )
    assert suffix.status_code == 200
    refork_history = history + suffix.body["tokens"]
    for id_slot in (0, 2):
        branch = completion(refork_history, id_slot)
        assert branch.status_code == 200
        assert branch.body["timings"]["cache_n"] > 0

    recommitted = server.make_request(
        "POST",
        "/slots/2?action=commit",
        data={"fork_id": refork.body["fork_id"]},
    )
    assert recommitted.status_code == 200
    assert recommitted.body["released"] == [0, 1]

    final_slots = server.make_request("GET", "/slots")
    assert final_slots.status_code == 200
    assert final_slots.body[2]["is_reserved"] is True
    assert final_slots.body[2]["fork_source_id"] == 2
    assert final_slots.body[2]["fork_id"] == refork.body["fork_id"]
    for id_slot in (0, 1):
        assert final_slots.body[id_slot]["is_reserved"] is False
        assert final_slots.body[id_slot].get("n_prompt_tokens", 0) == 0


def test_slot_commit_validation_is_atomic_and_root_may_be_erased():
    global server
    server.n_slots = 3
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    fork_id = fork.body["fork_id"]
    before_failures = server.make_request("GET", "/slots")
    assert before_failures.status_code == 200

    invalid_requests = [
        [],
        {},
        {"fork_id": None},
        {"fork_id": True},
        {"fork_id": "1"},
        {"fork_id": 1.0},
        {"fork_id": -1},
    ]
    for data in invalid_requests:
        invalid = server.make_request(
            "POST",
            "/slots/1?action=commit",
            data=data,
        )
        assert invalid.status_code == 400

    stale = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": fork_id + 1000},
    )
    assert stale.status_code == 503

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    for id_slot in (0, 1, 2):
        for key in ("n_prompt_tokens", "n_prompt_checkpoints", "fork_source_id", "fork_id"):
            assert slots.body[id_slot].get(key, 0) == before_failures.body[id_slot].get(key, 0)
        assert slots.body[id_slot]["is_reserved"] is True
        assert slots.body[id_slot]["fork_source_id"] == 0
        assert slots.body[id_slot]["fork_id"] == fork_id

    erased_root = server.make_request("POST", "/slots/0?action=erase")
    assert erased_root.status_code == 200

    committed = server.make_request(
        "POST",
        "/slots/2?action=commit",
        data={"fork_id": fork_id},
    )
    assert committed.status_code == 200
    assert committed.body["id_slot"] == 2
    assert committed.body["source_id"] == 0
    assert committed.body["released"] == [1]

    retry = server.make_request(
        "POST",
        "/slots/2?action=commit",
        data={"fork_id": fork_id},
    )
    assert retry.status_code == 200
    assert retry.body["id_slot"] == 2
    assert retry.body["source_id"] == 2
    assert retry.body["released"] == []
    assert retry.body["n_released"] == 0


def test_slot_commit_isolates_generations_with_reused_anchor():
    global server
    server.n_slots = 5
    server.start()

    source_a = completion(PREFIX, 0)
    assert source_a.status_code == 200
    fork_a = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork_a.status_code == 200

    erased_source = server.make_request("POST", "/slots/0?action=erase")
    assert erased_source.status_code == 200
    source_b = completion("A completely different cached prompt.", 0)
    assert source_b.status_code == 200
    fork_b = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [3, 4]},
    )
    assert fork_b.status_code == 200
    assert fork_b.body["fork_id"] != fork_a.body["fork_id"]

    commit_a = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": fork_a.body["fork_id"]},
    )
    assert commit_a.status_code == 200
    assert commit_a.body["released"] == [2]

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[1]["fork_source_id"] == 1
    assert slots.body[1]["fork_id"] == fork_a.body["fork_id"]
    for id_slot in (0, 3, 4):
        assert slots.body[id_slot]["fork_source_id"] == 0
        assert slots.body[id_slot]["fork_id"] == fork_b.body["fork_id"]

    commit_b = server.make_request(
        "POST",
        "/slots/3?action=commit",
        data={"fork_id": fork_b.body["fork_id"]},
    )
    assert commit_b.status_code == 200
    assert commit_b.body["released"] == [0, 4]

    final_slots = server.make_request("GET", "/slots")
    assert final_slots.status_code == 200
    for id_slot, fork_id in ((1, fork_a.body["fork_id"]), (3, fork_b.body["fork_id"])):
        assert final_slots.body[id_slot]["is_reserved"] is True
        assert final_slots.body[id_slot]["fork_source_id"] == id_slot
        assert final_slots.body[id_slot]["fork_id"] == fork_id
    for id_slot in (0, 2, 4):
        assert final_slots.body[id_slot]["is_reserved"] is False
        assert final_slots.body[id_slot]["fork_source_id"] == -1
        assert final_slots.body[id_slot]["fork_id"] == -1
        assert final_slots.body[id_slot].get("n_prompt_tokens", 0) == 0
        assert final_slots.body[id_slot]["n_prompt_checkpoints"] == 0


def test_committed_root_erase_and_restore_release_protection(tmp_path):
    global server
    server.n_slots = 3
    server.slot_save_path = f"{tmp_path}/"
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 200

    branch_prompt = PREFIX + " Answer with only the city name."
    branch = completion(branch_prompt, 1)
    assert branch.status_code == 200
    committed = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": fork.body["fork_id"]},
    )
    assert committed.status_code == 200

    saved = server.make_request(
        "POST",
        "/slots/1?action=save",
        data={"filename": "committed-root.bin"},
    )
    assert saved.status_code == 200
    restored = server.make_request(
        "POST",
        "/slots/1?action=restore",
        data={"filename": "committed-root.bin"},
    )
    assert restored.status_code == 200

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[1]["is_reserved"] is False
    assert slots.body[1]["fork_source_id"] == -1
    assert slots.body[1]["fork_id"] == -1
    assert slots.body[1]["state_id"] == -1
    assert slots.body[1]["n_prompt_tokens"] == saved.body["n_saved"]
    assert slots.body[1]["n_prompt_checkpoints"] == 0

    states = server.make_request("GET", "/states")
    assert states.status_code == 200
    assert states.body["states"] == []
    assert [entry["event"] for entry in states.body["journal"]] == [
        "fork",
        "commit",
        "restore",
    ]

    stale = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": fork.body["fork_id"]},
    )
    assert stale.status_code == 503

    restored_reuse = completion(branch_prompt + " Repeat it.", 1)
    assert restored_reuse.status_code == 200
    assert restored_reuse.body["timings"]["cache_n"] > 0

    refork = server.make_request(
        "POST",
        "/slots/1?action=fork",
        data={"destinations": [0]},
    )
    assert refork.status_code == 200
    recommitted = server.make_request(
        "POST",
        "/slots/1?action=commit",
        data={"fork_id": refork.body["fork_id"]},
    )
    assert recommitted.status_code == 200
    assert recommitted.body["released"] == [0]

    erased = server.make_request("POST", "/slots/1?action=erase")
    assert erased.status_code == 200
    assert erased.body["n_erased"] > 0

    final_slots = server.make_request("GET", "/slots")
    assert final_slots.status_code == 200
    assert final_slots.body[1]["is_reserved"] is False
    assert final_slots.body[1]["fork_source_id"] == -1
    assert final_slots.body[1]["fork_id"] == -1
    assert final_slots.body[1].get("n_prompt_tokens", 0) == 0
    assert final_slots.body[1]["n_prompt_checkpoints"] == 0


def test_slot_commit_rejects_busy_family_atomically():
    global server
    server.n_slots = 3
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200
    fork_id = fork.body["fork_id"]
    winner_node_id = next(
        node["node_id"] for node in fork.body["nodes"] if node["id_slot"] == 2
    )

    def run_long_branch():
        return server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": PREFIX + " Give a detailed answer.",
                "cache_prompt": True,
                "id_slot": 1,
                "n_predict": 128,
                "temperature": 0.0,
                "ignore_eos": True,
                "stop": [],
            },
            timeout=30,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        branch = executor.submit(run_long_branch)
        wait_for_slot_processing(1, True)

        rejected = server.make_request(
            "POST",
            f"/nodes/{winner_node_id}?action=commit",
            data={},
        )
        assert rejected.status_code == 503

        slots = server.make_request("GET", "/slots")
        assert slots.status_code == 200
        for id_slot in (0, 1, 2):
            assert slots.body[id_slot]["is_reserved"] is True
            assert slots.body[id_slot]["fork_source_id"] == 0
            assert slots.body[id_slot]["fork_id"] == fork_id

        branch_result = branch.result(timeout=30)
        assert branch_result.status_code == 200

    committed = server.make_request(
        "POST",
        "/slots/2?action=commit",
        data={"fork_id": fork_id},
    )
    assert committed.status_code == 200
    assert committed.body["released"] == [0, 1]


def test_slot_commit_wakes_one_deferred_request_per_loser():
    global server
    server.n_slots = 3
    server.server_metrics = True
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200
    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1, 2]},
    )
    assert fork.status_code == 200

    with ThreadPoolExecutor(max_workers=2) as executor:
        deferred_a = executor.submit(completion, "First deferred request.", None, 15)
        deferred_b = executor.submit(completion, "Second deferred request.", None, 15)
        wait_for_metric("requests_deferred", 2)

        committed = server.make_request(
            "POST",
            "/slots/1?action=commit",
            data={"fork_id": fork.body["fork_id"]},
        )
        assert committed.status_code == 200
        assert committed.body["released"] == [0, 2]

        result_a = deferred_a.result(timeout=15)
        result_b = deferred_b.result(timeout=15)
        assert result_a.status_code == 200
        assert result_b.status_code == 200
        assert {result_a.body["id_slot"], result_b.body["id_slot"]} == {0, 2}
        wait_for_metric("requests_deferred", 0)

    slots = server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert slots.body[1]["is_reserved"] is True
    assert slots.body[1]["fork_source_id"] == 1
    assert slots.body[1]["fork_id"] == fork.body["fork_id"]


def test_slot_fork_requires_unified_kv():
    global server
    server.kv_unified = False
    server.n_slots = 2
    server.start()

    source = completion(PREFIX, 0)
    assert source.status_code == 200

    fork = server.make_request(
        "POST",
        "/slots/0?action=fork",
        data={"destinations": [1]},
    )
    assert fork.status_code == 501

    commit = server.make_request(
        "POST",
        "/slots/0?action=commit",
        data={"fork_id": 0},
    )
    assert commit.status_code == 501
