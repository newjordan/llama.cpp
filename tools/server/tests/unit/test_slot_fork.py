import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from utils import *


server = ServerPreset.tinyllama2()

PREFIX = "What is the capital of France?"


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


def completion(prompt: str, id_slot: int | None = None, timeout: float | None = None):
    data = {
        "prompt": prompt,
        "cache_prompt": True,
        "n_predict": 1,
        "temperature": 0.0,
    }
    if id_slot is not None:
        data["id_slot"] = id_slot
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
        "destinations",
        "n_destinations",
        "n_tokens",
        "timings",
    }
    assert fork.body["id_slot"] == 0
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

    with ThreadPoolExecutor(max_workers=1) as executor:
        deferred = executor.submit(completion, "This request must use the released branch.", None, 10)
        wait_for_metric("requests_deferred", 1)
        assert not deferred.done()

        erase = server.make_request("POST", "/slots/1?action=erase")
        assert erase.status_code == 200
        assert erase.body["id_slot"] == 1
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
