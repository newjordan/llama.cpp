import hashlib
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


def completion(
    prompt: str | list[int],
    id_slot: int | None = None,
    timeout: float | None = None,
    *,
    n_predict: int = 1,
    return_tokens: bool = False,
    n_probs: int = 0,
    cache_prompt: bool = True,
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
        "fork_id",
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
        "source_id",
        "fork_id",
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
    assert [item["id"] for item in continued_probs["top_logprobs"][:5]] == [
        item["id"] for item in control_probs["top_logprobs"][:5]
    ]
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
    assert slots.body[1]["n_prompt_tokens"] == saved.body["n_saved"]
    assert slots.body[1]["n_prompt_checkpoints"] == 0

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
            "/slots/2?action=commit",
            data={"fork_id": fork_id},
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
