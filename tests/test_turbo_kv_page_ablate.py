import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-kv-page-ablate.py"
SPEC = importlib.util.spec_from_file_location("turbo_kv_page_ablate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ablate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ablate
SPEC.loader.exec_module(ablate)


def request(label: str, content: str, tokens: list[int], ok: bool = True) -> dict:
    return {
        "ok": ok,
        "label": label,
        "content": content,
        "tokens": tokens,
        "content_sha256": ablate.hash_content(content),
        "tokens_sha256": ablate.hash_tokens(tokens),
        "error": None if ok else "failed",
    }


class CompletionCaptureTests(unittest.TestCase):
    def test_completion_captures_tokens_content_and_hashes(self) -> None:
        response = {
            "content": "alpha",
            "tokens": [1, 2, 3],
            "timings": {
                "prompt_n": 7,
                "predicted_n": 3,
                "prompt_per_second": 10.0,
                "predicted_per_second": 20.0,
            },
        }
        with mock.patch.object(ablate, "http_json", return_value=response) as http_json:
            result = ablate.send_completion(8093, "test", 8, 3, 17)

        payload = http_json.call_args.args[2]
        self.assertTrue(payload["return_tokens"])
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "alpha")
        self.assertEqual(result.tokens, [1, 2, 3])
        self.assertEqual(result.content_sha256, ablate.hash_content("alpha"))
        self.assertEqual(result.tokens_sha256, ablate.hash_tokens([1, 2, 3]))

    def test_completion_rejects_missing_or_malformed_outputs(self) -> None:
        bad_responses = [
            {"tokens": [1]},
            {"content": "alpha"},
            {"content": "alpha", "tokens": [True]},
            {"content": "alpha", "tokens": "1"},
        ]
        for response in bad_responses:
            with self.subTest(response=response):
                with mock.patch.object(ablate, "http_json", return_value=response):
                    result = ablate.send_completion(8093, "test", 8, 3, 17)
                self.assertFalse(result.ok)
                self.assertIsNotNone(result.error)

    def test_nonfragmented_wave_pins_each_request_to_a_slot(self) -> None:
        slot_ids: list[int | None] = []

        def fake_completion(
            port: int,
            label: str,
            prompt_tokens: int,
            gen_tokens: int,
            seed: int,
            id_slot: int | None = None,
            cache_prompt: bool = False,
        ) -> object:
            del port, prompt_tokens, gen_tokens, seed, cache_prompt
            slot_ids.append(id_slot)
            return ablate.RequestResult(True, label, 0.1, 1, 1, 1.0, 1.0, "x", [1])

        with mock.patch.object(ablate, "send_completion", side_effect=fake_completion):
            results = ablate.run_wave(8093, 0, 3, [8], 2)

        self.assertEqual(len(results), 3)
        self.assertEqual(sorted(slot_ids), [0, 1, 2])


class OutputParityTests(unittest.TestCase):
    def test_identical_outputs_pass_and_token_changes_fail(self) -> None:
        reference = [request("same", "alpha", [1, 2])]
        matched = ablate.compare_request_outputs(reference, [request("same", "alpha", [1, 2])])
        changed = ablate.compare_request_outputs(reference, [request("same", "alpha", [1, 3])])

        self.assertTrue(matched["passed"])
        self.assertFalse(changed["passed"])
        self.assertIn("tokens", changed["mismatches"][0]["reasons"])
        self.assertNotIn("content", changed["mismatches"][0]["reasons"])

    def test_duplicate_labels_are_compared_by_ordinal(self) -> None:
        reference = [request("repeat", "one", [1]), request("repeat", "two", [2])]
        same = [request("repeat", "one", [1]), request("repeat", "two", [2])]
        swapped = [request("repeat", "two", [2]), request("repeat", "one", [1])]

        self.assertTrue(ablate.compare_request_outputs(reference, same)["passed"])
        self.assertEqual(ablate.compare_request_outputs(reference, swapped)["mismatch_count"], 2)

    def test_fragment_fill_mismatch_fails_aggregate_parity(self) -> None:
        baseline = {
            "variant": "baseline",
            "fragmentation": {"fill_requests": [request("fill", "one", [1])]},
            "requests": [request("measured", "same", [2])],
        }
        candidate = {
            "variant": "indexed",
            "fragmentation": {"fill_requests": [request("fill", "changed", [3])]},
            "requests": [request("measured", "same", [2])],
        }

        parity = ablate.compare_output_parity(baseline, candidate)
        self.assertFalse(parity["passed"])
        self.assertFalse(parity["phases"]["fragment_fill"]["passed"])
        self.assertTrue(parity["phases"]["measured"]["passed"])

    def test_warmup_mismatch_fails_aggregate_parity(self) -> None:
        baseline = {
            "variant": "baseline",
            "warmup_request": request("warmup", "one", [1]),
            "fragmentation": {"enabled": False},
            "requests": [request("measured", "same", [2])],
        }
        control = {
            "variant": "control",
            "warmup_request": request("warmup", "changed", [3]),
            "fragmentation": {"enabled": False},
            "requests": [request("measured", "same", [2])],
        }

        parity = ablate.compare_output_parity(baseline, control)

        self.assertFalse(parity["passed"])
        self.assertFalse(parity["phases"]["warmup"]["passed"])
        self.assertTrue(parity["phases"]["measured"]["passed"])

    def test_failed_reference_request_fails_reference_integrity(self) -> None:
        failed = {
            "variant": "baseline",
            "fragmentation": {"enabled": False},
            "requests": [request("measured", "", [], ok=False)],
        }

        parity = ablate.reference_output_parity(failed)

        self.assertEqual(parity["status"], "reference")
        self.assertFalse(parity["passed"])
        self.assertIn("request_error", parity["phases"]["measured"]["mismatches"][0]["reasons"])


class VariantResolutionTests(unittest.TestCase):
    def test_attach_is_honest_and_non_destructive(self) -> None:
        self.assertEqual(ablate.resolve_variants("attached", True, 0), ["attached"])
        with self.assertRaisesRegex(ValueError, "controlled ablations"):
            ablate.resolve_variants("baseline,indexed", True, 0)
        with self.assertRaisesRegex(ValueError, "erases live slot state"):
            ablate.resolve_variants("attached", True, 4)
        with self.assertRaisesRegex(ValueError, "requires --attach"):
            ablate.resolve_variants("attached", False, 0)

    def test_managed_variants_require_unique_baseline_first(self) -> None:
        self.assertEqual(ablate.resolve_variants("indexed,baseline,probe", False, 0), ["baseline", "indexed", "probe"])
        self.assertEqual(
            ablate.resolve_variants("indexed,control,baseline", False, 0),
            ["baseline", "control", "indexed"],
        )
        with self.assertRaisesRegex(ValueError, "require baseline"):
            ablate.resolve_variants("probe,indexed", False, 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ablate.resolve_variants("baseline,baseline", False, 0)


class MainParityTests(unittest.TestCase):
    @staticmethod
    def result(variant: str, tokens: list[int], ok: bool = True) -> dict:
        req = request("measured", "same", tokens, ok=ok)
        return {
            "kind": "turbo-kv-page-ablation",
            "schema_version": 2,
            "variant": variant,
            "run_id": 0,
            "fragmentation": {"enabled": False},
            "request_summary": {"failed_requests": 0 if ok else 1},
            "probe_summary": {},
            "log_path": f"{variant}.log",
            "requests": [req],
        }

    def test_main_writes_schema_v2_and_fails_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                str(SCRIPT_PATH),
                "--out-dir", temp_dir,
                "--variants", "baseline,probe",
                "--repeats", "1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ablate,
                    "run_variant",
                    side_effect=[self.result("baseline", [1]), self.result("probe", [2])],
                ),
            ):
                returncode = ablate.main()

            rows = [json.loads(line) for line in (Path(temp_dir) / "results.jsonl").read_text().splitlines()]

        self.assertEqual(returncode, 1)
        self.assertEqual([row["schema_version"] for row in rows], [2, 2])
        self.assertEqual(rows[0]["output_parity"]["status"], "reference")
        self.assertEqual(rows[1]["output_parity"]["status"], "mismatched")

    def test_lone_failed_baseline_fails_reference_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                str(SCRIPT_PATH),
                "--out-dir", temp_dir,
                "--variants", "baseline",
                "--repeats", "1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(ablate, "run_variant", return_value=self.result("baseline", [], ok=False)),
            ):
                returncode = ablate.main()

            row = json.loads((Path(temp_dir) / "results.jsonl").read_text())

        self.assertEqual(returncode, 1)
        self.assertEqual(row["output_parity"]["status"], "reference")
        self.assertFalse(row["output_parity"]["passed"])

    def test_completed_baseline_is_persisted_before_later_variant_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                str(SCRIPT_PATH),
                "--out-dir", temp_dir,
                "--variants", "baseline,probe",
                "--repeats", "1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ablate,
                    "run_variant",
                    side_effect=[self.result("baseline", [1]), RuntimeError("candidate launch failed")],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "candidate launch failed"):
                    ablate.main()

            rows = [json.loads(line) for line in (Path(temp_dir) / "results.jsonl").read_text().splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant"], "baseline")

    def test_zero_work_counts_are_rejected_by_argparse(self) -> None:
        for flag in ("--waves", "--repeats", "--parallel", "--gen-tokens"):
            with self.subTest(flag=flag):
                with mock.patch.object(sys, "argv", [str(SCRIPT_PATH), flag, "0"]):
                    with self.assertRaises(SystemExit):
                        ablate.main()

    def test_unpaired_attached_measurement_fails_on_request_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                str(SCRIPT_PATH),
                "--attach",
                "--out-dir", temp_dir,
                "--variants", "attached",
                "--repeats", "1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(ablate, "run_variant", return_value=self.result("attached", [], ok=False)),
            ):
                returncode = ablate.main()

        self.assertEqual(returncode, 1)


class AttachedPreflightTests(unittest.TestCase):
    def test_run_variant_rejects_more_fixed_slots_than_server_has(self) -> None:
        args = mock.Mock(
            attach=True,
            port=8093,
            startup_timeout=1.0,
            parallel=4,
            fragment_slots=0,
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(ablate, "wait_healthy"),
            mock.patch.object(ablate, "http_json", return_value={"total_slots": 2}),
        ):
            with self.assertRaisesRegex(RuntimeError, "server reports 2"):
                ablate.run_variant(args, "attached", 0, Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
