import argparse
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-speculative-breakout.py"
SPEC = importlib.util.spec_from_file_location("turbo_speculative_breakout", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
breakout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = breakout
SPEC.loader.exec_module(breakout)


class TerminalObjectivePassTests(unittest.TestCase):
    def test_json_terminal_pass_requires_one_strict_json_document(self) -> None:
        case = {"validator": "json_exact", "expected": {"ok": True}}
        valid = '{"ok":true}'
        validation = breakout.validate_objective(case, valid)

        self.assertTrue(breakout.is_terminal_objective_pass(case, valid, validation))
        self.assertFalse(breakout.validate_objective(case, 'answer: {"ok":true}')["passed"])
        self.assertFalse(breakout.validate_objective(case, '```json\n{"ok":true}\n```')["passed"])
        self.assertFalse(breakout.validate_objective(case, '{"ok":false,"ok":true}')["passed"])
        self.assertFalse(breakout.is_terminal_objective_pass(case, 'answer: {"ok":true}'))
        self.assertFalse(breakout.is_terminal_objective_pass(case, '```json\n{"ok":true}\n```'))
        self.assertFalse(breakout.is_terminal_objective_pass(case, '{"ok":false,"ok":true}'))
        self.assertFalse(breakout.is_terminal_objective_pass(case, ""))

    def test_json_exact_rejects_boolean_integer_aliases_at_any_depth(self) -> None:
        case = {
            "validator": "json_exact",
            "expected": {"enabled": True, "nested": [False, {"kept": True}]},
        }
        invalid = '{"enabled":1,"nested":[0,{"kept":1}]}'
        validation = breakout.validate_objective(case, invalid)

        self.assertFalse(validation["passed"])
        self.assertFalse(validation["checks"]["exact_match"])
        self.assertFalse(validation["checks"]["path_root_enabled"])
        self.assertFalse(validation["checks"]["path_root_nested_0"])
        self.assertFalse(breakout.is_terminal_objective_pass(case, invalid, validation))

    def test_partial_fixed_validator_is_never_terminal(self) -> None:
        lines = [
            "- Baseline: 1",
            "- Branch fanout: 2",
            "- Verifier quality: 3",
            "- Pass/fail gate: 4",
        ]
        case = {"validator": "rubric_signals"}

        self.assertFalse(breakout.is_terminal_objective_pass(case, "\n".join(lines)))
        self.assertFalse(breakout.is_terminal_objective_pass(case, "Rubric\n" + "\n".join(lines)))

        arithmetic = '{"alpha":410,"beta":70,"gamma":987,"checksum":1467}'
        self.assertTrue(breakout.validate_objective({"validator": "arithmetic_json"}, arithmetic)["passed"])
        self.assertFalse(breakout.is_terminal_objective_pass({"validator": "arithmetic_json"}, arithmetic))

    def test_exact_lines_contract_can_be_terminal(self) -> None:
        case = {"validator": "lines_exact", "expected_lines": ["alpha", "beta"]}

        self.assertTrue(breakout.is_terminal_objective_pass(case, "alpha\nbeta"))
        self.assertFalse(breakout.is_terminal_objective_pass(case, "alpha\nbeta\nextra"))


class CompletionContractTests(unittest.TestCase):
    def test_completion_records_exact_tokens_and_stable_hashes(self) -> None:
        response = {
            "content": "Answer: done",
            "id_slot": 2,
            "tokens": [17, 23, 19],
            "timings": {
                "cache_n": 2,
                "prompt_n": 4,
                "predicted_n": 3,
                "prompt_per_second": 8.0,
                "predicted_per_second": 6.0,
            },
        }
        with mock.patch.object(breakout, "http_json", return_value=response) as http_json:
            result = breakout.completion(
                8093,
                "test",
                "prompt",
                3,
                2,
                True,
                0.0,
                1,
                1.0,
                7,
                10.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.raw_content, "Answer: done")
        self.assertEqual(result.content, "done")
        self.assertEqual(result.tokens, [17, 23, 19])
        self.assertEqual(result.cache_tokens, 2)
        self.assertEqual(
            result.raw_content_sha256,
            hashlib.sha256(b"Answer: done").hexdigest(),
        )
        self.assertEqual(result.content_sha256, hashlib.sha256(b"done").hexdigest())
        self.assertEqual(
            result.tokens_sha256,
            hashlib.sha256(json.dumps([17, 23, 19], separators=(",", ":")).encode("ascii")).hexdigest(),
        )
        self.assertTrue(http_json.call_args.args[2]["return_tokens"])
        self.assertFalse(http_json.call_args.args[2]["ignore_eos"])
        self.assertEqual(http_json.call_args.args[2]["stop"], breakout.DEFAULT_STOPS)

    def test_completion_forwards_ignore_eos_true(self) -> None:
        response = {
            "content": "done",
            "tokens": [17],
            "timings": {"predicted_n": 1},
        }
        with mock.patch.object(breakout, "http_json", return_value=response) as http_json:
            result = breakout.completion(
                8093,
                "test",
                "prompt",
                1,
                None,
                False,
                0.0,
                1,
                1.0,
                7,
                10.0,
                ignore_eos=True,
            )

        self.assertTrue(result.ok)
        self.assertTrue(http_json.call_args.args[2]["ignore_eos"])

    def test_completion_can_disable_stop_strings(self) -> None:
        response = {
            "content": "done",
            "tokens": [17],
            "timings": {"predicted_n": 1},
        }
        with mock.patch.object(breakout, "http_json", return_value=response) as http_json:
            result = breakout.completion(
                8093,
                "test",
                "prompt",
                1,
                None,
                False,
                0.0,
                1,
                1.0,
                7,
                10.0,
                ignore_eos=True,
                use_stop_strings=False,
            )

        self.assertTrue(result.ok)
        self.assertEqual(http_json.call_args.args[2]["stop"], [])

    def test_fanout_forwards_ignore_eos(self) -> None:
        args = argparse.Namespace(port=8093, request_timeout=10.0, ignore_eos=True, stop_strings=False)
        request = (
            "branch-00",
            "prompt",
            0,
            False,
            0.0,
            1,
            1.0,
            7,
            1,
        )
        result = breakout.RequestResult(
            ok=True,
            label="branch-00",
            id_slot=0,
            start=1.0,
            end=2.0,
            wall_s=1.0,
            raw_content="done",
            content="done",
            prompt_tokens=1,
            predicted_tokens=1,
            prompt_tps=1.0,
            predicted_tps=1.0,
        )

        with mock.patch.object(breakout, "completion", return_value=result) as completion:
            self.assertEqual(breakout.fanout(args, [request]), [result])

        self.assertTrue(completion.call_args.kwargs["ignore_eos"])
        self.assertFalse(completion.call_args.kwargs["use_stop_strings"])

    def test_completion_rejects_missing_or_malformed_tokens(self) -> None:
        for response in (
            {"content": "done", "timings": {}},
            {"content": "done", "tokens": [1, True], "timings": {}},
            {"content": 7, "tokens": [1], "timings": {}},
        ):
            with self.subTest(response=response):
                with mock.patch.object(breakout, "http_json", return_value=response):
                    result = breakout.completion(
                        8093,
                        "test",
                        "prompt",
                        1,
                        0,
                        False,
                        0.0,
                        1,
                        1.0,
                        7,
                        10.0,
                    )
                self.assertFalse(result.ok)
                self.assertIsNone(result.tokens)

    def test_completion_rejects_slot_or_token_count_mismatch(self) -> None:
        responses = (
            {"content": "done", "id_slot": 1, "tokens": [1], "timings": {"predicted_n": 1}},
            {"content": "done", "id_slot": 0, "tokens": [1], "timings": {"predicted_n": 2}},
        )
        for response in responses:
            with self.subTest(response=response):
                with mock.patch.object(breakout, "http_json", return_value=response):
                    result = breakout.completion(
                        8093,
                        "test",
                        "prompt",
                        1,
                        0,
                        False,
                        0.0,
                        1,
                        1.0,
                        7,
                        10.0,
                    )
                self.assertFalse(result.ok)


class BaselineShortCircuitTests(unittest.TestCase):
    @staticmethod
    def args() -> argparse.Namespace:
        return argparse.Namespace(
            port=8093,
            attach=True,
            prefix_clone=True,
            prefix_clone_backend="file",
            branch_slots="0-11",
            prefix_slot=0,
            final_slot=0,
            score_slot=0,
            branch_tokens=220,
            baseline_tokens=64,
            verify_tokens=120,
            final_tokens=260,
            score_tokens=120,
            ignore_eos=False,
            stop_strings=True,
            repair_rounds=0,
            objective_repair_rounds=1,
            verifier_mode="model",
            objective_fast_path=True,
            objective_baseline_short_circuit=True,
            objective_fast_fallback_recombine=True,
            objective_fast_fallback_model_verifier=True,
            objective_fast_fallback_repair_rounds=1,
            accept_score=70,
            verify_prefix_chars=1600,
            run_baseline=True,
            score_final=False,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            seed=19907,
            request_timeout=10.0,
            started="test-run",
        )

    def test_config_records_ignore_eos(self) -> None:
        args = self.args()
        args.ignore_eos = True
        args.stop_strings = False

        self.assertTrue(breakout.breakout_run_config(args, [0, 1], "none")["ignore_eos"])
        self.assertFalse(breakout.breakout_run_config(args, [0, 1], "none")["stop_strings"])

    def test_terminal_baseline_returns_without_fanout(self) -> None:
        baseline = breakout.RequestResult(
            ok=True,
            label="baseline-single",
            id_slot=0,
            start=1.0,
            end=2.0,
            wall_s=1.0,
            raw_content='{"ok":true}',
            content='{"ok":true}',
            prompt_tokens=8,
            predicted_tokens=4,
            prompt_tps=8.0,
            predicted_tps=4.0,
        )
        case = {
            "id": "terminal-json",
            "validator": "json_exact",
            "expected": {"ok": True},
            "task": "Return the expected JSON.",
        }

        with (
            mock.patch.object(breakout, "http_json", return_value={"total_slots": 12}),
            mock.patch.object(breakout, "completion", return_value=baseline) as completion,
            mock.patch.object(breakout, "fanout") as fanout,
            mock.patch.object(breakout, "slot_action") as slot_action,
        ):
            result = breakout.run_breakout(
                self.args(),
                case["task"],
                Path("/tmp"),
                Path("/tmp/turbo-short-circuit-test.log"),
                case,
            )

        completion.assert_called_once()
        fanout.assert_not_called()
        slot_action.assert_not_called()
        self.assertEqual(result["final_answer"], baseline.content)
        self.assertEqual(result["selected"]["source"], "baseline")
        self.assertIsNone(result["selected"]["prefix_ok"])
        self.assertEqual(result["schema_version"], 3)
        self.assertFalse(result["config"]["ignore_eos"])
        self.assertEqual(result["decision"]["short_circuit_phase"], "baseline")
        self.assertEqual(result["decision"]["terminal_validation_surface"], "cleaned_content")
        self.assertEqual(result["decision"]["branches_launched"], 0)
        self.assertEqual(result["summaries"]["branches"]["requests"], 0)
        self.assertTrue(result["objective_benchmark"]["breakout"]["passed"])
        self.assertIsNone(result["objective_benchmark"]["initial_breakout"])

        summary = breakout.summarize_accuracy_and_throughput([result])
        self.assertIsNone(summary["accuracy"]["final_multipass_pass_rate"])
        self.assertEqual(summary["accuracy"]["final_system_pass_rate"], 1.0)
        self.assertIsNone(summary["accuracy"]["initial_multipass_pass_rate"])
        self.assertIsNone(summary["accuracy"]["all_multipass_passed"])
        self.assertTrue(summary["accuracy"]["all_final_passed"])
        self.assertEqual(summary["accuracy"]["baseline_short_circuits"], 1)
        self.assertEqual(summary["throughput"]["branch_fanout"]["requests"], 0)
        self.assertEqual(summary["throughput"]["multipass_core"]["mean_all_tasks_wall_s"], 0.0)
        self.assertIsNone(summary["throughput"]["multipass_with_baseline"]["mean_escalated_task_wall_s"])
        self.assertEqual(
            breakout.summarize_decisions([result]),
            {
                "tasks": 1,
                "baseline_short_circuits": 1,
                "escalated_tasks": 0,
                "branches_launched": 0,
                "fanout_waves": 0,
            },
        )

    def test_requested_model_scoring_does_not_short_circuit(self) -> None:
        args = self.args()
        args.prefix_clone = False
        args.score_final = True
        baseline = breakout.RequestResult(
            ok=True,
            label="baseline-single",
            id_slot=0,
            start=1.0,
            end=2.0,
            wall_s=1.0,
            raw_content='{"ok":true}',
            content='{"ok":true}',
            prompt_tokens=8,
            predicted_tokens=4,
            prompt_tps=8.0,
            predicted_tps=4.0,
        )
        case = {
            "id": "terminal-json",
            "validator": "json_exact",
            "expected": {"ok": True},
            "task": "Return the expected JSON.",
        }

        with (
            mock.patch.object(breakout, "http_json", return_value={"total_slots": 12}),
            mock.patch.object(breakout, "completion", return_value=baseline),
            mock.patch.object(breakout, "fanout", side_effect=RuntimeError("fanout reached")) as fanout,
        ):
            with self.assertRaisesRegex(RuntimeError, "fanout reached"):
                breakout.run_breakout(
                    args,
                    case["task"],
                    Path("/tmp"),
                    Path("/tmp/turbo-short-circuit-test.log"),
                    case,
                )

        fanout.assert_called_once()

    def test_fork_backend_clones_prefix_without_slot_files(self) -> None:
        args = self.args()
        args.prefix_clone_backend = "fork"
        args.score_final = True
        baseline = breakout.RequestResult(
            ok=True,
            label="baseline-single",
            id_slot=0,
            start=1.0,
            end=2.0,
            wall_s=1.0,
            raw_content="baseline",
            content="baseline",
            prompt_tokens=8,
            predicted_tokens=4,
            prompt_tps=8.0,
            predicted_tps=4.0,
        )
        prefix = breakout.RequestResult(
            ok=True,
            label="prefix-eval",
            id_slot=0,
            start=2.0,
            end=3.0,
            wall_s=1.0,
            raw_content="",
            content="",
            prompt_tokens=8,
            predicted_tokens=0,
            prompt_tps=8.0,
            predicted_tps=None,
        )

        with (
            mock.patch.object(breakout, "http_json", return_value={"total_slots": 12}),
            mock.patch.object(breakout, "completion", side_effect=[baseline, prefix]),
            mock.patch.object(breakout, "slot_fork", return_value={"id_slot": 0}) as slot_fork,
            mock.patch.object(breakout, "slot_action") as slot_action,
            mock.patch.object(breakout, "fanout", side_effect=RuntimeError("fanout reached")),
        ):
            with self.assertRaisesRegex(RuntimeError, "fanout reached"):
                breakout.run_breakout(args, "task", Path("/tmp"), Path("/tmp/fork-test.log"))

        slot_fork.assert_called_once_with(8093, 0, list(range(1, 12)), timeout=10.0)
        slot_action.assert_not_called()

    def test_cleanup_erases_only_reserved_requested_slots_source_last(self) -> None:
        rows = [
            {"id": 0, "is_reserved": True},
            {"id": 1, "is_reserved": False},
            {"id": 2, "is_reserved": True},
            {"id": 3, "is_reserved": True},
        ]
        with (
            mock.patch.object(breakout, "http_json", return_value=rows),
            mock.patch.object(breakout, "slot_action", return_value={}) as slot_action,
        ):
            errors = breakout.cleanup_fork_reservations(8093, [0, 1, 2], 0, 10.0)

        self.assertEqual(errors, [])
        self.assertEqual(
            [call.args[1] for call in slot_action.call_args_list],
            [2, 0],
        )


class ReportingSemanticsTests(unittest.TestCase):
    @staticmethod
    def phase(wall_s: float) -> dict:
        return {
            "requests": 1,
            "ok_requests": 1,
            "total_prompt_tokens": 10,
            "total_predicted_tokens": 5,
            "parallel_wall_s": wall_s,
        }

    def test_system_and_escalated_metrics_have_distinct_denominators(self) -> None:
        passed = {"score": 100, "passed": True}
        failed = {"score": 0, "passed": False}
        short_circuit = {
            "decision": {"short_circuit_phase": "baseline"},
            "objective_benchmark": {
                "baseline": passed,
                "initial_breakout": None,
                "breakout": passed,
                "score_delta": 0,
            },
            "summaries": {"baseline": self.phase(1.0), "branches": None},
        }
        escalated = {
            "decision": {"short_circuit_phase": None},
            "objective_benchmark": {
                "baseline": failed,
                "initial_breakout": passed,
                "breakout": passed,
                "initial_score_delta": 100,
                "score_delta": 100,
            },
            "summaries": {"baseline": self.phase(1.0), "branches": self.phase(4.0)},
        }

        summary = breakout.summarize_accuracy_and_throughput([short_circuit, escalated])
        accuracy = summary["accuracy"]
        combined = summary["throughput"]["multipass_with_baseline"]

        self.assertEqual(accuracy["final_system_pass_rate"], 1.0)
        self.assertEqual(accuracy["final_multipass_pass_rate"], 1.0)
        self.assertEqual(accuracy["escalated_tasks"], 1)
        self.assertEqual(combined["mean_escalated_task_wall_s"], 5.0)
        self.assertEqual(combined["mean_all_tasks_wall_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
