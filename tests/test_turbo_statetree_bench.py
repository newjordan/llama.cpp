import argparse
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-statetree-bench.py"
SPEC = importlib.util.spec_from_file_location("turbo_statetree_bench", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def metric(value: float) -> dict:
    return {"n": 1, "min": value, "p50": value, "p95": value, "max": value, "mean": value}


def result(label: str, fork_ms: float, branch_tps: float, rss: float, cleanup_ms: float) -> dict:
    return {
        "kind": "turbo-statetree-benchmark",
        "schema_version": bench.RESULT_SCHEMA_VERSION,
        "label": label,
        "config": {"model": "test.gguf", "ctx": 4096, "parallel": 2},
        "server": {"commit": label},
        "summary": {
            "failed_samples": 0,
            "commit_supported": True,
            "groups": {
                "p128:manual": {
                    "prefix_tokens": 128,
                    "metrics": {
                        "fork_server_ms": metric(fork_ms),
                        "branch_aggregate_predicted_tps": metric(branch_tps),
                        "rss_before_cleanup_bytes": metric(rss),
                        "cleanup_client_ms": metric(cleanup_ms),
                    },
                },
                "p128:commit": {
                    "prefix_tokens": 128,
                    "metrics": {"cleanup_client_ms": metric(cleanup_ms / 2)},
                },
            },
        },
    }


class ParsingTests(unittest.TestCase):
    def test_csv_and_mode_parsers_reject_ambiguous_input(self) -> None:
        self.assertEqual(bench.parse_csv_ints("128, 1024"), [128, 1024])
        self.assertEqual(bench.parse_modes("manual,commit"), ["manual", "commit"])
        with self.assertRaises(argparse.ArgumentTypeError):
            bench.parse_csv_ints("128,128")
        with self.assertRaises(argparse.ArgumentTypeError):
            bench.parse_modes("manual,unknown")

    def test_percentiles_are_interpolated_and_empty_safe(self) -> None:
        self.assertIsNone(bench.percentile([], 0.95))
        self.assertEqual(bench.percentile([7.0], 0.95), 7.0)
        self.assertEqual(bench.percentile([0.0, 10.0], 0.50), 5.0)
        summary = bench.summarize_values([1.0, 2.0, 3.0])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["p50"], 2.0)

    def test_layout_resolves_dense_and_fragmented_families(self) -> None:
        self.assertEqual(bench.resolve_family_ids(4, 2, "dense"), [0, 1, 2])
        self.assertEqual(bench.resolve_family_ids(8, 2, "fragmented"), [1, 3, 5])
        with self.assertRaisesRegex(ValueError, r"2 \*"):
            bench.resolve_family_ids(5, 2, "fragmented")

    def test_transaction_plan_rotates_a_shared_manual_commit_winner(self) -> None:
        self.assertEqual(bench.transaction_slot_plan([1, 3, 5], 0), (1, [3, 5], 1))
        self.assertEqual(bench.transaction_slot_plan([1, 3, 5], 1), (3, [1, 5], 3))
        self.assertEqual(bench.transaction_slot_plan([1, 3, 5], 2), (5, [1, 3], 5))


class TelemetryTests(unittest.TestCase):
    def test_runtime_provenance_hashes_shared_server_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "llama-server"
            binary.write_bytes(b"launcher")
            (root / "libllama-server-impl.so").write_bytes(b"server implementation")
            (root / "libllama.so").write_bytes(b"llama")
            (root / "libllama-bench-impl.so").write_bytes(b"unrelated benchmark")

            identities = bench.bundled_runtime_identities(str(binary))

        self.assertEqual(set(identities), {"libllama-server-impl.so", "libllama.so"})
        self.assertNotEqual(
            identities["libllama-server-impl.so"]["sha256"],
            identities["libllama.so"]["sha256"],
        )

    def test_slot_summary_exposes_exact_memory_totals(self) -> None:
        rows = [
            {
                "id": 0,
                "is_processing": False,
                "is_reserved": True,
                "fork_source_id": 0,
                "fork_id": 17,
                "n_prompt_checkpoints": 2,
                "n_prompt_tokens": 100,
                "n_prompt_data_bytes": 10,
                "n_prompt_checkpoint_bytes": 20,
                "n_prompt_state_bytes": 30,
            },
            {
                "id": 1,
                "is_processing": False,
                "is_reserved": True,
                "fork_source_id": 0,
                "fork_id": 17,
                "n_prompt_checkpoints": 1,
                "n_prompt_tokens": 110,
                "n_prompt_data_bytes": 40,
                "n_prompt_checkpoint_bytes": 50,
                "n_prompt_state_bytes": 90,
            },
        ]
        summary = bench.summarize_slots(rows)

        self.assertEqual(summary["reserved"], 2)
        self.assertEqual(summary["prompt_tokens"], 210)
        self.assertEqual(summary["prompt_checkpoint_bytes"], 70)
        self.assertEqual(summary["prompt_state_bytes"], 120)
        self.assertEqual(summary["by_id"]["1"]["fork_id"], 17)

    def test_old_server_without_byte_fields_reports_unknown_not_zero(self) -> None:
        summary = bench.summarize_slots([{"id": 0, "is_reserved": False}])
        self.assertIsNone(summary["prompt_data_bytes"])
        self.assertIsNone(summary["prompt_checkpoint_bytes"])
        self.assertIsNone(summary["prompt_state_bytes"])

    def test_prometheus_parser_ignores_comments_and_labels(self) -> None:
        body = """
# HELP llamacpp:kv_cache_usage_ratio ratio
llamacpp:kv_cache_usage_ratio 0.25
llamacpp:requests_processing{model=\"x\"} 3
garbage
"""
        self.assertEqual(
            bench.parse_prometheus(body),
            {"kv_cache_usage_ratio": 0.25, "requests_processing": 3.0},
        )

    def test_drm_fdinfo_parser_normalizes_xe_vram_bytes(self) -> None:
        parsed = bench.parse_drm_fdinfo("""
drm-driver:\txe
drm-client-id:\t11773
drm-total-vram0:\t100 KiB
drm-resident-vram0:\t80 KiB
""")
        self.assertEqual(
            parsed,
            {
                "driver": "xe",
                "client_id": "11773",
                "total_vram0_bytes": 100 * 1024,
                "resident_vram0_bytes": 80 * 1024,
            },
        )

    def test_shared_kv_estimate_counts_prefix_once_and_suffixes_per_branch(self) -> None:
        snapshot = {
            "all_slots": {
                "by_id": {
                    "0": {"n_prompt_tokens": 69},
                    "1": {"n_prompt_tokens": 70},
                    "2": {"n_prompt_tokens": 71},
                }
            }
        }
        self.assertEqual(bench.expected_shared_kv_cells(snapshot, 64, [0, 1, 2]), 82)


class RequestTests(unittest.TestCase):
    def test_launch_records_ragged_control_and_probe_environment(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(["run", "--label", "env-test", "--no-tree-ragged", "--kv-page-probe"])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(bench, "port_open", return_value=False), \
                mock.patch.object(bench.subprocess, "Popen") as popen:
            process = mock.Mock()
            popen.return_value = process
            bench.launch_server(args, Path(directory) / "server.log")
            environment = popen.call_args.kwargs["env"]

        self.assertEqual(environment["LLAMA_KV_TREE_RAGGED"], "0")
        self.assertEqual(environment["LLAMA_KV_PAGE_PROBE"], "1")

    def test_required_metrics_reject_missing_nonfinite_and_boolean_values(self) -> None:
        for value in (None, True, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    bench.required_nonnegative_number({"value": value}, "value", "test value")
        self.assertEqual(bench.required_nonnegative_number({"value": 0}, "value", "test value"), 0.0)

    def test_exact_tokens_grows_input_and_trims_exactly(self) -> None:
        with mock.patch.object(bench, "tokenize", side_effect=[[1, 2], list(range(20))]) as tokenize:
            tokens = bench.exact_tokens(8098, 10, "test")
        self.assertEqual(tokens, list(range(10)))
        self.assertEqual(tokenize.call_count, 2)

    def test_completion_pins_slot_and_records_cache_reuse(self) -> None:
        response = {
            "id_slot": 2,
            "tokens": [7, 8],
            "timings": {
                "prompt_n": 4,
                "cache_n": 100,
                "predicted_n": 2,
                "prompt_per_second": 10.0,
                "predicted_per_second": 20.0,
            },
        }
        with mock.patch.object(
            bench,
            "timed_json",
            return_value={"client_ms": 3.0, "response": response},
        ) as timed_json:
            measured = bench.completion(8098, [1, 2, 3], 2, 2, 17, 30.0, fork_id=42)

        payload = timed_json.call_args.args[2]
        self.assertEqual(payload["id_slot"], 2)
        self.assertEqual(payload["fork_id"], 42)
        self.assertTrue(payload["cache_prompt"])
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(measured["cache_n"], 100)
        self.assertEqual(measured["predicted_n"], 2)

    def test_zero_predict_prefill_accepts_mandatory_prompt_logit_token(self) -> None:
        response = {
            "id_slot": 0,
            "tokens": [7],
            "timings": {
                "prompt_n": 64,
                "cache_n": 0,
                "predicted_n": 1,
            },
        }
        with mock.patch.object(
            bench,
            "timed_json",
            return_value={"client_ms": 3.0, "response": response},
        ):
            measured = bench.completion(8098, list(range(64)), 0, 0, 17, 30.0)

        self.assertEqual(measured["predicted_n"], 1)


class ValidationTests(unittest.TestCase):
    @staticmethod
    def snapshot(reserved: int, rows: dict[str, dict]) -> dict:
        return {
            "all_slots": {"by_id": rows},
            "family_slots": {"reserved": reserved},
        }

    def test_commit_validation_checks_winner_losers_and_generation(self) -> None:
        winner = {
            "n_prompt_tokens": 140,
            "n_prompt_checkpoints": 2,
            "n_prompt_state_bytes": 4096,
        }
        empty = {"n_prompt_tokens": 0, "n_prompt_state_bytes": 0}
        sample = {
            "family_ids": [0, 1, 2],
            "prefix_tokens": 128,
            "winner_id": 2,
            "cleanup_mode": "commit",
            "supported": True,
            "fork": {"fork_id": 10},
            "refork": {"fork_id": 11},
            "branch_wave": {"min_cache_n": 128},
            "cleanup": {"response": {"released": [0, 1]}},
            "snapshots": {
                "after_fork": self.snapshot(3, {}),
                "before_cleanup": self.snapshot(3, {"2": winner}),
                "after_cleanup": self.snapshot(1, {"0": empty, "1": empty, "2": winner}),
                "final": self.snapshot(0, {}),
            },
        }
        self.assertEqual(bench.validate_sample(sample), [])

        sample["refork"]["fork_id"] = 10
        sample["snapshots"]["after_cleanup"]["all_slots"]["by_id"]["1"]["n_prompt_state_bytes"] = 1
        failures = bench.validate_sample(sample)
        self.assertIn("refork did not advance the generation", failures)
        self.assertIn("loser slot 1 retained prompt state bytes", failures)

    def test_unsupported_parent_commit_is_not_a_false_failure(self) -> None:
        sample = {
            "family_ids": [0, 1],
            "prefix_tokens": 128,
            "supported": False,
            "branch_wave": {"min_cache_n": 128},
            "snapshots": {"after_fork": self.snapshot(2, {})},
        }
        self.assertEqual(bench.validate_sample(sample), [])

    def test_manual_validation_catches_loser_and_final_state_leaks(self) -> None:
        winner = {
            "n_prompt_tokens": 140,
            "n_prompt_checkpoints": 2,
            "n_prompt_state_bytes": 4096,
        }
        empty = {"n_prompt_tokens": 0, "n_prompt_state_bytes": 0}
        sample = {
            "family_ids": [0, 1],
            "prefix_tokens": 128,
            "winner_id": 0,
            "cleanup_mode": "manual",
            "supported": True,
            "fork": {"fork_id": 10},
            "refork": None,
            "branch_wave": {"min_cache_n": 128},
            "cleanup": {},
            "snapshots": {
                "after_fork": self.snapshot(2, {}),
                "before_cleanup": self.snapshot(2, {"0": winner, "1": empty}),
                "after_cleanup": self.snapshot(1, {"0": winner, "1": empty}),
                "final": self.snapshot(0, {}),
            },
        }
        self.assertEqual(bench.validate_sample(sample), [])

        manual_leak = copy.deepcopy(sample)
        manual_leak["snapshots"]["after_cleanup"]["all_slots"]["by_id"]["1"] = {
            "n_prompt_tokens": 1,
            "n_prompt_state_bytes": 64,
        }
        failures = bench.validate_sample(manual_leak)
        self.assertIn("loser slot 1 retained prompt tokens", failures)
        self.assertIn("loser slot 1 retained prompt state bytes", failures)

        final_leak = copy.deepcopy(sample)
        final_leak["snapshots"]["final"]["all_slots"]["by_id"]["0"] = {
            "n_prompt_state_bytes": 64,
        }
        self.assertIn("final cleanup left prompt state bytes", bench.validate_sample(final_leak))

        budget_leak = copy.deepcopy(sample)
        after_fork = budget_leak["snapshots"]["after_fork"]
        after_fork["all_slots"]["prompt_state_bytes"] = 2
        after_fork["server_metrics"] = {
            "statetree_state_bytes": 2,
            "statetree_state_budget_bytes": 1,
            "statetree_state_high_water_bytes": 2,
        }
        failures = bench.validate_sample(budget_leak)
        self.assertIn("snapshot after_fork exceeded the StateTree state budget", failures)
        self.assertIn("snapshot after_fork StateTree high-water exceeded the state budget", failures)


class SummaryAndComparisonTests(unittest.TestCase):
    def test_summary_excludes_unsupported_commit_samples_from_metrics(self) -> None:
        samples = [
            {
                "prefix_tokens": 128,
                "cleanup_mode": "manual",
                "supported": True,
                "failures": [],
                "fork": {"client_ms": 2.0, "server_ms": 1.0},
                "prefill": {"client_ms": 3.0, "prompt_tps": 10.0},
                "branch_wave": {"wall_ms": 4.0, "aggregate_predicted_tps": 20.0},
                "cleanup": {"client_ms": 5.0},
                "snapshots": {},
            },
            {
                "prefix_tokens": 128,
                "cleanup_mode": "commit",
                "supported": False,
                "failures": [],
                "error": None,
            },
        ]
        summary = bench.summarize_samples(samples)
        self.assertEqual(summary["groups"]["p128:manual"]["usable_samples"], 1)
        self.assertEqual(summary["groups"]["p128:commit"]["usable_samples"], 0)
        self.assertFalse(summary["commit_supported"])

    def test_summary_excludes_contract_failures_and_requires_every_commit_sample(self) -> None:
        valid = {
            "prefix_tokens": 128,
            "cleanup_mode": "commit",
            "supported": True,
            "failures": [],
            "fork": {"server_ms": 1.0},
            "snapshots": {},
        }
        failed = copy.deepcopy(valid)
        failed["fork"]["server_ms"] = 999.0
        failed["failures"] = ["contract failure"]

        summary = bench.summarize_samples([valid, failed])
        group = summary["groups"]["p128:commit"]
        self.assertEqual(group["usable_samples"], 1)
        self.assertEqual(group["metrics"]["fork_server_ms"]["n"], 1)
        self.assertEqual(group["metrics"]["fork_server_ms"]["p50"], 1.0)
        self.assertFalse(summary["commit_supported"])

    def test_comparison_applies_latency_throughput_and_rss_gates(self) -> None:
        baseline = result("base", 1.0, 100.0, 1000.0, 8.0)
        candidate = result("candidate", 1.1, 98.0, 1200.0, 4.0)
        baseline["config"].update({"bin": "/tmp/parent", "port": 8098, "modes": ["manual"]})
        candidate["config"].update({
            "bin": "/tmp/candidate",
            "port": 8099,
            "modes": ["manual", "commit"],
        })
        comparison = bench.compare_results(baseline, candidate, 0.95, 1.10, 0.25, 1.0, 64.0)

        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["commit_vs_manual"][0]["speedup"], 2.0)

        candidate["summary"]["groups"]["p128:manual"]["metrics"][
            "branch_aggregate_predicted_tps"
        ] = metric(80.0)
        regression = bench.compare_results(baseline, candidate, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertFalse(regression["passed"])

    def test_comparison_rejects_kind_and_schema_mismatches(self) -> None:
        baseline = result("base", 1.0, 100.0, 1000.0, 8.0)

        wrong_kind = result("candidate", 1.0, 100.0, 1000.0, 4.0)
        wrong_kind["kind"] = "other-benchmark"
        comparison = bench.compare_results(baseline, wrong_kind, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("kind mismatch" in error for error in comparison["compatibility"]["errors"]))

        wrong_schema = result("candidate", 1.0, 100.0, 1000.0, 4.0)
        wrong_schema["schema_version"] += 1
        comparison = bench.compare_results(baseline, wrong_schema, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("schema mismatch" in error for error in comparison["compatibility"]["errors"]))

    def test_comparison_rejects_workload_config_mismatch(self) -> None:
        baseline = result("base", 1.0, 100.0, 1000.0, 8.0)
        candidate = result("candidate", 1.0, 100.0, 1000.0, 4.0)
        candidate["config"]["ctx"] = 8192

        comparison = bench.compare_results(baseline, candidate, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("config mismatch for ctx" in error for error in comparison["compatibility"]["errors"]))

    def test_comparison_allows_candidate_only_retention_controls(self) -> None:
        baseline = result("base", 1.0, 100.0, 1000.0, 8.0)
        candidate = result("candidate", 1.0, 100.0, 1000.0, 4.0)
        baseline["config"].update({"statetree_lease_ms": 0, "statetree_max_state_bytes": 0})
        candidate["config"].update({
            "statetree_lease_ms": 30000,
            "statetree_max_state_bytes": 1073741824,
        })

        comparison = bench.compare_results(baseline, candidate, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertTrue(comparison["compatibility"]["passed"])

    def test_comparison_rejects_missing_baseline_manual_group(self) -> None:
        baseline = result("base", 1.0, 100.0, 1000.0, 8.0)
        candidate = result("candidate", 1.0, 100.0, 1000.0, 4.0)
        extra_group = copy.deepcopy(baseline["summary"]["groups"]["p128:manual"])
        extra_group["prefix_tokens"] = 1024
        baseline["summary"]["groups"]["p1024:manual"] = extra_group

        comparison = bench.compare_results(baseline, candidate, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertFalse(comparison["passed"])
        self.assertIn(
            "candidate is missing baseline manual group p1024:manual",
            comparison["compatibility"]["errors"],
        )

    def test_comparison_rejects_missing_mandatory_metric(self) -> None:
        baseline = result("base", 1.0, 100.0, 1000.0, 8.0)
        candidate = result("candidate", 1.0, 100.0, 1000.0, 4.0)
        del candidate["summary"]["groups"]["p128:manual"]["metrics"][
            "branch_aggregate_predicted_tps"
        ]

        comparison = bench.compare_results(baseline, candidate, 0.95, 1.10, 0.25, 1.0, 64.0)
        self.assertFalse(comparison["passed"])
        self.assertIn(
            "candidate group p128:manual is missing mandatory metric branch_aggregate_predicted_tps",
            comparison["compatibility"]["errors"],
        )


class SafetyTests(unittest.TestCase):
    def test_attach_requires_explicit_destructive_permission(self) -> None:
        args = argparse.Namespace(
            attach=True,
            allow_destructive_attach=False,
            port=8098,
        )
        with self.assertRaisesRegex(SystemExit, "erases slot state"):
            bench.run_benchmark(args)

    def test_production_port_is_refused(self) -> None:
        args = argparse.Namespace(
            attach=False,
            allow_destructive_attach=False,
            port=8093,
            allow_production_port=False,
        )
        with self.assertRaisesRegex(SystemExit, "production port"):
            bench.run_benchmark(args)

    def test_persistent_fragmentation_requires_fragmented_layout(self) -> None:
        args = argparse.Namespace(
            attach=False,
            allow_destructive_attach=False,
            port=8098,
            persistent_fragmentation=True,
            layout="dense",
        )
        with self.assertRaisesRegex(SystemExit, "requires --layout fragmented"):
            bench.run_benchmark(args)

    def test_extra_server_args_cannot_override_controlled_shape(self) -> None:
        args = argparse.Namespace(
            attach=False,
            allow_destructive_attach=False,
            port=8098,
            extra_server_args="--ctx-size 1 --statetree-lease-ms=1",
        )
        with self.assertRaisesRegex(SystemExit, "cannot override controlled options"):
            bench.run_benchmark(args)


if __name__ == "__main__":
    unittest.main()
