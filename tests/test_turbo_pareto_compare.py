import argparse
import copy
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-pareto-compare.py"
SPEC = importlib.util.spec_from_file_location("turbo_pareto_compare", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
pareto_compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pareto_compare
SPEC.loader.exec_module(pareto_compare)


def stats(value: float) -> dict:
    return {
        "n": 5,
        "min": value - 1.0,
        "mean": value,
        "p50": value,
        "p95": value + 1.0,
        "max": value + 2.0,
    }


def row(
    agents: int,
    *,
    aggregate: float = 100.0,
    mean_slot: float = 40.0,
    min_slot: float = 35.0,
    p95_client_ms: float = 1000.0,
) -> dict:
    return {
        "active_agents": agents,
        "samples": 5,
        "failed_samples": 0,
        "aggregate_wall_tps": stats(aggregate),
        "mean_slot_tps": stats(mean_slot),
        "min_slot_tps": stats(min_slot),
        "p95_client_ms": stats(p95_client_ms),
    }


def pareto(agents: list[int]) -> dict:
    return {
        "schema_version": 1,
        "kind": "turbo-multiagent-pareto",
        "label": "fixture",
        "passed": True,
        "config": {
            "ctx": 262144,
            "n_predict": 256,
            "parallel": 12,
            "prompt": "fixed prompt",
            "repeats": 5,
            "agents": agents,
        },
        "summary": {"rows": [row(agent) for agent in agents]},
    }


def args(**changes: object) -> argparse.Namespace:
    values = {"max_regression_pct": 1.0}
    values.update(changes)
    return argparse.Namespace(**values)


class ParetoComparisonTests(unittest.TestCase):
    def test_intersection_emits_absolute_metrics_and_percent_deltas(self) -> None:
        reference = pareto([1, 8, 12])
        candidate = pareto([1, 12])
        candidate["summary"]["rows"] = [
            row(1, aggregate=105.0, mean_slot=42.0, min_slot=36.0, p95_client_ms=950.0),
            row(12, aggregate=104.0, mean_slot=41.0, min_slot=36.0, p95_client_ms=975.0),
        ]

        result = pareto_compare.compare(reference, candidate, args())

        self.assertTrue(result["passed"])
        self.assertEqual(result["intersecting_agents"], [1, 12])
        self.assertEqual(result["reference_only_agents"], [8])
        aggregate = result["rows"][0]["metrics"]["aggregate_wall_tps"]
        self.assertEqual(aggregate["reference"]["p50"], 100.0)
        self.assertEqual(aggregate["candidate"]["p50"], 105.0)
        self.assertEqual(aggregate["absolute_delta"], 5.0)
        self.assertEqual(aggregate["percent_delta"], 5.0)
        self.assertTrue(aggregate["passed"])

    def test_throughput_regression_exceeding_threshold_fails(self) -> None:
        reference = pareto([8])
        candidate = pareto([8])
        candidate["summary"]["rows"] = [row(8, aggregate=98.0)]

        result = pareto_compare.compare(reference, candidate, args(max_regression_pct=1.0))

        self.assertFalse(result["passed"])
        self.assertIn("agents_8_aggregate_wall_tps_regression", result["failed_gates"])
        aggregate = result["rows"][0]["metrics"]["aggregate_wall_tps"]
        self.assertEqual(aggregate["percent_delta"], -2.0)
        self.assertEqual(aggregate["regression_pct"], 2.0)

    def test_latency_uses_lower_is_better_direction(self) -> None:
        reference = pareto([8])
        candidate = pareto([8])
        candidate["summary"]["rows"] = [row(8, p95_client_ms=1030.0)]

        result = pareto_compare.compare(reference, candidate, args(max_regression_pct=2.0))

        self.assertFalse(result["passed"])
        self.assertIn("agents_8_p95_client_ms_regression", result["failed_gates"])
        latency = result["rows"][0]["metrics"]["p95_client_ms"]
        self.assertEqual(latency["direction"], "lower_is_better")
        self.assertEqual(latency["percent_delta"], 3.0)
        self.assertEqual(latency["regression_pct"], 3.0)

    def test_workload_mismatch_fails_before_claiming_a_valid_comparison(self) -> None:
        reference = pareto([1])
        candidate = copy.deepcopy(reference)
        candidate["config"]["ctx"] = 131072

        result = pareto_compare.compare(reference, candidate, args())

        self.assertFalse(result["passed"])
        self.assertIn("workload_identity", result["failed_gates"])

    def test_missing_agent_intersection_fails(self) -> None:
        reference = pareto([1])
        candidate = pareto([2])

        result = pareto_compare.compare(reference, candidate, args())

        self.assertFalse(result["passed"])
        self.assertIn("intersecting_agents", result["failed_gates"])
        self.assertEqual(result["rows"], [])


if __name__ == "__main__":
    unittest.main()
