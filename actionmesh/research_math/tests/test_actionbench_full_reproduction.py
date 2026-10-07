import json
from pathlib import Path
import sys
import tempfile
import unittest


ACTIONMESH = Path(__file__).resolve().parents[2]
if str(ACTIONMESH) not in sys.path:
    sys.path.insert(0, str(ACTIONMESH))

from research_math.actionbench_full_reproduction import (
    assess_official_reproduction,
    published_interval,
    validate_contract_data,
)


class FullPopulationReproductionContractTests(unittest.TestCase):
    def setUp(self):
        self.uids = [f"uid-{index:03d}" for index in range(128)]
        self.audit = {
            "published_aggregate_contract": {
                "source_path": "official/README.md",
                "population": "all 128 animated objects",
                "actionmesh_seed": 42,
                "metrics": {"cd_3d": 0.054, "cd_4d": 0.085,
                            "cd_motion": 0.153},
                "aggregation": "mean over successful samples as implemented by evaluator",
            }
        }
        self.population = {
            "dataset": "facebook/actionbench",
            "revision": "dataset-revision",
            "uids": self.uids,
        }
        self.contract = {
            "kind": "actionbench-full-population-reproduction-contract",
            "version": "1.0.0",
            "purpose": "official-full-population-baseline-reproduction",
            "benchmark": {
                "id": "facebook/actionbench",
                "revision": "dataset-revision",
                "population_size": 128,
                "frames_per_sample": 16,
                "population_ref": {"path": "population.json", "sha256": "0" * 64},
            },
            "method": {"name": "ActionMesh", "generation_seed": 42},
            "official_scorer": {
                "sampling_seed": 44,
                "n_pts_icp": 10000,
                "n_pts_chamfer": 100000,
                "metrics": ["cd_3d", "cd_4d", "cd_motion"],
            },
            "published_target": {
                "source_path": "official/README.md",
                "decimal_places": 3,
                "metrics": {"cd_3d": 0.054, "cd_4d": 0.085,
                            "cd_motion": 0.153},
            },
            "decision_rule": {
                "complete_population_required": True,
                "required_total": 128,
                "required_successes": 128,
                "required_failures": 0,
                "metric_rule": "published_rounding_interval",
                "all_metrics_required": True,
                "uncertainty": "finite_population_reproduction_no_sampling_ci",
            },
            "scientific_effect_qualification": False,
            "candidate_methods_tested": False,
            "dispatch_ready": False,
        }

    def test_published_interval_is_the_three_decimal_rounding_bin(self):
        self.assertEqual(published_interval(0.054, 3), (0.0535, 0.0545))

    def test_contract_separates_generation_and_scorer_seeds(self):
        validate_contract_data(self.contract, self.population, self.audit)
        changed = json.loads(json.dumps(self.contract))
        changed["official_scorer"]["sampling_seed"] = 42
        with self.assertRaisesRegex(ValueError, "sampling seed 44"):
            validate_contract_data(changed, self.population, self.audit)

    def test_contract_requires_the_complete_released_population(self):
        incomplete = dict(self.population, uids=self.uids[:-1])
        with self.assertRaisesRegex(ValueError, "128 unique"):
            validate_contract_data(self.contract, incomplete, self.audit)

    def test_contract_rejects_paper_values_in_place_of_current_readme_values(self):
        changed = json.loads(json.dumps(self.contract))
        changed["published_target"]["metrics"] = {
            "cd_3d": 0.053, "cd_4d": 0.081, "cd_motion": 0.148}
        with self.assertRaisesRegex(ValueError, "current official README"):
            validate_contract_data(changed, self.population, self.audit)

    def test_reproduction_rejects_a_one_uid_or_success_only_denominator(self):
        rows = [{"uid": self.uids[0], "status": "success", "n_frames": 16,
                 "cd_3d": 0.054, "cd_4d": 0.085, "cd_motion": 0.153}]
        summary = {"n_total": 1, "n_success": 1, "n_failed": 0,
                   "cd_3d_mean": 0.054, "cd_4d_mean": 0.085,
                   "cd_motion_mean": 0.153}
        with self.assertRaisesRegex(ValueError, "complete released population"):
            assess_official_reproduction(self.contract, self.population, rows, summary)

    def test_reproduction_requires_every_uid_to_succeed(self):
        rows = [
            {"uid": uid, "status": "success", "n_frames": 16,
             "cd_3d": 0.054, "cd_4d": 0.085, "cd_motion": 0.153}
            for uid in self.uids
        ]
        rows[-1].update(status="error", cd_3d=float("nan"))
        summary = {"n_total": 128, "n_success": 127, "n_failed": 1,
                   "cd_3d_mean": 0.054, "cd_4d_mean": 0.085,
                   "cd_motion_mean": 0.153}
        with self.assertRaisesRegex(ValueError, "128 successful"):
            assess_official_reproduction(self.contract, self.population, rows, summary)

    def test_reproduction_uses_all_three_published_rounding_intervals(self):
        rows = [
            {"uid": uid, "status": "success", "n_frames": 16,
             "cd_3d": 0.0541, "cd_4d": 0.0848, "cd_motion": 0.1534}
            for uid in self.uids
        ]
        summary = {"n_total": 128, "n_success": 128, "n_failed": 0,
                   "cd_3d_mean": 0.0541, "cd_4d_mean": 0.0848,
                   "cd_motion_mean": 0.1534}
        result = assess_official_reproduction(
            self.contract, self.population, rows, summary)
        self.assertEqual(result["decision"], "reproduced_within_published_precision")
        self.assertEqual(set(result["metric_checks"]),
                         {"cd_3d", "cd_4d", "cd_motion"})
        self.assertFalse(result["scientific_effect_qualification"])

    def test_one_metric_outside_the_interval_fails_reproduction(self):
        rows = [
            {"uid": uid, "status": "success", "n_frames": 16,
             "cd_3d": 0.0541, "cd_4d": 0.086, "cd_motion": 0.1534}
            for uid in self.uids
        ]
        summary = {"n_total": 128, "n_success": 128, "n_failed": 0,
                   "cd_3d_mean": 0.0541, "cd_4d_mean": 0.086,
                   "cd_motion_mean": 0.1534}
        result = assess_official_reproduction(
            self.contract, self.population, rows, summary)
        self.assertEqual(result["decision"], "not_reproduced")
        self.assertFalse(result["metric_checks"]["cd_4d"]["passed"])

    def test_assessment_rejects_a_weakened_decision_rule(self):
        changed = json.loads(json.dumps(self.contract))
        changed["decision_rule"]["required_successes"] = 127
        rows = [
            {"uid": uid, "status": "success", "n_frames": 16,
             "cd_3d": 0.054, "cd_4d": 0.085, "cd_motion": 0.153}
            for uid in self.uids
        ]
        summary = {"n_total": 128, "n_success": 128, "n_failed": 0,
                   "cd_3d_mean": 0.054, "cd_4d_mean": 0.085,
                   "cd_motion_mean": 0.153}
        with self.assertRaisesRegex(ValueError, "decision rule"):
            assess_official_reproduction(changed, self.population, rows, summary)


if __name__ == "__main__":
    unittest.main()
