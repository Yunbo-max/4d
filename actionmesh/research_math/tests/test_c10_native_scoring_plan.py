"""Local-only fail-closed contracts for the C10 scoring plan profile."""
from __future__ import annotations

import unittest

import prepare_c10_native_scoring as planner


class C10NativeScoringPlanTests(unittest.TestCase):
    def contract(self):
        names = {
            "treatment": "pinned_integrable_solve", "baseline": "b_star",
            "b0": "b0", "direct_common_lift": "direct_common_lift",
            "independent_local_repair": "independent_local_repair",
        }
        refs = {key: [{"path": name + ".py", "sha256": chr(97 + index) * 64}]
                for index, (key, name) in enumerate(names.items())}
        refs["baseline"] = refs["direct_common_lift"]
        contract = {
            "benchmark_id": "facebook/actionbench", "benchmark_revision": "rev",
            "primary_metric": "cd_3d",
            "metrics": [{"name": name, "output_path": ["readout", name]}
                        for name in ("cd_3d", "cd_4d", "cd_motion")],
            "contrasts": {
                "treatment": "pinned_integrable_solve", "baseline": "b_star",
                "controls": ["b0", "direct_common_lift",
                             "independent_local_repair"],
            },
            "arm_requirements": {
                key: {"name": name, "revision": name + "-method",
                      "implementation_refs": refs[key]}
                for key, name in names.items()
            },
            "scorer": {"kind": "official", "source_refs": [], "code_refs": []},
        }
        contract["arm_requirements"]["baseline"]["revision"] = (
            "direct_common_lift-method")
        return contract

    def comparison(self):
        rows = []
        hashes = {
            "b0": "a" * 64, "direct_common_lift": "c" * 64,
            "independent_local_repair": "d" * 64,
            "pinned_integrable_solve": "e" * 64,
        }
        for role in ("b0", "b_star", "direct_common_lift",
                     "independent_local_repair", "pinned_integrable_solve"):
            row = {"role": role, "method_id": role + "-method"}
            if role == "b_star":
                row.update(method_id="direct_common_lift-method",
                           alias_of="direct_common_lift")
            else:
                row.update(report_ref={"path": role + "/report.json",
                                       "sha256": hashes[role]},
                           implementation_sha256=hashes[role])
            rows.append(row)
        rows[-1]["method_id"] = "pinned_integrable_solve-method"
        return {"roles": rows}

    def test_contract_fixes_candidate_baseline_controls_and_official_scorer(self):
        planner.require_c14_contract(self.contract(), benchmark_revision="rev")
        for mutation in ("treatment", "baseline", "controls", "scorer"):
            changed = self.contract()
            if mutation == "treatment":
                changed["contrasts"]["treatment"] = "direct_common_lift"
            elif mutation == "baseline":
                changed["contrasts"]["baseline"] = "direct_common_lift"
            elif mutation == "controls":
                changed["contrasts"]["controls"].pop()
            else:
                changed.pop("scorer")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                planner.require_c14_contract(changed, benchmark_revision="rev")

    def test_frozen_arm_bindings_cover_every_physical_implementation(self):
        refs = planner.bind_contract_arms(self.contract(), self.comparison())
        self.assertEqual({ref["path"] for ref in refs}, {
            "b0.py", "direct_common_lift.py",
            "independent_local_repair.py", "pinned_integrable_solve.py"})
        changed = self.contract()
        changed["arm_requirements"]["treatment"]["revision"] = "caller-relabel"
        with self.assertRaisesRegex(ValueError, "method identity"):
            planner.bind_contract_arms(changed, self.comparison())

    def test_method_boundary_requires_c10_design_verification(self):
        ready = {"workflow_boundary": {
            "action": "dispatch", "candidate_id": "4d-math-20261006-c10",
            "required_stage": "design_verified", "ready": True}}
        planner.require_method_boundary(ready)
        blocked = {"workflow_boundary": {**ready["workflow_boundary"],
                                          "ready": False,
                                          "reason_codes": ["DESIGN_REVIEW_REQUIRED"]}}
        with self.assertRaisesRegex(ValueError, "design-verified"):
            planner.require_method_boundary(blocked)

    def test_scientific_job_is_one_confirmation_attempt_zero_retry(self):
        job, limits = planner.scientific_job_contract(
            {"evidence_mode": "prospective_confirmatory"}, wall_seconds=900)
        self.assertEqual(job, {"arm_role": "treatment"})
        self.assertEqual(limits["max_development_trials"], 0)
        self.assertEqual(limits["max_confirmation_trials"], 1)
        self.assertEqual(limits["max_retries_per_trial"], 0)

    def test_outputs_are_receipt_bound_raw_bundle_not_a_verdict(self):
        self.assertEqual(planner.scoring_outputs(), [
            "actionmesh/c10-scoring-output/result.json",
            "actionmesh/c10-scoring-output/raw-manifest.json",
            "actionmesh/c10-scoring-output/raw-evidence.tar",
        ])
        self.assertEqual(planner.AUTHORIZATION_SCOPE,
                         "single_c10_scoring_attempt")
        self.assertEqual(planner.FAMILY_SPLIT_KIND, "c10-family-split")
        self.assertEqual(planner.ANALYSIS_KIND, "c10-g01-analysis-plan")
        self.assertEqual(planner.ADMISSION_KIND,
                         "c10-scientific-dispatch-admission")


if __name__ == "__main__":
    unittest.main()
