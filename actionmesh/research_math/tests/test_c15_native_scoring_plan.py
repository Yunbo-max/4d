"""Static-fixture contracts for the admitted C15 scoring plan builder.

These tests do not invoke a scorer, harness, model, or GPU.  Local executes
them only through the common software-acceptance plan.
"""
from __future__ import annotations

import unittest

import prepare_c15_native_scoring as planner


class C15NativeScoringPlanTests(unittest.TestCase):
    def contract(self):
        names = {
            "treatment": "protected_residual_svt", "baseline": "b_star",
            "b0": "b0", "gaussian": "gaussian",
            "unprotected_svt": "unprotected_svt",
            "rank_matched_tsvd": "rank_matched_tsvd",
        }
        refs = {role: [{"path": role + ".py", "sha256": chr(97 + index) * 64}]
                for index, role in enumerate(names)}
        refs["baseline"] = refs["gaussian"]
        return {
            "benchmark_id": "facebook/actionbench", "benchmark_revision": "rev",
            "primary_metric": "cd_motion",
            "metrics": [{"name": name, "output_path": ["readout", name]}
                        for name in ("cd_3d", "cd_4d", "cd_motion")],
            "contrasts": {
                "treatment": "protected_residual_svt", "baseline": "b_star",
                "controls": ["b0", "gaussian", "unprotected_svt",
                             "rank_matched_tsvd"],
            },
            "arm_requirements": {
                role: {"name": name,
                       "revision": ("gaussian-method" if role == "baseline"
                                    else name + "-method"),
                       "implementation_refs": refs[role]}
                for role, name in names.items()},
            "scorer": {"kind": "official", "source_refs": [], "code_refs": []},
        }

    def comparison(self):
        rows = []
        for role, sha in zip(
                ("b0", "b_star", "gaussian", "unprotected_svt",
                 "rank_matched_tsvd", "protected_residual_svt"),
                ("c", "d", "d", "e", "f", "a")):
            row = {"role": role, "method_id": role + "-method"}
            if role == "b_star":
                row.update(method_id="gaussian-method", alias_of="gaussian")
            else:
                row.update(report_ref={"path": role + "/report.json",
                                       "sha256": sha * 64},
                           implementation_sha256=sha * 64)
            rows.append(row)
        return {"roles": rows}

    def test_contract_requires_six_roles_and_named_gaussian(self):
        planner.require_c15_contract(self.contract(), benchmark_revision="rev")
        changed = self.contract()
        changed["contrasts"]["controls"].remove("gaussian")
        with self.assertRaisesRegex(ValueError, "six-role"):
            planner.require_c15_contract(changed, benchmark_revision="rev")

    def test_contract_keeps_rank_matched_operation_control(self):
        changed = self.contract()
        changed["arm_requirements"].pop("rank_matched_tsvd")
        with self.assertRaisesRegex(ValueError, "six-role"):
            planner.require_c15_contract(changed, benchmark_revision="rev")

    def test_arm_binding_uses_all_physical_implementations(self):
        refs = planner.bind_contract_arms(self.contract(), self.comparison())
        self.assertEqual({ref["path"] for ref in refs}, {
            "treatment.py", "b0.py", "gaussian.py", "unprotected_svt.py",
            "rank_matched_tsvd.py"})
        changed = self.contract()
        changed["arm_requirements"]["treatment"]["revision"] = "other"
        with self.assertRaisesRegex(ValueError, "method identity"):
            planner.bind_contract_arms(changed, self.comparison())

    def test_method_boundary_must_be_design_verified_for_c15(self):
        ready = {"workflow_boundary": {
            "action": "dispatch", "candidate_id": "4d-math-20261006-c15",
            "required_stage": "design_verified", "ready": True}}
        planner.require_method_boundary(ready)
        blocked = {"workflow_boundary": {**ready["workflow_boundary"],
                                         "ready": False}}
        with self.assertRaises(ValueError):
            planner.require_method_boundary(blocked)

    def test_one_attempt_is_confirmation_only_and_zero_retry(self):
        job, limits = planner.scientific_job_contract(
            {"evidence_mode": "prospective_confirmatory"}, wall_seconds=900)
        self.assertEqual(job, {"arm_role": "treatment"})
        self.assertEqual(limits["max_confirmation_trials"], 1)
        self.assertEqual(limits["max_retries_per_trial"], 0)

    def test_outputs_require_result_manifest_and_raw_archive(self):
        self.assertEqual(planner.scoring_outputs(), [
            "actionmesh/c15-scoring-output/result.json",
            "actionmesh/c15-scoring-output/raw-manifest.json",
            "actionmesh/c15-scoring-output/raw-evidence.tar",
        ])


if __name__ == "__main__":
    unittest.main()
