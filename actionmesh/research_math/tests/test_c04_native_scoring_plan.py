"""Local-only fail-closed contracts for the C04 scoring plan profile."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import prepare_c04_native_scoring as planner


class C04NativeScoringPlanTests(unittest.TestCase):
    def contract(self):
        names = {
            "treatment": "robust_conic_protection", "baseline": "b_star",
            "b0": "b0", "deterministic_protection": "deterministic_protection",
            "strength_matched_repair": "strength_matched_repair",
        }
        refs = {key: [{"path": name + ".py", "sha256": chr(97 + index) * 64}]
                for index, (key, name) in enumerate(names.items())}
        refs["baseline"] = refs["deterministic_protection"]
        contract = {
            "benchmark_id": "facebook/actionbench", "benchmark_revision": "rev",
            "primary_metric": "cd_3d",
            "metrics": [{"name": name, "output_path": ["readout", name]}
                        for name in ("cd_3d", "cd_4d", "cd_motion")],
            "contrasts": {
                "treatment": "robust_conic_protection", "baseline": "b_star",
                "controls": ["b0", "deterministic_protection",
                             "strength_matched_repair"],
            },
            "arm_requirements": {
                key: {"name": name, "revision": name + "-method",
                      "implementation_refs": refs[key]}
                for key, name in names.items()
            },
            "scorer": {"kind": "official", "source_refs": [], "code_refs": []},
        }
        contract["arm_requirements"]["baseline"]["revision"] = (
            "deterministic_protection-method")
        return contract

    def comparison(self):
        rows = []
        hashes = {
            "b0": "c" * 64, "deterministic_protection": "d" * 64,
            "strength_matched_repair": "e" * 64,
            "robust_conic_protection": "a" * 64,
        }
        for role in ("b0", "b_star", "deterministic_protection",
                     "strength_matched_repair", "robust_conic_protection"):
            row = {"role": role, "method_id": role + "-method"}
            if role == "b_star":
                row.update(method_id="deterministic_protection-method",
                           alias_of="deterministic_protection")
            else:
                row.update(report_ref={"path": role + "/report.json",
                                       "sha256": hashes[role]},
                           implementation_sha256=hashes[role])
            rows.append(row)
        rows[-1]["method_id"] = "robust_conic_protection-method"
        return {"roles": rows}

    def test_contract_fixes_candidate_baseline_controls_and_official_scorer(self):
        planner.require_c14_contract(self.contract(), benchmark_revision="rev")
        for mutation in ("treatment", "baseline", "controls", "scorer"):
            changed = self.contract()
            if mutation == "treatment":
                changed["contrasts"]["treatment"] = "deterministic_protection"
            elif mutation == "baseline":
                changed["contrasts"]["baseline"] = "deterministic_protection"
            elif mutation == "controls":
                changed["contrasts"]["controls"].pop()
            else:
                changed.pop("scorer")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                planner.require_c14_contract(changed, benchmark_revision="rev")

    def test_frozen_arm_bindings_cover_every_physical_implementation(self):
        refs = planner.bind_contract_arms(self.contract(), self.comparison())
        self.assertEqual({ref["path"] for ref in refs}, {
            "b0.py", "deterministic_protection.py",
            "strength_matched_repair.py", "robust_conic_protection.py"})
        changed = self.contract()
        changed["arm_requirements"]["treatment"]["revision"] = "caller-relabel"
        with self.assertRaisesRegex(ValueError, "method identity"):
            planner.bind_contract_arms(changed, self.comparison())

    def test_native_context_b0_uses_generation_identity_not_candidate_report_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, comparison = self.contract(), self.comparison()
            comparison.update(uid="fixture", inference_seed=42, input_refs=[])
            by_role = {row["role"]: row for row in comparison["roles"]}
            for arm_name, arm in contract["arm_requirements"].items():
                if arm_name == "baseline":
                    continue
                role = arm["name"]
                implementation = root / (role + ".py")
                implementation.write_text("# " + role + "\n")
                implementation_ref = planner.scoring.file_ref(root, implementation)
                arm["implementation_refs"] = [implementation_ref]
                row = by_role[role]
                row["implementation_sha256"] = implementation_ref["sha256"]
                row["implementation_ref"] = implementation_ref
                comparison["input_refs"].append(implementation_ref)
                report = root / (role + "-report.json")
                report.write_text(json.dumps({} if role == "b0" else {
                    "implementation_sha256": implementation_ref["sha256"]}))
                row["report_ref"] = planner.scoring.file_ref(root, report)
            contract["arm_requirements"]["baseline"]["implementation_refs"] = (
                contract["arm_requirements"]["deterministic_protection"]["implementation_refs"])
            identity_path = root / "generation-identity.json"
            identity = {"kind": "native-context-generation-identity", "uid": "fixture",
                        "instrument_code_sha256": {
                "research_math/native_context_runner.py": by_role["b0"]["implementation_sha256"]}}
            identity_path.write_text(json.dumps(identity))
            by_role["b0"]["generation_identity_ref"] = planner.scoring.file_ref(root, identity_path)
            comparison["input_refs"].append(by_role["b0"]["generation_identity_ref"])
            planner.bind_contract_arms(contract, comparison, root)
            identity["instrument_code_sha256"]["research_math/native_context_runner.py"] = "0" * 64
            identity_path.write_text(json.dumps(identity))
            by_role["b0"]["generation_identity_ref"] = planner.scoring.file_ref(root, identity_path)
            comparison["input_refs"].append(by_role["b0"]["generation_identity_ref"])
            with self.assertRaisesRegex(ValueError, "producer implementation mismatch"):
                planner.bind_contract_arms(contract, comparison, root)

    def test_method_boundary_requires_c04_design_verification(self):
        ready = {"workflow_boundary": {
            "action": "dispatch", "candidate_id": "4d-math-20261006-c04",
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
            "actionmesh/c04-scoring-output/result.json",
            "actionmesh/c04-scoring-output/raw-manifest.json",
            "actionmesh/c04-scoring-output/raw-evidence.tar",
        ])
        self.assertEqual(planner.AUTHORIZATION_SCOPE,
                         "single_c04_scoring_attempt")
        self.assertEqual(planner.FAMILY_SPLIT_KIND, "c04-family-split")
        self.assertEqual(planner.ANALYSIS_KIND, "c04-g01-analysis-plan")
        self.assertEqual(planner.ADMISSION_KIND,
                         "c04-scientific-dispatch-admission")


if __name__ == "__main__":
    unittest.main()
