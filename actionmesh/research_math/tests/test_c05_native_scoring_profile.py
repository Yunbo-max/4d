"""Authored Local-only isolation checks for the C05 scoring profile."""
from __future__ import annotations

import unittest

import launch_c05_native_scoring as launcher
import prepare_c05_native_scoring as plan
from research_math import c05_native_comparison as comparison
from research_math import c05_native_scoring as scoring
from research_math import c14_native_scoring as c14_scoring


class C05NativeScoringProfileTests(unittest.TestCase):
    def test_profile_isolated_from_shared_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(scoring.REQUEST_KIND, "c05-native-scoring-request")
        self.assertEqual(scoring.CANDIDATE_ROLE, "joint_spatial_labels")
        self.assertEqual(len(comparison.ROLES), 7)
        self.assertEqual(comparison.ROLES[-1], "joint_spatial_labels")
        self.assertEqual(comparison.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(comparison.GUARDRAIL_METRICS, ("cd_4d", "cd_motion"))
        self.assertEqual(scoring.CONTROL_ROLES, comparison.ROLES[:-1])
        self.assertEqual(comparison.CANDIDATE_ID, "4d-math-20261006-c05")

    def test_plan_and_launcher_are_c05_only(self):
        self.assertEqual(plan.TASK_ID, "c05-seven-role-official-scoring")
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c05_scoring_attempt")
        self.assertEqual(plan.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(launcher.CLAIM_KIND, "c05-launch-claim")
        self.assertEqual(launcher.ENV_PREFIX, "C05")
        self.assertIn("actionmesh/research_math/spatial_mode_candidate.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/research_math/c05_mode_bank.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertNotIn("actionmesh/research_math/c11_native_comparison.py",
                         plan.EXTRA_CODE_SOURCES)

    def test_plan_binds_promoted_c05_source_without_synthetic_report_hash(self):
        source = {"path": "spatial_mode_candidate.py", "sha256": "a" * 64}
        b0_source = {"path": "native_context_runner.py", "sha256": "b" * 64}
        roles = []
        for role in comparison.ROLES:
            if role == "b_star":
                roles.append({"role": role,
                    "method_id": comparison.METHOD_IDS["localized_mean"],
                    "alias_of": "localized_mean"})
            elif role == "b0":
                roles.append({"role": role, "method_id": "native-actionmesh-b0",
                    "implementation_ref": b0_source,
                    "implementation_sha256": b0_source["sha256"]})
            else:
                roles.append({"role": role,
                    "method_id": comparison.METHOD_IDS[role],
                    "implementation_ref": source,
                    "implementation_sha256": source["sha256"]})
        names = dict(plan.ROLE_FOR_CONTRACT_ARM)
        refs = {arm: ([b0_source] if role == "b0" else [source])
                for arm, role in names.items()}
        contract = {"arm_requirements": {arm: {"name": role,
            "revision": next(row["method_id"] for row in roles
                             if row["role"] == role),
            "implementation_refs": refs[arm]}
            for arm, role in names.items()}}
        bound = plan.bind_contract_arms(contract, {"roles": roles})
        self.assertEqual({row["path"] for row in bound},
                         {source["path"], b0_source["path"]})

    def test_seven_role_readout_keeps_failures_and_aliases(self):
        roles = []
        for role in comparison.ROLES:
            roles.append({"role": role, "method_id": role,
                          "case_id": "u-" + role,
                          "preparation_status": "completed"})
        roles[1].update(method_id="localized_mean", alias_of="localized_mean",
                        case_id="u-localized_mean")
        roles[4].update(preparation_status="error", case_id=None,
                        preparation_error="retained projection failure")
        cases = [{"case_id": row["case_id"], "status": "success",
                  "cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0}
                 for row in roles if row.get("case_id") is not None
                 and row["role"] != "b_star"]
        readout = scoring.build_logical_readout({"roles": roles}, {"cases": cases})
        rows = {row["role"]: row for row in readout["roles"]}
        self.assertEqual(readout["logical_denominator"]["n_roles"], 7)
        self.assertEqual(readout["n_failed_or_missing_roles"], 1)
        self.assertEqual(rows["surface_projected_mean"]["status"],
                         "preparation_error")
        self.assertEqual(rows["b_star"]["shared_measurement_with"],
                         "localized_mean")


if __name__ == "__main__":
    unittest.main()
