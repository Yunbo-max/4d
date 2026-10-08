"""Local-only isolation checks for the C08 native scoring profile."""
from __future__ import annotations

import unittest
import inspect

import launch_c08_native_scoring as launcher
import prepare_c08_native_acceptance as acceptance
import prepare_c08_native_scoring as plan
from research_math import c08_native_comparison as comparison
from research_math import c08_native_scoring as scoring
from research_math import c11_native_comparison as c11_comparison
from research_math import c14_native_scoring as c14_scoring


class C08NativeScoringProfileTests(unittest.TestCase):
    def test_profile_isolated_from_shared_c14_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(scoring.REQUEST_KIND, "c08-native-scoring-request")
        self.assertEqual(scoring.CANDIDATE_ROLE, "endpoint_bridge")
        self.assertEqual(len(comparison.ROLES), 6)
        self.assertEqual(comparison.ROLES[-1], "endpoint_bridge")
        self.assertEqual(comparison.PRIMARY_METRIC, "cd_motion")
        self.assertEqual(comparison.GUARDRAIL_METRICS, ("cd_3d", "cd_4d"))
        self.assertEqual(scoring.CONTROL_ROLES, comparison.ROLES[:-1])
        self.assertEqual(c11_comparison.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(c11_comparison.GUARDRAIL_METRICS, ("cd_4d", "cd_motion"))

    def test_plan_and_launcher_are_c08_only(self):
        self.assertEqual(plan.TASK_ID, "c08-six-role-official-scoring")
        self.assertEqual(plan.PRIMARY_METRIC, "cd_motion")
        self.assertEqual(plan.GUARDRAIL_METRICS, ("cd_3d", "cd_4d"))
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c08_scoring_attempt")
        self.assertEqual(launcher.CLAIM_KIND, "c08-launch-claim")
        self.assertEqual(launcher.ENV_PREFIX, "C08")
        self.assertIn("actionmesh/research_math/trajectory_bridge_candidate.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/research_math/c11_native_comparison.py",
                      plan.EXTRA_CODE_SOURCES)

    def test_six_role_readout_deduplicates_alias_and_retains_failures(self):
        roles = []
        for role in comparison.ROLES:
            row = {"role": role, "method_id": role,
                   "case_id": "u-" + role, "preparation_status": "completed"}
            roles.append(row)
        roles[1].update(method_id="local_transition_tracker",
                        alias_of="local_transition_tracker",
                        case_id="u-local_transition_tracker")
        roles[3].update(preparation_status="error", case_id=None,
                        preparation_error="retained control failure")
        cases = [{"case_id": row["case_id"], "status": "success",
                  "cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0}
                 for row in roles if row.get("case_id") is not None
                 and row["role"] != "b_star"]
        readout = scoring.build_logical_readout({"roles": roles}, {"cases": cases})
        rows = {row["role"]: row for row in readout["roles"]}
        self.assertEqual(readout["logical_denominator"]["n_roles"], 6)
        self.assertEqual(readout["unique_physical_measurements"], 4)
        self.assertEqual(rows["coordinate_smoother"]["status"], "preparation_error")
        self.assertNotIn("metrics", rows["coordinate_smoother"])
        self.assertEqual(rows["b_star"]["shared_measurement_with"],
                         "local_transition_tracker")

    def test_acceptance_scoring_argv_contract_is_explicit(self):
        source = inspect.getsource(acceptance.build_plans)
        for token in ("sys.argv[3]", "sys.argv[4]", "sys.argv[5]",
                      "sys.argv[6]", "sys.argv[7]", "sys.argv[8]"):
            self.assertIn(token, source)
        self.assertIn("Path(sys.argv[7]).parent.parent", source)


if __name__ == "__main__":
    unittest.main()
