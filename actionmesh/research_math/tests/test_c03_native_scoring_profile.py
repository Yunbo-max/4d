"""Local software checks for the nine-role C03 native scoring profile."""
from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
import tempfile

from research_math import c03_native_comparison as comparison
from research_math import c03_native_scoring as scoring


class C03NativeScoringProfileTests(unittest.TestCase):
    def test_b_star_decision_is_content_bound_and_prospective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path = root / "g01-design.json"
            freeze_path = root / "g01-b-star-freeze.json"
            split_path = root / "g01-family-split.json"
            result_path = root / "d1-result.json"
            design_path.write_text('{"design_digest":"' + "d" * 64 + '"}\n')
            freeze_path.write_text('{"kind":"g01-b-star-freeze"}\n')
            split_path.write_text('{"kind":"g01-family-split"}\n')
            result_path.write_text('{"status":"reviewed"}\n')
            design = {"design_digest": "d" * 64}
            freeze = {"selected_by_candidate": {comparison.CANDIDATE_ID: {
                "role": "unit_C01", "selection_stage": "D1_only",
                "metric": "cd_motion"}}}
            split = {"d1_ids": [], "d2_ids": ["uid"],
                     "confirmation_ids": [], "split_digest": "s" * 64}
            with mock.patch.object(comparison.g01, "validate_design", return_value=design), \
                    mock.patch.object(comparison, "_g01_split",
                        return_value=(split, comparison.file_ref(root, split_path))), \
                    mock.patch.object(comparison, "_g01_selection",
                        return_value=(design, split, freeze,
                                      [design_path, split_path, freeze_path, result_path])):
                decision = comparison.make_decision(root, uid="uid",
                    inference_seed=314, application_stage="d2",
                    g01_design=design_path, g01_family_split=split_path,
                    g01_b_star_freeze=freeze_path,
                    decided_at="2026-10-08T00:00:00+00:00")
            self.assertTrue(decision[comparison.DECISION_NO_OUTCOMES_FIELD])
            self.assertEqual(decision["selected_role"], "unit_C01")
            self.assertEqual(decision["g01_selection"],
                             freeze["selected_by_candidate"][comparison.CANDIDATE_ID])
            self.assertEqual(decision["decision_digest"],
                             comparison.canonical_digest({key: value for key, value
                                 in decision.items() if key != "decision_digest"}))

    def test_all_g01_generation_seeds_are_declared(self):
        self.assertEqual(comparison.ALLOWED_INFERENCE_SEEDS, (42, 314, 2718))

    def test_successful_readout_keeps_nine_role_denominator(self):
        roles = []
        cases = []
        for index, role in enumerate(comparison.ROLES):
            case_id = "case-" + str(index)
            roles.append({"role": role, "method_id": "method-" + role,
                          "case_id": case_id, "preparation_status": "completed"})
            cases.append({"case_id": case_id, "status": "success",
                          "cd_motion": 1.0, "cd_3d": 2.0, "cd_4d": 3.0})
        result = scoring.build_logical_readout({"roles": roles}, {"cases": cases})
        self.assertEqual(result["logical_denominator"]["n_roles"], 9)
        self.assertEqual(result["n_successful_roles"], 9)
        self.assertEqual(len(result["contrasts"]), 8)
        self.assertFalse(result["native_qualified"])

    def test_failed_role_remains_in_denominator(self):
        roles = [{"role": role, "method_id": "m-" + role,
                  "case_id": "case-" + role,
                  "preparation_status": "error" if role == "full_squared" else "completed",
                  "preparation_error": "retained"}
                 for role in comparison.ROLES]
        cases = [{"case_id": "case-" + role, "status": "success",
                  "cd_motion": 1.0, "cd_3d": 2.0, "cd_4d": 3.0}
                 for role in comparison.ROLES if role != "full_squared"]
        result = scoring.build_logical_readout({"roles": roles}, {"cases": cases})
        self.assertEqual(result["n_failed_or_missing_roles"], 1)
        failed = next(row for row in result["roles"] if row["role"] == "full_squared")
        self.assertEqual(failed["status"], "preparation_error")


if __name__ == "__main__":
    unittest.main()
