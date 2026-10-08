"""Receipt/readout checks for C15 official scoring delivery."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from research_math import c15_native_scoring as scoring


class C15NativeScoringDeliveryTests(unittest.TestCase):
    def comparison(self):
        roles = [
            {"role": "b0", "method_id": "b0", "case_id": "u-b0",
             "preparation_status": "completed"},
            {"role": "b_star", "method_id": "world_gaussian",
             "alias_of": "gaussian", "case_id": "u-gaussian",
             "preparation_status": "completed"},
            {"role": "gaussian", "method_id": "world_gaussian",
             "case_id": "u-gaussian", "preparation_status": "completed"},
            {"role": "unprotected_svt", "method_id": "unprotected",
             "case_id": "u-unprotected", "preparation_status": "completed"},
            {"role": "rank_matched_tsvd", "method_id": "rank-matched",
             "case_id": "u-rank", "preparation_status": "completed"},
            {"role": "protected_residual_svt",
             "method_id": "4d-math-20261006-c15",
             "case_id": "u-protected", "preparation_status": "completed"},
        ]
        return {"uid": "u", "roles": roles,
                "logical_denominator": {"n_roles": 6}, "scoring_cases": []}

    def report(self, failed=()):
        cases = []
        for index, case_id in enumerate((
                "u-b0", "u-gaussian", "u-unprotected", "u-rank", "u-protected")):
            if case_id in failed:
                cases.append({"case_id": case_id, "uid": "u", "status": "error",
                              "error": "retained failure"})
            else:
                cases.append({"case_id": case_id, "uid": "u", "status": "success",
                              "n_frames": 16, "cd_3d": 10.0 - index,
                              "cd_4d": 20.0 - index,
                              "cd_motion": 30.0 - index})
        return {"cases": cases}

    def test_alias_expands_without_duplicate_official_measurement(self):
        readout = scoring.build_logical_readout(self.comparison(), self.report())
        self.assertEqual(readout["logical_denominator"]["n_roles"], 6)
        self.assertEqual(len(readout["roles"]), 6)
        self.assertEqual(readout["unique_physical_measurements"], 5)
        b_star = next(row for row in readout["roles"] if row["role"] == "b_star")
        gaussian = next(row for row in readout["roles"] if row["role"] == "gaussian")
        self.assertEqual(b_star["metrics"], gaussian["metrics"])
        self.assertEqual(b_star["shared_measurement_with"], "gaussian")
        self.assertEqual({row["control_role"] for row in readout["contrasts"]}, {
            "b0", "b_star", "gaussian", "unprotected_svt",
            "rank_matched_tsvd"})

    def test_failed_roles_remain_denominator_without_zero_imputation(self):
        frozen = self.comparison()
        rank = next(row for row in frozen["roles"]
                    if row["role"] == "rank_matched_tsvd")
        rank.update(preparation_status="error", case_id=None,
                    preparation_error="retained preparation failure")
        readout = scoring.build_logical_readout(
            frozen, self.report(failed={"u-protected"}))
        rows = {row["role"]: row for row in readout["roles"]}
        self.assertEqual(rows["rank_matched_tsvd"]["status"], "preparation_error")
        self.assertNotIn("metrics", rows["rank_matched_tsvd"])
        self.assertEqual(rows["protected_residual_svt"]["status"], "scoring_error")
        self.assertEqual(readout["n_failed_or_missing_roles"], 2)
        self.assertEqual(readout["contrasts"], [])

    def test_stage_cases_retains_certificates_and_full_input_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = self.comparison()
            cases = []
            for role in ("b0", "gaussian", "protected_residual_svt"):
                base = root / "inputs" / role
                base.mkdir(parents=True)
                report = base / "report.json"; report.write_text("{}\n")
                sequence = base / "sequence.npz"; sequence.write_bytes(role.encode())
                row = {"role": role, "method_id": role,
                       "case_id": "u-" + role,
                       "preparation_status": "completed",
                       "report_ref": scoring.file_ref(root, report),
                       "sequence_ref": scoring.file_ref(root, sequence)}
                if role == "protected_residual_svt":
                    certificate = base / "certificate.npz"
                    certificate.write_bytes(b"certificate")
                    row["certificate_ref"] = scoring.file_ref(root, certificate)
                cases.append(row)
            evidence = root / "basis.npy"; evidence.write_bytes(b"basis")
            frozen["scoring_cases"] = cases
            frozen["input_refs"] = [scoring.file_ref(root, evidence)]
            raw = root / "raw"; raw.mkdir()
            manifest = scoring.stage_cases(root, frozen, raw)
            self.assertEqual(len(manifest["cases"]), 3)
            self.assertEqual((raw / "cases/u-protected_residual_svt/certificate.npz").read_bytes(),
                             b"certificate")
            self.assertEqual((raw / "frozen-inputs/basis.npy").read_bytes(), b"basis")


if __name__ == "__main__":
    unittest.main()
