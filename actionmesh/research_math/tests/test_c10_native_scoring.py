"""Engineering contracts for C10 official scoring and retained failures.

These are Local-only source tests.  They use byte fixtures and never invoke
ActionBench, a model, the official scorer, or a GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research_math import c10_native_scoring as scoring


class C10NativeScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative: str, value) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value) + "\n")
        return path

    def test_unique_physical_cases_stage_once_and_keep_candidate_certificate(self):
        rows = []
        for role in ("b0", "direct_common_lift", "independent_local_repair",
                     "pinned_integrable_solve"):
            report = self.write(role + "/report.json", {"status": "completed"})
            sequence = self.write(role + "/sequence.npz", role.encode())
            row = {
                "role": role, "case_id": "u-" + role,
                "preparation_status": "completed",
                "report_ref": scoring.file_ref(self.root, report),
                "sequence_ref": scoring.file_ref(self.root, sequence),
            }
            if role in ("direct_common_lift", "independent_local_repair",
                        "pinned_integrable_solve"):
                certificate = self.write(role + "/certificate.npz",
                                         (role + "-certificate").encode())
                row["certificate_ref"] = scoring.file_ref(self.root, certificate)
            rows.append(row)
        comparison = {"uid": "u", "scoring_cases": rows}
        raw = self.root / "raw"
        raw.mkdir()
        staged = scoring.stage_cases(self.root, comparison, raw)
        self.assertEqual([row["case_id"] for row in staged["cases"]],
                         [row["case_id"] for row in rows])
        self.assertEqual(
            (raw / "cases/u-pinned_integrable_solve/certificate.npz").read_bytes(),
            b"pinned_integrable_solve-certificate")

    def test_failed_role_stays_in_logical_denominator_but_is_not_staged(self):
        report = self.write("b0/report.json", {"status": "completed"})
        sequence = self.write("b0/sequence.npz", b"native")
        comparison = {
            "uid": "u",
            "roles": [
                {"role": "b0", "case_id": "u-b0",
                 "preparation_status": "completed"},
                {"role": "independent_local_repair", "case_id": None,
                 "preparation_status": "error", "error": "retained failure"},
            ],
            "logical_denominator": {"n_roles": 5},
            "scoring_cases": [{
                "role": "b0", "case_id": "u-b0",
                "preparation_status": "completed",
                "report_ref": scoring.file_ref(self.root, report),
                "sequence_ref": scoring.file_ref(self.root, sequence),
            }],
        }
        raw = self.root / "raw"
        raw.mkdir()
        staged = scoring.stage_cases(self.root, comparison, raw)
        self.assertEqual(staged["cases"], [
            {"case_id": "u-b0", "uid": "u", "case_dir": "u-b0"}])
        self.assertFalse((raw / "cases/u-independent_local_repair").exists())
        self.assertEqual(comparison["logical_denominator"]["n_roles"], 5)

    def test_all_pinned_inputs_are_snapshotted_before_scoring(self):
        report = self.write("candidate/report.json", {"status": "completed"})
        sequence = self.write("candidate/sequence.npz", b"candidate")
        target = self.write("candidate/common-target.npz", b"target")
        comparison = {
            "uid": "u",
            "input_refs": [scoring.file_ref(self.root, path)
                           for path in (report, sequence, target)],
            "scoring_cases": [{
                "role": "pinned_integrable_solve",
                "case_id": "u-pinned", "preparation_status": "completed",
                "report_ref": scoring.file_ref(self.root, report),
                "sequence_ref": scoring.file_ref(self.root, sequence),
            }],
        }
        raw = self.root / "raw"
        raw.mkdir()
        scoring.stage_cases(self.root, comparison, raw)
        for ref in comparison["input_refs"]:
            snapshot = raw / "frozen-inputs" / ref["path"]
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                             ref["sha256"])

    def test_retained_json_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            scoring.read_json_bytes(b'{"request_digest":"a","request_digest":"b"}',
                                    "request.json")


if __name__ == "__main__":
    unittest.main()
