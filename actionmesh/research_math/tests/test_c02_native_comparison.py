"""Local-only end-to-end C02 artifact/freeze acceptance."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import c02_native_comparison as comparison
from research_math import protected_geometry_candidate as candidate


class C02NativeComparisonTests(unittest.TestCase):
    def test_real_candidate_artifacts_freeze_and_deduplicate(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            workspace = Path(directory)
            uid = "fixture-c02"
            source = workspace / "source"; source.mkdir()
            anchor = np.array([
                [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
                [0., 1., 0.], [1., 1., .1], [2., 1., 0.],
            ], dtype=np.float32)
            faces = np.array([[0, 1, 4], [0, 4, 3],
                              [1, 2, 5], [1, 5, 4]], dtype=np.int64)
            # Time-varying local deformation forces a nonzero common repair;
            # this test therefore reaches freeze-time certificate recomputation
            # instead of exercising only the trivial zero-step path.
            vertices = np.stack([anchor + np.array([0., .01 * t, 0.], np.float32)
                                 for t in range(16)])
            vertices[:, 4, 2] += np.linspace(0., .3, 16, dtype=np.float32)
            np.savez_compressed(
                source / "sequence.npz", vertices=vertices, faces=faces,
                timesteps=np.arange(16, dtype=np.float32),
                frame_indices=np.arange(16, dtype=np.int64),
                query_vertex_ids=np.arange(len(anchor), dtype=np.int64))
            sequence_sha = candidate.digest(source / "sequence.npz")
            source_report = {
                "status": "completed", "uid": uid, "seed": 42,
                "sha256": {"sequence.npz": sequence_sha},
            }
            (source / "report.json").write_text(json.dumps(source_report) + "\n")
            b0_implementation = root / "actionmesh/research_census_case.py"
            source_command = {
                "argv": [str(b0_implementation)], "uid": uid, "seed": 42,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": candidate.digest(b0_implementation),
                "offline": True,
                "scope": "native baseline fixture; no quality claim",
            }
            (source / "command.json").write_text(json.dumps(source_command) + "\n")
            output = workspace / "candidate"
            result = candidate.export_candidate(
                source, output, uid=uid, expected_sequence_sha256=sequence_sha,
                patch_count=2, arap_weight=1., temporal_weight=1.,
                iterations=100, cg_tolerance=1e-8, cg_max_iterations=500)
            self.assertEqual(result["status"], "completed")
            basis = workspace / "basis.json"; basis.write_text("{}\n")
            decision = {
                "kind": "c02-b-star-decision", "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42,
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "selected_role": "b0", "selected_method_id": "b0",
                "selected_without_c02_native_outcomes": True,
                "selection_basis_refs": [comparison.file_ref(root, basis)],
            }
            decision["decision_digest"] = comparison.canonical_digest(decision)
            decision_path = workspace / "decision.json"
            decision_path.write_text(json.dumps(decision) + "\n")
            implementation = root / "actionmesh/research_math/protected_geometry_candidate.py"
            certificate_ref = comparison.file_ref(root, output / "certificate.npz")
            rows = [{
                "role": "b0", "method_id": "b0",
                "report_ref": comparison.file_ref(root, source / "report.json"),
                "sequence_ref": comparison.file_ref(root, source / "sequence.npz"),
                "command_ref": comparison.file_ref(root, source / "command.json"),
                "implementation_ref": comparison.file_ref(root, b0_implementation),
            }, {
                "role": "b_star", "method_id": "b0", "alias_of": "b0",
            }]
            for role in candidate.ROLES:
                report = output / role / "report.json"
                sequence = output / role / "sequence.npz"
                rows.append({
                    "role": role,
                    "method_id": (candidate.CANDIDATE_ID if role == "protected_step"
                                  else "c02-control-" + role),
                    "report_ref": comparison.file_ref(root, report),
                    "sequence_ref": comparison.file_ref(root, sequence),
                    "implementation_ref": comparison.file_ref(root, implementation),
                    "certificate_ref": certificate_ref,
                })
            freeze = {
                "kind": "c02-native-comparison-freeze", "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "scoring_seed": 44,
                "primary_metric": "cd_3d",
                "guardrail_metrics": ["cd_4d", "cd_motion"],
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "source_sequence_ref": comparison.file_ref(root, source / "sequence.npz"),
                "source_report_ref": comparison.file_ref(root, source / "report.json"),
                "b_star_decision_ref": comparison.file_ref(root, decision_path),
                "roles": rows,
            }
            freeze["freeze_digest"] = comparison.canonical_digest(freeze)
            freeze_path = workspace / "freeze.json"
            freeze_path.write_text(json.dumps(freeze) + "\n")
            request = comparison.make_request(root, freeze_path=freeze_path)
            self.assertEqual([row["role"] for row in request["roles"]], list(comparison.ROLES))
            self.assertEqual(request["logical_denominator"]["n_roles"], 5)
            self.assertLessEqual(len(request["scoring_cases"]), 4)
            self.assertFalse(request["dispatch_ready"])
            projection = comparison.read_json(
                output / "protected_step" / "report.json")["projection"]
            self.assertLessEqual(projection["float32_strength_match_abs_delta"],
                                 projection["float32_strength_match_tolerance"])
            self.assertLessEqual(
                projection["float32_strength_norm2_relative_mismatch"],
                projection["float32_strength_norm2_relative_mismatch_cap"])

            # The source implementation identity is part of the frozen B0/B*
            # contract, not a caller-selected label.
            b0_row = freeze["roles"][0]
            original_implementation = b0_row["implementation_ref"]
            b0_row["implementation_ref"] = comparison.file_ref(
                root, Path(comparison.__file__))
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with self.assertRaisesRegex(ValueError, "B0 must be exact source refs"):
                comparison.make_request(root, freeze_path=freeze_path)
            b0_row["implementation_ref"] = original_implementation

            relabeled_row = next(row for row in freeze["roles"]
                                 if row["role"] == "geometry_only")
            original_method_id = relabeled_row["method_id"]
            relabeled_row["method_id"] = "caller-relabelled-control"
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with self.assertRaisesRegex(ValueError, "relabels the retained method"):
                comparison.make_request(root, freeze_path=freeze_path)
            relabeled_row["method_id"] = original_method_id

            # A terminal preparation failure remains a logical role but is not
            # staged or zero-imputed as a physical scoring case.
            failed_row = next(row for row in freeze["roles"]
                              if row["role"] == "strength_matched_blend")
            failed_report_path = output / "strength_matched_blend" / "report.json"
            completed_report = comparison.read_json(failed_report_path)
            failed_report = dict(completed_report)
            failed_report.update(status="error", exception_type="FixtureFailure",
                                 error="retained preparation failure")
            failed_report["sha256"] = {
                "certificate.npz": completed_report["sha256"]["certificate.npz"]}
            failed_report_path.write_text(json.dumps(failed_report) + "\n")
            failed_row["report_ref"] = comparison.file_ref(root, failed_report_path)
            completed_sequence_ref = failed_row["sequence_ref"]
            failed_row["sequence_ref"] = None
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            failed_request = comparison.make_request(root, freeze_path=freeze_path)
            failed_logical = next(row for row in failed_request["roles"]
                                  if row["role"] == "strength_matched_blend")
            self.assertEqual(failed_logical["preparation_status"], "error")
            self.assertIsNone(failed_logical["case_id"])
            self.assertNotIn(
                "strength_matched_blend",
                [row["role"] for row in failed_request["scoring_cases"]])
            failed_report_path.write_text(json.dumps(completed_report) + "\n")
            failed_row["report_ref"] = comparison.file_ref(root, failed_report_path)
            failed_row["sequence_ref"] = completed_sequence_ref

            # The independently fixed relative cap rejects native float32 arms
            # whose actual strength no longer matches the ideal equal-W-norm
            # construction, even when hashes/reports are updated consistently.
            matched_path = output / "strength_matched_blend" / "sequence.npz"
            original_matched = comparison._arrays(matched_path)
            excessive_matched = {key: value.copy()
                                 for key, value in original_matched.items()}
            excessive_matched["vertices"][1:, :, 0] += np.float32(1.0)
            np.savez_compressed(matched_path, **excessive_matched)
            matched_report_path = output / "strength_matched_blend" / "report.json"
            matched_report = comparison.read_json(matched_report_path)
            original_matched_report = json.loads(json.dumps(matched_report))
            matched_report["sha256"]["sequence.npz"] = candidate.digest(matched_path)
            matched_report_path.write_text(json.dumps(matched_report) + "\n")
            failed_row["report_ref"] = comparison.file_ref(root, matched_report_path)
            failed_row["sequence_ref"] = comparison.file_ref(root, matched_path)
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with self.assertRaisesRegex(ValueError, "materially destroys scored strength matching"):
                comparison.make_request(root, freeze_path=freeze_path)
            np.savez_compressed(matched_path, **original_matched)
            matched_report_path.write_text(json.dumps(original_matched_report) + "\n")
            failed_row["report_ref"] = comparison.file_ref(root, matched_report_path)
            failed_row["sequence_ref"] = comparison.file_ref(root, matched_path)

            # Hash-consistent native identity is insufficient: the scored arm
            # bytes must equal the independently recomputed C02 construction.
            tampered_path = output / "geometry_only" / "sequence.npz"
            tampered = comparison._arrays(tampered_path)
            tampered["vertices"][1, 0, 0] += np.float32(.01)
            np.savez_compressed(tampered_path, **tampered)
            tampered_report_path = output / "geometry_only" / "report.json"
            tampered_report = comparison.read_json(tampered_report_path)
            tampered_report["sha256"]["sequence.npz"] = candidate.digest(tampered_path)
            tampered_report_path.write_text(json.dumps(tampered_report) + "\n")
            tampered_row = next(row for row in freeze["roles"]
                                if row["role"] == "geometry_only")
            tampered_row["report_ref"] = comparison.file_ref(root, tampered_report_path)
            tampered_row["sequence_ref"] = comparison.file_ref(root, tampered_path)
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with self.assertRaisesRegex(ValueError, "differs from recomputed method"):
                comparison.make_request(root, freeze_path=freeze_path)

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"a": 1, "a": 2}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                comparison.read_json(path)


if __name__ == "__main__":
    unittest.main()
