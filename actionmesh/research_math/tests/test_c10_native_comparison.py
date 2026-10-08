"""Local-only real artifact to prospective C10 freeze integration tests.

The test uses the actual C10 constructor and comparison validator on a tiny
engineering mesh.  It is authored on Web but must be run only by Local through
the common CPU software-acceptance harness; it is not native evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import c10_native_comparison as comparison
from research_math import integrable_gradient_candidate as candidate


class C10NativeComparisonTests(unittest.TestCase):
    def test_real_common_target_artifact_freezes_all_five_roles(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            workspace = Path(directory)
            uid = "fixture-c10"
            source = workspace / "source"
            source.mkdir()
            anchor = np.array([
                [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            ], dtype=np.float32)
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
            vertices = np.stack([
                anchor + np.array([0., 0., .01 * frame], dtype=np.float32)
                for frame in range(16)])
            vertices[:, 2, 0] += np.linspace(0., .2, 16, dtype=np.float32)
            np.savez_compressed(
                source / "sequence.npz", vertices=vertices, faces=faces,
                timesteps=np.arange(16, dtype=np.float32),
                frame_indices=np.arange(16, dtype=np.int64),
                query_vertex_ids=np.arange(len(anchor), dtype=np.int64))
            sequence_ref = comparison.file_ref(root, source / "sequence.npz")
            source_report = {
                "status": "completed", "uid": uid, "seed": 42,
                "sha256": {"sequence.npz": sequence_ref["sha256"]},
            }
            (source / "report.json").write_text(json.dumps(source_report) + "\n")
            source_report_ref = comparison.file_ref(root, source / "report.json")
            b0_implementation = root / "actionmesh/research_math/native_context_runner.py"
            identity = {
                "kind": "native-context-generation-identity", "version": 1,
                "scope": "paired engineering observer/replay; no candidate or scorer execution",
                "uid": uid, "generation": {"seed": 42},
                "verified_unit_manifest": {"fixture": True},
                "retained_input_refs": {"fixture": {
                    "path": "inputs/fixture.json", "sha256": "a" * 64}},
                "upstream_source_sha256": {"model.py": "b" * 64},
                "gpu_uuid": "GPU-fixture",
                "native_context_qualified": False,
                "scientific_effect_qualification": False,
                "instrument_code_sha256": {
                    "research_math/native_context_runner.py":
                        comparison.digest(b0_implementation)},
            }
            identity_path = workspace / "generation-identity.json"
            identity_path.write_text(json.dumps(identity) + "\n")

            output = workspace / "candidate"
            result = candidate.export_integrable_candidate(
                source, output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_sequence_ref=sequence_ref["path"],
                source_report_ref=source_report_ref["path"],
                target_strength=.5, max_relative_change=.25,
                absolute_tolerance=1e-10, relative_tolerance=1e-8,
                max_iterations=1000, coordinate_lower=-10., coordinate_upper=10.,
                bounds_policy="preserve_and_report",
                max_artifact_bytes=16 * 1024 * 1024)
            self.assertEqual(result["status"], "completed")

            basis_path = workspace / "basis.json"
            basis_path.write_text("{}\n")
            decision = {
                "kind": "c10-b-star-decision", "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42,
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "selected_role": "direct_common_lift",
                "selected_method_id": comparison.METHOD_IDS["direct_common_lift"],
                "selected_without_c10_native_outcomes": True,
                "selection_basis_refs": [comparison.file_ref(root, basis_path)],
            }
            decision["decision_digest"] = comparison.canonical_digest(decision)
            decision_path = workspace / "decision.json"
            decision_path.write_text(json.dumps(decision) + "\n")
            implementation_ref = comparison.file_ref(
                root, root / "actionmesh/research_math/integrable_gradient_candidate.py")
            rows = [{
                "role": "b0", "method_id": "native-actionmesh-b0",
                "report_ref": source_report_ref, "sequence_ref": sequence_ref,
                "generation_identity_ref": comparison.file_ref(root, identity_path),
                "implementation_ref": comparison.file_ref(root, b0_implementation),
            }, {
                "role": "b_star",
                "method_id": comparison.METHOD_IDS["direct_common_lift"],
                "alias_of": "direct_common_lift",
            }]
            for role in candidate.ROLES:
                rows.append({
                    "role": role, "method_id": comparison.METHOD_IDS[role],
                    "report_ref": comparison.file_ref(
                        root, output / role / "report.json"),
                    "sequence_ref": comparison.file_ref(
                        root, output / role / "sequence.npz"),
                    "certificate_ref": comparison.file_ref(
                        root, output / role / "certificate.npz"),
                    "implementation_ref": implementation_ref,
                })
            freeze = {
                "kind": "c10-native-comparison-freeze", "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "scoring_seed": 44,
                "primary_metric": "cd_3d",
                "guardrail_metrics": ["cd_4d", "cd_motion"],
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "source_sequence_ref": sequence_ref,
                "source_report_ref": source_report_ref,
                "candidate_artifact_ref": comparison.file_ref(
                    root, output / "candidate.json"),
                "b_star_decision_ref": comparison.file_ref(root, decision_path),
                "roles": rows,
            }
            freeze["freeze_digest"] = comparison.canonical_digest(freeze)
            freeze_path = workspace / "freeze.json"
            freeze_path.write_text(json.dumps(freeze) + "\n")

            request = comparison.make_request(root, freeze_path=freeze_path)
            self.assertEqual([row["role"] for row in request["roles"]],
                             list(comparison.ROLES))
            self.assertEqual(request["logical_denominator"]["n_roles"], 5)
            self.assertEqual(request["role_to_case"]["b_star"],
                             request["role_to_case"]["direct_common_lift"])
            self.assertLessEqual(len(request["scoring_cases"]), 4)
            self.assertFalse(request["dispatch_ready"])
            self.assertTrue(request["generated_unexecuted"])

            # Hash-consistent relabelling cannot turn a simple control into C10.
            direct = next(row for row in freeze["roles"]
                          if row["role"] == "direct_common_lift")
            direct["method_id"] = candidate.CANDIDATE_ID
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items()
                if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with self.assertRaisesRegex(ValueError, "relabels retained C10 method"):
                comparison.make_request(root, freeze_path=freeze_path)

            # A real terminal construction failure remains one of the five
            # logical roles and is neither staged nor converted to score zero.
            direct["method_id"] = comparison.METHOD_IDS["direct_common_lift"]
            failed_role = "independent_local_repair"
            failed_row = next(row for row in freeze["roles"]
                              if row["role"] == failed_role)
            failed_report_path = output / failed_role / "report.json"
            failed_report = comparison.read_json(failed_report_path)
            failed_report.update(status="error", exception_type="FixtureFailure",
                                 error="retained construction failure")
            failed_report.pop("sha256")
            failed_report_path.write_text(json.dumps(failed_report) + "\n")
            (output / failed_role / "sequence.npz").unlink()
            (output / failed_role / "certificate.npz").unlink()
            candidate_record = comparison.read_json(output / "candidate.json")
            candidate_record["status"] = "incomplete"
            candidate_record["arms"] = [
                failed_report if report["candidate_arm"] == failed_role else report
                for report in candidate_record["arms"]]
            (output / "candidate.json").write_text(
                json.dumps(candidate_record) + "\n")
            manifest = comparison.read_json(output / "manifest.json")
            manifest["cases"] = [row for row in manifest["cases"]
                                 if row["arm_role"] != failed_role]
            (output / "manifest.json").write_text(json.dumps(manifest) + "\n")
            candidate._write_deterministic_archive(
                output, candidate_record["arms"], 16 * 1024 * 1024)
            failed_row["report_ref"] = comparison.file_ref(root, failed_report_path)
            failed_row["sequence_ref"] = None
            failed_row["certificate_ref"] = None
            freeze["candidate_artifact_ref"] = comparison.file_ref(
                root, output / "candidate.json")
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items()
                if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            failed_request = comparison.make_request(root, freeze_path=freeze_path)
            failed_logical = next(row for row in failed_request["roles"]
                                  if row["role"] == failed_role)
            self.assertEqual(failed_logical["preparation_status"], "error")
            self.assertIsNone(failed_logical["case_id"])
            self.assertNotIn(
                failed_role,
                [row["role"] for row in failed_request["scoring_cases"]])

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"a":1,"a":2}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                comparison.read_json(path)


if __name__ == "__main__":
    unittest.main()
