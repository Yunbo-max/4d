"""Local-only real artifact to prospective C04 freeze integration tests.

The test uses the actual C04 constructor and comparison validator on a tiny
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

from research_math import c04_native_comparison as comparison
from research_math import robust_motion_candidate as candidate


class C04NativeComparisonTests(unittest.TestCase):
    def test_real_common_target_artifact_freezes_all_five_roles(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            workspace = Path(directory)
            uid = "fixture-c04"
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
            result = candidate.export_robust_candidate(
                source, output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_sequence_ref=sequence_ref["path"],
                source_report_ref=source_report_ref["path"],
                parameters={"arap_weight": .1, "temporal_weight": .1,
                    "iterations": 2, "cg_tolerance": 1e-8,
                    "cg_max_iterations": 1000, "shape_floor": .01,
                    "shape_gain": .5, "radius": .2, "epsilon": .1,
                    "trust_radius": 1., "finite_budget": 1.,
                    "absolute_tolerance": 1e-8, "relative_tolerance": 1e-6,
                    "max_iterations": 10000, "max_backtracks": 20},
                coordinate_lower=-10., coordinate_upper=10.,
                bounds_policy="preserve_and_report",
                max_artifact_bytes=16 * 1024 * 1024)
            self.assertEqual(result["status"], "completed", result)

            basis_path = workspace / "basis.json"
            basis_path.write_text("{}\n")
            decision = {
                "kind": "c04-b-star-decision", "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42,
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "selected_role": "deterministic_protection",
                "selected_method_id": comparison.METHOD_IDS["deterministic_protection"],
                "selected_without_c04_native_outcomes": True,
                "selection_basis_refs": [comparison.file_ref(root, basis_path)],
            }
            decision["decision_digest"] = comparison.canonical_digest(decision)
            decision_path = workspace / "decision.json"
            decision_path.write_text(json.dumps(decision) + "\n")
            implementation_ref = comparison.file_ref(
                root, root / "actionmesh/research_math/robust_motion_candidate.py")
            rows = [{
                "role": "b0", "method_id": "native-actionmesh-b0",
                "report_ref": source_report_ref, "sequence_ref": sequence_ref,
                "generation_identity_ref": comparison.file_ref(root, identity_path),
                "implementation_ref": comparison.file_ref(root, b0_implementation),
            }, {
                "role": "b_star",
                "method_id": comparison.METHOD_IDS["deterministic_protection"],
                "alias_of": "deterministic_protection",
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
                "kind": "c04-native-comparison-freeze", "version": 1,
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
                             request["role_to_case"]["deterministic_protection"])
            self.assertLessEqual(len(request["scoring_cases"]), 4)
            self.assertFalse(request["dispatch_ready"])
            self.assertTrue(request["generated_unexecuted"])

            # A review reference must stage its per-evidence nested receipts and
            # sources as well as the wrapper; presence alone is not admission.
            retained = workspace / "held-out-raw.json"
            retained.write_text('{"engineering_fixture":true}\n')
            receipt = workspace / "held-out-receipt.json"
            receipt.write_text('{"engineering_fixture":true}\n')
            evidence = workspace / "held-out-evidence.json"
            evidence.write_text(json.dumps({
                "source_refs": [comparison.file_ref(root, retained)],
                "receipt_ref": comparison.file_ref(root, receipt)}))
            specification = workspace / "prospective-spec.json"
            specification.write_text(json.dumps({"implementation_ref": implementation_ref}))
            review = workspace / "semantic-review.json"
            review.write_text(json.dumps({
                "method_spec_ref": comparison.file_ref(root, specification),
                "evidence_refs": [comparison.file_ref(root, evidence)]}))
            freeze["semantic_review_ref"] = comparison.file_ref(root, review)
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with_review = comparison.make_request(root, freeze_path=freeze_path)
            for path in (retained, receipt, evidence, specification, review,
                         output / "artifact.tar", output / "artifact-archive.json"):
                self.assertIn(comparison.file_ref(root, path), with_review["input_refs"])
            self.assertFalse(with_review["dispatch_ready"])

            # Hash-consistent relabelling cannot turn a simple control into C04.
            direct = next(row for row in freeze["roles"]
                          if row["role"] == "deterministic_protection")
            direct["method_id"] = candidate.CANDIDATE_ID
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items()
                if key != "freeze_digest"})
            freeze_path.write_text(json.dumps(freeze) + "\n")
            with self.assertRaisesRegex(ValueError, "relabels retained C04 method"):
                comparison.make_request(root, freeze_path=freeze_path)

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"a":1,"a":2}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                comparison.read_json(path)


if __name__ == "__main__":
    unittest.main()
