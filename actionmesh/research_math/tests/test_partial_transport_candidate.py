import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
import platform
import sys

import numpy as np

from research_math import partial_transport_candidate as candidate
from research_math import c07_native_comparison as comparison


class PartialTransportKernelTests(unittest.TestCase):
    def test_partial_solver_obeys_capacity_and_rejects_cost_above_two_gamma(self):
        rows = np.array([0, 0, 1, 1]); columns = np.array([0, 1, 0, 1])
        costs = np.array([0.0, 10.0, 10.0, 10.0])
        mass = np.array([0.5, 0.5])
        plan, report = candidate.sparse_partial_transport(
            rows, columns, costs, mass, mass, gamma=1.0,
            epsilon=0.02, tolerance=1e-9, max_iterations=5000)
        row = np.zeros(2); column = np.zeros(2)
        np.add.at(row, rows, plan); np.add.at(column, columns, plan)
        self.assertTrue(np.all(row <= mass + 1e-9))
        self.assertTrue(np.all(column <= mass + 1e-9))
        self.assertGreater(row[0], 0.49)
        self.assertLess(row[1], 1e-6)
        self.assertGreater(report['unmatched_source_mass'], 0.49)
        self.assertLessEqual(report['maximum_capacity_violation'], 1e-9)

    def test_two_gamma_boundary_controls_matching(self):
        rows = columns = np.array([0])
        mass = np.array([1.0])
        low, _ = candidate.sparse_partial_transport(
            rows, columns, np.array([1.99]), mass, mass, gamma=1.0,
            epsilon=0.01, tolerance=1e-10, max_iterations=5000)
        high, _ = candidate.sparse_partial_transport(
            rows, columns, np.array([2.02]), mass, mass, gamma=1.0,
            epsilon=0.01, tolerance=1e-10, max_iterations=5000)
        self.assertGreater(low[0], 0.70)
        self.assertLess(high[0], 0.30)

    def test_partial_lift_blends_unmatched_mass_with_native_prediction(self):
        rows = np.array([0, 0, 1]); columns = np.array([0, 1, 1])
        plan = np.array([0.1, 0.1, 0.0]); source_mass = np.array([0.4, 0.6])
        target = np.array([[10., 0., 0.], [20., 0., 0.]])
        native = np.array([[2., 0., 0.], [7., 0., 0.]])
        output, fraction = candidate._partial_lift(
            rows, columns, plan, source_mass, target, native)
        np.testing.assert_allclose(fraction, [0.5, 0.0])
        np.testing.assert_allclose(output, [[8.5, 0., 0.], [7., 0., 0.]])

    def test_rectangular_active_sets_preserve_full_native_vertex_ids(self):
        vertices, _ = self._sequence()
        vertices = vertices.copy()
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]], dtype=np.int64)
        # Vertex 4 has positive anchor area but its only incident face becomes
        # degenerate in frame 1, so the first transport problem is 5 x 4.
        vertices[1, 4] = vertices[1, 0]
        for role in candidate.ROLES:
            output, certificate, report = candidate.build_role_sequence(
                vertices, faces, role, gamma=1.0, epsilon=.5, neighbors=3,
                tolerance=1e-9, max_iterations=3000)
            self.assertEqual(output.shape, vertices.shape)
            np.testing.assert_array_equal(output[0], vertices[0])
            self.assertTrue(np.isfinite(output).all())
            self.assertEqual(certificate["active_source_mask"][0].sum(), 5)
            self.assertEqual(certificate["active_target_mask"][0].sum(), 4)
            self.assertEqual(report["transport_active_source_vertices"], 5)

    def test_three_roles_share_geometry_and_keep_exact_anchor(self):
        vertices, faces = self._sequence()
        outputs, reports, certificates, shared = candidate.build_transport_sequences(
            vertices, faces, gamma=1.0, epsilon=.5, neighbors=3,
            tolerance=1e-9, max_iterations=3000)
        self.assertEqual(set(outputs), set(candidate.ROLES))
        for role, value in outputs.items():
            self.assertEqual(value.dtype, np.dtype(np.float32))
            np.testing.assert_array_equal(value[0], vertices[0])
            self.assertTrue(np.isfinite(value).all(), role)
            self.assertEqual(reports[role]["frames"], 15)
            self.assertEqual(certificates[role]["support_sha256"].shape, (15,))
        self.assertEqual(shared["feature_source"],
                         "predicted_mesh_geometry_only_no_gt_or_model_attention")
        self.assertFalse(np.array_equal(
            outputs["confidence_threshold_fallback"][1],
            outputs["partial_mass_native_fallback"][1]))
        np.testing.assert_array_equal(
            certificates["full_mass_transport"]["support_sha256"],
            certificates["partial_mass_native_fallback"]["support_sha256"])
        self.assertEqual(outputs["partial_mass_native_fallback"].shape, vertices.shape)

    def test_degenerate_face_is_removed_but_native_identity_is_retained(self):
        vertices, _ = self._sequence()
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 0, 1]], dtype=np.int64)
        output, certificate, report = candidate.build_role_sequence(
            vertices, faces, "partial_mass_native_fallback", gamma=1.0, epsilon=.5,
            neighbors=3, tolerance=1e-9, max_iterations=3000)
        self.assertEqual(output.shape, vertices.shape)
        np.testing.assert_array_equal(output[0], vertices[0])
        np.testing.assert_array_equal(output[:, 4], vertices[:, 4])
        self.assertEqual(certificate["active_source_mask"].shape, (15, 5))
        self.assertEqual(report["native_fallback_source_vertices"], 1)

    @staticmethod
    def _sequence():
        anchor = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [.5, .5, .7]], dtype=np.float32)
        faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                          [0, 3, 2], [0, 2, 1]], dtype=np.int64)
        frames = []
        for index in range(16):
            value = anchor.copy()
            value[:, 2] += np.float32(.01 * index) * np.array(
                [0., 1., 2., 1., 3.], dtype=np.float32)
            value[:, 0] += np.float32(.002 * index) * np.array(
                [0., -1., 0., 1., 0.], dtype=np.float32)
            frames.append(value)
        return np.stack(frames), faces


class C07RetainedNativeAcceptance(unittest.TestCase):
    def test_receipt_bound_native_artifact_reconstruction(self):
        root = os.environ.get('C07_NATIVE_ROOT')
        artifact = os.environ.get('C07_NATIVE_ARTIFACT')
        if not root or not artifact:
            self.skipTest('Real C07 native artifact required; fixtures cannot supply native acceptance')
        checked = candidate.validate_candidate_artifact(Path(root), Path(artifact))
        self.assertEqual(checked['candidate_id'], candidate.CANDIDATE_ID)
        self.assertEqual(tuple(checked['roles']), candidate.ROLES)
        self.assertEqual(checked['status'], 'completed',
                         'A retained method failure leaves Local acceptance pending')
        self.assertTrue(all(row['status'] == 'completed' for row in checked['arms']))
        self.assertFalse(checked['native_qualified'])
        self.assertFalse(checked['scientific_admission'])


class PartialTransportArtifactTests(unittest.TestCase):
    @staticmethod
    def _producer(root: Path, directory: Path, source_refs, code_paths=None):
        source_root = Path(candidate.__file__).resolve().parents[2]
        code_paths = []
        for relative in candidate.PRODUCER_CODE_PATHS:
            source = source_root / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                destination.write_bytes(source.read_bytes())
            code_paths.append(destination)
        refs = [{"path": path.resolve().relative_to(root.resolve()).as_posix(),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in code_paths]
        provenance = directory / "producer-provenance.json"
        provenance.write_text(json.dumps({
            "kind": "c07-candidate-producer-provenance", "version": 1,
            "code_refs": sorted(refs, key=lambda ref: ref["path"]),
            "environment": {
                "python_executable": sys.executable,
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "scipy": importlib.metadata.version("scipy"),
                "scope": "CPU partial-transport C07 artifacts; no model, GT, scorer or GPU",
                "feature_source": candidate.FEATURE_SOURCE},
            "source_refs": source_refs,
            "execution_contract": candidate.PRODUCER_EXECUTION_CONTRACT,
        }))
        return provenance, {
            "path": provenance.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()}

    def test_producer_closure_rejects_code_environment_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_refs = [
                {"path": "input/sequence.npz", "sha256": "1" * 64},
                {"path": "input/report.json", "sha256": "2" * 64},
            ]
            provenance, _ = self._producer(root, root, source_refs)
            original = json.loads(provenance.read_text())
            mutations = []
            changed = json.loads(json.dumps(original))
            changed["code_refs"][0]["sha256"] = "0" * 64
            mutations.append(changed)
            changed = json.loads(json.dumps(original))
            changed["environment"]["scope"] = "changed"
            mutations.append(changed)
            changed = json.loads(json.dumps(original))
            changed["source_refs"][0]["sha256"] = "0" * 64
            mutations.append(changed)
            changed = json.loads(json.dumps(original))
            changed["execution_contract"]["max_attempts"] = 2
            mutations.append(changed)
            expected = {"sequence": source_refs[0], "report": source_refs[1]}
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    with self.assertRaises(ValueError):
                        candidate._validate_producer_provenance(
                            root, mutation, expected)

    def test_export_and_replay_bind_real_source_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "inputs" / "case"; case.mkdir(parents=True)
            vertices, faces = PartialTransportKernelTests._sequence()
            sequence = case / "sequence.npz"
            np.savez_compressed(
                sequence, vertices=vertices, faces=faces,
                timesteps=np.arange(16, dtype=np.float32),
                frame_indices=np.arange(16, dtype=np.int64),
                query_vertex_ids=np.arange(vertices.shape[1], dtype=np.int64))
            sequence_sha = hashlib.sha256(sequence.read_bytes()).hexdigest()
            report = {"status": "completed", "uid": "unit", "seed": 42,
                      "sha256": {"sequence.npz": sequence_sha}}
            (case / "report.json").write_text(json.dumps(report))
            source_refs = {
                "sequence": {"path": "inputs/case/sequence.npz",
                             "sha256": sequence_sha},
                "report": {"path": "inputs/case/report.json",
                           "sha256": hashlib.sha256(
                               (case / "report.json").read_bytes()).hexdigest()},
            }
            output = root / "artifact"
            provenance, provenance_ref = self._producer(
                root, root, [source_refs["sequence"], source_refs["report"]])
            candidate.export_partial_transport_candidate(
                case, output, uid="unit", expected_sequence_sha256=sequence_sha,
                source_refs=source_refs, producer_provenance_ref=provenance_ref,
                producer_provenance_path=provenance, gamma=1.0,
                epsilon=.5, neighbors=3,
                tolerance=1e-9, max_iterations=3000,
                coordinate_lower=-10., coordinate_upper=10.,
                bounds_policy="preserve_and_report", max_artifact_bytes=10_000_000)
            verified = candidate.validate_candidate_artifact(
                root, output / "candidate.json")
            self.assertEqual(verified["completed_roles"], list(candidate.ROLES))
            self.assertEqual(set(verified["artifact_archive"]), {"archive", "record"})
            with np.load(output / "partial_mass_native_fallback" / "sequence.npz",
                         allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["vertices"][0], vertices[0])

            sequence_path = output / "partial_mass_native_fallback" / "sequence.npz"
            original = sequence_path.read_bytes()
            sequence_path.write_bytes(original + b"tamper")
            with self.assertRaises(ValueError):
                candidate.validate_candidate_artifact(root, output / "candidate.json")
            sequence_path.write_bytes(original)
            archive_path = output / "artifact.tar"
            archive = archive_path.read_bytes()
            archive_path.write_bytes(archive[:-1] + bytes([archive[-1] ^ 1]))
            with self.assertRaises(ValueError):
                candidate.validate_candidate_artifact(root, output / "candidate.json")

    def test_real_artifact_freezes_five_role_native_request(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            work = Path(directory); source = work / "source"; source.mkdir()
            uid = "fixture-c07"
            vertices, faces = PartialTransportKernelTests._sequence()
            np.savez_compressed(
                source / "sequence.npz", vertices=vertices, faces=faces,
                timesteps=np.arange(16, dtype=np.float32),
                frame_indices=np.arange(16, dtype=np.int64),
                query_vertex_ids=np.arange(vertices.shape[1], dtype=np.int64))
            sequence_ref = comparison.file_ref(root, source / "sequence.npz")
            (source / "report.json").write_text(json.dumps({
                "status": "completed", "uid": uid, "seed": 42,
                "sha256": {"sequence.npz": sequence_ref["sha256"]}}) + "\n")
            report_ref = comparison.file_ref(root, source / "report.json")
            output = work / "candidate"
            code_paths = [Path(candidate.__file__),
                          root / "actionmesh/research_math/protected_geometry_candidate.py",
                          root / "actionmesh/research_ten/m01_elasticity.py"]
            provenance, provenance_ref = self._producer(
                root, work, [sequence_ref, report_ref], code_paths)
            candidate.export_partial_transport_candidate(
                source, output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_refs={"sequence": sequence_ref, "report": report_ref},
                producer_provenance_ref=provenance_ref,
                producer_provenance_path=provenance,
                gamma=1.0, epsilon=.5, neighbors=3, tolerance=1e-9,
                max_iterations=3000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            basis = work / "basis.json"; basis.write_text("{}\n")
            stamp = datetime.now(timezone.utc).isoformat()
            decision = {
                "kind": comparison.DECISION_KIND, "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "decided_at": stamp,
                "selected_role": "full_mass_transport",
                "selected_method_id":
                    candidate.METHOD_IDS["full_mass_transport"],
                "selected_without_c07_native_outcomes": True,
                "selection_basis_refs": [comparison.file_ref(root, basis)],
            }
            decision["decision_digest"] = comparison.canonical_digest(decision)
            decision_path = work / "decision.json"
            decision_path.write_text(json.dumps(decision) + "\n")
            identity_path = work / "generation-identity.json"
            b0_implementation = root / "actionmesh/research_math/native_context_runner.py"
            identity_path.write_text(json.dumps({
                "kind": "native-context-generation-identity", "version": 1,
                "scope": "paired engineering observer/replay; no candidate or scorer execution",
                "uid": uid, "generation": {"seed": 42},
                "verified_unit_manifest": {"fixture": True},
                "retained_input_refs": {"fixture": {
                    "path": "inputs/fixture.json", "sha256": "a" * 64}},
                "upstream_source_sha256": {"model.py": "b" * 64},
                "gpu_uuid": "GPU-fixture", "native_context_qualified": False,
                "scientific_effect_qualification": False,
                "instrument_code_sha256": {
                    "research_math/native_context_runner.py":
                        comparison.digest(b0_implementation)},
            }) + "\n")
            implementation_ref = comparison.file_ref(root, Path(candidate.__file__))
            rows = [{
                "role": "b0", "method_id": "native-actionmesh-b0",
                "report_ref": report_ref, "sequence_ref": sequence_ref,
                "generation_identity_ref": comparison.file_ref(root, identity_path),
                "implementation_ref": comparison.file_ref(root, b0_implementation),
            }, {
                "role": "b_star",
                "method_id": candidate.METHOD_IDS["full_mass_transport"],
                "alias_of": "full_mass_transport",
            }]
            for role in candidate.ROLES:
                rows.append({
                    "role": role, "method_id": candidate.METHOD_IDS[role],
                    "report_ref": comparison.file_ref(
                        root, output / role / "report.json"),
                    "sequence_ref": comparison.file_ref(
                        root, output / role / "sequence.npz"),
                    "certificate_ref": comparison.file_ref(
                        root, output / role / "certificate.npz"),
                    "implementation_ref": implementation_ref,
                })
            freeze = {
                "kind": comparison.FREEZE_KIND, "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "scoring_seed": 44,
                "primary_metric": "cd_motion",
                "guardrail_metrics": ["cd_3d", "cd_4d"],
                "frozen_at": stamp, "source_sequence_ref": sequence_ref,
                "source_report_ref": report_ref,
                "candidate_artifact_ref": comparison.file_ref(
                    root, output / "candidate.json"),
                "b_star_decision_ref": comparison.file_ref(root, decision_path),
                "roles": rows,
            }
            freeze["freeze_digest"] = comparison.canonical_digest(freeze)
            freeze_path = work / "freeze.json"
            freeze_path.write_text(json.dumps(freeze) + "\n")
            request = comparison.make_request(root, freeze_path=freeze_path)
            self.assertEqual([row["role"] for row in request["roles"]],
                             list(comparison.ROLES))
            self.assertEqual(request["primary_metric"], "cd_motion")
            self.assertEqual(request["guardrail_metrics"], ["cd_3d", "cd_4d"])
            self.assertEqual(request["logical_denominator"]["n_roles"], 5)
            self.assertEqual(request["role_to_case"]["b_star"],
                             request["role_to_case"]["full_mass_transport"])
            self.assertFalse(request["dispatch_ready"])

            failed_output = work / "failed-candidate"
            candidate.export_partial_transport_candidate(
                source, failed_output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_refs={"sequence": sequence_ref, "report": report_ref},
                producer_provenance_ref=provenance_ref,
                producer_provenance_path=provenance,
                gamma=1.0, epsilon=.5, neighbors=3, tolerance=1e-9,
                max_iterations=3000, coordinate_lower=-1e-8,
                coordinate_upper=1e-8, bounds_policy="reject",
                max_artifact_bytes=32 * 1024 * 1024)
            failed_verified = candidate.validate_candidate_artifact(
                root, failed_output / "candidate.json")
            self.assertEqual(failed_verified["completed_roles"], [])
            failed_decision = {
                "kind": comparison.DECISION_KIND, "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "decided_at": stamp,
                "selected_role": "b0", "selected_method_id": "native-actionmesh-b0",
                "selected_without_c07_native_outcomes": True,
                "selection_basis_refs": [comparison.file_ref(root, basis)],
            }
            failed_decision["decision_digest"] = comparison.canonical_digest(
                failed_decision)
            failed_decision_path = work / "failed-decision.json"
            failed_decision_path.write_text(json.dumps(failed_decision) + "\n")
            failed_rows = [rows[0], {
                "role": "b_star", "method_id": "native-actionmesh-b0",
                "alias_of": "b0",
            }]
            for role in candidate.ROLES:
                failed_rows.append({
                    "role": role, "method_id": candidate.METHOD_IDS[role],
                    "report_ref": comparison.file_ref(
                        root, failed_output / role / "report.json"),
                    "sequence_ref": None, "certificate_ref": None,
                    "implementation_ref": implementation_ref,
                })
            failed_freeze = {
                **{key: value for key, value in freeze.items()
                   if key != "freeze_digest"},
                "candidate_artifact_ref": comparison.file_ref(
                    root, failed_output / "candidate.json"),
                "b_star_decision_ref": comparison.file_ref(
                    root, failed_decision_path),
                "roles": failed_rows,
            }
            failed_freeze["freeze_digest"] = comparison.canonical_digest(
                failed_freeze)
            failed_freeze_path = work / "failed-freeze.json"
            failed_freeze_path.write_text(json.dumps(failed_freeze) + "\n")
            failed_request = comparison.make_request(
                root, freeze_path=failed_freeze_path)
            self.assertEqual(failed_request["logical_denominator"]["n_roles"], 5)
            for row in failed_request["roles"][2:]:
                self.assertEqual(row["preparation_status"], "error")
                self.assertIn("preparation_error", row)
                self.assertIsNone(row["case_id"])


if __name__ == "__main__":
    unittest.main()
