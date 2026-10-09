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

from research_math import area_transport_candidate as candidate
from research_math import c06_native_comparison as comparison


class AreaTransportKernelTests(unittest.TestCase):
    def test_witness_accepts_roundoff_total_discrepancy(self):
        source = np.array([.5, .5])
        target = np.array([.5, np.nextafter(.5, 1.)])
        rows, columns, mass = candidate._greedy_witness(source, target)
        observed_source = np.zeros(2); observed_target = np.zeros(2)
        np.add.at(observed_source, rows, mass)
        np.add.at(observed_target, columns, mass)
        np.testing.assert_allclose(observed_source, source, atol=1e-15, rtol=0)
        np.testing.assert_allclose(observed_target, target, atol=1e-15, rtol=0)

    def test_lift_normalizes_actual_mass_and_commutes_with_translation(self):
        rows = np.array([0, 0, 1, 1]); columns = np.array([0, 1, 0, 1])
        plan = np.array([1e-18, 3e-18, .25, .75])
        requested_mass = np.array([1e-15, 1. - 1e-15])
        target = np.array([[1., 2., 3.], [5., 6., 7.]])
        shift = np.array([100., -20., 3.])
        actual = candidate._lift(rows, columns, plan, requested_mass, target)
        translated = candidate._lift(rows, columns, plan, requested_mass, target + shift)
        np.testing.assert_allclose(actual, [[4., 5., 6.], [4., 5., 6.]])
        np.testing.assert_allclose(translated, actual + shift, atol=1e-13, rtol=0)
        with self.assertRaisesRegex(RuntimeError, 'positive mass'):
            candidate._lift(rows, columns, np.array([0., 0., .25, .75]),
                            requested_mass, target)

    def test_uneven_masses_require_relative_marginal_closure(self):
        source = np.array([1e-14, 1. - 1e-14])
        target = np.array([.3, .7])
        rows = np.array([0, 0, 1, 1]); columns = np.array([0, 1, 0, 1])
        costs = np.array([0., 2., 1., 0.])
        plan, report = candidate.sparse_sinkhorn(
            rows, columns, costs, source, target,
            epsilon=.5, tolerance=1e-10, max_iterations=2000)
        self.assertLessEqual(report['maximum_relative_row_residual'], 1e-10)
        self.assertLessEqual(report['maximum_relative_column_residual'], 1e-10)
        self.assertTrue(np.isfinite(plan).all())
        with self.assertRaisesRegex(RuntimeError, 'both frozen marginals'):
            candidate.sparse_sinkhorn(rows, columns, costs, source, target,
                epsilon=.5, tolerance=1e-14, max_iterations=1)

    def test_tiny_positive_witness_mass_is_not_dropped(self):
        delta = np.finfo(np.float64).eps
        rows, columns, mass = candidate._greedy_witness(
            np.array([delta, 1. - delta]),
            np.array([2. * delta, 1. - 2. * delta]))
        row = np.zeros(2); column = np.zeros(2)
        np.add.at(row, rows, mass); np.add.at(column, columns, mass)
        np.testing.assert_allclose(row, [delta, 1. - delta], atol=1e-15, rtol=1e-14)
        np.testing.assert_allclose(column, [2. * delta, 1. - 2. * delta],
                                   atol=1e-15, rtol=1e-14)
        self.assertIn((1, 0), set(zip(rows, columns)))

    def test_knn_boundary_ties_use_original_target_id(self):
        source = np.zeros((2, 2), dtype=np.float64)
        target = np.zeros((4, 2), dtype=np.float64)
        support = candidate.build_sparse_support(
            source, target, np.array([.25, .75]), np.full(4, .25), neighbors=2,
            source_ids=np.array([7, 8]), target_ids=np.array([9, 3, 7, 1]))
        selected = support["knn_columns"][support["knn_rows"] == 0]
        np.testing.assert_array_equal(
            np.array([9, 3, 7, 1])[selected], np.array([3, 1]))

    def test_feasible_sparse_support_carries_both_marginals(self):
        source = np.array([.12, .31, .19, .38], dtype=np.float64)
        target = np.array([.27, .08, .29, .36], dtype=np.float64)
        features = np.arange(12, dtype=np.float64).reshape(4, 3)
        support = candidate.build_sparse_support(
            features, features[::-1], source, target, neighbors=2)
        witness = support["witness"]
        row = np.zeros(4); column = np.zeros(4)
        np.add.at(row, support["rows"], witness)
        np.add.at(column, support["columns"], witness)
        np.testing.assert_allclose(row, source, atol=1e-14, rtol=0)
        np.testing.assert_allclose(column, target, atol=1e-14, rtol=0)
        uniform_row = np.zeros(4); uniform_column = np.zeros(4)
        np.add.at(uniform_row, support["rows"], support["uniform_witness"])
        np.add.at(uniform_column, support["columns"], support["uniform_witness"])
        np.testing.assert_allclose(uniform_row, np.full(4, .25), atol=1e-14)
        np.testing.assert_allclose(uniform_column, np.full(4, .25), atol=1e-14)
        self.assertTrue(np.all(support["witness"] > 0.0))
        self.assertTrue(np.all(support["uniform_witness"] > 0.0))
        self.assertEqual(len(set(zip(support["rows"], support["columns"]))),
                         len(support["rows"]))

    def test_triangular_support_is_closed_before_sinkhorn(self):
        marginal = np.array([.5, .5], dtype=np.float64)
        triangular = {(0, 0), (0, 1), (1, 1)}
        first = candidate._strict_witness_on_support(
            triangular, marginal, marginal)
        self.assertIn((1, 0), first)
        closed = set(first)
        witness = candidate._strict_witness_on_support(
            closed, marginal, marginal)
        ordered = sorted(closed)
        rows = np.array([row for row, _ in ordered], dtype=np.int64)
        columns = np.array([column for _, column in ordered], dtype=np.int64)
        costs = np.array([0., 1., 1., 0.], dtype=np.float64)
        plan, diagnostics = candidate.sparse_sinkhorn(
            rows, columns, costs, marginal, marginal,
            epsilon=.5, tolerance=1e-12, max_iterations=2000)
        self.assertTrue(all(witness[key] > 0.0 for key in ordered))
        self.assertTrue(np.all(plan > 0.0))
        self.assertLessEqual(diagnostics["maximum_marginal_residual"], 1e-12)

    def test_geometry_descriptor_is_rigid_rotation_invariant(self):
        vertices, faces = self._sequence()
        angle = .731
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.],
            [0., 0., 1.],
        ])
        first = candidate.geometry_descriptors(vertices[3], faces)
        second = candidate.geometry_descriptors(
            vertices[3].astype(np.float64) @ rotation.T + np.array([3., -2., 5.]),
            faces)
        np.testing.assert_allclose(first[0], second[0], atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(first[1], second[1], atol=2e-12, rtol=2e-12)
        np.testing.assert_array_equal(first[2], second[2])

    def test_vertex_of_retained_near_threshold_face_stays_active(self):
        vertices = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 4e-14, 0.],
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        descriptor, mass, active_ids = candidate.geometry_descriptors(
            vertices, faces)
        self.assertEqual(descriptor.shape[0], 4)
        self.assertTrue(np.all(mass > 0.0))
        np.testing.assert_array_equal(active_ids, np.arange(4))

    def test_sparse_sinkhorn_closes_area_marginals(self):
        source = np.array([.2, .3, .5], dtype=np.float64)
        target = np.array([.4, .15, .45], dtype=np.float64)
        x = np.array([[0., 0.], [1., 0.], [2., 0.]])
        y = np.array([[0., 0.], [1.2, 0.], [2.1, 0.]])
        support = candidate.build_sparse_support(x, y, source, target, neighbors=2)
        plan, diagnostics = candidate.sparse_sinkhorn(
            support["rows"], support["columns"], support["costs"],
            source, target, epsilon=.4, tolerance=1e-10, max_iterations=5000)
        row = np.zeros(3); column = np.zeros(3)
        np.add.at(row, support["rows"], plan)
        np.add.at(column, support["columns"], plan)
        np.testing.assert_allclose(row, source, atol=1e-9, rtol=0)
        np.testing.assert_allclose(column, target, atol=1e-9, rtol=0)
        self.assertLessEqual(diagnostics["maximum_marginal_residual"], 1e-9)

    def test_rectangular_active_sets_preserve_full_native_vertex_ids(self):
        vertices, _ = self._sequence()
        vertices = vertices.copy()
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]], dtype=np.int64)
        # Vertex 4 has positive anchor area but its only incident face becomes
        # degenerate in frame 1, so the first transport problem is 5 x 4.
        vertices[1, 4] = vertices[1, 0]
        for role in candidate.ROLES:
            output, certificate, report = candidate.build_role_sequence(
                vertices, faces, role, epsilon=.5, neighbors=3,
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
            vertices, faces, epsilon=.5, neighbors=3,
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
            outputs["row_softmax"][1], outputs["area_marginal_transport"][1]))
        np.testing.assert_array_equal(
            certificates["row_softmax"]["support_sha256"],
            certificates["area_marginal_transport"]["support_sha256"])
        self.assertTrue(np.all(
            certificates["row_softmax"]["column_marginal_enforced"] == 0))

    def test_degenerate_face_is_removed_but_native_identity_is_retained(self):
        vertices, _ = self._sequence()
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 0, 1]], dtype=np.int64)
        output, certificate, report = candidate.build_role_sequence(
            vertices, faces, "area_marginal_transport", epsilon=.5,
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


class C06RetainedNativeAcceptance(unittest.TestCase):
    def test_receipt_bound_native_artifact_reconstruction(self):
        root = os.environ.get('C06_NATIVE_ROOT')
        artifact = os.environ.get('C06_NATIVE_ARTIFACT')
        if not root or not artifact:
            self.skipTest('Real C06 native artifact required; fixtures cannot supply native acceptance')
        checked = candidate.validate_candidate_artifact(Path(root), Path(artifact))
        self.assertEqual(checked['candidate_id'], candidate.CANDIDATE_ID)
        self.assertEqual(tuple(checked['roles']), candidate.ROLES)
        self.assertEqual(checked['status'], 'completed',
                         'A retained method failure leaves Local acceptance pending')
        self.assertTrue(all(row['status'] == 'completed' for row in checked['arms']))
        self.assertFalse(checked['native_qualified'])
        self.assertFalse(checked['scientific_admission'])


class AreaTransportArtifactTests(unittest.TestCase):
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
            "kind": "c06-candidate-producer-provenance", "version": 1,
            "code_refs": sorted(refs, key=lambda ref: ref["path"]),
            "environment": {
                "python_executable": sys.executable,
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "scipy": importlib.metadata.version("scipy"),
                "scope": "CPU geometry-only C06 artifacts; no model, GT, scorer or GPU",
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
            vertices, faces = AreaTransportKernelTests._sequence()
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
            candidate.export_area_transport_candidate(
                case, output, uid="unit", expected_sequence_sha256=sequence_sha,
                source_refs=source_refs, producer_provenance_ref=provenance_ref,
                producer_provenance_path=provenance, epsilon=.5, neighbors=3,
                tolerance=1e-9, max_iterations=3000,
                coordinate_lower=-10., coordinate_upper=10.,
                bounds_policy="preserve_and_report", max_artifact_bytes=10_000_000)
            verified = candidate.validate_candidate_artifact(
                root, output / "candidate.json")
            self.assertEqual(verified["completed_roles"], list(candidate.ROLES),
                [(path.as_posix(), path.read_text()) for path in output.glob("*/report.json")])
            self.assertEqual(set(verified["artifact_archive"]), {"archive", "record"})
            with np.load(output / "area_marginal_transport" / "sequence.npz",
                         allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["vertices"][0], vertices[0])

            sequence_path = output / "area_marginal_transport" / "sequence.npz"
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
            uid = "fixture-c06"
            vertices, faces = AreaTransportKernelTests._sequence()
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
            candidate.export_area_transport_candidate(
                source, output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_refs={"sequence": sequence_ref, "report": report_ref},
                producer_provenance_ref=provenance_ref,
                producer_provenance_path=provenance,
                epsilon=.5, neighbors=3, tolerance=1e-9,
                max_iterations=3000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            basis = work / "basis.json"; basis.write_text("{}\n")
            stamp = datetime.now(timezone.utc).isoformat()
            decision = {
                "kind": comparison.DECISION_KIND, "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "decided_at": stamp,
                "selected_role": "vertex_density_transport",
                "selected_method_id":
                    candidate.METHOD_IDS["vertex_density_transport"],
                "selected_without_c06_native_outcomes": True,
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
                "method_id": candidate.METHOD_IDS["vertex_density_transport"],
                "alias_of": "vertex_density_transport",
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
                "primary_metric": "cd_3d",
                "guardrail_metrics": ["cd_4d", "cd_motion"],
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
            self.assertEqual(request["logical_denominator"]["n_roles"], 5)
            self.assertEqual(request["role_to_case"]["b_star"],
                             request["role_to_case"]["vertex_density_transport"])
            self.assertFalse(request["dispatch_ready"])

            failed_output = work / "failed-candidate"
            candidate.export_area_transport_candidate(
                source, failed_output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_refs={"sequence": sequence_ref, "report": report_ref},
                producer_provenance_ref=provenance_ref,
                producer_provenance_path=provenance,
                epsilon=.5, neighbors=3, tolerance=1e-9,
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
                "selected_without_c06_native_outcomes": True,
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
