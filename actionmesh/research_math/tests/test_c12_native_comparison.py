"""Real C12 artifact-to-freeze integration fixture; not scientific evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import area_admission_candidate as candidate
from research_math import c12_native_comparison as comparison


class C12NativeComparisonTests(unittest.TestCase):
    def test_complete_artifact_freezes_five_roles_and_aliases_bstar(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            work = Path(directory); source = work / "source"; source.mkdir()
            uid = "fixture-c12"
            anchor = np.array([[0., 0., 0.], [1., 0., 0.],
                               [0., 1., 0.], [1., 1., 0.]], dtype=np.float32)
            vertices = np.repeat(anchor[None], 16, axis=0)
            vertices[:, :, 2] = np.linspace(0., .1, 16, dtype=np.float32)[:, None]
            faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
            np.savez_compressed(
                source / "sequence.npz", vertices=vertices, faces=faces,
                timesteps=np.arange(16, dtype=np.float32),
                frame_indices=np.arange(16, dtype=np.int64),
                query_vertex_ids=np.arange(4, dtype=np.int64))
            sequence_ref = comparison.file_ref(root, source / "sequence.npz")
            (source / "report.json").write_text(json.dumps({
                "status": "completed", "uid": uid, "seed": 42,
                "sha256": {"sequence.npz": sequence_ref["sha256"]}}) + "\n")
            report_ref = comparison.file_ref(root, source / "report.json")
            output = work / "candidate"
            candidate.export_area_admission_candidate(
                source, output, uid=uid,
                expected_sequence_sha256=sequence_ref["sha256"],
                source_sequence_ref=sequence_ref["path"],
                source_report_ref=report_ref["path"], beta=.2,
                fixed_alpha=.5, backtracking_factor=.5,
                backtracking_max_steps=8, degeneracy_epsilon=1e-12,
                root_tolerance=1e-10, absolute_margin=1e-8,
                relative_margin=1e-8, arap_weight=1., temporal_weight=.1,
                arap_iterations=2, cg_tolerance=1e-8,
                cg_max_iterations=1000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            basis = work / "basis.json"; basis.write_text("{}\n")
            decision = {
                "kind": comparison.DECISION_KIND, "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42,
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "selected_role": "generic_backtracking",
                "selected_method_id": candidate.METHOD_IDS["generic_backtracking"],
                "selected_without_c12_native_outcomes": True,
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
            implementation_ref = comparison.file_ref(
                root, root / "actionmesh/research_math/area_admission_candidate.py")
            rows = [{
                "role": "b0", "method_id": "native-actionmesh-b0",
                "report_ref": report_ref, "sequence_ref": sequence_ref,
                "generation_identity_ref": comparison.file_ref(root, identity_path),
                "implementation_ref": comparison.file_ref(root, b0_implementation),
            }, {
                "role": "b_star",
                "method_id": candidate.METHOD_IDS["generic_backtracking"],
                "alias_of": "generic_backtracking",
            }]
            for role in candidate.ROLES:
                rows.append({
                    "role": role, "method_id": candidate.METHOD_IDS[role],
                    "report_ref": comparison.file_ref(root, output / role / "report.json"),
                    "sequence_ref": comparison.file_ref(root, output / role / "sequence.npz"),
                    "certificate_ref": comparison.file_ref(root, output / role / "certificate.npz"),
                    "implementation_ref": implementation_ref,
                })
            freeze = {
                "kind": comparison.FREEZE_KIND, "version": 1,
                "candidate_id": candidate.CANDIDATE_ID, "uid": uid,
                "inference_seed": 42, "scoring_seed": 44,
                "primary_metric": "cd_3d",
                "guardrail_metrics": ["cd_4d", "cd_motion"],
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "source_sequence_ref": sequence_ref,
                "source_report_ref": report_ref,
                "candidate_artifact_ref": comparison.file_ref(root, output / "candidate.json"),
                "b_star_decision_ref": comparison.file_ref(root, decision_path),
                "roles": rows,
            }
            freeze["freeze_digest"] = comparison.canonical_digest(freeze)
            freeze_path = work / "freeze.json"
            freeze_path.write_text(json.dumps(freeze) + "\n")
            request = comparison.make_request(root, freeze_path=freeze_path)
            self.assertEqual(request["kind"], comparison.REQUEST_KIND)
            self.assertEqual([row["role"] for row in request["roles"]],
                             list(comparison.ROLES))
            self.assertEqual(request["role_to_case"]["b_star"],
                             request["role_to_case"]["generic_backtracking"])
            self.assertEqual(request["logical_denominator"]["n_roles"], 5)
            self.assertFalse(request["dispatch_ready"])


if __name__ == "__main__":
    unittest.main()
