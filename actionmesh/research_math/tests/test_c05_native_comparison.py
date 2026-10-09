"""C05 artifact-to-freeze checks; authored unexecuted by the Web supervisor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research_math import c05_candidate_artifacts as artifacts
from research_math import c05_native_comparison as comparison
from research_math import spatial_mode_candidate as methods
from research_math.tests import test_c05_candidate_artifacts as artifact_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def fixture(root: Path, sandbox: Path) -> Path:
    uid, seed = "software-uid", 42
    mode_root, parity, b0_sequence, b0_report = artifact_fixture.fixture(
        root, workspace=sandbox)
    output = sandbox / "candidate"
    artifacts.materialize_candidate(
        root, mode_root, parity, b0_sequence, b0_report, output,
        landmark_count=4, cluster_radius=0.05,
        natural_gate_min_fraction=0.75, localized_radius=1.0,
        temperature=2.0, unary_weight=0.1, spatial_weight=1.0,
        max_sweeps=20, displacement_clip_multiplier=4.0,
        max_artifact_bytes=8 * 1024 * 1024, face_chunk_size=2)

    native_source = Path(artifacts.__file__).with_name("native_context_runner.py")
    identity = sandbox / "generation-identity.json"
    _json(identity, {"kind": "native-context-generation-identity", "version": 1,
        "uid": uid, "generation": {"seed": seed},
        "native_context_qualified": False, "scientific_effect_qualification": False,
        "instrument_code_sha256": {"research_math/native_context_runner.py":
                                    comparison.digest(native_source)}})
    basis = sandbox / "prospective-basis.json"
    _json(basis, {"source": "pre-outcome"})
    decided_at = datetime.now(timezone.utc)
    decision = {"kind": comparison.DECISION_KIND, "version": 1,
        "candidate_id": comparison.CANDIDATE_ID, "uid": uid,
        "inference_seed": seed, "decided_at": decided_at.isoformat(),
        "selected_role": "localized_mean",
        "selected_method_id": methods.METHOD_IDS["localized_mean"],
        comparison.DECISION_NO_OUTCOMES_FIELD: True,
        "selection_basis_refs": [comparison.file_ref(root, basis)]}
    decision["decision_digest"] = comparison.canonical_digest(decision)
    decision_path = sandbox / "decision.json"
    _json(decision_path, decision)

    method_ref = comparison.file_ref(root, Path(methods.__file__))
    rows = [{"role": "b0", "method_id": "native-actionmesh-b0",
        "report_ref": comparison.file_ref(root, b0_report),
        "sequence_ref": comparison.file_ref(root, b0_sequence),
        "generation_identity_ref": comparison.file_ref(root, identity),
        "implementation_ref": comparison.file_ref(root, native_source)},
        {"role": "b_star", "method_id": methods.METHOD_IDS["localized_mean"],
         "alias_of": "localized_mean"}]
    rows.extend({"role": role, "method_id": methods.METHOD_IDS[role],
                 "implementation_ref": method_ref} for role in artifacts.ROLE_ORDER)
    freeze = {"kind": comparison.FREEZE_KIND, "version": 1,
        "candidate_id": comparison.CANDIDATE_ID, "uid": uid,
        "inference_seed": seed, "scoring_seed": 44,
        "primary_metric": comparison.PRIMARY_METRIC,
        "guardrail_metrics": list(comparison.GUARDRAIL_METRICS),
        "frozen_at": (decided_at + timedelta(seconds=1)).isoformat(),
        "source_sequence_ref": comparison.file_ref(root, b0_sequence),
        "source_report_ref": comparison.file_ref(root, b0_report),
        "candidate_artifact_ref": comparison.file_ref(root, output / "candidate.json"),
        "b_star_decision_ref": comparison.file_ref(root, decision_path), "roles": rows}
    freeze["freeze_digest"] = comparison.canonical_digest(freeze)
    freeze_path = sandbox / "freeze.json"
    _json(freeze_path, freeze)
    return freeze_path


class C05NativeComparisonTests(unittest.TestCase):
    def test_seven_roles_shared_json_certificates_and_recursive_closure(self):
        with tempfile.TemporaryDirectory(prefix="c05-comparison-", dir=PROJECT_ROOT) as directory:
            sandbox = Path(directory); freeze = fixture(PROJECT_ROOT, sandbox)
            request = comparison.make_request(PROJECT_ROOT, freeze_path=freeze)
            self.assertEqual([row["role"] for row in request["roles"]],
                             list(comparison.ROLES))
            self.assertEqual(request["logical_denominator"]["n_roles"], 7)
            self.assertEqual(request["role_to_case"]["b_star"],
                             request["role_to_case"]["localized_mean"])
            self.assertTrue(all("certificate_ref" not in row
                                for row in request["scoring_cases"]))
            method_rows = request["roles"][2:]
            self.assertTrue(all(row["shared_certificate_ref"]["path"].endswith(
                                "candidate/certificate.json") for row in method_rows))
            self.assertTrue(all(row["shared_solver_certificate_ref"]["path"].endswith(
                                "candidate/solver-certificate.json") for row in method_rows))
            paths = {ref["path"] for ref in request["input_refs"]}
            self.assertIn(request["candidate_solver_certificate_ref"]["path"], paths)
            self.assertTrue(any(path.endswith("sequences/branch-0000.npz")
                                for path in paths))
            for row in request["roles"]:
                if row.get("alias_of"):
                    target = next(item for item in request["roles"]
                                  if item["role"] == row["alias_of"])
                    self.assertEqual(row["case_id"], target["case_id"])
            comparison.verify_request(PROJECT_ROOT, request)

    def test_one_terminal_role_error_stays_in_denominator_without_a_case(self):
        with tempfile.TemporaryDirectory(prefix="c05-comparison-", dir=PROJECT_ROOT) as directory:
            sandbox = Path(directory)
            with mock.patch.object(
                    artifacts, "union_surface_projected_mean",
                    side_effect=RuntimeError("bounded projected-role failure")):
                freeze = fixture(PROJECT_ROOT, sandbox)
                request = comparison.make_request(PROJECT_ROOT, freeze_path=freeze)
            by_role = {row["role"]: row for row in request["roles"]}
            failed = by_role["surface_projected_mean"]
            self.assertEqual(request["logical_denominator"]["n_roles"], 7)
            self.assertEqual(request["candidate_role_denominator"], 5)
            self.assertEqual(request["candidate_roles_failed"],
                             ["surface_projected_mean"])
            self.assertEqual(failed["preparation_status"], "error")
            self.assertEqual(failed["exception_type"], "RuntimeError")
            self.assertIn("bounded projected-role failure", failed["preparation_error"])
            self.assertIsNone(failed["sequence_ref"])
            self.assertIsNone(failed["case_id"])
            self.assertNotIn("surface_projected_mean",
                {row["role"] for row in request["scoring_cases"]})
            self.assertEqual(by_role["localized_mean"]["preparation_status"],
                             "completed")
            self.assertEqual(by_role["joint_spatial_labels"]["preparation_status"],
                             "completed")

    def test_candidate_outer_seed_and_prospective_decision_are_enforced(self):
        with tempfile.TemporaryDirectory(prefix="c05-comparison-", dir=PROJECT_ROOT) as directory:
            sandbox = Path(directory); freeze_path = fixture(PROJECT_ROOT, sandbox)
            freeze = json.loads(freeze_path.read_text())
            freeze["inference_seed"] = 314
            freeze["freeze_digest"] = comparison.canonical_digest({
                key: value for key, value in freeze.items() if key != "freeze_digest"})
            _json(freeze_path, freeze)
            with self.assertRaisesRegex(ValueError, "outer-seed"):
                comparison.make_request(PROJECT_ROOT, freeze_path=freeze_path)

    def test_tampered_shared_solver_certificate_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="c05-comparison-", dir=PROJECT_ROOT) as directory:
            sandbox = Path(directory); freeze = fixture(PROJECT_ROOT, sandbox)
            solver = sandbox / "candidate/solver-certificate.json"
            solver.write_text(solver.read_text() + " ")
            with self.assertRaises(artifacts.C05ArtifactError):
                comparison.make_request(PROJECT_ROOT, freeze_path=freeze)

    def test_cli_only_builds_a_request(self):
        args = comparison.parse_args([
            "request", "--root", "/project", "--freeze", "/project/freeze.json",
            "--output", "/project/request.json"])
        self.assertEqual(args.operation, "request")


if __name__ == "__main__":  # pragma: no cover - Local execution only
    unittest.main()
