"""Authored plan checks for C15; Web does not execute this file."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from research_math.protected_lowrank_candidate import build_anchored_dct_basis


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProtectedLowrankPlanSourceTest(unittest.TestCase):
    def test_plan_is_single_attempt_cpu_only_and_all_policies_are_explicit(self):
        source = Path(__file__).parents[2] / "prepare_protected_lowrank_candidate.py"
        tree = ast.parse(source.read_text())
        text = source.read_text()
        for literal in (
                '"max_attempts": 1', '"max_retries_per_trial": 0',
                '"gpu_count": 0', '--lambda-value', '--residual-rank-policy',
                '--basis-source-policy', '--basis-orthogonality-tolerance',
                '--protected-coefficient-tolerance', '--numeric-rank-tolerance',
                '--max-artifact-bytes', '--basis-evidence'):
            self.assertIn(literal, text)
        self.assertTrue(any(
            isinstance(node, ast.FunctionDef) and node.name == "build_plans"
            for node in ast.walk(tree)))

    def test_real_stage_freezes_sequence_basis_evidence_and_nested_source_ref(self):
        from prepare_protected_lowrank_candidate import build_plans
        import run_experiments

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = Path(__file__).resolve().parents[3]
            package = root / "actionmesh/research_math"
            package.mkdir(parents=True)
            for name in ("__init__.py", "protected_lowrank_candidate.py"):
                shutil.copy2(project / "actionmesh/research_math" / name, package / name)
            shutil.copy2(project / "actionmesh/prepare_protected_lowrank_candidate.py",
                         root / "actionmesh/prepare_protected_lowrank_candidate.py")
            source = root / "external-retained-root/case"
            source.mkdir(parents=True)
            sequence = source / "sequence.npz"
            sequence.write_bytes(b"receipt-bound-sequence-fixture")
            sequence_sha = _sha(sequence)
            (source / "report.json").write_text(json.dumps({
                "status": "completed", "uid": "unit-c15", "seed": 42,
                "sha256": {"sequence.npz": sequence_sha},
            }))
            policy = root / "docs/c15-basis-policy.md"
            policy.parent.mkdir()
            policy.write_text("Prospective anchored DCT K=3.\n")
            basis = root / "inputs/c15-basis.npy"
            basis.parent.mkdir()
            np.save(basis, build_anchored_dct_basis(16, 3), allow_pickle=False)
            evidence = root / "inputs/c15-basis-evidence.json"
            evidence.write_text(json.dumps({
                "kind": "c15-protection-basis-evidence", "version": "1.0.0",
                "status": "completed", "basis_sha256": _sha(basis),
                "basis_source_policy": "analytic_anchored_dct",
                "confirmation_outcomes_used": False, "prospective_freeze": True,
                "frozen_at": "2026-10-08T12:00:00+00:00",
                "construction_rule": "exact_e0_plus_orthonormal_dct_ii_on_frames_1_to_T_minus_1",
                "frame_count": 16, "basis_rank": 3, "input_sequence_sha256": None,
                "source_refs": [{
                    "path": "docs/c15-basis-policy.md", "sha256": _sha(policy),
                }],
            }))
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run([
                "git", "-C", str(root), "-c", "user.name=Fixture",
                "-c", "user.email=fixture@example.invalid", "commit",
                "--allow-empty", "-qm", "fixture",
            ], check=True)
            native, outer = build_plans(
                root, source_sequence=sequence, basis=basis,
                basis_evidence=evidence,
                basis_source_policy="analytic_anchored_dct",
                lambda_value=0.1,
                residual_rank_policy="candidate_export_numeric_rank",
                basis_orthogonality_tolerance=1e-10,
                protected_coefficient_tolerance=1e-5,
                numeric_rank_tolerance=1e-6,
                max_artifact_bytes=8 * 1024 * 1024,
                run_id="c15-stage-fixture", plan_dir=root / "plans/c15",
                wall_seconds=300)
            job = native["jobs"][0]
            self.assertEqual(native["limits"]["max_attempts"], 1)
            self.assertEqual(native["limits"]["max_retries_per_trial"], 0)
            self.assertEqual(outer["tasks"][0]["resources"]["gpu_count"], 0)
            self.assertEqual({row["path"] for row in job["input_refs"]}, {
                "external-retained-root/case/sequence.npz",
                "external-retained-root/case/report.json",
                "inputs/c15-basis.npy", "inputs/c15-basis-evidence.json",
                "docs/c15-basis-policy.md",
            })
            self.assertEqual(job["output_paths"], [
                "actionmesh/c15-protected-lowrank-output/candidate.json",
                "actionmesh/c15-protected-lowrank-output/artifact-archive.json",
                "actionmesh/c15-protected-lowrank-output/artifact.tar",
            ])
            attempt = root / "attempt"
            attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, job, attempt)
            staged_sequence = Path(command[command.index("--source-sequence") + 1])
            staged_basis = Path(command[command.index("--basis") + 1])
            staged_evidence = Path(command[command.index("--basis-evidence") + 1])
            self.assertEqual(staged_sequence.read_bytes(), sequence.read_bytes())
            self.assertEqual(staged_basis.read_bytes(), basis.read_bytes())
            self.assertEqual(staged_evidence.read_bytes(), evidence.read_bytes())
            self.assertEqual(cwd, workspace / "actionmesh")


if __name__ == "__main__":
    unittest.main()
