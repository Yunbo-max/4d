"""Authored plan checks for C14; Web does not execute this file."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CorotationalResidualPlanSourceTest(unittest.TestCase):
    def test_plan_is_single_attempt_cpu_only_and_parameter_explicit(self):
        source = Path(__file__).parents[2] / "prepare_corotational_residual_candidate.py"
        tree = ast.parse(source.read_text())
        text = source.read_text()
        self.assertIn('"max_attempts": 1', text)
        self.assertIn('"max_retries_per_trial": 0', text)
        self.assertIn('"gpu_count": 0', text)
        self.assertIn("--weight", text)
        self.assertIn("--rho", text)
        self.assertIn("--absolute-tolerance", text)
        self.assertIn("--relative-tolerance", text)
        self.assertIn("--max-iterations", text)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "build_plans"
                            for node in ast.walk(tree)))

    def test_real_stage_rewrites_and_freezes_cross_root_source_pair(self):
        from prepare_corotational_residual_candidate import build_plans
        import run_experiments

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = Path(__file__).resolve().parents[3]
            package = root / "actionmesh/research_math"
            package.mkdir(parents=True)
            for name in ("__init__.py", "corotational_residual_candidate.py"):
                shutil.copy2(project / "actionmesh/research_math" / name, package / name)
            shutil.copy2(project / "actionmesh/prepare_corotational_residual_candidate.py",
                         root / "actionmesh/prepare_corotational_residual_candidate.py")
            source = root / "external-retained-root/case"
            source.mkdir(parents=True)
            sequence = source / "sequence.npz"
            sequence.write_bytes(b"receipt-bound-sequence-fixture")
            sequence_sha = hashlib.sha256(sequence.read_bytes()).hexdigest()
            (source / "report.json").write_text(json.dumps({
                "status": "completed", "uid": "unit-c14", "seed": 42,
                "sha256": {"sequence.npz": sequence_sha},
            }))
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run([
                "git", "-C", str(root), "-c", "user.name=Fixture",
                "-c", "user.email=fixture@example.invalid", "commit",
                "--allow-empty", "-qm", "fixture",
            ], check=True)
            native, outer = build_plans(
                root, source_sequence=sequence, run_id="c14-stage-fixture",
                weight=0.2, rho=1.0, absolute_tolerance=1e-7,
                relative_tolerance=1e-7, max_iterations=100,
                plan_dir=root / "plans/c14", wall_seconds=300)
            job = native["jobs"][0]
            self.assertEqual(native["limits"]["max_attempts"], 1)
            self.assertEqual(native["limits"]["max_retries_per_trial"], 0)
            self.assertEqual(outer["tasks"][0]["resources"]["gpu_count"], 0)
            self.assertEqual({row["path"] for row in job["input_refs"]}, {
                "external-retained-root/case/sequence.npz",
                "external-retained-root/case/report.json",
            })
            self.assertEqual(len(job["output_paths"]), 5)
            self.assertIsInstance(native["plan_digest"], str)
            self.assertEqual(len(native["plan_digest"]), 64)
            attempt = root / "attempt"
            attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, job, attempt)
            staged = Path(command[command.index("--source-sequence") + 1])
            self.assertEqual(staged, workspace / sequence.relative_to(root))
            staged_report = workspace / (source / "report.json").relative_to(root)
            original = staged.read_bytes()
            original_report = staged_report.read_bytes()
            sequence.write_bytes(b"controller-copy-mutated-after-stage")
            (source / "report.json").write_text("{}")
            self.assertEqual(staged.read_bytes(), original)
            self.assertEqual(staged_report.read_bytes(), original_report)
            self.assertEqual(cwd, workspace / "actionmesh")


if __name__ == "__main__":
    unittest.main()
