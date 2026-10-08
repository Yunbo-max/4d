"""Plan-only C10 contracts; no method, model, scorer or GPU is run here."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


try:
    planner = importlib.import_module("prepare_integrable_gradient_candidate")
except ModuleNotFoundError:
    planner = None


class IntegrableGradientPlanTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, "C10 CPU artifact plan builder is missing")

    def project(self, root):
        code = root / "actionmesh/research_math"
        code.mkdir(parents=True)
        (code / "__init__.py").write_text("")
        (code / "integrable_gradient_candidate.py").write_text("fixture source only\n")
        case = root / "inputs/native-case"
        case.mkdir(parents=True)
        sequence = case / "sequence.npz"
        sequence.write_bytes(b"opaque receipt-bound native sequence")
        digest = hashlib.sha256(sequence.read_bytes()).hexdigest()
        (case / "report.json").write_text(json.dumps({
            "status": "completed", "uid": "fixture", "seed": 42,
            "sha256": {"sequence.npz": digest}}))
        return sequence

    def build(self, root, sequence, plan_dir=None, **overrides):
        options = dict(
            root=root, source_sequence=sequence, run_id="c10-fixture",
            target_strength=.5, max_relative_change=.25,
            absolute_tolerance=1e-10, relative_tolerance=1e-9,
            max_iterations=10000, coordinate_lower=-1., coordinate_upper=1.,
            bounds_policy="preserve_and_report",
            max_artifact_bytes=16 * 1024 * 1024,
            plan_dir=plan_dir or root / "plans", wall_seconds=600)
        options.update(overrides)
        return planner.build_plans(**options)

    def test_plan_pins_receipt_and_is_single_attempt_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.project(root)
            native, outer = self.build(root, sequence)
            job = native["jobs"][0]
            self.assertEqual({item["path"] for item in job["input_refs"]}, {
                "inputs/native-case/sequence.npz", "inputs/native-case/report.json"})
            self.assertEqual(job["arm_role"],
                             "three-common-target-artifacts-no-scorer-no-admission")
            self.assertEqual(native["limits"]["max_attempts"], 1)
            self.assertEqual(native["limits"]["max_retries_per_trial"], 0)
            self.assertEqual(outer["tasks"][0]["resources"]["gpu_count"], 0)
            self.assertEqual(outer["limits"]["max_gpu_task_seconds"], 0)
            self.assertIn(str(sequence), job["command"])
            self.assertIn("--bounds-policy", job["command"])
            self.assertEqual(job["output_paths"], [
                "actionmesh/c10-integrable-output/candidate.json",
                "actionmesh/c10-integrable-output/artifact-archive.json",
                "actionmesh/c10-integrable-output/artifact.tar"])

    def test_plan_declares_fixed_terminal_archive_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.project(root)
            native, _ = self.build(root, sequence)
            outputs = native["jobs"][0]["output_paths"]
            self.assertIn("actionmesh/c10-integrable-output/artifact.tar", outputs)
            self.assertIn("--max-artifact-bytes", native["jobs"][0]["command"])

    def test_changed_receipt_invalid_parameters_and_outside_root_fail_before_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.project(root)
            sequence.write_bytes(b"changed after receipt")
            with self.assertRaises(ValueError):
                self.build(root, sequence)
            self.assertFalse((root / "plans").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.project(root)
            for override in (
                    {"target_strength": 0.}, {"max_relative_change": 1.1},
                    {"absolute_tolerance": float("inf")}, {"max_iterations": 0},
                    {"coordinate_lower": 1., "coordinate_upper": -1.},
                    {"bounds_policy": "clip"}, {"wall_seconds": 27000}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    self.build(root, sequence, plan_dir=root / ("plans-" + str(len(override))),
                               **override)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.project(root)
            other = root / "other"
            other.mkdir()
            with self.assertRaises(ValueError):
                self.build(other, sequence, plan_dir=other / "plans")

    def test_plan_directory_is_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.project(root)
            self.build(root, sequence)
            with self.assertRaises(FileExistsError):
                self.build(root, sequence)


if __name__ == "__main__":
    unittest.main()
