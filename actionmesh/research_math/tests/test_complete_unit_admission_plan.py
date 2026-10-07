import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_complete_unit_admission import (
    canonical_origin_paths,
    reference_closure,
    runner_output_paths,
)


class CompleteUnitAdmissionPlanTests(unittest.TestCase):
    def test_origin_closure_includes_harness_native_and_attempt_records(self):
        root = Path("/project")
        harness = {
            "output_root": "runs/harness", "batch_id": "complete-lowram-r7",
            "tasks": [{"task_id": "complete-fp16-lowram-v1-three-arm-unit"}],
        }
        native = {"output_root": "runs/attempts", "run_id": "complete-lowram-r7"}
        receipt = {"attempts": [{
            "attempt_id": "attempt-1",
            "attempt_path": "runs/attempts/complete-lowram-r7/attempt-1",
        }]}

        paths = set(canonical_origin_paths(root, harness, native, receipt))

        self.assertIn(root / "runs/harness/complete-lowram-r7/report.json", paths)
        self.assertIn(root / "runs/harness/complete-lowram-r7/state.json", paths)
        self.assertIn(root / "runs/attempts/complete-lowram-r7/receipt.json", paths)
        self.assertIn(root / "runs/attempts/complete-lowram-r7/attempt-1/attempt.json", paths)
        self.assertIn(
            root / "runs/harness/complete-lowram-r7/tasks/"
            "complete-fp16-lowram-v1-three-arm-unit/result.json", paths)
        self.assertIn(
            root / "runs/harness/complete-lowram-r7/tasks/"
            "complete-fp16-lowram-v1-three-arm-unit/execution-context.json", paths)

    def test_origin_closure_rejects_more_than_one_attempt(self):
        with self.assertRaises(ValueError):
            canonical_origin_paths(
                Path("/project"),
                {"tasks": [{"task_id": "task"}]},
                {},
                {"attempts": [{"attempt_id": "a"}, {"attempt_id": "b"}]},
            )

    def test_reference_closure_skips_external_diagnostic_refs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root.parent / "external-ground-truth.npy"
            external.write_bytes(b"ground truth")
            try:
                source = root / "official-scores.json"
                source.write_text(json.dumps({
                    "cases": [{
                        "inputs": {
                            "ground_truth": {
                                "path": str(external),
                                "sha256": "not-used-for-admission-staging",
                            }
                        }
                    }]
                }))
                refs = reference_closure(root, [source])
                self.assertEqual([row["path"] for row in refs],
                                 ["official-scores.json"])
                source.write_text(json.dumps({
                    "unexpected": {"path": str(external), "sha256": "unknown"}
                }))
                with self.assertRaises(ValueError):
                    reference_closure(root, [source])
            finally:
                external.unlink(missing_ok=True)

    def test_runner_output_paths_stages_nested_files_not_in_receipt_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt_path = Path("runs/attempts/unit/attempt-1")
            output = root / attempt_path / "workspace/actionmesh/unit-output"
            cache = output / "official/extra.pyc"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"incidental cache")
            result = output / "result.json"
            result.write_text(json.dumps({
                "outputs": [{"path": "official/extra.pyc", "bytes": cache.stat().st_size,
                             "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}]
            }))
            receipt = {"attempts": [{"attempt_path": attempt_path.as_posix()}]}
            self.assertEqual(runner_output_paths(root, receipt), [cache])

            cache.write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                runner_output_paths(root, receipt)
