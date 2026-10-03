"""Regression checks for real-input contracts and paired output accounting."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_ten.geometry_run import run_geometry


class GeometryRunTests(unittest.TestCase):
    def fixture(self, root, method=5, extra=None):
        rest = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        center = rest.mean(0)
        tracks = np.stack([(rest-center) * (1. + .01*t) + center + [t*.1, 0., 0.] for t in range(16)])
        arrays = dict(rest=rest, trajectories=tracks,
            faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]), times=np.arange(16.))
        if method == 5:
            arrays["stable_mask"] = np.ones(4, bool)
        elif method == 8:
            arrays.update(contact_frame=np.array([1]), contact_indices=np.array([[0, 1, 2, 3]]),
                contact_coefficients=np.array([[1., -1/3, -1/3, -1/3]]),
                contact_normals=np.array([[0., 0., -1.]]), contact_gap=np.array([0.]))
        elif method == 1:
            projection = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
            arrays.update(observed_uv=tracks[..., :2], confidence=np.ones((16, 4)),
                projection=np.broadcast_to(projection, (16, 3, 4)))
        arrays.update(extra or {})
        source = root/"input.npz"
        np.savez(source, **arrays)
        evidence = root/"evidence.json"
        row = dict(schema_version=1, uid="fixture", method=method,
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            input_evidence="constructed_control", measurement_provenance="analytical test fixture",
            qualification={"stable_region_verified": True, "signed_contacts_verified": True,
                "observed_tracks_and_camera_verified": True}, parameters={})
        evidence.write_text(json.dumps(row))
        return source, evidence, arrays

    def test_scale_pair_retains_pose_and_can_feed_existing_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, arrays = self.fixture(root)
            before = source.read_bytes()
            output = root/"out"
            self.assertEqual(run_geometry(5, source, evidence, output), 0)
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(len(manifest["cases"]), 4)
            with np.load(output/"variants/stable_region/sequence.npz") as data:
                expected = np.stack([arrays["rest"] + [t*.1, 0., 0.] for t in range(16)])
                np.testing.assert_allclose(data["vertices"], expected, atol=1e-12)
                np.testing.assert_array_equal(data["vertices"][0], arrays["trajectories"][0])
                np.testing.assert_array_equal(data["frame_indices"], np.arange(16))
            report = json.loads((output/"report.json").read_text())
            self.assertFalse(report["natural_quality_claim"])
            self.assertEqual(report["model_calls"], 0)
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                run_geometry(5, source, evidence, output)
            self.assertEqual(source.read_bytes(), before)

    def test_unqualified_stable_mask_and_hash_change_fail_before_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, _ = self.fixture(root)
            row = json.loads(evidence.read_text())
            row["qualification"]["stable_region_verified"] = False
            evidence.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                run_geometry(5, source, evidence, root/"out")
            self.assertFalse((root/"out").exists())
            row["qualification"]["stable_region_verified"] = True
            row["input_sha256"] = "0"*64
            evidence.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                run_geometry(5, source, evidence, root/"out")
            self.assertFalse((root/"out").exists())

    def test_privileged_extra_array_is_not_silently_consumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, _ = self.fixture(root, extra={"gt_vertices": np.zeros((16, 4, 3))})
            with self.assertRaises(ValueError):
                run_geometry(5, source, evidence, root/"out")
            self.assertFalse((root/"out").exists())

    def test_fractional_contact_frame_is_not_truncated_to_an_integer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, _ = self.fixture(root, method=8, extra={"contact_frame": np.array([1.5])})
            with self.assertRaises(ValueError):
                run_geometry(8, source, evidence, root/"out")
            self.assertFalse((root/"out").exists())

    def test_contact_and_elasticity_produce_separate_baselines(self):
        for method, candidate, count in [(8, "bounded_normal", 3), (1, "observed_elasticity", 8)]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, evidence, _ = self.fixture(root, method=method)
                # Convergence is retained in every variant, including failures.
                run_geometry(method, source, evidence, root/"out")
                manifest = json.loads((root/"out/manifest.json").read_text())
                self.assertEqual(len(manifest["cases"]), count)
                self.assertTrue((root/"out/variants"/candidate/"sequence.npz").is_file())
                for case in manifest["cases"]:
                    row = json.loads((root/"out"/case["case_dir"]/"report.json").read_text())
                    self.assertIn(row["status"], ("completed", "failed"))
                    self.assertEqual(row["model_calls"], 0)

    def test_infeasible_contact_is_counted_and_debug_sequence_is_not_a_success(self):
        from research_census_eval import discover_cases
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, _ = self.fixture(root, method=8, extra={"contact_gap": np.array([1.])})
            output = root/"out"
            self.assertEqual(run_geometry(8, source, evidence, output), 1)
            row = json.loads((output/"variants/bounded_normal/report.json").read_text())
            self.assertEqual(row["status"], "failed")
            self.assertGreater(max(row["diagnostics"]["final_violation"]), .1)
            self.assertTrue((output/"variants/bounded_normal/sequence.npz").is_file())
            cases, denominator = discover_cases(output, output/"manifest.json")
            self.assertEqual(len(cases), 3)
            self.assertEqual(denominator["n_declared"], 3)

    def test_source_change_invalidates_every_completed_arm_for_evaluation(self):
        from research_ten import geometry_run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, _ = self.fixture(root)
            save = geometry_run.save_sequence
            def changed(*args, **kwargs):
                result = save(*args, **kwargs)
                with source.open("ab") as stream:
                    stream.write(b"external source change")
                return result
            with patch.object(geometry_run, "save_sequence", side_effect=changed):
                self.assertEqual(run_geometry(5, source, evidence, root/"out"), 1)
            manifest = json.loads((root/"out/manifest.json").read_text())
            for case in manifest["cases"]:
                row = json.loads((root/"out"/case["case_dir"]/"report.json").read_text())
                self.assertEqual(row["status"], "failed")

    def test_source_deletion_invalidates_outputs_and_preserves_report(self):
        from research_ten import geometry_run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, evidence, _ = self.fixture(root)
            save = geometry_run.save_sequence
            def removed(*args, **kwargs):
                result = save(*args, **kwargs)
                source.unlink(missing_ok=True)
                return result
            with patch.object(geometry_run, "save_sequence", side_effect=removed):
                self.assertEqual(run_geometry(5, source, evidence, root/"out"), 1)
            report = json.loads((root/"out/report.json").read_text())
            self.assertFalse(report["inputs_unchanged"])
            for row in report["arms"].values():
                self.assertEqual(row["status"], "failed")


if __name__ == "__main__":
    unittest.main()
