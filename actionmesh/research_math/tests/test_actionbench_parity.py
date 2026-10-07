import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ACTIONMESH = Path(__file__).resolve().parents[2]
if str(ACTIONMESH) not in sys.path:
    sys.path.insert(0, str(ACTIONMESH))

import official_actionbench_adapter as official
from research_math.actionbench_parity import exact_comparison, one_case


class OfficialAdapterContractTests(unittest.TestCase):
    def test_manifest_preserves_frozen_denominator_and_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm").mkdir()
            path = root / "manifest.json"
            path.write_text(json.dumps({"cases": [
                {"case_id": "sample-native", "uid": "sample", "case_dir": "arm"}
            ]}))
            cases, denominator = official.manifest_cases(root, path)
            self.assertEqual(cases[0]["case_dir"], str((root / "arm").resolve()))
            self.assertEqual(denominator["n_declared"], 1)
            self.assertTrue(denominator["frozen"])

    def test_manifest_rejects_case_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text(json.dumps({"cases": [
                {"case_id": "escape", "uid": "sample", "case_dir": "../outside"}
            ]}))
            with self.assertRaisesRegex(ValueError, "nonescaping"):
                official.manifest_cases(root, path)

    def test_manifest_rejects_duplicate_case_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.json"
            path.write_text(json.dumps({"cases": [
                {"case_id": "same", "uid": "a", "case_dir": "a"},
                {"case_id": "same", "uid": "b", "case_dir": "b"},
            ]}))
            with self.assertRaisesRegex(ValueError, "unique"):
                official.manifest_cases(root, path)

    def test_official_csv_requires_one_successful_uid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "uid", "cd_3d", "cd_4d", "cd_motion", "n_frames",
                    "status", "error_message"))
                writer.writeheader()
                writer.writerow({"uid": "sample", "cd_3d": "1.0", "cd_4d": "2.0",
                                 "cd_motion": "3.0", "n_frames": "16",
                                 "status": "success", "error_message": ""})
            row = official.parse_official_csv(path, "sample")
            self.assertEqual(row["n_frames"], 16)
            self.assertEqual(row["cd_motion"], 3.0)

    def test_official_csv_rejects_failed_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.csv"
            path.write_text("uid,cd_3d,cd_4d,cd_motion,n_frames,status,error_message\n"
                            "sample,nan,nan,nan,0,error,failed\n")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                official.parse_official_csv(path, "sample")

    def test_official_cpu_rng_patch_is_exact_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "copy"
            source.mkdir()
            for name in official.OFFICIAL_FILES:
                text = "placeholder\n"
                if name == "sample_mesh.py":
                    text = "with torch.random.fork_rng(devices=[verts.device], enabled=True):\n    pass\n"
                (source / name).write_text(text)
            record = official.prepare_official_source(source, destination)
            patched = (destination / "sample_mesh.py").read_text()
            self.assertIn("if verts.is_cuda else []", patched)
            self.assertFalse(record["compatibility_patch"]["metric_or_draw_change"])
            self.assertNotEqual(record["original_sha256"]["sample_mesh.py"],
                                record["patched_source_sha256"]["sample_mesh.py"])

    def test_unreviewed_cpu_rng_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source", root / "copy"
            source.mkdir()
            for name in official.OFFICIAL_FILES:
                (source / name).write_text("changed upstream\n")
            with self.assertRaisesRegex(RuntimeError, "Unexpected official"):
                official.prepare_official_source(source, destination)


class ParityReadoutTests(unittest.TestCase):
    def test_one_case_extracts_only_frozen_metrics(self):
        output = {"denominator": {"n_declared": 1}, "cases": [{
            "uid": "sample", "status": "success", "cd_3d": 1.0,
            "cd_4d": 2.0, "cd_motion": 3.0,
        }]}
        self.assertEqual(one_case(output, "sample"),
                         {"cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0})

    def test_one_case_rejects_wrong_denominator(self):
        output = {"denominator": {"n_declared": 2}, "cases": [{
            "uid": "sample", "status": "success", "cd_3d": 1.0,
            "cd_4d": 2.0, "cd_motion": 3.0,
        }]}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            one_case(output, "sample")

    def test_one_case_rejects_nonfinite_metric(self):
        output = {"denominator": {"n_declared": 1}, "cases": [{
            "uid": "sample", "status": "success", "cd_3d": float("nan"),
            "cd_4d": 2.0, "cd_motion": 3.0,
        }]}
        with self.assertRaisesRegex(ValueError, "Invalid"):
            one_case(output, "sample")

    def test_exact_comparison_has_zero_tolerance(self):
        metrics = {"cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0}
        result = exact_comparison(metrics, dict(metrics))
        self.assertTrue(result["passed"])
        self.assertEqual(set(result["metric_tolerances"].values()), {0.0})

    def test_exact_comparison_rejects_small_difference(self):
        official_values = {"cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0}
        faithful_values = dict(official_values, cd_motion=3.0 + 1e-12)
        self.assertFalse(exact_comparison(official_values, faithful_values)["passed"])


if __name__ == "__main__":
    unittest.main()
