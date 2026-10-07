"""Engineering tests for ActionBench dataset semantic admission.

The generated arrays and PNGs are parser fixtures, not benchmark samples, and
these tests produce no prediction or scientific score.
"""
from __future__ import annotations

import binascii
import hashlib
import importlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from research_math import actionbench_dataset_semantics as semantics

try:
    planner = importlib.import_module("prepare_actionbench_dataset_semantics")
    run_experiments = importlib.import_module("run_experiments")
except ModuleNotFoundError:
    planner = None
    run_experiments = None


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF))


def rgba_png(width=2, height=2, colour=6):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0)
    channels = 4 if colour == 6 else 3
    rows = b"".join(b"\x00" + b"\x00" * (width * channels)
                    for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) +
            png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b""))


def camera():
    return {
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "T": [0.0, 0.0, -3.0],
        "focal_length_ndc": [2.0, 2.0],
        "principal_point_ndc": [0.0, 0.0],
    }


def contract(population_size=1):
    uids = ["uid-a"] if population_size == 1 else [f"uid-{i}" for i in range(population_size)]
    return {
        "kind": "actionbench-full128-dataset-semantics-contract",
        "version": "1.0.0",
        "dataset_revision": "a" * 40,
        "population_size": population_size,
        "frames_per_sample": 16,
        "uid_set_sha256": hashlib.sha256(semantics.canonical(uids)).hexdigest(),
        "surfaces": {
            "shape": [16, 100000, 6],
            "allowed_float_item_bytes": [4, 8],
            "position_bounds": [-1.0, 1.0],
            "position_bound_tolerance": 1e-6,
        },
        "camera": {
            "required_keys": ["R", "T", "focal_length_ndc",
                              "principal_point_ndc"],
            "rotation_tolerance": 1e-5,
        },
        "parser_resource_limits": {
            "camera_max_bytes": 65536,
            "surface_max_bytes": 83886080,
            "frame_max_bytes": 67108864,
        },
    }, {"dataset": "facebook/actionbench", "revision": "a" * 40,
        "uids": uids}


class SemanticHelperTests(unittest.TestCase):
    def test_complete_rgba_png_is_accepted(self):
        parsed = semantics._png_ihdr(rgba_png(), "frame.png")
        self.assertEqual(parsed["colour_type"], 6)
        self.assertEqual((parsed["width"], parsed["height"]), (2, 2))

    def test_rgb_png_without_alpha_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RGBA"):
            semantics._png_ihdr(rgba_png(colour=2), "frame.png")

    def test_non_square_png_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "square"):
            semantics._png_ihdr(rgba_png(width=2, height=1), "frame.png")

    def test_truncated_png_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete|Truncated"):
            semantics._png_ihdr(rgba_png()[:-8], "frame.png")

    def test_png_crc_mismatch_is_rejected(self):
        payload = bytearray(rgba_png())
        payload[29] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            semantics._png_ihdr(bytes(payload), "frame.png")

    def test_valid_camera_is_accepted(self):
        frozen, _ = contract()
        parsed = semantics._camera(json.dumps(camera()).encode(), frozen, "camera")
        self.assertAlmostEqual(parsed["rotation_determinant"], 1.0)

    def test_required_camera_keys_are_required(self):
        frozen, _ = contract()
        value = camera()
        del value["R"]
        with self.assertRaisesRegex(ValueError, "keys missing"):
            semantics._camera(json.dumps(value).encode(), frozen, "camera")

    def test_improper_camera_rotation_is_rejected(self):
        frozen, _ = contract()
        value = camera()
        value["R"][2][2] = -1.0
        with self.assertRaisesRegex(ValueError, "proper orthonormal"):
            semantics._camera(json.dumps(value).encode(), frozen, "camera")

    def test_nonpositive_focal_length_is_rejected(self):
        frozen, _ = contract()
        value = camera()
        value["focal_length_ndc"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "positive"):
            semantics._camera(json.dumps(value).encode(), frozen, "camera")

    def test_small_surface_helper_checks_shape_dtype_finite_and_bounds(self):
        frozen, _ = contract()
        frozen["surfaces"]["shape"] = [2, 3, 6]
        value = np.zeros((2, 3, 6), dtype=np.float32)
        buffer = tempfile.SpooledTemporaryFile()
        np.save(buffer, value)
        buffer.seek(0)
        parsed = semantics._surfaces(buffer.read(), frozen, "surface")
        self.assertEqual(parsed["shape"], [2, 3, 6])

    def test_surface_nonfinite_is_rejected(self):
        frozen, _ = contract()
        frozen["surfaces"]["shape"] = [1, 1, 6]
        value = np.zeros((1, 1, 6), dtype=np.float32)
        value[0, 0, 0] = np.nan
        buffer = tempfile.SpooledTemporaryFile()
        np.save(buffer, value)
        buffer.seek(0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            semantics._surfaces(buffer.read(), frozen, "surface")

    def test_surface_position_bound_is_rejected(self):
        frozen, _ = contract()
        frozen["surfaces"]["shape"] = [1, 1, 6]
        value = np.zeros((1, 1, 6), dtype=np.float32)
        value[0, 0, 0] = 1.1
        buffer = tempfile.SpooledTemporaryFile()
        np.save(buffer, value)
        buffer.seek(0)
        with self.assertRaisesRegex(ValueError, "normalized-space"):
            semantics._surfaces(buffer.read(), frozen, "surface")

    def test_integer_surface_tensor_is_rejected(self):
        frozen, _ = contract()
        frozen["surfaces"]["shape"] = [1, 1, 6]
        buffer = tempfile.SpooledTemporaryFile()
        np.save(buffer, np.zeros((1, 1, 6), dtype=np.int32))
        buffer.seek(0)
        with self.assertRaisesRegex(ValueError, "dtype"):
            semantics._surfaces(buffer.read(), frozen, "surface")


class SemanticAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frozen, self.population = contract()
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.files = []
        sample = self.dataset / "data/uid-a"
        (sample / "imgs").mkdir(parents=True)
        (sample / "camera.json").write_text(json.dumps(camera()))
        surface = np.zeros((16, 100000, 6), dtype=np.float32)
        np.save(sample / "surfaces.npy", surface)
        for frame in range(16):
            (sample / f"imgs/{frame:02d}.png").write_bytes(rgba_png())
        for path in sorted((self.dataset / "data").rglob("*")):
            if path.is_file():
                relative = path.relative_to(self.dataset).as_posix()
                payload = path.read_bytes()
                self.files.append({"path": relative, "bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest(),
                                   "etag": "fixture"})
        self.admission = {
            "kind": "actionbench-full128-snapshot-admission",
            "version": "1.0.0",
            "status": "admitted_engineering_snapshot",
            "all_revisions_immutable": True,
            "all_content_files_hashed": True,
            "scientific_effect_qualification": False,
            "dispatch_ready": False,
            "snapshots": {"dataset": {
                "repository": "facebook/actionbench",
                "revision": "a" * 40,
                "file_count": len(self.files),
                "manifest_sha256": hashlib.sha256(
                    semantics.canonical(self.files)).hexdigest(),
                "files": self.files,
            }},
            "dataset": {
                "dataset": "facebook/actionbench",
                "revision": "a" * 40,
                "population_size": 1,
                "frames_per_sample": 16,
                "uid_manifest_sha256": self.frozen["uid_set_sha256"],
                "required_files_per_sample": 18,
            },
        }

    def test_complete_sample_emits_semantic_digest_without_qualification(self):
        result = semantics.admit(
            self.frozen, self.population, self.admission, self.dataset,
            self.root / "out.json")
        self.assertEqual(result["status"], "admitted_engineering_dataset_semantics")
        self.assertEqual(result["population_size"], 1)
        self.assertFalse(result["scientific_effect_qualification"])
        self.assertFalse(result["dispatch_ready"])
        self.assertEqual(len(result["sample_semantics_sha256"]), 64)

    def test_post_admission_surface_mutation_is_rejected(self):
        path = self.dataset / "data/uid-a/surfaces.npy"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "differs from snapshot admission"):
            semantics.admit(self.frozen, self.population, self.admission,
                            self.dataset, self.root / "out.json")

    def test_existing_output_is_preserved(self):
        output = self.root / "out.json"
        output.write_text("preserve")
        with self.assertRaises(FileExistsError):
            semantics.admit(self.frozen, self.population, self.admission,
                            self.dataset, output)
        self.assertEqual(output.read_text(), "preserve")


class SemanticPlanTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, "Dataset semantics plan builder missing")
        self.assertIsNotNone(run_experiments, "Installed harness modules missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        code = self.root / "actionmesh/research_math"
        code.mkdir(parents=True)
        (code / "__init__.py").write_text("")
        (code / "actionbench_dataset_semantics.py").write_text("fixture\n")
        (self.root / "actionmesh/prepare_actionbench_dataset_semantics.py").write_text(
            "fixture\n")
        self.frozen, self.population = contract()
        population_path = self.root / "population.json"
        population_path.write_text(json.dumps(self.population))
        self.frozen["population_ref"] = planner.file_ref(self.root, population_path)
        contract_path = self.root / "contract.json"
        contract_path.write_text(json.dumps(self.frozen))
        admission_path = self.root / "admission.json"
        admission_path.write_text(json.dumps({
            "kind": "actionbench-full128-snapshot-admission",
            "status": "admitted_engineering_snapshot",
            "snapshots": {"dataset": {"revision": "a" * 40}}}))
        dataset = self.root / "dataset"
        dataset.mkdir()
        self.paths = (contract_path, population_path, admission_path, dataset)
        self.skill_dir = Path(run_experiments.__file__).resolve().parents[1]

    def build(self):
        completed = SimpleNamespace(stdout="f" * 40 + "\n")
        with patch.object(planner.subprocess, "run", return_value=completed):
            return planner.build_plan(
                self.root, skill_dir=self.skill_dir,
                contract_path=self.paths[0], population_path=self.paths[1],
                snapshot_admission_path=self.paths[2], dataset_root=self.paths[3],
                plan_dir=self.root / "plans", run_id="fixture-semantics",
                wall_seconds=300, ram_mib=512, cpu_cores=1)

    def test_plan_is_cpu_only_single_attempt_and_binds_admission(self):
        outer = self.build()
        native = json.loads((self.root / "plans/native.json").read_text())
        self.assertEqual(native["limits"]["max_attempts"], 1)
        self.assertEqual(outer["tasks"][0]["resources"]["gpu_count"], 0)
        self.assertEqual(outer["limits"]["max_gpu_task_seconds"], 0)
        self.assertIn("--snapshot-admission", native["jobs"][0]["command"])

    def test_contract_population_ref_mismatch_is_rejected(self):
        value = json.loads(self.paths[0].read_text())
        value["population_ref"]["sha256"] = "0" * 64
        self.paths[0].write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "exact population"):
            self.build()
