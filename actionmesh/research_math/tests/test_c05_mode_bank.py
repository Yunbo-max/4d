"""Authored checks for the C05 producer; Web does not execute them."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from research_math.c05_mode_bank import (
    C05ModeBankError,
    derive_inner_stage1_seeds,
    parse_args,
    produce_same_anchor_mode_bank,
    validate_retained_mode_bank,
)


class FakeMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int64)

    def copy(self):
        return FakeMesh(self.vertices.copy(), self.faces.copy())


class FakeBank:
    def __init__(self, items, timesteps):
        self.items = list(items)
        self.timesteps = list(timesteps)

    def get_ordered(self, device=None):
        order = np.argsort(self.timesteps)
        return ([self.items[index] for index in order],
                np.asarray([self.timesteps[index] for index in order], dtype=np.float32))


class FakeInput:
    def __init__(self):
        self.frames = [np.asarray([index], dtype=np.float32) for index in range(16)]
        self.timesteps = np.arange(16, dtype=np.float32)

    @property
    def n_frames(self):
        return len(self.frames)


class _IdentityProcessor:
    def process_images(self, frames):
        return frames


class FakePipeline:
    def __init__(self, *, corrupt_branch=None):
        self.cfg = SimpleNamespace(
            anchor_idx=0,
            model=SimpleNamespace(image_to_3D_denoiser=SimpleNamespace(
                num_inference_steps=None)),
        )
        self.scheduler = SimpleNamespace(num_inference_steps=None)
        self.mesh_process = SimpleNamespace(face_decimation=None, floaters_threshold=None)
        self.cf_guidance = SimpleNamespace(guidance_scales=None)
        self.background_removal = _IdentityProcessor()
        self.image_process = _IdentityProcessor()
        self.device = SimpleNamespace(type="cpu")
        self.loads = []
        self.unloads = []
        self.stage0_seeds = []
        self.stage1_seeds = []
        self.context_ids = []
        self.stage1_bank_ids = []
        self.stage2_mesh_bank_ids = []
        self.corrupt_branch = corrupt_branch

    def _load_background_removal(self): self.loads.append("background")
    def _load_image_to_3d(self): self.loads.append("stage0")
    def _load_image_encoder(self): self.loads.append("encoder")
    def _load_temporal_denoiser(self): self.loads.append("stage1")
    def _load_temporal_vae(self): self.loads.append("stage2")
    def _unload_model(self, name): self.unloads.append(name)

    @staticmethod
    def _anchor():
        return FakeMesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])

    def init_banks_from_anchor(self, model_input, seed):
        self.stage0_seeds.append(seed)
        return (FakeBank([np.asarray([[seed]], dtype=np.float32)], [0.0]),
                FakeBank([self._anchor()], [0.0]))

    def encode_all_frames(self, model_input):
        return np.arange(16, dtype=np.float32)[:, None]

    def generate_3d_latents(self, model_input, *, context, latent_bank, seed):
        # A later branch must never observe a bank mutated by an earlier branch.
        if len(latent_bank.items) != 1 or latent_bank.timesteps != [0.0]:
            raise AssertionError("non-pristine latent bank")
        self.stage1_bank_ids.append(id(latent_bank))
        self.context_ids.append(id(context))
        self.stage1_seeds.append(seed)
        latent_bank.items.append(np.asarray([[seed]], dtype=np.float32))
        latent_bank.timesteps.append(1.0)
        latent_bank.branch_seed = seed
        return latent_bank

    def generate_mesh_animation(self, *, latent_bank, mesh_bank):
        if len(mesh_bank.items) != 1 or mesh_bank.timesteps != [0.0]:
            raise AssertionError("non-pristine mesh bank")
        self.stage2_mesh_bank_ids.append(id(mesh_bank))
        seed = latent_bank.branch_seed
        branch = self.stage1_seeds.index(seed)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        for frame in range(1, 16):
            vertices = self._anchor().vertices.copy()
            vertices[:, 2] = frame * 0.01 + (seed % 997) * 1e-5
            current_faces = faces.copy()
            if self.corrupt_branch == branch and frame == 7:
                current_faces = np.asarray([[0, 2, 1]], dtype=np.int64)
            mesh_bank.items.append(FakeMesh(vertices, current_faces))
            mesh_bank.timesteps.append(float(frame))
        return mesh_bank


def _ref(name):
    return {"path": f"receipts/{name}.json", "sha256": hashlib.sha256(name.encode()).hexdigest()}


def provenance():
    return {
        "generation_uid": "unit-fixture",
        "gpu_uuid": "GPU-fixture",
        "source_ref": _ref("source"),
        "model_ref": _ref("model"),
        "input_ref": _ref("input"),
        "config_ref": _ref("config"),
        "environment_ref": _ref("environment"),
    }


def parameters():
    return {
        "stage_0_steps": 100,
        "stage_1_steps": 30,
        "face_decimation": 40000,
        "floaters_threshold": 0.02,
        "guidance_scales": [7.5],
        "anchor_idx": 0,
    }


def closure():
    return [{"path": "inputs/pinned.bin", "sha256": "7" * 64}]


class C05ModeBankTests(unittest.TestCase):
    def test_producer_has_no_external_coordinate_or_evaluator_input(self):
        parameters_by_name = inspect.signature(
            produce_same_anchor_mode_bank).parameters
        self.assertEqual(set(parameters_by_name), {
            "pipeline", "model_input", "output_dir", "outer_seed", "branch_count",
            "generation_parameters", "provenance", "producer_input_refs",
            "producer_code_refs", "expected_frames",
        })
        source = Path(inspect.getsourcefile(produce_same_anchor_mode_bank)).read_text()
        self.assertNotIn("official_actionbench_adapter", source)
        self.assertNotIn("surfaces.npy", source)

    def test_cli_is_validation_only(self):
        args = parse_args(["validate", "/retained/c05-output"])
        self.assertEqual(args.operation, "validate")
        self.assertEqual(args.output, Path("/retained/c05-output"))

    def test_seed_schedule_is_deterministic_unique_and_preserves_exact_b0(self):
        first = derive_inner_stage1_seeds(42, 5)
        self.assertEqual(first, derive_inner_stage1_seeds(42, 5))
        self.assertEqual(first[0], 42)
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first[1:], derive_inner_stage1_seeds(314, 5)[1:])

    def test_producer_freezes_stage0_context_and_pristine_banks(self):
        pipeline = FakePipeline()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c05-output"
            result = produce_same_anchor_mode_bank(
                pipeline, FakeInput(), output, outer_seed=42, branch_count=4,
                generation_parameters=parameters(), provenance=provenance(),
                producer_input_refs=closure(), producer_code_refs=closure())
            manifest = validate_retained_mode_bank(output)

            self.assertEqual(pipeline.stage0_seeds, [42])
            self.assertEqual(pipeline.stage1_seeds[0], 42)
            self.assertEqual(len(set(pipeline.stage1_bank_ids)), 4)
            self.assertEqual(len(set(pipeline.stage2_mesh_bank_ids)), 4)
            self.assertEqual(len(set(pipeline.context_ids)), 1)
            self.assertEqual(result["stage0_runs"], 1)
            self.assertEqual(result["context_encodes"], 1)
            self.assertEqual(result["full_sequences_retained"], 4)
            self.assertFalse(result["branch0_exact_b0"])
            self.assertEqual(manifest["branches"][0]["role"],
                             "b0_algorithmic_path_unverified")
            self.assertIn("separate receipt", result["branch0_parity_requirement"])
            self.assertEqual(manifest["mode_semantics"],
                             "equal-mass empirical model-output modes")
            self.assertFalse(manifest["posterior_correspondence_claim"])
            self.assertFalse(manifest["uniform_score_implications"]
                             ["temperature_reweighting_changes_weights"])

            with np.load(output / "mode-bank.npz", allow_pickle=False) as bank:
                self.assertEqual(bank["vertices"].shape, (4, 16, 3, 3))
                np.testing.assert_array_equal(bank["equal_mass_scores"],
                                              np.full(4, 0.25))
                # Branch zero follows the same seed formula as ordinary B0.
                expected = 0.01 + (42 % 997) * 1e-5
                self.assertAlmostEqual(float(bank["vertices"][0, 1, 0, 2]), expected)
                np.testing.assert_array_equal(bank["vertices"][:, 0],
                                              np.broadcast_to(FakePipeline._anchor().vertices,
                                                              (4, 3, 3)))

    def test_every_branch_retains_full_identical_topology_query_and_times(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c05-output"
            produce_same_anchor_mode_bank(
                FakePipeline(), FakeInput(), output, outer_seed=314, branch_count=3,
                generation_parameters=parameters(), provenance=provenance(),
                producer_input_refs=closure(), producer_code_refs=closure())
            arrays = []
            for index in range(3):
                with np.load(output / f"sequences/branch-{index:04d}.npz",
                             allow_pickle=False) as archive:
                    arrays.append({key: archive[key].copy() for key in archive.files})
            for current in arrays[1:]:
                for key in ("faces", "frame_indices", "timesteps", "query_vertex_ids"):
                    np.testing.assert_array_equal(current[key], arrays[0][key])
                self.assertEqual(current["vertices"].shape, arrays[0]["vertices"].shape)

    def test_topology_drift_fails_closed_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c05-output"
            with self.assertRaisesRegex(C05ModeBankError, "topology"):
                produce_same_anchor_mode_bank(
                    FakePipeline(corrupt_branch=1), FakeInput(), output,
                    outer_seed=42, branch_count=2,
                    generation_parameters=parameters(), provenance=provenance(),
                    producer_input_refs=closure(), producer_code_refs=closure())
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["scientific_effect_qualification"])
            self.assertFalse(result["dispatch_ready"])

    def test_receipt_validation_detects_changed_or_unlisted_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c05-output"
            produce_same_anchor_mode_bank(
                FakePipeline(), FakeInput(), output, outer_seed=2718, branch_count=2,
                generation_parameters=parameters(), provenance=provenance(),
                producer_input_refs=closure(), producer_code_refs=closure())
            branch = output / "sequences/branch-0001.npz"
            branch.write_bytes(branch.read_bytes() + b"changed")
            with self.assertRaisesRegex(C05ModeBankError, "hash/size"):
                validate_retained_mode_bank(output)

            branch.write_bytes(branch.read_bytes()[:-7])
            extra = output / "unlisted.bin"
            extra.write_bytes(b"not receipt-bound")
            with self.assertRaisesRegex(C05ModeBankError, "unlisted"):
                validate_retained_mode_bank(output)

    def test_output_is_append_only_and_receipts_reject_overclaim(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "c05-output"
            produce_same_anchor_mode_bank(
                FakePipeline(), FakeInput(), output, outer_seed=42, branch_count=2,
                generation_parameters=parameters(), provenance=provenance(),
                producer_input_refs=closure(), producer_code_refs=closure())
            with self.assertRaises(FileExistsError):
                produce_same_anchor_mode_bank(
                    FakePipeline(), FakeInput(), output, outer_seed=42, branch_count=2,
                    generation_parameters=parameters(), provenance=provenance(),
                    producer_input_refs=closure(), producer_code_refs=closure())
            manifest_path = output / "raw-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["scientific_effect_qualification"] = True
            manifest_path.write_text(json.dumps(manifest))
            result_path = output / "result.json"
            result = json.loads(result_path.read_text())
            result["manifest_ref"]["sha256"] = hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(C05ModeBankError, "overclaims"):
                validate_retained_mode_bank(output)

    def test_provenance_and_complete_generation_settings_are_mandatory(self):
        bad = provenance()
        bad["source_ref"] = {"path": "../escape", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "relative"):
                produce_same_anchor_mode_bank(
                    FakePipeline(), FakeInput(), Path(directory) / "bad-provenance",
                    outer_seed=42, branch_count=2,
                    generation_parameters=parameters(), provenance=bad,
                    producer_input_refs=closure(), producer_code_refs=closure())
            incomplete = parameters()
            incomplete.pop("stage_1_steps")
            with self.assertRaisesRegex(ValueError, "complete exact"):
                produce_same_anchor_mode_bank(
                    FakePipeline(), FakeInput(), Path(directory) / "bad-settings",
                    outer_seed=42, branch_count=2,
                    generation_parameters=incomplete, provenance=provenance(),
                    producer_input_refs=closure(), producer_code_refs=closure())


if __name__ == "__main__":
    unittest.main()
