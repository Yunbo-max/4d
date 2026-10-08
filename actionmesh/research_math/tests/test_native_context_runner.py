"""Unexecuted runner contracts; no native model, benchmark, scorer or GPU."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class NativeContextRunnerTests(unittest.TestCase):
    def generation(self, repair=False):
        from research_math.complete_unit_contract import REPAIR_PARENT_RECEIPT
        record = {
            "runtime_profile": "fp16-lowram-v1" if repair else "default",
            "seed": 42, "fast": False, "low_ram": repair,
            "dtype": "float16" if repair else "bfloat16",
            "config": "actionmesh/configs/actionmesh_lowram.yaml" if repair
                      else "actionmesh/configs/actionmesh.yaml",
            "effective_parameters": {
                "stage_0_steps": 100, "stage_1_steps": 30,
                "face_decimation": 40000, "floaters_threshold": .02,
                "guidance_scales": [7.5], "anchor_idx": 0,
                "temporal_context_size": 16, "sliding_window_denoiser": 15,
                "subsampling_level": 1, "sliding_window_autoencoder": 15,
            },
        }
        if repair:
            record["repair_parent_receipt_sha256"] = REPAIR_PARENT_RECEIPT
        return record

    def test_official_profile_mapping_has_no_silent_parameter_change(self):
        from research_math.native_context_runner import generation_settings
        for repair in (False, True):
            with self.subTest(repair=repair):
                pipeline, run = generation_settings(self.generation(repair),
                    Path("/source"), Path("/frames"), Path("/output"))
                self.assertEqual(pipeline, {
                    "config_name": "actionmesh_lowram.yaml" if repair else "actionmesh.yaml",
                    "config_dir": "/source/actionmesh/configs",
                    "dtype": "float16" if repair else "bfloat16",
                    "lazy_loading": repair,
                })
                self.assertEqual(run, {
                    "input": "/frames", "output_dir": "/output", "seed": 42,
                    "blender_path": None, "stage_0_steps": 100, "stage_1_steps": 30,
                    "face_decimation": 40000, "floaters_threshold": .02,
                    "guidance_scales": [7.5], "anchor_idx": 0,
                })

    def test_profile_or_effective_parameter_drift_is_rejected(self):
        from research_math.native_context_runner import generation_settings
        for key, value in (("stage_1_steps", 15), ("subsampling_level", 2),
                           ("guidance_scales", [8.]), ("anchor_idx", 1)):
            generation = self.generation()
            generation["effective_parameters"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                generation_settings(generation, Path("/s"), Path("/f"), Path("/o"))
        generation = self.generation()
        generation["seed"] = 44
        with self.assertRaises(ValueError):
            generation_settings(generation, Path("/s"), Path("/f"), Path("/o"))

    def test_cli_keeps_pricing_budget_distinct_from_paired_instrument_budget(self):
        from research_math.native_context_runner import parse_args, PREREQUISITE_PATHS
        args = []
        for name in PREREQUISITE_PATHS:
            args += ["--" + name.replace("_", "-"), "/inputs/" + name]
        args += ["--output", "/output", "--gpu-uuid", "GPU-fixture", "--wall-seconds", "1664",
                 "--instrument-wall-seconds", "5000", "--atol", "0", "--rtol", "0",
                 "--root", "/project", "--pricing", "/project/inputs/actionbench-full128-queue/pricing.json",
                 "--uid", "fixture-uid", "--window-id", "window-02",
                 "--historical-root", "/history", "--historical-manifest", "/history/archive.json"]
        parsed = parse_args(args)
        self.assertEqual(parsed.wall_seconds, 1664)
        self.assertEqual(parsed.instrument_wall_seconds, 5000)
        self.assertEqual(parsed.historical_root, Path("/history"))
        self.assertEqual(parsed.atol, 0.)
        self.assertEqual(parsed.rtol, 0.)
        self.assertFalse(parsed.source_time_query)

    def write_sequence(self, path, *, offset=0., changed_faces=False):
        vertices = np.zeros((16, 3, 3), dtype=np.float32)
        vertices[:, 1, 0] = .4
        vertices[:, 2, 1] = .4
        vertices[5, 1, 2] = offset
        faces = np.array([[0, 2, 1] if changed_faces else [0, 1, 2]], dtype=np.int32)
        np.savez_compressed(path, vertices=vertices, faces=faces,
                            frame_indices=np.arange(16, dtype=np.int64),
                            timesteps=np.arange(16, dtype=np.float32),
                            query_vertex_ids=np.arange(3, dtype=np.int64))

    def test_full_sequence_comparison_retains_coordinate_and_topology_mismatch(self):
        from research_math.native_context_runner import compare_sequences
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_sequence(root / "left.npz")
            self.write_sequence(root / "right.npz", offset=.125, changed_faces=True)
            result = compare_sequences(root / "left.npz", root / "right.npz", atol=0., rtol=0.)
            self.assertFalse(result["matches"])
            self.assertFalse(result["topology_matches"])
            self.assertFalse(result["coordinates_match"])
            self.assertEqual(result["max_abs_error"], .125)
            self.assertEqual(result["frames"], 16)

    def test_comparison_honors_explicit_tolerance_without_hiding_topology(self):
        from research_math.native_context_runner import compare_sequences
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_sequence(root / "left.npz")
            self.write_sequence(root / "right.npz", offset=.125)
            self.assertTrue(compare_sequences(root / "left.npz", root / "right.npz",
                                             atol=.125, rtol=0.)["matches"])
            self.write_sequence(root / "right.npz", changed_faces=True)
            self.assertFalse(compare_sequences(root / "left.npz", root / "right.npz",
                                              atol=1., rtol=1.)["matches"])

    def capture_fixture(self, root):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        from research_math.tests.test_pipeline_decoder_observer import FakePipeline, PipelineDecoderObserverTests
        fixture = PipelineDecoderObserverTests()
        pipeline = FakePipeline()
        with torch.inference_mode(), PipelineDecoderObserver(pipeline, root, fixture.identity()):
            pipeline._decode_displacement(**fixture.inputs())

    def test_capture_index_is_relocatable_but_rejects_escaping_or_changed_reference(self):
        from research_math.native_context_runner import capture_windows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            self.capture_fixture(capture)
            moved = root / "relocated"
            capture.rename(moved)
            self.assertEqual(capture_windows(moved), [moved / "window-0000"])
            index_path = moved / "index.json"
            original = json.loads(index_path.read_text())
            for changed in ("../outside", "/absolute", "window-0001"):
                index = json.loads(json.dumps(original))
                index["windows"][0]["path"] = changed
                index_path.write_text(json.dumps(index))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    capture_windows(moved)
            index_path.write_text(json.dumps(original))
            record = moved / "window-0000/record.json"
            record.write_text(record.read_text() + " ")
            with self.assertRaises(ValueError):
                capture_windows(moved)

    def test_missing_or_failed_replay_never_disappears_from_aggregate(self):
        from research_math.native_context_runner import aggregate_comparisons
        good = {"status": "replayed_unqualified", "raw_matches": True, "mesh_matches": True}
        bad = {"status": "replayed_unqualified", "raw_matches": False, "mesh_matches": True}
        for pair, rows in (({"matches": False}, [good]), ({"matches": True}, [good, bad]),
                           ({"matches": True}, []), ({"matches": True}, [good, {"status": "replay_failed"}])):
            with self.subTest(pair=pair, rows=rows):
                result = aggregate_comparisons(pair, rows)
                self.assertFalse(result["all_comparisons_match"])
                self.assertEqual(result["replay_windows"], len(rows))
                self.assertIs(result["native_context_qualified"], False)
                self.assertIs(result["scientific_effect_qualification"], False)

    def test_matching_comparison_is_still_unqualified(self):
        from research_math.native_context_runner import aggregate_comparisons
        result = aggregate_comparisons({"matches": True}, [
            {"status": "replayed_unqualified", "raw_matches": True, "mesh_matches": True}])
        self.assertTrue(result["all_comparisons_match"])
        self.assertIs(result["native_context_qualified"], False)
        self.assertIs(result["replay_qualified"], False)

    def test_replay_stage_status_retains_comparison_mismatch(self):
        from research_math.native_context_runner import stage_completion_status
        self.assertEqual(stage_completion_status('replay', {
            'all_comparisons_match': False}), 'comparison_mismatch')
        self.assertEqual(stage_completion_status('replay', {
            'all_comparisons_match': True}), 'completed_unqualified')
        self.assertEqual(stage_completion_status('observed', {}), 'completed_unqualified')

    def test_source_identity_failure_retains_stage_report_before_model_loading(self):
        from research_math.native_context_runner import _stage
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "kind": "native-context-instrument-stage", "version": 1,
                "stage": "unobserved", "gpu_uuid": "GPU-fixture",
                "output": str(root / "stage-output"),
                "identity": {"instrument_code_sha256": {
                    "research_math/native_context_runner.py": "0" * 64}},
            }
            path = root / "request.json"
            path.write_text(json.dumps(request))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            # Only the externally supplied environment is simulated. The real
            # stage, hash validation and failure persistence execute; the
            # deliberate code mismatch stops before importing/loading a model.
            with patch.dict(os.environ, {
                    "CUDA_VISIBLE_DEVICES": "GPU-fixture", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}):
                with self.assertRaisesRegex(ValueError, "Instrument source changed"):
                    _stage(path, expected)
            report = json.loads((root / "stage-output/stage-result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("Instrument source changed", report["error"])
            self.assertIs(report["native_context_qualified"], False)


if __name__ == "__main__":
    unittest.main()
