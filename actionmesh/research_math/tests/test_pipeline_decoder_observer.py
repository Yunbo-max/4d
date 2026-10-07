"""Unexecuted engineering contracts; these fixtures are not native evidence.

The real pipeline observer and decoder hooks are exercised against a small
in-memory decoder. No model weights, native assets, scorer, or GPU is loaded.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from safetensors.torch import load_file


class Decoder(torch.nn.Module):
    """Tensor-signature fixture with observable full-context dependence."""

    prediction_mode = "direct"

    def __init__(self):
        super().__init__()
        self.register_buffer("marker", torch.zeros(()))
        self.calls = []
        self.bias = 0.0
        self.fail = False

    @property
    def device(self):
        return self.marker.device

    def forward(self, latent, framestep, source_alpha, target_alphas, query,
                step_callback=None):
        values = dict(latent=latent, framestep=framestep,
                      source_alpha=source_alpha, target_alphas=target_alphas,
                      query=query)
        self.calls.append({
            "objects": values,
            "values": {key: value.detach().clone() for key, value in values.items()},
            "inference_mode": torch.is_inference_mode_enabled(),
            "grad_enabled": torch.is_grad_enabled(),
            "cpu_autocast_enabled": torch.is_autocast_cpu_enabled(),
            "cpu_autocast_dtype": str(torch.get_autocast_cpu_dtype()),
        })
        if self.fail:
            raise RuntimeError("fixture native forward failed")
        if step_callback is not None:
            step_callback(1, target_alphas.shape[1])
        context = latent.float().sum() * 0.0001 + framestep.sum() * 0.00001
        return (query[:, None, :, :3] * 0.2
                + target_alphas[:, :, None, None] * 0.05
                - source_alpha[:, None, None, None] * 0.01
                + context + self.bias)

    def apply_displacement(self, vertex, displacement, scale=1.0):
        # Native direct mode ignores vertex; preserve this signature/semantics.
        return displacement.clamp(min=-scale, max=scale)


class FakePipeline:
    """Only the native boundary is replaced; observer implementation is real."""

    def __init__(self):
        self.temporal_3D_vae = Decoder().eval()
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(anchor_idx=0, subsampling_level=1)
        self.calls = []
        self.last_output = None

    def _decode_displacement(self, latents, window_timesteps, source_alpha,
                             target_alphas, anchor_mesh, step_callback=None):
        self.calls.append((latents, window_timesteps, source_alpha,
                           target_alphas, anchor_mesh, step_callback))
        query = torch.as_tensor(np.concatenate(
            [anchor_mesh.vertices, anchor_mesh.vertex_normals], axis=-1),
            dtype=torch.float32, device=self.device)[None]
        raw = self.temporal_3D_vae(
            latent=latents, framestep=window_timesteps,
            source_alpha=source_alpha, target_alphas=target_alphas,
            query=query, step_callback=step_callback)
        vertices = self.temporal_3D_vae.apply_displacement(
            vertex=query[:3], displacement=raw).detach().cpu().numpy()
        self.last_output = [SimpleNamespace(
            vertices=vertices[0, index].astype(np.float64),
            faces=anchor_mesh.faces.copy())
            for index in range(target_alphas.shape[1])]
        return self.last_output


class PipelineDecoderObserverTests(unittest.TestCase):
    def identity(self):
        return {"scope": "engineering-fixture-only",
                "source_sha256": "1" * 64, "weights_sha256": "2" * 64,
                "config_sha256": "3" * 64, "generation": {"seed": 42}}

    def inputs(self):
        mesh = SimpleNamespace(
            vertices=np.array([[0., 0., 0.], [.4, 0., 0.], [0., .4, 0.]],
                              dtype=np.float64),
            vertex_normals=np.array([[0., 0., 1.]] * 3, dtype=np.float64),
            faces=np.array([[0, 1, 2]], dtype=np.int64))
        return dict(latents=torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
                    window_timesteps=torch.tensor([[10., 12., 16.]]),
                    source_alpha=torch.tensor([0.]),
                    target_alphas=torch.tensor([[1. / 3., 1.]]),
                    anchor_mesh=mesh)

    def capture(self, root, *, pipeline=None, inputs=None):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        pipeline = pipeline if pipeline is not None else FakePipeline()
        inputs = inputs if inputs is not None else self.inputs()
        with torch.inference_mode(), PipelineDecoderObserver(
                pipeline, root, self.identity()):
            output = pipeline._decode_displacement(**inputs)
        return pipeline, inputs, output

    def assert_restored(self, pipeline):
        self.assertNotIn("_decode_displacement", pipeline.__dict__)
        self.assertIs(pipeline._decode_displacement.__func__,
                      FakePipeline._decode_displacement)
        self.assertEqual(len(pipeline.temporal_3D_vae._forward_pre_hooks), 0)
        self.assertEqual(len(pipeline.temporal_3D_vae._forward_hooks), 0)

    def test_capture_preserves_call_objects_output_identity_and_rng(self):
        from research_math.pipeline_decoder_observer import load_window
        with tempfile.TemporaryDirectory() as directory:
            pipeline = FakePipeline()
            inputs = self.inputs()
            before = {key: value.clone() for key, value in inputs.items()
                      if isinstance(value, torch.Tensor)}
            rng = torch.get_rng_state().clone()
            root = Path(directory) / "capture"
            _, _, output = self.capture(root, pipeline=pipeline, inputs=inputs)
            self.assertIs(output, pipeline.last_output)
            self.assertEqual(len(pipeline.calls), 1)
            self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)
            for index, key in enumerate(("latents", "window_timesteps",
                                         "source_alpha", "target_alphas", "anchor_mesh")):
                self.assertIs(pipeline.calls[0][index], inputs[key])
            for key, value in before.items():
                self.assertTrue(torch.equal(value, inputs[key]))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assert_restored(pipeline)
            tensors, record = load_window(root / "window-0000")
            self.assertEqual(record["status"], "captured_unqualified")
            self.assertEqual(set(tensors), {"anchor_vertices", "faces", "vertices",
                                           "target_timesteps", "source_timesteps"})
            np.testing.assert_array_equal(tensors["anchor_vertices"].numpy(),
                                          inputs["anchor_mesh"].vertices)
            np.testing.assert_array_equal(tensors["faces"].numpy(), [[0, 1, 2]])
            np.testing.assert_allclose(tensors["target_timesteps"].numpy().reshape(-1),
                                       [12., 16.], rtol=0., atol=1e-6)
            np.testing.assert_array_equal(tensors["source_timesteps"].numpy().reshape(-1),
                                          [10.])
            np.testing.assert_array_equal(tensors["vertices"].numpy().reshape(2, 3, 3),
                                          np.stack([mesh.vertices for mesh in output]))

    def test_original_instance_override_is_restored_exactly(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        pipeline = FakePipeline()
        original = pipeline._decode_displacement
        pipeline._decode_displacement = original
        with tempfile.TemporaryDirectory() as directory:
            with PipelineDecoderObserver(pipeline, Path(directory) / "capture",
                                         self.identity()):
                pipeline._decode_displacement(**self.inputs())
        self.assertIs(pipeline.__dict__["_decode_displacement"], original)

    def test_legacy_capture_marker_rejects_attachment_without_side_effects(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        for marker in (object(), None):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as directory:
                pipeline = FakePipeline()
                original = pipeline._decode_displacement
                pipeline._research_decoder_capture = marker
                root = Path(directory) / "capture"
                with self.assertRaisesRegex(ValueError, "already attached"):
                    with PipelineDecoderObserver(pipeline, root, self.identity()):
                        self.fail("Legacy capture must retain exclusive ownership")
                self.assertFalse(root.exists())
                self.assertEqual(pipeline._decode_displacement, original)
                self.assertIs(pipeline._research_decoder_capture, marker)
                self.assertNotIn("_research_observer_active", pipeline.__dict__)
                self.assert_restored(pipeline)

    def test_lazy_loaded_decoder_is_resolved_at_decode_time(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver, load_window
        pipeline = FakePipeline()
        pipeline.temporal_3D_vae = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            with PipelineDecoderObserver(pipeline, root, self.identity()):
                self.assertIsNone(pipeline.temporal_3D_vae)
                pipeline.temporal_3D_vae = Decoder().eval()
                output = pipeline._decode_displacement(**self.inputs())
            self.assertIs(output, pipeline.last_output)
            self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)
            _, record = load_window(root / "window-0000")
            self.assertEqual(record["status"], "captured_unqualified")
            self.assert_restored(pipeline)

    def test_two_windows_have_separate_complete_records(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver, load_window
        pipeline = FakePipeline()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            with PipelineDecoderObserver(pipeline, root, self.identity()):
                pipeline._decode_displacement(**self.inputs())
                second = self.inputs()
                second["window_timesteps"] = torch.tensor([[20., 22., 26.]])
                pipeline._decode_displacement(**second)
            for name, first_time in (("window-0000", 10.), ("window-0001", 20.)):
                tensors, record = load_window(root / name)
                self.assertEqual(record["status"], "captured_unqualified")
                self.assertEqual(tensors["source_timesteps"].reshape(-1).tolist(),
                                 [first_time])
                self.assertTrue((root / name / "decoder/call-0000/record.json").is_file())
            index = json.loads((root / "index.json").read_text())
            self.assertEqual(len(index["windows"]), 2)
            self.assert_restored(pipeline)

    def test_failed_native_forward_retains_inputs_and_restores_method(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver, load_window
        pipeline = FakePipeline()
        pipeline.temporal_3D_vae.fail = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            with self.assertRaisesRegex(RuntimeError, "fixture native forward failed"):
                with PipelineDecoderObserver(pipeline, root, self.identity()):
                    pipeline._decode_displacement(**self.inputs())
            self.assert_restored(pipeline)
            call = root / "window-0000/decoder/call-0000"
            self.assertTrue((call / "inputs.safetensors").is_file())
            self.assertEqual(json.loads((call / "record.json").read_text())["status"],
                             "forward_failed")
            with self.assertRaises(ValueError):
                load_window(root / "window-0000")

    def test_invalid_storage_limit_never_replaces_native_method(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        for limit in (0, -1, 1.5, True):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                pipeline = FakePipeline()
                with self.assertRaises(ValueError):
                    with PipelineDecoderObserver(pipeline, Path(directory) / "capture",
                                                 self.identity(), max_bytes=limit):
                        self.fail("Invalid storage bound entered observer")
                self.assert_restored(pipeline)
                self.assertEqual(pipeline.calls, [])

    def test_window_limit_rejects_before_additional_native_call(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        pipeline = FakePipeline()
        with tempfile.TemporaryDirectory() as directory:
            with PipelineDecoderObserver(pipeline, Path(directory) / "capture",
                                         self.identity(), max_windows=1):
                pipeline._decode_displacement(**self.inputs())
                with self.assertRaises(ValueError):
                    pipeline._decode_displacement(**self.inputs())
            self.assertEqual(len(pipeline.calls), 1)
            self.assert_restored(pipeline)

    def test_persisted_tensor_payload_accepts_exact_bound(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver, load_window
        # Two 144-byte decoder input copies + 72-byte raw output + 252-byte
        # mesh payload, including 144 bytes of float64 output vertices.
        with tempfile.TemporaryDirectory() as directory:
            pipeline = FakePipeline()
            root = Path(directory) / "capture"
            with PipelineDecoderObserver(pipeline, root, self.identity(), max_bytes=612):
                pipeline._decode_displacement(**self.inputs())
            window = root / "window-0000"
            _, record = load_window(window)
            self.assertEqual(record["payload_bytes"], 612)
            files = (window / "decoder/call-0000/inputs.safetensors",
                     window / "decoder/call-0000/tensors.safetensors",
                     window / "meshes.safetensors")
            payload = sum(t.numel() * t.element_size()
                          for path in files for t in load_file(str(path)).values())
            self.assertEqual(payload, 612)
            self.assertGreater(sum(path.stat().st_size for path in files), 612)
            self.assert_restored(pipeline)

    def test_one_byte_short_rejects_before_persisting_raw_or_mesh_output(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        with tempfile.TemporaryDirectory() as directory:
            pipeline = FakePipeline()
            root = Path(directory) / "capture"
            with self.assertRaises(ValueError):
                with PipelineDecoderObserver(pipeline, root, self.identity(), max_bytes=611):
                    pipeline._decode_displacement(**self.inputs())
            window = root / "window-0000"
            call = window / "decoder/call-0000"
            self.assertTrue((call / "inputs.safetensors").is_file())
            self.assertFalse((call / "tensors.safetensors").exists())
            self.assertFalse((window / "meshes.safetensors").exists())
            self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)
            self.assert_restored(pipeline)

    def test_invalid_time_mapping_is_rejected_before_native_call(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver
        malformed = (
            ("window_timesteps", torch.tensor([[10., 10., 10.]])),
            ("target_alphas", torch.tensor([[float("nan"), 1.]])),
            ("target_alphas", torch.tensor([[1., .5]])),
            ("source_alpha", torch.tensor([2.])),
        )
        for key, value in malformed:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                inputs = self.inputs()
                inputs[key] = value
                pipeline = FakePipeline()
                with PipelineDecoderObserver(pipeline, Path(directory) / "capture",
                                             self.identity()):
                    with self.assertRaises(ValueError):
                        pipeline._decode_displacement(**inputs)
                self.assertEqual(pipeline.calls, [])
                self.assert_restored(pipeline)

    def test_mesh_and_decoder_byte_tampering_are_rejected(self):
        from research_math.pipeline_decoder_observer import load_window
        for relative in ("meshes.safetensors", "decoder/call-0000/tensors.safetensors"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                self.capture(root)
                target = root / "window-0000" / relative
                target.write_bytes(target.read_bytes() + b"tampered")
                with self.assertRaises(ValueError):
                    load_window(root / "window-0000")

    def test_replay_and_source_query_preserve_complete_context(self):
        from research_math.pipeline_decoder_observer import replay_window
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            inputs = self.inputs()
            inputs["latents"] = inputs["latents"].to(torch.bfloat16)
            pipeline, _, _ = self.capture(root, inputs=inputs)
            native_call = pipeline.temporal_3D_vae.calls[0]
            output = Path(directory) / "replay"
            replay_window(pipeline.temporal_3D_vae, root / "window-0000", output,
                          identity=self.identity(), atol=0., rtol=0., source_time_query=True)
            calls = pipeline.temporal_3D_vae.calls
            self.assertEqual(len(calls), 3)
            for key, expected in native_call["values"].items():
                self.assertTrue(torch.equal(calls[1]["values"][key], expected), key)
                self.assertEqual(calls[1]["values"][key].dtype, expected.dtype)
                if key != "target_alphas":
                    self.assertTrue(torch.equal(calls[2]["values"][key], expected), key)
            self.assertTrue(torch.equal(calls[2]["values"]["target_alphas"],
                                        native_call["values"]["source_alpha"][:, None]))
            self.assertTrue(calls[1]["inference_mode"])
            self.assertTrue(calls[2]["inference_mode"])
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["raw_matches"])
            self.assertTrue(report["mesh_matches"])
            self.assertIs(report["scientific_effect_qualification"], False)
            self.assertIs(report["replay_qualified"], False)
            source = load_file(str(output / "source-time.safetensors"))
            self.assertEqual(tuple(source["output"].shape), (1, 1, 3, 3))

    def test_replay_preserves_recorded_modes_and_restores_outer_state(self):
        from research_math.pipeline_decoder_observer import PipelineDecoderObserver, replay_window

        def mode_state():
            return (torch.is_inference_mode_enabled(), torch.is_grad_enabled(),
                    torch.is_autocast_cpu_enabled(), str(torch.get_autocast_cpu_dtype()))

        for grad, autocast in ((True, False), (False, True)):
            for fail_replay in (False, True):
                with self.subTest(grad=grad, autocast=autocast, fail_replay=fail_replay), \
                        tempfile.TemporaryDirectory() as directory:
                    pipeline = FakePipeline()
                    root = Path(directory) / "capture"
                    initial = mode_state()
                    expected = (False, grad, autocast, "torch.bfloat16")
                    with torch.inference_mode(False), torch.set_grad_enabled(grad), \
                            torch.autocast("cpu", enabled=autocast, dtype=torch.bfloat16):
                        with PipelineDecoderObserver(pipeline, root, self.identity()):
                            pipeline._decode_displacement(**self.inputs())
                        self.assertEqual(mode_state(), expected)
                    self.assertEqual(mode_state(), initial)
                    record = json.loads((root / "window-0000/decoder/call-0000/record.json").read_text())
                    keys = ("inference_mode", "grad_enabled",
                            "cpu_autocast_enabled", "cpu_autocast_dtype")
                    self.assertEqual(tuple(record[key] for key in keys), expected)
                    pipeline.temporal_3D_vae.fail = fail_replay
                    with torch.inference_mode(), torch.set_grad_enabled(not grad), \
                            torch.autocast("cpu", enabled=not autocast, dtype=torch.bfloat16):
                        outer = mode_state()
                        if fail_replay:
                            with self.assertRaisesRegex(RuntimeError, "fixture native forward failed"):
                                replay_window(pipeline.temporal_3D_vae, root / "window-0000",
                                              Path(directory) / "replay", identity=self.identity(),
                                              atol=0., rtol=0.)
                        else:
                            report = replay_window(
                                pipeline.temporal_3D_vae, root / "window-0000",
                                Path(directory) / "replay", identity=self.identity(), atol=0., rtol=0.)
                            self.assertTrue(report["raw_matches"])
                            self.assertTrue(report["mesh_matches"])
                        self.assertEqual(mode_state(), outer)
                    self.assertEqual(mode_state(), initial)
                    for call in pipeline.temporal_3D_vae.calls:
                        self.assertEqual(tuple(call[key] for key in keys), expected)
                    self.assertEqual(len(pipeline.temporal_3D_vae.calls), 2)
                    self.assert_restored(pipeline)

    def test_raw_values_outside_bounds_replay_as_clamped_float64_meshes(self):
        from research_math.pipeline_decoder_observer import load_window, replay_window
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            pipeline = FakePipeline()
            pipeline.temporal_3D_vae.bias = torch.tensor([-2., 0., 2.])
            self.capture(root, pipeline=pipeline)
            meshes, _ = load_window(root / "window-0000")
            captured = load_file(str(root / "window-0000/decoder/call-0000/tensors.safetensors"))
            raw = captured["output"]
            self.assertTrue((raw[..., 0] < -1).all().item())
            self.assertTrue((raw[..., 2] > 1).all().item())
            self.assertEqual(meshes["vertices"].dtype, torch.float64)
            self.assertTrue((meshes["vertices"][..., 0] == -1).all().item())
            self.assertTrue((meshes["vertices"][..., 2] == 1).all().item())
            self.assertTrue(torch.equal(meshes["vertices"][..., 1], raw[0, ..., 1].double()))
            report = replay_window(pipeline.temporal_3D_vae, root / "window-0000",
                                   Path(directory) / "replay", identity=self.identity(),
                                   atol=0., rtol=0.)
            self.assertTrue(report["raw_matches"])
            self.assertTrue(report["mesh_matches"])

    def test_replay_disagreement_is_retained_and_prevents_source_query(self):
        from research_math.pipeline_decoder_observer import replay_window
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            pipeline, _, _ = self.capture(root)
            pipeline.temporal_3D_vae.bias = .1
            output = Path(directory) / "replay"
            replay_window(pipeline.temporal_3D_vae, root / "window-0000", output,
                          identity=self.identity(), atol=0., rtol=0., source_time_query=True)
            self.assertEqual(len(pipeline.temporal_3D_vae.calls), 2)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["raw_matches"])
            self.assertFalse(report["mesh_matches"])
            self.assertFalse(report["replay_qualified"])
            self.assertFalse((output / "source-time.safetensors").exists())

    def test_replay_rejects_identity_mismatch_before_forward(self):
        from research_math.pipeline_decoder_observer import replay_window
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            pipeline, _, _ = self.capture(root)
            different = self.identity()
            different["weights_sha256"] = "4" * 64
            with self.assertRaises(ValueError):
                replay_window(pipeline.temporal_3D_vae, root / "window-0000",
                              Path(directory) / "replay", identity=different, atol=0., rtol=0.)
            self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)

    def test_replay_requires_explicit_finite_nonnegative_float_tolerances(self):
        from research_math.pipeline_decoder_observer import replay_window
        for atol, rtol in ((-1., 0.), (float("nan"), 0.), (0., float("inf")),
                           (True, 0.), (0., "0")):
            with self.subTest(atol=atol, rtol=rtol), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                pipeline, _, _ = self.capture(root)
                with self.assertRaises(ValueError):
                    replay_window(pipeline.temporal_3D_vae, root / "window-0000",
                                  Path(directory) / "replay", identity=self.identity(),
                                  atol=atol, rtol=rtol)
                self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)

    def test_replay_rejects_training_or_residual_mode(self):
        from research_math.pipeline_decoder_observer import replay_window
        for mode in ("training", "residual"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "capture"
                pipeline, _, _ = self.capture(root)
                if mode == "training":
                    pipeline.temporal_3D_vae.train()
                else:
                    pipeline.temporal_3D_vae.prediction_mode = "residual"
                with self.assertRaises(ValueError):
                    replay_window(pipeline.temporal_3D_vae, root / "window-0000",
                                  Path(directory) / "replay", identity=self.identity(),
                                  atol=0., rtol=0.)
                self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)

    def test_replay_refuses_captured_callback_semantics(self):
        from research_math.pipeline_decoder_observer import replay_window
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "capture"
            inputs = self.inputs()
            callbacks = []
            inputs["step_callback"] = lambda step, total: callbacks.append((step, total))
            pipeline, _, _ = self.capture(root, inputs=inputs)
            self.assertEqual(callbacks, [(1, 2)])
            with self.assertRaises(ValueError):
                replay_window(pipeline.temporal_3D_vae, root / "window-0000",
                              Path(directory) / "replay", identity=self.identity(),
                              atol=0., rtol=0.)
            self.assertEqual(len(pipeline.temporal_3D_vae.calls), 1)


if __name__ == "__main__":
    unittest.main()
