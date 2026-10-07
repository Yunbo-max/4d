"""CPU integration fixtures; never native data or scientific qualification."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from research_math.decoder_observer import load_capture
from research_math.pipeline_decoder_capture import PipelineDecoderCapture


class FixtureDecoder(torch.nn.Module):
    prediction_mode = 'direct'

    def forward(self, latent, framestep, source_alpha, target_alphas, query,
                step_callback=None):
        if step_callback is not None:
            step_callback(1, 1)
        return query[:, None, :, :3].expand(-1, target_alphas.shape[1], -1, -1).clone()


class FixturePipeline:
    """Loads a decoder after attachment, as the official low-RAM path does."""
    def __init__(self):
        self.temporal_3D_vae = None
        self.returned = None

    def _decode_displacement(self, latents, window_timesteps, source_alpha,
                             target_alphas, anchor_mesh, step_callback=None):
        query = torch.from_numpy(np.concatenate(
            [anchor_mesh.vertices, anchor_mesh.vertex_normals], axis=1)[None]).float()
        output = self.temporal_3D_vae(
            latent=latents, framestep=window_timesteps, source_alpha=source_alpha,
            target_alphas=target_alphas, query=query, step_callback=step_callback)
        vertices = output.clamp(-1, 1).cpu().numpy()[0]
        self.returned = [SimpleNamespace(vertices=v, faces=anchor_mesh.faces)
                         for v in vertices]
        return self.returned


class PipelineDecoderCaptureTests(unittest.TestCase):
    def test_torch_module_pipeline_keeps_positional_arguments_and_15_native_targets(self):
        class ModulePipeline(FixturePipeline, torch.nn.Module):
            def __init__(self):
                torch.nn.Module.__init__(self)
                FixturePipeline.__init__(self)

            def _decode_displacement(self, latents, window_timesteps, source_alpha,
                                     target_alphas, anchor_mesh, step_callback=None):
                self.received = (latents, window_timesteps, source_alpha,
                                 target_alphas, anchor_mesh, step_callback)
                return super()._decode_displacement(*self.received)

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ModulePipeline(); args = self.inputs()
            args['target_alphas'] = torch.arange(1, 16).float()[None]/15
            seen = []
            callback = lambda *a: seen.append(a)
            passed = (*args.values(), callback)
            with PipelineDecoderCapture(pipeline, Path(tmp)/'capture', {}):
                pipeline.temporal_3D_vae = FixtureDecoder().eval()
                result = pipeline._decode_displacement(*passed)
            self.assertTrue(all(a is b for a, b in zip(passed, pipeline.received)))
            self.assertEqual(seen, [(1, 1)])
            self.assertEqual(len(result), 15)
            self.assertNotIn('_decode_displacement', vars(pipeline))
            self.assertNotIn('_research_decoder_capture', vars(pipeline))

    def test_observer_command_preserves_frozen_generation_argv(self):
        from research_math.complete_unit_contract import generation_argv
        from research_math.complete_unit_runner import unit_generation_command
        args = SimpleNamespace(source_root=Path('/source'), dataset_root=Path('/data'),
                               capture_decoder=True)
        output = Path('/attempt/unit-output')
        original = generation_argv(args.source_root, Path('/data/data/uid/imgs'),
                                   output/'native-generation', profile='fp16-lowram-v1')
        observed = unit_generation_command(args, output, 'uid', 'fp16-lowram-v1')
        self.assertEqual(observed[:2], original[:2])
        self.assertEqual(observed[9:], original[3:])
        self.assertEqual(Path(observed[2]).name, 'observe_actionmesh_generation.py')
        self.assertEqual(observed[3:9], ['--source-root', '/source', '--capture-root',
            '/attempt/unit-output/decoder-capture', '--identity',
            '/attempt/unit-output/revalidated-unit-manifest.json'])
        args.capture_decoder = False
        self.assertEqual(unit_generation_command(args, output, 'uid', 'fp16-lowram-v1'), original)

    def test_observer_receipt_adds_archive_without_changing_existing_inventory(self):
        from research_math.complete_unit_plan import complete_output_inventory
        records = {'unit_manifest': {'calibration_unit': {'uid': 'uid'}}}
        baseline = complete_output_inventory(records)
        observed = complete_output_inventory(records, capture_decoder=True)
        self.assertEqual(len(baseline), 117)
        self.assertEqual(observed[:117], baseline)
        self.assertEqual(observed[117:], ['actionmesh/unit-output/decoder-capture.tar.gz',
                                        'actionmesh/unit-output/decoder-capture/manifest.json'])
        import prepare_control_scoring_checks as acceptance
        root = Path(__file__).resolve().parents[3]
        self.assertIn(root/'actionmesh/observe_actionmesh_generation.py',
                      acceptance.acceptance_sources(root))

    def test_observer_cannot_consume_existing_full128_pricing(self):
        from research_math.complete_unit_runner import validate_full128_mode
        args = SimpleNamespace(root=Path('/root'), uid='uid', window_id='full128-window-02',
            pricing=Path('/root/inputs/actionbench-full128-queue/pricing.json'),
            capture_decoder=True)
        with self.assertRaisesRegex(ValueError, 'separate unpriced'):
            validate_full128_mode(args)

    def inputs(self):
        mesh = SimpleNamespace(
            vertices=np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.]]),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            vertex_normals=np.tile([0., 0., 1.], (3, 1)))
        return dict(latents=torch.zeros(1, 16, 2, 4),
                    window_timesteps=torch.arange(16).float()[None],
                    source_alpha=torch.zeros(1),
                    target_alphas=torch.linspace(0., 1., 16)[None], anchor_mesh=mesh)

    def test_lazy_decoder_full_inputs_and_geometry_preserve_output_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'capture'
            pipeline = FixturePipeline()
            args = self.inputs()
            rng = torch.get_rng_state().clone()
            with PipelineDecoderCapture(pipeline, root, {'uid': 'fixture'}) as capture:
                pipeline.temporal_3D_vae = FixtureDecoder().eval()
                returned = pipeline._decode_displacement(**args)
                self.assertIs(returned, pipeline.returned)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertNotIn('_decode_displacement', vars(pipeline))
            tensors, record = load_capture(root/'window-0000/decoder/call-0000')
            self.assertEqual(tuple(tensors['latent'].shape), (1, 16, 2, 4))
            self.assertEqual(tuple(tensors['query'].shape), (1, 3, 6))
            np.testing.assert_array_equal(tensors['query'][0, :, 3:].numpy(),
                                          args['anchor_mesh'].vertex_normals)
            with np.load(root/'window-0000/geometry.npz', allow_pickle=False) as geometry:
                np.testing.assert_array_equal(geometry['vertices'],
                                              np.stack([v.vertices for v in returned]))
                np.testing.assert_array_equal(geometry['faces'], args['anchor_mesh'].faces)
            summary = json.loads((root/'manifest.json').read_text())
            self.assertEqual(summary['status'], 'captured_unqualified')
            self.assertEqual(summary['windows'], 1)
            self.assertFalse(summary['native_context_qualified'])
            self.assertFalse(summary['replay_qualified'])
            self.assertEqual(capture.windows, 1)

    def test_original_failure_is_retained_and_hooks_and_method_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = FixturePipeline(); root = Path(tmp)/'capture'
            def fail(*args):
                raise RuntimeError('original decoder failure')
            with self.assertRaisesRegex(RuntimeError, 'original decoder failure'):
                with PipelineDecoderCapture(pipeline, root, {}):
                    pipeline.temporal_3D_vae = FixtureDecoder().eval()
                    pipeline._decode_displacement(**self.inputs(), step_callback=fail)
            self.assertNotIn('_decode_displacement', vars(pipeline))
            self.assertFalse(pipeline.temporal_3D_vae._forward_pre_hooks)
            self.assertFalse(pipeline.temporal_3D_vae._forward_hooks)
            record = json.loads((root/'manifest.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertTrue((root/'window-0000/decoder/call-0000/inputs.safetensors').is_file())

    def test_no_decode_cannot_mint_complete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'capture'
            with self.assertRaisesRegex(ValueError, 'No decoder window'):
                with PipelineDecoderCapture(FixturePipeline(), root, {}):
                    pass
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'], 'failed')

    def test_existing_capture_and_nested_attachment_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'capture'; root.mkdir()
            with self.assertRaises(FileExistsError):
                with PipelineDecoderCapture(FixturePipeline(), root, {}): pass
            pipeline = FixturePipeline()
            with self.assertRaisesRegex(ValueError, 'already attached'):
                with PipelineDecoderCapture(pipeline, Path(tmp)/'first', {}):
                    with PipelineDecoderCapture(pipeline, Path(tmp)/'second', {}): pass
            self.assertFalse((Path(tmp)/'second').exists())
            self.assertNotIn('_decode_displacement', vars(pipeline))

    def test_active_pipeline_observer_is_rejected_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = FixturePipeline()
            original = pipeline._decode_displacement
            pipeline._decode_displacement = original
            pipeline._research_observer_active = True
            root = Path(tmp)/'capture'
            with self.assertRaisesRegex(ValueError, 'already attached'):
                with PipelineDecoderCapture(pipeline, root, {}):
                    self.fail('Another active observer must prevent attachment')
            self.assertIs(pipeline._decode_displacement, original)
            self.assertIs(pipeline._research_observer_active, True)
            self.assertNotIn('_research_decoder_capture', vars(pipeline))
            self.assertFalse(root.exists())

    def test_window_limit_preserves_first_capture_and_restores_prior_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = FixturePipeline()
            original = pipeline._decode_displacement
            pipeline._decode_displacement = original
            root = Path(tmp)/'capture'
            with self.assertRaisesRegex(ValueError, 'window bound'):
                with PipelineDecoderCapture(pipeline, root, {}, max_windows=1):
                    pipeline.temporal_3D_vae = FixtureDecoder().eval()
                    pipeline._decode_displacement(**self.inputs())
                    pipeline._decode_displacement(**self.inputs())
            self.assertIs(pipeline._decode_displacement, original)
            self.assertTrue((root/'window-0000/geometry.npz').is_file())
            self.assertFalse((root/'window-0001').exists())

    def test_changed_topology_is_not_admitted_as_native_context(self):
        class Changed(FixturePipeline):
            def _decode_displacement(self, latents, window_timesteps, source_alpha,
                                     target_alphas, anchor_mesh, step_callback=None):
                meshes = super()._decode_displacement(latents, window_timesteps,
                    source_alpha, target_alphas, anchor_mesh, step_callback)
                meshes[-1].faces = meshes[-1].faces[:, ::-1]
                return meshes
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Changed()
            with self.assertRaisesRegex(ValueError, 'topology'):
                with PipelineDecoderCapture(pipeline, Path(tmp)/'capture', {}):
                    pipeline.temporal_3D_vae = FixtureDecoder().eval()
                    pipeline._decode_displacement(**self.inputs())

    def test_caught_window_failure_cannot_become_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = FixturePipeline(); root = Path(tmp)/'capture'
            with self.assertRaisesRegex(ValueError, 'failed window'):
                with PipelineDecoderCapture(pipeline, root, {}):
                    pipeline.temporal_3D_vae = FixtureDecoder().eval()
                    pipeline._decode_displacement(**self.inputs())
                    try:
                        pipeline._decode_displacement(**self.inputs(),
                            step_callback=lambda *a: (_ for _ in ()).throw(RuntimeError('fail')))
                    except RuntimeError:
                        pass
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'], 'failed')
