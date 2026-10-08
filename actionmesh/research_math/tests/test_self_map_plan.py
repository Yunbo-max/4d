"""Real capture/transport/staging integration with an engineering decoder fixture.

No loader, capture, replay, consumed-bundle validator, or _stage boundary is
mocked. The decoder is deliberately a tiny software fixture, not ActionMesh or
scientific evidence. Separately run the retained-native C01 acceptance test.
"""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from research_math import self_map_candidate as candidate


def engineering_context(root):
    from research_math.tests.test_pipeline_decoder_observer import FakePipeline, PipelineDecoderObserverTests
    from research_math.pipeline_decoder_observer import PipelineDecoderObserver, replay_window
    from research_math.native_context_runner import compare_sequences, aggregate_comparisons
    from research_math.native_context_delivery import required_raw_paths, finalize_bundle, consume_bundle
    raw = root / 'producer'; raw.mkdir()
    identity = {'kind': 'native-context-generation-identity', 'version': 1,
                'uid': 'engineering-fixture', 'gpu_uuid': 'GPU-engineering-fixture',
                'generation': {'seed': 42}, 'native_context_qualified': False,
                'scientific_effect_qualification': False}
    candidate.write(raw / 'generation-identity.json', identity)
    fixture = PipelineDecoderObserverTests()
    inputs = fixture.inputs()
    inputs.update(latents=torch.arange(64, dtype=torch.float32).reshape(1, 16, 2, 2),
                  window_timesteps=torch.arange(16, dtype=torch.float32)[None],
                  target_alphas=torch.arange(1, 16, dtype=torch.float32)[None] / 15.)
    pipeline = FakePipeline()
    with torch.inference_mode(), PipelineDecoderObserver(pipeline, raw / 'capture', identity):
        output = pipeline._decode_displacement(**inputs)
    vertices = np.concatenate([inputs['anchor_mesh'].vertices[None],
                               np.stack([mesh.vertices for mesh in output])]).astype(np.float32)
    for name in ('observed', 'unobserved'):
        case = raw / name; case.mkdir()
        np.savez_compressed(case / 'sequence.npz', vertices=vertices,
            faces=inputs['anchor_mesh'].faces, frame_indices=np.arange(16),
            timesteps=np.arange(16, dtype=np.float32), query_vertex_ids=np.arange(vertices.shape[1]))
        candidate.write(case / 'report.json', {'status': 'completed', 'uid': identity['uid'], 'seed': 42,
            'sha256': {'sequence.npz': candidate.digest(case / 'sequence.npz')}})
    pair = compare_sequences(raw / 'unobserved/sequence.npz', raw / 'observed/sequence.npz', atol=0., rtol=0.)
    candidate.write(raw / 'paired-comparison.json', pair)
    replay = replay_window(pipeline.temporal_3D_vae, raw / 'capture/window-0000', raw / 'replay/window-0000',
                           identity=identity, atol=0., rtol=0., source_time_query=True)
    aggregate = aggregate_comparisons(pair, [replay])
    stage = {'kind': 'native-context-stage-result', 'version': 1, 'stage': 'replay',
             'status': 'completed_unqualified', 'paired_comparison': pair, 'windows': [replay], **aggregate}
    candidate.write(raw / 'replay/stage-result.json', stage)
    # Remaining transport inventory is inert fixture bytes. Its data is never
    # used as a model input, GLB/scorer evidence or native experiment result.
    for relative in required_raw_paths(source_time_query=True):
        path = raw / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'engineering transport fixture, not native evidence\n')
    bundle = finalize_bundle(raw, source_time_query=True)
    result = {'kind': 'paired-native-context-instrument', 'version': 1,
        'status': 'completed_unqualified', 'final_integrity': 'matched',
        'gpu_uuid': identity['gpu_uuid'], 'source_time_query_requested': True,
        'paired_comparison': pair, 'replay_reports': [replay], 'raw_bundle': bundle, **aggregate}
    candidate.write(raw / 'result.json', result)
    consumed = root / 'external-retained-root/context'
    consumed.parent.mkdir()
    consume_bundle(raw / 'result.json', raw / 'raw-manifest.json', raw / 'raw-evidence.tar', consumed,
        expected_result_sha256=candidate.digest(raw / 'result.json'),
        expected_manifest_sha256=candidate.digest(raw / 'raw-manifest.json'),
        expected_archive_sha256=candidate.digest(raw / 'raw-evidence.tar'),
        expected_uid=identity['uid'], expected_gpu_uuid=identity['gpu_uuid'],
        expected_generation_identity_sha256=candidate.digest(raw / 'generation-identity.json'),
        expected_source_time_query=True, max_files=500, max_unpacked_bytes=32 * 1024 * 1024,
        max_member_bytes=8 * 1024 * 1024, max_archive_bytes=64 * 1024 * 1024,
        max_metadata_bytes=4 * 1024 * 1024)
    return consumed


class SelfMapPlanIntegrationTests(unittest.TestCase):
    def test_real_cross_root_freeze_and_context_to_candidate(self):
        from prepare_self_map_candidate import build_plans
        import run_experiments
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = Path(__file__).resolve().parents[3]
            package = root / 'actionmesh/research_math'; package.mkdir(parents=True)
            for name in ('__init__.py', 'self_map_candidate.py', 'native_context_delivery.py',
                         'native_context_runner.py', 'pipeline_decoder_observer.py',
                         'decoder_observer.py', 'complete_unit_export.py'):
                shutil.copy2(project / 'actionmesh/research_math' / name, package / name)
            shutil.copy2(project / 'actionmesh/prepare_self_map_candidate.py',
                         root / 'actionmesh/prepare_self_map_candidate.py')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c',
                'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
            context = engineering_context(root)
            native, outer = build_plans(root, context_root=context, run_id='c01-cross-root-fixture',
                plan_dir=root / 'plans/c01', wall_seconds=300, ram_mib=4096,
                coordinate_bounds=[-1., 1.], bounds_policy='preserve_and_report')
            self.assertEqual(native['limits']['max_attempts'], 1)
            self.assertEqual(native['limits']['max_retries_per_trial'], 0)
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            attempt = root / 'attempt'; attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, native['jobs'][0], attempt)
            for option in ('--consumption', '--manifest', '--result'):
                staged = Path(command[command.index(option) + 1])
                self.assertTrue(staged.is_relative_to(workspace))
                self.assertTrue(staged.is_file())
            raw_files = {}
            for index, value in enumerate(command):
                if value == '--raw-file':
                    name, path = command[index + 1:index + 3]
                    self.assertTrue(Path(path).is_relative_to(workspace))
                    raw_files[name] = Path(path)
            self.assertEqual(set(raw_files), {row['path'] for row in
                json.loads((context / 'bundle-manifest.json').read_text())['files']})
            # Destroy live inputs after freeze. The actual candidate must still
            # finish from the staged snapshots, including native tensor replay.
            shutil.rmtree(context)
            result = candidate.export_self_map_candidate(
                command[command.index('--consumption') + 1], command[command.index('--manifest') + 1],
                command[command.index('--result') + 1], raw_files, cwd / 'c01-self-map-output',
                expected_consumption_sha256=command[command.index('--expected-consumption-sha256') + 1],
                coordinate_bounds=[-1., 1.], bounds_policy='preserve_and_report')
            self.assertEqual(result['status'], 'completed', result)
            verified = candidate.verify_candidate_artifacts(cwd / 'c01-self-map-output')
            self.assertEqual(set(verified['arms']), set(candidate.ROLES))
            self.assertTrue(all(path.is_relative_to(workspace) for path in verified['context_files']))
            altered = cwd / 'c01-self-map-output/context/raw/replay/window-0000/source-time.safetensors'
            altered.write_bytes(altered.read_bytes() + b'mutation')
            with self.assertRaises(ValueError):
                candidate.verify_candidate_artifacts(cwd / 'c01-self-map-output')


if __name__ == '__main__':
    unittest.main()
