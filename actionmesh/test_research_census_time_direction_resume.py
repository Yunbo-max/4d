"""CPU-only replay and tamper-rejection checks; no model or GPU initialization."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import research_census_time_direction as original
from research_census_time_direction_resume import verify_controls


def put(path, value):
    path.write_text(json.dumps(value))


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        root = root.resolve()
        case, attempt = root / 'case', root / 'v1'
        case.mkdir(); attempt.mkdir()
        checkpoint = root / 'repo/pretrained_weights/ActionMesh/autoencoder'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{}')
        (root / 'repo/model.py').write_text('# fixture')
        vertices = np.tile(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32), (16, 1, 1))
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        ids = np.arange(3, dtype=np.int64)
        latent = np.zeros((16, 2048, 64), dtype=np.float32)
        query = np.concatenate((vertices[0], np.tile(np.array([[0, 0, 1]], dtype=np.float32), (3, 1))), axis=1)
        np.savez_compressed(case / 'prepared.npz', timesteps=np.arange(16), anchor_latent=latent[:1].astype(np.float16),
            anchor_timesteps=np.array([0], dtype=np.float32), anchor_vertices=vertices[0], anchor_faces=faces,
            anchor_query_features=query, query_vertex_ids=ids, seed=np.array(42))
        np.savez_compressed(case / 'denoised.npz', latents=latent, timesteps=np.arange(16), seed=np.array(42))
        np.savez_compressed(case / 'sequence.npz', vertices=vertices, faces=faces, frame_indices=np.arange(16),
                            timesteps=np.arange(16), query_vertex_ids=ids)
        uid = original.FROZEN_UIDS[0]
        generation = dict(status='completed', frames=16, uid=uid, seed=42,
                          sha256={name: original.digest(case / name) for name in ('prepared.npz', 'denoised.npz', 'sequence.npz')})
        put(case / 'report.json', generation)
        put(case / 'code-provenance.json', {})
        hashes = {name: original.digest(case / name) for name in (*generation['sha256'], 'report.json', 'code-provenance.json')}
        diagonal = float(np.sqrt(2))
        provenance = dict(uid=uid, seed=42, source_hashes=hashes,
            code_sha256={'model.py': original.digest(root / 'repo/model.py')},
            checkpoint_sha256={'config.json': original.digest(checkpoint / 'config.json')})
        put(attempt / 'source-provenance.json', provenance)
        prior = dict(provenance, status='running', anchor_diagonal=diagonal,
            implementation_sha256=original.digest(Path(original.__file__)), arms={},
            source_latents_array_sha256=original.array_digest(latent), cached_query_sha256=original.array_digest(query),
            query_vertex_ids_sha256=original.array_digest(ids),
            control_protocol={'forward': 'bitwise equality to original native sequence required',
                'permutation_rms_over_D_limit': original.PERMUTATION_RMS_OVER_D_LIMIT,
                'permutation_max_coordinate_over_D_limit': original.PERMUTATION_MAX_COORD_OVER_D_LIMIT,
                'reversal_signal_floor_multiplier': 10})
        for arm in ('forward', 'row_permutation'):
            directory = attempt / 'variants' / arm
            directory.mkdir(parents=True)
            np.savez_compressed(directory / 'sequence.npz', vertices=vertices, faces=faces,
                query_vertex_ids=ids, frame_indices=np.arange(16), timesteps=np.arange(16), decoder_clock_times=np.arange(16))
            mapping = original.arm_mapping(arm)
            report = dict(status='completed', uid=uid, seed=42, variant=arm, control_pass=True,
                sequence_sha256=original.digest(directory / 'sequence.npz'),
                sha256={'sequence.npz': original.digest(directory / 'sequence.npz')},
                mapping={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in mapping.items()},
                latent_input_sha256=original.array_digest(latent[mapping['latent_row_physical_ids']]),
                discrepancy_vs_native=original.discrepancy(vertices, vertices, diagonal))
            if arm == 'row_permutation':
                report['discrepancy_vs_forward'] = original.discrepancy(vertices, vertices, diagonal)
            put(directory / 'report.json', report)
            prior['arms'][arm] = report
        put(attempt / 'report.json', prior)
        receipt = root / 'timeout.json'
        put(receipt, dict(status='stopped_after_failure', source_sha256=prior['implementation_sha256'],
            protocol_sha256='1' * 64, jobs=[dict(status='failed', termination_reason='deadline', exit_code=-15,
                argv=['python', 'v1.py', '--case-dir', str(case), '--output', str(attempt)])]))
        return argparse.Namespace(root=root, case_dir=case, source_attempt=attempt, timeout_receipt=receipt)

    def test_reuses_exact_controls_and_rejects_tampered_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            verified = verify_controls(args)
            self.assertEqual(set(verified['controls']), {'forward', 'row_permutation'})
            self.assertTrue(all(not item['gpu_recomputed'] for item in verified['controls'].values()))
            with (args.source_attempt / 'variants/row_permutation/sequence.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_controls(args)

    def test_rejects_different_timeout_case(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            receipt = json.loads(args.timeout_receipt.read_text())
            receipt['jobs'][0]['argv'][3] = str(Path(folder) / 'another-case')
            put(args.timeout_receipt, receipt)
            with self.assertRaisesRegex(ValueError, 'exact case'):
                verify_controls(args)


if __name__ == '__main__':
    unittest.main()
