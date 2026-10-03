"""CPU provenance fixtures: stale evidence must never qualify replacement arms."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import research_census_direction_motion as runner


UID = '000-037_1358c424008a43cbaa35eba5e58551ac'
ARMS = ('forward', 'row_permutation', 'time_reversal')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.args = SimpleNamespace(
            case_dir=base/'cases'/f'{UID}__seed42__attempt2',
            direction_dir=base/'time-direction'/'mannequin-seed42-v2',
            direction_eval_dir=base/'time-direction-eval'/'mannequin-seed42-v2',
            stage_gt_dir=base/'stage-gt'/'mannequin-seed42-v1',
            native_cache_dir=base/'motion-controls'/'mannequin-seed42-v1',
            gt_dir=base/'data')
        a = self.args
        for name in ('prepared.npz', 'denoised.npz', 'sequence.npz'):
            path = a.case_dir/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('native '+name).encode())
        self.common = dict(status='completed', uid=UID, seed=42)
        write_json(a.case_dir/'report.json', dict(self.common,
            sha256={'sequence.npz': sha(a.case_dir/'sequence.npz')}))
        self.matrix = a.stage_gt_dir/'shared-anchor-transform.npy'
        self.matrix.parent.mkdir(parents=True)
        self.matrix.write_bytes(b'frozen matrix')
        write_json(a.stage_gt_dir/'report.json', dict(self.common,
            shared_matrix_sha256=sha(self.matrix)))
        gt = a.gt_dir/UID/'surfaces.npy'
        gt.parent.mkdir(parents=True)
        gt.write_bytes(b'frozen tracked GT')
        for name in ('report.json', 'provenance.json', 'protocol.json', 'controls.json',
                     'per-frame-shape.json', 'sampling-and-maps.npz', 'aligned-material-clouds.npz'):
            path = a.native_cache_dir/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('native cache '+name).encode())
        write_json(a.direction_dir/'report.json', dict(self.common,
            controls_passed=True, GT_read=False, training=False,
            source_hashes={name: sha(a.case_dir/name) for name in
                           ('report.json', 'prepared.npz', 'denoised.npz', 'sequence.npz')}))
        for arm in ARMS:
            folder = a.direction_dir/'variants'/arm
            folder.mkdir(parents=True)
            (folder/'sequence.npz').write_bytes(('qualified '+arm).encode())
            write_json(folder/'report.json', dict(self.common, control_pass=True,
                sequence_sha256=sha(folder/'sequence.npz'),
                sha256={'sequence.npz': sha(folder/'sequence.npz')}))
        write_json(a.direction_eval_dir/'report.json', dict(self.common,
            independent_control_gates_passed=True, all_arms_retained=True))
        inputs = [a.case_dir/'report.json', a.case_dir/'sequence.npz', gt,
                  self.matrix, a.stage_gt_dir/'report.json', a.direction_dir/'report.json']
        inputs += [a.native_cache_dir/name for name in ('report.json', 'provenance.json',
                   'protocol.json', 'controls.json', 'per-frame-shape.json', 'sampling-and-maps.npz')]
        inputs += [a.direction_dir/'variants'/arm/name for arm in ARMS
                   for name in ('report.json', 'sequence.npz')]
        write_json(a.direction_eval_dir/'provenance.json', dict(
            source_files_sha256={str(path): sha(path) for path in inputs},
            same_matrix_sha256=sha(self.matrix), native_cache_qualified=True))

    def qualify(self):
        self.assertTrue(callable(getattr(runner, 'qualify_sources', None)),
                        'readout must bind its sources to the completed evaluator evidence')
        return runner.qualify_sources(self.args)

    def test_unchanged_completed_artifacts_qualify(self):
        result = self.qualify()
        self.assertEqual(result['generation']['uid'], UID)
        self.assertEqual(result['arm_reports']['time_reversal']['seed'], 42)

    def test_self_consistent_replacement_arm_rejected_by_stale_evaluation(self):
        folder = self.args.direction_dir/'variants'/'time_reversal'
        (folder/'sequence.npz').write_bytes(b'replacement with the same mesh identity')
        report = json.loads((folder/'report.json').read_text())
        report['sequence_sha256'] = sha(folder/'sequence.npz')
        report['sha256']['sequence.npz'] = sha(folder/'sequence.npz')
        write_json(folder/'report.json', report)
        with self.assertRaisesRegex(ValueError, 'evaluation source hash mismatch'):
            self.qualify()

    def test_stale_forward_control_also_rejected(self):
        (self.args.direction_dir/'variants'/'forward'/'sequence.npz').write_bytes(b'changed control')
        with self.assertRaisesRegex(ValueError, 'evaluation source hash mismatch'):
            self.qualify()

    def test_changed_latent_rejected_even_though_not_in_eval_provenance(self):
        (self.args.case_dir/'denoised.npz').write_bytes(b'different latents')
        with self.assertRaisesRegex(ValueError, 'direction source hash mismatch'):
            self.qualify()

    def test_seed_mismatch_is_not_relabelled_seed42(self):
        path = self.args.direction_eval_dir/'report.json'
        value = json.loads(path.read_text())
        value['seed'] = 43
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'UID/seed'):
            self.qualify()

    def test_source_change_after_qualification_rejected(self):
        qualified = self.qualify()
        (self.args.native_cache_dir/'aligned-material-clouds.npz').write_bytes(b'replaced cached cloud')
        with self.assertRaisesRegex(ValueError, 'changed during'):
            runner.revalidate_sources(qualified['source_sha256'])


if __name__ == '__main__':
    unittest.main()
