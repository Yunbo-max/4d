"""Local-only engineering closure checks; no model/scorer/benchmark execution.

Mutations caught: omitting raw decoder inputs, accepting link targets, and
publishing a summary without the actual bytes. These are file fixtures only.
"""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


class NativeContextDeliveryTests(unittest.TestCase):
    def populate(self, root, source_time=False):
        from research_math.native_context_delivery import required_raw_paths
        for name in required_raw_paths(source_time_query=source_time):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('engineering file: ' + name).encode())
        (root / 'result.json').write_text('{"status":"completed_unqualified"}')

    def test_archive_binds_every_raw_byte_including_optional_diagnostics(self):
        from research_math.native_context_delivery import finalize_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, True)
            (root / 'unobserved/grid_normal.mp4').write_bytes(b'opaque optional bytes')
            record = finalize_bundle(root, source_time_query=True)
            manifest = json.loads((root / 'raw-manifest.json').read_text())
            with tarfile.open(root / 'raw-evidence.tar', 'r:') as archive:
                self.assertEqual(archive.getnames(), [row['path'] for row in manifest['files']])
                self.assertNotIn('result.json', archive.getnames())
                self.assertIn('unobserved/grid_normal.mp4', archive.getnames())
                for row in manifest['files']:
                    data = archive.extractfile(row['path']).read()
                    self.assertEqual(len(data), row['bytes'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'])
            self.assertEqual(record['archive']['sha256'], hashlib.sha256(
                (root / 'raw-evidence.tar').read_bytes()).hexdigest())
            self.assertIs(manifest['native_context_qualified'], False)
            with self.assertRaises(FileExistsError):
                finalize_bundle(root, source_time_query=True)

    def test_missing_decoder_payload_rejects_before_archive_publication(self):
        from research_math.native_context_delivery import finalize_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            (root / 'capture/window-0000/decoder/call-0000/inputs.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing raw'):
                finalize_bundle(root, source_time_query=False)
            self.assertFalse((root / 'raw-evidence.tar').exists())
            self.assertFalse((root / 'raw-manifest.json').exists())

    def test_requested_source_time_output_is_mandatory(self):
        from research_math.native_context_delivery import finalize_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            with self.assertRaisesRegex(ValueError, 'Missing raw'):
                finalize_bundle(root, source_time_query=True)

    def test_linked_or_unexpected_special_file_is_rejected(self):
        from research_math.native_context_delivery import finalize_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            (root / 'linked').symlink_to(root / 'result.json')
            with self.assertRaisesRegex(ValueError, 'Physical'):
                finalize_bundle(root, source_time_query=False)
            self.assertFalse((root / 'raw-evidence.tar').exists())

    def test_collection_boundary_counts_archive_disk_footprint(self):
        from research_math.native_context_delivery import finalize_bundle, collection_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            before = collection_snapshot(root)
            record = finalize_bundle(root, source_time_query=False)
            after = collection_snapshot(root)
            self.assertGreaterEqual(after['output_bytes'] - before['output_bytes'],
                                    record['archive']['bytes'])
            self.assertGreater(after['collector_rss_bytes'], 0)
            self.assertIs(after['exact_peak'], False)

    def completed_bundle(self, root, *, source_time=True):
        from research_math.native_context_delivery import finalize_bundle
        self.populate(root, source_time)
        identity = {
            'kind': 'native-context-generation-identity', 'version': 1,
            'uid': 'fixture-uid', 'gpu_uuid': 'GPU-fixture',
            'scientific_effect_qualification': False,
            'native_context_qualified': False,
        }
        (root/'generation-identity.json').write_text(json.dumps(identity)+'\n')
        pair = {
            'kind': 'paired-native-generation-comparison', 'version': 1,
            'matches': True, 'scientific_effect_qualification': False,
            'native_context_qualified': False}
        (root/'paired-comparison.json').write_text(json.dumps(pair)+'\n')
        windows = [{'status': 'replayed_unqualified', 'raw_matches': True,
                    'mesh_matches': True}]
        replay = {
            'kind': 'native-context-stage-result', 'version': 1, 'stage': 'replay',
            'status': 'completed_unqualified', 'all_comparisons_match': True,
            'replay_qualified': False, 'native_context_qualified': False,
            'native_scientific_qualification': False,
            'scientific_effect_qualification': False,
            'candidate_methods_tested': False, 'dispatch_ready': False,
            'paired_comparison': pair, 'windows': windows}
        (root/'replay/stage-result.json').write_text(json.dumps(replay)+'\n')
        raw_bundle = finalize_bundle(root, source_time_query=source_time)
        result = {
            'kind': 'paired-native-context-instrument', 'version': 1,
            'status': 'completed_unqualified',
            'native_context_qualified': False,
            'native_scientific_qualification': False,
            'scientific_effect_qualification': False,
            'replay_qualified': False, 'candidate_methods_tested': False,
            'dispatch_ready': False,
            'all_comparisons_match': True,
            'final_integrity': 'matched', 'gpu_uuid': 'GPU-fixture',
            'source_time_query_requested': source_time,
            'paired_comparison': pair, 'replay_reports': windows,
            'raw_bundle': raw_bundle,
        }
        (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        values = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in ('result.json', 'raw-manifest.json', 'raw-evidence.tar')}
        values.update(uid='fixture-uid', gpu_uuid='GPU-fixture', source_time_query=source_time,
                      generation_identity_sha256=hashlib.sha256(
                          (root/'generation-identity.json').read_bytes()).hexdigest())
        return values

    def consumer_args(self, expected):
        return {
            'expected_result_sha256': expected['result.json'],
            'expected_manifest_sha256': expected['raw-manifest.json'],
            'expected_archive_sha256': expected['raw-evidence.tar'],
            'expected_uid': expected['uid'], 'expected_gpu_uuid': expected['gpu_uuid'],
            'expected_generation_identity_sha256': expected['generation_identity_sha256'],
            'expected_source_time_query': expected['source_time_query'],
            'max_files': 256, 'max_unpacked_bytes': 16 * 1024 * 1024,
            'max_member_bytes': 4 * 1024 * 1024,
            'max_archive_bytes': 16 * 1024 * 1024,
            'max_metadata_bytes': 1024 * 1024,
        }

    def test_consumer_rehashes_and_extracts_every_member_without_claiming_qualification(self):
        from research_math.native_context_delivery import consume_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle)
            record = consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                bundle/'raw-evidence.tar', root/'consumed', **self.consumer_args(expected))
            manifest = json.loads((bundle/'raw-manifest.json').read_text())
            self.assertEqual(record['extracted_files'], len(manifest['files']))
            self.assertFalse(record['native_context_qualified'])
            self.assertFalse(record['scientific_effect_qualification'])
            for row in manifest['files']:
                target = root/'consumed/raw'/row['path']
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), row['sha256'])
            self.assertEqual((root/'consumed/bundle-result.json').read_bytes(),
                             (bundle/'result.json').read_bytes())
            with self.assertRaises(FileExistsError):
                consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'consumed', **self.consumer_args(expected))

    def test_consumer_rejects_tampered_input_before_output_creation(self):
        from research_math.native_context_delivery import consume_bundle
        for name in ('result.json', 'raw-manifest.json', 'raw-evidence.tar'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
                expected = self.completed_bundle(bundle)
                with (bundle/name).open('ab') as stream: stream.write(b'tampered')
                with self.assertRaises(ValueError):
                    consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                        bundle/'raw-evidence.tar', root/'consumed', **self.consumer_args(expected))
                self.assertFalse((root/'consumed').exists())

    def test_consumer_rejects_unsafe_manifest_member_even_when_new_hashes_are_pinned(self):
        from research_math.native_context_delivery import consume_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle)
            manifest_path = bundle/'raw-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files'][0]['path'] = '../escape'
            manifest_path.write_text(json.dumps(manifest)+'\n')
            result_path = bundle/'result.json'; result = json.loads(result_path.read_text())
            result['raw_bundle']['manifest']['sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result)+'\n')
            with self.assertRaisesRegex(ValueError, 'canonical relative'):
                args = self.consumer_args(expected)
                args.update(expected_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                            expected_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            expected_archive_sha256=hashlib.sha256(
                                (bundle/'raw-evidence.tar').read_bytes()).hexdigest())
                consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'consumed', **args)
            self.assertFalse((root/'consumed').exists())

    def test_consumer_rejects_symlinked_output_parent_before_writing(self):
        from research_math.native_context_delivery import consume_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle)
            physical = root/'physical'; physical.mkdir()
            (root/'linked-parent').symlink_to(physical, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Physical output parent'):
                consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'linked-parent/consumed',
                    **self.consumer_args(expected))
            self.assertFalse((physical/'consumed').exists())

    def test_consumer_rejects_wrong_producer_identity_and_scope_overclaim(self):
        from research_math.native_context_delivery import consume_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle); args = self.consumer_args(expected)
            args['expected_uid'] = 'wrong-unit'
            with self.assertRaisesRegex(ValueError, 'identity'):
                consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'wrong-identity', **args)
            result_path = bundle/'result.json'; result = json.loads(result_path.read_text())
            result['dispatch_ready'] = True
            result_path.write_text(json.dumps(result)+'\n')
            args = self.consumer_args(expected)
            args['expected_result_sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'overclaimed'):
                consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'overclaim', **args)
            self.assertFalse((root/'wrong-identity').exists())
            self.assertFalse((root/'overclaim').exists())

    def test_consumer_rejects_missing_required_false_and_positive_manifest_scope(self):
        from research_math.native_context_delivery import consume_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle)
            result_path = bundle/'result.json'; result = json.loads(result_path.read_text())
            del result['native_scientific_qualification']
            result_path.write_text(json.dumps(result)+'\n')
            args = self.consumer_args(expected)
            args['expected_result_sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'overclaimed'):
                consume_bundle(result_path, bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'missing-false', **args)
            result['native_scientific_qualification'] = False
            manifest_path = bundle/'raw-manifest.json'
            manifest = json.loads(manifest_path.read_text()); manifest['dispatch_ready'] = True
            manifest_path.write_text(json.dumps(manifest)+'\n')
            result['raw_bundle']['manifest']['sha256'] = hashlib.sha256(
                manifest_path.read_bytes()).hexdigest()
            result_path.write_text(json.dumps(result)+'\n')
            args = self.consumer_args(expected)
            args.update(expected_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                        expected_manifest_sha256=hashlib.sha256(
                            manifest_path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'overclaimed'):
                consume_bundle(result_path, manifest_path, bundle/'raw-evidence.tar',
                    root/'manifest-overclaim', **args)
            self.assertFalse((root/'missing-false').exists())
            self.assertFalse((root/'manifest-overclaim').exists())

    def test_consumer_applies_preflight_file_and_byte_limits(self):
        from research_math.native_context_delivery import consume_bundle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle)
            for key, value in (('max_files', 1), ('max_unpacked_bytes', 1),
                               ('max_member_bytes', 1), ('max_archive_bytes', 1)):
                args = self.consumer_args(expected); args[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'limit'):
                    consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                        bundle/'raw-evidence.tar', root/('limited-'+key), **args)
                self.assertFalse((root/('limited-'+key)).exists())

    def test_consumer_bounds_metadata_and_preflights_real_free_space(self):
        from research_math.native_context_delivery import consume_bundle
        import shutil
        for name in ('result.json', 'raw-manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
                expected = self.completed_bundle(bundle)
                limit = max((bundle/'result.json').stat().st_size,
                            (bundle/'raw-manifest.json').stat().st_size)
                with (bundle/name).open('ab') as stream:
                    stream.write(b'x' * (limit + 1))
                args = self.consumer_args(expected); args['max_metadata_bytes'] = limit
                args['expected_'+('result' if name == 'result.json' else 'manifest')+'_sha256'] = \
                    hashlib.sha256((bundle/name).read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                        bundle/'raw-evidence.tar', root/('metadata-'+name), **args)
                self.assertFalse((root/('metadata-'+name)).exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); bundle = root/'bundle'; bundle.mkdir()
            expected = self.completed_bundle(bundle)
            args = self.consumer_args(expected)
            args['max_unpacked_bytes'] = shutil.disk_usage(root).free + 1
            with self.assertRaisesRegex(ValueError, 'free disk'):
                consume_bundle(bundle/'result.json', bundle/'raw-manifest.json',
                    bundle/'raw-evidence.tar', root/'no-space', **args)
            self.assertFalse((root/'no-space').exists())


if __name__ == '__main__':
    unittest.main()
