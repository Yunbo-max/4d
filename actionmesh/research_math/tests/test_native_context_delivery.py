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


if __name__ == '__main__':
    unittest.main()
