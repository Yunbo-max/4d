"""Input integrity tests only: no Blender, GPU, or remote access."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from research_census_export import digest, load_case


class ExportInputTests(unittest.TestCase):
    def fixture(self, root, *, status='completed', nonfinite=False, mismatched=False):
        vertices = np.zeros((16, 3, 3), dtype=np.float32)
        vertices[:, 1, 0] = 1
        vertices[:, 2, 1] = 1
        if nonfinite:
            vertices[2, 0, 0] = np.nan
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        np.save(root / 'deformations_vertices.npy', vertices)
        np.save(root / 'deformations_faces.npy', faces)
        np.savez(root / 'sequence.npz', vertices=vertices,
                 faces=faces[:, ::-1] if mismatched else faces,
                 frame_indices=np.arange(16), timesteps=np.arange(16))
        report = {'status': status, 'frames': 16, 'sha256': {name: digest(root / name) for name in
                  ('deformations_vertices.npy', 'deformations_faces.npy', 'sequence.npz')}}
        (root / 'report.json').write_text(json.dumps(report))

    def test_rejects_incomplete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root, status='failed')
            with self.assertRaisesRegex(ValueError, 'completed'):
                load_case(root)

    def test_rejects_nonfinite_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root, nonfinite=True)
            with self.assertRaisesRegex(ValueError, 'finite'):
                load_case(root)

    def test_rejects_topology_disagreement_despite_valid_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root, mismatched=True)
            with self.assertRaisesRegex(ValueError, 'topology mismatch'):
                load_case(root)

    def test_rejects_changed_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            load_case(root)
            with (root / 'sequence.npz').open('ab') as stream:
                stream.write(b'changed-after-inference')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                load_case(root)


if __name__ == '__main__':
    unittest.main()
