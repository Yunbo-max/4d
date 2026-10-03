"""Small CPU fixtures for new variant integrity and damage diagnostics only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from research_census_surface_eval import FRAMES, geometry_diagnostics, validate_variant


class SurfaceEvaluationTests(unittest.TestCase):
    def fixture(self):
        vertices = np.tile(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=np.float32), (16, 1, 1))
        return vertices, np.array([[0, 1, 2]], dtype=np.int64)

    def test_identity_has_no_damage(self):
        vertices, faces = self.fixture()
        geometry = geometry_diagnostics(vertices[0], vertices[0], faces, 2 ** .5)
        self.assertEqual(geometry['total_area_ratio'], 1)
        self.assertEqual(geometry['new_degenerate_triangles'], 0)
        self.assertEqual(geometry['normal_orientation_reversals'], 0)
        self.assertTrue(all(v == 1 for v in geometry['edge_length_ratio_quantiles'].values()))
        self.assertTrue(all(v == 0 for v in geometry['displacement_quantiles'].values()))

    def test_rejects_changed_topology_and_anchor(self):
        vertices, faces = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            sequence = Path(folder) / 'sequence.npz'
            np.savez(sequence, vertices=vertices[list(FRAMES)], faces=faces[:, ::-1], frame_indices=FRAMES)
            with self.assertRaisesRegex(ValueError, 'topology'):
                validate_variant(sequence, vertices, faces)
            corrected = vertices[list(FRAMES)].copy()
            corrected[0, 0, 0] = .01
            np.savez(sequence, vertices=corrected, faces=faces, frame_indices=FRAMES)
            with self.assertRaisesRegex(ValueError, 'anchor'):
                validate_variant(sequence, vertices, faces)

    def test_identity_cannot_contain_a_moving_frame_edit(self):
        vertices, faces = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            sequence = Path(folder) / 'sequence.npz'
            corrected = vertices[list(FRAMES)].copy()
            corrected[1, 0, 2] = .01
            np.savez(sequence, vertices=corrected, faces=faces, frame_indices=FRAMES)
            with self.assertRaisesRegex(ValueError, 'Identity'):
                validate_variant(sequence, vertices, faces, identity=True)

    def test_reports_new_collapse_and_reversal(self):
        vertices, faces = self.fixture()
        flipped = vertices[0].copy()
        flipped[2, 1] = -1
        self.assertEqual(geometry_diagnostics(flipped, vertices[0], faces, 2 ** .5)['normal_orientation_reversals'], 1)
        collapsed = vertices[0].copy()
        collapsed[2] = collapsed[1]
        self.assertEqual(geometry_diagnostics(collapsed, vertices[0], faces, 2 ** .5)['new_degenerate_triangles'], 1)


if __name__ == '__main__':
    unittest.main()
