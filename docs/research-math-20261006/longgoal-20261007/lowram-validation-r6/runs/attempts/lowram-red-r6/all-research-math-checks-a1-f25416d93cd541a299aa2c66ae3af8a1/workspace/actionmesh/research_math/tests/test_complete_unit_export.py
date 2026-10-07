import unittest
import numpy as np
from research_math.complete_unit_export import restore_vertices, validate_sequence

class CompleteUnitExportTests(unittest.TestCase):
    def test_inverse_official_axis_transform(self):
        original = np.arange(16*4*3, dtype=np.float32).reshape(16,4,3)
        exported = original[:,:,[2,0,1]].copy()
        exported[:,:,0] *= -1
        np.testing.assert_array_equal(restore_vertices(exported), original)

    def test_rejects_nonfinite_or_missing_frames(self):
        for vertices in (np.zeros((15,4,3)), np.full((16,4,3), np.nan)):
            with self.assertRaises(ValueError): restore_vertices(vertices)

    def test_rejects_invalid_faces(self):
        vertices = np.zeros((16,4,3))
        for faces in (np.array([[0,1,4]]), np.array([[0.,1.,2.]]), np.zeros((0,3),int)):
            with self.assertRaises(ValueError): validate_sequence(vertices,faces)

    def test_preserves_identity_timeline_and_vertices(self):
        vertices = np.arange(192, dtype=np.float32).reshape(16,4,3)
        faces = np.array([[0,1,2],[1,2,3]],dtype=np.int32)
        result=validate_sequence(vertices,faces)
        np.testing.assert_array_equal(result['vertices'],vertices)
        np.testing.assert_array_equal(result['query_vertex_ids'],np.arange(4))
        np.testing.assert_array_equal(result['frame_indices'],np.arange(16))
        np.testing.assert_array_equal(result['timesteps'],np.arange(16))
