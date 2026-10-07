"""Protect source identity, distinct control/reference spaces and GPU probe limits."""
import unittest
import numpy as np


class NativeProbeTests(unittest.TestCase):
    def test_contract_preserves_middle_frame_and_edit_targets_late_frame(self):
        from research_math.native_probe import measurement_maps
        A, B = measurement_maps(4)
        x = np.zeros((3, 4, 3)); x[1, :, 0] = 2.; x[2, :, 1] = 3.
        np.testing.assert_allclose(A @ x.ravel(), [2, 0, 0])
        np.testing.assert_allclose(B @ x.ravel(), [0, 3, 0])
        self.assertEqual(A.shape, (3, 36))

    def test_query_controls_keep_normals_unchanged(self):
        from research_math.native_probe import query_control_basis
        q = np.array([[0.,0,0,1,0,0],[1,0,0,0,1,0],[0,1,0,0,0,1],[0,0,1,1,0,0]])
        D = query_control_basis(q)
        self.assertEqual(D.shape, (4,6,6))
        self.assertTrue(np.all(D[:,3:,:] == 0))
        self.assertEqual(np.linalg.matrix_rank(D.reshape(24,6)),6)

    def test_nonfinite_or_degenerate_controls_rejected(self):
        from research_math.native_probe import query_control_basis
        with self.assertRaises(ValueError): query_control_basis(np.zeros((4,6)))
        with self.assertRaises(ValueError): query_control_basis(np.full((4,6),np.nan))

    def test_native_source_hash_mismatch_rejected(self):
        from research_math.native_probe import verify_file
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_file(p, '0'*64)


if __name__ == '__main__': unittest.main()
