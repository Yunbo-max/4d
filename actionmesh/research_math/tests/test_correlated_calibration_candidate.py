"""Local-only C03 candidate export and retained-artifact acceptance checks."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from research_math import correlated_calibration_candidate as c03


class C03CandidateBoundaryTests(unittest.TestCase):
    def test_bounds_reports_without_clipping(self):
        value = np.asarray([[[-2.0, 0.0, 3.0]]], dtype=np.float32)
        report = c03._bounds(value, -1.0, 1.0, "preserve_and_report")
        self.assertEqual((report["below"], report["above"]), (1, 1))
        self.assertFalse(report["clipped"])
        np.testing.assert_array_equal(value, [[[-2.0, 0.0, 3.0]]])

    def test_bounds_reject_policy_fails_closed(self):
        with self.assertRaises(ValueError):
            c03._bounds(np.asarray([[[2.0, 0.0, 0.0]]]), -1.0, 1.0, "reject")


class C03RetainedNativeAcceptance(unittest.TestCase):
    """Run only through the zero-GPU Local acceptance plan."""

    def test_retained_artifact_recomputes_all_declared_roles(self):
        root = os.environ.get("C03_NATIVE_ROOT")
        artifact = os.environ.get("C03_NATIVE_ARTIFACT")
        if not root or not artifact:
            self.skipTest("C03 retained-artifact environment is Local-only")
        result = c03.validate_candidate_artifact(Path(root), Path(artifact))
        self.assertEqual(result["candidate_id"], c03.CANDIDATE_ID)
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
