"""Generated plan-builder contract for C12; Local executes this test."""
from __future__ import annotations

import inspect
import unittest

import prepare_area_admission_candidate as builder


class AreaAdmissionPlanTests(unittest.TestCase):
    def test_builder_is_cpu_only_zero_retry_and_has_no_execution_call(self):
        source = inspect.getsource(builder)
        self.assertIn('"gpu_count": 0', source)
        self.assertIn('"max_retries_per_trial": 0', source)
        self.assertIn('"max_confirmation_trials": 0', source)
        self.assertIn('"execution_started": False', source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("run_harness.main", source)


if __name__ == "__main__":
    unittest.main()
