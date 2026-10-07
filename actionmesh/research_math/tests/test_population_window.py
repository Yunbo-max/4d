"""Reject omitted, reordered, or overbudget population windows before launch."""
import unittest
from research_math.population_window_plan import window_indices


class PopulationWindowTests(unittest.TestCase):
    def test_contiguous_window_and_last_partial_window(self):
        self.assertEqual(window_indices(128, 1, 9, 2700), list(range(1, 10)))
        self.assertEqual(window_indices(128, 125, 9, 2700), [125, 126, 127])

    def test_rejects_overbudget_and_invalid_windows(self):
        for args in ((128, 0, 11, 2700), (128, -1, 1, 2700),
                     (128, 128, 1, 2700), (128, 0, 0, 2700), (128, 0, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                window_indices(*args)
