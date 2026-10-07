"""Constructed controls, not natural 4D failure evidence."""
import importlib.util
import unittest

import numpy as np

MODULE_AVAILABLE = importlib.util.find_spec("research_math.contact_events") is not None
if MODULE_AVAILABLE:
    from research_math.contact_events import (
        ContactEvent, ContactEvidence, align_events, repair_trajectory,
        retime_event_times, retime_trajectory, retiming_baseline,
        solve_contact_events, run_demo,
    )


class ImplementationExists(unittest.TestCase):
    def test_contact_module_is_implemented(self):
        self.assertTrue(MODULE_AVAILABLE, "contact event prototype is not implemented yet")


@unittest.skipUnless(MODULE_AVAILABLE, "implementation pending")
class ContactEventControls(unittest.TestCase):
    def test_unknown_observation_does_not_become_noncontact(self):
        native = np.zeros((5, 1))
        events = [ContactEvent("A", 1, 3)]
        unknown = [ContactEvidence(None, 0, 4, state="unknown")]
        result = solve_contact_events(native, {"A": np.array([0.0])}, events,
                                      unknown, max_displacement=0.0)
        np.testing.assert_array_equal(result["trajectory"], native)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["alignment"][0].kind, "null")
        explicit = repair_trajectory(native, {"A": np.array([0.0])},
                                     [ContactEvidence("A", 1, 3, state="noncontact")],
                                     max_displacement=0.0, clearance=0.1)
        self.assertFalse(explicit["feasible"])
        self.assertEqual(explicit["status"], "certified_infeasible")

    def test_partner_and_contact_mode_are_not_interchangeable(self):
        native = [ContactEvent("A", 2, 2, mode="touch")]
        observations = [ContactEvidence("B", 2, 2, mode="touch")]
        best = align_events(native, observations)[0]
        self.assertEqual([op.kind for op in best.operations], ["replace"])
        self.assertEqual(best.cost, 1.0)
        changed_mode = align_events(native, [ContactEvidence("A", 2, 2, mode="grasp")])[0]
        self.assertEqual(changed_mode.operations[0].kind, "replace")

    def test_insert_delete_and_unknown_are_separate_operations(self):
        self.assertEqual(align_events([], [ContactEvidence("A", 1, 1)])[0]
                         .operations[0].kind, "insert")
        native = [ContactEvent("A", 1, 1)]
        deleted = align_events(native, [ContactEvidence("A", 1, 1, state="noncontact")])[0]
        self.assertEqual(deleted.operations[0].kind, "delete")
        unobserved = align_events(native, [])[0]
        self.assertEqual(unobserved.operations[0].kind, "null")

    def test_strict_monotone_retiming_preserves_event_order(self):
        events = [ContactEvent("A", 1, 1), ContactEvent("B", 3, 3)]
        mapped = retime_event_times(events, np.array([0., .5, 1.5, 3.5, 4.]))
        self.assertLess(mapped[0][1], mapped[1][0])
        self.assertAlmostEqual(mapped[0][0], 1.5)
        self.assertAlmostEqual(mapped[1][0], 2.75)
        with self.assertRaises(ValueError):
            retime_trajectory(np.arange(5.)[:, None], [0., 2., 1., 3., 4.])
        with self.assertRaises(ValueError):
            retime_trajectory(np.arange(5.)[:, None], [0., 1., 1., 3., 4.])

    def test_missing_contact_is_repaired_within_actual_l2_budget(self):
        native = np.zeros((7, 2))
        target = {"B": np.array([0.3, 0.4])}
        evidence = [ContactEvidence("B", 2, 4)]
        repaired = repair_trajectory(native, target, evidence, max_displacement=0.5)
        self.assertTrue(repaired["feasible"])
        np.testing.assert_allclose(repaired["trajectory"][2:5],
                                   np.array([[.3, .4], [.3, .4], [.3, .4]]), atol=1e-10)
        self.assertLessEqual(repaired["max_displacement"], 0.5 + 1e-10)
        impossible = repair_trajectory(native, target, evidence, max_displacement=0.49)
        self.assertFalse(impossible["feasible"])
        self.assertEqual(impossible["status"], "certified_infeasible")
        np.testing.assert_array_equal(native, np.zeros((7, 2)))

    def test_simultaneous_incompatible_contacts_are_infeasible(self):
        result = repair_trajectory(
            np.zeros((5, 1)), {"A": np.array([-1.]), "B": np.array([1.])},
            [ContactEvidence("A", 2, 2), ContactEvidence("B", 2, 2)],
            max_displacement=2.)
        self.assertEqual(result["status"], "certified_infeasible")
        self.assertFalse(result["feasible"])

    def test_feasible_timing_error_is_solved_by_retiming_baseline(self):
        native = np.array([0., 0., 1., 1., 0.])[:, None]
        result = retiming_baseline(native, {"A": np.array([1.])},
                                  [ContactEvidence("A", 1, 1)], max_displacement=1.,
                                  grid_size=33)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["trajectory"][1, 0], 1.)
        self.assertTrue(np.all(np.diff(result["sample_times"]) > 0))

    def test_event_repair_does_not_claim_advantage_over_same_information_solver(self):
        native = np.linspace(-1., 1., 7)[:, None]
        targets = {"A": np.array([-1.]), "B": np.array([1.])}
        evidence = [ContactEvidence("B", 1, 1), ContactEvidence("A", 5, 5)]
        repaired = solve_contact_events(
            native, targets, [ContactEvent("A", 0, 0), ContactEvent("B", 6, 6)],
            evidence, max_displacement=2.)
        direct = repair_trajectory(native, targets, evidence, max_displacement=2.)
        retimed = retiming_baseline(native, targets, evidence, max_displacement=2.)
        self.assertTrue(repaired["feasible"])
        self.assertFalse(retimed["feasible"])
        np.testing.assert_allclose(repaired["trajectory"], direct["trajectory"], atol=1e-10)
        self.assertEqual(repaired["continuous_comparator"], "same_information_direct_solver")

    def test_retiming_prioritizes_exact_contact_over_smaller_displacement(self):
        # A weighted motion tie-breaker used to prefer a 1e-6 contact miss,
        # although [0, 3, 3.25, 3.5, 4] is a feasible clock on this grid.
        native = np.array([0., 1.-1e-6, 0., 1., 2.])[:, None]
        result = retiming_baseline(native, {"A": np.array([1.])},
                                  [ContactEvidence("A", 1, 1)],
                                  max_displacement=2., grid_size=17)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["max_contact_error"], 0.)
        self.assertEqual(result["sample_times"][1], 3.)
        self.assertTrue(np.all(np.diff(result["sample_times"]) > 0))
        self.assertLessEqual(result["max_displacement"], 2.)
        self.assertFalse(result["continuous_impossibility_claim"])

    def test_bounded_search_reports_candidate_cap(self):
        result = solve_contact_events(
            np.zeros((5, 1)), {"A": np.array([0.])},
            [ContactEvent("A", 1, 1)], [ContactEvidence("A", 2, 2)],
            max_displacement=0., max_candidates=2)
        self.assertLessEqual(result["candidate_count"], 2)
        self.assertFalse(result["global_joint_optimality_claim"])

    def test_invalid_or_nonfinite_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            repair_trajectory(np.array([[np.nan], [0.]]), {"A": np.array([1.])},
                              [], max_displacement=1.)
        with self.assertRaises(ValueError):
            repair_trajectory(np.zeros((2, 1)), {"A": np.array([1.])},
                              [ContactEvidence("A", 0, 2)], max_displacement=1.)
        with self.assertRaises(ValueError):
            repair_trajectory(np.zeros((2, 1)), {"A": np.array([1.])},
                              [], max_displacement=-1.)

    def test_demo_has_success_failure_and_no_natural_evidence_claim(self):
        result = run_demo()
        self.assertEqual(result["evidence_kind"], "constructed_toy_controls")
        self.assertTrue(result["controls"]["wrong_partner"]["repair_feasible"])
        self.assertFalse(result["controls"]["budget_impossible"]["repair_feasible"])
        self.assertTrue(result["controls"]["timing_only"]["retiming_feasible"])
        self.assertTrue(result["direct_solver_matches_all_repairs"])


if __name__ == "__main__":
    unittest.main()
