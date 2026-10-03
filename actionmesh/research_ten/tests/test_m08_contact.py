"""Tests explicitly concern sampled linear contact constraints, not CCD."""
import importlib
import unittest
from unittest.mock import patch

import numpy as np


def api():
    try:
        return importlib.import_module("research_ten.m08_contact")
    except ModuleNotFoundError:
        raise AssertionError("bounded contact repair is missing")


class ContactTests(unittest.TestCase):
    def test_contact_projection_moves_only_normal_and_preserves_other_vertices(self):
        m = api()
        points = np.array([[2., 3, -.2], [0, 0, 0], [8, 9, 10]])
        c = m.ContactConstraint((0, 1), (1., -1.), (0., 0., 1.), .1)
        result = m.repair_contacts(points, [c], max_displacement=.5)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.vertices[:, :2], points[:, :2], atol=0)
        np.testing.assert_array_equal(result.vertices[2], points[2])
        np.testing.assert_allclose(result.vertices[:2, 2], [-.05, -.15], atol=1e-8)
        self.assertLess(result.final_violation, 1e-8)

    def test_infeasible_displacement_bound_is_reported_and_pins_hold(self):
        m = api()
        points = np.array([[0., 0, -.2], [0, 0, 0]])
        c = m.ContactConstraint((0, 1), (1., -1.), (0., 0., 1.), .1)
        result = m.repair_contacts(points, [c], max_displacement=.05, fixed_vertices=[1])
        self.assertFalse(result.converged)
        self.assertGreater(result.final_violation, .2)
        self.assertLessEqual(np.linalg.norm(result.displacement, axis=1).max(), .050000001)
        np.testing.assert_array_equal(result.vertices[1], points[1])

    def test_coupled_halfspaces_converge_and_standard_control_is_feasible(self):
        m = api()
        points = np.zeros((3, 3))
        cs = [m.ContactConstraint((0, 1), (1., -1.), (0., 0., 1.), 1.),
              m.ContactConstraint((1, 2), (1., -1.), (0., 0., 1.), 1.)]
        result = m.repair_contacts(points, cs, max_displacement=2.)
        np.testing.assert_allclose(result.vertices[:, 2], [1, 0, -1], atol=1e-6)
        self.assertTrue(result.converged)
        baseline = m.project_contact_constraints(points, cs)
        self.assertLess(baseline.final_violation, 1e-7)

    def test_no_contacts_is_exact_identity_and_trajectory_anchor_is_exact(self):
        m = api()
        points = np.array([[0., 0, -.2], [0, 0, 0]])
        np.testing.assert_array_equal(m.repair_contacts(points, []).vertices, points)
        c = m.ContactConstraint((0, 1), (1., -1.), (0., 0., 1.), .1)
        repaired, reports = m.repair_contact_trajectory(np.stack([points, points]), [[c], [c]],
                                                        max_displacement=.5, anchor_frame=0)
        np.testing.assert_array_equal(repaired[0], points)
        self.assertLess(reports[1].final_violation, 1e-8)
        self.assertFalse(reports[0].converged)

    def test_geometry_detection_uses_reference_side_and_excludes_adjacent_vertices(self):
        m = api()
        reference = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [.25, .25, .1]])
        current = reference.copy()
        current[3, 2] = -.02
        found = m.detect_vertex_triangle_contacts(reference, current, [[0, 1, 2]],
                                                    clearance=.01, search_radius=.1)
        self.assertEqual(len(found.contacts), 1)
        c = found.contacts[0]
        self.assertEqual(c.indices[0], 3)
        fixed = m.repair_contacts(current, found.contacts, max_displacement=.1, fixed_vertices=[0, 1, 2])
        self.assertAlmostEqual(fixed.vertices[3, 2], .01, places=7)

    def test_dykstra_minimizes_change_when_plain_projection_overshoots(self):
        m = api()
        points = np.zeros((2, 3))
        contacts = [m.ContactConstraint((0, 1), (1., -1.), (1., 0., 0.), 1.),
                    m.ContactConstraint((0, 1), (1., -1.), (1., 1., 0.), 2.)]
        repaired = m.repair_contacts(points, contacts, fixed_vertices=[1], max_displacement=3.)
        standard = m.project_contact_constraints(points, contacts, fixed_vertices=[1])
        np.testing.assert_allclose(repaired.vertices[0], [np.sqrt(2), np.sqrt(2), 0], atol=1e-6)
        self.assertLess(np.linalg.norm(repaired.displacement), np.linalg.norm(standard.displacement) - .05)

    def test_invalid_constraint_and_fractional_pin_ids_are_rejected(self):
        m = api()
        points = np.zeros((2, 3))
        for contact in [m.ContactConstraint((0, 1), (1., 1.), (0., 0., 1.)),
                        m.ContactConstraint((0, 1), (1., -1.), (0., 0., 0.))]:
            with self.assertRaises(ValueError):
                m.repair_contacts(points, [contact])
        with self.assertRaises(ValueError):
            m.repair_contacts(points, [], fixed_vertices=[.5])

    def test_large_hash_box_falls_back_without_cartesian_enumeration(self):
        m = api()
        reference = np.array([[0., 0, 0], [1, 1, 0], [0, 0, 1], [.2000001, .2, .2]])
        current = reference.copy()
        current[3, 0] = .19999999
        # The incorrect int64 product wraps negative for this finite geometry.
        # Guard the dangerous enumerator; assert actual detection still works.
        with patch.object(m, "product", side_effect=AssertionError("unsafe Cartesian enumeration")):
            result = m.detect_vertex_triangle_contacts(reference, current, [[0, 1, 2]],
                                                        search_radius=2e-7, clearance=0.)
        self.assertEqual(len(result.contacts), 1)

    def test_unrepresentable_hash_coordinates_use_finite_aabb_scan(self):
        m = api()
        reference = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [.25, .25, .1]])
        current = reference.copy()
        current[3, 2] = -1e-101
        with patch.object(m, "product", side_effect=AssertionError("unsafe Cartesian enumeration")):
            result = m.detect_vertex_triangle_contacts(reference, current, [[0, 1, 2]],
                                                        search_radius=1e-100, clearance=0.)
        self.assertEqual(len(result.contacts), 1)


if __name__ == "__main__":
    unittest.main()
