import unittest

from geofem_app.cad_constraints import solve_cad_constraints


class CADConstraintSolverTests(unittest.TestCase):
    def test_length_and_horizontal_constraints_move_free_point(self) -> None:
        result = solve_cad_constraints(
            {
                "p0": {"x": 0.0, "y": 0.0, "locked": True},
                "p1": {"x": 2.0, "y": 0.2},
            },
            [
                {"id": "L1", "type": "length", "p1": "p0", "p2": "p1", "value": 2.5},
                {"id": "H1", "type": "horizontal", "p1": "p0", "p2": "p1"},
            ],
            tolerance=1.0e-9,
        )
        self.assertEqual(result["diagnostics"]["status"], "solved")
        self.assertFalse(result["diagnostics"]["inconsistent"])
        self.assertAlmostEqual(result["points"]["p1"][0], 2.5, places=7)
        self.assertAlmostEqual(result["points"]["p1"][1], 0.0, places=7)

    def test_duplicate_constraints_are_reported_as_redundant(self) -> None:
        result = solve_cad_constraints(
            {"p0": {"x": 0.0, "y": 0.0, "locked": True}, "p1": {"x": 1.0, "y": 0.0}},
            [
                {"id": "L1", "type": "length", "p1": "p0", "p2": "p1", "value": 1.0},
                {"id": "L2", "type": "length", "p1": "p0", "p2": "p1", "value": 1.0},
            ],
            tolerance=1.0e-9,
        )
        self.assertEqual(result["diagnostics"]["status"], "solved")
        self.assertGreaterEqual(result["diagnostics"]["redundant_count"], 1)
        self.assertFalse(result["diagnostics"]["inconsistent"])

    def test_conflicting_constraints_are_reported_as_inconsistent(self) -> None:
        result = solve_cad_constraints(
            {"p0": {"x": 0.0, "y": 0.0, "locked": True}, "p1": {"x": 1.0, "y": 0.0}},
            [
                {"id": "L1", "type": "length", "p1": "p0", "p2": "p1", "value": 1.0},
                {"id": "L2", "type": "length", "p1": "p0", "p2": "p1", "value": 2.0},
            ],
            tolerance=1.0e-9,
        )
        self.assertEqual(result["diagnostics"]["status"], "inconsistent")
        self.assertTrue(result["diagnostics"]["inconsistent"])
        self.assertGreater(result["diagnostics"]["max_abs_residual"], 1.0e-3)

    def test_tangent_constraint_aligns_segment_to_reference(self) -> None:
        result = solve_cad_constraints(
            {
                "a0": {"x": 0.0, "y": 0.0, "locked": True},
                "a1": {"x": 1.0, "y": 0.4},
                "b0": {"x": 0.0, "y": 1.0, "locked": True},
                "b1": {"x": 2.0, "y": 1.0, "locked": True},
            },
            [
                {"id": "A_len", "type": "length", "p1": "a0", "p2": "a1", "value": 1.0},
                {"id": "G1", "type": "tangent", "p1": "a0", "p2": "a1", "reference_p1": "b0", "reference_p2": "b1"},
            ],
            tolerance=1.0e-9,
        )
        self.assertEqual(result["diagnostics"]["status"], "solved")
        self.assertAlmostEqual(result["points"]["a1"][0], 1.0, places=7)
        self.assertAlmostEqual(result["points"]["a1"][1], 0.0, places=7)

    def test_concentric_constraint_coincides_centers(self) -> None:
        result = solve_cad_constraints(
            {
                "c1": {"x": 0.0, "y": 0.0, "locked": True},
                "c2": {"x": 0.2, "y": -0.3},
            },
            [{"id": "C0", "type": "concentric", "p1": "c1", "p2": "c2"}],
            tolerance=1.0e-9,
        )
        self.assertEqual(result["diagnostics"]["status"], "solved")
        self.assertAlmostEqual(result["points"]["c2"][0], 0.0, places=7)
        self.assertAlmostEqual(result["points"]["c2"][1], 0.0, places=7)

    def test_curvature_continuity_moves_third_control_point(self) -> None:
        result = solve_cad_constraints(
            {
                "p1": {"x": 0.0, "y": 0.0, "locked": True},
                "p2": {"x": 1.0, "y": 0.0, "locked": True},
                "p3": {"x": 2.0, "y": 1.0, "locked": True},
                "q1": {"x": 0.0, "y": 0.0, "locked": True},
                "q2": {"x": 1.0, "y": 0.0, "locked": True},
                "q3": {"x": 2.0, "y": 1.4},
            },
            [
                {
                    "id": "G2",
                    "type": "curvature_continuity",
                    "p1": "p1",
                    "p2": "p2",
                    "p3": "p3",
                    "reference_p1": "q1",
                    "reference_p2": "q2",
                    "reference_p3": "q3",
                }
            ],
            tolerance=1.0e-9,
        )
        self.assertEqual(result["diagnostics"]["status"], "solved")
        self.assertAlmostEqual(result["points"]["q3"][0], 2.0, places=6)
        self.assertAlmostEqual(result["points"]["q3"][1], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
