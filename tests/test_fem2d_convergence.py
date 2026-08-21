from __future__ import annotations

import math
import unittest

import numpy as np

import geofem_app.fem2d_solver as solver
from geofem_app.fem2d_convergence import (
    dynamic_residual_metrics,
    newton_convergence_metrics,
    newton_convergence_with_force_norm,
    riks_convergence_metrics,
)


class FEM2DConvergenceTests(unittest.TestCase):
    def test_solver_uses_split_convergence_module(self) -> None:
        self.assertIs(solver._riks_convergence_metrics, riks_convergence_metrics)
        self.assertIs(solver._dynamic_residual_metrics, dynamic_residual_metrics)
        self.assertIs(solver._newton_convergence_metrics, newton_convergence_metrics)
        self.assertIs(solver._newton_convergence_with_force_norm, newton_convergence_with_force_norm)

    def test_riks_uses_each_residual_group_reference_without_unit_floor(self) -> None:
        metrics = riks_convergence_metrics(
            force_norm=2.0e-10,
            force_reference=1.0e-6,
            pressure_norm=2.0e-5,
            pressure_reference=1.0,
            pressure_enabled=True,
            arc_residual=2.0e-8,
            arc_reference=1.0e-2,
            mpc_norm=2.0e-10,
            mpc_reference=1.0e-6,
            mpc_enabled=True,
            riks_cfg={"tol_rel": 1.0e-4, "tol_abs": 0.0},
            legacy_tol=1.0e-4,
        )
        self.assertAlmostEqual(metrics["force_tolerance"], 1.0e-10)
        self.assertFalse(metrics["force_converged"])
        self.assertTrue(metrics["pressure_converged"])
        self.assertTrue(metrics["arc_converged"])
        self.assertFalse(metrics["mpc_converged"])
        self.assertFalse(metrics["converged"])

    def test_dynamic_pressure_failure_cannot_be_hidden_by_force_convergence(self) -> None:
        metrics = dynamic_residual_metrics(
            force_norm=1.0e-8,
            force_reference=1.0,
            pressure_norm=1.0e-2,
            pressure_reference=1.0,
            pressure_enabled=True,
            settings={"tol_rel": 1.0e-6, "tol_abs": 0.0, "tol_pressure_rel": 1.0e-4, "tol_pressure_abs": 0.0},
        )
        self.assertTrue(metrics["force_converged"])
        self.assertFalse(metrics["pressure_converged"])
        self.assertFalse(metrics["converged"])
        self.assertTrue(math.isfinite(metrics["normalized_residual_merit"]))

    def test_newton_force_override_recomputes_combined_decision(self) -> None:
        settings = {
            "tol_rel": 1.0e-3,
            "tol_abs": 0.0,
            "tol_displacement_rel": 1.0,
            "tol_displacement_abs": 0.0,
            "tol_energy_rel": 1.0,
            "tol_energy_abs": 0.0,
            "mixed_convergence": True,
            "strict_force_bypass_ratio": 0.5,
        }
        metrics = newton_convergence_metrics(
            residual_free=np.array([1.0e-4]),
            external_free=np.array([1.0]),
            displacement_free=np.array([1.0]),
            previous_update_free=np.array([1.0e-5]),
            has_previous_update=True,
            constraint_norm=0.0,
            settings=settings,
        )
        self.assertTrue(metrics["converged"])
        updated = newton_convergence_with_force_norm(metrics, 2.0e-3, settings)
        self.assertFalse(updated["force_converged"])
        self.assertFalse(updated["converged"])


if __name__ == "__main__":
    unittest.main()
