from __future__ import annotations

import unittest

from geofem_app import fem2d_materials, fem2d_solver
from geofem_app.srm_reporting import (
    srm_fos_display,
    srm_fos_is_confirmed,
    srm_result_confidence,
    srm_result_status,
    srm_safety_verdict,
)


class SrmReportingTests(unittest.TestCase):
    def test_confirmed_bracket_may_report_ok(self) -> None:
        srm = {
            "factor_of_safety": 1.25,
            "stable_factor": 1.25,
            "failed_factor": 1.275,
            "factor_of_safety_status": "confirmed_bracket",
            "factor_of_safety_confidence": "high",
        }
        self.assertTrue(srm_fos_is_confirmed(srm))
        self.assertEqual(srm_safety_verdict(srm), "OK")
        self.assertEqual(srm_result_status(srm), "confirmed_bracket")
        self.assertIn("FOS=1.25", srm_fos_display(srm))

    def test_indeterminate_result_is_a_lower_bound_and_never_ok(self) -> None:
        srm = {
            "factor_of_safety": 1.25,
            "stable_factor": 1.25,
            "failed_factor": None,
            "factor_of_safety_status": "lower_bound_indeterminate",
            "factor_of_safety_confidence": "limited",
        }
        self.assertFalse(srm_fos_is_confirmed(srm))
        self.assertEqual(srm_safety_verdict(srm), "WARN")
        self.assertIn("FOS>=1.25", srm_fos_display(srm))
        self.assertIn("下限値", srm_fos_display(srm, locale="ja"))

    def test_unresolved_interval_is_presented_as_a_lower_bound(self) -> None:
        srm = {
            "factor_of_safety": 1.2,
            "stable_factor": 1.2,
            "failed_factor": 1.25,
            "factor_of_safety_status": "unresolved_indeterminate_interval",
            "factor_of_safety_confidence": "limited",
            "factor_of_safety_certified": False,
        }
        self.assertFalse(srm_fos_is_confirmed(srm))
        self.assertEqual(srm_safety_verdict(srm), "WARN")
        self.assertIn("FOS>=1.2", srm_fos_display(srm))
        self.assertIn("indeterminate trial", srm_fos_display(srm))

    def test_wide_confirmed_interval_is_not_presented_as_final_fos(self) -> None:
        srm = {
            "factor_of_safety": 1.2,
            "stable_factor": 1.2,
            "failed_factor": 1.4,
            "factor_of_safety_status": "bracket_tolerance_not_met",
            "factor_of_safety_confidence": "limited",
            "factor_of_safety_certified": False,
        }
        self.assertFalse(srm_fos_is_confirmed(srm))
        self.assertIn("FOS>=1.2", srm_fos_display(srm))
        self.assertIn("tolerance not met", srm_fos_display(srm))

    def test_nonmonotonic_result_is_not_presented_as_final(self) -> None:
        srm = {
            "factor_of_safety": 1.1,
            "stable_factor": 1.2,
            "failed_factor": 1.1,
            "factor_of_safety_status": "nonmonotonic_evidence",
            "factor_of_safety_confidence": "low",
        }
        self.assertEqual(srm_safety_verdict(srm), "WARN")
        self.assertIn("FOS~1.1", srm_fos_display(srm))

    def test_explicit_uncertified_interval_is_not_presented_as_final_fos(self) -> None:
        srm = {
            "factor_of_safety": 1.0,
            "stable_factor": 1.0,
            "failed_factor": 1.4,
            "factor_of_safety_status": "confirmed_bracket",
            "factor_of_safety_confidence": "limited",
            "factor_of_safety_certified": False,
        }
        self.assertFalse(srm_fos_is_confirmed(srm))
        self.assertEqual(srm_safety_verdict(srm), "WARN")
        self.assertIn("FOS>=1", srm_fos_display(srm))

    def test_legacy_summary_with_stable_and_failed_factors_is_confirmed(self) -> None:
        srm = {"factor_of_safety": 1.25, "stable_factor": 1.25, "failed_factor": 1.5}
        self.assertEqual(srm_result_status(srm), "confirmed_bracket")
        self.assertEqual(srm_result_confidence(srm), "high")
        self.assertEqual(srm_safety_verdict(srm), "OK")

    def test_material_fallback_telemetry_preserves_point_and_violation(self) -> None:
        fem2d_materials.reset_mohr_coulomb_fallback_telemetry()
        fem2d_materials._record_mohr_coulomb_fallback("numba_to_python", diagnostic_context=("E7", 2))
        fem2d_materials._record_mohr_coulomb_fallback(
            "regularized_projection",
            diagnostic_context=("E7", 2),
            yield_violation=0.25,
            relative_yield_violation=0.005,
            relaxed_tolerance=0.1,
        )
        telemetry = fem2d_materials.mohr_coulomb_fallback_telemetry()
        self.assertEqual(telemetry["numba_to_python_count"], 1)
        self.assertEqual(telemetry["regularized_projection_count"], 1)
        self.assertEqual(telemetry["regularized_projection_above_relaxed_tolerance_count"], 1)
        self.assertEqual(telemetry["samples"][0]["element_id"], "E7")
        self.assertEqual(telemetry["samples"][0]["integration_point"], 2)

    def test_verified_associated_apex_policy_preserves_configured_model_certification(self) -> None:
        info = {
            "factor_of_safety_status": "confirmed_bracket",
            "factor_of_safety_confidence": "high",
            "bracketed": True,
            "factor_of_safety_boundary_certified": True,
            "factor_of_safety_certified": True,
            "factor_of_safety_value_kind": "certified_stable_lower_bound",
            "boundary_quality": "verified_failure_boundary",
        }
        trials = [
            {
                "mc_numba_to_python_fallback_count": 2,
                "mc_numba_regularized_projection_count": 1,
                "mc_regularized_projection_count": 1,
                "mc_associated_apex_projection_count": 1,
                "mc_legacy_bounded_projection_count": 0,
                "mc_regularization_method": "associated_multisurface_apex",
                "mc_regularized_projection_above_relaxed_tolerance_count": 0,
                "mc_regularized_projection_max_yield_violation": 1.0e-8,
                "mc_regularized_projection_max_relative_yield_violation": 1.0e-10,
                "mc_regularized_projection_samples": [{"element_id": "E1", "integration_point": 0}],
            }
        ]
        fem2d_solver._srm_attach_material_fallback_search_info(info, trials)
        self.assertEqual(info["factor_of_safety_status"], "confirmed_bracket")
        self.assertEqual(info["factor_of_safety_confidence"], "high")
        self.assertTrue(info["bracketed"])
        self.assertFalse(info["material_fallback_verification_required"])
        self.assertTrue(info["material_fallback_within_tolerance"])
        self.assertTrue(info["factor_of_safety_boundary_certified"])
        self.assertTrue(info["factor_of_safety_certified"])
        self.assertEqual(
            info["factor_of_safety_certification_scope"],
            "mohr_coulomb_with_associated_multisurface_apex_policy",
        )
        self.assertTrue(info["mohr_coulomb_fallback"]["flow_rule_verified"])
        self.assertTrue(
            info["mohr_coulomb_fallback"]["constitutive_model_fidelity"]
        )
        self.assertFalse(
            info["mohr_coulomb_fallback"]["base_nonassociated_flow_rule_verified"]
        )
        self.assertEqual(
            info["mohr_coulomb_fallback"]["regularization_quality"],
            "yield_surface_verified",
        )
        self.assertEqual(info["mohr_coulomb_fallback"]["numba_regularized_projection_count"], 1)
        self.assertEqual(info["mohr_coulomb_fallback"]["regularized_projection_count"], 1)

    def test_material_fallback_above_tolerance_downgrades_srm_confidence(self) -> None:
        info = {
            "factor_of_safety_status": "confirmed_bracket",
            "factor_of_safety_confidence": "high",
            "bracketed": True,
            "factor_of_safety_boundary_certified": True,
            "factor_of_safety_certified": True,
            "factor_of_safety_value_kind": "certified_stable_lower_bound",
            "boundary_quality": "verified_failure_boundary",
        }
        trials = [
            {
                "mc_numba_regularized_projection_count": 1,
                "mc_regularized_projection_count": 1,
                "mc_associated_apex_projection_count": 1,
                "mc_regularization_method": "associated_multisurface_apex",
                "mc_regularized_projection_above_relaxed_tolerance_count": 1,
                "mc_regularized_projection_max_yield_violation": 1.0,
                "mc_regularized_projection_max_relative_yield_violation": 1.0e-3,
            }
        ]

        fem2d_solver._srm_attach_material_fallback_search_info(info, trials)

        self.assertEqual(info["factor_of_safety_status"], "material_fallback_evidence")
        self.assertEqual(info["factor_of_safety_confidence"], "limited")
        self.assertFalse(info["bracketed"])
        self.assertTrue(info["material_fallback_verification_required"])
        self.assertFalse(info["factor_of_safety_boundary_certified"])
        self.assertFalse(info["factor_of_safety_certified"])
        self.assertEqual(info["factor_of_safety_value_kind"], "verification_required")

    def test_legacy_bounded_apex_projection_requires_fos_verification(self) -> None:
        info = {
            "factor_of_safety_status": "confirmed_bracket",
            "factor_of_safety_confidence": "high",
            "bracketed": True,
            "factor_of_safety_boundary_certified": True,
            "factor_of_safety_certified": True,
            "factor_of_safety_value_kind": "certified_stable_lower_bound",
            "boundary_quality": "verified_failure_boundary",
        }
        trials = [
            {
                "mc_numba_regularized_projection_count": 4,
                "mc_regularized_projection_count": 4,
                "mc_associated_apex_projection_count": 0,
                "mc_legacy_bounded_projection_count": 4,
                "mc_regularization_method": "bounded_sequential_cone_tip",
                "mc_regularized_projection_above_relaxed_tolerance_count": 0,
                "mc_regularized_projection_max_relative_yield_violation": 1.0e-9,
            }
        ]

        fem2d_solver._srm_attach_material_fallback_search_info(info, trials)

        self.assertEqual(info["factor_of_safety_status"], "material_fallback_evidence")
        self.assertFalse(info["factor_of_safety_certified"])
        self.assertTrue(info["material_fallback_within_tolerance"])
        self.assertTrue(info["material_fallback_verification_required"])
        self.assertEqual(
            info["factor_of_safety_certification_scope"],
            "unverified_apex_projection_points",
        )


if __name__ == "__main__":
    unittest.main()
