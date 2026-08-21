from __future__ import annotations

import csv
import io
import json
import tempfile
import threading
import time
import unittest
import math
import os
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np
import yaml
from scipy.sparse import csr_matrix

import geofem_app.fem2d_solver as fem2d_solver_module
import geofem_app.fem2d_linear_solver as linear_solver_core
import geofem_app.fem2d_materials as fem2d_materials_module
import geofem_app.fem2d_plastic_batch as fem2d_plastic_batch_module
from geofem_app.fem2d_nonlinear_assembly import (
    assemble_internal_force_candidates,
)
from geofem_app.cli import _apply_cli_solver_runtime_defaults, run_solve
from geofem_app.fem2d_types import Element2D, Interface2D, Mesh2D, StructuralElement2D
from geofem_app.sparse_assembly import (
    SparseAssemblyBuilder,
    reset_sparse_assembly_diagnostics,
    set_sparse_assembly_diagnostics_enabled,
    sparse_assembly_diagnostics,
)
from geofem_app.fem2d import (
    ElasticPlaneStrainMaterial,
    FEM2DError,
    PlasticState2D,
    PlasticStateView2D,
    PUBLIC_PROFILE,
    _NUMBA_AVAILABLE,
    _ADV_STATE_EFFECTIVE_E,
    _ADV_STATE_GAMMA_EQ,
    _ADV_STATE_MODULUS_RATIO,
    _ADV_STATE_RU,
    _advanced_effective_material,
    _advanced_history_array,
    _advanced_history_state,
    _apply_tension_cutoff,
    _j2dp_tension_cutoff_numerical_tangent_numba,
    _quad4_advanced_elastic_bbar_post_fast,
    _quad4_advanced_elastic_post_fast,
    _quad4_advanced_elastic_tension_bbar_post_fast,
    _quad4_advanced_elastic_tension_post_fast,
    _quad4_advanced_strength_j2dp_bbar_post_fast,
    _quad4_advanced_strength_j2dp_post_fast,
    _quad4_advanced_strength_j2dp_tangent_force_fast,
    _quad4_advanced_strength_mc_tangent_force_fast,
    _quad8_advanced_elastic_bbar_post_fast,
    _quad8_advanced_elastic_post_fast,
    _quad8_advanced_elastic_tension_bbar_post_fast,
    _quad8_advanced_elastic_tension_post_fast,
    _quad8_advanced_strength_j2dp_bbar_post_fast,
    _quad8_advanced_strength_j2dp_post_fast,
    _quad8_advanced_strength_j2dp_tangent_force_fast,
    _quad8_advanced_strength_mc_bbar_post_fast,
    _quad8_advanced_strength_mc_post_fast,
    _quad8_advanced_strength_mc_tangent_force_fast,
    _quad8_biot_matrix_fast,
    _quad8_axisymmetric_biot_matrix_fast,
    _quad8_axisymmetric_edge_traction_fast,
    _quad8_axisymmetric_element_stiffness_fast,
    _quad8_axisymmetric_internal_force_elastic_fast,
    _quad8_axisymmetric_j2dp_bbar_post_fast,
    _quad8_axisymmetric_j2dp_post_fast,
    _quad8_axisymmetric_j2dp_tangent_force_fast,
    _quad8_axisymmetric_pressure_matrices_fast,
    _quad8_consistent_mass_matrix_fast,
    _quad8_elastic_bbar_post_fast,
    _quad8_elastic_post_fast,
    _quad8_elastic_tension_bbar_post_fast,
    _quad8_elastic_tension_post_fast,
    _quad8_elastic_tension_tangent_force_fast,
    _quad8_element_stiffness_fast,
    _quad8_internal_force_elastic_fast,
    _quad8_j2dp_bbar_post_fast,
    _quad8_j2dp_post_fast,
    _quad8_j2dp_tangent_force_fast,
    _quad8_mc_bbar_post_fast,
    _quad8_mc_internal_force_fast,
    _quad8_mc_post_fast,
    _quad8_mc_tangent_force_fast,
    _quad8_pressure_matrices_fast,
    _mc_consistent_tangent_spectral_numba,
    _mc_tension_cutoff_consistent_tangent_numba,
    _mc_tension_cutoff_numerical_tangent_numba,
    _mc_plane_coeffs,
    _mc_reduced_parameters,
    _mc_return_mapping_principal_numba,
    _mc_return_mapping_principal_python,
    _quad4_j2dp_tangent_force_fast,
    _quad4_mc_internal_force_fast,
    _quad4_mc_tangent_force_fast,
    _mc_yield_tol,
    _yield_surface_parameters,
    algorithmic_material_tangent,
    assemble_algorithmic_tangent_stiffness,
    assemble_biot_coupling_matrix,
    assemble_biot_coupling_matrix_cached,
    assemble_axisymmetric_biot_coupling_matrix,
    assemble_axisymmetric_algorithmic_tangent_stiffness,
    assemble_axisymmetric_internal_force,
    assemble_axisymmetric_tangent_and_internal_force,
    assemble_axisymmetric_load_vector,
    assemble_axisymmetric_pressure_boundary_terms,
    assemble_axisymmetric_pressure_matrices,
    assemble_axisymmetric_stiffness,
    assemble_internal_force,
    assemble_tangent_and_internal_force,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_mass_matrix_cached,
    assemble_pressure_boundary_terms,
    assemble_pressure_boundary_terms_cached,
    assemble_pressure_matrices,
    assemble_pressure_matrices_cached,
    assemble_pore_pressure_load,
    assemble_pore_pressure_load_cached,
    axisymmetric_element_stiffness,
    axisymmetric_strain_displacement_matrix,
    clear_linear_factor_cache,
    clear_material_strength_parameter_array_cache,
    clear_strength_parameter_cache,
    compare_external_seepage_roundtrip,
    compare_external_seepage_results,
    compare_geofeas_dynamic_sample,
    compare_geofeas_reference_csv,
    compare_geofeas_stage_package,
    compute_integration_point_results,
    compute_axisymmetric_integration_point_results,
    compute_element_results_and_state,
    diagnose_external_seepage_version,
    estimate_joint_mohr_coulomb_parameters,
    element_stiffness,
    import_external_seepage_results,
    integration_points,
    load_combination_template_manifest,
    mesh_from_config,
    validate_mesh_quality_for_solve,
    numerical_material_tangent,
    plane_strain_materials,
    principal_stresses,
    shape_functions,
    linear_factor_cache_info,
    material_strength_parameter_array_cache_info,
    build_large_deformation_step_cache,
    build_mass_matrix_assembly_cache,
    build_dynamic_mass_step_cache,
    build_biot_coupling_assembly_cache,
    build_small_deformation_step_cache,
    build_initial_stress_array_cache,
    build_pore_pressure_load_cache,
    build_pressure_boundary_term_cache,
    build_pressure_matrix_assembly_cache,
    initial_stress_array_cache_info,
    build_plastic_state_array_cache,
    build_reduced_matrix_cache_from_csr,
    fill_updated_coords,
    large_deformation_kernel_contract,
    mesh_with_updated_coords,
    solve_plane_strain_config,
    solve_plane_strain_stage,
    solve_linear_system,
    solve_reduced_linear_system,
    strain_displacement_matrix,
    plastic_state_array_cache_info,
    strength_parameter_cache_info,
    SparseAssemblyPattern,
    update_plane_strain_stress,
    run_geofeas_benchmark_suite,
    write_geofeas_tolerance_report,
    write_joint_standard_report,
    write_load_combination_template_manifest,
)
from geofem_app.fem2d_structural_assembly import (
    assemble_axisymmetric_stiffness_cached,
    assemble_global_stiffness,
    assemble_global_stiffness_cached,
    assemble_load_vector_cached,
    build_axisymmetric_stiffness_assembly_cache,
    build_global_stiffness_assembly_cache,
    build_load_vector_assembly_cache,
)
from geofem_app.fem2d_mpc import mpc_arc_length_stage_plan, mpc_constraint_matrix, mpc_elimination_requested, mpc_lagrange_requested, mpc_stage_plan
from geofem_app.fem2d_hydro_iteration import SeepageActiveSetState, advance_seepage_active_set, observe_seepage_active_set, seepage_outer_iteration_limit
from geofem_app.fem2d_element_interpolation import element_interpolation_contract
from geofem_app.fem2d_element_elastic_kernels import (
    _quad4_element_stiffness_fast as elastic_kernel_module_quad4_stiffness_fast,
    elastic_element_kernel_contract,
)
from geofem_app.fem2d_element_elastic_post import element_elastic_post_contract, _quad8_elastic_post_fast as elastic_post_module_quad8_fast
from geofem_app.fem2d_element_fast_paths import element_fast_path_contract, _quad4_elastic_post_fast_path, _quad4_post_state_arrays
from geofem_app.fem2d_element_j2dp_kernels import (
    _quad8_j2dp_tangent_force_fast as j2dp_module_quad8_tangent_force_fast,
    j2dp_element_kernel_contract,
)
from geofem_app.fem2d_element_numba_primitives import (
    _quad4_b_det_numba as primitive_quad4_b_det_numba,
    _quad8_gp_full as primitive_quad8_gp_full,
    element_numba_primitives_contract,
)
from geofem_app.fem2d_element_tension_cutoff_kernels import (
    _quad8_elastic_tension_tangent_force_fast as tension_cutoff_module_quad8_fast,
    tension_cutoff_element_kernel_contract,
)
from geofem_app.fem2d_element_post_processing import (
    compute_integration_point_results as compute_integration_point_results_impl,
    element_post_processing_contract,
)
from geofem_app.fem2d_element_result_rows import element_result_row_contract, _inactive_element_result, _quad4_elastic_post_result_rows
from geofem_app.fem2d_solver_progress import (
    stage_boundary_conditions,
    stage_display_name,
    stage_loads,
    stage_mpc_constraints,
    stage_sequence_from_config,
    stage_solver_config,
    stage_state_after_result,
    stage_time,
    stage_type,
)
from geofem_app.fem2d_solver import (
    _SRMTopologyDiagnosticsCache,
    _srm_evaluate_records_parallel,
    _run_srm_trial_search,
    _srm_build_coarse_mesh,
    _srm_early_failure_cutback_decision,
    _srm_final_factors_from_coarse,
    _srm_numeric_thread_context,
    _srm_parallel_settings,
    _srm_selected_factor,
    _srm_solver_with_retry_override,
)
from geofem_app.fem2d_solver_controls import increment_settings, scale_loads
from geofem_app.fem2d_nonlinear_assembly import _quad4_sri_bbar_plastic_batch_blocks
from geofem_app.fem2d_performance_contract import deformation_mode_from_config, large_deformation_performance_contract
from geofem_app.fem2d_io import write_deferred_run_artifacts, write_deferred_run_artifacts_from_files
from geofem_app.numba_warmup import NumbaWarmupKernel, _representative_numba_kernels, clear_numba_warmup_state, gui_numba_warmup_enabled, warmup_numba_kernels
from geofem_app.fem2d_plastic_batch import (
    MohrCoulombActiveSetCache,
    _quad4_generic_tangent_force_state,
    build_plastic_element_blocks,
    empty_up_coupling_block_result,
    evaluate_plastic_tangent_block,
    evaluate_up_coupling_block,
    plastic_batch_contract,
)
from geofem_app.analysis_log import build_structured_analysis_log
from geofem_app.fem2d_types import SolveResult2D, StageResult2D
from geofem_app.result_viewer import build_result_view_index
from geofem_app.samples import plane_strain_patch_sample, plane_strain_quad4_sample


class NumbaAccelerationTests(unittest.TestCase):
    def test_numba_is_required_for_2d_core(self) -> None:
        self.assertTrue(_NUMBA_AVAILABLE)

    def test_numba_warmup_reports_times_and_skips_already_warmed_profiles(self) -> None:
        clear_numba_warmup_state()
        calls: list[str] = []
        kernels = (
            NumbaWarmupKernel("dummy_a", lambda: calls.append("a"), ("unit",)),
            NumbaWarmupKernel("dummy_b", lambda: calls.append("b"), ("unit",)),
        )
        first = warmup_numba_kernels(profile="unit", kernels=kernels)
        second = warmup_numba_kernels(profile="unit", kernels=kernels)

        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(first["schema"], "geofem.numba_warmup.v1")
        self.assertEqual(first["kernel_count"], 2)
        self.assertEqual(first["warmed_count"], 2)
        self.assertEqual(first["failed_count"], 0)
        self.assertGreaterEqual(first["elapsed_seconds"], 0.0)
        self.assertFalse(first["already_warmed"])
        self.assertTrue(second["already_warmed"])
        self.assertTrue(second["cached"])
        self.assertEqual(second["elapsed_seconds"], 0.0)
        self.assertFalse(gui_numba_warmup_enabled({"QT_QPA_PLATFORM": "offscreen"}))
        self.assertTrue(gui_numba_warmup_enabled({"QT_QPA_PLATFORM": "offscreen", "GEOFEM_GUI_NUMBA_WARMUP": "1"}))

    def test_representative_numba_warmup_includes_tri_pressure_biot_and_mass(self) -> None:
        names = {kernel.name for kernel in _representative_numba_kernels()}
        for name in (
            "tri3_pressure_matrices",
            "tri6_pressure_matrices",
            "tri3_biot_matrix",
            "tri6_biot_matrix",
            "tri3_consistent_mass",
            "tri6_consistent_mass",
            "axisymmetric_tri3_pressure_matrices",
            "axisymmetric_tri6_biot_matrix",
            "quad4_j2dp_post",
            "quad8_j2dp_post_full",
            "quad4_mohr_coulomb_post",
            "quad4_mohr_coulomb_regularized_projection",
            "quad8_mohr_coulomb_post",
            "quad8_elastic_post_full",
        ):
            self.assertIn(name, names)

    def test_plane_strain_principal_stress_and_tension_cutoff_match_dense_eigensolve(self) -> None:
        stress = np.array([10.0, 2.0, 6.0, 3.0], dtype=float)
        tensor = np.array([[stress[0], stress[3], 0.0], [stress[3], stress[1], 0.0], [0.0, 0.0, stress[2]]], dtype=float)
        self.assertTrue(np.allclose(principal_stresses(stress), np.linalg.eigvalsh(tensor)[::-1], rtol=1.0e-12, atol=1.0e-12))

        material = ElasticPlaneStrainMaterial("soil", 10000.0, 0.30, tension_cutoff=True, tensile_strength=5.0)
        updated, clipped, excess = _apply_tension_cutoff(stress, material)
        vals, vecs = np.linalg.eigh(tensor)
        expected_excess = float(np.max(vals) - material.tensile_strength)
        expected_tensor = vecs @ np.diag(np.minimum(vals, material.tensile_strength)) @ vecs.T
        expected = np.array([expected_tensor[0, 0], expected_tensor[1, 1], expected_tensor[2, 2], expected_tensor[0, 1]], dtype=float)
        self.assertTrue(clipped)
        self.assertAlmostEqual(excess, expected_excess, places=12)
        self.assertTrue(np.allclose(updated, expected, rtol=1.0e-12, atol=1.0e-12))

    def test_tension_cutoff_numerical_tangent_uses_batched_kernel(self) -> None:
        material = ElasticPlaneStrainMaterial("soil", 10000.0, 0.30, tension_cutoff=True, tensile_strength=5.0)
        strain = np.array([0.0015, -0.0001, 0.0003, 0.0008], dtype=float)
        tangent = numerical_material_tangent(material, strain)

        base = update_plane_strain_stress(material, strain).stress
        delta = 1.0e-8 * max(1.0, float(np.linalg.norm(strain)))
        expected = np.zeros((4, 4), dtype=float)
        for i in range(4):
            perturbed = strain.copy()
            perturbed[i] += delta
            expected[:, i] = (update_plane_strain_stress(material, perturbed).stress - base) / delta
        self.assertTrue(np.allclose(tangent, expected, rtol=1.0e-12, atol=1.0e-9))

    def test_strength_parameter_cache_reuses_material_factor_parameters(self) -> None:
        clear_strength_parameter_cache()
        material = ElasticPlaneStrainMaterial(
            "soil",
            50000.0,
            0.30,
            model="mohr_coulomb",
            cohesion=25.0,
            friction_angle=30.0,
            dilation_angle=10.0,
        )

        first_surface = _yield_surface_parameters(material, 1.25)
        first_mc = _mc_reduced_parameters(material, 1.25)
        first_info = strength_parameter_cache_info()
        second_surface = _yield_surface_parameters(material, 1.25)
        second_mc = _mc_reduced_parameters(material, 1.25)
        second_info = strength_parameter_cache_info()

        self.assertEqual(first_surface, second_surface)
        self.assertEqual(first_mc, second_mc)
        self.assertEqual(first_info["yield_surface"]["misses"], 1)
        self.assertEqual(first_info["mohr_coulomb"]["misses"], 1)
        self.assertGreater(second_info["yield_surface"]["hits"], first_info["yield_surface"]["hits"])
        self.assertGreater(second_info["mohr_coulomb"]["hits"], first_info["mohr_coulomb"]["hits"])

    def test_mohr_coulomb_return_mapping_regularizes_tension_side_corner(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "soil",
            14000.0,
            0.30,
            model="mohr_coulomb",
            cohesion=10.0,
            friction_angle=25.0,
            dilation_angle=0.0,
        )
        c, phi, psi = _mc_reduced_parameters(material, 1.0)
        cohesion_term = 2.0 * c * math.cos(phi)
        yield_coeffs = _mc_plane_coeffs(phi)
        flow_coeffs = _mc_plane_coeffs(psi)
        sig_tr = np.array([-36.14151638706245, 111.4935656604855, 113.33635888626321], dtype=float)

        sig_corr, active_ids, gamma, vals_corr = _mc_return_mapping_principal_python(
            sig_tr,
            yield_coeffs=yield_coeffs,
            flow_coeffs=flow_coeffs,
            cohesion_term=cohesion_term,
            Cn=material.D4[:3, :3],
            hardening=material.hardening,
            kappa=0.0,
            tol=_mc_yield_tol(sig_tr, cohesion_term),
            apex_policy="associated_multisurface",
        )

        self.assertTrue(np.all(np.isfinite(sig_corr)))
        self.assertLessEqual(len(active_ids), 3)
        self.assertTrue(all(0 <= active_id < 6 for active_id in active_ids))
        self.assertGreaterEqual(float(np.sum(gamma)), 0.0)
        self.assertLessEqual(float(np.max(vals_corr)), 1.0e-10)

    def test_mohr_coulomb_low_friction_apex_has_verified_associated_return(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "soil",
            14000.0,
            0.30,
            model="mohr_coulomb",
            cohesion=10.0,
            friction_angle=5.0,
            dilation_angle=0.0,
            mohr_coulomb_apex_policy="associated_multisurface",
        )

        replay = fem2d_materials_module.mohr_coulomb_principal_return_feasibility(
            material,
            np.array([500.0, 500.0, 500.0]),
        )

        self.assertFalse(replay["strict_nonassociated"]["exact_complementarity"])
        apex = replay["associated_multisurface_apex"]
        self.assertTrue(apex["exact_complementarity"])
        self.assertLessEqual(apex["max_relative_yield_violation"], 1.0e-12)
        self.assertGreaterEqual(apex["minimum_multiplier"], 0.0)

        fem2d_materials_module.reset_mohr_coulomb_fallback_telemetry()
        update_plane_strain_stress(
            material,
            np.linalg.solve(material.D4, np.array([500.0, 500.0, 500.0, 0.0])),
        )
        telemetry = fem2d_materials_module.mohr_coulomb_fallback_telemetry()
        self.assertEqual(telemetry["associated_apex_projection_count"], 1)
        self.assertEqual(telemetry["legacy_bounded_projection_count"], 0)
        self.assertTrue(telemetry["configured_apex_policy_verified"])
        self.assertTrue(telemetry["constitutive_model_fidelity"])
        self.assertFalse(telemetry["base_nonassociated_flow_rule_verified"])
        self.assertLessEqual(telemetry["max_relative_yield_violation"], 1.0e-12)

    def test_mohr_coulomb_apex_policy_is_parsed_and_validated(self) -> None:
        cfg = {
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 5.0,
                    "dilation_angle": 0.0,
                    "apex_return": {"mode": "associated_multisurface"},
                }
            }
        }
        material = plane_strain_materials(cfg)["soil"]
        self.assertEqual(
            material.mohr_coulomb_apex_policy,
            "associated_multisurface",
        )
        self.assertTrue(material.mohr_coulomb_apex_policy_explicit)
        default_material = plane_strain_materials(
            {
                "materials": {
                    "soil": {
                        "model": "mohr_coulomb",
                        "E": 14000.0,
                        "nu": 0.30,
                        "cohesion": 10.0,
                        "friction_angle": 25.0,
                    }
                }
            }
        )["soil"]
        self.assertEqual(default_material.mohr_coulomb_apex_policy, "legacy_bounded")
        self.assertFalse(default_material.mohr_coulomb_apex_policy_explicit)
        with self.assertRaises(FEM2DError):
            ElasticPlaneStrainMaterial(
                "invalid",
                14000.0,
                0.30,
                model="mohr_coulomb",
                mohr_coulomb_apex_policy="unknown",
            )

    def test_mohr_coulomb_python_fallback_reuses_candidate_lu_cache(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "soil",
            50000.0,
            0.30,
            model="mohr_coulomb",
            cohesion=5.0,
            friction_angle=30.0,
            dilation_angle=0.0,
        )
        c, phi, psi = _mc_reduced_parameters(material, 1.0)
        cohesion_term = 2.0 * c * math.cos(phi)
        yield_coeffs = _mc_plane_coeffs(phi)
        flow_coeffs = _mc_plane_coeffs(psi)
        sig_tr = np.array([10.316916381021441, 116.924410453221, 432.6979347595025])
        kwargs = {
            "yield_coeffs": yield_coeffs,
            "flow_coeffs": flow_coeffs,
            "cohesion_term": cohesion_term,
            "Cn": material.D4[:3, :3],
            "hardening": material.hardening,
            "kappa": 0.0,
            "tol": _mc_yield_tol(sig_tr, cohesion_term),
        }

        cache = fem2d_materials_module._mc_python_candidate_matrix_cache
        cache.cache_clear()
        first = _mc_return_mapping_principal_python(sig_tr, **kwargs)
        first_info = cache.cache_info()
        second = _mc_return_mapping_principal_python(sig_tr, **kwargs)
        second_info = cache.cache_info()

        self.assertEqual(first_info.misses, 1)
        self.assertEqual(second_info.misses, 1)
        self.assertEqual(second_info.hits, first_info.hits + 1)
        self.assertGreaterEqual(len(first[1]), 1)
        self.assertTrue(all(0 <= active_id < 6 for active_id in first[1]))
        self.assertEqual(second[1], first[1])
        self.assertTrue(np.array_equal(second[0], first[0]))
        self.assertTrue(np.array_equal(second[2], first[2]))
        self.assertTrue(np.array_equal(second[3], first[3]))
        self.assertTrue(np.all(np.isfinite(first[0])))
        self.assertTrue(np.all(first[2] >= 0.0))
        self.assertLessEqual(float(np.max(first[3])), kwargs["tol"])

    def test_mohr_coulomb_production_fallback_shortlists_only_safe_friction_range(self) -> None:
        def mapping_inputs(friction_angle: float, sig_tr: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
            material = ElasticPlaneStrainMaterial(
                "soil",
                14000.0,
                0.30,
                model="mohr_coulomb",
                cohesion=10.0,
                friction_angle=friction_angle,
                dilation_angle=0.0,
            )
            c, phi, psi = _mc_reduced_parameters(material, 1.0)
            cohesion_term = 2.0 * c * math.cos(phi)
            kwargs = {
                "yield_coeffs": _mc_plane_coeffs(phi),
                "flow_coeffs": _mc_plane_coeffs(psi),
                "cohesion_term": cohesion_term,
                "Cn": material.D4[:3, :3],
                "hardening": material.hardening,
                "kappa": 0.0,
                "tol": _mc_yield_tol(sig_tr, cohesion_term),
            }
            return kwargs, sig_tr

        high_kwargs, high_sig = mapping_inputs(
            30.0,
            np.array([20.633832762042882, 233.848820906442, 865.395869519005]),
        )
        high_reference = _mc_return_mapping_principal_python(high_sig, **high_kwargs)
        with patch.object(
            fem2d_materials_module,
            "_mc_shortlisted_singular_candidate_scan",
            wraps=fem2d_materials_module._mc_shortlisted_singular_candidate_scan,
        ) as shortlist:
            high_result = fem2d_materials_module._mc_return_mapping_principal(high_sig, **high_kwargs)
        self.assertTrue(shortlist.called)
        for actual, expected in zip(high_result, high_reference):
            if isinstance(actual, np.ndarray):
                self.assertTrue(np.array_equal(actual, expected))
            else:
                self.assertEqual(actual, expected)

        low_kwargs, low_sig = mapping_inputs(
            5.0,
            np.array([53.568608272419866, 126.81017631501027, 717.8702352268871]),
        )
        low_reference = _mc_return_mapping_principal_python(low_sig, **low_kwargs)
        with patch.object(
            fem2d_materials_module,
            "_mc_shortlisted_singular_candidate_scan",
            wraps=fem2d_materials_module._mc_shortlisted_singular_candidate_scan,
        ) as shortlist:
            low_result = fem2d_materials_module._mc_return_mapping_principal(low_sig, **low_kwargs)
        shortlist.assert_not_called()
        for actual, expected in zip(low_result, low_reference):
            if isinstance(actual, np.ndarray):
                self.assertTrue(np.array_equal(actual, expected))
            else:
                self.assertEqual(actual, expected)


class SolverBoundaryModuleTests(unittest.TestCase):
    def test_reduced_matrix_cache_matches_csr_free_block_slicing(self) -> None:
        matrix = csr_matrix(
            np.array(
                [
                    [10.0, 2.0, 0.0, 0.0, 1.0],
                    [2.0, 9.0, 3.0, 0.0, 0.0],
                    [0.0, 3.0, 8.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0, 7.0, 5.0],
                    [1.0, 0.0, 0.0, 5.0, 6.0],
                ],
                dtype=float,
            )
        )
        rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        free = np.array([0, 2, 4], dtype=int)
        fixed = np.array([1, 3], dtype=int)
        fixed_values = np.array([0.25, -0.5], dtype=float)
        cache = build_reduced_matrix_cache_from_csr(matrix, free, fixed)

        self.assertTrue(np.allclose(cache.extract_free_free(matrix).toarray(), matrix[free][:, free].toarray()))
        self.assertTrue(np.allclose(cache.reduced_rhs(matrix, rhs, fixed_values), rhs[free] - matrix[free][:, fixed] @ fixed_values))

        solution, info, reused_cache = solve_reduced_linear_system(
            matrix,
            rhs,
            free,
            fixed,
            fixed_values=fixed_values,
            reduction_cache=cache,
            stage_name="cache-test",
            solver={"linear": {"cache_factorization": False}},
            validate_cache=False,
        )
        expected, _ = solve_linear_system(matrix[free][:, free], rhs[free] - matrix[free][:, fixed] @ fixed_values, stage_name="expected", solver={"linear": {"cache_factorization": False}})
        self.assertIs(reused_cache, cache)
        self.assertTrue(info["reduced_matrix_cache"]["reused"])
        self.assertTrue(np.allclose(solution, expected))

    def test_solver_control_module_parses_increments_and_scales_loads(self) -> None:
        increments = increment_settings({"increment": {"steps": 3, "max_cutbacks": 2}})
        self.assertTrue(increments["enabled"])
        self.assertEqual(increments["steps"], 3)
        self.assertEqual(increments["max_cutbacks"], 2)

        scaled = scale_loads([{"fx": 2.0, "type": "gravity", "scale": 1.5}, {"label": "unchanged"}], 0.25)
        self.assertEqual(scaled[0]["fx"], 0.5)
        self.assertEqual(scaled[0]["scale"], 0.375)
        self.assertEqual(scaled[0]["load_case_factor"], 0.25)
        self.assertEqual(scaled[1]["label"], "unchanged")
        body_scaled = scale_loads([{"type": "body", "by": -2.0, "scale": 2.0}], 0.25)[0]
        self.assertAlmostEqual(body_scaled["by"], -0.5)
        self.assertAlmostEqual(body_scaled["scale"], 2.0)

    def test_material_body_force_and_linear_edge_traction_load_vector(self) -> None:
        cfg = {
            "analysis": {"type": "static_plane_strain"},
            "materials": {
                "soil": {"model": "elastic", "E": 1000.0, "nu": 0.3, "gamma": 0.0},
                "clay": {"model": "elastic", "E": 1200.0, "nu": 0.32, "gamma": 0.0},
            },
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [1.0, 1.0],
                    "4": [0.0, 1.0],
                    "5": [2.0, 0.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"},
                    {"id": "2", "type": "QUAD4", "nodes": ["2", "5", "6", "3"], "material": "clay"},
                ],
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        body = assemble_load_vector(mesh, materials, [{"type": "body", "material": "clay", "bx": 0.0, "by": -2.0}])
        self.assertAlmostEqual(float(body[2 * mesh.node_index["1"] + 1]), 0.0)
        self.assertAlmostEqual(float(body[2 * mesh.node_index["2"] + 1]), -0.5)
        self.assertAlmostEqual(float(body[2 * mesh.node_index["3"] + 1]), -0.5)
        self.assertAlmostEqual(float(body[2 * mesh.node_index["5"] + 1]), -0.5)
        self.assertAlmostEqual(float(body[2 * mesh.node_index["6"] + 1]), -0.5)
        self.assertAlmostEqual(float(body[1::2].sum()), -2.0)

        surface = assemble_load_vector(mesh, materials, [{"edge": ["1", "2"], "distribution": "linear", "ty1": 0.0, "ty2": -6.0}])
        self.assertAlmostEqual(float(surface[2 * mesh.node_index["1"] + 1]), -1.0)
        self.assertAlmostEqual(float(surface[2 * mesh.node_index["2"] + 1]), -2.0)
        self.assertAlmostEqual(float(surface[1::2].sum()), -3.0)

    def test_mpc_module_builds_constraints_and_detects_methods(self) -> None:
        matrix, values = mpc_constraint_matrix(
            {"equations": [{"slave_dof": 1, "master_dof": 0, "coefficient": 2.0, "value": 0.3}]},
            3,
            "unit",
        )
        self.assertTrue(np.allclose(matrix.toarray(), np.array([[-2.0, 1.0, 0.0]])))
        self.assertTrue(np.allclose(values, np.array([0.3])))
        self.assertTrue(mpc_elimination_requested([{"method": "transform"}], None))
        self.assertTrue(mpc_lagrange_requested([], {"mpc": {"method": "lm"}}))
        plan = mpc_stage_plan(
            [{"method": "elimination"}],
            {},
            {"count": 1},
            nonlinear=True,
            add_plain_penalty_to_stage_matrix=True,
        )
        self.assertEqual(plan.applied_method, "penalty_fallback")
        self.assertFalse(plan.use_elimination_linear)
        self.assertFalse(plan.add_penalty_to_stage_matrix)
        lagrange_plan = mpc_stage_plan(
            [{"method": "lagrange"}],
            {},
            {"count": 1},
            nonlinear=False,
        )
        self.assertTrue(lagrange_plan.use_lagrange_linear)
        self.assertEqual(lagrange_plan.applied_method, "lagrange")
        monolithic_dynamic_plan = mpc_stage_plan(
            [{"method": "lagrange"}],
            {},
            {"count": 1},
            nonlinear=True,
            add_penalty_when_lagrange_linear_blocked=True,
        )
        self.assertFalse(monolithic_dynamic_plan.use_lagrange_linear)
        self.assertTrue(monolithic_dynamic_plan.add_penalty_to_stage_matrix)
        self.assertEqual(monolithic_dynamic_plan.applied_method, "penalty_fallback")
        arc_length_plan = mpc_arc_length_stage_plan(
            [{"method": "lagrange"}],
            {},
            {"count": 1},
        )
        self.assertTrue(arc_length_plan.lagrange_requested)
        self.assertFalse(arc_length_plan.add_penalty_to_stage_matrix)
        self.assertEqual(arc_length_plan.applied_method, "lagrange")
        arc_length_exact_fallback = mpc_arc_length_stage_plan(
            [{"method": "elimination"}],
            {},
            {"count": 1},
        )
        self.assertFalse(arc_length_exact_fallback.use_elimination_linear)
        self.assertTrue(arc_length_exact_fallback.add_penalty_to_stage_matrix)
        self.assertEqual(arc_length_exact_fallback.applied_method, "penalty_fallback")

    def test_hydro_iteration_module_tracks_seepage_active_set(self) -> None:
        self.assertEqual(seepage_outer_iteration_limit({"max_outer": 0}), 1)
        self.assertEqual(seepage_outer_iteration_limit({"seepage_max_outer": 3}), 3)
        state = SeepageActiveSetState()
        state, done = advance_seepage_active_set(state, {"seepage_count": 1, "seepage_active_edges": 1, "seepage_inactive_edges": 2})
        self.assertFalse(done)
        self.assertEqual(state.signature, (1, 2))
        self.assertEqual(state.toggle_count, 0)
        state, done = advance_seepage_active_set(state, {"seepage_count": 1, "seepage_active_edges": 2, "seepage_inactive_edges": 1})
        self.assertFalse(done)
        self.assertEqual(state.toggle_count, 1)
        state, done = advance_seepage_active_set(state, {"seepage_count": 1, "seepage_active_edges": 2, "seepage_inactive_edges": 1})
        self.assertTrue(done)
        self.assertEqual(state.toggle_count, 1)
        observed = observe_seepage_active_set(state, {"seepage_active_edges": 0, "seepage_inactive_edges": 3})
        self.assertEqual(observed.toggle_count, 2)

    def test_solver_progress_module_normalizes_stage_sequence_and_inputs(self) -> None:
        default_stages = stage_sequence_from_config({"boundary_conditions": [{"node": "n1"}], "loads": [{"fy": -1.0}]})
        self.assertEqual(default_stages[0]["name"], "Stage-1")
        self.assertEqual(default_stages[0]["loads"], [{"fy": -1.0}])
        stage = {
            "name": "掘削1",
            "type": "GeoStatic",
            "t": "2.5",
            "bc": [{"node": "n2"}],
            "mpc": [{"method": "penalty"}],
            "increment": {"steps": 2},
        }
        self.assertEqual(stage_display_name(stage, 3), "掘削1")
        self.assertEqual(stage_type(stage), "geostatic")
        self.assertEqual(stage_time(stage, 3), 2.5)
        self.assertEqual(stage_boundary_conditions({"bc": [{"node": "n1"}]}, stage), [{"node": "n1"}, {"node": "n2"}])
        self.assertEqual(stage_mpc_constraints({"mpc_constraints": [{"method": "elimination"}]}, stage), [{"method": "elimination"}, {"method": "penalty"}])
        self.assertEqual(stage_loads({}, stage, "geostatic"), [{"type": "gravity", "gx": 0.0, "gy": -1.0, "scale": 1.0}])
        self.assertEqual(stage_solver_config({"linear": {"method": "direct"}}, stage)["increments"], {"steps": 2})

        class _Result:
            pore_pressure = np.array([1.0, 2.0])
            plastic_state = {"e1:0": object()}

        previous, plastic = stage_state_after_result(_Result(), None, copy_pressure=True)
        self.assertTrue(np.allclose(previous, np.array([1.0, 2.0])))
        self.assertIsNot(previous, _Result.pore_pressure)
        self.assertEqual(set(plastic), {"e1:0"})


class TwoDOnlyApplicationTests(unittest.TestCase):
    def test_3d_input_is_rejected_without_backend_delegation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text('{"analysis": {"dimension": "3D", "type": "static"}}', encoding="utf-8")
            with self.assertRaisesRegex(FEM2DError, "3D analysis has been removed"):
                run_solve(path, output_dir=str(Path(tmp) / "out"))


class ShapeFunctionTests(unittest.TestCase):
    def test_element_interpolation_module_contract_covers_shape_and_b_matrices(self) -> None:
        contract = element_interpolation_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_interpolation.v1")
        self.assertIn("shape_functions", contract["covered_surfaces"])
        self.assertIn("gauss_integration_points", contract["covered_surfaces"])
        self.assertIn("axisymmetric_b_matrix", contract["covered_surfaces"])
        self.assertIn("strain_displacement_matrix", contract["functions"])

    def test_element_numba_primitives_module_exports_quad4_quad8_kernels(self) -> None:
        contract = element_numba_primitives_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_numba_primitives.v1")
        self.assertIn("quad4_b_matrix_and_jacobian", contract["covered_surfaces"])
        self.assertIn("quad8_gauss_rules", contract["covered_surfaces"])

        coords4 = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
        B4, det4 = primitive_quad4_b_det_numba(coords4, 0.0, 0.0)
        self.assertAlmostEqual(float(det4), 0.5)
        self.assertEqual(B4.shape, (4, 8))
        xi, eta, weight = primitive_quad8_gp_full(4)
        self.assertEqual((xi, eta), (0.0, 0.0))
        self.assertAlmostEqual(weight, 64.0 / 81.0)

    def test_element_elastic_post_module_exports_quad4_quad8_stress_kernels(self) -> None:
        contract = element_elastic_post_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_elastic_post.v1")
        self.assertIn("quad4_elastic_post", contract["covered_surfaces"])
        self.assertIn("quad8_elastic_tension_cutoff_post", contract["covered_surfaces"])

        cfg = plane_strain_patch_sample("QUAD8", "FULL")
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        u = np.zeros(mesh.coords.shape[0] * 2, dtype=float)
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        data = elastic_post_module_quad8_fast(coords, u[dofs], material)
        self.assertEqual(data.shape, (9, 20))
        self.assertTrue(np.allclose(data[:, 6:20], 0.0))

    def test_element_elastic_kernel_module_matches_public_stiffness_wrapper(self) -> None:
        contract = elastic_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_elastic_kernels.v1")
        self.assertIn("quad4_plane_strain_stiffness_internal_force", contract["covered_surfaces"])
        self.assertIn("quad8_axisymmetric_stiffness_internal_force", contract["covered_surfaces"])

        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
        material = ElasticPlaneStrainMaterial("soil", 10000.0, 0.30)
        direct = elastic_kernel_module_quad4_stiffness_fast(coords, material, "FULL")
        public = element_stiffness("QUAD4", coords, material, "FULL")
        self.assertTrue(np.allclose(direct, public, rtol=1.0e-12, atol=1.0e-12))

    def test_partition_of_unity(self) -> None:
        for etype in ["TRI3", "TRI6", "QUAD4", "QUAD8"]:
            points = [(0.2, 0.2)] if etype.startswith("TRI") else [(0.0, 0.0), (-0.3, 0.4)]
            for xi, eta in points:
                N, dN = shape_functions(etype, xi, eta)
                self.assertAlmostEqual(float(np.sum(N)), 1.0, places=13, msg=etype)
                self.assertTrue(np.allclose(np.sum(dN, axis=1), 0.0, atol=1e-13), etype)

    def test_element_fast_path_module_selects_elastic_post_and_state_arrays(self) -> None:
        contract = element_fast_path_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_fast_paths.v1")
        self.assertIn("elastic_post_fast_path_selection", contract["covered_surfaces"])
        self.assertIn("plastic_state_arrays", contract["covered_surfaces"])

        element = Element2D("E1", "QUAD4", ("1", "2", "3", "4"), "soil", integration="FULL")
        elastic = ElasticPlaneStrainMaterial("soil", 10000.0, 0.30)
        self.assertTrue(_quad4_elastic_post_fast_path(element, elastic, None))
        plastic = ElasticPlaneStrainMaterial("soil", 10000.0, 0.30, model="von_mises", yield_stress=10.0)
        self.assertFalse(_quad4_elastic_post_fast_path(element, plastic, None))

        states = {"E1:2": PlasticState2D(np.array([0.1, 0.2, 0.3, 0.4], dtype=float), 0.5)}
        plastic_strains, kappas = _quad4_post_state_arrays("E1", states)
        self.assertTrue(np.allclose(plastic_strains[2], [0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(kappas[2], 0.5)

    def test_element_result_row_module_formats_fast_post_rows(self) -> None:
        contract = element_result_row_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_result_rows.v1")
        self.assertIn("elastic_post_result_rows", contract["covered_surfaces"])
        self.assertIn("inactive_result_rows", contract["covered_surfaces"])

        element = Element2D("E1", "QUAD4", ("1", "2", "3", "4"), "soil", integration="FULL")
        material = ElasticPlaneStrainMaterial("soil", 10000.0, 0.30)
        data = np.array([[0.0, 0.0, 1.0, 0.5, 0.5, 2.0, 0.1, 0.2, 0.3, 0.4, 10.0, 20.0, 30.0, 4.0, 31.0, 19.0, 10.0, 10.5, 12.0, 13.0]])
        row = _quad4_elastic_post_result_rows(element, material, data)[0]
        self.assertEqual(row["element_id"], "E1")
        self.assertEqual(row["state_key"], "E1:0")
        self.assertEqual(row["sigma_x"], 10.0)
        self.assertEqual(row["active"], 1.0)
        inactive = _inactive_element_result(element)
        self.assertEqual(inactive["active"], 0.0)
        self.assertEqual(inactive["integration"], "FULL")

    def test_element_post_processing_module_matches_public_wrapper(self) -> None:
        contract = element_post_processing_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_post_processing.v1")
        self.assertIn("integration_point_post_orchestration", contract["covered_surfaces"])
        self.assertIn("fast_post_kernel_dispatch", contract["covered_surfaces"])

        element = Element2D("E1", "QUAD4", ("1", "2", "3", "4"), "soil", integration="FULL")
        mesh = Mesh2D(
            node_ids=["1", "2", "3", "4"],
            coords=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float),
            elements=[element],
        )
        materials = {"soil": ElasticPlaneStrainMaterial("soil", 10000.0, 0.30)}
        u = np.zeros(8, dtype=float)
        self.assertEqual(
            compute_integration_point_results(mesh, materials, u),
            compute_integration_point_results_impl(mesh, materials, u),
        )


class PlaneStrainPatchTests(unittest.TestCase):
    def assert_patch(self, element_type: str, integration: str = "FULL") -> None:
        cfg = plane_strain_patch_sample(element_type, integration)
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].element_results[0]
        expected_strain = np.array([0.001, -0.0003, 0.0, 0.0006])
        mat = ElasticPlaneStrainMaterial("soil", E=50000.0, nu=0.30)
        expected_stress = mat.D4 @ expected_strain
        actual_strain = np.array([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]])
        actual_stress = np.array([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]])
        self.assertTrue(np.allclose(actual_strain, expected_strain, rtol=1e-10, atol=1e-12), (element_type, integration, actual_strain))
        self.assertTrue(np.allclose(actual_stress, expected_stress, rtol=1e-10, atol=1e-8), (element_type, integration, actual_stress))
        s = principal_stresses(actual_stress)
        self.assertGreaterEqual(s[0], s[1])
        self.assertGreaterEqual(s[1], s[2])

    def test_full_patch_elements(self) -> None:
        for etype in ["TRI3", "TRI6", "QUAD4", "QUAD8"]:
            with self.subTest(etype=etype):
                self.assert_patch(etype, "FULL")

    def test_quad4_locking_modes_keep_patch(self) -> None:
        for mode in ["SRI", "B-bar"]:
            with self.subTest(mode=mode):
                self.assert_patch("QUAD4", mode)

    def test_plane_strain_patch_matches_full_plane_strain_constitutive_matrix(self) -> None:
        cfg = plane_strain_patch_sample("QUAD4", "FULL")
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].element_results[0]
        E = 50000.0
        nu = 0.30
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        D6 = np.array(
            [
                [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
                [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
                [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, mu, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, mu, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, mu],
            ],
            dtype=float,
        )
        strain6 = np.array([0.001, -0.0003, 0.0, 0.0006, 0.0, 0.0])
        stress6 = D6 @ strain6
        actual = np.array([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]])
        self.assertTrue(np.allclose(actual, stress6[[0, 1, 2, 3]], rtol=1.0e-10, atol=1.0e-8))


class PlaneStrainSolveTests(unittest.TestCase):
    @staticmethod
    def quad8_rectangle_config(integration: str = "FULL", material: dict[str, object] | None = None, *, axisymmetric: bool = False) -> dict[str, object]:
        return {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static" if axisymmetric else "static_plane_strain"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0] if axisymmetric else [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD8",
                "integration": integration,
                "material": "soil",
            },
            "materials": {"soil": material or {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }

    @staticmethod
    def displacement_from_coords(coords: np.ndarray) -> np.ndarray:
        u = np.zeros(coords.shape[0] * 2, dtype=float)
        for i, (x, y) in enumerate(coords):
            u[2 * i] = 0.001 * x + 0.0002 * y + 0.0001 * x * y
            u[2 * i + 1] = -0.0003 * y + 0.0004 * x - 0.00005 * x * y
        return u

    def test_mesh_solve_preflight_is_scale_relative(self) -> None:
        element = Element2D("1", "QUAD4", ("1", "2", "3", "4"), "soil")
        summaries = []
        for scale in (1.0e-9, 1.0, 1.0e9):
            mesh = Mesh2D(
                node_ids=["1", "2", "3", "4"],
                coords=scale * np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float),
                elements=[element],
            )
            summaries.append(validate_mesh_quality_for_solve(mesh))
        self.assertTrue(all(summary["passed"] for summary in summaries))
        self.assertTrue(
            np.allclose(
                [summary["min_normalized_jacobian"] for summary in summaries],
                summaries[0]["min_normalized_jacobian"],
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        )

    def test_mesh_solve_preflight_rejects_positive_but_nearly_singular_jacobian(self) -> None:
        mesh = Mesh2D(
            node_ids=["1", "2", "3", "4"],
            coords=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0e-14], [0.0, 1.0e-14]], dtype=float),
            elements=[Element2D("thin", "QUAD4", ("1", "2", "3", "4"), "soil")],
        )
        with self.assertRaisesRegex(FEM2DError, "mesh solve preflight failed") as caught:
            validate_mesh_quality_for_solve(mesh)
        self.assertEqual(caught.exception.diagnostics.get("status"), "mesh_quality_preflight_failed")

    def test_mesh_solve_preflight_allows_node_only_exchange_model(self) -> None:
        mesh = Mesh2D(
            node_ids=["1"],
            coords=np.array([[0.0, 0.0]], dtype=float),
            elements=[],
        )
        summary = validate_mesh_quality_for_solve(mesh)
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["geometry_checks_skipped"])
        self.assertEqual(summary["checked_element_count"], 0)
        self.assertTrue(math.isfinite(summary["coordinate_ulp_ratio"]))

    @staticmethod
    def quad8_mixed_integration_chain_config() -> dict[str, object]:
        nodes: dict[str, list[float]] = {}
        elements: list[dict[str, object]] = []
        top_nodes: list[str] = []
        for index, integration in enumerate(("FULL", "SRI", "B-bar")):
            x0 = float(index)
            x1 = float(index + 1)
            ids = [f"{index + 1}-{local}" for local in range(1, 9)]
            coords = [
                [x0, 0.0],
                [x1, 0.0],
                [x1, 1.0],
                [x0, 1.0],
                [(x0 + x1) * 0.5, 0.0],
                [x1, 0.5],
                [(x0 + x1) * 0.5, 1.0],
                [x0, 0.5],
            ]
            for nid, xy in zip(ids, coords):
                nodes[nid] = xy
            top_nodes.extend([ids[2], ids[3], ids[6]])
            elements.append({"id": f"q8-{index + 1}", "type": "QUAD8", "nodes": ids, "material": "soil", "integration": integration})
        bcs = []
        for nid, (x, y) in nodes.items():
            bcs.append({"node": nid, "ux": 1.0e-5 * x + 2.0e-6 * y, "uy": -3.0e-6 * y + 1.0e-6 * x})
        return {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": nodes, "elements": elements, "node_sets": {"top": top_nodes}},
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 500.0, "friction_angle": 30.0}},
            "boundary_conditions": bcs,
            "stages": [{"name": "large-q8-mixed", "type": "large_deformation", "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}],
        }

    def test_quad4_sample_writes_outputs(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
            self.assertEqual(len(result.mesh.elements), 16)
            self.assertTrue(np.all(np.isfinite(stage.displacements)))
            self.assertTrue((stage.output_dir / "displacements.csv").exists())
            self.assertTrue((stage.output_dir / "element_stress.csv").exists())
            ip_csv = stage.output_dir / "integration_point_stress.csv"
            self.assertTrue(ip_csv.exists())
            with ip_csv.open(encoding="utf-8") as f:
                ip_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(ip_rows), len(result.mesh.elements) * 4)
            for key in ["stage", "time", "element_id", "ip", "x", "y", "q", "p", "kappa", "plastic_strain_x", "state_key"]:
                self.assertIn(key, ip_rows[0])
            self.assertTrue((stage.output_dir / "results.vtk").exists())
            report = result.output_dir / "calculation_report.html"
            self.assertTrue(report.exists())
            report_text = report.read_text(encoding="utf-8")
            for heading in ["入力条件", "材料表", "境界/荷重図", "ステージ一覧", "解析結果図", "判定表", "ログ", "再現条件"]:
                self.assertIn(heading, report_text)
            self.assertIn("calculation_report.html", (result.output_dir / "summary.json").read_text(encoding="utf-8"))
            post_info = stage.solver_info["postprocess_state_commit"]
            self.assertTrue(post_info["integration_point_second_pass_skipped"])
            self.assertGreaterEqual(post_info["integration_point_rows"], len(result.mesh.elements) * 4)

    def test_lazy_report_generation_defers_heavy_run_artifacts(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        cfg["output"] = {"lazy_reports": True}
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["output_generation"]["lazy_reports"])
            self.assertTrue(summary["result_view_index"]["deferred"])
            self.assertTrue(summary["standard_report"]["deferred"])
            self.assertTrue(summary["calculation_report"]["deferred"])
            self.assertFalse((result.output_dir / "result_view_index.html").exists())
            self.assertFalse((result.output_dir / "standard_report.html").exists())
            self.assertFalse((result.output_dir / "calculation_report.html").exists())
            self.assertTrue((result.output_dir / "summary.json").exists())
            self.assertTrue((result.stages[0].output_dir / "displacements.csv").exists())
            generated = write_deferred_run_artifacts(result)
            self.assertIn("result_view_index", generated["generated"])
            self.assertTrue((result.output_dir / "result_view_index.html").exists())
            self.assertTrue((result.output_dir / "standard_report.html").exists())
            self.assertTrue((result.output_dir / "calculation_report.html").exists())
            updated = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("standard_report", updated["output_generation"]["deferred_artifacts_materialized"])
            self.assertFalse(updated["calculation_report"]["deferred"])

    def test_lazy_report_generation_materializes_from_persisted_gui_run_files(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        cfg["output"] = {"lazy_reports": True}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            input_path = run_dir / "input.yaml"
            input_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
            result = solve_plane_strain_config(cfg, run_dir / "results")
            self.assertFalse((result.output_dir / "calculation_report.html").exists())
            generated = write_deferred_run_artifacts_from_files(result.output_dir, input_path=input_path)
            self.assertIn("calculation_report", generated["generated"])
            self.assertTrue((result.output_dir / "calculation_report.html").exists())
            self.assertTrue((result.output_dir / "calculation_report.pdf").exists())
            self.assertTrue((result.output_dir / "result_view_index.html").exists())
            updated = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(updated["calculation_report"]["deferred"])
            self.assertIn("result_view_index", updated["output_generation"]["deferred_artifacts_materialized"])

    def test_large_deformation_geometry_update_kernels_are_vectorized_and_numba_ready(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
        }
        mesh = mesh_from_config(cfg)
        disp = np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2], dtype=float)
        updated = mesh_with_updated_coords(mesh, disp, backend="vectorized")
        updated_fast = mesh_with_updated_coords(mesh, disp, backend="numba")
        reusable = np.empty_like(mesh.coords)
        returned = fill_updated_coords(mesh.coords, disp, out=reusable, backend="vectorized")
        self.assertIs(returned, reusable)
        self.assertTrue(np.allclose(updated.coords, updated_fast.coords))
        self.assertTrue(np.allclose(updated.coords, reusable))
        self.assertTrue(np.allclose(updated.coords[2], [1.1, 1.2]))
        contract = large_deformation_kernel_contract()
        self.assertIn("_updated_coords_numba", contract["numba_functions"])
        self.assertIn("_fill_updated_coords_numba", contract["numba_functions"])
        self.assertIn("updated_coords_vectorized", contract["vectorized_functions"])
        self.assertIn("fill_updated_coords", contract["vectorized_functions"])
        self.assertIn("geometry_update_cache", contract)
        self.assertIn("topology_cache", contract)

    def test_large_deformation_topology_cache_reuses_stiffness_pattern(self) -> None:
        cfg = {
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
            "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_large_deformation_step_cache(mesh, materials, cfg["boundary_conditions"])
        disp = np.linspace(0.0, 0.1, len(mesh.node_ids) * 2, dtype=float)
        updated = mesh_with_updated_coords(mesh, disp, backend="vectorized")
        cached_k = assemble_global_stiffness_cached(cache.stiffness_cache, updated, materials)
        normal_k = assemble_global_stiffness(updated, materials)
        self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray()))
        self.assertEqual(cache.solver_info()["constrained_dofs"], 6)
        self.assertEqual(cache.solver_info()["stiffness_blocks"], 2)
        self.assertEqual(cache.solver_info()["batched_quad4_elastic_elements"], 2)
        self.assertTrue(cache.solver_info()["reduced_matrix_cached"])
        self.assertGreater(cache.solver_info()["reduced_matrix_cache"]["free_free"]["nnz"], 0)

    def test_global_stiffness_cache_precomputes_linear_interface_and_structural_blocks(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [1.0, 0.0], "4": [1.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        interfaces = [Interface2D(id="joint", minus_nodes=("1", "2"), plus_nodes=("3", "4"), kn=1000.0, kt=500.0)]
        structural = [StructuralElement2D(id="bar", type="BAR2", nodes=("3", "4"), section={"kx": 250.0})]
        cache = build_global_stiffness_assembly_cache(mesh, materials, interfaces=interfaces, structural_elements=structural)
        cached_k = assemble_global_stiffness_cached(cache, mesh, materials, interfaces=interfaces, structural_elements=structural)
        normal_k = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural)
        info = cache.info()
        self.assertEqual(info["precomputed_interface_blocks"], 1)
        self.assertEqual(info["precomputed_structural_blocks"], 1)
        self.assertEqual(info["precomputed_linear_blocks"], 2)
        self.assertEqual(info["precomputed_linear_batches"]["batch_count"], 2)
        self.assertEqual(info["precomputed_linear_batches"]["batched_blocks"], 2)
        self.assertTrue(info["direct_fill"]["enabled"])
        topology = build_small_deformation_step_cache(
            mesh,
            materials,
            [{"nodes": ["1", "2"], "fixed": True}],
            interfaces=interfaces,
            structural_elements=structural,
        ).solver_info()
        self.assertEqual(topology["precomputed_interface_blocks"], 1)
        self.assertEqual(topology["precomputed_structural_blocks"], 1)
        self.assertEqual(topology["stiffness_assembly_cache"]["precomputed_linear_blocks"], 2)
        self.assertEqual(topology["precomputed_linear_stiffness_batch_count"], 2)
        self.assertEqual(topology["precomputed_linear_stiffness_batched_blocks"], 2)
        self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray()))

    def test_global_stiffness_cache_batches_mixed_linear_interfaces_and_structural_blocks(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil"},
                ],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        interfaces = [
            Interface2D(id="joint-1", minus_nodes=("1", "4"), plus_nodes=("2", "5"), kn=1000.0, kt=300.0),
            Interface2D(id="joint-2", minus_nodes=("2", "5"), plus_nodes=("3", "6"), kn=800.0, kt=250.0),
        ]
        structural = [
            StructuralElement2D(id="bar", type="BAR2", nodes=("1", "2"), section={"kx": 250.0}),
            StructuralElement2D(id="frame", type="FRAME2", nodes=("4", "5"), material="soil", section={"A": 0.2, "I": 0.01}),
            StructuralElement2D(id="shear", type="SHEAR_SPRING2", nodes=("5", "6"), section={"ky": 150.0}),
        ]
        cache = build_global_stiffness_assembly_cache(mesh, materials, interfaces=interfaces, structural_elements=structural)
        cached_k = assemble_global_stiffness_cached(cache, mesh, materials, interfaces=interfaces, structural_elements=structural)
        normal_k = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural)
        info = cache.info()
        batches = info["precomputed_linear_batches"]
        self.assertEqual(info["precomputed_interface_blocks"], 2)
        self.assertEqual(info["precomputed_structural_blocks"], 3)
        self.assertEqual(batches["interface_batches"], 1)
        self.assertEqual(batches["structural_batches"], 1)
        self.assertEqual(batches["batch_count"], 2)
        self.assertEqual(batches["batched_blocks"], 5)
        self.assertGreater(batches["flat_value_size"], 0)
        self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray(), rtol=1.0e-12, atol=1.0e-9))

    def test_precomputed_linear_element_stiffness_matches_uncached_for_plane_elements(self) -> None:
        for element_type, integration in (("QUAD4", "FULL"), ("QUAD8", "B-bar"), ("TRI3", "FULL"), ("TRI6", "SRI")):
            with self.subTest(element_type=element_type, integration=integration):
                cfg = {
                    "analysis": {"dimension": "2D", "type": "static_plane_strain"},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [0.0, 2.0],
                        "y_range": [0.0, 1.0],
                        "nx": 2,
                        "ny": 2,
                        "element_type": element_type,
                        "integration": integration,
                        "material": "soil",
                    },
                    "materials": {"soil": {"model": "elastic", "E": 12000.0, "nu": 0.28}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                cache = build_global_stiffness_assembly_cache(mesh, materials, precompute_linear_element_stiffness=True)
                cached_k = assemble_global_stiffness_cached(cache, mesh, materials)
                normal_k = assemble_global_stiffness(mesh, materials)
                info = cache.info()
                batches = info["precomputed_linear_batches"]
                self.assertTrue(batches["enabled"])
                self.assertEqual(batches["element_batches"], 1)
                self.assertEqual(info["precomputed_element_blocks"], len(mesh.elements))
                self.assertEqual(batches["batched_blocks"], len(mesh.elements))
                self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray(), rtol=1.0e-11, atol=1.0e-8))

    def test_dynamic_step_cache_precomputed_stiffness_mass_and_load_match_uncached(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
                "node_sets": {"right": ["3", "6"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
        }
        loads = [{"set": "right", "fx": 0.5}, {"type": "body", "by": -0.2}]
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_dynamic_mass_step_cache(mesh, materials, loads)
        assert cache.stiffness_cache is not None
        assert cache.mass_cache is not None
        assert cache.load_vector_cache is not None

        cached_k = assemble_global_stiffness_cached(cache.stiffness_cache, mesh, materials)
        cached_m = assemble_mass_matrix_cached(cache.mass_cache, mesh, materials)
        cached_f = assemble_load_vector_cached(cache.load_vector_cache, mesh, materials)
        info = cache.solver_info()

        self.assertTrue(info["stiffness_cache"]["precomputed_linear_batches"]["enabled"])
        self.assertEqual(info["stiffness_cache"]["precomputed_linear_batches"]["batched_blocks"], len(mesh.elements))
        self.assertTrue(info["mass_cache"]["direct_fill"]["enabled"])
        self.assertTrue(np.allclose(cached_k.toarray(), assemble_global_stiffness(mesh, materials).toarray(), rtol=1.0e-12, atol=1.0e-9))
        self.assertTrue(np.allclose(cached_m.toarray(), assemble_mass_matrix(mesh, materials).toarray(), rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(np.allclose(cached_f, assemble_load_vector(mesh, materials, loads), rtol=1.0e-12, atol=1.0e-12))

    def test_load_vector_cache_reuses_body_and_edge_targets_with_updated_coords(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil"},
                ],
                "node_sets": {"top": ["4", "5", "6"]},
                "element_sets": {"right": ["e2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3, "gamma": 18.0}},
        }
        loads = [
            {"node": "6", "fx": 0.2, "fy": -0.3},
            {"type": "body", "by": -1.5, "elements": ["e1", "e2"]},
            {"edges": "top", "tx1": 0.1, "ty1": -0.4, "tx2": 0.2, "ty2": -0.8},
        ]
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_load_vector_assembly_cache(mesh, materials, loads)
        self.assertIsNotNone(cache)
        assert cache is not None
        disp = np.linspace(0.0, 0.08, len(mesh.node_ids) * 2, dtype=float)
        updated = mesh_with_updated_coords(mesh, disp, backend="vectorized")

        cached = assemble_load_vector_cached(cache, updated, materials)
        normal = assemble_load_vector(updated, materials, loads)
        info = cache.info()
        self.assertEqual(info["body_blocks"], 2)
        self.assertEqual(info["edge_blocks"], 2)
        self.assertTrue(info["node_vector_cached"])
        self.assertTrue(info["geometry_dependent"])
        self.assertTrue(np.allclose(cached, normal, rtol=1.0e-12, atol=1.0e-12))

    def test_large_deformation_quad4_batch_supports_full_sri_and_bbar_together(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [3.0, 0.0],
                    "5": [0.0, 1.0],
                    "6": [1.0, 1.0],
                    "7": [2.0, 1.0],
                    "8": [3.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "6", "5"], "material": "soil", "integration": "FULL"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "7", "6"], "material": "soil", "integration": "SRI"},
                    {"id": "e3", "type": "QUAD4", "nodes": ["3", "4", "8", "7"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {"soil": {"model": "elastic", "E": 1200.0, "nu": 0.32}},
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "ux": 0.0, "uy": 0.0}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_large_deformation_step_cache(mesh, materials, cfg["boundary_conditions"])
        disp = np.linspace(0.0, 0.03, len(mesh.node_ids) * 2, dtype=float)
        updated = mesh_with_updated_coords(mesh, disp, backend="vectorized")
        cached_k = assemble_global_stiffness_cached(cache.stiffness_cache, updated, materials)
        normal_k = assemble_global_stiffness(updated, materials)
        self.assertEqual(cache.solver_info()["batched_quad4_elastic_elements"], 3)
        self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray()))

    def test_quad8_global_stiffness_cache_batches_full_sri_and_bbar(self) -> None:
        for integration in ("FULL", "SRI", "B-bar"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config(integration)
                cfg["mesh"].update({"nx": 2, "ny": 2})
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                cache = build_global_stiffness_assembly_cache(mesh, materials)
                cached_k = assemble_global_stiffness_cached(cache, mesh, materials)
                normal_k = assemble_global_stiffness(mesh, materials)
                info = cache.info()
                self.assertEqual(info["batched_quad4_elastic_elements"], 0)
                self.assertEqual(info["batched_quad8_elastic_elements"], len(mesh.elements))
                self.assertEqual(info["batched_elastic_elements"], len(mesh.elements))
                self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray(), rtol=1.0e-11, atol=1.0e-8))

    def test_tri_global_stiffness_cache_batches_tri3_tri6_full_sri_and_bbar(self) -> None:
        for element_type in ("TRI3", "TRI6"):
            for integration in ("FULL", "SRI", "B-bar"):
                with self.subTest(element_type=element_type, integration=integration):
                    cfg = {
                        "analysis": {"dimension": "2D", "type": "static_plane_strain"},
                        "mesh": {
                            "generator": "rectangle",
                            "x_range": [0.0, 1.0],
                            "y_range": [0.0, 1.0],
                            "nx": 2,
                            "ny": 2,
                            "element_type": element_type,
                            "integration": integration,
                            "material": "soil",
                        },
                        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
                    }
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    cache = build_global_stiffness_assembly_cache(mesh, materials)
                    cached_k = assemble_global_stiffness_cached(cache, mesh, materials)
                    normal_k = assemble_global_stiffness(mesh, materials)
                    info = cache.info()
                    tri3_expected = len(mesh.elements) if element_type == "TRI3" else 0
                    tri6_expected = len(mesh.elements) if element_type == "TRI6" else 0
                    self.assertEqual(info["batched_tri3_elastic_elements"], tri3_expected)
                    self.assertEqual(info["batched_tri6_elastic_elements"], tri6_expected)
                    self.assertEqual(info["batched_elastic_elements"], len(mesh.elements))
                    self.assertTrue(np.allclose(cached_k.toarray(), normal_k.toarray(), rtol=1.0e-11, atol=1.0e-8))

    def test_large_deformation_quad8_elastic_batch_supports_full_sri_and_bbar_together(self) -> None:
        cfg = self.quad8_mixed_integration_chain_config()
        cfg["materials"]["soil"] = {"model": "elastic", "E": 50000.0, "nu": 0.30}
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        topology = stage.solver_info["large_deformation"]["topology_cache"]
        self.assertEqual(topology["batched_quad4_elastic_elements"], 0)
        self.assertEqual(topology["batched_quad8_elastic_elements"], 3)
        self.assertEqual(topology["batched_elastic_elements"], 3)
        self.assertEqual(stage.solver_info["batched_elements"], 3)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_deformation_mode_contract_defaults_stage_and_solver_info(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "deformation_mode": "large_deformation"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"], "top": ["3", "4"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}, {"set": "top", "uy": 0.001}],
            "solver": {"large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}},
        }
        self.assertEqual(deformation_mode_from_config(cfg), "large_deformation")
        self.assertEqual(stage_sequence_from_config(cfg)[0]["type"], "large_deformation")
        explicit_static = dict(cfg)
        explicit_static["stages"] = [{"name": "explicit-static", "type": "static"}]
        self.assertEqual(stage_sequence_from_config(explicit_static)[0]["deformation_mode"], "large_deformation")
        contract = large_deformation_performance_contract()
        self.assertIn("geometry_mode", contract["common_solver_info_fields"])
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        with tempfile.TemporaryDirectory() as tmp:
            explicit_stage = solve_plane_strain_config(explicit_static, tmp).stages[0]
        self.assertEqual(stage.solver_info["method"], "updated_lagrangian")
        self.assertEqual(explicit_stage.solver_info["method"], "updated_lagrangian")
        for key in ("geometry_mode", "element_type", "integration", "material_model", "batched_elements", "fallback_count", "fallback_reasons"):
            self.assertIn(key, stage.solver_info)

    def test_large_deformation_plastic_step_cache_normalizes_topology_and_state_layout(self) -> None:
        cfg = {
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 100.0, "friction_angle": 30.0}},
            "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_large_deformation_step_cache(mesh, materials, cfg["boundary_conditions"])
        info = cache.solver_info()
        self.assertEqual(info["active_element_array_size"], 2)
        self.assertEqual(info["connectivity_shape"], [2, 4])
        self.assertEqual(info["element_dof_shape"], [2, 8])
        self.assertEqual(info["state_point_count_max"], 4)
        self.assertEqual(info["plastic_blocks"][0]["material_model"], "drucker_prager")
        self.assertIn("strength_factor", info["cache_inputs"])
        self.assertEqual(info["plastic_state_layout"]["order"], "element_major_integration_point_minor")

    def test_quad4_full_plastic_batch_outputs_match_existing_j2dp_kernel(self) -> None:
        cfg = {
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 100.0, "friction_angle": 30.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        blocks = build_plastic_element_blocks(mesh, materials)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(plastic_batch_contract()["outputs"], ["ke_values", "internal_force_values", "updated_state_values", "status_flags"])
        result = evaluate_plastic_tangent_block(blocks[0], mesh, materials, u)
        self.assertTrue(np.all(result.status_flags == 0))
        first = mesh.elements[int(blocks[0].element_indices[0])]
        conn = [mesh.node_index[nid] for nid in first.nodes]
        coords = mesh.coords[conn]
        ue = u[result.dofs[0]]
        material = materials[first.material]
        alpha, cohesion = _yield_surface_parameters(material, 1.0)
        expected_ke, expected_fe = _quad4_j2dp_tangent_force_fast(
            coords,
            ue,
            material,
            initial_stress=np.zeros(4, dtype=float),
            plastic_strains=np.zeros((4, 4), dtype=float),
            kappas=np.zeros(4, dtype=float),
            alpha=alpha,
            cohesion_term=cohesion,
        )
        self.assertTrue(np.allclose(result.ke_values[0].reshape(8, 8), expected_ke))
        self.assertTrue(np.allclose(result.internal_force_values[0], expected_fe))
        self.assertEqual(result.updated_state_values.shape, (2, 4, 5))

    def test_plastic_batch_reuses_material_strength_arrays_by_factor(self) -> None:
        clear_material_strength_parameter_array_cache()
        clear_strength_parameter_cache()
        cfg = {
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 80.0, "friction_angle": 28.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        blocks = build_plastic_element_blocks(mesh, materials)
        self.assertEqual(len(blocks), 1)

        result_1 = evaluate_plastic_tangent_block(blocks[0], mesh, materials, u, strength_factor=1.2)
        first = material_strength_parameter_array_cache_info()
        result_2 = evaluate_plastic_tangent_block(blocks[0], mesh, materials, u, strength_factor=1.2)
        second = material_strength_parameter_array_cache_info()
        result_3 = evaluate_plastic_tangent_block(blocks[0], mesh, materials, u, strength_factor=1.3)
        third = material_strength_parameter_array_cache_info()

        self.assertTrue(np.all(result_1.status_flags == 0))
        self.assertTrue(np.all(result_2.status_flags == 0))
        self.assertTrue(np.all(result_3.status_flags == 0))
        self.assertEqual(first["misses"], 1)
        self.assertEqual(second["misses"], first["misses"])
        self.assertGreater(second["hits"], first["hits"])
        self.assertEqual(third["misses"], second["misses"] + 1)

    def test_plastic_state_array_cache_uses_active_element_point_order(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "FULL"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "SRI"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 80.0, "friction_angle": 28.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        plastic_state = {
            "e1:2": PlasticState2D(np.array([0.1, 0.2, 0.3, 0.4], dtype=float), 0.5),
            "e2:1": PlasticState2D(state_vars={"ru": 0.25}),
            "e2:4": PlasticState2D(np.array([0.4, 0.3, 0.2, 0.1], dtype=float), 0.75),
        }

        cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
        info = plastic_state_array_cache_info(cache)
        e1_strains, e1_kappas = cache.state_arrays("e1", 4)
        e2_strains, e2_kappas = cache.state_arrays("e2", 5)

        self.assertEqual(cache.element_ids, ("e1", "e2"))
        self.assertEqual(cache.state_point_counts.tolist(), [4, 5])
        self.assertEqual(info["layout"], "active_element_major_integration_point_minor")
        self.assertEqual(info["present_points"], 3)
        self.assertEqual(info["numeric_state_view_points"], 2)
        self.assertEqual(info["numeric_state_view"], "array_backed")
        self.assertTrue(np.allclose(e1_strains[2], [0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(e1_kappas[2], 0.5)
        self.assertTrue(np.allclose(e2_strains[4], [0.4, 0.3, 0.2, 0.1]))
        self.assertEqual(e2_kappas[4], 0.75)
        self.assertTrue(cache.has_state_vars("e2", 5))
        self.assertIsNone(cache.state_objects[cache.element_row["e1"], 2])
        numeric_view = cache.state_view_for_gp("e1", 2)
        self.assertIsInstance(numeric_view, PlasticStateView2D)
        self.assertNotIsInstance(numeric_view, PlasticState2D)
        self.assertTrue(np.shares_memory(numeric_view.plastic_strain, cache.plastic_strains[cache.element_row["e1"], 2, :]))
        compat_state = cache.state_for_gp("e1", 2)
        self.assertIsInstance(compat_state, PlasticState2D)
        self.assertFalse(np.shares_memory(compat_state.plastic_strain, cache.plastic_strains[cache.element_row["e1"], 2, :]))
        view_update = update_plane_strain_stress(materials["soil"], np.array([0.001, -0.0002, 0.0, 0.0003], dtype=float), state=numeric_view)
        object_update = update_plane_strain_stress(
            materials["soil"],
            np.array([0.001, -0.0002, 0.0, 0.0003], dtype=float),
            state=plastic_state["e1:2"],
        )
        self.assertTrue(np.allclose(view_update.stress, object_update.stress))
        self.assertEqual(cache.state_for_gp("e2", 1).state_vars["ru"], 0.25)
        self.assertIs(cache.state_view_for_gp("e2", 1), plastic_state["e2:1"])

        numeric_state = {key: value for key, value in plastic_state.items() if not value.state_vars}
        numeric_cache = build_plastic_state_array_cache(mesh, materials, numeric_state)
        u = self.displacement_from_coords(mesh.coords)
        tangent_from_cache = assemble_algorithmic_tangent_stiffness(mesh, materials, u, plastic_state=numeric_state, plastic_state_cache=numeric_cache)
        internal_from_cache = assemble_internal_force(mesh, materials, u, plastic_state=numeric_state, plastic_state_cache=numeric_cache)
        tangent_from_mapping = assemble_algorithmic_tangent_stiffness(mesh, materials, u, plastic_state=numeric_state)
        internal_from_mapping = assemble_internal_force(mesh, materials, u, plastic_state=numeric_state)
        self.assertTrue(np.allclose(tangent_from_cache.toarray(), tangent_from_mapping.toarray()))
        self.assertTrue(np.allclose(internal_from_cache, internal_from_mapping))

    def test_plastic_state_array_cache_fallback_uses_views_without_compat_materialization(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [0.0, 1.0],
                },
                "elements": [
                    {"id": "tri", "type": "TRI3", "nodes": ["1", "2", "3"], "material": "soil", "integration": "FULL"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 80.0, "friction_angle": 28.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        plastic_state = {
            "tri:0": PlasticState2D(np.array([0.015, -0.005, 0.002, 0.004], dtype=float), 0.02),
        }
        cache = build_plastic_state_array_cache(mesh, materials, plastic_state)

        with patch.object(type(cache), "state_for_gp", side_effect=AssertionError("compat materialization path should not be used")):
            tangent_from_cache = assemble_algorithmic_tangent_stiffness(mesh, materials, u, plastic_state=plastic_state, plastic_state_cache=cache)
            internal_from_cache = assemble_internal_force(mesh, materials, u, plastic_state=plastic_state, plastic_state_cache=cache)
            combined_tangent, combined_internal = assemble_tangent_and_internal_force(
                mesh,
                materials,
                u,
                plastic_state=plastic_state,
                plastic_state_cache=cache,
            )

        tangent_from_mapping = assemble_algorithmic_tangent_stiffness(mesh, materials, u, plastic_state=plastic_state)
        internal_from_mapping = assemble_internal_force(mesh, materials, u, plastic_state=plastic_state)
        self.assertTrue(np.allclose(tangent_from_cache.toarray(), tangent_from_mapping.toarray()))
        self.assertTrue(np.allclose(internal_from_cache, internal_from_mapping))
        self.assertTrue(np.allclose(combined_tangent.toarray(), tangent_from_mapping.toarray()))
        self.assertTrue(np.allclose(combined_internal, internal_from_mapping))

    def test_tangent_internal_combined_assembly_matches_separate_paths(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [3.0, 0.0],
                    "5": [0.0, 1.0],
                    "6": [1.0, 1.0],
                    "7": [2.0, 1.0],
                    "8": [3.0, 1.0],
                },
                "elements": [
                    {"id": "full", "type": "QUAD4", "nodes": ["1", "2", "6", "5"], "material": "soil", "integration": "FULL"},
                    {"id": "sri", "type": "QUAD4", "nodes": ["2", "3", "7", "6"], "material": "soil", "integration": "SRI"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["3", "4", "8", "7"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 80.0, "friction_angle": 28.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        plastic_state = {
            "full:2": PlasticState2D(np.array([0.03, 0.01, 0.0, 0.02], dtype=float), 0.04),
            "sri:4": PlasticState2D(np.array([0.02, 0.00, 0.01, 0.03], dtype=float), 0.03),
            "bbar:1": PlasticState2D(np.array([0.01, 0.02, 0.00, 0.01], dtype=float), 0.02),
        }
        cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
        initial = {
            "full": np.array([1.0, -0.5, 0.25, 0.2], dtype=float),
            "sri": np.array([0.5, 0.2, -0.1, -0.3], dtype=float),
            "bbar": np.array([-0.2, 0.4, 0.1, 0.15], dtype=float),
        }
        initial_cache = build_initial_stress_array_cache(mesh, initial)
        initial_cache_info = initial_stress_array_cache_info(initial_cache)
        self.assertTrue(initial_cache_info["enabled"])
        self.assertEqual(initial_cache_info["layout"], "active_element_major")
        self.assertEqual(initial_cache_info["present_elements"], 3)

        combined_tangent, combined_internal = assemble_tangent_and_internal_force(
            mesh,
            materials,
            u,
            initial_stress_cache=initial_cache,
            strength_factor=1.25,
            plastic_state=plastic_state,
            plastic_state_cache=cache,
        )
        expected_tangent = assemble_algorithmic_tangent_stiffness(
            mesh,
            materials,
            u,
            initial_stress_cache=initial_cache,
            strength_factor=1.25,
            plastic_state=plastic_state,
            plastic_state_cache=cache,
        )
        expected_internal = assemble_internal_force(
            mesh,
            materials,
            u,
            initial_stresses=initial,
            strength_factor=1.25,
            plastic_state=plastic_state,
            plastic_state_cache=cache,
        )
        self.assertTrue(np.allclose(combined_tangent.toarray(), expected_tangent.toarray()))
        self.assertTrue(np.allclose(combined_internal, expected_internal))

        dof_blocks = []
        for element in mesh.elements:
            dofs: list[int] = []
            for node in element.nodes:
                idx = mesh.node_index[node]
                dofs.extend([2 * idx, 2 * idx + 1])
            dof_blocks.append(dofs)
        sparse_pattern = SparseAssemblyPattern.from_square_blocks(dof_blocks, combined_tangent.shape)
        direct_info = sparse_pattern.direct_fill_info()
        self.assertTrue(direct_info["enabled"])
        self.assertEqual(direct_info["mode"], "flat_offset_direct_fill")
        self.assertIn(direct_info["scatter_mode"], {"direct", "accumulate"})
        self.assertEqual(sparse_pattern.flat_value_size, sum(len(dofs) * len(dofs) for dofs in dof_blocks))
        disjoint_pattern = SparseAssemblyPattern.from_square_blocks([np.array([0, 1]), np.array([2, 3])], (4, 4))
        disjoint_values = disjoint_pattern.empty_flat_values()
        disjoint_pattern.fill_block(disjoint_values, 0, np.eye(2))
        disjoint_pattern.fill_block(disjoint_values, 1, 2.0 * np.eye(2))
        self.assertEqual(disjoint_pattern.direct_fill_info()["scatter_mode"], "direct")
        self.assertTrue(np.allclose(disjoint_pattern.assemble_flat_values(disjoint_values).diagonal(), [1.0, 1.0, 2.0, 2.0]))
        direct_blocks = [
            (np.arange(len(dofs) * len(dofs), dtype=float).reshape((len(dofs), len(dofs))) + 100.0 * (index + 1))
            for index, dofs in enumerate(dof_blocks)
        ]
        direct_flat = sparse_pattern.empty_flat_values()
        for index, block in enumerate(direct_blocks):
            sparse_pattern.fill_block(direct_flat, index, block)
        self.assertTrue(np.allclose(sparse_pattern.assemble_flat_values(direct_flat).toarray(), sparse_pattern.assemble(direct_blocks).toarray()))
        cached_tangent, cached_internal = assemble_tangent_and_internal_force(
            mesh,
            materials,
            u,
            initial_stresses=initial,
            strength_factor=1.25,
            plastic_state=plastic_state,
            plastic_state_cache=cache,
            sparse_pattern=sparse_pattern,
        )
        cached_tangent_only = assemble_algorithmic_tangent_stiffness(
            mesh,
            materials,
            u,
            initial_stresses=initial,
            strength_factor=1.25,
            plastic_state=plastic_state,
            plastic_state_cache=cache,
            sparse_pattern=sparse_pattern,
        )
        self.assertTrue(np.allclose(cached_tangent.toarray(), expected_tangent.toarray()))
        self.assertTrue(np.allclose(cached_tangent_only.toarray(), expected_tangent.toarray()))
        self.assertTrue(np.allclose(cached_internal, expected_internal))

    def test_sparse_assembly_diagnostics_count_fallback_and_duplicate_scatter(self) -> None:
        set_sparse_assembly_diagnostics_enabled(True)
        reset_sparse_assembly_diagnostics()
        try:
            builder = SparseAssemblyBuilder()
            builder.add_block([0, 1], [0, 1], np.eye(2))
            self.assertTrue(np.allclose(builder.to_csr((2, 2)).toarray(), np.eye(2)))

            pattern = SparseAssemblyPattern.from_square_blocks(
                [np.array([0, 1]), np.array([0, 1])],
                (2, 2),
            )
            self.assertEqual(pattern.direct_fill_info()["scatter_mode"], "accumulate")
            assembled = pattern.assemble([np.eye(2), 2.0 * np.eye(2)])
            self.assertTrue(np.allclose(assembled.toarray(), 3.0 * np.eye(2)))
            diagnostics = sparse_assembly_diagnostics()
        finally:
            set_sparse_assembly_diagnostics_enabled(False)

        self.assertEqual(diagnostics["builder_to_csr_count"], 1)
        self.assertEqual(diagnostics["builder_block_count"], 1)
        self.assertEqual(diagnostics["builder_value_count"], 4)
        self.assertEqual(diagnostics["pattern_build_count"], 1)
        self.assertEqual(diagnostics["pattern_duplicate_scatter_count"], 1)
        self.assertTrue(diagnostics["fallback_builder_used"])
        self.assertTrue(diagnostics["duplicate_scatter_used"])

    def test_nonlinear_solver_reports_combined_tangent_internal_assembly(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "FULL"}],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 500.0, "friction_angle": 30.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0, "uy": 0.0}],
            "loads": [{"node": "3", "fy": -10.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        self.assertEqual(stage.solver_info["method"], "newton")
        self.assertTrue(stage.solver_info["combined_tangent_internal_assembly"])
        topology = stage.solver_info["topology_cache"]
        self.assertTrue(topology["enabled"])
        self.assertTrue(topology["auto_generated"])
        self.assertEqual(topology["cache_kind"], "small_deformation_step_cache")
        self.assertTrue(stage.solver_info["sparse_pattern_cached"])
        self.assertTrue(stage.solver_info["constraint_dofs_cached"])
        self.assertTrue(stage.solver_info["reduced_matrix_cache"]["enabled"])
        first_iteration = stage.solver_info["convergence_history"][0]
        self.assertIn("tangent_internal_assembly_elapsed_seconds", first_iteration)
        self.assertIn("reduced_matrix_elapsed_seconds", first_iteration)
        self.assertIn("linear_solve_elapsed_seconds", first_iteration)
        self.assertIn("line_search_elapsed_seconds", first_iteration)
        self.assertIn("elapsed_seconds", first_iteration)
        factor_cache = stage.solver_info["nonlinear_factor_cache"]
        self.assertEqual(factor_cache["mode"], "auto")
        self.assertGreaterEqual(factor_cache["solves"], 1)
        self.assertEqual(
            factor_cache["solves"],
            factor_cache["hits"]
            + factor_cache["misses"]
            + factor_cache["auto_disabled_solves"]
            + factor_cache["disabled_solves"],
        )
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_nonlinear_factor_cache_summary_reports_auto_disable(self) -> None:
        summary = fem2d_solver_module._nonlinear_factor_cache_summary(
            [
                {"lu_factor_cache_state": "miss"},
                {"lu_factor_cache_state": "hit"},
                {"lu_factor_cache_state": "miss"},
                {"lu_factor_cache_state": "miss"},
                {"lu_factor_cache_state": "miss"},
                {"lu_factor_cache_state": "auto_disabled"},
            ],
            mode="auto",
        )
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(summary["misses"], 4)
        self.assertEqual(summary["max_consecutive_misses"], 3)
        self.assertTrue(summary["disabled_after_misses"])
        self.assertEqual(summary["disable_reason"], "consecutive_factorization_cache_misses")
        self.assertEqual(summary["control_scope"], "sparse_pattern_global_with_periodic_reprobe")

    def test_static_step_cache_can_be_disabled_for_nonlinear_stage(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "FULL"}],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 500.0, "friction_angle": 30.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0, "uy": 0.0}],
            "loads": [{"node": "3", "fy": -10.0}],
            "solver": {"static_step_cache": {"enabled": False}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        topology = stage.solver_info["topology_cache"]
        self.assertFalse(topology["enabled"])
        self.assertEqual(topology["reason"], "disabled_by_solver_setting")
        self.assertFalse(stage.solver_info["sparse_pattern_cached"])
        self.assertFalse(stage.solver_info["constraint_dofs_cached"])

    def test_nonlinear_solver_reuses_initial_stress_array_cache(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "FULL"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 12000.0, "nu": 0.30, "yield_stress": 1.0e6}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0}, {"nodes": ["1", "2"], "uy": 0.0}],
            "loads": [{"node": "3", "fy": -5.0}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        step_cache = build_small_deformation_step_cache(mesh, materials, cfg["boundary_conditions"])
        initial = {"e1": np.array([2.0, -1.0, 0.5, 0.25], dtype=float)}

        stage = solve_plane_strain_stage(
            mesh=mesh,
            materials=materials,
            boundary_conditions=cfg["boundary_conditions"],
            loads=cfg["loads"],
            stage_name="initial-cache",
            output_dir=None,
            solver=None,
            initial_stresses=initial,
            step_cache=step_cache,
            postprocess_results=False,
        )

        info = stage.solver_info["initial_stress_array_cache"]
        self.assertTrue(info["enabled"])
        self.assertEqual(info["layout"], "active_element_major")
        self.assertEqual(info["active_elements"], 1)
        self.assertEqual(info["present_elements"], 1)
        self.assertTrue(stage.solver_info["combined_tangent_internal_assembly"])
        self.assertTrue(stage.solver_info["sparse_pattern_cached"])
        reduced = stage.solver_info["reduced_matrix_cache"]
        self.assertTrue(reduced["enabled"])
        self.assertGreaterEqual(reduced["solves"], 1)
        self.assertGreaterEqual(reduced["hits"], 1)

    def test_lightweight_stage_keeps_plastic_state_as_array_cache(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "FULL"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 12000.0, "nu": 0.30, "yield_stress": 1.0e6}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0}, {"nodes": ["1", "2"], "uy": 0.0}],
            "loads": [{"node": "3", "fy": -5.0}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)

        stage = solve_plane_strain_stage(
            mesh=mesh,
            materials=materials,
            boundary_conditions=cfg["boundary_conditions"],
            loads=cfg["loads"],
            stage_name="array-only-state",
            output_dir=None,
            solver=None,
            postprocess_results=False,
        )

        self.assertEqual(stage.element_results, [])
        self.assertEqual(stage.plastic_state, {})
        self.assertIsNotNone(stage.plastic_state_array_cache)
        self.assertEqual(stage.solver_info["plastic_ratio_source"], "plastic_state_array_cache")
        self.assertEqual(stage.solver_info["postprocess_state_commit"]["state_commit"], "array_only")
        self.assertFalse(stage.solver_info["postprocess_state_commit"]["dict_materialized"])
        self.assertTrue(stage.solver_info["plastic_state_array_cache"]["enabled"])

    def test_quad4_plastic_batch_supports_full_sri_bbar_and_tension_cutoff(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [3.0, 0.0],
                    "5": [0.0, 1.0],
                    "6": [1.0, 1.0],
                    "7": [2.0, 1.0],
                    "8": [3.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "6", "5"], "material": "soil", "integration": "FULL"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "7", "6"], "material": "soil", "integration": "SRI"},
                    {"id": "e3", "type": "QUAD4", "nodes": ["3", "4", "8", "7"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 50.0, "friction_angle": 25.0, "tension_cutoff": 5.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        contract_blocks = plastic_batch_contract()["supported_blocks"]
        self.assertIn("QUAD4:SRI:drucker_prager", contract_blocks)
        self.assertIn("QUAD4:B-BAR:mohr_coulomb", contract_blocks)
        for block in build_plastic_element_blocks(mesh, materials):
            result = evaluate_plastic_tangent_block(block, mesh, materials, u)
            element_index = int(block.element_indices[0])
            element = mesh.elements[element_index]
            self.assertTrue(np.all(result.status_flags == 0))
            self.assertEqual(result.ke_values.shape[1], 64)
            self.assertEqual(result.internal_force_values.shape[1], 8)
            self.assertEqual(result.updated_state_values.shape[1], 5 if block.integration == "SRI" else 4)
            single_mesh = Mesh2D(
                node_ids=list(mesh.node_ids),
                coords=mesh.coords.copy(),
                elements=[element],
                node_sets=dict(mesh.node_sets),
                element_sets=dict(mesh.element_sets),
            )
            expected_tangent = assemble_algorithmic_tangent_stiffness(single_mesh, materials, u)
            expected_internal = assemble_internal_force(single_mesh, materials, u)
            self.assertTrue(np.allclose(result.ke_values[0].reshape(8, 8), expected_tangent.toarray()[np.ix_(result.dofs[0], result.dofs[0])]))
            self.assertTrue(np.allclose(result.internal_force_values[0], expected_internal[result.dofs[0]]))

    def test_quad4_sri_bbar_drucker_prager_uses_numba_batch_kernel(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "sri", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "SRI"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 50.0, "friction_angle": 25.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        kernels = {}
        for block in build_plastic_element_blocks(mesh, materials):
            result = evaluate_plastic_tangent_block(block, mesh, materials, u)
            self.assertTrue(np.all(result.status_flags == 0))
            kernels[block.integration] = result.solver_info()["kernel"]
            self.assertEqual(result.updated_state_values.shape[1], 5 if block.integration == "SRI" else 4)

        self.assertEqual(kernels["SRI"], "quad4_sri_j2dp_batch_numba")
        self.assertEqual(kernels["B-BAR"], "quad4_bbar_j2dp_batch_numba")
        batched = _quad4_sri_bbar_plastic_batch_blocks(
            mesh,
            materials,
            u,
            initial_stresses=None,
            plastic_state=None,
            strength_factor=1.0,
        )
        self.assertEqual(set(batched), {0, 1})
        tangent = assemble_algorithmic_tangent_stiffness(mesh, materials, u)
        internal = assemble_internal_force(mesh, materials, u)
        self.assertTrue(np.all(np.isfinite(tangent.data)))
        self.assertTrue(np.all(np.isfinite(internal)))

    def test_quad4_sri_bbar_mohr_coulomb_nonassociated_uses_numba_batch_kernel(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "sri", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "SRI"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 50000.0,
                    "nu": 0.3,
                    "cohesion": 50.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        u = self.displacement_from_coords(mesh.coords)
        kernels = {}

        for block in build_plastic_element_blocks(mesh, materials):
            result = evaluate_plastic_tangent_block(block, mesh, materials, u)
            self.assertTrue(np.all(result.status_flags == 0))
            kernels[block.integration] = result.solver_info()["kernel"]
            self.assertEqual(result.updated_state_values.shape[1], 5 if block.integration == "SRI" else 4)
            element_index = int(block.element_indices[0])
            element = mesh.elements[element_index]
            conn = [mesh.node_index[node_id] for node_id in element.nodes]
            coords = mesh.coords[conn]
            ue = u[result.dofs[0]]
            expected_ke, expected_fe, expected_state = _quad4_generic_tangent_force_state(
                element,
                coords,
                ue,
                material,
                np.zeros(4, dtype=float),
                strength_factor=1.0,
                plastic_state=None,
            )
            self.assertTrue(np.allclose(result.ke_values[0].reshape(8, 8), expected_ke, rtol=1.0e-10, atol=1.0e-8))
            self.assertTrue(np.allclose(result.internal_force_values[0], expected_fe, rtol=1.0e-10, atol=1.0e-8))
            self.assertTrue(np.allclose(result.updated_state_values[0], expected_state, rtol=1.0e-10, atol=1.0e-8))

        self.assertEqual(kernels["SRI"], "quad4_sri_mc_batch_numba")
        self.assertEqual(kernels["B-BAR"], "quad4_bbar_mc_batch_numba")
        batched = _quad4_sri_bbar_plastic_batch_blocks(
            mesh,
            materials,
            u,
            initial_stresses=None,
            plastic_state=None,
            strength_factor=1.0,
        )
        self.assertEqual(set(batched), {0, 1})

    def test_quad4_mc_regularized_projection_numba_matches_case2_case4_python_path(self) -> None:
        from geofem_app.fem2d_element_mohr_coulomb_kernels import (
            _quad4_mc_stress_tangent_state_regularized_numba,
        )
        from geofem_app.fem2d_plastic_batch import _build_mc_material_arrays

        cases = (
            (
                10.0,
                25.0,
                1.340625,
                np.array([-30.322152808175474, 199.43875977836774, 258.087431004466, 0.0]),
            ),
            (
                20.0,
                35.0,
                2.1,
                np.array([255.73412667203388, 102.09362207583081, 105.66077975802853, 60.09832906726717]),
            ),
        )
        for cohesion, friction_angle, strength_factor, trial_stress in cases:
            material = ElasticPlaneStrainMaterial(
                "soil",
                E=14000.0,
                nu=0.30,
                model="mohr_coulomb",
                cohesion=cohesion,
                friction_angle=friction_angle,
                dilation_angle=0.0,
                mohr_coulomb_apex_policy="associated_multisurface",
            )
            strain = np.linalg.solve(material.D4, trial_stress)
            state = PlasticState2D()
            expected_update = update_plane_strain_stress(
                material,
                strain,
                state=state,
                strength_factor=strength_factor,
            )
            expected_tangent = algorithmic_material_tangent(
                material,
                strain,
                state=state,
                strength_factor=strength_factor,
            )
            (
                d4,
                s4,
                yield_coeffs,
                flow_coeffs,
                cohesion_terms,
                hardening,
                _thickness,
                _operator_indices,
                operator1,
                operator2,
                operator3,
                candidate_h,
            ) = _build_mc_material_arrays([material], strength_factor)
            result = _quad4_mc_stress_tangent_state_regularized_numba(
                np.ascontiguousarray(strain),
                np.zeros(4, dtype=float),
                0.0,
                d4[0],
                s4[0],
                np.zeros(4, dtype=float),
                yield_coeffs[0],
                flow_coeffs[0],
                cohesion_terms[0],
                hardening[0],
                operator1[0],
                operator2[0],
                operator3[0],
                candidate_h[0],
            )
            self.assertTrue(result[0])
            self.assertGreater(result[5], 0)
            self.assertTrue(np.allclose(result[1], expected_update.stress, rtol=0.0, atol=2.0e-12))
            self.assertTrue(np.allclose(result[2], expected_tangent, rtol=1.0e-10, atol=1.0e-6))
            self.assertTrue(np.allclose(result[3], expected_update.plastic_strain, rtol=0.0, atol=2.0e-12))
            self.assertAlmostEqual(result[4], expected_update.kappa, places=12)

    def test_quad4_sri_bbar_mc_regularized_projection_stays_in_numba_batch(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "sri", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "SRI"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        target_strain = np.array([0.04535905, 0.04075716, 0.0, -0.01517363])
        displacement = np.zeros(2 * len(mesh.node_ids), dtype=float)
        for node_index, (x, y) in enumerate(mesh.coords):
            displacement[2 * node_index] = target_strain[0] * x + 0.5 * target_strain[3] * y
            displacement[2 * node_index + 1] = target_strain[1] * y + 0.5 * target_strain[3] * x

        fem2d_materials_module.reset_mohr_coulomb_fallback_telemetry()
        batch_results = {}
        for block in build_plastic_element_blocks(mesh, materials):
            result = evaluate_plastic_tangent_block(
                block,
                mesh,
                materials,
                displacement,
                strength_factor=1.340625,
            )
            self.assertTrue(np.all(result.status_flags == 0))
            self.assertGreater(result.mc_numba_regularized_projection_count, 0)
            batch_results[block.integration] = (block, result)
        telemetry = fem2d_materials_module.mohr_coulomb_fallback_telemetry()
        self.assertEqual(telemetry["numba_to_python_count"], 0)
        self.assertEqual(
            telemetry["numba_regularized_projection_count"],
            sum(result.mc_numba_regularized_projection_count for _block, result in batch_results.values()),
        )
        self.assertEqual(
            telemetry["regularized_projection_count"],
            telemetry["numba_regularized_projection_count"],
        )
        self.assertEqual(telemetry["associated_apex_projection_count"], 0)
        self.assertEqual(
            telemetry["legacy_bounded_projection_count"],
            telemetry["numba_regularized_projection_count"],
        )
        self.assertFalse(telemetry["configured_apex_policy_verified"])
        self.assertEqual(
            telemetry["regularization_method"],
            "bounded_sequential_cone_tip",
        )

        for block, result in batch_results.values():
            element = mesh.elements[int(block.element_indices[0])]
            conn = [mesh.node_index[node_id] for node_id in element.nodes]
            coords = mesh.coords[conn]
            expected_ke, expected_fe, expected_state = _quad4_generic_tangent_force_state(
                element,
                coords,
                displacement[result.dofs[0]],
                materials["soil"],
                np.zeros(4, dtype=float),
                strength_factor=1.340625,
                plastic_state=None,
            )
            self.assertTrue(np.allclose(result.ke_values[0].reshape(8, 8), expected_ke, rtol=1.0e-9, atol=2.0e-7))
            self.assertTrue(np.allclose(result.internal_force_values[0], expected_fe, rtol=1.0e-11, atol=1.0e-10))
            self.assertTrue(np.allclose(result.updated_state_values[0], expected_state, rtol=1.0e-11, atol=1.0e-11))

        cfg["materials"]["soil"]["mohr_coulomb_apex_policy"] = (
            "associated_multisurface"
        )
        associated_materials = plane_strain_materials(cfg)
        associated_block = next(
            block
            for block in build_plastic_element_blocks(mesh, associated_materials)
            if block.integration == "SRI"
        )
        fem2d_materials_module.reset_mohr_coulomb_fallback_telemetry()
        associated_result = evaluate_plastic_tangent_block(
            associated_block,
            mesh,
            associated_materials,
            displacement,
            strength_factor=1.340625,
        )
        associated_telemetry = (
            fem2d_materials_module.mohr_coulomb_fallback_telemetry()
        )
        self.assertTrue(np.all(associated_result.status_flags == 0))
        self.assertGreater(associated_result.mc_numba_regularized_projection_count, 0)
        self.assertEqual(associated_telemetry["legacy_bounded_projection_count"], 0)
        self.assertEqual(
            associated_telemetry["associated_apex_projection_count"],
            associated_result.mc_numba_regularized_projection_count,
        )
        self.assertTrue(associated_telemetry["configured_apex_policy_verified"])

    def test_quad4_sri_bbar_mc_geometry_cache_reuses_exact_coefficients(self) -> None:
        from geofem_app.fem2d_plastic_batch import (
            build_quad4_mc_geometry_cache,
        )

        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "sri", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "SRI"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        displacement = np.linspace(0.0, 2.0e-3, 2 * len(mesh.node_ids))
        geometry_cache = build_quad4_mc_geometry_cache(mesh, materials)

        for block in build_plastic_element_blocks(mesh, materials):
            baseline = evaluate_plastic_tangent_block(
                block, mesh, materials, displacement, strength_factor=1.2
            )
            cached = evaluate_plastic_tangent_block(
                block,
                mesh,
                materials,
                displacement,
                strength_factor=1.2,
                quad4_mc_geometry_cache=geometry_cache,
            )
            self.assertTrue(np.array_equal(cached.ke_values, baseline.ke_values))
            self.assertTrue(
                np.array_equal(
                    cached.internal_force_values,
                    baseline.internal_force_values,
                )
            )
            self.assertTrue(
                np.array_equal(
                    cached.updated_state_values,
                    baseline.updated_state_values,
                )
            )
        cache_info = geometry_cache.solver_info()
        self.assertEqual(cache_info["scope"], "small_deformation_step_and_srm_factor")
        self.assertEqual(cache_info["element_count"], 2)
        self.assertEqual(cache_info["block_cache_hits"], 2)
        self.assertEqual(cache_info["block_cache_misses"], 0)

    def test_quad4_sri_bbar_mc_active_set_reuses_regularized_tangent_without_changing_state(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "sri", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "SRI"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        target_strain = np.array([0.04535905, 0.04075716, 0.0, -0.01517363])
        displacement = np.zeros(2 * len(mesh.node_ids), dtype=float)
        for node_index, (x, y) in enumerate(mesh.coords):
            displacement[2 * node_index] = target_strain[0] * x + 0.5 * target_strain[3] * y
            displacement[2 * node_index + 1] = target_strain[1] * y + 0.5 * target_strain[3] * x

        blocks = build_plastic_element_blocks(mesh, materials)
        baseline = [
            evaluate_plastic_tangent_block(
                block,
                mesh,
                materials,
                displacement,
                strength_factor=1.340625,
            )
            for block in blocks
        ]
        cache = MohrCoulombActiveSetCache()
        first = [
            evaluate_plastic_tangent_block(
                block,
                mesh,
                materials,
                displacement,
                strength_factor=1.340625,
                mohr_coulomb_active_set_cache=cache,
            )
            for block in blocks
        ]
        second = [
            evaluate_plastic_tangent_block(
                block,
                mesh,
                materials,
                displacement,
                strength_factor=1.340625,
                mohr_coulomb_active_set_cache=cache,
            )
            for block in blocks
        ]

        for expected, initial, reused in zip(baseline, first, second):
            self.assertTrue(np.array_equal(initial.ke_values, expected.ke_values))
            self.assertTrue(np.array_equal(initial.internal_force_values, expected.internal_force_values))
            self.assertTrue(np.array_equal(initial.updated_state_values, expected.updated_state_values))
            self.assertTrue(np.allclose(reused.ke_values, expected.ke_values, rtol=0.0, atol=5.0e-12))
            self.assertTrue(np.allclose(reused.internal_force_values, expected.internal_force_values, rtol=0.0, atol=2.0e-13))
            self.assertTrue(np.allclose(reused.updated_state_values, expected.updated_state_values, rtol=0.0, atol=1.0e-15))
            self.assertGreater(reused.mc_active_set_update_attempt_count, 0)
            self.assertGreater(reused.mc_active_set_update_hit_count, 0)
            self.assertEqual(
                reused.mc_active_set_update_hit_count,
                reused.mc_active_set_regularized_update_hit_count,
            )
        cache_info = cache.solver_info()
        self.assertEqual(cache_info["refresh_interval"], 8)
        self.assertEqual(cache_info["secant_update"], "rank_one_broyden")
        self.assertGreater(cache_info["block_cache_hits"], 0)

    def test_mc_active_set_strict_tangent_preserves_ids_and_invalidates_only_tangent(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {
                        "id": "sri",
                        "type": "QUAD4",
                        "nodes": ["1", "2", "5", "4"],
                        "material": "soil",
                        "integration": "SRI",
                    },
                    {
                        "id": "bbar",
                        "type": "QUAD4",
                        "nodes": ["2", "3", "6", "5"],
                        "material": "soil",
                        "integration": "B-BAR",
                    },
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        strain = np.array([0.04535905, 0.04075716, 0.0, -0.01517363])
        displacement = np.zeros(2 * len(mesh.node_ids), dtype=float)
        for node_index, (x, y) in enumerate(mesh.coords):
            displacement[2 * node_index] = (
                strain[0] * x + 0.5 * strain[3] * y
            )
            displacement[2 * node_index + 1] = (
                strain[1] * y + 0.5 * strain[3] * x
            )
        block = next(
            block
            for block in build_plastic_element_blocks(mesh, materials)
            if block.integration == "SRI"
        )
        baseline = evaluate_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            strength_factor=1.340625,
        )
        cache = MohrCoulombActiveSetCache(tangent_reuse_enabled=False)
        first = evaluate_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            strength_factor=1.340625,
            mohr_coulomb_active_set_cache=cache,
        )
        cached = cache.hint_arrays(block, 1.340625)
        ids_before = cached.active_ids.copy()
        cached.tangent_valid.fill(True)
        invalidated = cache.force_numerical_tangent(
            "test_line_search_storm"
        )
        repeated_invalidation = cache.force_numerical_tangent(
            "test_line_search_storm"
        )
        second = evaluate_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            strength_factor=1.340625,
            mohr_coulomb_active_set_cache=cache,
        )

        self.assertGreater(invalidated, 0)
        self.assertEqual(repeated_invalidation, 0)
        self.assertTrue(np.any(ids_before >= 0))
        self.assertTrue(np.array_equal(ids_before, cached.active_ids))
        self.assertTrue(np.array_equal(first.ke_values, baseline.ke_values))
        self.assertTrue(
            np.allclose(second.ke_values, baseline.ke_values, rtol=1.0e-9, atol=2.0e-5)
        )
        self.assertGreater(second.mc_active_set_update_attempt_count, 0)
        cache_info = cache.solver_info()
        self.assertFalse(cache_info["tangent_reuse_enabled"])
        self.assertFalse(cache_info["direct_consistent_tangent_enabled"])
        self.assertEqual(cache_info["tangent_invalidation_count"], 1)
        self.assertEqual(
            cache_info["consistent_tangent"],
            "numerical_regularized_projection",
        )
        self.assertEqual(cache_info["numerical_tangent_switch_count"], 1)

        selective_strict = MohrCoulombActiveSetCache(
            tangent_reuse_enabled=False,
            direct_consistent_tangent_enabled=True,
            strict_unstable_points_only=True,
        ).solver_info()
        self.assertTrue(selective_strict["direct_consistent_tangent_enabled"])
        self.assertTrue(selective_strict["strict_unstable_points_only"])
        self.assertEqual(
            selective_strict["consistent_tangent"],
            "strict_direct_stable_then_numerical_unstable_points",
        )

    def test_mc_adaptive_numerical_tangent_uses_batch_instability_metrics(self) -> None:
        config = {
            "regularized_projection_invalidation_min_count": 32,
            "regularized_projection_invalidation_fraction": 0.05,
            "active_set_miss_invalidation_min_attempts": 32,
            "active_set_miss_invalidation_fraction": 0.35,
        }
        regularized = fem2d_solver_module._mc_adaptive_numerical_tangent_decision(
            (10, 100, 90, 0),
            (50, 200, 190, 0),
            point_capacity=500,
            config=config,
        )
        self.assertEqual(
            regularized["switch_reason"], "regularized_projection_density"
        )
        self.assertAlmostEqual(
            regularized["regularized_projection_fraction"], 0.08
        )

        churn = fem2d_solver_module._mc_adaptive_numerical_tangent_decision(
            (0, 0, 0, 0),
            (0, 64, 32, 0),
            point_capacity=500,
            config=config,
        )
        self.assertEqual(churn["switch_reason"], "active_set_churn")
        self.assertAlmostEqual(churn["active_set_miss_fraction"], 0.5)

        sparse = fem2d_solver_module._mc_adaptive_numerical_tangent_decision(
            (0, 0, 0, 0),
            (8, 16, 8, 0),
            point_capacity=500,
            config=config,
        )
        self.assertEqual(sparse["switch_reason"], "")

    def test_mc_sri_bbar_line_search_force_candidates_match_scalar_assembly(
        self,
    ) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {
                        "id": "sri",
                        "type": "QUAD4",
                        "nodes": ["1", "2", "5", "4"],
                        "material": "soil",
                        "integration": "SRI",
                    },
                    {
                        "id": "bbar",
                        "type": "QUAD4",
                        "nodes": ["2", "3", "6", "5"],
                        "material": "soil",
                        "integration": "B-BAR",
                    },
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        strain = np.array([0.04535905, 0.04075716, 0.0, -0.01517363])
        displacement = np.zeros(2 * len(mesh.node_ids), dtype=float)
        for node_index, (x, y) in enumerate(mesh.coords):
            displacement[2 * node_index] = (
                strain[0] * x + 0.5 * strain[3] * y
            )
            displacement[2 * node_index + 1] = (
                strain[1] * y + 0.5 * strain[3] * x
            )
        candidates = np.vstack(
            [displacement, 0.5 * displacement, 0.25 * displacement]
        )

        batched = assemble_internal_force_candidates(
            mesh,
            materials,
            candidates,
            strength_factor=1.340625,
        )
        expected = np.vstack(
            [
                assemble_internal_force(
                    mesh,
                    materials,
                    candidate,
                    strength_factor=1.340625,
                )
                for candidate in candidates
            ]
        )

        self.assertIsNotNone(batched)
        assert batched is not None
        self.assertTrue(np.allclose(batched, expected, rtol=1.0e-12, atol=1.0e-10))

    def test_mc_precomputed_fixed_active_set_tangent_matches_direct_solve(self) -> None:
        cfg = {
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 14000.0,
                    "nu": 0.30,
                    "cohesion": 10.0,
                    "friction_angle": 25.0,
                    "dilation_angle": 0.0,
                }
            }
        }
        material = plane_strain_materials(cfg)["soil"]
        (
            d4,
            _s4,
            yield_coeffs,
            flow_coeffs,
            _cohesion,
            _hardening,
            _thickness,
            operator_indices,
            operator1,
            operator2,
            operator3,
            candidate_h,
        ) = fem2d_plastic_batch_module._mc_material_arrays(
            [material],
            1.340625,
        )
        operator_index = int(operator_indices[0])
        sig_trial = np.array([100.0, 60.0, 20.0], dtype=float)
        sig_corrected = np.array([85.0, 58.0, 22.0], dtype=float)
        active_ids = np.array([0, -1, -1], dtype=np.int64)
        old_ok, old_tangent = (
            fem2d_materials_module._mc_consistent_tangent_spectral_numba(
                sig_trial,
                sig_corrected,
                np.eye(3, dtype=float),
                active_ids,
                1,
                yield_coeffs[0],
                flow_coeffs[0],
                d4[0, :3, :3],
                d4[0],
                float(material.hardening),
            )
        )
        new_ok, new_tangent = (
            fem2d_materials_module._mc_consistent_tangent_spectral_precomputed_numba(
                sig_trial,
                sig_corrected,
                np.eye(3, dtype=float),
                active_ids,
                1,
                yield_coeffs[0],
                flow_coeffs[0],
                d4[0, :3, :3],
                d4[0],
                operator1[operator_index],
                operator2[operator_index],
                operator3[operator_index],
                candidate_h[operator_index],
            )
        )
        self.assertTrue(old_ok)
        self.assertTrue(new_ok)
        self.assertTrue(
            np.allclose(
                new_tangent,
                old_tangent,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )

    def test_quad8_plastic_batch_uses_fixed_state_layout_for_full_sri_and_bbar(self) -> None:
        cfg = self.quad8_mixed_integration_chain_config()
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        blocks = build_plastic_element_blocks(mesh, materials)
        self.assertEqual({"FULL", "SRI", "B-BAR"}, {block.integration for block in blocks})
        for block in blocks:
            result = evaluate_plastic_tangent_block(block, mesh, materials, u)
            self.assertEqual(result.dofs.shape[1], 16)
            self.assertEqual(result.ke_values.shape[1], 16 * 16)
            self.assertEqual(result.internal_force_values.shape[1], 16)
            self.assertEqual(result.updated_state_values.shape[1], 13 if block.integration == "SRI" else 9)
            self.assertTrue(np.all(result.status_flags == 0))

    def test_up_batch_contract_exposes_split_coupling_outputs(self) -> None:
        out = empty_up_coupling_block_result(displacement_dofs=8, pressure_dofs=4, element_count=2)
        self.assertEqual(out.kuu_values.shape, (2, 8, 8))
        self.assertEqual(out.kup_values.shape, (2, 8, 4))
        self.assertEqual(out.kpu_values.shape, (2, 4, 8))
        self.assertEqual(out.kpp_values.shape, (2, 4, 4))
        self.assertEqual(out.flow_residual_values.shape, (2, 4))
        self.assertTrue(np.all(out.status_flags == 0))
        cfg = plane_strain_quad4_sample()
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        pressure = np.linspace(1.0, 4.0, len(mesh.node_ids))
        block = evaluate_up_coupling_block(mesh, materials, element_indices=[0], pressure=pressure, storage=0.5, permeability=1.2, biot_alpha=0.8, dt=0.25)
        self.assertEqual(block.kuu_values.shape, (1, 8, 8))
        self.assertEqual(block.kup_values.shape, (1, 8, 4))
        self.assertEqual(block.kpu_values.shape, (1, 4, 8))
        self.assertEqual(block.kpp_values.shape, (1, 4, 4))
        self.assertTrue(np.all(np.isfinite(block.flow_residual_values)))
        self.assertTrue(np.all(block.status_flags == 0))

    def test_large_deformation_stage_accumulates_displacement_and_updates_geometry(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "uy": 0.2},
            ],
            "stages": [
                {
                    "name": "large",
                    "type": "large_deformation",
                    "large_deformation": {"steps": 4, "backend": "vectorized"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
        self.assertEqual(stage.solver_info["method"], "updated_lagrangian")
        history = stage.solver_info["large_deformation"]["history"]
        self.assertEqual(len(history), 4)
        self.assertEqual([row["postprocessed"] for row in history], [False, False, False, True])
        self.assertTrue(all(row["increment_solver"] == "large_deformation_internal_loop" for row in history))
        self.assertEqual(stage.solver_info["large_deformation"]["increment_solver"], "internal_loop")
        self.assertTrue(stage.solver_info["large_deformation"]["adaptive_steps"])
        self.assertEqual(stage.solver_info["large_deformation"]["accepted_steps"], 4)
        self.assertAlmostEqual(stage.solver_info["large_deformation"]["final_load_fraction"], 1.0, places=12)
        self.assertTrue(stage.solver_info["large_deformation"]["skip_intermediate_postprocessing"])
        self.assertTrue(stage.solver_info["large_deformation"]["topology_cache"]["stiffness_pattern_cached"])
        self.assertEqual(stage.solver_info["large_deformation"]["topology_cache"]["batched_quad4_elastic_elements"], 1)
        geometry_cache = stage.solver_info["large_deformation"]["geometry_update_cache"]
        self.assertTrue(geometry_cache["enabled"])
        self.assertTrue(geometry_cache["coordinate_buffer_reused"])
        self.assertEqual(geometry_cache["mesh_object_rebuilds"], 0)
        self.assertEqual(geometry_cache["mode"], "coordinate_buffer_temporary_mesh_coords")
        self.assertGreaterEqual(geometry_cache["updated_coordinate_calls"], 4)
        self.assertEqual(len(stage.element_results), 1)
        self.assertGreaterEqual(len(stage.integration_point_results), 4)
        self.assertTrue(stage.solver_info["postprocess_state_commit"]["integration_point_second_pass_skipped"])
        self.assertAlmostEqual(stage.displacements[2 * 2 + 1], 0.2, places=12)
        self.assertAlmostEqual(stage.displacements[2 * 3 + 1], 0.2, places=12)
        self.assertGreater(stage.solver_info["large_deformation"]["max_displacement_to_model_diagonal"], 0.1)
        self.assertIn("min_detJ", history[-1])
        self.assertEqual(stage.solver_info["large_deformation"]["cutbacks"], 0)
        for key in ("assembly_elapsed_seconds", "linear_solve_elapsed_seconds", "postprocess_elapsed_seconds", "coupled_assembly_elapsed_seconds"):
            self.assertIn(key, stage.solver_info["performance"])
            self.assertIn(key, history[-1])
        log = build_structured_analysis_log(result)
        self.assertIn("large_deformation_increment", {row["event_type"] for row in log["events"]})
        view = build_result_view_index(result)
        self.assertTrue(view["stages"][0]["final_increment_only_postprocess"])
        self.assertIn("fallback_count", view["stages"][0])

    def test_large_deformation_increment_reuses_node_load_and_mpc_penalty_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 2000.0, "nu": 0.3}},
            "boundary_conditions": [{"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0}],
            "loads": [{"node": "3", "fy": -0.5}, {"node": "4", "fy": -0.5}],
            "mpc_constraints": [{"master": "3", "slave": "4", "dof": "uy", "coefficient": 1.0, "penalty": 1.0e7}],
            "stages": [
                {
                    "name": "large-load-mpc-cache",
                    "type": "large_deformation",
                    "large_deformation": {"steps": 3, "adaptive_steps": False, "backend": "vectorized", "precompute_topology": True},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]

        self.assertEqual(stage.solver_info["method"], "updated_lagrangian")
        history = stage.solver_info["large_deformation"]["history"]
        self.assertEqual(len(history), 3)
        self.assertTrue(all(row["load_vector_reused"] for row in history))
        self.assertTrue(all(row["mpc_penalty_reused"] for row in history))
        cache_info = stage.solver_info["large_deformation"]["topology_cache"]["factor_invariant_load_mpc_cache"]
        self.assertTrue(cache_info["enabled"])
        self.assertTrue(cache_info["load_vector_cached"])
        self.assertTrue(cache_info["mpc_penalty_cached"])
        self.assertEqual(cache_info["mpc_equations"], 1)
        self.assertEqual(cache_info["load_vector_reason"], "cached_large_deformation_increment_base_load_vector")
        self.assertEqual(cache_info["mpc_penalty_reason"], "cached_large_deformation_increment_mpc_penalty")

    def test_large_deformation_increment_reuses_edge_body_load_template(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil"},
                ],
                "node_sets": {"base": ["1", "2", "3"], "top": ["4", "5", "6"]},
                "element_sets": {"soil_block": ["e1", "e2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 5000.0, "nu": 0.3, "gamma": 18.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "loads": [
                {"type": "body", "by": -0.02, "set": "soil_block"},
                {"edges": "top", "ty": -0.03},
            ],
            "stages": [
                {
                    "name": "large-load-template-cache",
                    "type": "large_deformation",
                    "large_deformation": {"steps": 3, "adaptive_steps": False, "backend": "vectorized", "precompute_topology": True},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]

        history = stage.solver_info["large_deformation"]["history"]
        self.assertEqual(len(history), 3)
        self.assertTrue(all(row["load_vector_reused"] for row in history))
        cache_info = stage.solver_info["large_deformation"]["topology_cache"]["factor_invariant_load_mpc_cache"]
        self.assertTrue(cache_info["enabled"])
        self.assertFalse(cache_info["load_vector_cached"])
        self.assertTrue(cache_info["load_vector_template_cached"])
        self.assertEqual(cache_info["load_vector_reason"], "cached_geometry_dependent_load_template")
        self.assertEqual(cache_info["load_vector_template_reason"], "cached_large_deformation_updated_coordinate_load_template")
        self.assertEqual(cache_info["load_vector_template"]["body_blocks"], 2)
        self.assertEqual(cache_info["load_vector_template"]["edge_blocks"], 2)

    def test_large_deformation_adaptive_steps_grow_for_small_deformation(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "uy": 0.004},
            ],
            "stages": [{"name": "large-small", "type": "large_deformation", "large_deformation": {"steps": 8, "backend": "vectorized"}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        info = stage.solver_info["large_deformation"]
        self.assertLess(info["accepted_steps"], info["initial_steps"])
        self.assertTrue(any(row["adaptive_action"] == "grow" for row in info["history"]))
        self.assertTrue(info["history"][-1]["postprocessed"])
        self.assertAlmostEqual(info["final_load_fraction"], 1.0, places=12)
        self.assertAlmostEqual(stage.displacements[2 * 2 + 1], 0.004, places=12)

    def test_small_deformation_and_large_deformation_match_in_small_displacement_limit(self) -> None:
        nodes = {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]}
        bcs = []
        for nid, (x, y) in nodes.items():
            bcs.append({"node": nid, "ux": 1.0e-7 * x + 2.0e-8 * y, "uy": -3.0e-8 * y + 4.0e-8 * x})
        base = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": nodes, "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "FULL"}]},
            "materials": {"soil": {"model": "elastic", "E": 50000.0, "nu": 0.30}},
            "boundary_conditions": bcs,
        }
        small_cfg = dict(base, stages=[{"name": "small", "type": "static"}])
        large_cfg = dict(base, stages=[{"name": "large", "type": "large_deformation", "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}])
        with tempfile.TemporaryDirectory() as tmp:
            small = solve_plane_strain_config(small_cfg, Path(tmp) / "small").stages[0]
            large = solve_plane_strain_config(large_cfg, Path(tmp) / "large").stages[0]
        tol = large_deformation_performance_contract()["comparison_tolerances"]["displacement"]
        self.assertTrue(np.allclose(large.displacements, small.displacements, rtol=tol["rtol"], atol=tol["atol"]))
        self.assertTrue(np.allclose(large.reactions, small.reactions, rtol=2.0e-5, atol=1.0e-8))

    def test_large_deformation_quad4_mixed_integration_mesh_runs_through_solver(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [3.0, 0.0],
                    "5": [0.0, 1.0],
                    "6": [1.0, 1.0],
                    "7": [2.0, 1.0],
                    "8": [3.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "6", "5"], "material": "soil", "integration": "FULL"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "7", "6"], "material": "soil", "integration": "SRI"},
                    {"id": "e3", "type": "QUAD4", "nodes": ["3", "4", "8", "7"], "material": "soil", "integration": "B-bar"},
                ],
            },
            "materials": {"soil": {"model": "elastic", "E": 1200.0, "nu": 0.32}},
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "ux": 0.0, "uy": 0.0}, {"nodes": ["5", "6", "7", "8"], "uy": 1.0e-5}],
            "stages": [{"name": "mixed", "type": "large_deformation", "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        self.assertEqual(stage.solver_info["integration"], "mixed:B-BAR,FULL,SRI")
        self.assertEqual(stage.solver_info["large_deformation"]["topology_cache"]["batched_quad4_elastic_elements"], 3)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_large_deformation_quad8_mixed_integration_mesh_runs_through_solver(self) -> None:
        cfg = self.quad8_mixed_integration_chain_config()
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        self.assertEqual(stage.solver_info["element_type"], "QUAD8")
        self.assertEqual(stage.solver_info["integration"], "mixed:B-BAR,FULL,SRI")
        self.assertEqual(stage.solver_info["large_deformation"]["topology_cache"]["element_type_counts"]["QUAD8"], 3)
        self.assertTrue(np.all(np.isfinite(stage.reactions)))

    def test_large_deformation_hydro_stage_keeps_pressure_cache_and_history_contract(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"], "top": ["3", "4"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "permeability": 1.0, "storage": 1.0, "biot": 1.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}, {"set": "top", "uy": -1.0e-5}],
            "stages": [{"name": "large-up", "type": "large_deformation", "hydro": {"initial_pressure": 10.0}, "large_deformation": {"steps": 2, "adaptive_steps": False, "backend": "vectorized"}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        self.assertIsNotNone(stage.pore_pressure)
        self.assertTrue(stage.solver_info["hydro_coupled"])
        hydro_cache = stage.solver_info["large_deformation"]["topology_cache"]["hydro_cache"]
        self.assertTrue(hydro_cache["enabled"])
        self.assertEqual(hydro_cache["pressure_dofs"], 4)
        pressure_load_cache = hydro_cache["pore_pressure_load_cache"]
        self.assertTrue(pressure_load_cache["enabled"])
        self.assertTrue(pressure_load_cache["direct_vector_scatter"]["enabled"])
        self.assertTrue(stage.solver_info["pore_pressure_load_cache"]["reused"])
        self.assertTrue(all("coupled_assembly_elapsed_seconds" in row for row in stage.solver_info["large_deformation"]["history"]))
        self.assertTrue(all(row["pore_pressure_load_cache_enabled"] for row in stage.solver_info["large_deformation"]["history"]))
        self.assertTrue(all(row["pore_pressure_load_reused"] for row in stage.solver_info["large_deformation"]["history"]))
        self.assertTrue(all("pore_pressure_load_assembly_elapsed_seconds" in row for row in stage.solver_info["large_deformation"]["history"]))
        up_info = stage.solver_info["large_deformation"]["up_coupling"]
        self.assertEqual(up_info["increment_solver"], "large_deformation_up_internal_loop")
        self.assertEqual(up_info["pressure_dof_count"], 4)
        self.assertTrue(all("pressure_dof_count" in row for row in stage.solver_info["large_deformation"]["history"]))

    def test_large_deformation_cutback_retries_before_failure(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"], "top": ["3", "4"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 50000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}, {"node": "1", "dof": "bad", "value": 0.0}],
            "stages": [
                {
                    "name": "large-cutback-fail",
                    "type": "large_deformation",
                    "large_deformation": {"steps": 1, "adaptive_steps": True, "initial_step": 1.0, "min_step": 0.1, "max_cutbacks": 1, "backend": "vectorized", "precompute_topology": False},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FEM2DError, "cutback"):
                solve_plane_strain_config(cfg, tmp)

    def test_large_deformation_tension_cutoff_and_srm_below_one_are_covered(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"], "top": ["3", "4"]},
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 1.0, "friction_angle": 20.0, "tension_cutoff": 0.1}},
            "boundary_conditions": [
                {"node": "1", "ux": 0.0, "uy": 0.0},
                {"node": "2", "ux": 0.0, "uy": 0.0},
                {"node": "3", "ux": 0.0, "uy": 1.0e-4},
                {"node": "4", "ux": 0.0, "uy": 1.0e-4},
            ],
            "stages": [{"name": "large-tension", "type": "large_deformation", "solver": {"large_deformation": {"enabled": True, "steps": 2, "adaptive_steps": False, "backend": "vectorized", "precompute_topology": False}}, "srm": {"factors": [0.8, 0.9], "failure_plastic_ratio": 1.0, "parallel": {"enabled": True, "max_workers": 2}}}],
        }
        cfg["stages"][0]["type"] = "srm"
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        serial_cfg = json.loads(json.dumps(cfg))
        serial_cfg["stages"][0]["srm"]["parallel"] = {"enabled": False, "max_workers": 1}
        with tempfile.TemporaryDirectory() as tmp:
            serial_stage = solve_plane_strain_config(serial_cfg, tmp).stages[0]
        self.assertAlmostEqual(
            stage.solver_info["srm"]["factor_of_safety"],
            serial_stage.solver_info["srm"]["factor_of_safety"],
            places=12,
        )
        self.assertTrue(np.allclose(stage.displacements, serial_stage.displacements, rtol=1.0e-12, atol=1.0e-14))
        self.assertLess(stage.solver_info["strength_factor"], 1.0)
        self.assertEqual(stage.solver_info["method"], "updated_lagrangian")
        self.assertEqual(stage.solver_info["material_model"], "drucker_prager")
        self.assertTrue(stage.solver_info["srm"]["factor_of_safety"] < 1.0)
        self.assertTrue(stage.solver_info["srm"]["factor_cache"]["enabled"])
        self.assertTrue(stage.solver_info["srm"]["factor_cache"]["shared_across_trials"])
        self.assertTrue(stage.solver_info["srm"]["factor_cache"]["shared_across_srm_factors"])
        self.assertEqual(stage.solver_info["srm"]["factor_cache"]["cache_kind"], "large_deformation_step_cache")
        self.assertTrue(stage.solver_info["srm"]["factor_cache"]["stiffness_pattern_cached"])
        self.assertTrue(stage.solver_info["srm"]["factor_cache"]["reduced_matrix_cached"])
        self.assertEqual(stage.solver_info["srm"]["factor_cache"]["reuse_scope"], "srm_factor_trials")
        self.assertTrue(stage.solver_info["srm"]["factor_cache"]["topology_cache_id"])
        parallel = stage.solver_info["srm"]["parallel"]
        self.assertTrue(parallel["enabled"])
        self.assertTrue(parallel["parallel_requested"])
        self.assertFalse(parallel["thread_safety_guard_active"])
        self.assertEqual(parallel["max_workers"], 2)
        self.assertEqual(parallel["disabled_reason"], "")
        workspace = stage.solver_info["srm"]["trial_workspace"]
        self.assertTrue(workspace["independent_geometry"])
        self.assertTrue(workspace["coordinate_buffer_per_trial"])
        self.assertTrue(workspace["thread_safe_parallel_trials"])
        self.assertTrue(workspace["topology_cache_shared"])
        large_load_mpc_cache = stage.solver_info["srm"]["factor_cache"]["factor_invariant_load_mpc_cache"]
        self.assertTrue(large_load_mpc_cache["enabled"])
        self.assertTrue(large_load_mpc_cache["load_vector_cached"])
        self.assertTrue(large_load_mpc_cache["mpc_penalty_cached"])
        self.assertTrue(stage.solver_info["sparse_pattern_cached"])
        self.assertEqual(stage.solver_info["large_deformation"]["topology_cache"]["cache_kind"], "large_deformation_step_cache")
        self.assertTrue(stage.solver_info["large_deformation"]["topology_cache"]["stiffness_pattern_cached"])
        self.assertTrue(stage.solver_info["large_deformation"]["topology_cache_supplied"])
        self.assertTrue(stage.solver_info["large_deformation"]["topology_cache_reused"])
        self.assertTrue(stage.solver_info["large_deformation"]["topology_shared_across_srm_factors"])
        self.assertEqual(
            stage.solver_info["large_deformation"]["topology_cache"]["topology_cache_id"],
            stage.solver_info["srm"]["factor_cache"]["topology_cache_id"],
        )
        self.assertEqual(
            {row["topology_cache_id"] for row in stage.solver_info["srm"]["trials"]},
            {stage.solver_info["srm"]["factor_cache"]["topology_cache_id"]},
        )
        self.assertTrue(all(row["topology_shared_across_srm_factors"] for row in stage.solver_info["srm"]["trials"]))
        self.assertTrue(stage.solver_info["srm"]["lightweight_postprocess"]["enabled"])
        self.assertTrue(stage.solver_info["postprocess_results"])
        self.assertTrue(all(row["postprocess_results"] is False for row in stage.solver_info["srm"]["trials"]))
        self.assertTrue(all(row["plastic_ratio_source"] == "plastic_state_array_cache" for row in stage.solver_info["srm"]["trials"]))
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["constraint_dofs_cached"])
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["reduced_matrix_cache_enabled"])
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["topology_shared_across_srm_factors"])
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["load_vector_reused"])
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["mpc_penalty_reused"])
        self.assertTrue(stage.solver_info["plastic_state_array_cache"]["enabled"])
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["plastic_state_array_cache_enabled"])
        self.assertTrue(stage.solver_info["large_deformation"]["history"][-1]["postprocessed"])

    def test_element_state_update_can_skip_postprocess_rows(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 1000.0, "nu": 0.3}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        disp = np.array([0.0, 0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.2], dtype=float)
        info: dict[str, object] = {}
        rows, state = compute_element_results_and_state(mesh, materials, disp, collect_results=False, postprocess_info=info)
        self.assertEqual(rows, [])
        self.assertEqual(state, {})
        self.assertEqual(info["state_commit"], "elastic_array_no_state")
        self.assertGreaterEqual(info["elastic_state_skipped_points"], 4)

    def test_final_postprocess_commits_state_with_array_fast_path(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "full", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "FULL"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 80.0, "friction_angle": 28.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        initial = {
            "full": np.array([1.0, -0.5, 0.25, 0.2], dtype=float),
            "bbar": np.array([-0.2, 0.4, 0.1, 0.15], dtype=float),
        }
        fast_info: dict[str, object] = {}
        loop_info: dict[str, object] = {}
        fast_rows, fast_state = compute_element_results_and_state(
            mesh,
            materials,
            u,
            initial_stresses=initial,
            strength_factor=1.15,
            postprocess_info=fast_info,
        )
        loop_rows, loop_state = compute_element_results_and_state(
            mesh,
            materials,
            u,
            initial_stresses=initial,
            strength_factor=1.15,
            postprocess_info=loop_info,
            use_array_postprocess=False,
        )

        self.assertEqual(fast_info["state_commit"], "array_batch")
        self.assertEqual(fast_info["array_committed_elements"], 2)
        self.assertEqual(fast_info["array_row_elements"], 2)
        self.assertEqual(fast_info["row_generation"], "array_backed_lazy")
        self.assertEqual(fast_info["state_mapping"], "array_backed_lazy")
        self.assertFalse(fast_info["dict_materialized"])
        self.assertFalse(isinstance(fast_state, dict))
        self.assertEqual(loop_info["state_commit"], "element_loop")
        self.assertFalse(isinstance(fast_rows[0], dict))
        self.assertEqual(len(fast_rows), len(loop_rows))
        numeric_keys = ["eps_x", "eps_y", "eps_z", "gamma_xy", "sigma_x", "sigma_y", "sigma_z", "tau_xy", "plastic", "yield_value", "p", "q"]
        for fast_row, loop_row in zip(fast_rows, loop_rows, strict=True):
            for key in numeric_keys:
                self.assertAlmostEqual(float(fast_row[key]), float(loop_row[key]), delta=1.0e-8)
        self.assertEqual(set(fast_state), set(loop_state))
        for key, state in fast_state.items():
            self.assertTrue(np.allclose(state.plastic_strain, loop_state[key].plastic_strain))
            self.assertAlmostEqual(state.kappa, loop_state[key].kappa, delta=1.0e-12)

    def test_elastic_final_postprocess_uses_array_backed_lazy_rows(self) -> None:
        for element_type, integration in (("QUAD4", "B-bar"), ("QUAD8", "B-bar")):
            with self.subTest(element_type=element_type):
                cfg = self.quad8_rectangle_config(integration) if element_type == "QUAD8" else {
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": integration}],
                    },
                    "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                u = self.displacement_from_coords(mesh.coords)
                fast_info: dict[str, object] = {}
                loop_info: dict[str, object] = {}
                fast_rows, fast_state = compute_element_results_and_state(mesh, materials, u, postprocess_info=fast_info)
                loop_rows, loop_state = compute_element_results_and_state(mesh, materials, u, postprocess_info=loop_info, use_array_postprocess=False)

                self.assertEqual(fast_state, {})
                self.assertGreaterEqual(len(loop_state), 4)
                self.assertEqual(fast_info["state_commit"], "elastic_array_no_state")
                self.assertEqual(fast_info["row_generation"], "array_backed_lazy")
                self.assertFalse(isinstance(fast_rows[0], dict))
                for key in ("eps_x", "eps_y", "eps_z", "gamma_xy", "sigma_x", "sigma_y", "sigma_z", "tau_xy", "p", "q"):
                    self.assertAlmostEqual(float(fast_rows[0][key]), float(loop_rows[0][key]), delta=1.0e-8)

    def test_final_postprocess_can_emit_integration_point_rows_in_same_pass(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {
                    "1": [0.0, 0.0],
                    "2": [1.0, 0.0],
                    "3": [2.0, 0.0],
                    "4": [0.0, 1.0],
                    "5": [1.0, 1.0],
                    "6": [2.0, 1.0],
                },
                "elements": [
                    {"id": "full", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil", "integration": "FULL"},
                    {"id": "bbar", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil", "integration": "B-BAR"},
                ],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.3, "cohesion": 80.0, "friction_angle": 28.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = self.displacement_from_coords(mesh.coords)
        info: dict[str, object] = {}
        element_rows, _state = compute_element_results_and_state(
            mesh,
            materials,
            u,
            postprocess_info=info,
            collect_integration_point_rows=True,
        )
        same_pass_rows = info.pop("_integration_point_rows")
        normal_rows = compute_integration_point_results(mesh, materials, u)

        self.assertEqual(len(element_rows), 2)
        self.assertEqual(info["integration_point_row_generation"], "array_post_data")
        self.assertTrue(info["integration_point_second_pass_skipped"])
        self.assertEqual(len(same_pass_rows), len(normal_rows))
        for same_pass, normal in zip(same_pass_rows, normal_rows, strict=True):
            self.assertEqual(same_pass["element_id"], normal["element_id"])
            self.assertEqual(same_pass["ip"], normal["ip"])
            for key in ("eps_x", "eps_y", "eps_z", "gamma_xy", "sigma_x", "sigma_y", "sigma_z", "tau_xy", "p", "q", "plastic", "yield_value"):
                self.assertAlmostEqual(float(same_pass[key]), float(normal[key]), delta=1.0e-8)

    def test_mohr_coulomb_final_postprocess_fast_path_matches_element_loop(self) -> None:
        cases = [
            (
                "quad4-full",
                {
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "q4", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "FULL"}],
                    },
                    "materials": {"soil": {"model": "mohr_coulomb", "E": 50000.0, "nu": 0.30, "cohesion": 200.0, "friction_angle": 28.0, "dilation_angle": 5.0}},
                },
            ),
            (
                "quad8-full",
                self.quad8_rectangle_config(
                    "FULL",
                    material={"model": "mohr_coulomb", "E": 50000.0, "nu": 0.30, "cohesion": 200.0, "friction_angle": 28.0, "dilation_angle": 5.0},
                ),
            ),
        ]
        numeric_keys = ("eps_x", "eps_y", "eps_z", "gamma_xy", "sigma_x", "sigma_y", "sigma_z", "tau_xy", "plastic", "yield_value", "p", "q")
        for name, cfg in cases:
            with self.subTest(name=name):
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                u = self.displacement_from_coords(mesh.coords)
                initial = {element.id: np.array([0.5, -0.2, 0.15, 0.05], dtype=float) for element in mesh.elements}
                fast_info: dict[str, object] = {}
                loop_info: dict[str, object] = {}
                fast_rows, fast_state = compute_element_results_and_state(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial,
                    strength_factor=1.10,
                    postprocess_info=fast_info,
                )
                loop_rows, loop_state = compute_element_results_and_state(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial,
                    strength_factor=1.10,
                    postprocess_info=loop_info,
                    use_array_postprocess=False,
                )

                self.assertEqual(fast_info["state_commit"], "array_batch")
                self.assertEqual(fast_info["row_generation"], "array_backed_lazy")
                self.assertFalse(fast_info["dict_materialized"])
                self.assertEqual(loop_info["state_commit"], "element_loop")
                self.assertEqual(len(fast_rows), len(loop_rows))
                for fast_row, loop_row in zip(fast_rows, loop_rows, strict=True):
                    self.assertEqual(fast_row["element_id"], loop_row["element_id"])
                    for key in numeric_keys:
                        self.assertAlmostEqual(float(fast_row[key]), float(loop_row[key]), delta=1.0e-8)
                self.assertEqual(set(fast_state), set(loop_state))
                for key, state in fast_state.items():
                    self.assertTrue(np.allclose(state.plastic_strain, loop_state[key].plastic_strain, rtol=1.0e-10, atol=1.0e-10))
                    self.assertAlmostEqual(state.kappa, loop_state[key].kappa, delta=1.0e-12)
                fast_ip_info: dict[str, object] = {}
                loop_ip_info: dict[str, object] = {}
                compute_element_results_and_state(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial,
                    strength_factor=1.10,
                    postprocess_info=fast_ip_info,
                    collect_integration_point_rows=True,
                )
                compute_element_results_and_state(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial,
                    strength_factor=1.10,
                    postprocess_info=loop_ip_info,
                    use_array_postprocess=False,
                    collect_integration_point_rows=True,
                )
                fast_ip_rows = fast_ip_info["_integration_point_rows"]
                loop_ip_rows = loop_ip_info["_integration_point_rows"]
                self.assertEqual(len(fast_ip_rows), len(loop_ip_rows))
                for fast_row, loop_row in zip(fast_ip_rows, loop_ip_rows, strict=True):
                    self.assertEqual(fast_row["state_key"], loop_row["state_key"])
                    for key in (*numeric_keys, "kappa", "plastic_strain_x", "plastic_strain_y", "plastic_strain_z", "plastic_strain_gamma_xy"):
                        self.assertAlmostEqual(float(fast_row[key]), float(loop_row[key]), delta=1.0e-8)

    def test_quad4_mass_matrix_uses_fast_consistent_kernel(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 9.80665}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        mass = assemble_mass_matrix(mesh, materials)
        lumped = assemble_mass_matrix(mesh, materials, lumped=True)
        self.assertEqual(mass.shape, (8, 8))
        self.assertAlmostEqual(float(mass.sum()), 2.0, places=12)
        self.assertAlmostEqual(float(lumped.sum()), 2.0, places=12)
        self.assertAlmostEqual(float(mass[0, 1]), 0.0, places=12)
        self.assertGreater(float(mass[0, 2]), 0.0)

    def test_mass_matrix_cache_matches_builder_and_batches_quad4(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 4.0],
                "y_range": [0.0, 2.0],
                "nx": 4,
                "ny": 2,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 9.80665}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_mass_matrix_assembly_cache(mesh, materials)
        cached = assemble_mass_matrix_cached(cache, mesh, materials)
        reference = assemble_mass_matrix(mesh, materials)
        self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(np.allclose(assemble_mass_matrix_cached(cache, mesh, materials, lumped=True).toarray(), assemble_mass_matrix(mesh, materials, lumped=True).toarray()))
        info = cache.info()
        self.assertEqual(info["batched_elements"], 8)
        self.assertTrue(info["direct_fill"]["enabled"])

    def test_mass_matrix_cache_batches_tri_and_precomputes_structural_mass(self) -> None:
        for element_type in ("TRI3", "TRI6"):
            with self.subTest(element_type=element_type):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [0.0, 2.0],
                        "y_range": [0.0, 1.0],
                        "nx": 2,
                        "ny": 2,
                        "element_type": element_type,
                        "material": "soil",
                    },
                    "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 9.80665}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                cache = build_mass_matrix_assembly_cache(mesh, materials)
                cached = assemble_mass_matrix_cached(cache, mesh, materials)
                reference = assemble_mass_matrix(mesh, materials)
                self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-12, atol=1.0e-12))
                self.assertEqual(cache.info()["batched_elements"], len(mesh.elements))
                self.assertEqual(cache.info()["batch_groups"][0]["element_type"], element_type)

                small_cfg = dict(cfg)
                small_cfg["mesh"] = {**cfg["mesh"], "nx": 1, "ny": 1}
                small_mesh = mesh_from_config(small_cfg)
                small_cache = build_mass_matrix_assembly_cache(small_mesh, materials)
                self.assertEqual(small_cache.info()["batched_elements"], 0)
                small_cached = assemble_mass_matrix_cached(small_cache, small_mesh, materials)
                small_reference = assemble_mass_matrix(small_mesh, materials)
                self.assertTrue(np.allclose(small_cached.toarray(), small_reference.toarray(), rtol=1.0e-12, atol=1.0e-12))

        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [2.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        structural = [StructuralElement2D(id="bar", type="BAR2", nodes=("1", "2"), section={"mass_per_length": 3.0})]
        cache = build_mass_matrix_assembly_cache(mesh, materials, structural_elements=structural)
        cached = assemble_mass_matrix_cached(cache, mesh, materials, structural_elements=structural)
        reference = assemble_mass_matrix(mesh, materials, structural_elements=structural)
        self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-12, atol=1.0e-12))
        self.assertEqual(cache.info()["precomputed_structural_mass_blocks"], 1)

    def test_tri_pressure_biot_and_mass_kernels_match_reference_integrals(self) -> None:
        for element_type in ("TRI3", "TRI6"):
            with self.subTest(element_type=element_type):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [0.0, 1.0],
                        "y_range": [0.0, 1.0],
                        "nx": 1,
                        "ny": 1,
                        "element_type": element_type,
                        "material": "soil",
                    },
                    "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 9.80665}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                density = material.gamma / 9.80665
                storage = 0.8
                permeability = 0.05
                alpha = 0.65
                pressure_mass_expected = np.zeros((len(mesh.node_ids), len(mesh.node_ids)), dtype=float)
                pressure_cond_expected = np.zeros_like(pressure_mass_expected)
                mass_expected = np.zeros((len(mesh.node_ids) * 2, len(mesh.node_ids) * 2), dtype=float)
                biot_expected = np.zeros((len(mesh.node_ids) * 2, len(mesh.node_ids)), dtype=float)
                m = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
                for element in mesh.elements:
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    for gp in integration_points(element.type, "FULL"):
                        N, dN = shape_functions(element.type, gp[0], gp[1])
                        jac = dN @ coords
                        detJ = float(np.linalg.det(jac))
                        grad = np.linalg.inv(jac) @ dN
                        B4, detB, N_b = strain_displacement_matrix(element.type, coords, gp)
                        self.assertAlmostEqual(detB, detJ, places=12)
                        self.assertTrue(np.allclose(N_b, N, rtol=1.0e-12, atol=1.0e-12))
                        dV = detJ * gp[2] * material.thickness
                        for a, Na in enumerate(N):
                            for b, Nb in enumerate(N):
                                pressure_mass_expected[conn[a], conn[b]] += storage * Na * Nb * dV
                                pressure_cond_expected[conn[a], conn[b]] += permeability * float(grad[:, a] @ grad[:, b]) * dV
                                value = density * Na * Nb * dV
                                mass_expected[2 * conn[a], 2 * conn[b]] += value
                                mass_expected[2 * conn[a] + 1, 2 * conn[b] + 1] += value
                        biot_expected[np.ix_(dofs, conn)] += B4.T @ (alpha * np.outer(m, N)) * dV

                pressure_mass, pressure_cond = assemble_pressure_matrices(mesh, materials, storage=storage, permeability=permeability)
                mass = assemble_mass_matrix(mesh, materials)
                biot = assemble_biot_coupling_matrix(mesh, materials, alpha=alpha)
                self.assertTrue(np.allclose(pressure_mass.toarray(), pressure_mass_expected, rtol=1.0e-11, atol=1.0e-10))
                self.assertTrue(np.allclose(pressure_cond.toarray(), pressure_cond_expected, rtol=1.0e-11, atol=1.0e-10))
                self.assertTrue(np.allclose(mass.toarray(), mass_expected, rtol=1.0e-11, atol=1.0e-10))
                self.assertTrue(np.allclose(biot.toarray(), biot_expected, rtol=1.0e-11, atol=1.0e-10))

    def test_quad4_plane_strain_elastic_tangent_and_internal_force_use_fast_kernels(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        u = np.array([0.0, 0.0, 0.001, 0.0, 0.0012, -0.0004, 0.0, -0.0003], dtype=float)

        tangent = assemble_algorithmic_tangent_stiffness(mesh, materials, u)
        internal = assemble_internal_force(mesh, materials, u)

        expected_tangent = np.zeros((len(u), len(u)), dtype=float)
        ke = element_stiffness("QUAD4", coords, material, "FULL")
        expected_tangent[np.ix_(dofs, dofs)] = ke
        expected_internal = np.zeros_like(u)
        expected_internal[dofs] = ke @ u[dofs]
        self.assertTrue(np.allclose(tangent.toarray(), expected_tangent, rtol=1.0e-11, atol=1.0e-8))
        self.assertTrue(np.allclose(internal, expected_internal, rtol=1.0e-11, atol=1.0e-8))

    def test_quad8_plane_strain_sri_bbar_elastic_tangent_and_internal_force_follow_integration_mode(self) -> None:
        for integration in ("SRI", "B-bar"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config(integration)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords)

                tangent = assemble_algorithmic_tangent_stiffness(mesh, materials, u)
                internal = assemble_internal_force(mesh, materials, u)

                ke = element_stiffness("QUAD8", coords, material, integration)
                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = ke
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = ke @ u[dofs]
                self.assertTrue(np.allclose(tangent.toarray(), expected_tangent, rtol=1.0e-10, atol=1.0e-8))
                self.assertTrue(np.allclose(internal, expected_internal, rtol=1.0e-10, atol=1.0e-8))

    def test_quad8_plane_strain_elastic_kernels_match_reference_integrals(self) -> None:
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                expected = np.zeros((16, 16), dtype=float)
                Pvol = material.volumetric_projector
                if integration == "FULL":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected += B4.T @ material.D4 @ B4 * detJ * gp[2] * material.thickness
                elif integration == "SRI":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected += B4.T @ material.C_dev @ B4 * detJ * gp[2] * material.thickness
                    for gp in integration_points("QUAD8", "REDUCED"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        Bv = Pvol @ B4
                        expected += Bv.T @ material.C_vol @ Bv * detJ * gp[2] * material.thickness
                else:
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[np.ndarray, float]] = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((B4, dV))
                    Bv_bar = Bv_acc / volume
                    Pdev = np.eye(4) - Pvol
                    for B4, dV in cached:
                        Bdev = Pdev @ B4
                        expected += Bdev.T @ material.C_dev @ Bdev * dV
                        expected += Bv_bar.T @ material.C_vol @ Bv_bar * dV
                fast = _quad8_element_stiffness_fast(coords, material, integration)
                self.assertTrue(np.allclose(fast, 0.5 * (expected + expected.T), rtol=1.0e-11, atol=1.0e-8))

    def test_quad8_plane_strain_mass_pressure_and_biot_kernels_match_reference_integrals(self) -> None:
        cfg = self.quad8_rectangle_config("FULL", {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 9.80665})
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        density = 1.0
        storage = 2.0
        permeability = 0.5
        alpha = 0.7
        mass_expected = np.zeros((16, 16), dtype=float)
        pressure_mass_expected = np.zeros((8, 8), dtype=float)
        pressure_cond_expected = np.zeros((8, 8), dtype=float)
        biot_expected = np.zeros((16, 8), dtype=float)
        m = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        for gp in integration_points("QUAD8", "FULL"):
            N, dN = shape_functions("QUAD8", gp[0], gp[1])
            jac = dN @ coords
            detJ = float(np.linalg.det(jac))
            grad = np.linalg.inv(jac) @ dN
            dV = detJ * gp[2] * material.thickness
            B4, detB, N_b = strain_displacement_matrix("QUAD8", coords, gp)
            self.assertAlmostEqual(detB, detJ, places=12)
            self.assertTrue(np.allclose(N_b, N, rtol=1.0e-12, atol=1.0e-12))
            for a, Na in enumerate(N):
                for b, Nb in enumerate(N):
                    value = density * Na * Nb * dV
                    mass_expected[2 * a, 2 * b] += value
                    mass_expected[2 * a + 1, 2 * b + 1] += value
                    pressure_mass_expected[a, b] += storage * Na * Nb * dV
                    pressure_cond_expected[a, b] += permeability * float(grad[:, a] @ grad[:, b]) * dV
            biot_expected += B4.T @ (alpha * np.outer(m, N)) * dV
        mass_fast = _quad8_consistent_mass_matrix_fast(coords, material, density)
        pressure_mass_fast, pressure_cond_fast = _quad8_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
        biot_fast = _quad8_biot_matrix_fast(coords, material, alpha)
        self.assertTrue(np.allclose(mass_fast, mass_expected, rtol=1.0e-11, atol=1.0e-10))
        self.assertTrue(np.allclose(pressure_mass_fast, pressure_mass_expected, rtol=1.0e-11, atol=1.0e-10))
        self.assertTrue(np.allclose(pressure_cond_fast, pressure_cond_expected, rtol=1.0e-11, atol=1.0e-10))
        self.assertTrue(np.allclose(biot_fast, biot_expected, rtol=1.0e-11, atol=1.0e-10))

    def test_quad8_plane_strain_biot_kernel_follows_sri_and_bbar_modes(self) -> None:
        alpha = 0.7
        m = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                expected = np.zeros((16, 8), dtype=float)
                Pvol = material.volumetric_projector
                if integration == "SRI":
                    for gp in integration_points("QUAD8", "REDUCED"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        Bv = Pvol @ B4
                        expected += Bv.T @ (alpha * np.outer(m, N)) * detJ * gp[2] * material.thickness
                elif integration == "B-BAR":
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[np.ndarray, float]] = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((N, dV))
                    Bv_bar = Bv_acc / volume
                    for N, dV in cached:
                        expected += Bv_bar.T @ (alpha * np.outer(m, N)) * dV
                else:
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected += B4.T @ (alpha * np.outer(m, N)) * detJ * gp[2] * material.thickness

                fast = _quad8_biot_matrix_fast(coords, material, alpha, integration)
                assembled = assemble_biot_coupling_matrix(mesh, materials, alpha).toarray()
                expected_global = np.zeros_like(assembled)
                expected_global[np.ix_(dofs, conn)] = expected
                self.assertTrue(np.allclose(fast, expected, rtol=1.0e-11, atol=1.0e-10))
                self.assertTrue(np.allclose(assembled, expected_global, rtol=1.0e-11, atol=1.0e-10))

    def test_pressure_direct_fill_caches_match_builder_assemblies(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 8,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        pressure_cache = build_pressure_matrix_assembly_cache(mesh)
        pressure_info = pressure_cache.info()
        self.assertEqual(pressure_info["batched_elements"], 8)
        self.assertEqual(pressure_info["batch_groups"][0]["element_type"], "QUAD4")
        mass_cached, conductivity_cached = assemble_pressure_matrices_cached(pressure_cache, mesh, materials, storage=1.2, permeability=0.04)
        mass_expected, conductivity_expected = assemble_pressure_matrices(mesh, materials, storage=1.2, permeability=0.04)
        self.assertTrue(np.allclose(mass_cached.toarray(), mass_expected.toarray()))
        self.assertTrue(np.allclose(conductivity_cached.toarray(), conductivity_expected.toarray()))

        biot_cache = build_biot_coupling_assembly_cache(mesh)
        biot_info = biot_cache.info()
        self.assertEqual(biot_info["batched_elements"], 8)
        self.assertEqual(biot_info["batch_groups"][0]["element_type"], "QUAD4")
        biot_cached = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=0.8)
        biot_expected = assemble_biot_coupling_matrix(mesh, materials, alpha=0.8)
        self.assertTrue(np.allclose(biot_cached.toarray(), biot_expected.toarray()))

        pore_load_cache = build_pore_pressure_load_cache(mesh)
        pore_load_info = pore_load_cache.info()
        self.assertEqual(pore_load_info["batched_elements"], 8)
        self.assertTrue(pore_load_info["direct_vector_scatter"]["enabled"])
        pore_pressure = np.linspace(1.0, 2.0, len(mesh.node_ids), dtype=float)
        displacement = np.linspace(0.0, 0.02, len(mesh.node_ids) * 2, dtype=float)
        updated = mesh_with_updated_coords(mesh, displacement, backend="vectorized")
        pore_load_cached = assemble_pore_pressure_load_cached(pore_load_cache, updated, materials, pore_pressure, alpha=0.8)
        pore_load_expected = assemble_biot_coupling_matrix(updated, materials, alpha=0.8) @ pore_pressure
        self.assertTrue(np.allclose(pore_load_cached, pore_load_expected, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(np.allclose(assemble_pore_pressure_load(updated, materials, pore_pressure, alpha=0.8), pore_load_expected, rtol=1.0e-12, atol=1.0e-12))

        hydro = {
            "pore_flux_bcs": [{"set": "bottom", "flux": 2.0}],
            "pore_robin_bcs": [{"set": "top", "beta": 0.5, "pressure": -10.0}],
        }
        boundary_cache = build_pressure_boundary_term_cache(mesh, hydro)
        boundary_cached, rhs_cached, info_cached = assemble_pressure_boundary_terms_cached(boundary_cache)
        boundary_expected, rhs_expected, info_expected = assemble_pressure_boundary_terms(mesh, hydro)
        self.assertTrue(np.allclose(boundary_cached.toarray(), boundary_expected.toarray()))
        self.assertTrue(np.allclose(rhs_cached, rhs_expected))
        self.assertEqual(info_cached["direct_fill"]["cache_kind"], "pressure_boundary_term_cache")
        self.assertAlmostEqual(info_cached["flux_total"], info_expected["flux_total"])
        self.assertAlmostEqual(info_cached["robin_conductance_total"], info_expected["robin_conductance_total"])

    def test_pressure_and_biot_cache_batches_tri3_tri6(self) -> None:
        for element_type in ("TRI3", "TRI6"):
            with self.subTest(element_type=element_type):
                cfg = {
                    "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [0.0, 2.0],
                        "y_range": [0.0, 1.0],
                        "nx": 2,
                        "ny": 2,
                        "element_type": element_type,
                        "material": "soil",
                    },
                    "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                pressure_cache = build_pressure_matrix_assembly_cache(mesh)
                pressure_info = pressure_cache.info()
                self.assertEqual(pressure_info["batched_elements"], len(mesh.elements))
                self.assertEqual(pressure_info["batch_groups"][0]["element_type"], element_type)
                mass_cached, conductivity_cached = assemble_pressure_matrices_cached(pressure_cache, mesh, materials, storage=0.8, permeability=0.05)
                mass_expected, conductivity_expected = assemble_pressure_matrices(mesh, materials, storage=0.8, permeability=0.05)
                self.assertTrue(np.allclose(mass_cached.toarray(), mass_expected.toarray(), rtol=1.0e-11, atol=1.0e-10))
                self.assertTrue(np.allclose(conductivity_cached.toarray(), conductivity_expected.toarray(), rtol=1.0e-11, atol=1.0e-10))

                biot_cache = build_biot_coupling_assembly_cache(mesh)
                biot_info = biot_cache.info()
                self.assertEqual(biot_info["batched_elements"], len(mesh.elements))
                self.assertEqual(biot_info["batch_groups"][0]["element_type"], element_type)
                biot_cached = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=0.65)
                biot_expected = assemble_biot_coupling_matrix(mesh, materials, alpha=0.65)
                self.assertTrue(np.allclose(biot_cached.toarray(), biot_expected.toarray(), rtol=1.0e-11, atol=1.0e-10))

                pore_load_cache = build_pore_pressure_load_cache(mesh)
                self.assertEqual(pore_load_cache.info()["batched_elements"], len(mesh.elements))
                pore_pressure = np.linspace(0.25, 1.5, len(mesh.node_ids), dtype=float)
                pore_load_cached = assemble_pore_pressure_load_cached(pore_load_cache, mesh, materials, pore_pressure, alpha=0.65)
                pore_load_expected = assemble_biot_coupling_matrix(mesh, materials, alpha=0.65) @ pore_pressure
                self.assertTrue(np.allclose(pore_load_cached, pore_load_expected, rtol=1.0e-11, atol=1.0e-10))

    def test_axisymmetric_pressure_direct_fill_caches_match_builder_assemblies(self) -> None:
        cfg = self.quad8_rectangle_config("FULL", axisymmetric=True)
        cfg["mesh"]["nx"] = 3
        cfg["mesh"]["ny"] = 3
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        pressure_cache = build_pressure_matrix_assembly_cache(mesh, axisymmetric=True)
        pressure_info = pressure_cache.info()
        self.assertEqual(pressure_info["batched_elements"], 9)
        self.assertEqual(pressure_info["batch_groups"][0]["element_type"], "QUAD8")
        mass_cached, conductivity_cached = assemble_pressure_matrices_cached(pressure_cache, mesh, materials, storage=0.9, permeability=0.03)
        mass_expected, conductivity_expected = assemble_axisymmetric_pressure_matrices(mesh, materials, storage=0.9, permeability=0.03)
        self.assertTrue(np.allclose(mass_cached.toarray(), mass_expected.toarray()))
        self.assertTrue(np.allclose(conductivity_cached.toarray(), conductivity_expected.toarray()))

        biot_cache = build_biot_coupling_assembly_cache(mesh, axisymmetric=True)
        biot_info = biot_cache.info()
        self.assertEqual(biot_info["batched_elements"], 9)
        self.assertEqual(biot_info["batch_groups"][0]["element_type"], "QUAD8")
        biot_cached = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=0.75)
        biot_expected = assemble_axisymmetric_biot_coupling_matrix(mesh, materials, alpha=0.75)
        self.assertTrue(np.allclose(biot_cached.toarray(), biot_expected.toarray()))

        hydro = {
            "pore_flux_bcs": [{"set": "bottom", "flux": 1.5}],
            "pore_robin_bcs": [{"set": "top", "beta": 0.4, "pressure": 2.0}],
        }
        boundary_cache = build_pressure_boundary_term_cache(mesh, hydro, axisymmetric=True)
        boundary_cached, rhs_cached, _info_cached = assemble_pressure_boundary_terms_cached(boundary_cache)
        boundary_expected, rhs_expected, _info_expected = assemble_axisymmetric_pressure_boundary_terms(mesh, hydro)
        self.assertTrue(np.allclose(boundary_cached.toarray(), boundary_expected.toarray()))
        self.assertTrue(np.allclose(rhs_cached, rhs_expected))

    def test_axisymmetric_pressure_cache_batches_tri3_tri6(self) -> None:
        for element_type in ("TRI3", "TRI6"):
            with self.subTest(element_type=element_type):
                cfg = {
                    "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [1.0, 2.0],
                        "y_range": [0.0, 1.0],
                        "nx": 2,
                        "ny": 2,
                        "element_type": element_type,
                        "material": "soil",
                    },
                    "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                pressure_cache = build_pressure_matrix_assembly_cache(mesh, axisymmetric=True)
                pressure_info = pressure_cache.info()
                self.assertEqual(pressure_info["batched_elements"], len(mesh.elements))
                self.assertEqual(pressure_info["batch_groups"][0]["element_type"], element_type)
                mass_cached, conductivity_cached = assemble_pressure_matrices_cached(pressure_cache, mesh, materials, storage=0.9, permeability=0.03)
                mass_expected, conductivity_expected = assemble_axisymmetric_pressure_matrices(mesh, materials, storage=0.9, permeability=0.03)
                self.assertTrue(np.allclose(mass_cached.toarray(), mass_expected.toarray(), rtol=1.0e-11, atol=1.0e-10))
                self.assertTrue(np.allclose(conductivity_cached.toarray(), conductivity_expected.toarray(), rtol=1.0e-11, atol=1.0e-10))

                biot_cache = build_biot_coupling_assembly_cache(mesh, axisymmetric=True)
                biot_info = biot_cache.info()
                self.assertEqual(biot_info["batched_elements"], len(mesh.elements))
                self.assertEqual(biot_info["batch_groups"][0]["element_type"], element_type)
                biot_cached = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=0.55)
                biot_expected = assemble_axisymmetric_biot_coupling_matrix(mesh, materials, alpha=0.55)
                self.assertTrue(np.allclose(biot_cached.toarray(), biot_expected.toarray(), rtol=1.0e-11, atol=1.0e-10))

    def test_quad8_pressure_and_biot_batches_match_builder_assemblies(self) -> None:
        for integration in ("FULL", "SRI", "B-bar"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config(integration)
                cfg["mesh"]["nx"] = 3
                cfg["mesh"]["ny"] = 3
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)

                pressure_cache = build_pressure_matrix_assembly_cache(mesh)
                self.assertEqual(pressure_cache.info()["batched_elements"], 9)
                mass_cached, conductivity_cached = assemble_pressure_matrices_cached(pressure_cache, mesh, materials, storage=0.7, permeability=0.02)
                mass_expected, conductivity_expected = assemble_pressure_matrices(mesh, materials, storage=0.7, permeability=0.02)
                self.assertTrue(np.allclose(mass_cached.toarray(), mass_expected.toarray()))
                self.assertTrue(np.allclose(conductivity_cached.toarray(), conductivity_expected.toarray()))

                biot_cache = build_biot_coupling_assembly_cache(mesh)
                self.assertEqual(biot_cache.info()["batched_elements"], 9)
                biot_cached = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=0.6)
                biot_expected = assemble_biot_coupling_matrix(mesh, materials, alpha=0.6)
                self.assertTrue(np.allclose(biot_cached.toarray(), biot_expected.toarray()))

    def test_quad8_plane_strain_elastic_internal_force_kernel_matches_reference_integrals(self) -> None:
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                ue = self.displacement_from_coords(coords)
                initial = np.array([1.0, -0.5, 0.25, 0.1], dtype=float)
                Pvol = material.volumetric_projector
                expected = np.zeros(16, dtype=float)
                if integration == "FULL":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        stress = material.D4 @ strain + initial
                        expected += B4.T @ stress * detJ * gp[2] * material.thickness
                elif integration == "SRI":
                    Pdev = np.eye(4) - Pvol
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        stress = material.D4 @ (B4 @ ue) + initial
                        expected += (Pdev @ B4).T @ stress * detJ * gp[2] * material.thickness
                    for gp in integration_points("QUAD8", "REDUCED"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        stress = material.D4 @ (B4 @ ue) + initial
                        expected += (Pvol @ B4).T @ stress * detJ * gp[2] * material.thickness
                else:
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[np.ndarray, float]] = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((B4, dV))
                    Pdev = np.eye(4) - Pvol
                    Bv_bar = Bv_acc / volume
                    for B4, dV in cached:
                        B_eff = Pdev @ B4 + Bv_bar
                        stress = material.D4 @ (B_eff @ ue) + initial
                        expected += B_eff.T @ stress * dV
                fast = _quad8_internal_force_elastic_fast(coords, ue, material, integration, initial)
                self.assertTrue(np.allclose(fast, expected, rtol=1.0e-11, atol=1.0e-8))

    def test_quad8_plane_strain_elastic_post_uses_fast_arrays(self) -> None:
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords)
                ue = u[dofs]
                initial = np.array([1.0, -0.5, 0.25, 0.1], dtype=float)

                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial})
                direct_data = (
                    _quad8_elastic_bbar_post_fast(coords, ue, material, initial)
                    if integration == "B-BAR"
                    else _quad8_elastic_post_fast(coords, ue, material, initial)
                )

                self.assertEqual(len(rows), 9)
                Pvol = material.volumetric_projector
                if integration == "B-BAR":
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                for gp_index, (gp, strain, dV, N) in enumerate(expected):
                    stress = material.D4 @ strain + initial
                    principal = principal_stresses(stress)
                    row = rows[gp_index]
                    row_data = direct_data[gp_index]
                    self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                    self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], stress, rtol=1.0e-12, atol=1.0e-9))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-12, atol=1.0e-9))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertTrue(np.allclose(row_data[6:10], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose(row_data[10:14], stress, rtol=1.0e-12, atol=1.0e-9))
                    self.assertEqual(float(row["plastic"]), 0.0)
                    self.assertEqual(row["material_model"], "elastic")

    def test_tension_cutoff_kernel_module_matches_public_quad8_wrapper(self) -> None:
        contract = tension_cutoff_element_kernel_contract()
        self.assertEqual(contract["schema"], "geofem.fem2d.element_tension_cutoff_kernels.v1")
        self.assertIn("quad8_elastic_tension_cutoff_tangent", contract["covered_surfaces"])
        cfg = self.quad8_rectangle_config("FULL", {"model": "elastic", "E": 50000.0, "nu": 0.30, "tension_cutoff": 2.0})
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        u = self.displacement_from_coords(mesh.coords) * 20.0
        ue = u[dofs]
        initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)

        module_ke, module_fe = tension_cutoff_module_quad8_fast(coords, ue, material, "FULL", initial)
        public_ke, public_fe = _quad8_elastic_tension_tangent_force_fast(coords, ue, material, "FULL", initial)

        self.assertTrue(np.allclose(module_ke, public_ke, rtol=1.0e-12, atol=1.0e-12))
        self.assertTrue(np.allclose(module_fe, public_fe, rtol=1.0e-12, atol=1.0e-12))

    def test_quad8_plane_strain_elastic_tension_cutoff_tangent_internal_and_post_use_fast_kernels(self) -> None:
        material_cfg = {"model": "elastic", "E": 50000.0, "nu": 0.30, "tension_cutoff": 2.0}
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords) * 20.0
                ue = u[dofs]
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                Pvol = material.volumetric_projector
                Pdev = np.eye(4) - Pvol
                tangent_ref = np.zeros((16, 16), dtype=float)
                internal_ref = np.zeros(16, dtype=float)

                if integration == "FULL":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        tangent_gp = algorithmic_material_tangent(material, strain, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, initial_stress=initial)
                        dV = detJ * gp[2] * material.thickness
                        tangent_ref += B4.T @ tangent_gp @ B4 * dV
                        internal_ref += B4.T @ update.stress * dV
                elif integration == "SRI":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        tangent_gp = algorithmic_material_tangent(material, strain, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, initial_stress=initial)
                        dV = detJ * gp[2] * material.thickness
                        tangent_ref += (Pdev @ B4).T @ tangent_gp @ B4 * dV
                        internal_ref += (Pdev @ B4).T @ update.stress * dV
                    for gp in integration_points("QUAD8", "REDUCED"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        tangent_gp = algorithmic_material_tangent(material, strain, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, initial_stress=initial)
                        dV = detJ * gp[2] * material.thickness
                        tangent_ref += (Pvol @ B4).T @ tangent_gp @ B4 * dV
                        internal_ref += (Pvol @ B4).T @ update.stress * dV
                else:
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[np.ndarray, float]] = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((B4, dV))
                    Bv_bar = Bv_acc / volume
                    for B4, dV in cached:
                        B_eff = (Pdev @ B4) + Bv_bar
                        strain = B_eff @ ue
                        tangent_gp = algorithmic_material_tangent(material, strain, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, initial_stress=initial)
                        tangent_ref += B_eff.T @ tangent_gp @ B_eff * dV
                        internal_ref += B_eff.T @ update.stress * dV

                direct_ke, direct_fe = _quad8_elastic_tension_tangent_force_fast(coords, ue, material, integration, initial)
                tangent_fast = assemble_algorithmic_tangent_stiffness(mesh, materials, u, initial_stresses={element.id: initial})
                internal_fast = assemble_internal_force(mesh, materials, u, initial_stresses={element.id: initial})
                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = internal_ref
                self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial})
                direct_data = (
                    _quad8_elastic_tension_bbar_post_fast(coords, ue, material, initial)
                    if integration == "B-BAR"
                    else _quad8_elastic_tension_post_fast(coords, ue, material, initial)
                )
                if integration == "B-BAR":
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    cached_rows: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached_rows.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected_rows = [(gp, (Pdev @ strain) + epsv_bar, dV, N) for gp, strain, dV, N in cached_rows]
                else:
                    expected_rows = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected_rows.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 9)
                clipped_count = 0
                for gp_index, (gp, strain, dV, N) in enumerate(expected_rows):
                    update = update_plane_strain_stress(material, strain, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    row_data = direct_data[gp_index]
                    clipped_count += int(update.plastic)
                    self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                    self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertTrue(np.allclose(row_data[6:10], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose(row_data[10:14], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertEqual(row["active_set"], "tension_cutoff" if update.plastic else "")
                    self.assertEqual(float(row_data[20]), 1.0 if update.plastic else 0.0)
                    self.assertAlmostEqual(float(row_data[21]), float(update.yield_value), delta=1.0e-8)
                self.assertGreater(clipped_count, 0)

    def test_quad8_plane_strain_sri_bbar_constitutive_results_are_finite(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 100.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 100.0, "friction_angle": 25.0, "hardening": 10.0},
            "mohr_coulomb": {"model": "mohr_coulomb", "E": 50000.0, "nu": 0.30, "cohesion": 100.0, "friction_angle": 30.0, "dilation_angle": 5.0},
            "tension_cutoff": {"model": "elastic", "E": 50000.0, "nu": 0.30, "tension_cutoff": 100.0},
            "advanced": {"model": "hardin_drnevich", "E": 50000.0, "G0": 20000.0, "gamma_ref": 0.001, "nu": 0.30},
            "liquefaction": {"model": "bilinear_liquefaction", "E": 50000.0, "G0": 25000.0, "gamma_ref": 0.001, "nu": 0.30, "liquefaction": {"ru": 0.2}},
        }
        for integration in ("SRI", "B-bar"):
            for name, material_cfg in material_cases.items():
                with self.subTest(integration=integration, material=name):
                    cfg = self.quad8_rectangle_config(integration, material_cfg)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    u = self.displacement_from_coords(mesh.coords)

                    tangent = assemble_algorithmic_tangent_stiffness(mesh, materials, u)
                    internal = assemble_internal_force(mesh, materials, u)
                    element_rows, state = compute_element_results_and_state(mesh, materials, u)
                    ip_rows = compute_integration_point_results(mesh, materials, u)

                    self.assertTrue(np.all(np.isfinite(tangent.toarray())))
                    self.assertTrue(np.all(np.isfinite(internal)))
                    self.assertEqual(element_rows[0]["integration"], "B-BAR" if integration == "B-bar" else "SRI")
                    self.assertEqual(len(ip_rows), 9)
                    self.assertTrue(all(row["type"] == "QUAD8" for row in ip_rows))
                    self.assertTrue(all(np.isfinite(float(row["sigma_x"])) for row in ip_rows))
                    expected_state_count = 13 if integration == "SRI" else 9
                    self.assertGreaterEqual(len(state), expected_state_count)

    def test_quad8_plane_strain_j2dp_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
            "von_mises_tension_cutoff": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0, "tension_cutoff": 2.0},
            "drucker_prager_tension_cutoff": {
                "model": "drucker_prager",
                "E": 50000.0,
                "nu": 0.30,
                "cohesion": 3.0,
                "friction_angle": 25.0,
                "hardening": 10.0,
                "tension_cutoff": 2.0,
            },
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            for name, material_cfg in material_cases.items():
                with self.subTest(integration=integration, model=name):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = self.displacement_from_coords(mesh.coords) * 6.0
                    ue = u[dofs]
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    state_count = 13 if integration == "SRI" else 9
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float) * (1.0 + 0.1 * gp_index),
                            0.01 + 0.001 * gp_index,
                        )
                        for gp_index in range(state_count)
                    }
                    plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(state_count)])
                    kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(state_count)], dtype=float)
                    alpha, cohesion_term = _yield_surface_parameters(material, 1.0)

                    direct_ke, direct_fe = _quad8_j2dp_tangent_force_fast(
                        coords,
                        ue,
                        material,
                        integration,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    )
                    tangent_fast = assemble_algorithmic_tangent_stiffness(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)
                    internal_fast = assemble_internal_force(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                    if integration == "FULL" and name == "drucker_prager_tension_cutoff":
                        contract = j2dp_element_kernel_contract()
                        self.assertEqual(contract["schema"], "geofem.fem2d.element_j2dp_kernels.v1")
                        self.assertIn("quad8_j2dp_tangent_internal_force", contract["covered_surfaces"])
                        module_ke, module_fe = j2dp_module_quad8_tangent_force_fast(
                            coords,
                            ue,
                            material,
                            integration,
                            initial_stress=initial,
                            plastic_strains=plastic_strains,
                            kappas=kappas,
                            alpha=alpha,
                            cohesion_term=cohesion_term,
                        )
                        self.assertTrue(np.allclose(module_ke, direct_ke, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose(module_fe, direct_fe, rtol=1.0e-12, atol=1.0e-12))

                    tangent_ref = np.zeros((16, 16), dtype=float)
                    internal_ref = np.zeros(16, dtype=float)
                    Pvol = material.volumetric_projector
                    Pdev = np.eye(4) - Pvol
                    if integration == "FULL":
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness
                            tangent_ref += B4.T @ tangent_gp @ B4 * dV
                            internal_ref += B4.T @ update.stress * dV
                    elif integration == "SRI":
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness
                            Bdev = Pdev @ B4
                            tangent_ref += Bdev.T @ tangent_gp @ B4 * dV
                            internal_ref += Bdev.T @ update.stress * dV
                        offset = 9
                        for red_index, gp in enumerate(integration_points("QUAD8", "REDUCED")):
                            gp_index = offset + red_index
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness
                            Bv = Pvol @ B4
                            tangent_ref += Bv.T @ tangent_gp @ B4 * dV
                            internal_ref += Bv.T @ update.stress * dV
                    else:
                        volume = 0.0
                        Bv_acc = np.zeros((4, 16), dtype=float)
                        cached: list[tuple[int, np.ndarray, float]] = []
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            dV = detJ * gp[2] * material.thickness
                            volume += dV
                            Bv_acc += (Pvol @ B4) * dV
                            cached.append((gp_index, B4, dV))
                        Bv_bar = Bv_acc / volume
                        for gp_index, B4, dV in cached:
                            B_eff = Pdev @ B4 + Bv_bar
                            strain = B_eff @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            tangent_ref += B_eff.T @ tangent_gp @ B_eff * dV
                            internal_ref += B_eff.T @ update.stress * dV

                    expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                    expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                    expected_internal = np.zeros_like(u)
                    expected_internal[dofs] = internal_ref
                    self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-9, atol=1.0e-5))
                    self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-9, atol=1.0e-5))
                    self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad8_plane_strain_j2dp_post_uses_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
            "von_mises_tension_cutoff": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0, "tension_cutoff": 2.0},
            "drucker_prager_tension_cutoff": {
                "model": "drucker_prager",
                "E": 50000.0,
                "nu": 0.30,
                "cohesion": 3.0,
                "friction_angle": 25.0,
                "hardening": 10.0,
                "tension_cutoff": 2.0,
            },
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            for name, material_cfg in material_cases.items():
                with self.subTest(integration=integration, model=name):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = self.displacement_from_coords(mesh.coords) * 6.0
                    ue = u[dofs]
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    state_count = 13 if integration == "SRI" else 9
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float) * (1.0 + 0.1 * gp_index),
                            0.01 + 0.001 * gp_index,
                        )
                        for gp_index in range(state_count)
                    }
                    plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(9)])
                    kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(9)], dtype=float)
                    alpha, cohesion_term = _yield_surface_parameters(material, 1.0)
                    direct_data = (
                        _quad8_j2dp_bbar_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_strains=plastic_strains,
                            kappas=kappas,
                            alpha=alpha,
                            cohesion_term=cohesion_term,
                        )
                        if integration == "B-BAR"
                        else _quad8_j2dp_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_strains=plastic_strains,
                            kappas=kappas,
                            alpha=alpha,
                            cohesion_term=cohesion_term,
                        )
                    )
                    rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                    Pvol = material.volumetric_projector
                    if integration == "B-BAR":
                        cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                        volume = 0.0
                        epsv_acc = np.zeros(4, dtype=float)
                        for gp in integration_points("QUAD8", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            dV = detJ * gp[2] * material.thickness
                            volume += dV
                            epsv_acc += (Pvol @ strain) * dV
                            cached.append((gp, strain, dV, N))
                        epsv_bar = epsv_acc / volume
                        expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                    else:
                        expected = []
                        for gp in integration_points("QUAD8", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                            expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                    self.assertEqual(len(rows), 9)
                    for gp_index, (gp, strain, dV, N) in enumerate(expected):
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        principal = principal_stresses(update.stress)
                        row = rows[gp_index]
                        row_data = direct_data[gp_index]
                        self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                        self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                        self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                        self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                        self.assertTrue(np.allclose(row_data[6:10], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose(row_data[10:14], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                        self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-7)
                        self.assertAlmostEqual(float(row["p"]), float(update.p), delta=1.0e-9)
                        self.assertAlmostEqual(float(row["q"]), float(update.q), delta=1.0e-7)
                        self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-12)
                        self.assertTrue(np.allclose(row_data[23:27], update.plastic_strain, rtol=1.0e-10, atol=1.0e-10))
                        if material.tension_cutoff:
                            self.assertLessEqual(float(np.max(principal)), float(material.tensile_strength) + 1.0e-8)

    def test_quad8_plane_strain_mc_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cases = {
            "mohr_coulomb": {
                "model": "mohr_coulomb",
                "E": 50000.0,
                "nu": 0.30,
                "cohesion": 5.0,
                "friction_angle": 10.0,
                "dilation_angle": 5.0,
            },
            "mohr_coulomb_tension_cutoff": {
                "model": "mohr_coulomb",
                "E": 50000.0,
                "nu": 0.30,
                "cohesion": 5.0,
                "friction_angle": 10.0,
                "dilation_angle": 5.0,
                "tension_cutoff": 2.0,
            },
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            for case_name, material_cfg in material_cases.items():
                with self.subTest(integration=integration, case=case_name):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords) * 4.0
                ue = u[dofs]
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                strength_factor = 1.0
                state_count = 13 if integration == "SRI" else 9
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(
                        np.array([1.0e-4, -5.0e-5, 2.5e-5, 5.0e-5], dtype=float) * (1.0 + 0.05 * gp_index),
                        0.002 + 0.0002 * gp_index,
                    )
                    for gp_index in range(state_count)
                }
                plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(state_count)])
                kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(state_count)], dtype=float)

                direct_fast = _quad8_mc_tangent_force_fast(
                    coords,
                    ue,
                    material,
                    integration,
                    initial_stress=initial,
                    plastic_strains=plastic_strains,
                    kappas=kappas,
                    strength_factor=strength_factor,
                )
                self.assertIsNotNone(direct_fast)
                direct_ke, direct_fe = direct_fast
                direct_internal = _quad8_mc_internal_force_fast(
                    coords,
                    ue,
                    material,
                    integration,
                    initial_stress=initial,
                    plastic_strains=plastic_strains,
                    kappas=kappas,
                    strength_factor=strength_factor,
                )
                self.assertIsNotNone(direct_internal)
                tangent_fast = assemble_algorithmic_tangent_stiffness(
                    mesh,
                    materials,
                    u,
                    initial_stresses={element.id: initial},
                    plastic_state=plastic_state,
                    strength_factor=strength_factor,
                )
                internal_fast = assemble_internal_force(
                    mesh,
                    materials,
                    u,
                    initial_stresses={element.id: initial},
                    plastic_state=plastic_state,
                    strength_factor=strength_factor,
                )

                tangent_ref = np.zeros((16, 16), dtype=float)
                internal_ref = np.zeros(16, dtype=float)
                Pvol = material.volumetric_projector
                Pdev = np.eye(4) - Pvol
                plastic_count = 0
                if integration == "FULL":
                    for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        plastic_count += int(update.plastic)
                        dV = detJ * gp[2] * material.thickness
                        tangent_ref += B4.T @ tangent_gp @ B4 * dV
                        internal_ref += B4.T @ update.stress * dV
                elif integration == "SRI":
                    for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        plastic_count += int(update.plastic)
                        dV = detJ * gp[2] * material.thickness
                        Bdev = Pdev @ B4
                        tangent_ref += Bdev.T @ tangent_gp @ B4 * dV
                        internal_ref += Bdev.T @ update.stress * dV
                    offset = 9
                    for red_index, gp in enumerate(integration_points("QUAD8", "REDUCED")):
                        gp_index = offset + red_index
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        plastic_count += int(update.plastic)
                        dV = detJ * gp[2] * material.thickness
                        Bv = Pvol @ B4
                        tangent_ref += Bv.T @ tangent_gp @ B4 * dV
                        internal_ref += Bv.T @ update.stress * dV
                else:
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[int, np.ndarray, float]] = []
                    for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((gp_index, B4, dV))
                    Bv_bar = Bv_acc / volume
                    for gp_index, B4, dV in cached:
                        B_eff = Pdev @ B4 + Bv_bar
                        strain = B_eff @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
                        plastic_count += int(update.plastic)
                        tangent_ref += B_eff.T @ tangent_gp @ B_eff * dV
                        internal_ref += B_eff.T @ update.stress * dV

                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = internal_ref
                self.assertGreater(plastic_count, 0)
                self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(direct_internal, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad8_plane_strain_mc_post_uses_state_arrays(self) -> None:
        material_cases = {
            "mohr_coulomb": {
                "model": "mohr_coulomb",
                "E": 50000.0,
                "nu": 0.30,
                "cohesion": 5.0,
                "friction_angle": 30.0,
                "dilation_angle": 10.0,
            },
            "mohr_coulomb_tension_cutoff": {
                "model": "mohr_coulomb",
                "E": 50000.0,
                "nu": 0.30,
                "cohesion": 5.0,
                "friction_angle": 10.0,
                "dilation_angle": 5.0,
                "tension_cutoff": 2.0,
            },
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            for case_name, material_cfg in material_cases.items():
                with self.subTest(integration=integration, case=case_name):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords) * 8.0
                ue = u[dofs]
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                state_count = 13 if integration == "SRI" else 9
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(
                        np.array([1.0e-4, -5.0e-5, 2.5e-5, 5.0e-5], dtype=float) * (1.0 + 0.05 * gp_index),
                        0.002 + 0.0002 * gp_index,
                    )
                    for gp_index in range(state_count)
                }
                plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(9)])
                kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(9)], dtype=float)
                direct_data = (
                    _quad8_mc_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        strength_factor=1.0,
                    )
                    if integration == "B-BAR"
                    else _quad8_mc_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        strength_factor=1.0,
                    )
                )
                self.assertIsNotNone(direct_data)
                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                Pvol = material.volumetric_projector
                if integration == "B-BAR":
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 9)
                self.assertTrue(any(float(row["plastic"]) == 1.0 for row in rows))
                for gp_index, (gp, strain, dV, N) in enumerate(expected):
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    row_data = direct_data[gp_index]
                    self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                    self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["p"]), float(update.p), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["q"]), float(update.q), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                    self.assertTrue(np.allclose(row_data[23:27], update.plastic_strain, rtol=1.0e-10, atol=1.0e-9))
                    self.assertEqual(bool(row["active_set"]), bool(update.active_set))
                    if material.tension_cutoff:
                        self.assertLessEqual(float(np.max(principal)), float(material.tensile_strength) + 1.0e-8)

    def test_quad4_plane_strain_elastic_post_uses_fast_arrays(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        u = np.array([0.0, 0.0, 0.001, 0.0, 0.0012, -0.0004, 0.0, -0.0003], dtype=float)
        rows = compute_integration_point_results(mesh, materials, u)
        self.assertEqual(len(rows), 4)
        ue = u[dofs]
        for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
            B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
            strain = B4 @ ue
            stress = material.D4 @ strain
            principal = principal_stresses(stress)
            row = rows[gp_index]
            self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
            self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], stress, rtol=1.0e-12, atol=1.0e-9))
            self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-12, atol=1.0e-9))
            self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
            self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
            self.assertAlmostEqual(float(row["dV"]), detJ * gp[2] * material.thickness, places=12)
            self.assertEqual(row["material_model"], "elastic")
            self.assertEqual(float(row["plastic"]), 0.0)
            self.assertEqual(float(row["liquefaction_FL"]), 0.0)

    def test_quad4_plane_strain_bbar_elastic_post_uses_fast_arrays(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "B-bar"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        u = np.array([0.0, 0.0, 0.001, 0.0, 0.0012, -0.0004, 0.0, -0.0003], dtype=float)
        rows = compute_integration_point_results(mesh, materials, u)
        ue = u[dofs]
        Pvol = material.volumetric_projector
        cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
        volume = 0.0
        epsv_acc = np.zeros(4, dtype=float)
        for gp in integration_points("QUAD4", "FULL"):
            B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
            strain = B4 @ ue
            dV = detJ * gp[2] * material.thickness
            volume += dV
            epsv_acc += (Pvol @ strain) * dV
            cached.append((gp, strain, dV, N))
        epsv_bar = epsv_acc / volume
        for gp_index, (gp, strain, dV, N) in enumerate(cached):
            strain_eff = (np.eye(4) - Pvol) @ strain + epsv_bar
            stress = material.D4 @ strain_eff
            principal = principal_stresses(stress)
            row = rows[gp_index]
            self.assertEqual(row["integration"], "B-BAR")
            self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain_eff, rtol=1.0e-12, atol=1.0e-12))
            self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], stress, rtol=1.0e-12, atol=1.0e-9))
            self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-12, atol=1.0e-9))
            self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
            self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
            self.assertAlmostEqual(float(row["dV"]), dV, places=12)
            self.assertEqual(float(row["plastic"]), 0.0)

    def test_quad4_plane_strain_advanced_elastic_post_uses_state_arrays(self) -> None:
        for integration in ("FULL", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": integration}],
                    },
                    "materials": {"soil": {"model": "hardin_drnevich", "E": 50000.0, "nu": 0.30, "G0": 25000.0, "gamma_ref": 0.001}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.001, 0.0, 0.0012, -0.0004, 0.0, -0.0003], dtype=float)
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(state_vars={"gamma_eq": 1.0e-5 * (gp_index + 1), "cycles": float(gp_index)})
                    for gp_index in range(4)
                }

                if integration == "B-BAR":
                    direct_data = _quad4_advanced_elastic_bbar_post_fast(
                        coords,
                        u[dofs],
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                else:
                    direct_data = _quad4_advanced_elastic_post_fast(
                        coords,
                        u[dofs],
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                self.assertEqual(direct_data.shape, (4, 36))
                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                ue = u[dofs]
                if integration == "B-BAR":
                    Pvol = material.volumetric_projector
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD4", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD4", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 4)
                for gp_index, (_gp, strain, dV, N) in enumerate(expected):
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    self.assertEqual(row["integration"], integration)
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["modulus_ratio"]), float(update.state_vars["modulus_ratio"]), delta=1.0e-12)
                    self.assertAlmostEqual(float(row["effective_E"]), float(update.state_vars["effective_E"]), delta=1.0e-8)
                    self.assertEqual(row["advanced_model"], material.advanced_model)
                    self.assertEqual(row["material_model"], material.model)
                    self.assertEqual(float(row["plastic"]), 0.0)

    def test_quad4_plane_strain_advanced_elastic_tension_post_uses_state_arrays(self) -> None:
        for integration in ("FULL", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": integration}],
                    },
                    "materials": {
                        "soil": {
                            "model": "hardin_drnevich",
                            "E": 50000.0,
                            "nu": 0.30,
                            "G0": 25000.0,
                            "gamma_ref": 0.001,
                            "tension_cutoff": 8.0,
                        }
                    },
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.002, 0.0, 0.0022, -0.0003, 0.0, -0.0002], dtype=float)
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(state_vars={"gamma_eq": 1.0e-5 * (gp_index + 1), "cycles": float(gp_index)})
                    for gp_index in range(4)
                }

                if integration == "B-BAR":
                    direct_data = _quad4_advanced_elastic_tension_bbar_post_fast(
                        coords,
                        u[dofs],
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                else:
                    direct_data = _quad4_advanced_elastic_tension_post_fast(
                        coords,
                        u[dofs],
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                self.assertEqual(direct_data.shape, (4, 38))
                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                ue = u[dofs]
                if integration == "B-BAR":
                    Pvol = material.volumetric_projector
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD4", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD4", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 4)
                self.assertGreater(sum(float(row["plastic"]) for row in rows), 0.0)
                for gp_index, (_gp, strain, dV, N) in enumerate(expected):
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    self.assertEqual(row["integration"], integration)
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["modulus_ratio"]), float(update.state_vars["modulus_ratio"]), delta=1.0e-12)
                    self.assertAlmostEqual(float(row["effective_E"]), float(update.state_vars["effective_E"]), delta=1.0e-8)
                    self.assertEqual(row["advanced_model"], material.advanced_model)
                    self.assertEqual(row["material_model"], material.model)

    def test_quad8_plane_strain_advanced_elastic_post_uses_state_arrays(self) -> None:
        material_cfg = {"model": "hardin_drnevich", "E": 50000.0, "nu": 0.30, "G0": 25000.0, "gamma_ref": 0.001}
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords) * 1.5
                ue = u[dofs]
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(state_vars={"gamma_eq": 1.0e-5 * (gp_index + 1), "cycles": float(gp_index)})
                    for gp_index in range(9)
                }

                direct_data = (
                    _quad8_advanced_elastic_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                    if integration == "B-BAR"
                    else _quad8_advanced_elastic_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                )
                self.assertEqual(direct_data.shape, (9, 36))
                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                if integration == "B-BAR":
                    Pvol = material.volumetric_projector
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 9)
                for gp_index, (gp, strain, dV, N) in enumerate(expected):
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    row_data = direct_data[gp_index]
                    self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                    self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["modulus_ratio"]), float(update.state_vars["modulus_ratio"]), delta=1.0e-12)
                    self.assertAlmostEqual(float(row["effective_E"]), float(update.state_vars["effective_E"]), delta=1.0e-8)
                    self.assertEqual(row["advanced_model"], material.advanced_model)
                    self.assertEqual(row["material_model"], material.model)
                    self.assertEqual(float(row["plastic"]), 0.0)

    def test_quad8_plane_strain_advanced_elastic_tension_post_uses_state_arrays(self) -> None:
        material_cfg = {
            "model": "hardin_drnevich",
            "E": 50000.0,
            "nu": 0.30,
            "G0": 25000.0,
            "gamma_ref": 0.001,
            "tension_cutoff": 8.0,
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords) * 3.0
                ue = u[dofs]
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(state_vars={"gamma_eq": 1.0e-5 * (gp_index + 1), "cycles": float(gp_index)})
                    for gp_index in range(9)
                }

                direct_data = (
                    _quad8_advanced_elastic_tension_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                    if integration == "B-BAR"
                    else _quad8_advanced_elastic_tension_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    )
                )
                self.assertEqual(direct_data.shape, (9, 38))
                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                if integration == "B-BAR":
                    Pvol = material.volumetric_projector
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 9)
                self.assertGreater(sum(float(row["plastic"]) for row in rows), 0.0)
                for gp_index, (gp, strain, dV, N) in enumerate(expected):
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    row_data = direct_data[gp_index]
                    self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                    self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["modulus_ratio"]), float(update.state_vars["modulus_ratio"]), delta=1.0e-12)
                    self.assertAlmostEqual(float(row["effective_E"]), float(update.state_vars["effective_E"]), delta=1.0e-8)
                    self.assertEqual(row["advanced_model"], material.advanced_model)
                    self.assertEqual(row["material_model"], material.model)
                    self.assertLessEqual(float(np.max(principal)), float(material.tensile_strength) + 1.0e-8)

    def test_quad4_plane_strain_advanced_strength_post_uses_state_arrays(self) -> None:
        material_cases = {
            "liquefaction_vm": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "liquefaction_vm_cutoff": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "tension_cutoff": 4.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "pz_sand_dp": {
                "model": "pastor_zienkiewicz_sand",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "cohesion": 2.0,
                "friction_angle": 28.0,
                "phi_cs": 32.0,
                "peak_dilation_angle": 8.0,
                "residual_dilation_angle": 1.0,
                "advanced_hardening": 15.0,
            },
        }
        for case_name, material_cfg in material_cases.items():
            for integration in ("FULL", "B-BAR"):
                with self.subTest(case=case_name, integration=integration):
                    cfg = {
                        "analysis": {"dimension": "2D"},
                        "mesh": {
                            "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                            "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": integration}],
                        },
                        "materials": {"soil": material_cfg},
                    }
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = np.array([0.0, 0.0, 0.002, -0.0001, 0.0024, -0.0008, 0.0, -0.0005], dtype=float)
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                            0.01 * (gp_index + 1),
                            {"gamma_eq": 1.0e-5 * (gp_index + 1), "ru": 0.2 + 0.02 * gp_index, "hardening_variable": 0.1 * gp_index},
                        )
                        for gp_index in range(4)
                    }
                    if integration == "B-BAR":
                        direct_data = _quad4_advanced_strength_j2dp_bbar_post_fast(
                            coords,
                            u[dofs],
                            material,
                            initial_stress=initial,
                            plastic_state=plastic_state,
                            element_id=element.id,
                            strength_factor=1.0,
                        )
                    else:
                        direct_data = _quad4_advanced_strength_j2dp_post_fast(
                            coords,
                            u[dofs],
                            material,
                            initial_stress=initial,
                            plastic_state=plastic_state,
                            element_id=element.id,
                            strength_factor=1.0,
                        )
                    self.assertEqual(direct_data.shape, (4, 43))
                    rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)
                    if "cutoff" in case_name:
                        self.assertGreater(sum(float(row["plastic"]) for row in rows), 0.0)

                    ue = u[dofs]
                    if integration == "B-BAR":
                        Pvol = material.volumetric_projector
                        cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                        volume = 0.0
                        epsv_acc = np.zeros(4, dtype=float)
                        for gp in integration_points("QUAD4", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                            strain = B4 @ ue
                            dV = detJ * gp[2] * material.thickness
                            volume += dV
                            epsv_acc += (Pvol @ strain) * dV
                            cached.append((gp, strain, dV, N))
                        epsv_bar = epsv_acc / volume
                        expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                    else:
                        expected = []
                        for gp in integration_points("QUAD4", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                            expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                    self.assertEqual(len(rows), 4)
                    for gp_index, (_gp, strain, dV, N) in enumerate(expected):
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        principal = principal_stresses(update.stress)
                        row = rows[gp_index]
                        self.assertEqual(row["integration"], integration)
                        self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                        self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                        self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                        self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                        self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                        self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                        self.assertTrue(
                            np.allclose(
                                [row["plastic_strain_x"], row["plastic_strain_y"], row["plastic_strain_z"], row["plastic_strain_gamma_xy"]],
                                update.plastic_strain,
                                rtol=1.0e-10,
                                atol=1.0e-9,
                            )
                        )
                        for key in ("modulus_ratio", "effective_E", "hardening_variable", "dilatancy", "plastic_multiplier"):
                            self.assertAlmostEqual(float(row[key]), float(update.state_vars[key]), delta=1.0e-8)
                        if "liquefaction" in case_name:
                            self.assertAlmostEqual(float(row["ru"]), float(update.state_vars["ru"]), delta=1.0e-12)
                        self.assertEqual(row["advanced_model"], material.advanced_model)
                        self.assertEqual(row["material_model"], material.model)

    def test_quad4_plane_strain_advanced_strength_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cases = {
            "liquefaction_vm": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "liquefaction_vm_cutoff": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "tension_cutoff": 4.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "pz_sand_dp": {
                "model": "pastor_zienkiewicz_sand",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "cohesion": 2.0,
                "friction_angle": 28.0,
                "phi_cs": 32.0,
                "advanced_hardening": 15.0,
            },
        }
        for case_name, material_cfg in material_cases.items():
            with self.subTest(case=case_name):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    },
                    "materials": {"soil": material_cfg},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.002, -0.0001, 0.0024, -0.0008, 0.0, -0.0005], dtype=float)
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(
                        np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                        0.01 * (gp_index + 1),
                        {"gamma_eq": 1.0e-5 * (gp_index + 1), "ru": 0.2 + 0.02 * gp_index, "hardening_variable": 0.1 * gp_index},
                    )
                    for gp_index in range(4)
                }
                direct_ke, direct_fe = _quad4_advanced_strength_j2dp_tangent_force_fast(
                    coords,
                    u[dofs],
                    material,
                    initial_stress=initial,
                    plastic_state=plastic_state,
                    element_id=element.id,
                    strength_factor=1.0,
                )
                tangent_fast = assemble_algorithmic_tangent_stiffness(
                    mesh,
                    materials,
                    u,
                    initial_stresses={element.id: initial},
                    plastic_state=plastic_state,
                )
                internal_fast = assemble_internal_force(
                    mesh,
                    materials,
                    u,
                    initial_stresses={element.id: initial},
                    plastic_state=plastic_state,
                )

                ue = u[dofs]
                tangent_ref = np.zeros((8, 8), dtype=float)
                internal_ref = np.zeros(8, dtype=float)
                for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
                    B4, detJ, _N = strain_displacement_matrix("QUAD4", coords, gp)
                    strain = B4 @ ue
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    dV = detJ * gp[2] * material.thickness
                    tangent_ref += B4.T @ tangent_gp @ B4 * dV
                    internal_ref += B4.T @ update.stress * dV

                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = internal_ref
                self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad8_plane_strain_advanced_strength_post_uses_state_arrays(self) -> None:
        material_cases = {
            "liquefaction_vm": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "liquefaction_vm_cutoff": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "tension_cutoff": 4.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "pz_sand_dp": {
                "model": "pastor_zienkiewicz_sand",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "cohesion": 2.0,
                "friction_angle": 28.0,
                "phi_cs": 32.0,
                "peak_dilation_angle": 8.0,
                "residual_dilation_angle": 1.0,
                "advanced_hardening": 15.0,
            },
        }
        for case_name, material_cfg in material_cases.items():
            for integration in ("FULL", "SRI", "B-BAR"):
                with self.subTest(case=case_name, integration=integration):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = self.displacement_from_coords(mesh.coords) * 2.8
                    ue = u[dofs]
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                            0.01 * (gp_index + 1),
                            {"gamma_eq": 1.0e-5 * (gp_index + 1), "ru": 0.2 + 0.02 * gp_index, "hardening_variable": 0.1 * gp_index},
                        )
                        for gp_index in range(9)
                    }
                    direct_data = (
                        _quad8_advanced_strength_j2dp_bbar_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_state=plastic_state,
                            element_id=element.id,
                            strength_factor=1.0,
                        )
                        if integration == "B-BAR"
                        else _quad8_advanced_strength_j2dp_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_state=plastic_state,
                            element_id=element.id,
                            strength_factor=1.0,
                        )
                    )
                    self.assertEqual(direct_data.shape, (9, 43))
                    rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)
                    if "cutoff" in case_name:
                        self.assertGreater(sum(float(row["plastic"]) for row in rows), 0.0)

                    if integration == "B-BAR":
                        Pvol = material.volumetric_projector
                        cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                        volume = 0.0
                        epsv_acc = np.zeros(4, dtype=float)
                        for gp in integration_points("QUAD8", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            dV = detJ * gp[2] * material.thickness
                            volume += dV
                            epsv_acc += (Pvol @ strain) * dV
                            cached.append((gp, strain, dV, N))
                        epsv_bar = epsv_acc / volume
                        expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                    else:
                        expected = []
                        for gp in integration_points("QUAD8", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                            expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                    self.assertEqual(len(rows), 9)
                    for gp_index, (gp, strain, dV, N) in enumerate(expected):
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        principal = principal_stresses(update.stress)
                        row = rows[gp_index]
                        row_data = direct_data[gp_index]
                        self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                        self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                        self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                        self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                        self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                        self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                        self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                        self.assertTrue(
                            np.allclose(
                                [row["plastic_strain_x"], row["plastic_strain_y"], row["plastic_strain_z"], row["plastic_strain_gamma_xy"]],
                                update.plastic_strain,
                                rtol=1.0e-10,
                                atol=1.0e-9,
                            )
                        )
                        for key in ("modulus_ratio", "effective_E", "hardening_variable", "dilatancy", "plastic_multiplier"):
                            self.assertAlmostEqual(float(row[key]), float(update.state_vars[key]), delta=1.0e-8)
                        if "liquefaction" in case_name:
                            self.assertAlmostEqual(float(row["ru"]), float(update.state_vars["ru"]), delta=1.0e-12)
                        self.assertEqual(row["advanced_model"], material.advanced_model)
                        self.assertEqual(row["material_model"], material.model)

    def test_quad8_plane_strain_advanced_strength_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cases = {
            "liquefaction_vm_cutoff": {
                "model": "bilinear_liquefaction",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "yield_stress": 12.0,
                "hardening": 20.0,
                "tension_cutoff": 4.0,
                "liquefaction": {"ru": 0.35, "post_liquefaction_strength_ratio": 0.05},
            },
            "pz_sand_dp": {
                "model": "pastor_zienkiewicz_sand",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "cohesion": 2.0,
                "friction_angle": 28.0,
                "phi_cs": 32.0,
                "advanced_hardening": 15.0,
            },
        }
        for case_name, material_cfg in material_cases.items():
            for integration in ("FULL", "SRI", "B-BAR"):
                with self.subTest(case=case_name, integration=integration):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = self.displacement_from_coords(mesh.coords) * 2.8
                    ue = u[dofs]
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    state_count = 13 if integration == "SRI" else 9
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                            0.01 * (gp_index + 1),
                            {"gamma_eq": 1.0e-5 * (gp_index + 1), "ru": 0.2 + 0.02 * gp_index, "hardening_variable": 0.1 * gp_index},
                        )
                        for gp_index in range(state_count)
                    }

                    direct_ke, direct_fe = _quad8_advanced_strength_j2dp_tangent_force_fast(
                        coords,
                        ue,
                        material,
                        integration,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                        strength_factor=1.0,
                    )
                    tangent_fast = assemble_algorithmic_tangent_stiffness(
                        mesh,
                        materials,
                        u,
                        initial_stresses={element.id: initial},
                        plastic_state=plastic_state,
                    )
                    internal_fast = assemble_internal_force(
                        mesh,
                        materials,
                        u,
                        initial_stresses={element.id: initial},
                        plastic_state=plastic_state,
                    )

                    tangent_ref = np.zeros((16, 16), dtype=float)
                    internal_ref = np.zeros(16, dtype=float)
                    Pvol = material.volumetric_projector
                    Pdev = np.eye(4) - Pvol
                    if integration == "FULL":
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness
                            tangent_ref += B4.T @ tangent_gp @ B4 * dV
                            internal_ref += B4.T @ update.stress * dV
                    elif integration == "SRI":
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness
                            Bdev = Pdev @ B4
                            tangent_ref += Bdev.T @ tangent_gp @ B4 * dV
                            internal_ref += Bdev.T @ update.stress * dV
                        for red_index, gp in enumerate(integration_points("QUAD8", "REDUCED")):
                            gp_index = 9 + red_index
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness
                            Bv = Pvol @ B4
                            tangent_ref += Bv.T @ tangent_gp @ B4 * dV
                            internal_ref += Bv.T @ update.stress * dV
                    else:
                        volume = 0.0
                        Bv_acc = np.zeros((4, 16), dtype=float)
                        cached: list[tuple[int, np.ndarray, float]] = []
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                            dV = detJ * gp[2] * material.thickness
                            volume += dV
                            Bv_acc += (Pvol @ B4) * dV
                            cached.append((gp_index, B4, dV))
                        Bv_bar = Bv_acc / volume
                        for gp_index, B4, dV in cached:
                            B_eff = Pdev @ B4 + Bv_bar
                            strain = B_eff @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            tangent_ref += B_eff.T @ tangent_gp @ B_eff * dV
                            internal_ref += B_eff.T @ update.stress * dV

                    expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                    expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                    expected_internal = np.zeros_like(u)
                    expected_internal[dofs] = internal_ref
                    self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-8, atol=1.0e-4))
                    self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-8, atol=1.0e-4))
                    self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad8_plane_strain_advanced_strength_mc_post_uses_state_arrays(self) -> None:
        material_cases = {
            "pz_sand_mc": {
                "model": "pastor_zienkiewicz_sand",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "cohesion": 1.5,
                "friction_angle": 24.0,
                "phi_cs": 28.0,
                "peak_dilation_angle": 6.0,
                "residual_dilation_angle": 1.0,
                "advanced_hardening": 12.0,
                "strength_model": "mohr_coulomb",
            },
            "pz_sand_mc_cutoff": {
                "model": "pastor_zienkiewicz_sand",
                "E": 50000.0,
                "nu": 0.30,
                "G0": 25000.0,
                "gamma_ref": 0.001,
                "cohesion": 1.5,
                "friction_angle": 24.0,
                "phi_cs": 28.0,
                "peak_dilation_angle": 6.0,
                "residual_dilation_angle": 1.0,
                "advanced_hardening": 12.0,
                "strength_model": "mohr_coulomb",
                "tension_cutoff": 4.0,
            },
        }
        for case_name, material_cfg in material_cases.items():
            for integration in ("FULL", "SRI", "B-BAR"):
                with self.subTest(case=case_name, integration=integration):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = self.displacement_from_coords(mesh.coords) * 0.2
                    ue = u[dofs]
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                            0.01 * (gp_index + 1),
                            {"gamma_eq": 1.0e-5 * (gp_index + 1), "hardening_variable": 0.1 * gp_index},
                        )
                        for gp_index in range(9)
                    }
                    direct_data = (
                        _quad8_advanced_strength_mc_bbar_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_state=plastic_state,
                            element_id=element.id,
                            strength_factor=1.0,
                        )
                        if integration == "B-BAR"
                        else _quad8_advanced_strength_mc_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_state=plastic_state,
                            element_id=element.id,
                            strength_factor=1.0,
                        )
                    )
                    self.assertIsNotNone(direct_data)
                    rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                    Pvol = material.volumetric_projector
                    if integration == "B-BAR":
                        cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                        volume = 0.0
                        epsv_acc = np.zeros(4, dtype=float)
                        for gp in integration_points("QUAD8", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            dV = detJ * gp[2] * material.thickness
                            volume += dV
                            epsv_acc += (Pvol @ strain) * dV
                            cached.append((gp, strain, dV, N))
                        epsv_bar = epsv_acc / volume
                        expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                    else:
                        expected = []
                        for gp in integration_points("QUAD8", "FULL"):
                            B4, detJ, N = strain_displacement_matrix("QUAD8", coords, gp)
                            expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                    self.assertEqual(len(rows), 9)
                    for gp_index, (gp, strain, dV, N) in enumerate(expected):
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        principal = principal_stresses(update.stress)
                        row = rows[gp_index]
                        row_data = direct_data[gp_index]
                        self.assertEqual(update.state_vars["yield_surface"], "mohr_coulomb")
                        self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else integration)
                        self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                        self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                        self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                        self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                        self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-7)
                        self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                        self.assertTrue(
                            np.allclose(
                                [row["plastic_strain_x"], row["plastic_strain_y"], row["plastic_strain_z"], row["plastic_strain_gamma_xy"]],
                                update.plastic_strain,
                                rtol=1.0e-10,
                                atol=1.0e-9,
                            )
                        )
                        for key in ("modulus_ratio", "effective_E", "hardening_variable", "dilatancy", "plastic_multiplier"):
                            self.assertAlmostEqual(float(row[key]), float(update.state_vars[key]), delta=1.0e-8)
                        if material.tension_cutoff:
                            self.assertLessEqual(float(np.max(principal)), float(material.tensile_strength) + 1.0e-8)

    def test_quad8_plane_strain_advanced_strength_mc_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cfg = {
            "model": "pastor_zienkiewicz_sand",
            "E": 50000.0,
            "nu": 0.30,
            "G0": 25000.0,
            "gamma_ref": 0.001,
            "cohesion": 1.5,
            "friction_angle": 24.0,
            "phi_cs": 28.0,
            "peak_dilation_angle": 6.0,
            "residual_dilation_angle": 1.0,
            "advanced_hardening": 12.0,
            "strength_model": "mohr_coulomb",
            "tension_cutoff": 4.0,
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords) * 0.2
                ue = u[dofs]
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                state_count = 13 if integration == "SRI" else 9
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(
                        np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                        0.01 * (gp_index + 1),
                        {"gamma_eq": 1.0e-5 * (gp_index + 1), "hardening_variable": 0.1 * gp_index},
                    )
                    for gp_index in range(state_count)
                }

                direct = _quad8_advanced_strength_mc_tangent_force_fast(
                    coords,
                    ue,
                    material,
                    integration,
                    initial_stress=initial,
                    plastic_state=plastic_state,
                    element_id=element.id,
                    strength_factor=1.0,
                )
                self.assertIsNotNone(direct)
                direct_ke, direct_fe = direct
                tangent_fast = assemble_algorithmic_tangent_stiffness(
                    mesh,
                    materials,
                    u,
                    initial_stresses={element.id: initial},
                    plastic_state=plastic_state,
                )
                internal_fast = assemble_internal_force(
                    mesh,
                    materials,
                    u,
                    initial_stresses={element.id: initial},
                    plastic_state=plastic_state,
                )

                tangent_ref = np.zeros((16, 16), dtype=float)
                internal_ref = np.zeros(16, dtype=float)
                Pvol = material.volumetric_projector
                Pdev = np.eye(4) - Pvol
                if integration == "FULL":
                    for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        self.assertEqual(update.state_vars["yield_surface"], "mohr_coulomb")
                        dV = detJ * gp[2] * material.thickness
                        tangent_ref += B4.T @ tangent_gp @ B4 * dV
                        internal_ref += B4.T @ update.stress * dV
                elif integration == "SRI":
                    for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        dV = detJ * gp[2] * material.thickness
                        Bdev = Pdev @ B4
                        tangent_ref += Bdev.T @ tangent_gp @ B4 * dV
                        internal_ref += Bdev.T @ update.stress * dV
                    for red_index, gp in enumerate(integration_points("QUAD8", "REDUCED")):
                        gp_index = 9 + red_index
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        strain = B4 @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        dV = detJ * gp[2] * material.thickness
                        Bv = Pvol @ B4
                        tangent_ref += Bv.T @ tangent_gp @ B4 * dV
                        internal_ref += Bv.T @ update.stress * dV
                else:
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[int, np.ndarray, float]] = []
                    for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                        B4, detJ, _N = strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((gp_index, B4, dV))
                    Bv_bar = Bv_acc / volume
                    for gp_index, B4, dV in cached:
                        B_eff = Pdev @ B4 + Bv_bar
                        strain = B_eff @ ue
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        tangent_ref += B_eff.T @ tangent_gp @ B_eff * dV
                        internal_ref += B_eff.T @ update.stress * dV

                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = internal_ref
                self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-8, atol=1.0e-4))
                self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad4_plane_strain_advanced_strength_mc_tangent_and_internal_force_use_state_arrays(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {
                "soil": {
                    "model": "pastor_zienkiewicz_sand",
                    "E": 50000.0,
                    "nu": 0.30,
                    "G0": 25000.0,
                    "gamma_ref": 0.001,
                    "cohesion": 1.5,
                    "friction_angle": 24.0,
                    "phi_cs": 28.0,
                    "peak_dilation_angle": 6.0,
                    "residual_dilation_angle": 1.0,
                    "advanced_hardening": 12.0,
                    "strength_model": "mohr_coulomb",
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        u = np.array([0.0, 0.0, 0.002, -0.0001, 0.0024, -0.0008, 0.0, -0.0005], dtype=float)
        initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
        plastic_state = {
            f"{element.id}:{gp_index}": PlasticState2D(
                np.array([1.0e-5 * (gp_index + 1), -0.5e-5, 0.0, 0.25e-5], dtype=float),
                0.01 * (gp_index + 1),
                {"gamma_eq": 1.0e-5 * (gp_index + 1), "hardening_variable": 0.1 * gp_index},
            )
            for gp_index in range(4)
        }
        direct = _quad4_advanced_strength_mc_tangent_force_fast(
            coords,
            u[dofs],
            material,
            initial_stress=initial,
            plastic_state=plastic_state,
            element_id=element.id,
            strength_factor=1.0,
        )
        self.assertIsNotNone(direct)
        direct_ke, direct_fe = direct
        tangent_fast = assemble_algorithmic_tangent_stiffness(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
        )
        internal_fast = assemble_internal_force(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
        )

        ue = u[dofs]
        tangent_ref = np.zeros((8, 8), dtype=float)
        internal_ref = np.zeros(8, dtype=float)
        for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
            B4, detJ, _N = strain_displacement_matrix("QUAD4", coords, gp)
            strain = B4 @ ue
            state = plastic_state[f"{element.id}:{gp_index}"]
            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
            self.assertEqual(update.state_vars["yield_surface"], "mohr_coulomb")
            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
            dV = detJ * gp[2] * material.thickness
            tangent_ref += B4.T @ tangent_gp @ B4 * dV
            internal_ref += B4.T @ update.stress * dV

        expected_tangent = np.zeros((len(u), len(u)), dtype=float)
        expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
        expected_internal = np.zeros_like(u)
        expected_internal[dofs] = internal_ref
        self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-8, atol=1.0e-4))
        self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
        self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-8, atol=1.0e-4))
        self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad4_plane_strain_j2dp_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
        }
        for name, material_cfg in material_cases.items():
            with self.subTest(model=name):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    },
                    "materials": {"soil": material_cfg},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.01, 0.0, 0.012, -0.002, 0.0, -0.001], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float), 0.01)
                    for gp_index in range(4)
                }

                tangent_fast = assemble_algorithmic_tangent_stiffness(mesh, materials, u, plastic_state=plastic_state)
                internal_fast = assemble_internal_force(mesh, materials, u, plastic_state=plastic_state)

                ue = u[dofs]
                tangent_ref = np.zeros((8, 8), dtype=float)
                internal_ref = np.zeros(8, dtype=float)
                initial = np.zeros(4, dtype=float)
                for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
                    B4, detJ, _N = strain_displacement_matrix("QUAD4", coords, gp)
                    strain = B4 @ ue
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    dV = detJ * gp[2] * material.thickness
                    tangent_ref += B4.T @ tangent_gp @ B4 * dV
                    internal_ref += B4.T @ update.stress * dV

                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = internal_ref
                self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad4_plane_strain_mc_tangent_and_internal_force_use_state_arrays(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 50000.0,
                    "nu": 0.30,
                    "cohesion": 5.0,
                    "friction_angle": 10.0,
                    "dilation_angle": 5.0,
                }
            },
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
        u = np.array([0.0, 0.0, 0.002, 0.0, 0.0022, -0.0001, 0.0, -0.00005], dtype=float)
        initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
        strength_factor = 1.0
        plastic_state = {
            f"{element.id}:{gp_index}": PlasticState2D(np.array([1.0e-4, -5.0e-5, 2.5e-5, 5.0e-5], dtype=float), 0.002)
            for gp_index in range(4)
        }
        plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(4)])
        kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(4)], dtype=float)

        direct_fast = _quad4_mc_tangent_force_fast(
            coords,
            u[dofs],
            material,
            initial_stress=initial,
            plastic_strains=plastic_strains,
            kappas=kappas,
            strength_factor=strength_factor,
        )
        self.assertIsNotNone(direct_fast)
        direct_ke, direct_fe = direct_fast
        direct_internal = _quad4_mc_internal_force_fast(
            coords,
            u[dofs],
            material,
            initial_stress=initial,
            plastic_strains=plastic_strains,
            kappas=kappas,
            strength_factor=strength_factor,
        )
        self.assertIsNotNone(direct_internal)
        tangent_fast = assemble_algorithmic_tangent_stiffness(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
            strength_factor=strength_factor,
        )
        internal_fast = assemble_internal_force(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
            strength_factor=strength_factor,
        )

        ue = u[dofs]
        tangent_ref = np.zeros((8, 8), dtype=float)
        internal_ref = np.zeros(8, dtype=float)
        plastic_count = 0
        for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
            B4, detJ, _N = strain_displacement_matrix("QUAD4", coords, gp)
            strain = B4 @ ue
            state = plastic_state[f"{element.id}:{gp_index}"]
            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
            plastic_count += int(update.plastic)
            dV = detJ * gp[2] * material.thickness
            tangent_ref += B4.T @ tangent_gp @ B4 * dV
            internal_ref += B4.T @ update.stress * dV

        expected_tangent = np.zeros((len(u), len(u)), dtype=float)
        expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
        expected_internal = np.zeros_like(u)
        expected_internal[dofs] = internal_ref
        self.assertGreater(plastic_count, 0)
        self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-10, atol=1.0e-7))
        self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
        self.assertTrue(np.allclose(direct_internal, internal_ref, rtol=1.0e-10, atol=1.0e-7))
        self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-10, atol=1.0e-7))
        self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_quad4_plane_strain_j2dp_post_uses_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
        }
        for name, material_cfg in material_cases.items():
            with self.subTest(model=name):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                    },
                    "materials": {"soil": material_cfg},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.01, 0.0, 0.012, -0.002, 0.0, -0.001], dtype=float)
                initial = np.array([1.0, -2.0, 0.5, 0.2], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float), 0.01)
                    for gp_index in range(4)
                }

                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                self.assertEqual(len(rows), 4)
                ue = u[dofs]
                for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
                    B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                    strain = B4 @ ue
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(
                        np.allclose(
                            [row["plastic_strain_x"], row["plastic_strain_y"], row["plastic_strain_z"], row["plastic_strain_gamma_xy"]],
                            update.plastic_strain,
                            rtol=1.0e-10,
                            atol=1.0e-9,
                        )
                    )
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), detJ * gp[2] * material.thickness, places=12)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["p"]), float(update.p), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["q"]), float(update.q), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                    self.assertEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0)
                    self.assertEqual(row["active_set"], "")
                    self.assertEqual(row["material_model"], material.model)
                    self.assertEqual(float(row["liquefaction_FL"]), 0.0)

    def test_quad4_plane_strain_bbar_j2dp_post_uses_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
        }
        for name, material_cfg in material_cases.items():
            with self.subTest(model=name):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": "B-bar"}],
                    },
                    "materials": {"soil": material_cfg},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.01, 0.0, 0.012, -0.002, 0.0, -0.001], dtype=float)
                initial = np.array([1.0, -2.0, 0.5, 0.2], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float), 0.01)
                    for gp_index in range(4)
                }

                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                ue = u[dofs]
                Pvol = material.volumetric_projector
                cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                volume = 0.0
                epsv_acc = np.zeros(4, dtype=float)
                for gp in integration_points("QUAD4", "FULL"):
                    B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                    strain = B4 @ ue
                    dV = detJ * gp[2] * material.thickness
                    volume += dV
                    epsv_acc += (Pvol @ strain) * dV
                    cached.append((gp, strain, dV, N))
                epsv_bar = epsv_acc / volume
                for gp_index, (gp, strain, dV, N) in enumerate(cached):
                    strain_eff = (np.eye(4) - Pvol) @ strain + epsv_bar
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain_eff, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    self.assertEqual(row["integration"], "B-BAR")
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain_eff, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(
                        np.allclose(
                            [row["plastic_strain_x"], row["plastic_strain_y"], row["plastic_strain_z"], row["plastic_strain_gamma_xy"]],
                            update.plastic_strain,
                            rtol=1.0e-10,
                            atol=1.0e-9,
                        )
                    )
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["p"]), float(update.p), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["q"]), float(update.q), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                    self.assertEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0)
                    self.assertEqual(row["active_set"], "")
                    self.assertEqual(row["material_model"], material.model)

    def test_quad4_plane_strain_mc_post_uses_numba_candidate_scan(self) -> None:
        for integration in ("FULL", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = {
                    "analysis": {"dimension": "2D"},
                    "mesh": {
                        "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                        "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil", "integration": integration}],
                    },
                    "materials": {
                        "soil": {
                            "model": "mohr_coulomb",
                            "E": 50000.0,
                            "nu": 0.30,
                            "cohesion": 5.0,
                            "friction_angle": 30.0,
                            "dilation_angle": 10.0,
                        }
                    },
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.012, 0.0, 0.014, -0.004, 0.0, -0.002], dtype=float)
                initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(np.array([1.0e-4, -5.0e-5, 2.5e-5, 5.0e-5], dtype=float), 0.002)
                    for gp_index in range(4)
                }

                rows = compute_integration_point_results(mesh, materials, u, initial_stresses={element.id: initial}, plastic_state=plastic_state)

                ue = u[dofs]
                if integration == "B-BAR":
                    Pvol = material.volumetric_projector
                    cached: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
                    volume = 0.0
                    epsv_acc = np.zeros(4, dtype=float)
                    for gp in integration_points("QUAD4", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                        strain = B4 @ ue
                        dV = detJ * gp[2] * material.thickness
                        volume += dV
                        epsv_acc += (Pvol @ strain) * dV
                        cached.append((gp, strain, dV, N))
                    epsv_bar = epsv_acc / volume
                    expected = [(gp, (np.eye(4) - Pvol) @ strain + epsv_bar, dV, N) for gp, strain, dV, N in cached]
                else:
                    expected = []
                    for gp in integration_points("QUAD4", "FULL"):
                        B4, detJ, N = strain_displacement_matrix("QUAD4", coords, gp)
                        expected.append((gp, B4 @ ue, detJ * gp[2] * material.thickness, N))

                self.assertEqual(len(rows), 4)
                self.assertTrue(any(float(row["plastic"]) == 1.0 for row in rows))
                for gp_index, (_gp, strain, dV, N) in enumerate(expected):
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    principal = principal_stresses(update.stress)
                    row = rows[gp_index]
                    self.assertEqual(row["integration"], "B-BAR" if integration == "B-BAR" else "FULL")
                    self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                    self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(
                        np.allclose(
                            [row["plastic_strain_x"], row["plastic_strain_y"], row["plastic_strain_z"], row["plastic_strain_gamma_xy"]],
                            update.plastic_strain,
                            rtol=1.0e-10,
                            atol=1.0e-9,
                        )
                    )
                    self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                    self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                    self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                    self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["p"]), float(update.p), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["q"]), float(update.q), delta=1.0e-8)
                    self.assertAlmostEqual(float(row["kappa"]), float(update.kappa), delta=1.0e-10)
                    self.assertEqual(bool(row["active_set"]), bool(update.active_set))
                    self.assertEqual(row["material_model"], material.model)

    def test_axisymmetric_quad4_up_matrices_use_fast_kernels(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        mass, conductivity = assemble_axisymmetric_pressure_matrices(mesh, materials, storage=1.0, permeability=1.0)
        biot = assemble_axisymmetric_biot_coupling_matrix(mesh, materials)
        self.assertEqual(mass.shape, (4, 4))
        self.assertEqual(conductivity.shape, (4, 4))
        self.assertEqual(biot.shape, (8, 4))
        self.assertAlmostEqual(float(mass.sum()), 3.0 * math.pi, places=12)
        self.assertAlmostEqual(float(conductivity.diagonal().sum()), 8.0 * math.pi, places=12)
        self.assertTrue(np.all(np.isfinite(biot.toarray())))

    def test_axisymmetric_quad4_elastic_tangent_and_internal_force_use_fast_kernels(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        u = np.array([0.0, 0.0, 0.001, 0.0, 0.0012, -0.0004, 0.0, -0.0003], dtype=float)
        stiffness = assemble_axisymmetric_stiffness(mesh, materials)
        tangent = assemble_axisymmetric_algorithmic_tangent_stiffness(mesh, materials, u)
        internal = assemble_axisymmetric_internal_force(mesh, materials, u)
        self.assertTrue(np.allclose(tangent.toarray(), stiffness.toarray(), rtol=1.0e-11, atol=1.0e-8))
        self.assertTrue(np.allclose(internal, tangent @ u, rtol=1.0e-11, atol=1.0e-8))
        self.assertGreater(stiffness.nnz, 0)

    def test_axisymmetric_stiffness_cache_matches_uncached_and_batches_quad4(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        cache = build_axisymmetric_stiffness_assembly_cache(mesh, materials)
        cached = assemble_axisymmetric_stiffness_cached(cache, mesh, materials)
        reference = assemble_axisymmetric_stiffness(mesh, materials)
        self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-12, atol=1.0e-9))
        info = cache.info()
        self.assertEqual(info["batched_quad4_axisymmetric_elastic_elements"], 2)
        self.assertTrue(info["direct_fill"]["enabled"])

    def test_axisymmetric_step_cache_precomputes_linear_element_blocks(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "loads": [{"set": "right", "fx": 1.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        topology = stage.solver_info["topology_cache"]
        batches = topology["stiffness_assembly_cache"]["precomputed_linear_batches"]
        self.assertTrue(batches["enabled"])
        self.assertEqual(batches["element_batches"], 1)
        self.assertEqual(topology["precomputed_linear_stiffness_batched_blocks"], len(stage.active_elements))
        self.assertTrue(stage.solver_info["axisymmetric_linear_static_cache"]["stiffness_cache_used"])

    def test_axisymmetric_precomputed_linear_element_stiffness_matches_uncached_for_high_order_and_tri(self) -> None:
        for element_type, integration in (("QUAD8", "FULL"), ("QUAD8", "B-bar"), ("TRI3", "FULL"), ("TRI6", "SRI")):
            with self.subTest(element_type=element_type, integration=integration):
                cfg = {
                    "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [1.0, 2.0],
                        "y_range": [0.0, 1.0],
                        "nx": 2,
                        "ny": 2,
                        "element_type": element_type,
                        "integration": integration,
                        "material": "soil",
                    },
                    "materials": {"soil": {"model": "elastic", "E": 9000.0, "nu": 0.27}},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                cache = build_axisymmetric_stiffness_assembly_cache(mesh, materials, precompute_linear_element_stiffness=True)
                cached = assemble_axisymmetric_stiffness_cached(cache, mesh, materials)
                reference = assemble_axisymmetric_stiffness(mesh, materials)
                info = cache.info()
                batches = info["precomputed_linear_batches"]
                self.assertTrue(batches["enabled"])
                self.assertEqual(batches["element_batches"], 1)
                self.assertEqual(info["precomputed_element_blocks"], len(mesh.elements))
                self.assertEqual(batches["batched_blocks"], len(mesh.elements))
                self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-10, atol=1.0e-8))

    def test_axisymmetric_stiffness_cache_batches_quad8_full_sri_and_bbar(self) -> None:
        for integration in ("FULL", "SRI", "B-bar"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config(integration, axisymmetric=True)
                cfg["mesh"].update({"nx": 2, "ny": 2})
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                cache = build_axisymmetric_stiffness_assembly_cache(mesh, materials)
                cached = assemble_axisymmetric_stiffness_cached(cache, mesh, materials)
                reference = assemble_axisymmetric_stiffness(mesh, materials)
                info = cache.info()
                self.assertEqual(info["batched_quad4_axisymmetric_elastic_elements"], 0)
                self.assertEqual(info["batched_quad8_axisymmetric_elastic_elements"], len(mesh.elements))
                self.assertEqual(info["batched_axisymmetric_elastic_elements"], len(mesh.elements))
                self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-11, atol=1.0e-8))

    def test_axisymmetric_stiffness_cache_batches_tri3_tri6_full_sri_and_bbar(self) -> None:
        for element_type in ("TRI3", "TRI6"):
            for integration in ("FULL", "SRI", "B-bar"):
                with self.subTest(element_type=element_type, integration=integration):
                    cfg = {
                        "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
                        "mesh": {
                            "generator": "rectangle",
                            "x_range": [1.0, 2.0],
                            "y_range": [0.0, 1.0],
                            "nx": 2,
                            "ny": 2,
                            "element_type": element_type,
                            "integration": integration,
                            "material": "soil",
                        },
                        "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
                    }
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    cache = build_axisymmetric_stiffness_assembly_cache(mesh, materials)
                    cached = assemble_axisymmetric_stiffness_cached(cache, mesh, materials)
                    reference = assemble_axisymmetric_stiffness(mesh, materials)
                    info = cache.info()
                    tri3_expected = len(mesh.elements) if element_type == "TRI3" else 0
                    tri6_expected = len(mesh.elements) if element_type == "TRI6" else 0
                    self.assertEqual(info["batched_tri3_axisymmetric_elastic_elements"], tri3_expected)
                    self.assertEqual(info["batched_tri6_axisymmetric_elastic_elements"], tri6_expected)
                    self.assertEqual(info["batched_axisymmetric_elastic_elements"], len(mesh.elements))
                    self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-10, atol=1.0e-8))

    def test_axisymmetric_stiffness_cache_batches_mixed_linear_interfaces_and_structural_blocks(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {
                    "1": [1.0, 0.0],
                    "2": [2.0, 0.0],
                    "3": [3.0, 0.0],
                    "4": [1.0, 1.0],
                    "5": [2.0, 1.0],
                    "6": [3.0, 1.0],
                },
                "elements": [
                    {"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "5", "4"], "material": "soil"},
                    {"id": "e2", "type": "QUAD4", "nodes": ["2", "3", "6", "5"], "material": "soil"},
                ],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        interfaces = [
            Interface2D(id="joint-1", minus_nodes=("1", "4"), plus_nodes=("2", "5"), kn=1000.0, kt=300.0),
            Interface2D(id="joint-2", minus_nodes=("2", "5"), plus_nodes=("3", "6"), kn=800.0, kt=250.0),
        ]
        structural = [
            StructuralElement2D(id="bar", type="BAR2", nodes=("1", "2"), section={"kx": 250.0}),
            StructuralElement2D(id="shear", type="SHEAR_SPRING2", nodes=("5", "6"), section={"ky": 150.0}),
        ]
        cache = build_axisymmetric_stiffness_assembly_cache(mesh, materials, interfaces=interfaces, structural_elements=structural)
        cached = assemble_axisymmetric_stiffness_cached(cache, mesh, materials, interfaces=interfaces, structural_elements=structural)
        reference = assemble_axisymmetric_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural)
        info = cache.info()
        batches = info["precomputed_linear_batches"]
        self.assertEqual(info["precomputed_interface_blocks"], 2)
        self.assertEqual(info["precomputed_structural_blocks"], 2)
        self.assertEqual(batches["interface_batches"], 1)
        self.assertEqual(batches["structural_batches"], 1)
        self.assertEqual(batches["batch_count"], 2)
        self.assertEqual(batches["batched_blocks"], 4)
        self.assertGreater(batches["flat_value_size"], 0)
        self.assertTrue(np.allclose(cached.toarray(), reference.toarray(), rtol=1.0e-12, atol=1.0e-8))

    def test_axisymmetric_combined_tangent_internal_force_matches_separate_assemblies(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0}},
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        element = mesh.elements[0]
        u = np.array([0.0, 0.0, 0.01, 0.0, 0.012, -0.002, 0.0, -0.001], dtype=float)
        initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
        plastic_state = {
            f"{element.id}:{gp_index}": PlasticState2D(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float), 0.01)
            for gp_index in range(4)
        }
        state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)

        tangent, internal = assemble_axisymmetric_tangent_and_internal_force(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
            plastic_state_cache=state_cache,
        )
        tangent_expected = assemble_axisymmetric_algorithmic_tangent_stiffness(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
            plastic_state_cache=state_cache,
        )
        internal_expected = assemble_axisymmetric_internal_force(
            mesh,
            materials,
            u,
            initial_stresses={element.id: initial},
            plastic_state=plastic_state,
            plastic_state_cache=state_cache,
        )

        self.assertTrue(np.allclose(tangent.toarray(), tangent_expected.toarray(), rtol=1.0e-12, atol=1.0e-8))
        self.assertTrue(np.allclose(internal, internal_expected, rtol=1.0e-12, atol=1.0e-8))

    def test_axisymmetric_quad8_elastic_kernels_match_reference_integrals(self) -> None:
        for integration in ("FULL", "SRI", "B-BAR"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, axisymmetric=True)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                expected = np.zeros((16, 16), dtype=float)
                Pvol = material.volumetric_projector
                if integration == "FULL":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                        expected += B4.T @ material.D4 @ B4 * detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                elif integration == "SRI":
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                        expected += B4.T @ material.C_dev @ B4 * detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                    for gp in integration_points("QUAD8", "REDUCED"):
                        B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                        Bv = Pvol @ B4
                        expected += Bv.T @ material.C_vol @ Bv * detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                else:
                    volume = 0.0
                    Bv_acc = np.zeros((4, 16), dtype=float)
                    cached: list[tuple[np.ndarray, float]] = []
                    for gp in integration_points("QUAD8", "FULL"):
                        B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                        dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                        volume += dV
                        Bv_acc += (Pvol @ B4) * dV
                        cached.append((B4, dV))
                    Bv_bar = Bv_acc / volume
                    Pdev = np.eye(4) - Pvol
                    for B4, dV in cached:
                        Bdev = Pdev @ B4
                        expected += Bdev.T @ material.C_dev @ Bdev * dV
                        expected += Bv_bar.T @ material.C_vol @ Bv_bar * dV
                fast = _quad8_axisymmetric_element_stiffness_fast(coords, material, integration)
                self.assertTrue(np.allclose(fast, 0.5 * (expected + expected.T), rtol=1.0e-11, atol=1.0e-7))

    def test_axisymmetric_quad8_pressure_biot_and_internal_kernels_match_reference_integrals(self) -> None:
        cfg = self.quad8_rectangle_config("B-bar", {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 9.80665}, axisymmetric=True)
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        material = materials["soil"]
        element = mesh.elements[0]
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        ue = self.displacement_from_coords(coords)
        initial = np.array([1.0, -0.5, 0.25, 0.1], dtype=float)
        storage = 2.0
        permeability = 0.5
        alpha = 0.7
        mass_expected = np.zeros((8, 8), dtype=float)
        cond_expected = np.zeros((8, 8), dtype=float)
        biot_expected = np.zeros((16, 8), dtype=float)
        internal_expected = np.zeros(16, dtype=float)
        m = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        volume = 0.0
        Bv_acc = np.zeros((4, 16), dtype=float)
        cached: list[tuple[np.ndarray, float]] = []
        for gp in integration_points("QUAD8", "FULL"):
            xi, eta, weight = gp
            N, dN = shape_functions("QUAD8", xi, eta)
            jac = dN @ coords
            detJ = float(np.linalg.det(jac))
            grad = np.linalg.inv(jac) @ dN
            B4, detB, N_b, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
            self.assertAlmostEqual(detB, detJ, places=12)
            self.assertTrue(np.allclose(N_b, N, rtol=1.0e-12, atol=1.0e-12))
            dV = detJ * weight * material.thickness * 2.0 * math.pi * radius
            mass_expected += storage * np.outer(N, N) * dV
            cond_expected += permeability * (grad.T @ grad) * dV
            biot_expected += B4.T @ (alpha * np.outer(m, N)) * dV
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((B4, dV))
        Bv_bar = Bv_acc / volume
        for B4, dV in cached:
            B_eff = Pdev @ B4 + Bv_bar
            stress = material.D4 @ (B_eff @ ue) + initial
            internal_expected += B_eff.T @ stress * dV
        mass_fast, cond_fast = _quad8_axisymmetric_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
        biot_fast = _quad8_axisymmetric_biot_matrix_fast(coords, material, alpha)
        internal_fast = _quad8_axisymmetric_internal_force_elastic_fast(coords, ue, material, "B-BAR", initial)
        self.assertTrue(np.allclose(mass_fast, mass_expected, rtol=1.0e-11, atol=1.0e-10))
        self.assertTrue(np.allclose(cond_fast, cond_expected, rtol=1.0e-11, atol=1.0e-10))
        self.assertTrue(np.allclose(biot_fast, biot_expected, rtol=1.0e-11, atol=1.0e-10))
        self.assertTrue(np.allclose(internal_fast, internal_expected, rtol=1.0e-11, atol=1.0e-7))

    def test_axisymmetric_quad8_sri_bbar_elastic_tangent_and_internal_force_follow_integration_mode(self) -> None:
        for integration in ("SRI", "B-bar"):
            with self.subTest(integration=integration):
                cfg = self.quad8_rectangle_config(integration, axisymmetric=True)
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = self.displacement_from_coords(mesh.coords)

                tangent = assemble_axisymmetric_algorithmic_tangent_stiffness(mesh, materials, u)
                internal = assemble_axisymmetric_internal_force(mesh, materials, u)

                ke = axisymmetric_element_stiffness("QUAD8", coords, material, integration)
                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = ke
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = ke @ u[dofs]
                self.assertTrue(np.allclose(tangent.toarray(), expected_tangent, rtol=1.0e-10, atol=1.0e-8))
                self.assertTrue(np.allclose(internal, expected_internal, rtol=1.0e-10, atol=1.0e-8))

    def test_axisymmetric_quad8_three_node_edge_traction_is_supported(self) -> None:
        cfg = self.quad8_rectangle_config("FULL", axisymmetric=True)
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        F = assemble_axisymmetric_load_vector(mesh, materials, [{"edge": ["2", "6", "4"], "ty": -1.0}])
        pts = np.array([mesh.coords[mesh.node_index[nid]] for nid in ("2", "6", "4")], dtype=float)
        direct = _quad8_axisymmetric_edge_traction_fast(pts, 0.0, -1.0)
        y_force = sum(float(F[2 * mesh.node_index[nid] + 1]) for nid in ("2", "6", "4"))
        x_force = sum(float(F[2 * mesh.node_index[nid]]) for nid in ("2", "6", "4"))
        self.assertTrue(np.allclose(direct[1::2], [F[2 * mesh.node_index[nid] + 1] for nid in ("2", "6", "4")], rtol=1.0e-12, atol=1.0e-12))
        self.assertAlmostEqual(x_force, 0.0, places=12)
        self.assertAlmostEqual(y_force, -4.0 * math.pi, places=12)

    def test_axisymmetric_quad4_j2dp_tangent_and_internal_force_use_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
        }
        for name, material_cfg in material_cases.items():
            with self.subTest(model=name):
                cfg = {
                    "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
                    "mesh": {
                        "generator": "rectangle",
                        "x_range": [1.0, 2.0],
                        "y_range": [0.0, 1.0],
                        "nx": 1,
                        "ny": 1,
                        "element_type": "QUAD4",
                        "material": "soil",
                    },
                    "materials": {"soil": material_cfg},
                }
                mesh = mesh_from_config(cfg)
                materials = plane_strain_materials(cfg)
                material = materials["soil"]
                element = mesh.elements[0]
                conn = [mesh.node_index[nid] for nid in element.nodes]
                coords = mesh.coords[conn]
                dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                u = np.array([0.0, 0.0, 0.01, 0.0, 0.012, -0.002, 0.0, -0.001], dtype=float)
                plastic_state = {
                    f"{element.id}:{gp_index}": PlasticState2D(np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float), 0.01)
                    for gp_index in range(4)
                }

                tangent_fast = assemble_axisymmetric_algorithmic_tangent_stiffness(mesh, materials, u, plastic_state=plastic_state)
                internal_fast = assemble_axisymmetric_internal_force(mesh, materials, u, plastic_state=plastic_state)

                ue = u[dofs]
                tangent_ref = np.zeros((8, 8), dtype=float)
                internal_ref = np.zeros(8, dtype=float)
                initial = np.zeros(4, dtype=float)
                for gp_index, gp in enumerate(integration_points("QUAD4", "FULL")):
                    B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix("QUAD4", coords, gp)
                    strain = B4 @ ue
                    state = plastic_state[f"{element.id}:{gp_index}"]
                    tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                    update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                    dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                    tangent_ref += B4.T @ tangent_gp @ B4 * dV
                    internal_ref += B4.T @ update.stress * dV

                expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                expected_internal = np.zeros_like(u)
                expected_internal[dofs] = internal_ref
                self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-10, atol=1.0e-7))
                self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))

    def test_axisymmetric_quad8_j2dp_tangent_internal_and_post_use_state_arrays(self) -> None:
        material_cases = {
            "von_mises": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0},
            "drucker_prager": {"model": "drucker_prager", "E": 50000.0, "nu": 0.30, "cohesion": 3.0, "friction_angle": 25.0, "hardening": 10.0},
        }
        for integration in ("FULL", "SRI", "B-BAR"):
            for name, material_cfg in material_cases.items():
                with self.subTest(integration=integration, model=name):
                    cfg = self.quad8_rectangle_config("B-bar" if integration == "B-BAR" else integration, material_cfg, axisymmetric=True)
                    mesh = mesh_from_config(cfg)
                    materials = plane_strain_materials(cfg)
                    material = materials["soil"]
                    element = mesh.elements[0]
                    conn = [mesh.node_index[nid] for nid in element.nodes]
                    coords = mesh.coords[conn]
                    dofs = [dof for idx in conn for dof in (2 * idx, 2 * idx + 1)]
                    u = self.displacement_from_coords(mesh.coords) * 6.0
                    ue = u[dofs]
                    initial = np.array([0.5, -0.25, 0.1, 0.05], dtype=float)
                    state_count = 13 if integration == "SRI" else 9
                    plastic_state = {
                        f"{element.id}:{gp_index}": PlasticState2D(
                            np.array([2.0e-4, -1.0e-4, 5.0e-5, 1.0e-4], dtype=float) * (1.0 + 0.1 * gp_index),
                            0.01 + 0.001 * gp_index,
                        )
                        for gp_index in range(state_count)
                    }
                    plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(state_count)])
                    kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(state_count)], dtype=float)
                    post_plastic_strains = np.vstack([plastic_state[f"{element.id}:{gp_index}"].plastic_strain for gp_index in range(9)])
                    post_kappas = np.array([plastic_state[f"{element.id}:{gp_index}"].kappa for gp_index in range(9)], dtype=float)
                    alpha, cohesion_term = _yield_surface_parameters(material, 1.0)

                    direct_ke, direct_fe = _quad8_axisymmetric_j2dp_tangent_force_fast(
                        coords,
                        ue,
                        material,
                        integration,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    )
                    direct_post = (
                        _quad8_axisymmetric_j2dp_bbar_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_strains=post_plastic_strains,
                            kappas=post_kappas,
                            alpha=alpha,
                            cohesion_term=cohesion_term,
                        )
                        if integration == "B-BAR"
                        else _quad8_axisymmetric_j2dp_post_fast(
                            coords,
                            ue,
                            material,
                            initial_stress=initial,
                            plastic_strains=post_plastic_strains,
                            kappas=post_kappas,
                            alpha=alpha,
                            cohesion_term=cohesion_term,
                        )
                    )
                    tangent_fast = assemble_axisymmetric_algorithmic_tangent_stiffness(
                        mesh,
                        materials,
                        u,
                        initial_stresses={element.id: initial},
                        plastic_state=plastic_state,
                    )
                    internal_fast = assemble_axisymmetric_internal_force(
                        mesh,
                        materials,
                        u,
                        initial_stresses={element.id: initial},
                        plastic_state=plastic_state,
                    )
                    rows = compute_axisymmetric_integration_point_results(
                        mesh,
                        materials,
                        u,
                        initial_stresses={element.id: initial},
                        plastic_state=plastic_state,
                    )

                    tangent_ref = np.zeros((16, 16), dtype=float)
                    internal_ref = np.zeros(16, dtype=float)
                    Pvol = material.volumetric_projector
                    Pdev = np.eye(4) - Pvol
                    post_expected: list[tuple[tuple[float, float, float], np.ndarray, float, np.ndarray, float]] = []
                    if integration == "FULL":
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                            tangent_ref += B4.T @ tangent_gp @ B4 * dV
                            internal_ref += B4.T @ update.stress * dV
                            post_expected.append((gp, strain, dV, N, radius))
                    elif integration == "SRI":
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                            Bdev = Pdev @ B4
                            tangent_ref += Bdev.T @ tangent_gp @ B4 * dV
                            internal_ref += Bdev.T @ update.stress * dV
                            post_expected.append((gp, strain, dV, N, radius))
                        offset = 9
                        for red_index, gp in enumerate(integration_points("QUAD8", "REDUCED")):
                            gp_index = offset + red_index
                            B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                            Bv = Pvol @ B4
                            tangent_ref += Bv.T @ tangent_gp @ B4 * dV
                            internal_ref += Bv.T @ update.stress * dV
                    else:
                        volume = 0.0
                        Bv_acc = np.zeros((4, 16), dtype=float)
                        epsv_acc = np.zeros(4, dtype=float)
                        cached: list[tuple[int, tuple[float, float, float], np.ndarray, np.ndarray, float, np.ndarray, float]] = []
                        for gp_index, gp in enumerate(integration_points("QUAD8", "FULL")):
                            B4, detJ, N, radius = axisymmetric_strain_displacement_matrix("QUAD8", coords, gp)
                            strain = B4 @ ue
                            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                            volume += dV
                            Bv_acc += (Pvol @ B4) * dV
                            epsv_acc += (Pvol @ strain) * dV
                            cached.append((gp_index, gp, B4, strain, dV, N, radius))
                        Bv_bar = Bv_acc / volume
                        epsv_bar = epsv_acc / volume
                        for gp_index, gp, B4, strain, dV, N, radius in cached:
                            B_eff = Pdev @ B4 + Bv_bar
                            strain_eff = Pdev @ strain + epsv_bar
                            state = plastic_state[f"{element.id}:{gp_index}"]
                            tangent_gp = algorithmic_material_tangent(material, B_eff @ ue, state=state, initial_stress=initial)
                            update = update_plane_strain_stress(material, B_eff @ ue, state=state, initial_stress=initial)
                            tangent_ref += B_eff.T @ tangent_gp @ B_eff * dV
                            internal_ref += B_eff.T @ update.stress * dV
                            post_expected.append((gp, strain_eff, dV, N, radius))

                    expected_tangent = np.zeros((len(u), len(u)), dtype=float)
                    expected_tangent[np.ix_(dofs, dofs)] = tangent_ref
                    expected_internal = np.zeros_like(u)
                    expected_internal[dofs] = internal_ref
                    self.assertTrue(np.allclose(direct_ke, tangent_ref, rtol=1.0e-9, atol=1.0e-5))
                    self.assertTrue(np.allclose(direct_fe, internal_ref, rtol=1.0e-10, atol=1.0e-7))
                    self.assertTrue(np.allclose(tangent_fast.toarray(), expected_tangent, rtol=1.0e-9, atol=1.0e-5))
                    self.assertTrue(np.allclose(internal_fast, expected_internal, rtol=1.0e-10, atol=1.0e-7))
                    self.assertEqual(len(rows), 9)

                    for gp_index, (gp, strain, dV, N, radius) in enumerate(post_expected):
                        state = plastic_state[f"{element.id}:{gp_index}"]
                        update = update_plane_strain_stress(material, strain, state=state, initial_stress=initial)
                        principal = principal_stresses(update.stress)
                        row = rows[gp_index]
                        row_data = direct_post[gp_index]
                        self.assertEqual(row["integration"], integration)
                        self.assertTrue(np.allclose(row_data[0:3], np.array(gp, dtype=float), rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["eps_x"], row["eps_y"], row["eps_z"], row["gamma_xy"]], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose([row["sigma_x"], row["sigma_y"], row["sigma_z"], row["tau_xy"]], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertTrue(np.allclose([row["sigma_1"], row["sigma_2"], row["sigma_3"]], principal, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row["x"]), float(N @ coords[:, 0]), places=12)
                        self.assertAlmostEqual(float(row["y"]), float(N @ coords[:, 1]), places=12)
                        self.assertAlmostEqual(float(row["radius"]), float(radius), places=12)
                        self.assertAlmostEqual(float(row["dV"]), dV, places=12)
                        self.assertTrue(np.allclose(row_data[6:10], strain, rtol=1.0e-12, atol=1.0e-12))
                        self.assertTrue(np.allclose(row_data[10:14], update.stress, rtol=1.0e-10, atol=1.0e-7))
                        self.assertAlmostEqual(float(row_data[27]), float(radius), places=12)
                        self.assertAlmostEqual(float(row["plastic"]), 1.0 if update.plastic else 0.0, delta=1.0e-12)
                        self.assertAlmostEqual(float(row["yield_value"]), float(update.yield_value), delta=1.0e-7)

    def test_static_steps_alias_stays_in_2d_core(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg.pop("loads", None)
        cfg["steps"] = [{"name": "load", "type": "static", "loads": [{"edge": ["9", "18"], "ty": -50.0}]}]
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        self.assertEqual(result.stages[0].name, "load")
        self.assertTrue(np.all(np.isfinite(result.stages[0].displacements)))

    def test_linear_solver_control_cg_matches_direct(self) -> None:
        direct_cfg = plane_strain_quad4_sample(integration="B-bar")
        cg_cfg = plane_strain_quad4_sample(integration="B-bar")
        cg_cfg["solver"] = {"linear": {"method": "cg", "tol_rel": 1.0e-11, "max_iter": 1000}}
        with tempfile.TemporaryDirectory() as direct_tmp, tempfile.TemporaryDirectory() as cg_tmp:
            direct = solve_plane_strain_config(direct_cfg, direct_tmp)
            cg_result = solve_plane_strain_config(cg_cfg, cg_tmp)
        self.assertEqual(cg_result.stages[0].solver_info["method"], "cg")
        self.assertGreater(cg_result.stages[0].solver_info["iterations"], 0)
        self.assertTrue(np.allclose(cg_result.stages[0].displacements, direct.stages[0].displacements, rtol=1.0e-8, atol=1.0e-10))

    def test_linear_solver_cg_jacobi_preconditioner_matches_direct(self) -> None:
        direct_cfg = plane_strain_quad4_sample(integration="B-bar")
        cg_cfg = plane_strain_quad4_sample(integration="B-bar")
        cg_cfg["solver"] = {"linear": {"method": "cg", "preconditioner": "jacobi", "tol_rel": 1.0e-11, "max_iter": 1000}}
        with tempfile.TemporaryDirectory() as direct_tmp, tempfile.TemporaryDirectory() as cg_tmp:
            direct = solve_plane_strain_config(direct_cfg, direct_tmp)
            cg_result = solve_plane_strain_config(cg_cfg, cg_tmp)
        self.assertEqual(cg_result.stages[0].solver_info["method"], "cg")
        self.assertEqual(cg_result.stages[0].solver_info["preconditioner"], "jacobi")
        self.assertTrue(np.allclose(cg_result.stages[0].displacements, direct.stages[0].displacements, rtol=1.0e-8, atol=1.0e-10))

    def test_linear_solver_gmres_ilu_preconditioner_matches_direct(self) -> None:
        matrix = csr_matrix(
            [
                [5.0, -1.0, 0.0, 0.2],
                [-1.0, 4.0, -0.5, 0.0],
                [0.0, -0.5, 3.5, -1.0],
                [0.1, 0.0, -1.0, 3.0],
            ],
            dtype=float,
        )
        rhs = np.array([1.0, 2.0, -1.0, 0.5], dtype=float)
        direct, _direct_info = solve_linear_system(matrix, rhs, stage_name="direct", solver={"linear": {"method": "direct", "cache_factorization": False}})
        gmres, info = solve_linear_system(
            matrix,
            rhs,
            stage_name="gmres-ilu",
            solver={"linear": {"method": "gmres", "preconditioner": "ilu", "tol_rel": 1.0e-12, "max_iter": 100, "ilu_drop_tol": 0.0}},
        )
        self.assertEqual(info["method"], "gmres")
        self.assertEqual(info["preconditioner"], "ilu")
        self.assertTrue(info["preconditioner_info"]["enabled"])
        self.assertEqual(info["preconditioner_info"]["type"], "ilu")
        self.assertTrue(np.allclose(gmres, direct, rtol=1.0e-10, atol=1.0e-12))

    def test_linear_solver_auto_can_select_iterative_preconditioned_path(self) -> None:
        matrix = csr_matrix(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=float,
        )
        rhs = np.array([1.0, 2.0, 3.0], dtype=float)
        solution, info = solve_linear_system(
            matrix,
            rhs,
            stage_name="auto-linear",
            solver={"linear": {"method": "auto", "auto_iterative_size": 0, "auto_iterative_method": "gmres", "auto_preconditioner": "ilu", "tol_rel": 1.0e-12, "max_iter": 100}},
        )
        self.assertEqual(info["method_requested"], "auto")
        self.assertEqual(info["method"], "gmres")
        self.assertEqual(info["auto_selection"]["selected"], "gmres")
        self.assertEqual(info["preconditioner"], "ilu")
        self.assertTrue(info["preconditioner_info"]["enabled"])
        self.assertTrue(np.allclose(matrix @ solution, rhs, rtol=1.0e-10, atol=1.0e-12))

    def test_linear_solver_large_scale_profile_defaults_to_iterative_ilu_path(self) -> None:
        matrix = csr_matrix(
            [
                [5.0, -1.0, 0.0, 0.0],
                [-1.0, 4.0, -0.5, 0.0],
                [0.0, -0.5, 3.5, -1.0],
                [0.1, 0.0, -1.0, 3.0],
            ],
            dtype=float,
        )
        rhs = np.array([1.0, 2.0, -1.0, 0.5], dtype=float)
        solution, info = solve_linear_system(
            matrix,
            rhs,
            stage_name="large-scale-linear",
            solver={"linear": {"profile": "large_scale", "auto_iterative_size": 0, "tol_rel": 1.0e-12, "max_iter": 100, "ilu_drop_tol": 0.0}},
        )
        self.assertEqual(info["method_requested"], "auto")
        self.assertEqual(info["method"], "gmres")
        self.assertEqual(info["linear_profile"], "large_scale")
        self.assertTrue(info["equilibrated"])
        self.assertEqual(info["preconditioner"], "ilu")
        self.assertEqual(info["preconditioner_info"]["type"], "ilu")
        self.assertTrue(np.allclose(matrix @ solution, rhs, rtol=1.0e-10, atol=1.0e-12))

    def test_linear_solver_amg_preconditioner_falls_back_when_pyamg_is_unavailable(self) -> None:
        matrix = csr_matrix(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=float,
        )
        rhs = np.array([1.0, 2.0, 3.0], dtype=float)
        with patch.object(linear_solver_core, "_load_pyamg", return_value=(None, ModuleNotFoundError("pyamg"))):
            solution, info = solve_linear_system(
                matrix,
                rhs,
                stage_name="amg-fallback",
                solver={"linear": {"method": "gmres", "preconditioner": "amg", "preconditioner_fallback": "jacobi", "tol_rel": 1.0e-12, "max_iter": 100}},
            )
        self.assertEqual(info["preconditioner"], "amg")
        self.assertTrue(info["preconditioner_info"]["enabled"])
        self.assertEqual(info["preconditioner_info"]["type"], "amg")
        self.assertFalse(info["preconditioner_info"]["available"])
        self.assertEqual(info["preconditioner_info"]["effective_type"], "jacobi")
        self.assertTrue(np.allclose(matrix @ solution, rhs, rtol=1.0e-10, atol=1.0e-12))

    def test_linear_solver_auto_direct_failure_falls_back_to_iterative(self) -> None:
        matrix = csr_matrix(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=float,
        )
        rhs = np.array([1.0, 2.0, 3.0], dtype=float)
        with patch.object(linear_solver_core, "_solve_direct", side_effect=FEM2DError("direct backend failed")):
            solution, info = solve_linear_system(
                matrix,
                rhs,
                stage_name="direct-fallback",
                solver={
                    "linear": {
                        "method": "auto",
                        "auto_iterative_size": 9999,
                        "auto_iterative_method": "gmres",
                        "auto_preconditioner": "jacobi",
                        "tol_rel": 1.0e-12,
                        "max_iter": 100,
                    }
                },
            )
        self.assertEqual(info["method_requested"], "auto")
        self.assertEqual(info["method"], "gmres")
        self.assertTrue(info["direct_fallback"]["enabled"])
        self.assertEqual(info["preconditioner"], "jacobi")
        self.assertTrue(np.allclose(matrix @ solution, rhs, rtol=1.0e-10, atol=1.0e-12))

    def test_linear_solver_external_direct_backend_falls_back_to_scipy_when_unavailable(self) -> None:
        matrix = csr_matrix(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ],
            dtype=float,
        )
        rhs = np.array([1.0, 2.0, 3.0], dtype=float)
        with patch.object(linear_solver_core, "_load_external_direct_solver", return_value=(None, ModuleNotFoundError("pypardiso"))):
            solution, info = solve_linear_system(
                matrix,
                rhs,
                stage_name="external-direct-fallback",
                solver={"linear": {"method": "direct", "direct_backend": "pypardiso", "direct_backend_fallback": True, "cache_factorization": False}},
            )
        self.assertEqual(info["method"], "direct")
        self.assertTrue(str(info["factor_cache"]).startswith("pypardiso_fallback_"))
        self.assertTrue(info["symbolic_cache"]["direct_backend"]["fallback"])
        self.assertEqual(info["symbolic_cache"]["direct_backend"]["used"], "scipy")
        self.assertTrue(np.allclose(matrix @ solution, rhs, rtol=1.0e-10, atol=1.0e-12))

    def test_linear_solver_equilibration_matches_direct(self) -> None:
        direct_cfg = plane_strain_quad4_sample(integration="B-bar")
        scaled_cfg = plane_strain_quad4_sample(integration="B-bar")
        scaled_cfg["solver"] = {"linear": {"method": "direct", "equilibrate": True}}
        with tempfile.TemporaryDirectory() as direct_tmp, tempfile.TemporaryDirectory() as scaled_tmp:
            direct = solve_plane_strain_config(direct_cfg, direct_tmp)
            scaled = solve_plane_strain_config(scaled_cfg, scaled_tmp)
        self.assertTrue(scaled.stages[0].solver_info["equilibrated"])
        self.assertTrue(np.allclose(scaled.stages[0].displacements, direct.stages[0].displacements, rtol=1.0e-10, atol=1.0e-12))

    def test_direct_linear_solver_reuses_cached_factorization(self) -> None:
        clear_linear_factor_cache()
        matrix = csr_matrix(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ]
        )
        solver = {"linear": {"method": "direct", "cache_factorization": True, "cache_min_size": 0}}
        first, first_info = solve_linear_system(matrix, np.array([1.0, 2.0, 3.0]), stage_name="cache-a", solver=solver)
        second, second_info = solve_linear_system(matrix, np.array([2.0, 1.0, 0.0]), stage_name="cache-b", solver=solver)
        cache = linear_factor_cache_info()
        self.assertTrue(np.allclose(matrix @ first, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.allclose(matrix @ second, np.array([2.0, 1.0, 0.0])))
        self.assertEqual(first_info["factor_cache"], "miss")
        self.assertEqual(second_info["factor_cache"], "hit")
        self.assertGreaterEqual(cache["hits"], 1)
        self.assertEqual(cache["solve_lock_scope"], "per_lu_entry")
        clear_linear_factor_cache()

    def test_direct_linear_solver_builds_same_lu_once_across_threads(self) -> None:
        clear_linear_factor_cache()
        matrix = csr_matrix(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 3.0],
            ]
        )
        rhs = np.array([1.0, 2.0, 3.0], dtype=float)
        solver = {
            "linear": {
                "method": "direct",
                "cache_factorization": True,
                "cache_min_size": 0,
                "cache_symbolic": False,
            }
        }
        worker_count = 4
        start = threading.Barrier(worker_count)
        call_lock = threading.Lock()
        splu_calls = 0
        solutions: list[np.ndarray] = []
        errors: list[BaseException] = []
        real_splu = linear_solver_core.splu

        def delayed_splu(*args: Any, **kwargs: Any) -> Any:
            nonlocal splu_calls
            with call_lock:
                splu_calls += 1
            time.sleep(0.05)
            return real_splu(*args, **kwargs)

        def solve_worker() -> None:
            try:
                start.wait()
                solution, _info = solve_linear_system(
                    matrix, rhs, stage_name="threaded-lu", solver=solver
                )
                solutions.append(solution)
            except BaseException as exc:
                errors.append(exc)

        try:
            with patch.object(linear_solver_core, "splu", side_effect=delayed_splu):
                threads = [threading.Thread(target=solve_worker) for _index in range(worker_count)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5.0)
            cache = linear_factor_cache_info()
        finally:
            clear_linear_factor_cache()

        self.assertEqual(errors, [])
        self.assertEqual(len(solutions), worker_count)
        self.assertTrue(all(np.allclose(matrix @ value, rhs) for value in solutions))
        self.assertEqual(splu_calls, 1)
        self.assertEqual(cache["misses"], 1)
        self.assertEqual(cache["hits"], worker_count - 1)
        self.assertEqual(cache["inflight_builds"], 0)

    def test_direct_linear_solver_allows_distinct_cached_lu_solves_to_overlap(self) -> None:
        clear_linear_factor_cache()
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        start_barrier = threading.Barrier(2)

        class FakeLU:
            def __init__(self, matrix: Any) -> None:
                self.matrix = np.asarray(matrix.toarray(), dtype=float)

            def solve(self, rhs: np.ndarray) -> np.ndarray:
                nonlocal active, max_active
                with active_lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    time.sleep(0.03)
                    return np.linalg.solve(self.matrix, rhs)
                finally:
                    with active_lock:
                        active -= 1

        matrices = [csr_matrix([[4.0, -1.0], [-1.0, 3.0]]), csr_matrix([[6.0, -1.0], [-1.0, 5.0]])]
        rhs = np.array([1.0, 2.0], dtype=float)
        errors: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                start_barrier.wait()
                solution, _info = solve_linear_system(
                    matrices[index],
                    rhs,
                    stage_name=f"parallel-lu-{index}",
                    solver={"linear": {"cache_min_size": 0, "cache_symbolic": False}},
                )
                self.assertTrue(np.allclose(matrices[index] @ solution, rhs))
            except BaseException as exc:
                errors.append(exc)

        with patch.object(linear_solver_core, "splu", side_effect=lambda matrix, permc_spec="COLAMD": FakeLU(matrix)):
            threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertFalse(errors)
        self.assertEqual(max_active, 2)
        clear_linear_factor_cache()

    def test_direct_linear_solver_auto_disables_numeric_cache_for_changing_values(self) -> None:
        clear_linear_factor_cache()
        rhs = np.array([1.0, 2.0, 3.0], dtype=float)
        solver = {
            "linear": {
                "cache_min_size": 0,
                "cache_symbolic": False,
                "cache_auto_disable_miss_streak": 2,
                "cache_auto_reprobe_interval": 100,
            }
        }
        states: list[str] = []
        for offset in (0.0, 1.0, 2.0):
            matrix = csr_matrix(
                [[4.0 + offset, -1.0, 0.0], [-1.0, 4.0 + offset, -1.0], [0.0, -1.0, 3.0 + offset]]
            )
            solution, info = solve_linear_system(matrix, rhs, stage_name="changing-lu", solver=solver)
            self.assertTrue(np.allclose(matrix @ solution, rhs))
            states.append(str(info["factor_cache"]))

        cache = linear_factor_cache_info()
        self.assertEqual(states, ["miss", "miss", "auto_disabled"])
        self.assertGreaterEqual(cache["auto_skips"], 1)
        self.assertGreaterEqual(cache["auto_disabled_patterns"], 1)
        clear_linear_factor_cache()

    def test_direct_linear_solver_reuses_symbolic_ordering_for_same_pattern(self) -> None:
        clear_linear_factor_cache()
        matrix_a = csr_matrix(
            [
                [6.0, -2.0, 0.0, 0.0],
                [-2.0, 7.0, -1.0, 0.0],
                [0.0, -1.0, 5.0, -1.5],
                [0.0, 0.0, -1.5, 4.0],
            ]
        )
        matrix_b = csr_matrix(
            [
                [7.0, -1.5, 0.0, 0.0],
                [-1.5, 8.0, -0.5, 0.0],
                [0.0, -0.5, 6.0, -1.0],
                [0.0, 0.0, -1.0, 5.0],
            ]
        )
        solver = {"linear": {"method": "direct", "cache_factorization": True, "cache_min_size": 0, "symbolic_cache_min_size": 0}}
        rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        first, first_info = solve_linear_system(matrix_a, rhs, stage_name="symbolic-a", solver=solver)
        second, second_info = solve_linear_system(matrix_b, rhs, stage_name="symbolic-b", solver=solver)
        cache = linear_factor_cache_info()

        self.assertTrue(np.allclose(matrix_a @ first, rhs))
        self.assertTrue(np.allclose(matrix_b @ second, rhs))
        self.assertEqual(first_info["factor_cache"], "miss")
        self.assertEqual(second_info["factor_cache"], "miss")
        self.assertEqual(first_info["symbolic_cache"]["state"], "miss")
        self.assertEqual(second_info["symbolic_cache"]["state"], "hit")
        self.assertEqual(second_info["symbolic_cache"]["permc_spec"], "NATURAL")
        self.assertTrue(second_info["symbolic_cache"]["direct_fill_permutation"])
        self.assertGreaterEqual(cache["symbolic_hits"], 1)
        self.assertGreaterEqual(cache["symbolic_misses"], 1)
        clear_linear_factor_cache()

    def test_mpc_constraint_ties_slave_dof_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "loads": [{"node": "3", "fy": -10.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertIn("mpc", stage.solver_info)
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-8)
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=8)

    def test_mpc_elimination_ties_slave_dof_without_penalty(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "loads": [{"node": "3", "fy": -10.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "elimination"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertEqual(stage.solver_info["method"], "mpc_elimination")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=12)

    def test_mpc_lagrange_ties_slave_dof_without_penalty(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "loads": [{"node": "3", "fy": -10.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertEqual(stage.solver_info["method"], "mpc_lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=12)
        self.assertEqual(len(stage.solver_info["multipliers"]), 1)
        self.assertTrue(stage.solver_info["reduced_matrix_cache"]["enabled"])
        self.assertEqual(stage.solver_info["reduced_matrix_cache"]["source"], "mpc_lagrange_free_dofs")
        self.assertTrue(stage.solver_info["lagrange_augmented_assembly"]["enabled"])
        self.assertEqual(stage.solver_info["lagrange_augmented_assembly"]["mode"], "lagrange_saddle_direct_fill")

    def test_nonlinear_mpc_lagrange_ties_slave_dof_without_penalty(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 10000.0, "nu": 0.30, "yield_stress": 1.0, "hardening": 100.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "loads": [{"node": "3", "fy": -10.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertEqual(stage.solver_info["method"], "newton")
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=12)
        lagrange_cache = stage.solver_info["lagrange_linear_cache"]
        self.assertTrue(lagrange_cache["enabled"])
        self.assertEqual(lagrange_cache["builds"], 1)
        self.assertGreaterEqual(lagrange_cache["hits"], 1)
        self.assertEqual(lagrange_cache["current"]["cache_kind"], "mpc_lagrange_linear_correction_cache")
        self.assertTrue(lagrange_cache["current"]["direct_fill"]["enabled"])

    def test_axisymmetric_static_elastic_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [1.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0}, {"nodes": ["1", "2"], "uy": 0.0}],
            "loads": [{"edge": ["3", "4"], "ty": -1.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
            self.assertEqual(stage.solver_info["geometry"], "axisymmetric")
            self.assertTrue(np.all(np.isfinite(stage.displacements)))
            self.assertTrue((stage.output_dir / "element_stress.csv").exists())

    def test_axisymmetric_nonlinear_mpc_lagrange_uses_direct_fill_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [1.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 10000.0, "nu": 0.30, "yield_stress": 1.0, "hardening": 100.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0}, {"nodes": ["1", "2"], "uy": 0.0}],
            "loads": [{"edge": ["3", "4"], "ty": -1.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "ux", "coefficient": 1.0, "method": "lagrange"}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertEqual(stage.solver_info["method"], "axisymmetric_newton")
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertAlmostEqual(stage.displacements[2 * idx2], stage.displacements[2 * idx3], places=12)
        lagrange_cache = stage.solver_info["lagrange_linear_cache"]
        self.assertTrue(lagrange_cache["enabled"])
        self.assertEqual(lagrange_cache["builds"], 1)
        self.assertGreaterEqual(lagrange_cache["hits"], 1)
        self.assertEqual(lagrange_cache["current"]["cache_kind"], "mpc_lagrange_linear_correction_cache")
        self.assertTrue(lagrange_cache["current"]["direct_fill"]["enabled"])

    def test_axisymmetric_linear_disabled_step_cache_still_uses_reduced_matrix_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [1.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0}, {"nodes": ["1", "2"], "uy": 0.0}],
            "loads": [{"edge": ["3", "4"], "ty": -1.0}],
            "solver": {"static_step_cache": {"enabled": False}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        self.assertFalse(stage.solver_info["topology_cache"]["enabled"])
        self.assertFalse(stage.solver_info["axisymmetric_linear_static_cache"]["enabled"])
        reduced = stage.solver_info["reduced_matrix_cache"]
        self.assertTrue(reduced["enabled"])
        self.assertEqual(reduced["source"], "csr_matrix")
        self.assertTrue(reduced["built"])
        self.assertFalse(reduced["reused"])
        self.assertGreater(reduced["free_free"]["nnz"], 0)
        self.assertIn("reduced_matrix_elapsed_seconds", stage.solver_info["linear_solver"])
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_axisymmetric_linear_step_cache_on_off_produces_same_solution(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 12000.0, "nu": 0.29}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "loads": [{"set": "right", "fx": 1.0}, {"set": "top", "fy": -0.25}],
        }
        uncached_cfg = {
            **cfg,
            "solver": {"static_step_cache": {"enabled": False}},
        }
        with tempfile.TemporaryDirectory() as cached_tmp, tempfile.TemporaryDirectory() as uncached_tmp:
            cached = solve_plane_strain_config(cfg, cached_tmp).stages[0]
            uncached = solve_plane_strain_config(uncached_cfg, uncached_tmp).stages[0]

        self.assertTrue(cached.solver_info["topology_cache"]["stiffness_assembly_cache"]["precomputed_linear_batches"]["enabled"])
        self.assertFalse(uncached.solver_info["topology_cache"]["enabled"])
        self.assertTrue(np.allclose(cached.displacements, uncached.displacements, rtol=1.0e-10, atol=1.0e-12))
        self.assertTrue(np.allclose(cached.reactions, uncached.reactions, rtol=1.0e-10, atol=1.0e-10))
        self.assertEqual(len(cached.element_results), len(uncached.element_results))
        for cached_row, uncached_row in zip(cached.element_results, uncached.element_results, strict=True):
            self.assertEqual(cached_row["element_id"], uncached_row["element_id"])
            for key in ("sigma_x", "sigma_y", "sigma_z", "tau_xy", "eps_x", "eps_y", "eps_z", "gamma_xy", "p", "q"):
                self.assertAlmostEqual(float(cached_row[key]), float(uncached_row[key]), delta=1.0e-8)

    def test_axisymmetric_von_mises_plastic_history_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [1.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 5.0, "hardening": 10.0}},
            "boundary_conditions": [
                {"nodes": ["1", "4"], "ux": 0.0},
                {"nodes": ["2", "3"], "ux": 0.01},
                {"nodes": ["1", "2", "3", "4"], "uy": 0.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        self.assertEqual(stage.solver_info["method"], "axisymmetric_newton")
        self.assertEqual(stage.element_results[0]["plastic"], 1.0)
        self.assertGreater(max(state.kappa for state in stage.plastic_state.values()), 0.0)

    def test_axisymmetric_newton_reports_combined_assembly_and_caches(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [1.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 1.0e9, "hardening": 10.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0}, {"nodes": ["1", "2"], "uy": 0.0}],
            "loads": [{"edge": ["2", "3"], "tx": 1.0}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        info = result.stages[0].solver_info
        self.assertEqual(info["method"], "axisymmetric_newton")
        self.assertTrue(info["combined_tangent_internal_assembly"])
        self.assertTrue(info["plastic_state_array_cache"]["enabled"])
        self.assertGreater(info["plastic_state_array_cache"]["state_points"], 0)
        topology = info["topology_cache"]
        self.assertTrue(topology["enabled"])
        self.assertTrue(topology["auto_generated"])
        self.assertEqual(topology["cache_kind"], "axisymmetric_step_cache")
        self.assertTrue(info["sparse_pattern_cached"])
        self.assertTrue(info["constraint_dofs_cached"])
        self.assertTrue(info["reduced_matrix_cache"]["enabled"])
        self.assertGreaterEqual(info["reduced_matrix_cache"]["solves"], 1)
        self.assertGreaterEqual(info["symbolic_ordering_cache"]["solves"], 1)

    def test_axisymmetric_srm_strength_reduction_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [2.0, 0.0], "3": [2.0, 1.0], "4": [1.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 50000.0, "nu": 0.30, "yield_stress": 60.0}},
            "boundary_conditions": [
                {"nodes": ["1", "4"], "ux": 0.0},
                {"nodes": ["2", "3"], "ux": 0.001},
                {"nodes": ["1", "2", "3", "4"], "uy": 0.0},
            ],
            "steps": [{"name": "axisym-srm", "type": "srm", "srm": {"factors": [1.0, 2.0], "failure_plastic_ratio": 0.0}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        srm = result.stages[0].solver_info["srm"]
        self.assertEqual(len(srm["trials"]), 2)
        self.assertEqual(srm["factor_of_safety"], 1.0)
        self.assertTrue(srm["lightweight_postprocess"]["enabled"])
        self.assertTrue(result.stages[0].solver_info["postprocess_results"])
        self.assertTrue(all(row["postprocess_results"] is False for row in srm["trials"]))
        self.assertTrue(all(row["plastic_ratio_source"] == "plastic_state" for row in srm["trials"]))

    def test_axisymmetric_geostatic_k0_initial_stress_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 2.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 10.0}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [{"name": "k0", "type": "geostatic", "apply_gravity": False, "surface_y": 2.0, "k0": 0.4}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].element_results[0]
        info = result.stages[0].solver_info
        self.assertEqual(info["geometry"], "axisymmetric")
        topology = info["topology_cache"]
        self.assertTrue(topology["enabled"])
        self.assertTrue(topology["stiffness_pattern_cached"])
        self.assertTrue(topology["stiffness_assembly_cache"]["direct_fill"]["enabled"])
        self.assertEqual(topology["batched_quad4_axisymmetric_elastic_elements"], 1)
        self.assertTrue(info["axisymmetric_linear_static_cache"]["stiffness_cache_used"])
        self.assertTrue(info["reduced_matrix_cache"]["enabled"])
        post_commit = info["postprocess_state_commit"]
        self.assertEqual(post_commit["geometry"], "axisymmetric")
        self.assertEqual(post_commit["state_commit"], "element_loop")
        self.assertEqual(post_commit["row_generation"], "element_loop")
        self.assertGreaterEqual(post_commit["committed_points"], 1)
        performance = info["performance"]
        for key in (
            "assembly_elapsed_seconds",
            "stiffness_assembly_elapsed_seconds",
            "load_assembly_elapsed_seconds",
            "linear_solve_elapsed_seconds",
            "postprocess_elapsed_seconds",
            "cache_build_elapsed_seconds",
            "coupled_assembly_elapsed_seconds",
        ):
            self.assertIn(key, performance)
            self.assertGreaterEqual(performance[key], 0.0)
        self.assertAlmostEqual(row["sigma_y"], -10.0, places=10)
        self.assertAlmostEqual(row["sigma_x"], -4.0, places=10)
        self.assertAlmostEqual(row["sigma_z"], -4.0, places=10)

    def test_axisymmetric_excavation_stage_deactivates_elements_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 3.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "sets": {"elements": {"right": ["2"]}},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "steps": [
                {"name": "before", "type": "static", "loads": [{"edge": ["4", "5"], "ty": -1.0}]},
                {"name": "excavate-right", "type": "excavation", "set": "right", "loads": [{"edge": ["4", "5"], "ty": -1.0}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        self.assertEqual([stage.name for stage in result.stages], ["before", "excavate-right"])
        self.assertEqual(len(result.stages[0].active_elements), 2)
        self.assertEqual(result.stages[1].active_elements, ["1"])
        inactive_row = next(row for row in result.stages[1].element_results if row["element_id"] == "2")
        self.assertEqual(inactive_row["active"], 0.0)
        self.assertTrue(np.all(np.isfinite(result.stages[1].displacements)))

    def test_axisymmetric_consolidation_pressure_diffusion_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [
                {
                    "name": "axisym-consolidation",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 100.0,
                        "dt": 0.1,
                        "steps": 2,
                        "storage": 1.0,
                        "permeability": 1.0,
                        "pressure_bcs": [{"set": "top", "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "pore_pressure.csv").exists())
        stage = result.stages[0]
        pressure = stage.pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        self.assertEqual(stage.solver_info["method"], "axisymmetric_monolithic_up")
        self.assertEqual(stage.solver_info["consolidation"]["unknowns"], 12)
        step_row = stage.solver_info["consolidation"]["step_history"][0]
        self.assertIn("coupled_assembly_elapsed_seconds", step_row)
        self.assertIn("reduced_matrix_elapsed_seconds", step_row)
        self.assertIn("linear_solve_elapsed_seconds", step_row)
        self.assertIn("elapsed_seconds", step_row)
        self.assertIn("performance", stage.solver_info)
        self.assertTrue(np.allclose(pressure[[2, 3]], 0.0))
        self.assertTrue(np.all((pressure[[0, 1]] > 0.0) & (pressure[[0, 1]] < 100.0)))

    def test_axisymmetric_nonlinear_up_reuses_hydraulic_boundary_and_reduced_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 4,
                "ny": 2,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 10.0, "friction_angle": 25.0}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "solver": {"linear": {"cache_min_size": 0, "symbolic_cache_min_size": 0}},
            "steps": [
                {
                    "name": "axisym-nonlinear-up-cache",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 50.0,
                        "dt": 0.1,
                        "steps": 3,
                        "storage": 1.0,
                        "permeability": 0.5,
                        "pressure_bcs": [{"set": "top", "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        info = stage.solver_info["consolidation"]
        cache = info["step_cache"]
        self.assertFalse(cache["enabled"])
        self.assertEqual(cache["reason"], "disabled_for_nonlinear_coupled_stage")
        self.assertEqual(cache["boundary_cache_reuses"], 3)
        self.assertEqual(cache["pressure_lhs_reuses"], 3)
        self.assertGreaterEqual(cache["reduced_cache_reuses"], 1)
        self.assertGreaterEqual(cache["factor_cache_hits"], 1)
        hydraulic = info["hydraulic_cache"]
        self.assertEqual(hydraulic["pressure_matrices"]["cache_kind"], "pressure_matrix_assembly_cache")
        self.assertEqual(hydraulic["biot_coupling"]["cache_kind"], "biot_coupling_assembly_cache")
        self.assertEqual(hydraulic["boundary_terms"]["cache_kind"], "pressure_boundary_term_cache")
        self.assertGreaterEqual(hydraulic["pressure_matrices"]["batched_elements"], 1)
        self.assertGreaterEqual(hydraulic["biot_coupling"]["batched_elements"], 1)
        monolithic = info["monolithic_lhs_pattern_cache"]
        self.assertTrue(monolithic["enabled"])
        self.assertGreaterEqual(monolithic["builds"], 1)
        self.assertTrue(monolithic["current"]["direct_fill"]["enabled"])
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_axisymmetric_consolidation_robin_mass_balance_diagnostics_run_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [
                {
                    "name": "axisym-consolidation",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 1.0,
                        "dt": 1.0,
                        "steps": 1,
                        "storage": 1.0,
                        "permeability": 0.0,
                        "pore_flux_bcs": [{"set": "bottom", "flux": 2.0}],
                        "pore_robin_bcs": [{"set": "top", "beta": 0.5, "pressure": -10.0, "seepage_face": True}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        pressure = stage.pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        info = stage.solver_info["consolidation"]
        self.assertAlmostEqual(info["boundary"]["flux_total"], 6.0 * math.pi)
        self.assertAlmostEqual(info["boundary"]["robin_conductance_total"], 1.5 * math.pi)
        self.assertEqual(info["seepage_active_edges"], 1)
        self.assertLess(info["mass_balance"], 1.0e-10)
        self.assertGreater(float(np.max(pressure)), 0.0)

    def test_axisymmetric_consolidation_interface_hydraulic_transfer_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [1.0, 1.0], "3": [1.0, 0.0], "4": [1.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 1000.0,
                    "behavior": {"hydro": {"transfer": 2.0}},
                }
            ],
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "fixed": True}],
            "steps": [
                {
                    "name": "axisym-interface-transfer",
                    "type": "consolidation",
                    "hydro": {
                        "dt": 1.0,
                        "steps": 1,
                        "storage": 1.0,
                        "permeability": 0.0,
                        "pore_flux_bcs": [{"nodes": ["1", "2"], "flux": 1.0}],
                        "pressure_bcs": [{"nodes": ["3", "4"], "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        pressure = result.stages[0].pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        info = result.stages[0].solver_info["consolidation"]
        self.assertEqual(info["interface_transfer"]["count"], 1)
        self.assertAlmostEqual(info["interface_transfer"]["conductance_total"], 4.0 * math.pi)
        self.assertAlmostEqual(info["boundary"]["flux_total"], 2.0 * math.pi)
        self.assertTrue(np.all(pressure[[0, 1]] > 0.0))
        self.assertTrue(np.allclose(pressure[[2, 3]], 0.0))

    def test_axisymmetric_consolidation_mpc_lagrange_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "mpc_constraints": [{"master": "2", "slave": "4", "dof": "ux", "coefficient": 1.0, "method": "lagrange"}],
            "steps": [
                {
                    "name": "axisym-up-lm",
                    "type": "consolidation",
                    "hydro": {"dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0, "pressure_bcs": [{"set": "top", "pressure": 0.0}]},
                    "loads": [{"edge": ["3", "4"], "ty": -1.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx4 = result.mesh.node_index["4"]
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertAlmostEqual(stage.displacements[2 * idx2], stage.displacements[2 * idx4], places=12)
        consolidation = stage.solver_info["consolidation"]
        self.assertTrue(consolidation["lagrange_linear_cache"]["enabled"])
        self.assertEqual(consolidation["lagrange_linear_cache"]["cache_kind"], "mpc_lagrange_linear_correction_cache")
        self.assertEqual(consolidation["step_cache"]["lagrange_linear_cache_builds"], 1)

    def test_excavation_stage_deactivates_elements_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 2,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "sets": {"elements": {"right": ["2"]}},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0, "uy": 0.0}, {"set": "bottom", "uy": 0.0}],
            "steps": [
                {"name": "before", "type": "static", "loads": [{"edge": ["4", "5"], "ty": -5.0}]},
                {"name": "excavate-right", "type": "excavation", "set": "right", "loads": [{"edge": ["4", "5"], "ty": -5.0}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        self.assertEqual([stage.name for stage in result.stages], ["before", "excavate-right"])
        self.assertEqual(len(result.stages[0].active_elements), 2)
        self.assertEqual(result.stages[1].active_elements, ["1"])
        inactive_row = next(row for row in result.stages[1].element_results if row["element_id"] == "2")
        self.assertEqual(inactive_row["active"], 0.0)
        self.assertEqual(inactive_row["sigma_y"], 0.0)
        self.assertTrue(np.all(np.isfinite(result.stages[1].displacements)))

    def test_geostatic_k0_initial_stress_stays_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 2.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 20000.0, "nu": 0.30, "gamma": 18.0, "k0": 0.5}},
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "fixed": True}],
            "steps": [{"name": "k0", "type": "geostatic", "surface_y": 2.0, "apply_gravity": False}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].element_results[0]
        self.assertAlmostEqual(row["sigma_y"], -18.0)
        self.assertAlmostEqual(row["sigma_x"], -9.0)
        self.assertAlmostEqual(row["sigma_z"], -9.0)

    def test_linear_interface_reaction_stays_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [0.0, 0.0], "4": [0.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [{"id": "joint", "minus_nodes": ["1", "2"], "plus_nodes": ["3", "4"], "kn": 1000.0, "kt": 500.0}],
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "ux": 0.01, "uy": 0.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        self.assertEqual(len(result.interfaces), 1)
        plus_reaction_x = stage.reactions[4] + stage.reactions[6]
        self.assertAlmostEqual(abs(plus_reaction_x), 10.0)

    def test_structural_line_elements_assemble_and_write_post_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [
                {"id": "bar", "type": "BAR2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0}},
                {"id": "shear", "type": "SHEAR_SPRING2", "nodes": ["1", "2"], "stiffness": {"ky": 500.0}},
            ],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}],
            "steps": [{"name": "load", "loads": [{"node": "2", "fx": 10.0, "fy": 5.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
            self.assertTrue((stage.output_dir / "structural_state.csv").exists())
            with (stage.output_dir / "structural_state.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(stage.displacements[2], 0.01)
        self.assertAlmostEqual(stage.displacements[3], 0.01)
        axial = {row["element_id"]: float(row["axial_force"]) for row in rows}
        spring = {row["element_id"]: float(row["spring_reaction"]) for row in rows}
        self.assertAlmostEqual(axial["bar"], 10.0)
        self.assertAlmostEqual(spring["shear"], 5.0)

    def test_response_seismic_time_history_and_layer_design_are_processed(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "steps": [
                {
                    "name": "eq",
                    "time": 1.0,
                    "seismic": {
                        "method": "response_seismic",
                        "time_history": [{"time": 0.0, "kh": 0.05, "kv": 0.0}, {"time": 1.0, "kh": 0.10, "kv": 0.02}],
                        "layers": [{"name": "all", "y_min": 0.0, "y_max": 1.0, "kh": 0.20, "kv": 0.03}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        self.assertEqual(stage.solver_info["seismic"]["method"], "response_seismic")
        self.assertEqual(stage.solver_info["seismic"]["layer_count"], 1)
        self.assertEqual(stage.solver_info["seismic"]["generated_loads"], 1)
        self.assertGreater(float(np.linalg.norm(stage.displacements)), 0.0)

    def test_load_combination_scales_cases_and_writes_report_tables(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [{"id": "bar", "type": "BAR2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0}}],
            "load_cases": [{"name": "LC1", "type": "static", "scale": 2.0, "active": True, "description": "dead"}],
            "load_combinations": [{"name": "COMB1", "factors": {"LC1": 0.5}}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [{"name": "comb", "load_combination": "COMB1", "loads": [{"node": "2", "fx": 10.0, "load_case": "LC1"}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.output_dir / "load_combinations.csv").exists())
            self.assertTrue((result.output_dir / "post_case_comparison.csv").exists())
            report = (result.output_dir / "calculation_report.html").read_text(encoding="utf-8")
        stage = result.stages[0]
        self.assertAlmostEqual(stage.displacements[2], 0.01, places=12)
        self.assertEqual(stage.solver_info["load_processing"]["load_combination"], "COMB1")
        self.assertIn("荷重組合せ照査", report)

    def test_calculation_report_writes_direct_pdf_template_manifest_and_frozen_inputs(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [{"id": "bar", "type": "BAR2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0}}],
            "load_cases": [{"name": "D", "type": "dead"}, {"name": "EQ", "type": "earthquake"}],
            "load_combinations": [{"name": "COMB", "factors": {"D": 1.0, "EQ": 0.5}}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [{"name": "report-case", "load_combination": "COMB", "loads": [{"node": "2", "fx": 10.0, "load_case": "D"}, {"node": "2", "fx": 4.0, "load_case": "EQ"}]}],
            "report": {
                "template": {"id": "geofeas_review", "template_revision": "test-rev"},
                "title": "Project X Calculation Report",
                "project": "Project X",
                "company": "Example Engineering",
                "report_no": "GX-001",
                "prepared_by": "analyst",
                "checked_by": "checker",
                "approved_by": "approver",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            html_path = result.output_dir / "calculation_report.html"
            pdf_path = result.output_dir / "calculation_report.pdf"
            manifest_path = result.output_dir / "calculation_report_manifest.json"
            snapshot_path = result.output_dir / "calculation_report_input_snapshot.json"
            summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            html_text = html_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            pdf_bytes = pdf_path.read_bytes()
        self.assertTrue(pdf_bytes.startswith(b"%PDF-1.4"))
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertIn("data-report-feature", html_text)
        self.assertIn("Table 1. Title Block", html_text)
        self.assertIn("Figure 1.", html_text)
        self.assertIn("reproducibility-freeze", html_text)
        self.assertEqual(manifest["template"]["template_id"], "geofeas_review")
        self.assertIn("direct_pdf", manifest["features"])
        self.assertEqual(len(manifest["reproducibility"]["input_sha256"]), 64)
        self.assertTrue(manifest["reproducibility"]["frozen"])
        self.assertTrue(any(row["stage"] == "report-case" for row in manifest["case_comparison"]))
        self.assertEqual(snapshot["report"]["project"], "Project X")
        self.assertIn("calculation_report.pdf", summary["report_pdf"])
        self.assertIn("calculation_report_manifest.json", summary["report_manifest"])

    def test_geofeas_public_profile_records_workflow_release_and_post_metadata(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "profile": PUBLIC_PROFILE},
            "mesh": {"generator": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0, "uy": 0.0}, {"set": "bottom", "uy": 0.0}],
            "steps": [
                {"name": "heading-release", "type": "static", "geofeas_workflow": "tunnel_excavation", "stress_release": 40, "loads": [{"edge": ["3", "4"], "ty": -1.0}]},
                {"name": "bench-release", "type": "static", "geofeas_workflow": "tunnel_excavation", "stress_release": 0.6, "loads": [{"edge": ["3", "4"], "ty": -1.0}]},
            ],
            "post": {"geofeas_style": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((result.output_dir / "calculation_report_manifest.json").read_text(encoding="utf-8"))
            with (result.output_dir / "post_case_comparison.csv").open(encoding="utf-8") as f:
                comparison_rows = list(csv.DictReader(f))
            operation_log_path = result.stages[0].output_dir / "geofeas_public_operation_log.json"
            operation_log = json.loads(operation_log_path.read_text(encoding="utf-8"))
            public_profile = json.loads((result.output_dir / "geofeas_public_profile.json").read_text(encoding="utf-8"))
            output_conditions = json.loads((result.output_dir / "geofeas_public_output_conditions.json").read_text(encoding="utf-8"))
            operation_log_exists = operation_log_path.exists()
            public_profile_exists = (result.output_dir / "geofeas_public_profile.json").exists()
            output_conditions_exists = (result.output_dir / "geofeas_public_output_conditions.json").exists()
        first_public = result.stages[0].solver_info["geofeas_public"]
        second_public = result.stages[1].solver_info["geofeas_public"]
        self.assertEqual(first_public["workflow"], "tunnel_excavation")
        self.assertAlmostEqual(first_public["stress_release"]["stress_release"], 0.4)
        self.assertAlmostEqual(second_public["stress_release"]["cumulative_release"], 1.0)
        self.assertTrue(second_public["stress_release"]["release_ok"])
        self.assertTrue(operation_log_exists)
        self.assertTrue(public_profile_exists)
        self.assertTrue(output_conditions_exists)
        self.assertEqual(summary["geofeas_public"]["profile"], PUBLIC_PROFILE)
        self.assertIn("geofeas_public_profile", manifest["features"])
        self.assertIn("geofeas_public_output_conditions", manifest["features"])
        self.assertIn("principal_stress_figure", first_public["post"]["views"])
        self.assertTrue(any("rectangle selection" in row["action"] for row in operation_log["operation_log"]))
        self.assertTrue(any("output condition" in row["action"] for row in operation_log["operation_log"]))
        self.assertTrue(any("youtube.com/watch" in source["url"] for source in public_profile["sources"]))
        self.assertTrue(output_conditions["save_behavior"]["supports_explicit_overwrite"])
        self.assertFalse(output_conditions["save_behavior"]["commercial_oss_roundtrip"])
        self.assertEqual(output_conditions["movie_observed_items"][2]["id"], "principal_stress_figure")
        self.assertEqual(comparison_rows[0]["geofeas_workflow"], "tunnel_excavation")
        self.assertEqual(comparison_rows[0]["relative_stage"], "previous")

    def test_geofeas_public_liquefaction_metadata_tracks_method_and_stage_roles(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"], "profile": PUBLIC_PROFILE},
            "mesh": {"generator": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "sand"},
            "materials": {
                "sand": {
                    "model": "bilinear_liquefaction",
                    "E": 10000.0,
                    "nu": 0.30,
                    "gamma": 18.0,
                    "G0": 6000.0,
                    "gamma_ref": 0.001,
                    "liquefaction": {
                        "cyclic_stress_method": "gauss_overburden",
                        "initial_effective_stress": 5.0,
                        "cyclic_resistance_ratio": 0.2,
                        "cyclic_stress_ratio": 0.18,
                        "generation_rate": 0.2,
                        "cycles_per_step": 1.0,
                    },
                }
            },
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [{"name": "river-liq", "type": "consolidation", "geofeas_workflow": "river_liquefaction_h28", "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0}}],
            "post": {"geofeas_style": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            public = result.stages[0].solver_info["geofeas_public"]
            summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
        liquefaction = public["liquefaction"]
        self.assertEqual(public["workflow"], "river_liquefaction_h28")
        self.assertEqual(liquefaction["cyclic_stress_methods"], ["gauss_overburden"])
        self.assertTrue(liquefaction["second_order_required"])
        self.assertFalse(liquefaction["second_order_present"])
        self.assertTrue(any("second-order" in warning for warning in summary["warnings"]))

    def test_geofeas_like_benchmark_suite_validates_values_post_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_geofeas_benchmark_suite(Path(tmp) / "benchmarks")
            out = Path(tmp) / "benchmarks"
            with (out / "benchmark_tolerance.csv").open(encoding="utf-8") as f:
                tolerance_rows = list(csv.DictReader(f))
            public_matrix = json.loads((out / "geofeas_public_compatibility_matrix.json").read_text(encoding="utf-8"))
            artifact_exists = {
                "summary": (out / "benchmark_summary.json").exists(),
                "metrics": (out / "benchmark_metrics.csv").exists(),
                "report": (out / "benchmark_report.html").exists(),
                "public_csv": (out / "geofeas_public_compatibility_matrix.csv").exists(),
                "public_html": (out / "geofeas_public_compatibility_matrix.html").exists(),
            }
            manifests_exist = [Path(case["reports"]["manifest"]).exists() for case in summary["cases"]]
        self.assertTrue(summary["passed"], summary)
        self.assertEqual(summary["case_count"], 7)
        self.assertEqual(summary["failed_count"], 0)
        names = {case["name"] for case in summary["cases"]}
        self.assertEqual(
            names,
            {
                "thin_axisymmetric_k0",
                "srm_strength_reduction",
                "excavation_death",
                "liquefaction_up_history",
                "seepage_water_pressure",
                "lagrange_mpc",
                "joint_mohr_coulomb",
            },
        )
        self.assertTrue(artifact_exists["summary"])
        self.assertTrue(artifact_exists["metrics"])
        self.assertTrue(artifact_exists["report"])
        self.assertTrue(artifact_exists["public_csv"])
        self.assertTrue(artifact_exists["public_html"])
        self.assertTrue(summary["public_compatibility"]["passed"])
        self.assertGreaterEqual(public_matrix["public_implemented_count"], 1)
        self.assertGreaterEqual(public_matrix["blocked_proprietary_count"], 1)
        self.assertTrue(any(row["public_status"] == "implemented_public" for row in public_matrix["rows"]))
        self.assertTrue(any(row["public_status"] == "blocked_proprietary" for row in public_matrix["rows"]))
        self.assertTrue(any(row["check"] == "srm.factor_of_safety" and row["passed"] == "True" for row in tolerance_rows))
        self.assertTrue(any(row["check"] == "report.calculation_report.pdf" and row["passed"] == "True" for row in tolerance_rows))
        self.assertTrue(any(row["check"] == "post.svg_count" and row["passed"] == "True" for row in tolerance_rows))
        self.assertTrue(all(manifests_exist))
        for case in summary["cases"]:
            self.assertGreaterEqual(case["post_appearance"]["svg_count"], 1)

    def test_seepage_time_sync_head_conversion_and_water_level_update(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"top": ["3", "4"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "ux": 0.0, "uy": 0.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            seepage = Path(tmp) / "seepage.csv"
            seepage.write_text("time,node_id,head\n0.0,1,1.0\n1.0,1,2.0\n1.0,2,2.0\n", encoding="utf-8")
            cfg["steps"] = [
                {
                    "name": "hydro-static",
                    "time": 1.0,
                    "hydro": {
                        "initial_pressure": 0.0,
                        "seepage_csv": str(seepage),
                        "water_levels": [{"set": "top", "water_level": 2.0}],
                    },
                }
            ]
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "pore_pressure.csv").exists())
        stage = result.stages[0]
        self.assertIsNotNone(stage.pore_pressure)
        assert stage.pore_pressure is not None
        self.assertAlmostEqual(stage.pore_pressure[0], 9.80665 * 2.0, places=8)
        self.assertAlmostEqual(stage.pore_pressure[2], 9.80665 * 1.0, places=8)
        self.assertEqual(stage.solver_info["hydro_sync"]["seepage_sync_count"], 2)
        self.assertEqual(stage.solver_info["hydro_sync"]["water_level_update_count"], 2)

    def test_newmark_dynamic_time_history_uses_mass_damping_and_writes_history(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "steps": [
                {
                    "name": "dyn",
                    "type": "dynamic_time_history",
                    "dynamic": {"rayleigh_alpha": 0.02},
                    "seismic": {
                        "method": "response_seismic",
                        "time_history": [{"time": 0.0, "kh": 0.0}, {"time": 0.1, "kh": 0.10}, {"time": 0.2, "kh": 0.0}],
                    },
                }
            ],
        }
        clear_linear_factor_cache()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = solve_plane_strain_config(cfg, tmp)
                with (result.stages[0].output_dir / "dynamic_history.csv").open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
        finally:
            linear_cache = linear_factor_cache_info()
            clear_linear_factor_cache()
        stage = result.stages[0]
        self.assertEqual(stage.solver_info["method"], "newmark")
        self.assertEqual(stage.solver_info["dynamic"]["steps"], 2)
        self.assertEqual(len(rows), 3)
        self.assertGreater(float(rows[1]["max_acceleration"]), 0.0)
        cache = stage.solver_info["dynamic"]["effective_stiffness_cache"]
        self.assertTrue(cache["enabled"])
        self.assertEqual(cache["builds"], 1)
        self.assertGreaterEqual(cache["hits"], 1)
        self.assertGreaterEqual(cache["linear_combination_builds"], 1)
        self.assertTrue(cache["reduced_matrix_cached"])
        damping_cache = stage.solver_info["dynamic"]["damping_matrix_cache"]
        self.assertTrue(damping_cache["enabled"])
        self.assertEqual(damping_cache["mode"], "csr_linear_combination_direct_data")
        mass_step_cache = stage.solver_info["dynamic"]["mass_step_cache"]
        self.assertTrue(mass_step_cache["enabled"])
        self.assertTrue(mass_step_cache["mass_cache"]["direct_fill"]["enabled"])
        self.assertTrue(mass_step_cache["stiffness_cache"]["precomputed_linear_batches"]["enabled"])
        self.assertEqual(mass_step_cache["stiffness_cache"]["precomputed_linear_batches"]["element_batches"], 1)
        self.assertTrue(stage.solver_info["dynamic"]["mass"]["assembly_cache"]["enabled"])
        self.assertTrue(stage.solver_info["dynamic"]["load_vector_reused"])
        self.assertEqual(stage.solver_info["dynamic"]["history"][1]["effective_stiffness_cache_state"], "miss")
        self.assertEqual(stage.solver_info["dynamic"]["history"][2]["effective_stiffness_cache_state"], "hit")
        self.assertEqual(stage.solver_info["dynamic"]["history"][1]["effective_stiffness_linear_combination_state"], "miss")
        self.assertEqual(stage.solver_info["dynamic"]["history"][2]["effective_stiffness_linear_combination_state"], "matrix_reused")
        self.assertTrue(stage.solver_info["dynamic"]["history"][1]["mass_matrix_cache_enabled"])
        self.assertTrue(stage.solver_info["dynamic"]["history"][2]["mass_matrix_cache_enabled"])
        self.assertTrue(stage.solver_info["dynamic"]["history"][2]["reduced_matrix_cache_reused"])
        self.assertEqual(stage.solver_info["dynamic"]["history"][1]["lu_factor_cache_state"], "miss")
        self.assertEqual(stage.solver_info["dynamic"]["history"][2]["lu_factor_cache_state"], "hit")
        self.assertGreaterEqual(stage.solver_info["dynamic"]["effective_stiffness_cache"]["factor_cache_hits"], 1)
        self.assertGreaterEqual(linear_cache["hits"], 1)
        self.assertGreaterEqual(stage.solver_info["performance"]["cache_build_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(stage.solver_info["performance"]["mass_assembly_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(stage.solver_info["performance"]["damping_assembly_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(stage.solver_info["performance"]["effective_stiffness_assembly_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(stage.solver_info["performance"]["linear_solve_elapsed_seconds"], 0.0)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_dynamic_mass_step_cache_on_off_produces_same_time_history(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "steps": [
                {
                    "name": "dyn",
                    "type": "dynamic_time_history",
                    "dynamic": {"rayleigh_alpha": 0.02},
                    "seismic": {
                        "method": "response_seismic",
                        "time_history": [{"time": 0.0, "kh": 0.0}, {"time": 0.1, "kh": 0.10}, {"time": 0.2, "kh": 0.0}],
                    },
                }
            ],
        }
        disabled_cfg = json.loads(json.dumps(cfg))
        disabled_cfg["steps"][0]["dynamic"]["mass_step_cache"] = False
        with tempfile.TemporaryDirectory() as cached_tmp, tempfile.TemporaryDirectory() as uncached_tmp:
            cached_result = solve_plane_strain_config(cfg, cached_tmp)
            uncached_result = solve_plane_strain_config(disabled_cfg, uncached_tmp)
            with (cached_result.stages[0].output_dir / "dynamic_history.csv").open(encoding="utf-8") as f:
                cached_rows = list(csv.DictReader(f))
            with (uncached_result.stages[0].output_dir / "dynamic_history.csv").open(encoding="utf-8") as f:
                uncached_rows = list(csv.DictReader(f))
        cached = cached_result.stages[0]
        uncached = uncached_result.stages[0]

        self.assertTrue(cached.solver_info["dynamic"]["mass_step_cache"]["enabled"])
        self.assertFalse(uncached.solver_info["dynamic"]["mass_step_cache"]["enabled"])
        self.assertTrue(np.allclose(cached.displacements, uncached.displacements, rtol=1.0e-10, atol=1.0e-12))
        self.assertEqual(len(cached_rows), len(uncached_rows))
        for cached_row, uncached_row in zip(cached_rows, uncached_rows, strict=True):
            self.assertEqual(cached_row["step"], uncached_row["step"])
            for key in ("time", "dt", "kh", "load_scale", "max_displacement", "max_velocity", "max_acceleration"):
                self.assertAlmostEqual(float(cached_row[key]), float(uncached_row[key]), delta=1.0e-10)

    def test_dynamic_time_dependent_displacement_boundary_is_enforced(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [{"id": "bar", "type": "BAR2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0, "mass_per_length": 1.0}}],
            "boundary_conditions": [
                {"node": "1", "ux": 0.0, "uy": 0.0},
                {"node": "2", "uy": 0.0},
                {"node": "2", "dof": "ux", "unit": "mm", "time_history": [{"time": 0.0, "value": 0.0}, {"time": 0.1, "value": 10.0}]},
            ],
            "steps": [{"name": "forced-dynamic", "type": "newmark", "dynamic": {"dt": 0.1, "steps": 1}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        self.assertAlmostEqual(result.stages[0].displacements[2], 0.01, places=12)
        self.assertEqual(result.stages[0].solver_info["dynamic"]["steps"], 1)

    def test_geofeas_like_dynamic_profile_records_damping_and_convergence_controls(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "steps": [
                {
                    "name": "profile-dyn",
                    "type": "newmark",
                    "dynamic": {
                        "profile": "geofeas_like",
                        "dt": 0.1,
                        "steps": 1,
                        "damping": {"damping_ratios": [0.05, 0.05], "frequencies_hz": [1.0, 3.0]},
                    },
                    "seismic": {"kh": 0.02},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        info = result.stages[0].solver_info["dynamic"]
        self.assertEqual(info["profile"]["name"], "geofeas_like")
        self.assertGreater(info["rayleigh_alpha"], 0.0)
        self.assertGreater(info["rayleigh_beta"], 0.0)
        self.assertIn("damping_ratios", info["damping_spec"])

    def test_nonlinear_dynamic_newmark_uses_newton_path_and_cutback_metadata(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "gamma": 20.0, "cohesion": 500.0, "friction_angle": 30.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "solver": {"newton": {"max_iter": 8, "tol_abs": 1.0e-7}, "linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [
                {
                    "name": "nonlinear-dyn",
                    "type": "dynamic_time_history",
                    "dynamic": {"nonlinear": True, "max_cutbacks": 1, "dt": 0.1, "steps": 2},
                    "seismic": {"kh": 0.05},
                }
            ],
        }
        clear_linear_factor_cache()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = solve_plane_strain_config(cfg, tmp)
        finally:
            clear_linear_factor_cache()
        info = result.stages[0].solver_info["dynamic"]
        self.assertTrue(info["nonlinear"])
        self.assertEqual(info["steps"], 2)
        self.assertEqual(info["cutbacks"], 0)
        newton_cache = info["newton_cache"]
        self.assertTrue(newton_cache["combined_tangent_internal_assembly"])
        self.assertTrue(newton_cache["plastic_state_array_cache"]["enabled"])
        self.assertEqual(newton_cache["reduced_matrix_cache"]["builds"], 1)
        self.assertGreaterEqual(newton_cache["reduced_matrix_cache"]["hits"], 1)
        first_step = info["history"][1]["convergence_history"][0]
        second_step = info["history"][2]["convergence_history"][0]
        self.assertGreaterEqual(first_step["tangent_internal_assembly_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(first_step["effective_stiffness_assembly_elapsed_seconds"], 0.0)
        self.assertTrue(first_step["reduced_matrix_cache_built"])
        self.assertTrue(second_step["reduced_matrix_cache_reused"])
        self.assertGreaterEqual(result.stages[0].solver_info["performance"]["effective_stiffness_assembly_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(result.stages[0].solver_info["performance"]["linear_solve_elapsed_seconds"], 0.0)
        self.assertIn(result.stages[0].solver_info["method"], {"newmark"})

    def test_dynamic_up_monolithic_updates_pressure_history(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "ux": 0.0, "uy": 0.0}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [
                {
                    "name": "dynamic-up",
                    "type": "dynamic_time_history",
                    "dynamic": {"dt": 0.1, "steps": 2, "up_coupled": True},
                    "hydro": {"initial_pressure": 0.0, "storage": 1.0, "permeability": 0.1, "pressure_bcs": [{"node": "1", "pressure": 5.0}]},
                }
            ],
        }
        clear_linear_factor_cache()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = solve_plane_strain_config(cfg, tmp)
                with (result.stages[0].output_dir / "dynamic_history.csv").open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
        finally:
            clear_linear_factor_cache()
        stage = result.stages[0]
        self.assertTrue(stage.solver_info["dynamic"]["up_coupled"])
        cache = stage.solver_info["dynamic"]["newton_cache"]["reduced_matrix_cache"]
        self.assertEqual(cache["builds"], 1)
        self.assertGreaterEqual(cache["hits"], 1)
        first_step = stage.solver_info["dynamic"]["history"][1]["convergence_history"][0]
        second_step = stage.solver_info["dynamic"]["history"][2]["convergence_history"][0]
        self.assertGreater(first_step["monolithic_assembly_elapsed_seconds"], 0.0)
        self.assertTrue(first_step["monolithic_lhs_direct_fill_built"])
        self.assertTrue(second_step["monolithic_lhs_direct_fill_reused"])
        self.assertTrue(first_step["reduced_matrix_cache_built"])
        self.assertTrue(second_step["reduced_matrix_cache_reused"])
        monolithic = stage.solver_info["dynamic"]["newton_cache"]["monolithic_lhs_pattern_cache"]
        self.assertTrue(monolithic["enabled"])
        self.assertEqual(monolithic["builds"], 1)
        self.assertGreaterEqual(monolithic["hits"], 1)
        combo_cache = stage.solver_info["dynamic"]["newton_cache"]["effective_stiffness_linear_combination_cache"]
        self.assertTrue(combo_cache["enabled"])
        self.assertGreaterEqual(combo_cache["builds"], 1)
        self.assertGreaterEqual(combo_cache["hits"], 1)
        self.assertIsNotNone(stage.pore_pressure)
        assert stage.pore_pressure is not None
        self.assertAlmostEqual(stage.pore_pressure[0], 5.0)
        self.assertIn("max_pore_pressure", rows[-1])

    def test_dynamic_newmark_cutback_retries_before_failure(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "solver": {"newton": {"max_iter": 0}},
            "steps": [{"name": "dyn-cutback", "type": "newmark", "dynamic": {"nonlinear": True, "dt": 0.1, "steps": 1, "max_cutbacks": 1, "cutback_factor": 0.5}, "seismic": {"kh": 0.1}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FEM2DError, "failed after 1 cutbacks"):
                solve_plane_strain_config(cfg, tmp)

    def test_dynamic_newmark_does_not_commit_allow_nonconvergence_step(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
                "node_sets": {"base": ["1", "2"]},
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30, "gamma": 20.0}},
            "boundary_conditions": [{"set": "base", "ux": 0.0, "uy": 0.0}],
            "solver": {"newton": {"max_iter": 0, "allow_nonconvergence": True}},
            "steps": [
                {
                    "name": "dyn-no-unconverged-commit",
                    "type": "newmark",
                    "dynamic": {"nonlinear": True, "dt": 0.1, "steps": 1, "max_cutbacks": 1, "cutback_factor": 0.5},
                    "seismic": {"kh": 0.1},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FEM2DError, "failed after 1 cutbacks"):
                solve_plane_strain_config(cfg, tmp)

    def test_external_seepage_geo_product_headers_are_normalized_and_compared(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            seepage = Path(tmp) / "GeoFEAS_seepage.tsv"
            seepage.write_text("時刻\t節点番号\t全水頭\n1.0\t1\t2.0\n", encoding="utf-8")
            rows = import_external_seepage_results(seepage)
            self.assertEqual(rows[0]["source_product"], "GeoFEAS")
            self.assertEqual(rows[0]["node_id"], "1")
            self.assertAlmostEqual(rows[0]["head"], 2.0)
            comparison = compare_external_seepage_results(seepage, seepage)
            self.assertTrue(comparison["passed"])
            diagnosis = diagnose_external_seepage_version(seepage)
            self.assertEqual(diagnosis["product"], "GeoFEAS")
            exported = Path(tmp) / "normalized_seepage.csv"
            roundtrip = compare_external_seepage_roundtrip(seepage, exported)
            self.assertTrue(roundtrip["passed"])
            cfg["steps"] = [{"name": "external-seepage", "time": 1.0, "hydro": {"initial_pressure": 0.0, "seepage_results": str(seepage)}}]
            result = solve_plane_strain_config(cfg, tmp)
        pressure = result.stages[0].pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        self.assertAlmostEqual(pressure[0], 9.80665 * 2.0, places=8)

    def test_load_combination_standard_templates_are_available_to_solver_and_report(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [{"id": "bar", "type": "BAR2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0}}],
            "load_combination_standard": "river_seismic",
            "load_cases": [{"name": "D", "type": "dead"}, {"name": "EQ", "type": "earthquake"}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [{"name": "river-combo", "load_combination": "river_l1_seismic", "loads": [{"node": "2", "fx": 10.0, "load_case": "D"}, {"node": "2", "fx": 5.0, "load_case": "EQ"}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            with (result.output_dir / "load_combinations.csv").open(encoding="utf-8") as f:
                combo_rows = list(csv.DictReader(f))
        self.assertAlmostEqual(result.stages[0].displacements[2], 0.015, places=12)
        self.assertTrue(any(row["combination"] == "river_l1_seismic" and row["case"] == "EQ" for row in combo_rows))

    def test_load_combination_template_manifest_records_revision_and_clauses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "combination_manifest.csv"
            write_load_combination_template_manifest(manifest_path, "river_seismic")
            with manifest_path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        manifest = load_combination_template_manifest("river_seismic")
        self.assertTrue(any(row["combination"] == "river_l1_seismic" and row["revision"] for row in manifest))
        self.assertTrue(any(row["clause"].startswith("open-template:river") for row in rows))

    def test_structural_stage_birth_death_controls_active_post(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [{"id": "bar", "type": "BAR2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0}}],
            "boundary_conditions": [{"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0}],
            "steps": [
                {"name": "active"},
                {"name": "inactive", "structural_deactivate": {"ids": ["bar"]}},
                {"name": "reactivated", "structural_activate": ["bar"]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        self.assertEqual([row["state"] for row in result.stages[0].structural_results], ["active"])
        self.assertEqual([row["state"] for row in result.stages[1].structural_results], ["inactive"])
        self.assertEqual([row["state"] for row in result.stages[2].structural_results], ["active"])

    def test_euler_beam_uses_rotational_dofs_and_section_forces(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [2.0, 0.0]}, "elements": []},
            "materials": {"steel": {"model": "elastic", "E": 3000.0, "nu": 0.30}},
            "structural_elements": [{"id": "beam", "type": "BEAM2", "nodes": ["1", "2"], "material": "steel", "section": {"A": 10.0, "I": 1.0}}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0, "rz": 0.0}],
            "steps": [{"name": "tip-load", "loads": [{"node": "2", "fy": -12.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        self.assertEqual(stage.displacements.size, 6)
        self.assertAlmostEqual(stage.displacements[3], -12.0 * 2.0**3 / (3.0 * 3000.0), places=10)
        row = stage.structural_results[0]
        self.assertAlmostEqual(row["rotation_j"], -12.0 * 2.0**2 / (2.0 * 3000.0), places=10)
        self.assertAlmostEqual(abs(row["end_moment_i"]), 24.0, places=10)

    def test_beam_uniform_load_and_end_release_are_reflected_in_post(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [2.0, 0.0]}, "elements": []},
            "materials": {"steel": {"model": "elastic", "E": 3000.0, "nu": 0.30}},
            "structural_elements": [{"id": "beam", "type": "BEAM2", "nodes": ["1", "2"], "material": "steel", "section": {"A": 10.0, "I": 1.0, "release_j": True}}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0, "rz": 0.0}],
            "steps": [{"name": "uniform-load", "loads": [{"type": "beam_uniform", "element": "beam", "qy_local": -6.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].structural_results[0]
        self.assertEqual(row["release_j"], 1)
        self.assertAlmostEqual(row["end_moment_j"], 0.0, places=10)
        self.assertLess(result.stages[0].displacements[3], 0.0)

    def test_beam_section_library_and_semi_rigid_end_spring_are_connected(self) -> None:
        base = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [2.0, 0.0]}, "elements": []},
            "materials": {"steel": {"model": "elastic", "E": 3000.0, "nu": 0.30}},
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0, "rz": 0.0}],
            "steps": [{"name": "tip-load", "loads": [{"node": "2", "fy": -12.0}]}],
        }
        rigid = dict(base)
        rigid["structural_elements"] = [{"id": "beam", "type": "BEAM2", "nodes": ["1", "2"], "material": "steel", "section": {"section_name": "RC_RECT_1M"}}]
        semi = dict(base)
        semi["structural_elements"] = [{"id": "beam", "type": "BEAM2", "nodes": ["1", "2"], "material": "steel", "section": {"section_name": "RC_RECT_1M", "rotational_spring_i": 100.0}}]
        with tempfile.TemporaryDirectory() as tmp:
            rigid_result = solve_plane_strain_config(rigid, Path(tmp) / "rigid")
            semi_result = solve_plane_strain_config(semi, Path(tmp) / "semi")
        row = semi_result.stages[0].structural_results[0]
        self.assertEqual(row["section_name"], "RC_RECT_1M")
        self.assertEqual(row["connection_i"], "semi_rigid")
        self.assertAlmostEqual(row["section_area"], 1.0, places=12)
        self.assertAlmostEqual(row["rotational_spring_i"], 100.0, places=12)
        self.assertGreater(abs(semi_result.stages[0].displacements[3]), abs(rigid_result.stages[0].displacements[3]))

    def test_bilinear_axial_spring_uses_nonlinear_tangent_and_history_post(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "structural_elements": [{"id": "spring", "type": "AXIAL_SPRING2", "nodes": ["1", "2"], "stiffness": {"kx": 1000.0, "law": "bilinear", "yield_force": 5.0, "post_yield_stiffness": 100.0}}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [{"name": "yield", "loads": [{"node": "2", "fx": 15.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        self.assertAlmostEqual(stage.displacements[2], 0.105, places=10)
        row = stage.structural_results[0]
        self.assertEqual(row["spring_state_axial"], "yielded")
        self.assertGreater(row["plastic_deformation_axial"], 0.0)
        self.assertAlmostEqual(row["axial_force"], 15.0, places=8)

    def test_hysteretic_spring_tracks_reversal_energy_and_degradation(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "solver": {"newton": {"tol_abs": 1.0e-4}},
            "structural_elements": [
                {
                    "id": "spring",
                    "type": "AXIAL_SPRING2",
                    "nodes": ["1", "2"],
                    "stiffness": {"kx": 1000.0, "law": "hysteretic", "yield_force": 5.0, "post_yield_stiffness": 100.0, "degradation": 0.1, "pinching": 0.3, "damping": 5.0},
                }
            ],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [
                {"name": "push", "loads": [{"node": "2", "fx": 15.0}]},
                {"name": "pull", "loads": [{"node": "2", "fx": -10.0}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[1].structural_results[0]
        self.assertGreaterEqual(row["reversal_count_axial"], 1.0)
        self.assertGreater(row["cumulative_energy_axial"], 0.0)
        self.assertLess(row["force_degradation_axial"], 1.0)
        self.assertLessEqual(row["pinching_factor_axial"], 1.0)

    def test_spring_hysteresis_model_preset_drives_pinching_history(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "solver": {"newton": {"tol_abs": 1.0e-4}},
            "structural_elements": [
                {
                    "id": "spring",
                    "type": "AXIAL_SPRING2",
                    "nodes": ["1", "2"],
                    "stiffness": {"kx": 1000.0, "hysteresis_model": "PINCHING_CLOUGH_LIKE", "yield_force": 5.0},
                }
            ],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [
                {"name": "push", "loads": [{"node": "2", "fx": 15.0}]},
                {"name": "pull", "loads": [{"node": "2", "fx": -10.0}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[1]
        row = stage.structural_results[0]
        self.assertEqual(row["spring_law"], "pinching")
        self.assertEqual(row["hysteresis_model"], "PINCHING_CLOUGH_LIKE")
        self.assertGreaterEqual(row["reversal_count_axial"], 1.0)
        self.assertLess(row["pinching_factor_axial"], 1.0)
        self.assertTrue(stage.solver_info["converged"])
        self.assertTrue(any(bool(item.get("final_residual_check", False)) for item in stage.solver_info["convergence_history"]))

    def test_spring_external_parameter_system_is_normalized(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0]}, "elements": []},
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "solver": {"newton": {"tol_abs": 1.0e-4}},
            "structural_elements": [
                {
                    "id": "spring",
                    "type": "AXIAL_SPRING2",
                    "nodes": ["1", "2"],
                    "stiffness": {
                        "kx": 1000.0,
                        "hysteresis_model": "GEOFEAS_LIKE_DEGRADING",
                        "commercial_parameter_system": "user_geo_feas_table",
                        "commercial_parameters": {
                            "yield_load": 5.0,
                            "hardening_ratio": 0.10,
                            "strength_loss_per_cycle": 0.20,
                            "pinch_ratio": 0.25,
                        },
                    },
                }
            ],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0}, {"node": "2", "uy": 0.0}],
            "steps": [{"name": "yield", "loads": [{"node": "2", "fx": 15.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].structural_results[0]
        self.assertEqual(row["spring_law"], "degrading")
        self.assertEqual(row["spring_parameter_system"], "user_geo_feas_table")
        self.assertEqual(row["spring_state_axial"], "hysteretic_yielded")
        self.assertGreater(row["plastic_deformation_axial"], 0.0)

    def test_beam_section_force_distribution_csv_is_written(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {"nodes": {"1": [0.0, 0.0], "2": [2.0, 0.0]}, "elements": []},
            "materials": {"steel": {"model": "elastic", "E": 3000.0, "nu": 0.30}},
            "structural_elements": [{"id": "beam", "type": "BEAM2", "nodes": ["1", "2"], "material": "steel", "section": {"A": 10.0, "I": 1.0}}],
            "boundary_conditions": [{"node": "1", "ux": 0.0, "uy": 0.0, "rz": 0.0}],
            "steps": [{"name": "uniform-load", "loads": [{"type": "beam_uniform", "element": "beam", "qy_local": -6.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            path = result.stages[0].output_dir / "structural_section_forces.csv"
            self.assertTrue(path.exists())
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        self.assertIn("bending_moment", rows[0])

    def test_friction_interface_limits_shear_reaction_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [0.0, 0.0], "4": [0.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 1000.0,
                    "behavior": {"friction": 0.5, "cohesion": 0.0},
                }
            ],
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "ux": -0.01, "uy": 0.02},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "interface_state.csv").exists())
        stage = result.stages[0]
        plus_reaction_y = stage.reactions[5] + stage.reactions[7]
        self.assertAlmostEqual(abs(plus_reaction_y), 5.0, places=10)
        self.assertTrue(any(row["state"] == "slip" for row in stage.interface_results))
        self.assertTrue(all(row["material_model"] == "mohr_coulomb" for row in stage.interface_results))
        self.assertTrue(any(row["contact_state"] == "closed_slip" for row in stage.interface_results))
        self.assertGreater(max(row["slip_abs"] for row in stage.interface_results), 0.0)

    def test_joint_mohr_coulomb_parameter_estimation_from_direct_shear_points(self) -> None:
        fit = estimate_joint_mohr_coulomb_parameters([(10.0, 7.0), (20.0, 12.0), (30.0, 17.0)])
        self.assertAlmostEqual(fit["cohesion"], 2.0, places=12)
        self.assertAlmostEqual(fit["friction"], 0.5, places=12)
        self.assertAlmostEqual(fit["friction_angle_deg"], math.degrees(math.atan(0.5)), places=12)

    def test_geofeas_reference_csv_comparison_accepts_matching_structural_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actual = Path(tmp) / "actual.csv"
            reference = Path(tmp) / "reference.csv"
            actual.write_text("element_id,axial_force,shear_force\nbeam,10.000001,2.0\n", encoding="utf-8")
            reference.write_text("element_id,axial_force,shear_force\nbeam,10.0,2.0\n", encoding="utf-8")
            summary = compare_geofeas_reference_csv(actual, reference, rtol=1.0e-5, atol=1.0e-7)
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["failed_count"], 0)

    def test_geofeas_tolerance_report_is_written_from_comparison_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actual = Path(tmp) / "actual.csv"
            reference = Path(tmp) / "reference.csv"
            report = Path(tmp) / "tolerance.html"
            actual.write_text("element_id,axial_force,shear_force\nbeam,10.0,2.0\n", encoding="utf-8")
            reference.write_text("element_id,axial_force,shear_force\nbeam,10.0,2.0\n", encoding="utf-8")
            summary = compare_geofeas_reference_csv(actual, reference)
            write_geofeas_tolerance_report(summary, report)
            text = report.read_text(encoding="utf-8")
        self.assertIn("Status: PASSED", text)
        self.assertIn("max_abs_error", text)

    def test_geofeas_stage_package_comparison_writes_structural_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actual_dir = Path(tmp) / "actual"
            reference_dir = Path(tmp) / "reference"
            out_dir = Path(tmp) / "report"
            actual_dir.mkdir()
            reference_dir.mkdir()
            structural = "element_id,axial_force,shear_force,end_moment_i,end_moment_j,rotation_i,rotation_j,spring_reaction\nbeam,10.0,2.0,-4.0,0.0,0.0,0.1,11.0\n"
            section = "element_id,type,node_i,node_j,x,ratio,axial_force,shear_force,bending_moment\nbeam,BEAM2,1,2,0.0,0.0,10.0,2.0,-4.0\n"
            interface = "interface_id,gp,gap_t,gap_n,traction_t,traction_n,slip_abs,opening,closure,friction_limit,effective_roughness\njoint,0,0.01,-0.02,5.0,-20.0,0.005,0.0,0.02,10.0,8.0\n"
            for root in (actual_dir, reference_dir):
                (root / "structural_state.csv").write_text(structural, encoding="utf-8")
                (root / "structural_section_forces.csv").write_text(section, encoding="utf-8")
                (root / "interface_state.csv").write_text(interface, encoding="utf-8")
            summary = compare_geofeas_stage_package(actual_dir, reference_dir, output_dir=out_dir)
            self.assertTrue((out_dir / "geofeas_package_tolerance.html").exists())
            self.assertTrue((out_dir / "structural_state_comparison.csv").exists())
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["compared_count"], 3)

    def test_geofeas_dynamic_sample_comparison_includes_dynamic_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actual_dir = Path(tmp) / "actual"
            reference_dir = Path(tmp) / "reference"
            out_dir = Path(tmp) / "report"
            actual_dir.mkdir()
            reference_dir.mkdir()
            history = "step,time,dt,kh,kv,load_scale,max_displacement,max_velocity,max_acceleration,max_pore_pressure,min_pore_pressure\n0,0.0,0.0,0,0,1,0,0,0,0,0\n1,0.1,0.1,0.05,0,1,0.01,0.1,1.0,5.0,0.0\n"
            for root in (actual_dir, reference_dir):
                (root / "dynamic_history.csv").write_text(history, encoding="utf-8")
            summary = compare_geofeas_dynamic_sample(actual_dir, reference_dir, output_dir=out_dir)
        self.assertTrue(summary["passed"])
        self.assertTrue(any(file["file"] == "dynamic_history.csv" and file["status"] == "compared" for file in summary["files"]))

    def test_no_tension_interface_releases_opening_reaction_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [0.0, 0.0], "4": [0.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 500.0,
                    "behavior": {"no_tension": True},
                }
            ],
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "ux": 0.01, "uy": 0.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        plus_reaction_x = stage.reactions[4] + stage.reactions[6]
        self.assertAlmostEqual(plus_reaction_x, 0.0, places=12)

    def test_joint_open_close_history_and_roughness_degradation_persist_across_stages(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [0.0, 0.0], "4": [0.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 1000.0,
                    "behavior": {
                        "friction": 0.5,
                        "cohesion": 0.0,
                        "no_tension": True,
                        "roughness": 10.0,
                        "dilatancy_angle": 8.0,
                        "roughness_degradation": 0.1,
                        "residual_roughness_ratio": 0.5,
                    },
                }
            ],
            "boundary_conditions": [{"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0}],
            "steps": [
                {"name": "closed-slip", "boundary_conditions": [{"nodes": ["3", "4"], "ux": -0.01, "uy": 0.02}]},
                {"name": "open", "boundary_conditions": [{"nodes": ["3", "4"], "ux": 0.01, "uy": 0.0}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[1].interface_results[0]
        self.assertEqual(row["normal_state"], "open")
        self.assertGreaterEqual(row["open_close_cycles"], 1.0)
        self.assertGreater(row["roughness_loss"], 0.0)
        self.assertLess(row["effective_roughness"], row["roughness"])

    def test_joint_standard_report_writes_history_and_roughness_summary(self) -> None:
        rows = [
            {
                "interface_id": "joint",
                "gp": 0,
                "state": "slip",
                "normal_state": "closed",
                "slip_abs": 0.01,
                "opening": 0.0,
                "closure": 0.02,
                "open_close_cycles": 1.0,
                "traction_t": 5.0,
                "traction_n": -20.0,
                "friction_limit": 10.0,
                "roughness": 10.0,
                "effective_roughness": 8.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "joint_report.html"
            write_joint_standard_report(rows, report, metadata={"standard": "project-template"})
            text = report.read_text(encoding="utf-8")
        self.assertIn("Joint element standard report", text)
        self.assertIn("最大すべり量", text)
        self.assertIn("effective_roughness", text)

    def test_axisymmetric_linear_interface_reaction_uses_ring_measure(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [1.0, 1.0], "3": [1.0, 0.0], "4": [1.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [{"id": "joint", "minus_nodes": ["1", "2"], "plus_nodes": ["3", "4"], "kn": 1000.0, "kt": 500.0}],
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "ux": 0.01, "uy": 0.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "interface_state.csv").exists())
        stage = result.stages[0]
        plus_reaction_x = stage.reactions[4] + stage.reactions[6]
        self.assertEqual(len(result.interfaces), 1)
        self.assertAlmostEqual(abs(plus_reaction_x), 20.0 * math.pi)
        self.assertTrue(all(row["geometry"] == "axisymmetric" for row in stage.interface_results))

    def test_axisymmetric_friction_interface_limits_shear_reaction_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [1.0, 1.0], "3": [1.0, 0.0], "4": [1.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 1000.0,
                    "behavior": {"friction": 0.5, "cohesion": 0.0},
                }
            ],
            "boundary_conditions": [
                {"nodes": ["1", "2"], "ux": 0.0, "uy": 0.0},
                {"nodes": ["3", "4"], "ux": -0.01, "uy": 0.02},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        plus_reaction_y = stage.reactions[5] + stage.reactions[7]
        self.assertEqual(stage.solver_info["method"], "axisymmetric_newton")
        self.assertAlmostEqual(abs(plus_reaction_y), 10.0 * math.pi, places=10)
        self.assertTrue(any(row["state"] == "slip" for row in stage.interface_results))
        self.assertGreater(max(row["slip_abs"] for row in stage.interface_results), 0.0)

    def test_von_mises_plastic_stress_update_runs_in_2d_core(self) -> None:
        cfg = plane_strain_patch_sample("QUAD4", "FULL")
        cfg["materials"]["soil"].update({"model": "von_mises", "yield_stress": 30.0})
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        row = result.stages[0].element_results[0]
        self.assertEqual(row["plastic"], 1.0)
        self.assertLessEqual(row["q"], 30.0 + 1.0e-10)
        self.assertGreater(max(state.kappa for state in result.stages[0].plastic_state.values()), 0.0)

    def test_mohr_coulomb_elastic_trial_uses_analytic_principal_fast_path(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "mc",
            E=50000.0,
            nu=0.30,
            model="mohr_coulomb",
            cohesion=500.0,
            friction_angle=30.0,
            dilation_angle=10.0,
        )
        strain = np.array([1.0e-5, -2.0e-6, 0.0, 1.0e-6])
        update = update_plane_strain_stress(material, strain)
        self.assertFalse(update.plastic)
        self.assertEqual(update.active_set, ())
        self.assertTrue(np.allclose(update.stress, material.D4 @ strain, rtol=1.0e-12, atol=1.0e-12))
        tangent = algorithmic_material_tangent(material, strain)
        self.assertTrue(np.allclose(tangent, material.D4, rtol=1.0e-12, atol=1.0e-12))

    def test_mohr_coulomb_return_mapping_principal_uses_numba_candidate_scan(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "mc",
            E=50000.0,
            nu=0.30,
            model="mohr_coulomb",
            cohesion=5.0,
            friction_angle=30.0,
            dilation_angle=10.0,
        )
        strain = np.array([0.002, -0.001, 0.0, 0.001])
        trial = material.D4 @ strain
        tensor = np.array([[trial[0], trial[3], 0.0], [trial[3], trial[1], 0.0], [0.0, 0.0, trial[2]]])
        sig_tr_p = np.linalg.eigvalsh(tensor)
        c, phi, psi = _mc_reduced_parameters(material, 1.0)
        yield_coeffs = _mc_plane_coeffs(phi)
        flow_coeffs = _mc_plane_coeffs(psi)
        cohesion_term = 2.0 * c * math.cos(phi)
        tol = _mc_yield_tol(sig_tr_p, cohesion_term)

        ok, sig_fast, ids_fast, active_count, gamma_fast, vals_fast = _mc_return_mapping_principal_numba(
            np.ascontiguousarray(sig_tr_p, dtype=np.float64),
            np.ascontiguousarray(yield_coeffs, dtype=np.float64),
            np.ascontiguousarray(flow_coeffs, dtype=np.float64),
            float(cohesion_term),
            np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
            float(material.hardening),
            0.0,
            float(tol),
        )
        self.assertTrue(ok)
        sig_ref, ids_ref, gamma_ref, vals_ref = _mc_return_mapping_principal_python(
            sig_tr_p,
            yield_coeffs=yield_coeffs,
            flow_coeffs=flow_coeffs,
            cohesion_term=cohesion_term,
            Cn=material.D4[:3, :3],
            hardening=material.hardening,
            kappa=0.0,
            tol=tol,
        )
        count = int(active_count)
        self.assertGreater(count, 0)
        self.assertEqual(len(ids_ref), len(gamma_ref))
        self.assertTrue(np.allclose(sig_fast, sig_ref, rtol=1.0e-12, atol=1.0e-9))
        self.assertGreaterEqual(float(np.min(gamma_fast[:count])), -1.0e-12)
        self.assertLessEqual(float(np.max(vals_fast)), 100.0 * tol)
        self.assertLessEqual(float(np.max(vals_ref)), 100.0 * tol)

    def test_exact_mohr_coulomb_active_set_return_mapping_runs_in_2d_core(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "mc",
            E=50000.0,
            nu=0.30,
            model="mohr_coulomb",
            cohesion=5.0,
            friction_angle=30.0,
            dilation_angle=10.0,
        )
        strain = np.array([0.002, -0.001, 0.0, 0.001])
        update = update_plane_strain_stress(material, strain)
        self.assertTrue(update.plastic)
        self.assertTrue(update.active_set)
        tensor = np.array([[update.stress[0], update.stress[3], 0.0], [update.stress[3], update.stress[1], 0.0], [0.0, 0.0, update.stress[2]]])
        principals = np.linalg.eigvalsh(tensor)
        phi = math.radians(30.0)
        coeffs = []
        for i, j in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            row = np.zeros(3)
            row[i] += 1.0 + math.sin(phi)
            row[j] -= 1.0 - math.sin(phi)
            coeffs.append(row)
        fmax = float(np.max(np.vstack(coeffs) @ principals - 2.0 * 5.0 * math.cos(phi)))
        self.assertLessEqual(fmax, 1.0e-8)
        fallback_route = algorithmic_material_tangent(material, strain, method="numerical")
        self.assertTrue(np.all(np.isfinite(fallback_route)))

    def test_mohr_coulomb_consistent_tangent_uses_numba_spectral_kernel(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "mc",
            E=50000.0,
            nu=0.30,
            model="mohr_coulomb",
            cohesion=5.0,
            friction_angle=10.0,
            dilation_angle=10.0,
        )
        strain = np.array([0.0005, -0.0002, 0.0, 0.001])
        trial = material.D4 @ strain
        sig_tr_p, vecs = np.linalg.eigh(np.array([[trial[0], trial[3], 0.0], [trial[3], trial[1], 0.0], [0.0, 0.0, trial[2]]]))
        c, phi, psi = _mc_reduced_parameters(material, 1.0)
        yield_coeffs = _mc_plane_coeffs(phi)
        flow_coeffs = _mc_plane_coeffs(psi)
        cohesion_term = 2.0 * c * math.cos(phi)
        tol = _mc_yield_tol(sig_tr_p, cohesion_term)
        ok, sig_corr_p, active_ids, active_count, _gamma, _vals_corr = _mc_return_mapping_principal_numba(
            np.ascontiguousarray(sig_tr_p, dtype=np.float64),
            np.ascontiguousarray(yield_coeffs, dtype=np.float64),
            np.ascontiguousarray(flow_coeffs, dtype=np.float64),
            float(cohesion_term),
            np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
            float(material.hardening),
            0.0,
            float(tol),
        )
        self.assertTrue(ok)
        tangent_ok, tangent = _mc_consistent_tangent_spectral_numba(
            np.ascontiguousarray(sig_tr_p, dtype=np.float64),
            np.ascontiguousarray(sig_corr_p, dtype=np.float64),
            np.ascontiguousarray(vecs, dtype=np.float64),
            active_ids,
            int(active_count),
            np.ascontiguousarray(yield_coeffs, dtype=np.float64),
            np.ascontiguousarray(flow_coeffs, dtype=np.float64),
            np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
            np.ascontiguousarray(material.D4, dtype=np.float64),
            float(material.hardening),
        )
        self.assertTrue(tangent_ok)
        numerical = algorithmic_material_tangent(material, strain, method="numerical")
        self.assertTrue(np.allclose(tangent, numerical, rtol=2.0e-5, atol=2.0))

    def test_exact_mohr_coulomb_consistent_tangent_matches_numerical_backup(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "mc",
            E=50000.0,
            nu=0.30,
            model="mohr_coulomb",
            cohesion=5.0,
            friction_angle=10.0,
            dilation_angle=10.0,
        )
        strain = np.array([0.0005, -0.0002, 0.0, 0.001])
        update = update_plane_strain_stress(material, strain)
        self.assertEqual(len(update.active_set), 1)
        analytic = algorithmic_material_tangent(material, strain, method="analytic")
        numerical = algorithmic_material_tangent(material, strain, method="numerical")
        self.assertTrue(np.allclose(analytic, numerical, rtol=2.0e-5, atol=2.0))
        self.assertTrue(np.allclose(analytic, analytic.T, rtol=1.0e-10, atol=1.0e-6))

    def test_tension_cutoff_caps_principal_stress_with_numerical_tangent_backup(self) -> None:
        material = plane_strain_materials({"materials": {"cutoff": {"E": 50000.0, "nu": 0.30, "tension_cutoff": 10.0}}})["cutoff"]
        self.assertEqual(material.tensile_strength, 10.0)
        strain = np.array([0.001, 0.0, 0.0, 0.0])
        update = update_plane_strain_stress(material, strain)
        self.assertTrue(update.plastic)
        self.assertLessEqual(principal_stresses(update.stress)[0], 10.0 + 1.0e-10)
        analytic_route = algorithmic_material_tangent(material, strain, method="analytic")
        numerical_route = algorithmic_material_tangent(material, strain, method="numerical")
        self.assertTrue(np.allclose(analytic_route, numerical_route, rtol=1.0e-12, atol=1.0e-12))

    def test_j2dp_tension_cutoff_numerical_tangent_uses_batched_kernel(self) -> None:
        cases = [
            ElasticPlaneStrainMaterial(
                "j2-cutoff",
                E=50000.0,
                nu=0.30,
                model="von_mises",
                yield_stress=20.0,
                hardening=40.0,
                tension_cutoff=True,
                tensile_strength=12.0,
            ),
            ElasticPlaneStrainMaterial(
                "dp-cutoff",
                E=50000.0,
                nu=0.30,
                model="drucker_prager",
                cohesion=20.0,
                friction_angle=20.0,
                hardening=40.0,
                tension_cutoff=True,
                tensile_strength=12.0,
            ),
        ]
        strain = np.array([0.0012, 0.0002, 0.0, 0.0008], dtype=float)
        state = PlasticState2D(np.array([1.0e-4, -5.0e-5, 2.0e-5, 7.0e-5], dtype=float), 0.015)
        for material in cases:
            with self.subTest(model=material.model):
                alpha, cohesion_term = _yield_surface_parameters(material, 1.0)
                fast = _j2dp_tension_cutoff_numerical_tangent_numba(
                    np.ascontiguousarray(strain, dtype=np.float64),
                    np.ascontiguousarray(state.plastic_strain, dtype=np.float64),
                    np.ascontiguousarray(material.D4, dtype=np.float64),
                    np.zeros(4, dtype=np.float64),
                    float(alpha),
                    float(cohesion_term),
                    float(material.hardening),
                    float(material.shear_mu),
                    float(state.kappa),
                    float(material.tensile_strength),
                )
                base = update_plane_strain_stress(material, strain, state=state).stress
                self.assertLessEqual(principal_stresses(base)[0], material.tensile_strength + 1.0e-10)
                delta = 1.0e-8 * max(1.0, float(np.linalg.norm(strain)))
                expected = np.zeros((4, 4), dtype=float)
                for i in range(4):
                    perturbed = strain.copy()
                    perturbed[i] += delta
                    expected[:, i] = (update_plane_strain_stress(material, perturbed, state=state).stress - base) / delta
                self.assertTrue(np.allclose(fast, expected, rtol=1.0e-10, atol=1.0e-6))
                self.assertTrue(np.allclose(numerical_material_tangent(material, strain, state=state), expected, rtol=1.0e-10, atol=1.0e-6))
                self.assertTrue(np.allclose(algorithmic_material_tangent(material, strain, state=state), expected, rtol=1.0e-10, atol=1.0e-6))

    def test_mohr_coulomb_tension_cutoff_numerical_tangent_uses_batched_kernel(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "mc-cutoff",
            E=50000.0,
            nu=0.30,
            model="mohr_coulomb",
            cohesion=5.0,
            friction_angle=10.0,
            dilation_angle=10.0,
            hardening=20.0,
            tension_cutoff=True,
            tensile_strength=12.0,
        )
        strain = np.array([0.0005, -0.0002, 0.0, 0.001], dtype=float)
        state = PlasticState2D(np.array([1.0e-5, -2.0e-5, 0.0, 1.0e-5], dtype=float), 0.01)
        c, phi, psi = _mc_reduced_parameters(material, 1.0)
        ok, fast = _mc_tension_cutoff_numerical_tangent_numba(
            np.ascontiguousarray(strain, dtype=np.float64),
            np.ascontiguousarray(state.plastic_strain, dtype=np.float64),
            np.ascontiguousarray(material.D4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            float(math.sin(phi)),
            float(2.0 * c * math.cos(phi)),
            float(material.hardening),
            float(state.kappa),
            np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
            np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
            np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
            float(material.tensile_strength),
        )
        self.assertTrue(ok)
        ok_consistent, consistent = _mc_tension_cutoff_consistent_tangent_numba(
            np.ascontiguousarray(strain, dtype=np.float64),
            np.ascontiguousarray(state.plastic_strain, dtype=np.float64),
            np.ascontiguousarray(material.D4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            float(math.sin(phi)),
            float(2.0 * c * math.cos(phi)),
            float(material.hardening),
            float(state.kappa),
            np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
            np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
            np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
            float(material.tensile_strength),
        )
        self.assertTrue(ok_consistent)
        base = update_plane_strain_stress(material, strain, state=state).stress
        self.assertLessEqual(principal_stresses(base)[0], material.tensile_strength + 1.0e-10)
        delta = 1.0e-8 * max(1.0, float(np.linalg.norm(strain)))
        expected = np.zeros((4, 4), dtype=float)
        for i in range(4):
            perturbed = strain.copy()
            perturbed[i] += delta
            expected[:, i] = (update_plane_strain_stress(material, perturbed, state=state).stress - base) / delta
        self.assertTrue(np.allclose(fast, expected, rtol=1.0e-8, atol=1.0e-4))
        self.assertTrue(np.allclose(consistent, expected, rtol=2.0e-5, atol=2.0))
        self.assertTrue(np.allclose(numerical_material_tangent(material, strain, state=state), expected, rtol=1.0e-8, atol=1.0e-4))
        self.assertTrue(np.allclose(algorithmic_material_tangent(material, strain, state=state), consistent, rtol=1.0e-10, atol=1.0e-7))

    def test_j2_analytic_tangent_matches_numerical_backup(self) -> None:
        material = ElasticPlaneStrainMaterial("j2", E=50000.0, nu=0.30, model="von_mises", yield_stress=30.0, hardening=50.0)
        strain = np.array([0.002, -0.0006, 0.0, 0.0012])
        analytic = algorithmic_material_tangent(material, strain, method="analytic")
        numerical = numerical_material_tangent(material, strain)
        self.assertTrue(np.allclose(analytic, numerical, rtol=2.0e-6, atol=5.0e-2))
        self.assertTrue(np.allclose(analytic, analytic.T, rtol=1.0e-12, atol=1.0e-8))

    def test_dp_analytic_tangent_matches_numerical_backup(self) -> None:
        material = ElasticPlaneStrainMaterial(
            "dp",
            E=50000.0,
            nu=0.30,
            model="drucker_prager",
            cohesion=50.0,
            friction_angle=20.0,
            hardening=50.0,
        )
        strain = np.array([0.002, -0.0006, 0.0, 0.0012])
        analytic = algorithmic_material_tangent(material, strain, method="analytic")
        numerical = algorithmic_material_tangent(material, strain, method="numerical")
        self.assertTrue(np.allclose(analytic, numerical, rtol=2.0e-6, atol=5.0e-2))
        self.assertGreater(np.linalg.norm(analytic - analytic.T), 1.0)

    def test_plastic_history_persists_across_2d_stages(self) -> None:
        cfg = plane_strain_patch_sample("QUAD4", "FULL")
        cfg["materials"]["soil"].update({"model": "von_mises", "yield_stress": 30.0, "hardening": 10.0})
        cfg["steps"] = [{"name": "load-1", "type": "static"}, {"name": "hold", "type": "static"}]
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        kappa_1 = max(state.kappa for state in result.stages[0].plastic_state.values())
        kappa_2 = max(state.kappa for state in result.stages[1].plastic_state.values())
        self.assertGreater(kappa_1, 0.0)
        self.assertAlmostEqual(kappa_2, kappa_1)

    def test_newton_line_search_control_is_reported_in_2d_core(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg["materials"]["soil"].update({"model": "von_mises", "yield_stress": 2.0, "hardening": 10.0})
        cfg["solver"] = {"newton": {"line_search": True, "max_line_search": 4}}
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        self.assertTrue(result.stages[0].solver_info["converged"])
        self.assertIn("line_search_reductions", result.stages[0].solver_info)
        line_search_batch = result.stages[0].solver_info["line_search_batch"]
        self.assertTrue(line_search_batch["requested"])
        self.assertFalse(line_search_batch["enabled"])
        self.assertEqual(
            line_search_batch["disabled_reason"],
            "mixed_or_unsupported_elements",
        )

    def test_advanced_material_catalog_models_connect_to_2d_core(self) -> None:
        material_specs = {
            "hardin": {"model": "elastic", "gui_model": "hardin_drnevich", "G0": 25000.0, "gamma_ref": 0.001},
            "ramberg": {"model": "ramberg_osgood", "G0": 25000.0, "gamma_ref": 0.001, "alpha": 1.0, "r": 2.2},
            "uw": {"model": "uw_clay", "G0": 18000.0, "gamma_ref": 0.002, "su": 20.0},
            "pz_sand": {"model": "pastor_zienkiewicz_sand", "G0": 30000.0, "gamma_ref": 0.001, "phi_cs": 34.0},
            "pz_clay": {"model": "pastor_zienkiewicz_clay", "G0": 18000.0, "gamma_ref": 0.002, "su": 15.0, "phi_cs": 24.0},
            "liq": {"model": "bilinear_liquefaction", "G0": 25000.0, "gamma_ref": 0.001, "friction_angle": 32.0, "liquefaction": {"ru": 0.4, "post_liquefaction_stiffness_ratio": 0.05}},
        }
        strain = np.array([0.001, -0.0002, 0.0, 0.001], dtype=float)
        for name, spec in material_specs.items():
            with self.subTest(name=name):
                cfg = {"materials": {"soil": {"E": 50000.0, "nu": 0.30, "cohesion": 10.0, **spec}}}
                material = plane_strain_materials(cfg)["soil"]
                update = update_plane_strain_stress(material, strain)
                tangent = algorithmic_material_tangent(material, strain)
                self.assertEqual(material.advanced_model, material.model)
                self.assertTrue(np.all(np.isfinite(update.stress)))
                self.assertTrue(np.all(np.isfinite(tangent)))
                self.assertIn("modulus_ratio", update.state_vars)
                self.assertIn("dilatancy", update.state_vars)
                if "liq" in name:
                    self.assertIn("ru", update.state_vars)

    def test_hardin_drnevich_stiffness_reduces_with_strain(self) -> None:
        material = plane_strain_materials({"materials": {"soil": {"model": "hardin_drnevich", "E": 50000.0, "nu": 0.30, "G0": 25000.0, "gamma_ref": 0.001}}})["soil"]
        small = _advanced_effective_material(material, np.array([1.0e-6, 0.0, 0.0, 1.0e-6]))
        large = _advanced_effective_material(material, np.array([1.0e-3, 0.0, 0.0, 1.0e-3]))
        self.assertGreater(small.E, large.E)

    def test_advanced_material_history_uses_fixed_state_array_kernel(self) -> None:
        material = plane_strain_materials(
            {
                "materials": {
                    "sand": {
                        "model": "bilinear_liquefaction",
                        "E": 50000.0,
                        "nu": 0.30,
                        "G0": 25000.0,
                        "gamma_ref": 0.001,
                        "friction_angle": 32.0,
                        "liquefaction": {
                            "cyclic_resistance_ratio": 0.2,
                            "cyclic_stress_ratio": 0.18,
                            "generation_rate": 0.2,
                            "dissipation_rate": 0.1,
                            "cycles_per_step": 0.5,
                        },
                    }
                }
            }
        )["sand"]
        strain = np.array([0.001, -0.0002, 0.0, 0.001], dtype=float)
        state = PlasticState2D(state_vars={"gamma_eq": 1.0e-4, "cyclic_strain": 0.01, "cycles": 2.0, "ru": 0.2})
        source_model = material.advanced_model or material.model

        history_array = _advanced_history_array(material, strain, state, source_model)
        history = _advanced_history_state(material, strain, state, source_model)
        update = update_plane_strain_stress(material, strain, state=state)

        self.assertAlmostEqual(float(history["gamma_eq"]), float(history_array[_ADV_STATE_GAMMA_EQ]), places=15)
        self.assertAlmostEqual(float(history["ru"]), float(history_array[_ADV_STATE_RU]), places=15)
        self.assertAlmostEqual(float(history["modulus_ratio"]), float(history_array[_ADV_STATE_MODULUS_RATIO]), places=15)
        self.assertAlmostEqual(float(history["effective_E"]), float(history_array[_ADV_STATE_EFFECTIVE_E]), places=10)
        self.assertGreater(float(history["ru_generation_increment"]), 0.0)
        self.assertGreaterEqual(update.state_vars["ru"], 0.2)
        self.assertAlmostEqual(update.state_vars["effective_E"], float(history_array[_ADV_STATE_EFFECTIVE_E]), places=10)

    def test_liquefaction_strength_uses_history_ru(self) -> None:
        material = plane_strain_materials(
            {
                "materials": {
                    "sand": {
                        "model": "bilinear_liquefaction",
                        "E": 50000.0,
                        "nu": 0.30,
                        "G0": 25000.0,
                        "gamma_ref": 0.01,
                        "yield_stress": 100.0,
                        "liquefaction": {"post_liquefaction_strength_ratio": 0.05},
                    }
                }
            }
        )["sand"]
        strain = np.array([0.0, 0.0, 0.0, 0.02], dtype=float)
        low_ru = update_plane_strain_stress(material, strain, state=PlasticState2D(state_vars={"ru": 0.0}))
        high_ru = update_plane_strain_stress(material, strain, state=PlasticState2D(state_vars={"ru": 0.9}))
        self.assertLess(high_ru.q, low_ru.q)
        self.assertGreaterEqual(high_ru.state_vars["ru"], 0.9)

    def test_advanced_material_solves_and_writes_ip_history(self) -> None:
        cfg = plane_strain_patch_sample("QUAD4", "FULL")
        cfg["materials"]["soil"].update({"model": "hardin_drnevich", "G0": 25000.0, "gamma_ref": 0.001})
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
            self.assertTrue((stage.output_dir / "integration_point_stress.csv").exists())
            with (stage.output_dir / "integration_point_stress.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(stage.solver_info["method"], "newton")
        self.assertTrue(np.all(np.isfinite(stage.displacements)))
        self.assertIn("modulus_ratio", rows[0])
        self.assertLess(float(rows[0]["modulus_ratio"]), 1.0)

    def test_srm_strength_reduction_runs_in_2d_core(self) -> None:
        cfg = plane_strain_patch_sample("QUAD4", "FULL")
        cfg["materials"]["soil"].update({"model": "von_mises", "yield_stress": 80.0})
        cfg["steps"] = [{"name": "srm", "type": "srm", "srm": {"factors": [1.0, 2.0], "failure_plastic_ratio": 0.0}}]
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "integration_point_stress.csv").exists())
            run_log = (result.output_dir / "run.log").read_text(encoding="utf-8")
            self.assertIn("FOS=1", run_log)
            self.assertIn("[srm-trial] srm #1", run_log)
            analysis_log = json.loads((result.output_dir / "analysis_log.json").read_text(encoding="utf-8"))
            self.assertTrue(any(row.get("event_type") == "srm_summary" and row.get("factor_of_safety") == 1.0 for row in analysis_log["events"]))
            self.assertTrue(any(row.get("event_type") == "srm_trial" and row.get("srm_factor") == 2.0 for row in analysis_log["events"]))
            analysis_csv = (result.output_dir / "analysis_log.csv").read_text(encoding="utf-8")
            self.assertIn("factor_of_safety", analysis_csv)
            self.assertIn("srm_factor", analysis_csv)
            self.assertIn("trial_status", analysis_csv)
            self.assertIn("last_accepted_plastic_ratio", analysis_csv)
            report_html = (result.output_dir / "calculation_report.html").read_text(encoding="utf-8")
            self.assertIn("SRM FOS Trial Results", report_html)
            self.assertIn("SRM Trial Results", report_html)
            standard_html = (result.output_dir / "standard_report.html").read_text(encoding="utf-8")
            self.assertIn("SRM Trial Results", standard_html)
            self.assertIn("<th>FOS</th>", standard_html)
        srm = result.stages[0].solver_info["srm"]
        self.assertEqual(srm["factor_of_safety"], 1.0)
        self.assertEqual(len(srm["trials"]), 2)
        self.assertTrue(srm["lightweight_postprocess"]["enabled"])
        self.assertTrue(srm["lightweight_postprocess"]["nonlinear_reanalysis_avoided"])
        self.assertIsNone(srm["lightweight_postprocess"]["final_factor_reprocessed"])
        self.assertTrue(result.stages[0].solver_info["postprocess_results"])
        self.assertEqual(result.stages[0].solver_info["postprocess_state_commit"]["state_commit"], "array_batch")
        retained_postprocess = result.stages[0].solver_info["srm_retained_trial_postprocess"]
        self.assertTrue(retained_postprocess["enabled"])
        self.assertEqual(retained_postprocess["state_commit_authority"], "retained_solver_trial")
        self.assertEqual(retained_postprocess["output_state_recomputed_from"], "stage_initial_state")
        self.assertFalse(retained_postprocess["requires_full_reanalysis"])
        self.assertFalse(retained_postprocess["state_drift_exceeds_tolerance"])
        self.assertGreater(result.stages[0].solver_info["postprocess_state_commit"]["array_committed_elements"], 0)
        self.assertTrue(all(row["postprocess_results"] is False for row in srm["trials"]))
        self.assertTrue(all(row["plastic_ratio_source"] == "plastic_state_array_cache" for row in srm["trials"]))
        self.assertGreater(len(result.stages[0].element_results), 0)
        self.assertGreater(len(result.stages[0].integration_point_results), 0)
        self.assertTrue(srm["factor_cache"]["enabled"])
        self.assertTrue(srm["factor_cache"]["shared_across_trials"])
        self.assertEqual(srm["factor_cache"]["cache_kind"], "small_deformation_step_cache")
        self.assertTrue(srm["factor_cache"]["stiffness_pattern_cached"])
        self.assertTrue(srm["factor_cache"]["reduced_matrix_cached"])
        self.assertTrue(all(row["reduced_matrix_cache_enabled"] for row in srm["trials"]))
        strength_cache = srm["factor_cache"]["material_strength_parameter_cache"]
        self.assertTrue(strength_cache["enabled"])
        self.assertEqual(strength_cache["scope"], "material_strength_factor")
        self.assertIn("scalar_parameters", strength_cache)
        self.assertIn("batch_arrays", strength_cache)
        load_mpc_cache = srm["factor_cache"]["factor_invariant_load_mpc_cache"]
        self.assertTrue(load_mpc_cache["enabled"])
        self.assertTrue(load_mpc_cache["load_vector_cached"])
        self.assertTrue(load_mpc_cache["mpc_penalty_cached"])
        self.assertTrue(load_mpc_cache["constraint_scale_template_cached"])
        self.assertTrue(result.stages[0].solver_info["load_mpc_factor_cache"]["load_vector_reused"])
        self.assertTrue(result.stages[0].solver_info["load_mpc_factor_cache"]["mpc_penalty_reused"])
        self.assertTrue(result.stages[0].solver_info["plastic_state_array_cache"]["enabled"])
        self.assertEqual(result.stages[0].solver_info["plastic_state_array_cache"]["layout"], "active_element_major_integration_point_minor")
        self.assertEqual(result.stages[0].solver_info["topology_cache"]["cache_kind"], "small_deformation_step_cache")
        self.assertTrue(result.stages[0].solver_info["topology_cache"]["stiffness_pattern_cached"])
        self.assertTrue(result.stages[0].solver_info["topology_cache"]["reduced_matrix_cached"])

    def test_srm_plastic_state_drift_reanalyzes_only_non_numeric_state(self) -> None:
        cfg = plane_strain_patch_sample("QUAD4", "FULL")
        cfg["materials"]["soil"].update({"model": "von_mises", "yield_stress": 10.0})
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        element_id = str(mesh.elements[0].id)
        first = build_plastic_state_array_cache(
            mesh,
            materials,
            {
                f"{element_id}:0": PlasticState2D(
                    np.array([0.01, 0.0, 0.0, 0.0], dtype=float), 0.01
                )
            },
        )
        changed = build_plastic_state_array_cache(
            mesh,
            materials,
            {
                f"{element_id}:0": PlasticState2D(
                    np.array([0.02, 0.0, 0.0, 0.0], dtype=float), 0.02
                )
            },
        )
        numeric_drift = fem2d_solver_module._srm_plastic_state_cache_drift(first, changed)
        self.assertTrue(numeric_drift["exceeds_tolerance"])
        self.assertFalse(numeric_drift["requires_full_reanalysis"])

        advanced = build_plastic_state_array_cache(
            mesh,
            materials,
            {
                f"{element_id}:0": PlasticState2D(
                    np.array([0.01, 0.0, 0.0, 0.0], dtype=float),
                    0.01,
                    {"ru": 0.1},
                )
            },
        )
        advanced_drift = fem2d_solver_module._srm_plastic_state_cache_drift(advanced, advanced)
        self.assertTrue(advanced_drift["state_vars_require_reanalysis"])
        self.assertTrue(advanced_drift["requires_full_reanalysis"])
        self.assertEqual(advanced_drift["reason"], "advanced_state_vars_require_full_reanalysis")

    def test_srm_factor_cache_reuses_load_vector_and_mpc_penalty(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "von_mises", "E": 10000.0, "nu": 0.30, "yield_stress": 1.0e6}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "loads": [{"node": "3", "fy": -10.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "penalty": 1.0e8}],
            "steps": [{"name": "srm-load-mpc-cache", "type": "srm", "srm": {"factors": [1.0, 1.1], "failure_plastic_ratio": 1.0}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        factor_cache = stage.solver_info["srm"]["factor_cache"]["factor_invariant_load_mpc_cache"]
        trial_cache = stage.solver_info["load_mpc_factor_cache"]
        self.assertTrue(factor_cache["load_vector_cached"])
        self.assertTrue(factor_cache["mpc_penalty_cached"])
        self.assertEqual(factor_cache["mpc_equations"], 1)
        self.assertTrue(factor_cache["constraint_scale_template_cached"])
        self.assertTrue(trial_cache["load_vector_reused"])
        self.assertTrue(trial_cache["mpc_penalty_reused"])
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "penalty")
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=6)

    def test_srm_incremental_trials_share_small_deformation_step_cache(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        cfg["materials"]["soil"].update({"model": "von_mises", "yield_stress": 1.0e6})
        cfg["steps"] = [
            {
                "name": "srm-inc-cache",
                "type": "srm",
                "solver": {"increments": {"enabled": True, "steps": 2}, "newton": {"max_iter": 20, "line_search": True}},
                "srm": {"factors": [1.0, 1.1], "failure_plastic_ratio": 1.0},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]

        srm = stage.solver_info["srm"]
        self.assertEqual(len(srm["trials"]), 2)
        self.assertTrue(srm["factor_cache"]["enabled"])
        self.assertTrue(srm["factor_cache"]["shared_across_trials"])
        self.assertEqual(srm["factor_cache"]["cache_kind"], "small_deformation_step_cache")
        self.assertTrue(srm["factor_cache"]["stiffness_pattern_cached"])
        self.assertTrue(srm["factor_cache"]["reduced_matrix_cached"])
        self.assertEqual(srm["factor_cache"]["active_elements"], len(stage.active_elements))
        self.assertTrue(srm["lightweight_postprocess"]["enabled"])
        self.assertTrue(stage.solver_info["postprocess_results"])
        self.assertTrue(all(row["postprocess_results"] is False for row in srm["trials"]))
        self.assertTrue(all(row["plastic_ratio_source"] == "plastic_state_array_cache" for row in srm["trials"]))
        self.assertTrue(stage.solver_info["increments"]["step_cache_shared"])
        self.assertEqual(stage.solver_info["increments"]["step_cache_kind"], "small_deformation_step_cache")
        self.assertTrue(stage.solver_info["increments"]["step_cache"]["stiffness_pattern_cached"])
        self.assertTrue(stage.solver_info["increments"]["step_cache"]["reduced_matrix_cached"])
        self.assertTrue(all(row["step_cache_used"] for row in stage.solver_info["increments"]["log"]))
        self.assertTrue(stage.solver_info["sparse_pattern_cached"])
        self.assertTrue(stage.solver_info["constraint_dofs_cached"])

    def test_srm_two_branch_search_starts_from_anchor_without_trial_continuation(self) -> None:
        evaluated: list[float] = []

        def fake_trial(factor: float) -> StageResult2D:
            evaluated.append(round(float(factor), 3))
            plastic = 0.0 if factor <= 0.8 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            {
                "search_mode": "two_branch",
                "anchor_factor": 1.0,
                "lower_branch": {"factor_min": 0.5, "factor_step": 0.1},
                "upper_branch": {"factor_max": 1.1, "factor_step": 0.1},
                "factor_tol": 0.05,
                "max_bisection": 4,
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(evaluated[:3], [1.0, 0.9, 0.8])
        self.assertLess(fos, 1.0)
        self.assertGreaterEqual(fos, 0.8)
        self.assertTrue(info["bracketed"])
        self.assertEqual(info["trial_state"], "independent_from_stage_start")
        self.assertEqual(info["final_display_state"], "last_stable")
        self.assertLess(info["stable_factor"], 1.0)
        self.assertIn("plastic_divergence", info["failure_reasons"])
        self.assertEqual([row["factor"] for row in trials[:3]], [1.0, 0.9, 0.8])

    def test_srm_explicit_factors_parallel_respects_worker_limit_and_order(self) -> None:
        active = 0
        max_active = 0
        evaluated: list[float] = []
        lock = threading.Lock()

        def fake_trial(factor: float) -> StageResult2D:
            nonlocal active, max_active
            factor = round(float(factor), 3)
            with lock:
                evaluated.append(factor)
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.02)
                plastic = 0.0 if factor <= 1.1 else 1.0
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": plastic}],
                    {},
                    [],
                    {"converged": True},
                )
            finally:
                with lock:
                    active -= 1

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.1, 1.2, 1.3],
            {"factors": [1.0, 1.1, 1.2, 1.3], "parallel": {"enabled": True, "max_workers": 2}},
            0.0,
            fake_trial,
        )

        self.assertEqual(fos, 1.1)
        self.assertEqual(info["search_mode"], "explicit_factors")
        self.assertEqual([row["factor"] for row in trials], [1.0, 1.1, 1.2])
        self.assertEqual(info["parallel"]["max_workers"], 2)
        self.assertEqual(info["parallel"]["evaluated_trials"], 4)
        self.assertEqual(info["parallel"]["reported_trials"], 3)
        self.assertCountEqual(evaluated, [1.0, 1.1, 1.2, 1.3])
        self.assertGreater(max_active, 1)
        self.assertLessEqual(max_active, 2)

    def test_srm_parallel_evaluation_uses_numeric_thread_context(self) -> None:
        seen_controls: list[dict[str, Any]] = []

        def fake_thread_context(control: Mapping[str, Any] | None):
            seen_controls.append(dict(control or {}))
            return nullcontext()

        def fake_trial(factor: float) -> StageResult2D:
            plastic = 0.0 if float(factor) <= 1.0 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        with (
            patch("geofem_app.fem2d_solver.os.cpu_count", return_value=4),
            patch("geofem_app.fem2d_solver._srm_physical_cpu_count", return_value=2),
            patch("geofem_app.fem2d_solver._srm_numeric_thread_context", side_effect=fake_thread_context),
        ):
            _result, _fos, _trials, info = _run_srm_trial_search(
                [1.0, 1.1],
                {"factors": [1.0, 1.1], "parallel": {"enabled": True, "max_workers": 2, "threads_per_worker": 1}},
                0.0,
                fake_trial,
            )

        self.assertEqual(len(seen_controls), 1)
        self.assertEqual(seen_controls[0]["threads_per_worker"], 1)
        self.assertEqual(seen_controls[0]["numeric_thread_env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(info["parallel"]["selected_threads_per_worker"], 1)

    def test_srm_parallel_thread_safety_guard_forces_serial_execution(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_trial(factor: float) -> StageResult2D:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.01)
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    [],
                    {"converged": True},
                )
            finally:
                with lock:
                    active -= 1

        _result, _fos, trials, info = _run_srm_trial_search(
            [1.0, 1.1],
            {"factors": [1.0, 1.1], "parallel": {"enabled": True, "max_workers": 2}},
            1.0,
            fake_trial,
            parallel_trials_supported=False,
            parallel_disabled_reason="shared_mutable_test_state",
        )

        self.assertEqual(len(trials), 2)
        self.assertEqual(max_active, 1)
        self.assertFalse(info["parallel"]["enabled"])
        self.assertTrue(info["parallel"]["parallel_requested"])
        self.assertEqual(info["parallel"]["disabled_reason"], "shared_mutable_test_state")

    def test_srm_numeric_thread_context_falls_back_to_scoped_environment(self) -> None:
        control = {
            "threads_per_worker": 2,
            "numeric_thread_env": {
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMBA_NUM_THREADS": "2",
            },
        }
        real_import = __import__

        def fake_import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple[Any, ...] = (), level: int = 0) -> Any:
            if name == "threadpoolctl":
                raise ImportError("threadpoolctl intentionally unavailable")
            return real_import(name, globals, locals, fromlist, level)

        baseline = {
            "OMP_NUM_THREADS": "99",
            "MKL_NUM_THREADS": "99",
            "OPENBLAS_NUM_THREADS": "99",
            "NUMBA_NUM_THREADS": "99",
        }
        with patch.dict(os.environ, baseline, clear=False):
            with patch("builtins.__import__", side_effect=fake_import):
                with _srm_numeric_thread_context(control):
                    self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
                    self.assertEqual(os.environ["MKL_NUM_THREADS"], "2")
                    self.assertTrue(control["applied"])
                    self.assertEqual(control["apply_method"], "environment")
                    self.assertFalse(control["threadpoolctl_available"])
                    self.assertFalse(control["environment_restored"])
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "99")
                self.assertEqual(os.environ["MKL_NUM_THREADS"], "99")
                self.assertTrue(control["environment_restored"])

    def test_srm_batch_parallel_auto_workers_uses_cpu_and_memory_budget(self) -> None:
        with (
            patch("geofem_app.fem2d_solver.os.cpu_count", return_value=8),
            patch("geofem_app.fem2d_solver._srm_physical_cpu_count", return_value=None),
            patch("geofem_app.fem2d_solver._srm_available_memory_mb", return_value=16000.0),
        ):
            settings = _srm_parallel_settings(
                {
                    "_runtime": {"context": "cli", "profile": "batch"},
                    "parallel": {
                        "enabled": True,
                        "policy": "batch",
                        "max_workers": "auto",
                        "memory_limit_mb": 2048,
                        "memory_per_worker_mb": 512,
                    },
                },
                10,
            )

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["policy"], "batch")
        self.assertEqual(settings["context"], "cli")
        self.assertEqual(settings["requested_workers"], 7)
        self.assertEqual(settings["memory_worker_cap"], 4)
        self.assertTrue(settings["memory_limited"])
        self.assertEqual(settings["max_workers"], 4)

    def test_srm_parallel_auto_records_environment_mesh_and_thread_plan(self) -> None:
        mesh = Mesh2D(
            node_ids=["1", "2", "3", "4"],
            coords=np.zeros((4, 2), dtype=float),
            elements=[Element2D("E1", "QUAD4", ("1", "2", "3", "4"), "soil")],
        )
        with (
            patch("geofem_app.fem2d_solver.os.cpu_count", return_value=8),
            patch("geofem_app.fem2d_solver._srm_physical_cpu_count", return_value=4),
            patch("geofem_app.fem2d_solver._srm_available_memory_mb", return_value=4096.0),
        ):
            settings = _srm_parallel_settings(
                {
                    "_runtime": {"context": "gui"},
                    "parallel": {
                        "enabled": True,
                        "policy": "auto",
                        "strategy": "lookahead",
                        "preserve_decision_order": True,
                        "lookahead_depth": 1,
                        "bisection_speculation": False,
                        "max_workers": "auto",
                        "memory_fraction": 0.5,
                    },
                },
                6,
                mesh=mesh,
            )

        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["context"], "gui")
        self.assertEqual(settings["policy"], "auto")
        self.assertEqual(settings["max_workers"], 3)
        self.assertEqual(settings["strategy"], "lookahead")
        self.assertTrue(settings["preserve_decision_order"])
        self.assertEqual(settings["lookahead_depth"], 1)
        self.assertFalse(settings["bisection_speculation"])
        self.assertEqual(settings["environment"]["logical_cpu_count"], 8)
        self.assertEqual(settings["environment"]["physical_cpu_count"], 4)
        self.assertEqual(settings["available_memory_mb"], 4096.0)
        self.assertEqual(settings["mesh"]["node_count"], 4)
        self.assertEqual(settings["mesh"]["dof_count"], 8)
        self.assertEqual(settings["memory_limit_mb"], 2048.0)
        self.assertEqual(settings["memory_per_worker_source"], "mesh_heuristic")
        self.assertEqual(settings["selected_threads_per_worker"], 2)
        self.assertEqual(settings["numeric_thread_env"]["OMP_NUM_THREADS"], "2")

        stage = StageResult2D(
            "srm",
            np.zeros(8, dtype=float),
            np.zeros(8, dtype=float),
            [],
            {},
            ["E1"],
            {"srm": {"factor_of_safety": 1.0, "trials": [], "parallel": settings}},
        )
        solve = SolveResult2D(mesh, {}, [stage], Path("."))
        events = build_structured_analysis_log(solve)["events"]
        stage_event = next(row for row in events if row.get("event_type") == "stage_completed")
        summary_event = next(row for row in events if row.get("event_type") == "srm_summary")
        self.assertEqual(stage_event["srm_parallel_max_workers"], 3)
        self.assertEqual(stage_event["srm_parallel_selected_threads_per_worker"], 2)
        self.assertEqual(summary_event["srm_parallel_dof_count"], 8)

    def test_cli_runtime_defaults_mark_solve_as_batch_and_accept_srm_worker_overrides(self) -> None:
        cfg = _apply_cli_solver_runtime_defaults(
            {"analysis": {"type": "static_plane_strain"}},
            srm_workers="auto",
            srm_parallel_policy="batch",
            srm_memory_limit_mb=4096,
            srm_memory_per_worker_mb=512,
            cancel_file="cancel.flag",
        )

        self.assertEqual(cfg["solver"]["execution"]["context"], "cli")
        self.assertEqual(cfg["solver"]["execution"]["profile"], "batch")
        self.assertTrue(str(cfg["solver"]["execution"]["cancel_file"]).endswith("cancel.flag"))
        parallel = cfg["solver"]["srm"]["parallel"]
        self.assertTrue(parallel["enabled"])
        self.assertEqual(parallel["max_workers"], "auto")
        self.assertEqual(parallel["memory_limit_mb"], 4096.0)
        self.assertEqual(parallel["memory_per_worker_mb"], 512.0)

    def test_srm_cancel_file_token_from_runtime_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cancel_file = Path(tmp) / "cancel.flag"
            token = fem2d_solver_module._srm_cancel_token_from_config(
                {"_runtime": {"cancel_file": str(cancel_file)}},
                {},
            )
            self.assertFalse(fem2d_solver_module._srm_cancel_requested(token))
            cancel_file.write_text("cancel", encoding="utf-8")
            self.assertTrue(fem2d_solver_module._srm_cancel_requested(token))

    def test_srm_running_trial_cancel_reports_solver_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cancel_file = Path(tmp) / "cancel.flag"
            cancel_file.write_text("cancel", encoding="utf-8")

            def fake_trial(factor: float) -> StageResult2D:
                fem2d_solver_module._raise_if_solver_cancel_requested(
                    {"execution": {"cancel_file": str(cancel_file)}},
                    f"FS{factor:g}",
                    "newton_iteration_start",
                    attempted_load_factor=0.5,
                    last_accepted_load_factor=0.25,
                    accepted_increment_count=1,
                    newton_iterations_total=2,
                )
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    [],
                    {"converged": True},
                )

            _result, fos, trials, info = _run_srm_trial_search([1.0], {"factors": [1.0]}, 0.0, fake_trial)

        self.assertEqual(fos, 0.0)
        self.assertEqual(info["search_mode"], "explicit_factors")
        self.assertEqual(len(trials), 1)
        failed = trials[0]
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["failure_reason"], "cancelled")
        self.assertEqual(failed["trial_status"], "solver_cancelled")
        self.assertTrue(failed["solver_cancel_requested"])
        self.assertEqual(failed["solver_cancel_checkpoint"], "newton_iteration_start")
        self.assertEqual(failed["last_accepted_load_factor"], 0.25)

        stage = StageResult2D(
            "srm",
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            [],
            {},
            [],
            {"srm": {"factor_of_safety": fos, "trials": trials, **info}},
        )
        solve = SolveResult2D(Mesh2D([], np.zeros((0, 2), dtype=float), []), {}, [stage], Path("."))
        events = build_structured_analysis_log(solve)["events"]
        event = next(row for row in events if row.get("event_type") == "srm_trial")
        self.assertTrue(event["solver_cancel_requested"])
        self.assertEqual(event["solver_cancel_checkpoint"], "newton_iteration_start")

    def test_srm_lightweight_trial_compacts_fallback_rows_from_non_adopted_trials(self) -> None:
        def fake_trial(factor: float) -> StageResult2D:
            plastic = 0.0 if factor <= 1.0 else 1.0
            result = StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                ["1"],
                {"converged": True, "postprocess_results": False},
            )
            result.integration_point_results = [{"element": "1", "gp": 1}]
            result.interface_results = [{"interface": "i1"}]
            result.structural_results = [{"element": "b1"}]
            return result

        _result, fos, trials, info = _run_srm_trial_search([1.0, 1.2], {"factors": [1.0, 1.2]}, 0.0, fake_trial)

        self.assertEqual(fos, 1.0)
        self.assertEqual(info["search_mode"], "explicit_factors")
        self.assertEqual([row["trial_element_rows"] for row in trials], [0, 0])
        self.assertEqual([row["trial_integration_point_rows"] for row in trials], [0, 0])
        self.assertTrue(all(row["lightweight_trial_compacted"] for row in trials))
        self.assertEqual(trials[0]["lightweight_removed_element_rows"], 1)
        self.assertEqual(trials[0]["lightweight_removed_integration_point_rows"], 1)

    def test_srm_progress_stdout_reports_trials_during_search(self) -> None:
        def fake_trial(factor: float) -> StageResult2D:
            plastic = 0.0 if factor <= 1.0 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        out = io.StringIO()
        with redirect_stdout(out):
            _result, fos, trials, info = _run_srm_trial_search(
                [1.0, 1.2],
                {"factors": [1.0, 1.2], "progress_stdout": True, "_runtime": {"context": "gui"}},
                0.0,
                fake_trial,
                progress_stage_name="srm",
            )

        text = out.getvalue()
        self.assertEqual(fos, 1.0)
        self.assertEqual(info["search_mode"], "explicit_factors")
        self.assertEqual(len(trials), 2)
        self.assertIn("[GeoFEM][SRM trial] stage=srm trial=1/2 factor=1.0", text)
        self.assertIn("current_FOS=1", text)
        self.assertIn("factor=1.2 ok=False", text)
        self.assertIn("diag=(", text)

    def test_srm_failed_trial_keeps_solver_diagnostics_in_analysis_log(self) -> None:
        def fake_trial(factor: float) -> StageResult2D:
            if factor > 1.0:
                raise FEM2DError(
                    "increment cutback limit reached",
                    diagnostics={
                        "trial_status": "increment_cutback_limit",
                        "attempted_load_factor": 0.875,
                        "last_accepted_load_factor": 0.75,
                        "accepted_increment_count": 3,
                        "cutback_count": 2,
                        "failed_step_size": 0.125,
                        "next_step_size": 0.0625,
                        "last_accepted_plastic_ratio": 0.42,
                        "last_accepted_plastic_point_count": 12,
                        "last_accepted_residual_norm": 1.0e-5,
                    },
                )
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        _result, fos, trials, info = _run_srm_trial_search([1.0, 1.2], {"factors": [1.0, 1.2]}, 0.0, fake_trial)

        self.assertEqual(fos, 1.0)
        self.assertEqual(info["search_mode"], "explicit_factors")
        self.assertEqual(len(trials), 2)
        failed = trials[1]
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["trial_status"], "increment_cutback_limit")
        self.assertEqual(failed["last_accepted_load_factor"], 0.75)
        self.assertEqual(failed["last_accepted_plastic_ratio"], 0.42)
        self.assertIn("diagnostic_summary", failed)
        self.assertIn("increment_cutback_limit", failed["diagnostic_summary"])

        stage = StageResult2D(
            "srm",
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            [],
            {},
            ["e1"],
            {"srm": {"factor_of_safety": fos, "trials": trials, **info}},
        )
        solve = SolveResult2D(Mesh2D([], np.zeros((0, 2), dtype=float), []), {}, [stage], Path("."))
        events = build_structured_analysis_log(solve)["events"]
        failed_event = next(row for row in events if row.get("event_type") == "srm_trial" and row.get("srm_factor") == 1.2)
        self.assertEqual(failed_event["trial_status"], "increment_cutback_limit")
        self.assertEqual(failed_event["last_accepted_load_factor"], 0.75)
        self.assertEqual(failed_event["last_accepted_plastic_ratio"], 0.42)
        self.assertIn("last_pr=0.42", failed_event["message"])

    def test_srm_trial_diagnostics_include_plastic_cluster_and_strain_metrics(self) -> None:
        mesh = Mesh2D(
            ["1", "2", "3", "4", "5", "6"],
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                ],
                dtype=float,
            ),
            [
                Element2D("e1", "QUAD4", ("1", "2", "5", "4"), "soil"),
                Element2D("e2", "QUAD4", ("2", "3", "6", "5"), "soil"),
            ],
        )

        def fake_trial(factor: float) -> StageResult2D:
            plastic_state = {}
            if factor > 1.0:
                plastic_state = {
                    "e1:0": PlasticState2D(np.array([0.02, -0.01, -0.01, 0.004], dtype=float), 0.03),
                    "e2:0": PlasticState2D(np.array([0.04, -0.02, -0.02, 0.008], dtype=float), 0.06),
                }
            return StageResult2D(
                f"FS{factor:g}",
                np.array([0.0, 0.0, 0.0, -0.01 * factor, 0.0, -0.02 * factor, 0.0, 0.0, 0.0, -0.01 * factor, 0.0, -0.02 * factor], dtype=float),
                np.zeros(12, dtype=float),
                [],
                {},
                ["e1", "e2"],
                {
                    "converged": True,
                    "residual_norm": 1.0e-8,
                    "iterations": int(2 + factor),
                    "line_search_reductions": 1 if factor > 1.0 else 0,
                    "internal_external_work_ratio": 0.99,
                    "convergence_history": [{"residual_norm": 1.0}, {"residual_norm": 1.0e-8}],
                },
                plastic_state=plastic_state,
            )

        _result, fos, trials, info = _run_srm_trial_search([1.0, 1.2], {"factors": [1.0, 1.2]}, 0.5, fake_trial, mesh=mesh)

        self.assertEqual(fos, 1.0)
        self.assertEqual(info["search_mode"], "explicit_factors")
        failed = trials[1]
        for key in (
            "plastic_ratio_delta",
            "max_equivalent_plastic_strain",
            "mean_equivalent_plastic_strain",
            "top_percentile_equivalent_plastic_strain",
            "yielded_element_count",
            "connected_plastic_cluster_size",
            "plastic_cluster_spans_boundary",
            "final_step_size",
            "newton_iterations_total",
            "newton_iterations_max",
            "line_search_reductions_total",
            "residual_norm_final",
            "residual_reduction_ratio",
            "min_det_j",
            "max_displacement_norm",
            "displacement_increment_norm",
            "internal_external_work_ratio",
        ):
            self.assertIn(key, failed)
        self.assertGreater(failed["plastic_ratio_delta"], 0.0)
        self.assertAlmostEqual(failed["max_equivalent_plastic_strain"], 0.06)
        self.assertEqual(failed["yielded_element_count"], 2)
        self.assertEqual(failed["connected_plastic_cluster_size"], 2)
        self.assertTrue(failed["plastic_cluster_spans_boundary"])
        self.assertGreater(failed["min_det_j"], 0.0)

    def test_srm_selected_factor_ignores_all_failed_trials(self) -> None:
        self.assertIsNone(_srm_selected_factor(0.0, [{"factor": 1.2, "ok": False, "failure_reason": "nonconvergence"}]))
        self.assertEqual(_srm_selected_factor(1.1, [{"factor": 1.0, "ok": True}, {"factor": 1.2, "ok": False}]), 1.1)
        self.assertEqual(_srm_selected_factor(0.0, [{"factor": 1.0, "ok": True}, {"factor": 1.2, "ok": False}]), 1.0)

    def test_srm_adaptive_bracket_reduces_explicit_factor_trials(self) -> None:
        evaluated: list[float] = []

        def fake_trial(factor: float) -> StageResult2D:
            factor = round(float(factor), 6)
            evaluated.append(factor)
            plastic = 0.0 if factor <= 1.23 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        factors = [round(1.0 + 0.1 * i, 3) for i in range(11)]
        _result, fos, trials, info = _run_srm_trial_search(
            factors,
            {
                "factors": factors,
                "adaptive": True,
                "bracket_stride": 4,
                "factor_tol": 0.02,
                "max_bisection": 4,
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(info["search_mode"], "adaptive_bracket")
        self.assertTrue(info["bracketed"])
        self.assertLess(len(evaluated), len(factors))
        self.assertGreaterEqual(fos, 1.2)
        self.assertLess(fos, 1.25)
        self.assertLess(info["parallel"]["evaluated_trials"], len(factors))
        self.assertEqual(trials[0]["factor"], 1.0)

    def test_srm_adaptive_bracket_lookahead_prefetches_window_preserving_order(self) -> None:
        active = 0
        max_active = 0
        evaluated: list[float] = []
        lock = threading.Lock()

        def fake_trial(factor: float) -> StageResult2D:
            nonlocal active, max_active
            factor = round(float(factor), 6)
            with lock:
                evaluated.append(factor)
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.02)
                plastic = 0.0 if factor <= 1.05 else 1.0
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": plastic}],
                    {},
                    [],
                    {"converged": True},
                )
            finally:
                with lock:
                    active -= 1

        factors = [1.0, 1.1, 1.2, 1.3, 1.4]
        _result, fos, trials, info = _run_srm_trial_search(
            factors,
            {
                "factors": factors,
                "adaptive": True,
                "bracket_stride": 2,
                "factor_tol": 0.03,
                "max_bisection": 3,
                "parallel": {
                    "enabled": True,
                    "max_workers": 2,
                    "strategy": "lookahead",
                    "preserve_decision_order": True,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertTrue(info["bracketed"])
        self.assertGreaterEqual(fos, 1.0)
        self.assertLessEqual(fos, 1.05)
        self.assertEqual(info["parallel"]["strategy"], "adaptive_bracket_lookahead")
        self.assertTrue(info["parallel"]["enabled"])
        self.assertEqual(info["parallel"]["lookahead_depth"], 2)
        self.assertEqual(info["parallel"]["speculative_trial_count"], 2)
        self.assertEqual(info["parallel"]["unused_speculative_trial_count"], 1)
        self.assertEqual(info["parallel"]["canceled_speculative_trial_count"], 0)
        self.assertGreaterEqual(info["parallel"]["speculative_prefetch_call_count"], 1)
        self.assertGreater(info["parallel"]["speculative_prefetch_wall_elapsed_seconds"], 0.0)
        self.assertGreater(info["parallel"]["speculative_trial_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(info["parallel"]["speculative_queue_wait_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(info["parallel"]["speculative_estimated_wall_clock_saving_seconds"], 0.0)
        self.assertEqual(info["parallel"]["speculative_unused_factors"], [1.4])
        self.assertNotIn(1.4, [row["factor"] for row in trials])
        self.assertIn(1.4, evaluated)
        self.assertGreater(max_active, 1)

    def test_srm_parallel_evaluation_skips_not_started_trials_when_cancel_requested(self) -> None:
        evaluated: list[float] = []

        def fake_trial(factor: float) -> StageResult2D:
            evaluated.append(float(factor))
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                [],
                {"converged": True},
            )

        stats: dict[str, Any] = {}
        records = _srm_evaluate_records_parallel(
            [1.0, 1.1, 1.2],
            0.0,
            fake_trial,
            workers=2,
            cancel_token=lambda: True,
            cancellation_stats=stats,
        )

        self.assertEqual(records, [])
        self.assertEqual(evaluated, [])
        self.assertTrue(stats["requested"])
        self.assertEqual(stats["skipped_count"], 3)
        self.assertEqual(stats.get("canceled_count", 0), 0)

    def test_srm_parallel_decision_boundary_stops_running_out_of_range_trial(self) -> None:
        speculative_started = threading.Event()

        def fake_trial(
            factor: float,
            *,
            solver_override: Mapping[str, Any] | None = None,
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor == 1.1:
                self.assertTrue(speculative_started.wait(timeout=1.0))
                plastic = 1.0
            else:
                speculative_started.set()
                for _ in range(200):
                    fem2d_solver_module._raise_if_solver_cancel_requested(
                        solver_override,
                        f"FS{factor:g}",
                        "newton_iteration_start",
                    )
                    time.sleep(0.002)
                plastic = 0.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        stats: dict[str, Any] = {}
        records = _srm_evaluate_records_parallel(
            [1.1, 1.2],
            0.0,
            fake_trial,
            workers=2,
            cancellation_stats=stats,
            decision_cancel_callback=(
                lambda index, record, factors: [1]
                if index == 0 and not bool(record.get("ok", False))
                else []
            ),
        )

        self.assertEqual([record["factor"] for record in records], [1.1])
        self.assertEqual(stats["decision_linked_requested_count"], 1)
        self.assertEqual(stats["decision_linked_safe_stop_count"], 1)
        self.assertEqual(stats["decision_linked_requested_factors"], [1.2])
        self.assertGreaterEqual(stats["canceled_count"], 1)

    def test_srm_adaptive_lookahead_reports_decision_linked_safe_stop(self) -> None:
        speculative_started = threading.Event()

        def fake_trial(
            factor: float,
            *,
            solver_override: Mapping[str, Any] | None = None,
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor == 1.1:
                self.assertTrue(speculative_started.wait(timeout=1.0))
                plastic = 1.0
            elif factor == 1.2:
                speculative_started.set()
                for _ in range(200):
                    fem2d_solver_module._raise_if_solver_cancel_requested(
                        solver_override,
                        f"FS{factor:g}",
                        "increment_start",
                    )
                    time.sleep(0.002)
                plastic = 1.0
            else:
                plastic = 0.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.1, 1.2],
            {
                "factors": [1.0, 1.1, 1.2],
                "adaptive": True,
                "bracket_stride": 1,
                "factor_tol": 0.05,
                "max_bisection": 0,
                "parallel": {
                    "enabled": True,
                    "max_workers": 2,
                    "strategy": "lookahead",
                    "preserve_decision_order": True,
                    "decision_linked_cancellation": True,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(fos, 1.0)
        self.assertEqual(info["stable_factor"], 1.0)
        self.assertEqual(info["failed_factor"], 1.1)
        self.assertEqual([row["factor"] for row in trials], [1.0, 1.1])
        parallel = info["parallel"]
        self.assertTrue(parallel["decision_linked_cancellation_enabled"])
        self.assertEqual(parallel["decision_linked_requested_count"], 1)
        self.assertEqual(parallel["decision_linked_safe_stop_count"], 1)
        self.assertEqual(parallel["decision_linked_requested_factors"], [1.2])

    def test_increment_boundary_verification_resumes_accepted_checkpoint(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        reference_traction = abs(float(cfg["loads"][0]["ty"]))
        phase = "initial"
        targets: dict[str, list[float]] = {
            "initial": [],
            "resume": [],
            "fallback": [],
        }

        def fake_plane_strain_stage(**kwargs: Any) -> StageResult2D:
            load_rows = kwargs.get("loads", [])
            target = (
                abs(float(load_rows[0]["ty"])) / reference_traction
                if load_rows
                else 0.0
            )
            targets[phase].append(round(target, 6))
            if phase == "initial" and target > 0.75:
                raise FEM2DError("controlled boundary failure")
            result = StageResult2D(
                str(kwargs.get("stage_name", "increment")),
                np.full(2 * len(mesh.node_ids), target, dtype=float),
                np.zeros(2 * len(mesh.node_ids), dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                [element.id for element in mesh.elements if element.active],
                {
                    "converged": True,
                    "iterations": 2,
                    "line_search_reductions": 0,
                    "residual_norm": 1.0e-9,
                },
            )
            return result

        base_increments = {
            "enabled": True,
            "steps": 2,
            "max_cutbacks": 0,
            "min_step": 0.01,
            "cutback_factor": 0.5,
            "growth": 1.0,
            "share_step_cache": False,
        }
        base_solver: dict[str, Any] = {
            "increments": dict(base_increments),
            "newton": {"max_iter": 10},
            "_srm_capture_increment_checkpoint": True,
        }
        solve_kwargs = {
            "mesh": mesh,
            "materials": materials,
            "boundary_conditions": cfg["boundary_conditions"],
            "loads": cfg["loads"],
            "mpc_constraints": None,
            "stage_name": "checkpoint-test",
            "output_dir": None,
            "initial_stresses": None,
            "interfaces": None,
            "structural_elements": None,
            "strength_factor": 1.2,
            "pore_pressure": None,
            "time": 1.0,
            "plastic_state": None,
            "postprocess_results": False,
        }

        with patch.object(
            fem2d_solver_module,
            "solve_plane_strain_stage",
            side_effect=fake_plane_strain_stage,
        ):
            with self.assertRaises(FEM2DError) as captured:
                fem2d_solver_module.solve_incremental_stage(
                    solver=base_solver,
                    **solve_kwargs,
                )
            checkpoint = getattr(
                captured.exception,
                "_srm_increment_continuation_checkpoint",
                None,
            )
            self.assertIsNotNone(checkpoint)
            self.assertEqual(targets["initial"], [0.5, 1.0])
            self.assertAlmostEqual(checkpoint.target, 0.5)
            self.assertEqual(checkpoint.accepted_steps, 1)

            phase = "resume"
            resume_solver = {
                "increments": {
                    **base_increments,
                    "max_cutbacks": 2,
                    "min_step": 0.005,
                },
                "newton": {"max_iter": 20},
                "_srm_increment_checkpoint": checkpoint,
            }
            resumed = fem2d_solver_module.solve_incremental_stage(
                solver=resume_solver,
                **solve_kwargs,
            )

            phase = "fallback"
            mismatch_solver = {
                "increments": {
                    **base_increments,
                    "max_cutbacks": 2,
                    "min_step": 0.005,
                },
                "newton": {"max_iter": 20, "tolerance": 1.0e-7},
                "_srm_increment_checkpoint": checkpoint,
            }
            fallback = fem2d_solver_module.solve_incremental_stage(
                solver=mismatch_solver,
                **solve_kwargs,
            )

        self.assertEqual(targets["resume"], [0.75, 1.0])
        resumed_info = resumed.solver_info["increments"]
        self.assertTrue(resumed_info["checkpoint_continuation_requested"])
        self.assertTrue(resumed_info["checkpoint_continuation_used"])
        self.assertEqual(resumed_info["checkpoint_source_load_factor"], 0.5)
        self.assertEqual(resumed_info["checkpoint_resumed_accepted_steps"], 1)
        self.assertEqual(resumed_info["checkpoint_reused_history_rows"], 2)

        self.assertEqual(targets["fallback"], [0.5, 1.0])
        fallback_info = fallback.solver_info["increments"]
        self.assertTrue(fallback_info["checkpoint_continuation_requested"])
        self.assertFalse(fallback_info["checkpoint_continuation_used"])
        self.assertEqual(
            fallback_info["checkpoint_fallback_reason"],
            "fingerprint_mismatch",
        )

    def test_srm_adaptive_lookahead_cancels_speculative_prefetch_without_changing_order(self) -> None:
        evaluated: list[float] = []
        cancel_event = threading.Event()
        cancel_event.set()

        def fake_trial(factor: float) -> StageResult2D:
            factor = round(float(factor), 6)
            evaluated.append(factor)
            plastic = 0.0 if factor <= 1.1 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        factors = [1.0, 1.1, 1.2, 1.3]
        _result, fos, trials, info = _run_srm_trial_search(
            factors,
            {
                "factors": factors,
                "adaptive": True,
                "bracket_stride": 1,
                "factor_tol": 0.05,
                "max_bisection": 0,
                "parallel": {
                    "enabled": True,
                    "max_workers": 2,
                    "strategy": "lookahead",
                    "preserve_decision_order": True,
                    "_cancel_requested": cancel_event.is_set,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(fos, 1.1)
        self.assertEqual([row["factor"] for row in trials], [1.0, 1.1, 1.2])
        self.assertEqual(evaluated, [1.0, 1.1, 1.2])
        self.assertTrue(info["bracketed"])
        self.assertTrue(info["parallel"]["enabled"])
        self.assertEqual(info["parallel"]["speculative_trial_count"], 0)
        self.assertEqual(info["parallel"]["canceled_speculative_trial_count"], 2)
        self.assertTrue(info["parallel"]["speculative_cancellation_requested"])
        self.assertIn("cancellation", info["parallel"]["speculative_cancellation_note"])

    def test_srm_adaptive_bisection_speculates_next_midpoints_preserving_bracket(self) -> None:
        active = 0
        max_active = 0
        evaluated: list[float] = []
        lock = threading.Lock()

        def fake_trial(factor: float) -> StageResult2D:
            nonlocal active, max_active
            factor = round(float(factor), 6)
            with lock:
                evaluated.append(factor)
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.02)
                plastic = 0.0 if factor <= 1.2 else 1.0
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": plastic}],
                    {},
                    [],
                    {"converged": True},
                )
            finally:
                with lock:
                    active -= 1

        factors = [1.0, 1.1, 1.2, 1.3, 1.4]
        _result, fos, trials, info = _run_srm_trial_search(
            factors,
            {
                "factors": factors,
                "adaptive": True,
                "bracket_stride": 4,
                "factor_tol": 0.05,
                "max_bisection": 3,
                "parallel": {
                    "enabled": True,
                    "max_workers": 3,
                    "strategy": "lookahead",
                    "preserve_decision_order": True,
                    "bisection_speculation": True,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertTrue(info["bracketed"])
        self.assertAlmostEqual(fos, 1.2)
        self.assertAlmostEqual(info["stable_factor"], 1.2)
        self.assertAlmostEqual(info["failed_factor"], 1.25)
        self.assertEqual(info["parallel"]["strategy"], "adaptive_bracket_lookahead")
        self.assertTrue(info["parallel"]["bisection_speculation_enabled"])
        self.assertGreaterEqual(info["parallel"]["bisection_speculative_trial_count"], 3)
        self.assertGreaterEqual(info["parallel"]["bisection_used_speculative_trial_count"], 2)
        self.assertGreaterEqual(info["parallel"]["bisection_unused_speculative_trial_count"], 1)
        self.assertGreaterEqual(info["parallel"]["speculative_prefetch_call_count"], 1)
        self.assertGreater(info["parallel"]["speculative_trial_elapsed_seconds"], 0.0)
        self.assertGreaterEqual(info["parallel"]["speculative_estimated_wall_clock_saving_seconds"], 0.0)
        self.assertIn(1.1, info["parallel"]["bisection_speculative_unused_factors"])
        self.assertIn(1.1, evaluated)
        self.assertNotIn(1.1, [row["factor"] for row in trials])
        self.assertGreater(max_active, 1)

    def test_srm_adaptive_bracket_searches_lower_branch_when_anchor_fails(self) -> None:
        evaluated: list[float] = []

        def fake_trial(factor: float) -> StageResult2D:
            factor = round(float(factor), 6)
            evaluated.append(factor)
            plastic = 0.0 if factor <= 0.86 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.1, 1.2, 1.3],
            {
                "search_mode": "two_branch",
                "adaptive": True,
                "anchor_factor": 1.0,
                "lower_branch": {"factor_min": 0.5, "factor_step": 0.1},
                "upper_branch": {"factor_max": 1.3, "factor_step": 0.1},
                "bracket_stride": 2,
                "factor_tol": 0.02,
                "max_bisection": 4,
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(info["search_mode"], "adaptive_bracket")
        self.assertEqual(info["scan_direction"], "lower")
        self.assertTrue(info["bracketed"])
        self.assertLess(fos, 1.0)
        self.assertGreaterEqual(fos, 0.8)
        self.assertLess(len(evaluated), 8)
        self.assertEqual([row["factor"] for row in trials[:2]], [1.0, 0.8])

    def test_srm_auto_lower_projection_lookahead_prefetches_probe_window(self) -> None:
        active = 0
        max_active = 0
        evaluated: list[float] = []
        lock = threading.Lock()

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            nonlocal active, max_active
            factor = round(float(factor), 6)
            with lock:
                evaluated.append(factor)
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.02)
                if factor <= 0.9:
                    return StageResult2D(
                        f"FS{factor:g}",
                        np.zeros(0, dtype=float),
                        np.zeros(0, dtype=float),
                        [{"active": 1.0, "plastic": 0.0}],
                        {},
                        ["e1"],
                        {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                    )
                raise FEM2DError(
                    "increment cutback limit reached",
                    diagnostics={
                        "trial_status": "increment_cutback_limit",
                        "last_accepted_load_factor": 0.8,
                        "last_accepted_plastic_ratio": 0.4,
                        "active_element_count": 10,
                        "connected_plastic_cluster_size": 4,
                        "plastic_cluster_spans_boundary": False,
                        "cutback_count": 10,
                        "max_cutbacks": 12,
                        "final_step_size": 1.0e-4,
                    },
                )
            finally:
                with lock:
                    active -= 1

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "lower_branch": {"factor_min": 0.5, "factor_step": 0.1},
                "upper_branch": {"factor_max": 1.2, "factor_step": 0.1},
                "factor_tol": 0.02,
                "max_bisection": 3,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "lower_projection_enabled": True,
                    "lower_projection_multipliers": [1.10, 1.20, 0.98],
                    "lower_projection_max_probes": 3,
                    "lower_projection_skip_coarse_scan_on_bracket": True,
                },
                "parallel": {
                    "enabled": True,
                    "max_workers": 2,
                    "strategy": "lookahead",
                    "preserve_decision_order": True,
                },
            },
            0.0,
            fake_trial,
        )

        projection = info["lower_projection"]
        self.assertEqual(info["search_mode"], "auto")
        self.assertEqual(info["scan_direction"], "lower")
        self.assertTrue(projection["used"])
        self.assertTrue(projection["parallel_prefetch_enabled"])
        self.assertEqual(projection["parallel_prefetch_count"], 3)
        self.assertEqual(projection["parallel_prefetch_used_count"], 3)
        self.assertEqual(projection["parallel_prefetch_unused_count"], 0)
        self.assertEqual(info["parallel"]["strategy"], "auto_diagnostic_bracket_lookahead")
        self.assertIn(0.784, evaluated)
        self.assertIn(0.784, [row["factor"] for row in trials])
        self.assertGreater(max_active, 1)
        self.assertGreaterEqual(fos, 0.88)
        self.assertLess(fos, 0.91)
        self.assertFalse(info["bracketed"])
        self.assertEqual(info["factor_of_safety_status"], "lower_bound_indeterminate")
        self.assertEqual(info["factor_of_safety_confidence"], "limited")
        self.assertIsNone(info["failed_factor"])

    def test_srm_auto_suspect_failure_keeps_broad_bracket_unresolved(self) -> None:
        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor <= 1.2:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            if factor < 1.4:
                raise FEM2DError(
                    "increment cutback limit reached",
                    diagnostics={
                        "trial_status": "increment_cutback_limit",
                        "last_accepted_load_factor": 0.85,
                        "last_accepted_plastic_ratio": 0.35,
                        "active_element_count": 10,
                        "connected_plastic_cluster_size": 3,
                        "plastic_cluster_spans_boundary": False,
                        "cutback_count": 12,
                        "max_cutbacks": 12,
                    },
                )
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.98,
                    "last_accepted_plastic_ratio": 0.80,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.2, 1.4],
                "factor_tol": 0.02,
                "max_bisection": 4,
                "auto": {"enabled": True, "retry_suspect_failures": False, "max_suspect_retries": 0},
            },
            0.0,
            fake_trial,
        )

        suspect = next(row for row in trials if row.get("factor") == 1.3)
        self.assertEqual(suspect["auto_decision"], "suspect_failure")
        self.assertEqual(suspect["srm_trial_state"], "indeterminate")
        self.assertAlmostEqual(fos, 1.2)
        self.assertTrue(info["bracketed"])
        self.assertAlmostEqual(info["stable_factor"], 1.2)
        self.assertAlmostEqual(info["failed_factor"], 1.4)
        self.assertEqual(
            info["factor_of_safety_status"],
            "unresolved_indeterminate_interval",
        )
        self.assertFalse(info["bracket_resolved"])
        self.assertEqual(info["indeterminate_factors_inside_bracket"], [1.3])
        self.assertFalse(info["factor_of_safety_certified"])
        self.assertEqual(
            info["factor_of_safety_value_kind"],
            "unresolved_interval_lower_bound",
        )
        self.assertIn(1.3, info["indeterminate_factors"])

    def test_srm_auto_retries_suspect_failure_before_bracketing(self) -> None:
        calls: list[tuple[float, bool]] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            retry = isinstance(solver_override, Mapping) and bool(solver_override.get("_srm_auto_retry", False))
            calls.append((factor, retry))
            if factor <= 1.2 or (retry and factor <= 1.3):
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            if factor <= 1.3:
                raise FEM2DError(
                    "increment cutback limit reached",
                    diagnostics={
                        "trial_status": "increment_cutback_limit",
                        "last_accepted_load_factor": 0.85,
                        "last_accepted_plastic_ratio": 0.42,
                        "active_element_count": 10,
                        "connected_plastic_cluster_size": 4,
                        "plastic_cluster_spans_boundary": False,
                        "cutback_count": 12,
                        "max_cutbacks": 12,
                        "final_step_size": 1.0e-4,
                    },
                )
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.96,
                    "last_accepted_plastic_ratio": 0.76,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                    "final_step_size": 1.0e-4,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "upper_branch": {"factor_max": 1.4, "factor_step": 0.2},
                "factor_tol": 0.05,
                "max_bisection": 3,
                "auto": {"enabled": True, "max_suspect_retries": 1},
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(info["search_mode"], "auto")
        self.assertTrue(info["auto"]["enabled"])
        self.assertGreaterEqual(info["auto"]["retry_count"], 1)
        self.assertIn((1.3, True), calls)
        self.assertGreaterEqual(fos, 1.3)
        suspect = next(row for row in trials if row.get("factor") == 1.3 and not row.get("auto_retry"))
        retry = next(row for row in trials if row.get("factor") == 1.3 and row.get("auto_retry"))
        self.assertEqual(suspect["auto_decision"], "suspect_failure")
        self.assertTrue(suspect["auto_superseded_by_retry"])
        self.assertTrue(retry["ok"])
        self.assertEqual(retry["auto_decision"], "stable")

    def test_srm_auto_verifies_suspect_and_early_failure_boundary(self) -> None:
        calls: list[tuple[float, bool]] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            verification = isinstance(solver_override, Mapping) and isinstance(
                solver_override.get("_srm_boundary_verification"), Mapping
            )
            calls.append((factor, verification))
            if factor <= 1.0 or (verification and factor <= 1.2):
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            diagnostics = {
                "trial_status": "increment_cutback_limit",
                "last_accepted_load_factor": 0.82 if factor <= 1.2 else 0.97,
                "last_accepted_plastic_ratio": 0.35 if factor <= 1.2 else 0.78,
                "active_element_count": 10,
                "connected_plastic_cluster_size": 3 if factor <= 1.2 else 8,
                "plastic_cluster_spans_boundary": factor > 1.2,
                "cutback_count": 12,
                "max_cutbacks": 12,
                "final_step_size": 1.0e-4,
            }
            if factor > 1.2:
                diagnostics["early_failure_stop"] = True
            raise FEM2DError("increment cutback limit reached", diagnostics=diagnostics)

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "upper_branch": {"factor_max": 1.4, "factor_step": 0.2},
                "factor_tol": 0.05,
                "max_bisection": 3,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "max_suspect_retries": 0,
                    "boundary_verification_enabled": True,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertIn((1.2, False), calls)
        self.assertIn((1.2, True), calls)
        self.assertGreaterEqual(fos, 1.2)
        verified_stable = next(
            row for row in trials if row.get("factor") == 1.2 and row.get("boundary_verification")
        )
        self.assertTrue(verified_stable["ok"])
        self.assertEqual(verified_stable["boundary_verification_result"], "stable")
        self.assertTrue(info["boundary_verified"])
        self.assertEqual(info["boundary_quality"], "verified_failure_boundary")
        self.assertTrue(info["factor_of_safety_boundary_certified"])
        self.assertTrue(info["factor_of_safety_tolerance_met"])
        self.assertTrue(info["factor_of_safety_certified"])
        self.assertEqual(info["factor_of_safety_value_kind"], "certified_stable_lower_bound")
        self.assertGreaterEqual(info["auto"]["boundary_verification_count"], 2)
        self.assertLessEqual(info["factor_of_safety_interval"]["width"], 0.05 + 1.0e-12)

    def test_srm_auto_defers_intermediate_early_failure_verification(self) -> None:
        calls: list[tuple[float, bool]] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            verification = isinstance(solver_override, Mapping) and isinstance(
                solver_override.get("_srm_boundary_verification"), Mapping
            )
            calls.append((factor, verification))
            if factor <= 1.2:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "srm_early_confirmed_failure" if not verification else "increment_cutback_limit",
                    "early_failure_stop": not verification,
                    "last_accepted_load_factor": 0.97,
                    "last_accepted_plastic_ratio": 0.80,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.4],
                "factor_tol": 0.05,
                "max_bisection": 4,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "boundary_verification_enabled": True,
                    "boundary_verification_strategy": "deferred_final",
                },
            },
            0.0,
            fake_trial,
        )

        verification_factors = [factor for factor, verification in calls if verification]
        self.assertEqual(verification_factors, [1.25])
        self.assertAlmostEqual(fos, 1.2)
        self.assertAlmostEqual(info["stable_factor"], 1.2)
        self.assertAlmostEqual(info["failed_factor"], 1.25)
        self.assertTrue(info["boundary_verified"])
        self.assertEqual(info["auto"]["boundary_verification_strategy"], "deferred_final")
        self.assertEqual(info["auto"]["boundary_verification_deferred_count"], 3)
        self.assertEqual(info["auto"]["boundary_verification_executed_count"], 1)
        deferred = [row for row in trials if row.get("boundary_verification_deferred")]
        self.assertEqual([row["factor"] for row in deferred], [1.4, 1.3, 1.25])

    def test_srm_auto_deferred_verification_rolls_back_stable_reversal(self) -> None:
        calls: list[tuple[float, bool]] = []

        def stable_result(factor: float) -> StageResult2D:
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            verification = isinstance(solver_override, Mapping) and isinstance(
                solver_override.get("_srm_boundary_verification"), Mapping
            )
            calls.append((factor, verification))
            if factor <= 1.15 or (verification and factor <= 1.2):
                return stable_result(factor)
            if abs(factor - 1.2) <= 1.0e-9:
                diagnostics = {
                    "trial_status": "srm_early_confirmed_failure",
                    "early_failure_stop": True,
                    "last_accepted_load_factor": 0.97,
                    "last_accepted_plastic_ratio": 0.80,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                }
            elif factor < 1.4:
                diagnostics = {
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.82,
                    "last_accepted_plastic_ratio": 0.25,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 2,
                    "plastic_cluster_spans_boundary": False,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                }
            else:
                diagnostics = {
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.97,
                    "last_accepted_plastic_ratio": 0.80,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                }
            raise FEM2DError("increment cutback limit reached", diagnostics=diagnostics)

        _result, fos, _trials, info = _run_srm_trial_search(
            [1.0, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.4],
                "factor_tol": 0.05,
                "max_bisection": 6,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "boundary_verification_enabled": True,
                    "boundary_verification_strategy": "deferred_final",
                },
            },
            0.0,
            fake_trial,
        )

        self.assertIn((1.2, False), calls)
        self.assertIn((1.2, True), calls)
        self.assertIn((1.3, True), calls)
        self.assertAlmostEqual(fos, 1.2)
        self.assertAlmostEqual(info["stable_factor"], 1.2)
        self.assertAlmostEqual(info["failed_factor"], 1.4)
        self.assertIn(1.3, info["indeterminate_factors"])
        self.assertEqual(info["auto"]["boundary_verification_recovery_count"], 1)
        self.assertEqual(info["auto"]["boundary_verification_stable_reversal_count"], 1)
        self.assertFalse(info["factor_of_safety_tolerance_met"])

    def test_srm_auto_resumes_upper_scan_after_verified_stable_endpoint(self) -> None:
        calls: list[tuple[float, bool]] = []

        def stable_result(factor: float) -> StageResult2D:
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        def fake_trial(
            factor: float, *, solver_override: Mapping[str, Any] | None = None
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            verification = isinstance(solver_override, Mapping) and isinstance(
                solver_override.get("_srm_boundary_verification"), Mapping
            )
            calls.append((factor, verification))
            if factor < 1.2 or (abs(factor - 1.2) <= 1.0e-9 and verification):
                return stable_result(factor)
            if abs(factor - 1.2) <= 1.0e-9:
                raise FEM2DError(
                    "provisional early failure",
                    diagnostics={
                        "trial_status": "srm_early_confirmed_failure",
                        "early_failure_stop": True,
                        "last_accepted_load_factor": 0.97,
                        "last_accepted_plastic_ratio": 0.80,
                        "active_element_count": 10,
                        "connected_plastic_cluster_size": 8,
                        "plastic_cluster_spans_boundary": True,
                        "cutback_count": 12,
                        "max_cutbacks": 12,
                    },
                )
            if factor < 1.4:
                return stable_result(factor)
            raise FEM2DError(
                "confirmed upper failure",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.99,
                    "last_accepted_plastic_ratio": 0.90,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 9,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                },
            )

        _result, fos, _trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.2, 1.4],
                "factor_tol": 0.05,
                "max_bisection": 6,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "boundary_verification_enabled": True,
                    "boundary_verification_strategy": "deferred_final",
                },
            },
            0.0,
            fake_trial,
        )

        verified_index = calls.index((1.2, True))
        upper_index = next(
            index for index, call in enumerate(calls) if call[0] == 1.4
        )
        self.assertGreater(upper_index, verified_index)
        self.assertGreaterEqual(fos, 1.35)
        self.assertAlmostEqual(info["failed_factor"], 1.4)
        self.assertEqual(info["factor_of_safety_status"], "confirmed_bracket")

    def test_srm_auto_cold_retries_checkpoint_indeterminate_boundary(self) -> None:
        calls: list[tuple[float, bool, bool]] = []

        def stable_result(factor: float) -> StageResult2D:
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        def fake_trial(
            factor: float, *, solver_override: Mapping[str, Any] | None = None
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            verification_cfg = (
                solver_override.get("_srm_boundary_verification")
                if isinstance(solver_override, Mapping)
                else None
            )
            verification = isinstance(verification_cfg, Mapping)
            cold_start = bool(
                verification
                and verification_cfg.get("cold_start", False)
                and str(verification_cfg.get("reason", "")).startswith(
                    "cold_indeterminate:"
                )
            )
            calls.append((factor, verification, cold_start))
            if factor <= 1.0 or (cold_start and factor < 1.3):
                return stable_result(factor)
            confirmed = factor >= 1.3
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.98 if confirmed else 0.82,
                    "last_accepted_plastic_ratio": 0.80 if confirmed else 0.25,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8 if confirmed else 2,
                    "plastic_cluster_spans_boundary": confirmed,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.4],
                "factor_tol": 0.10,
                "max_bisection": 5,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "boundary_verification_enabled": True,
                    "boundary_verification_cold_retry_on_indeterminate": True,
                    "boundary_verification_cold_retry_max_per_factor": 1,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertIn((1.2, True, True), calls)
        cold_rows = [
            row for row in trials if row.get("boundary_verification_cold_retry")
        ]
        self.assertEqual([row["factor"] for row in cold_rows], [1.2])
        self.assertTrue(cold_rows[0]["ok"])
        self.assertEqual(info["auto"]["boundary_verification_cold_retry_count"], 1)
        self.assertAlmostEqual(fos, 1.2)
        self.assertAlmostEqual(info["failed_factor"], 1.3)
        self.assertEqual(info["factor_of_safety_status"], "confirmed_bracket")

    def test_srm_auto_deferred_strategy_verifies_weak_provisional_failure_immediately(self) -> None:
        calls: list[tuple[float, bool]] = []

        def stable_result(factor: float) -> StageResult2D:
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            verification = isinstance(solver_override, Mapping) and isinstance(
                solver_override.get("_srm_boundary_verification"), Mapping
            )
            calls.append((factor, verification))
            if factor <= 1.0 or (verification and factor <= 1.2):
                return stable_result(factor)
            if abs(factor - 1.2) <= 1.0e-9:
                diagnostics = {
                    "trial_status": "srm_early_confirmed_failure",
                    "early_failure_stop": True,
                    "last_accepted_load_factor": 0.94,
                    "last_accepted_plastic_ratio": 0.60,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 6,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 9,
                    "max_cutbacks": 12,
                }
            else:
                diagnostics = {
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.97,
                    "last_accepted_plastic_ratio": 0.80,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                }
            raise FEM2DError("increment cutback limit reached", diagnostics=diagnostics)

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.4],
                "factor_tol": 0.05,
                "max_bisection": 4,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "boundary_verification_enabled": True,
                    "boundary_verification_strategy": "deferred_final",
                    "boundary_verification_defer_min_failure_score": 6,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertIn((1.2, False), calls)
        self.assertIn((1.2, True), calls)
        verified = next(
            row for row in trials if row.get("factor") == 1.2 and row.get("boundary_verification")
        )
        self.assertEqual(verified["boundary_verification_trigger"], "weak_provisional_failure")
        self.assertTrue(verified["ok"])
        self.assertAlmostEqual(fos, 1.2)
        self.assertEqual(info["auto"]["boundary_verification_recovery_count"], 0)
        self.assertEqual(info["auto"]["boundary_verification_stable_reversal_count"], 1)

    def test_srm_auto_does_not_move_failure_boundary_on_indeterminate_trial(self) -> None:
        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor <= 1.0:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            confirmed = factor >= 1.4
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.97 if confirmed else 0.82,
                    "last_accepted_plastic_ratio": 0.78 if confirmed else 0.25,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8 if confirmed else 2,
                    "plastic_cluster_spans_boundary": confirmed,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                    "final_step_size": 1.0e-4,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "upper_branch": {"factor_max": 1.4, "factor_step": 0.2},
                "factor_tol": 0.05,
                "max_bisection": 3,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "max_suspect_retries": 0,
                    "boundary_verification_enabled": True,
                },
            },
            0.0,
            fake_trial,
        )

        indeterminate = [row for row in trials if row.get("factor") == 1.2]
        self.assertTrue(indeterminate)
        self.assertTrue(all(row.get("srm_trial_state") == "indeterminate" for row in indeterminate))
        self.assertEqual(fos, 1.0)
        self.assertEqual(info["stable_factor"], 1.0)
        self.assertEqual(info["failed_factor"], 1.4)
        self.assertAlmostEqual(info["factor_of_safety_interval"]["width"], 0.4)
        self.assertTrue(info["factor_of_safety_boundary_certified"])
        self.assertFalse(info["factor_of_safety_tolerance_met"])
        self.assertFalse(info["factor_of_safety_certified"])
        self.assertEqual(
            info["factor_of_safety_status"],
            "unresolved_indeterminate_interval",
        )
        self.assertEqual(
            info["factor_of_safety_value_kind"],
            "unresolved_interval_lower_bound",
        )
        self.assertEqual(info["auto"]["indeterminate_search_factors"], [1.2])

    def test_srm_auto_factor_tol_rejects_solver_only_failure_boundary(
        self,
    ) -> None:
        calls: list[tuple[float, bool]] = []

        def fake_trial(
            factor: float,
            *,
            solver_override: Mapping[str, Any] | None = None,
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            verification = isinstance(
                solver_override, Mapping
            ) and isinstance(
                solver_override.get("_srm_boundary_verification"), Mapping
            )
            calls.append((factor, verification))
            if factor <= 1.2:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {
                        "converged": True,
                        "residual_norm": 1.0e-8,
                        "iterations": 2,
                    },
                )
            confirmed = factor >= 1.4
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": (
                        0.97 if confirmed else 0.82
                    ),
                    "last_accepted_plastic_ratio": (
                        0.78 if confirmed else 0.25
                    ),
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": (
                        8 if confirmed else 2
                    ),
                    "plastic_cluster_spans_boundary": confirmed,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                    "final_step_size": 1.0e-4,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "upper_branch": {
                    "factor_max": 1.4,
                    "factor_step": 0.2,
                },
                "factor_tol": 0.05,
                "max_bisection": 4,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "max_suspect_retries": 0,
                    "boundary_verification_enabled": True,
                    "factor_tol_enforcement_enabled": True,
                    "factor_tol_enforcement_accept_verified_numerical_failure": True,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertAlmostEqual(fos, 1.2)
        self.assertAlmostEqual(
            info["factor_of_safety_interval"]["width"], 0.2
        )
        self.assertFalse(info["factor_of_safety_tolerance_met"])
        self.assertEqual(
            info["auto"]["factor_tol_numerical_failure_boundary_count"],
            0,
        )
        rejected = [
            row
            for row in trials
            if row.get("factor_tol_numerical_failure_rejected")
        ]
        self.assertTrue(rejected)
        self.assertTrue(
            all(
                row.get("factor_tol_enforcement_reason")
                == "no_boundary_spanning_plastic_cluster"
                for row in rejected
            )
        )
        self.assertTrue(any(verification for _factor, verification in calls))

    def test_srm_auto_factor_tol_enforcement_extends_bisection_budget(
        self,
    ) -> None:
        def fake_trial(factor: float) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor <= 1.2:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True},
                )
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": 0.98,
                    "last_accepted_plastic_ratio": 0.80,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                },
            )

        _result, _fos, _trials, info = _run_srm_trial_search(
            [1.0, 1.4],
            {
                "search_mode": "auto",
                "factors": [1.0, 1.4],
                "factor_tol": 0.05,
                "max_bisection": 0,
                "auto": {
                    "enabled": True,
                    "factor_tol_enforcement_enabled": True,
                    "factor_tol_enforcement_max_extra_bisections": 4,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertLessEqual(
            info["factor_of_safety_interval"]["width"],
            0.05 + 1.0e-12,
        )
        self.assertTrue(info["factor_of_safety_tolerance_met"])
        self.assertGreaterEqual(
            info["auto"]["factor_tol_enforcement_extra_bisections_used"],
            1,
        )

    def test_srm_case2_factor_tol_keeps_verified_suspect_gap_uncertified(
        self,
    ) -> None:
        def fake_trial(
            factor: float,
            *,
            solver_override: Mapping[str, Any] | None = None,
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor <= 1.33125:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True},
                )
            confirmed = factor >= 1.3375
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": (
                        0.97 if confirmed else 0.80
                    ),
                    "last_accepted_plastic_ratio": (
                        0.80 if confirmed else 0.25
                    ),
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": (
                        8 if confirmed else 2
                    ),
                    "plastic_cluster_spans_boundary": confirmed,
                    "cutback_count": 14,
                    "max_cutbacks": 14,
                },
            )

        _result, fos, _trials, info = _run_srm_trial_search(
            [1.3, 1.33125, 1.3375],
            {
                "search_mode": "auto",
                "factors": [1.3, 1.33125, 1.3375],
                "factor_tol": 0.005,
                "max_bisection": 1,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "max_suspect_retries": 0,
                    "boundary_verification_enabled": True,
                    "boundary_verification_strategy": "deferred_final",
                    "factor_tol_enforcement_enabled": True,
                    "factor_tol_enforcement_accept_verified_numerical_failure": True,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertAlmostEqual(fos, 1.33125)
        self.assertAlmostEqual(info["stable_factor"], 1.33125)
        self.assertAlmostEqual(info["failed_factor"], 1.3375)
        self.assertAlmostEqual(
            info["factor_of_safety_interval"]["width"], 0.00625
        )
        self.assertFalse(info["factor_of_safety_tolerance_met"])
        self.assertEqual(
            info["auto"][
                "factor_tol_numerical_failure_boundary_factors"
            ],
            [],
        )

    def test_srm_boundary_verification_is_preserved_in_structured_log(self) -> None:
        srm = {
            "factor_of_safety": 1.2,
            "stable_factor": 1.2,
            "failed_factor": 1.25,
            "factor_of_safety_status": "confirmed_bracket",
            "factor_of_safety_confidence": "high",
            "factor_of_safety_interval": {"stable": 1.2, "failed": 1.25, "width": 0.05},
            "factor_of_safety_boundary_certified": True,
            "factor_of_safety_tolerance_met": True,
            "factor_of_safety_certified": True,
            "factor_of_safety_value_kind": "certified_stable_lower_bound",
            "boundary_quality": "verified_failure_boundary",
            "boundary_verified": True,
            "bracket_resolved": True,
            "indeterminate_factors_inside_bracket": [],
            "auto": {
                "boundary_verification_strategy": "deferred_final",
                "boundary_verification_defer_min_failure_score": 6,
                "boundary_verification_deferred_count": 3,
                "boundary_verification_executed_count": 1,
                "boundary_verification_recovery_count": 0,
                "boundary_verification_stable_reversal_count": 0,
                "boundary_verification_cold_retry_on_indeterminate": True,
                "boundary_verification_cold_retry_count": 1,
                "boundary_verification_cold_retry_factors": [1.25],
                "boundary_checkpoint_continuation_enabled": True,
                "boundary_checkpoint_continuation_extra_cutbacks": 1,
                "boundary_verification_strict_tangent": True,
                "retry_strict_tangent": True,
                "boundary_checkpoint_continuation_requested_count": 1,
                "boundary_checkpoint_continuation_used_count": 1,
                "boundary_checkpoint_fallback_count": 0,
                "factor_tol_enforcement_enabled": True,
                "factor_tol_enforcement_extra_bisections_used": 1,
                "factor_tol_numerical_failure_boundary_count": 1,
                "factor_tol_numerical_failure_boundary_factors": [1.25],
            },
            "parallel": {
                "enabled": True,
                "decision_linked_cancellation_enabled": True,
                "decision_linked_requested_count": 2,
                "decision_linked_pending_cancel_count": 1,
                "decision_linked_safe_stop_count": 1,
                "decision_linked_completed_after_request_count": 0,
                "decision_linked_requested_factors": [1.3, 1.4],
                "cost_aware_lookahead_enabled": True,
                "cost_aware_depth_limited_count": 2,
                "cost_aware_asymmetric_bisection_count": 1,
                "cost_aware_deferred_candidate_count": 3,
                "cost_aware_observation": {
                    "sample_count": 4,
                    "stable_median_elapsed_seconds": 20.0,
                    "failed_median_elapsed_seconds": 100.0,
                    "failure_to_stable_cost_ratio": 5.0,
                    "reason": "failure_cost_ratio",
                },
            },
            "trials": [
                {
                    "factor": 1.25,
                    "ok": False,
                    "converged": False,
                    "srm_trial_state": "confirmed_failure",
                    "boundary_verification": True,
                    "boundary_verification_reason": "suspect_failure",
                    "boundary_verification_result": "confirmed_failure",
                    "boundary_checkpoint_continuation_requested": True,
                    "boundary_checkpoint_continuation_used": True,
                    "increment_checkpoint_reused_history_rows": 8,
                    "factor_tol_numerical_failure_boundary": True,
                    "factor_tol_enforcement_original_state": "indeterminate",
                    "factor_tol_enforcement_reason": "verified_nonconvergence",
                    "mc_apex_regularization_count": 3,
                    "mc_regularization_method": "bounded_sequential_cone_tip",
                    "mc_constitutive_model_fidelity": False,
                }
            ],
        }
        stage = StageResult2D(
            "srm",
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            [],
            {},
            [],
            {"srm": srm},
        )
        solve = SolveResult2D(
            Mesh2D([], np.zeros((0, 2), dtype=float), []), {}, [stage], Path(".")
        )
        events = build_structured_analysis_log(solve)["events"]
        summary = next(row for row in events if row.get("event_type") == "srm_summary")
        trial = next(row for row in events if row.get("event_type") == "srm_trial")
        self.assertTrue(summary["factor_of_safety_certified"])
        self.assertEqual(summary["boundary_quality"], "verified_failure_boundary")
        self.assertEqual(summary["srm_boundary_verification_strategy"], "deferred_final")
        self.assertEqual(summary["srm_boundary_verification_deferred_count"], 3)
        self.assertEqual(summary["srm_boundary_verification_executed_count"], 1)
        self.assertEqual(
            summary["srm_boundary_checkpoint_continuation_used_count"], 1
        )
        self.assertEqual(
            summary["srm_boundary_checkpoint_continuation_extra_cutbacks"], 1
        )
        self.assertTrue(summary["srm_boundary_verification_strict_tangent"])
        self.assertTrue(
            summary["srm_boundary_verification_cold_retry_on_indeterminate"]
        )
        self.assertEqual(summary["srm_boundary_verification_cold_retry_count"], 1)
        self.assertEqual(
            summary["srm_boundary_verification_cold_retry_factors"], [1.25]
        )
        self.assertTrue(summary["bracket_resolved"])
        self.assertEqual(
            summary["srm_parallel_decision_linked_safe_stop_count"], 1
        )
        self.assertTrue(summary["srm_parallel_cost_aware_lookahead_enabled"])
        self.assertEqual(
            summary["srm_parallel_cost_aware_deferred_candidate_count"], 3
        )
        self.assertEqual(
            summary["srm_parallel_cost_aware_failure_to_stable_cost_ratio"],
            5.0,
        )
        self.assertTrue(summary["srm_factor_tol_enforcement_enabled"])
        self.assertEqual(
            summary["srm_factor_tol_numerical_failure_boundary_count"], 1
        )
        self.assertEqual(trial["srm_trial_state"], "confirmed_failure")
        self.assertTrue(trial["boundary_verification"])
        self.assertEqual(trial["boundary_verification_result"], "confirmed_failure")
        self.assertTrue(trial["boundary_checkpoint_continuation_used"])
        self.assertEqual(trial["increment_checkpoint_reused_history_rows"], 8)
        self.assertTrue(trial["factor_tol_numerical_failure_boundary"])
        self.assertEqual(
            trial["factor_tol_enforcement_original_state"], "indeterminate"
        )
        self.assertEqual(trial["mc_apex_regularization_count"], 3)
        self.assertEqual(
            trial["mc_regularization_method"], "bounded_sequential_cone_tip"
        )
        self.assertFalse(trial["mc_constitutive_model_fidelity"])

    def test_case1_to_case4_phase6_yaml_share_guarded_accuracy_contract(self) -> None:
        case_dir = (
            Path(__file__).resolve().parents[1]
            / "dist"
            / "sustainability_2024_case1-4_auto_srm_speed_guarded_20260612"
        )
        paths = sorted(
            case_dir.glob(
                "sustainability_2024_case*_quad4_sri_auto_srm_speed_guarded.yaml"
            )
        )
        self.assertEqual(len(paths), 4)
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(len(cfg["stages"]), 1, path.name)
            stage = cfg["stages"][0]
            self.assertEqual(stage["type"], "srm", path.name)
            srm = stage["srm"]
            self.assertEqual(srm["search_mode"], "auto", path.name)
            self.assertEqual(float(srm["failure_plastic_ratio"]), 1.0, path.name)
            self.assertTrue(srm["lightweight_postprocess"], path.name)
            self.assertTrue(srm["parallel"]["preserve_decision_order"], path.name)
            self.assertTrue(
                srm["parallel"]["decision_linked_cancellation"], path.name
            )
            self.assertEqual(srm["parallel"]["executor"], "process", path.name)
            self.assertTrue(srm["parallel"]["bisection_speculation"], path.name)
            self.assertEqual(int(srm["parallel"]["lookahead_depth"]), 3, path.name)
            self.assertTrue(srm["parallel"]["cost_aware_lookahead"], path.name)
            self.assertTrue(
                srm["parallel"]["event_driven_cost_cancellation"],
                path.name,
            )
            self.assertEqual(
                int(srm["parallel"]["cost_aware_min_elapsed_seconds"]),
                30,
                path.name,
            )
            self.assertEqual(
                float(srm["parallel"]["cost_aware_failure_ratio"]),
                3.0,
                path.name,
            )
            auto = srm["auto"]
            self.assertFalse(auto["retry_suspect_failures"], path.name)
            self.assertEqual(int(auto["max_suspect_retries"]), 0, path.name)
            self.assertTrue(auto["boundary_verification_enabled"], path.name)
            self.assertTrue(
                auto["boundary_checkpoint_continuation_enabled"], path.name
            )
            self.assertEqual(
                int(auto["boundary_checkpoint_continuation_extra_cutbacks"]),
                1,
                path.name,
            )
            self.assertTrue(
                auto["boundary_verification_strict_tangent"],
                path.name,
            )
            self.assertTrue(
                auto["boundary_verification_cold_retry_on_indeterminate"],
                path.name,
            )
            self.assertEqual(
                int(auto["boundary_verification_cold_retry_max_per_factor"]),
                1,
                path.name,
            )
            self.assertTrue(auto["factor_tol_enforcement_enabled"], path.name)
            self.assertFalse(
                auto[
                    "factor_tol_enforcement_accept_verified_numerical_failure"
                ],
                path.name,
            )
            self.assertTrue(
                auto["factor_tol_require_physical_failure_evidence"],
                path.name,
            )
            self.assertTrue(
                auto["boundary_checkpoint_residual_prediction_enabled"],
                path.name,
            )
            self.assertEqual(
                int(
                    auto[
                        "boundary_checkpoint_residual_prediction_max_extra_cutbacks"
                    ]
                ),
                4,
                path.name,
            )
            self.assertEqual(
                int(auto["factor_tol_enforcement_max_extra_bisections"]),
                8,
                path.name,
            )
            self.assertEqual(auto["boundary_verification_strategy"], "deferred_final", path.name)
            self.assertEqual(int(auto["boundary_verification_max_recoveries"]), 4, path.name)
            self.assertEqual(int(auto["boundary_verification_defer_min_failure_score"]), 6, path.name)
            self.assertTrue(auto["boundary_verification_suspect"], path.name)
            self.assertTrue(auto["boundary_verification_early_failure"], path.name)
            self.assertEqual(int(auto["boundary_verification_extra_cutbacks"]), 2, path.name)
            self.assertTrue(auto["early_failure_strong_collapse_enabled"], path.name)
            self.assertEqual(int(auto["early_failure_strong_min_cutbacks"]), 4, path.name)
            self.assertEqual(
                float(auto["early_failure_strong_min_cutback_ratio"]),
                0.50,
                path.name,
            )
            self.assertEqual(float(auto["early_failure_strong_max_last_load"]), 0.85, path.name)
            self.assertEqual(float(auto["early_failure_strong_cluster_fraction"]), 0.80, path.name)
            self.assertEqual(float(auto["early_failure_strong_plastic_ratio"]), 0.70, path.name)
            self.assertEqual(
                float(auto["early_failure_strong_residual_reduction_min"]),
                0.90,
                path.name,
            )
            self.assertTrue(auto["early_failure_strong_require_boundary_span"], path.name)
            active_set = stage["solver"]["newton"][
                "mohr_coulomb_active_set_update"
            ]
            self.assertTrue(active_set["tangent_reuse_enabled"], path.name)
            self.assertTrue(
                active_set["direct_consistent_tangent_enabled"], path.name
            )
            self.assertTrue(
                active_set["adaptive_numerical_tangent_enabled"], path.name
            )
            self.assertTrue(
                active_set["line_search_invalidation_enabled"],
                path.name,
            )
            self.assertTrue(
                active_set["regularized_projection_invalidation_enabled"],
                path.name,
            )
            self.assertEqual(
                int(active_set["regularized_projection_invalidation_min_count"]),
                32,
                path.name,
            )
            self.assertAlmostEqual(
                float(active_set["regularized_projection_invalidation_fraction"]),
                0.05,
                msg=path.name,
            )
            self.assertTrue(
                active_set["active_set_miss_invalidation_enabled"],
                path.name,
            )
            self.assertEqual(
                int(active_set["active_set_miss_invalidation_min_attempts"]),
                32,
                path.name,
            )
            self.assertAlmostEqual(
                float(active_set["active_set_miss_invalidation_fraction"]),
                0.35,
                msg=path.name,
            )
            line_search_batch = stage["solver"]["newton"][
                "line_search_batch"
            ]
            self.assertTrue(line_search_batch["enabled"], path.name)
            self.assertEqual(
                int(line_search_batch["chunk_size"]), 4, path.name
            )
            self.assertEqual(
                int(active_set["line_search_invalidation_threshold"]),
                4,
                path.name,
            )

    def test_real_small_srm_serial_and_parallel_search_are_numerically_equivalent(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [
                    {
                        "id": "e1",
                        "type": "QUAD4",
                        "nodes": ["1", "2", "3", "4"],
                        "material": "soil",
                        "integration": "SRI",
                    }
                ],
            },
            "materials": {
                "soil": {"model": "von_mises", "E": 12000.0, "nu": 0.30, "yield_stress": 2.0}
            },
            "boundary_conditions": [
                {"nodes": ["1", "4"], "ux": 0.0},
                {"nodes": ["1", "2"], "uy": 0.0},
            ],
            "loads": [{"node": "3", "fy": -0.5}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        base_solver = {
            "newton": {"max_iter": 30},
            "linear": {"cache_factorization": False},
        }
        process_spec = fem2d_solver_module._srm_plane_process_trial_spec(
            mesh=mesh,
            materials=materials,
            boundary_conditions=cfg["boundary_conditions"],
            loads=cfg["loads"],
            mpc_constraints=None,
            stage_name="parity",
            solver=base_solver,
            srm_cfg={},
            initial_stresses=None,
            interfaces=None,
            structural_elements=None,
            plastic_state=None,
            lightweight_trials=True,
            large_deformation_trials=False,
        )

        def run(
            parallel: Mapping[str, Any],
        ) -> tuple[StageResult2D | None, float, list[dict[str, Any]], dict[str, Any]]:
            step_cache = build_small_deformation_step_cache(
                mesh, materials, cfg["boundary_conditions"]
            )

            def solve_trial(
                factor: float, *, solver_override: Mapping[str, Any] | None = None
            ) -> StageResult2D:
                return solve_plane_strain_stage(
                    mesh=mesh,
                    materials=materials,
                    boundary_conditions=cfg["boundary_conditions"],
                    loads=cfg["loads"],
                    stage_name=f"parity-FS{factor:g}",
                    output_dir=None,
                    solver=base_solver,
                    strength_factor=factor,
                    step_cache=step_cache,
                    postprocess_results=False,
                )

            return _run_srm_trial_search(
                [1.0, 1.5, 2.0],
                {
                    "search_mode": "adaptive_bracket",
                    "anchor_factor": 1.0,
                    "upper_branch": {"factor_max": 2.0, "factor_step": 0.5},
                    "factor_tol": 0.125,
                    "max_bisection": 3,
                    "parallel": dict(parallel),
                },
                0.0,
                solve_trial,
                mesh=mesh,
                process_trial_spec=process_spec,
            )

        serial = run({"enabled": False, "max_workers": 1})
        parallel = run(
            {
                "enabled": True,
                "max_workers": 2,
                "strategy": "lookahead",
                "preserve_decision_order": True,
                "lookahead_depth": 2,
                "bisection_speculation": False,
            }
        )
        process = run(
            {
                "enabled": True,
                "max_workers": 2,
                "executor": "process",
                "strategy": "lookahead",
                "preserve_decision_order": True,
                "lookahead_depth": 2,
                "bisection_speculation": False,
            }
        )
        self.assertAlmostEqual(serial[1], parallel[1], places=12)
        self.assertAlmostEqual(serial[1], process[1], places=12)
        self.assertEqual(serial[3]["stable_factor"], parallel[3]["stable_factor"])
        self.assertEqual(serial[3]["failed_factor"], parallel[3]["failed_factor"])
        self.assertEqual(serial[3]["stable_factor"], process[3]["stable_factor"])
        self.assertEqual(serial[3]["failed_factor"], process[3]["failed_factor"])
        self.assertEqual(process[3]["parallel"]["effective_executor"], "process")
        self.assertEqual(process[3]["parallel"]["process_executor_fallback_count"], 0)
        self.assertIsNotNone(serial[0])
        self.assertIsNotNone(parallel[0])
        self.assertIsNotNone(process[0])
        assert serial[0] is not None and parallel[0] is not None and process[0] is not None
        self.assertTrue(
            np.allclose(
                serial[0].displacements,
                parallel[0].displacements,
                rtol=1.0e-11,
                atol=1.0e-12,
            )
        )
        self.assertTrue(
            np.allclose(
                serial[0].displacements,
                process[0].displacements,
                rtol=1.0e-11,
                atol=1.0e-12,
            )
        )

    def test_real_small_srm_process_executor_matches_serial_trials(self) -> None:
        cfg = {
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [
                    {
                        "id": "e1",
                        "type": "QUAD4",
                        "nodes": ["1", "2", "3", "4"],
                        "material": "soil",
                        "integration": "SRI",
                    }
                ],
            },
            "materials": {
                "soil": {
                    "model": "mohr_coulomb",
                    "E": 12000.0,
                    "nu": 0.30,
                    "cohesion": 100.0,
                    "friction_angle": 30.0,
                    "dilation_angle": 0.0,
                }
            },
            "boundary_conditions": [
                {"nodes": ["1", "4"], "ux": 0.0},
                {"nodes": ["1", "2"], "uy": 0.0},
            ],
            "loads": [{"node": "3", "fy": -0.5}],
        }
        mesh = mesh_from_config(cfg)
        materials = plane_strain_materials(cfg)
        solver = {"newton": {"max_iter": 20}, "linear": {"cache_factorization": False}}
        process_spec = fem2d_solver_module._srm_plane_process_trial_spec(
            mesh=mesh,
            materials=materials,
            boundary_conditions=cfg["boundary_conditions"],
            loads=cfg["loads"],
            mpc_constraints=None,
            stage_name="process-parity",
            solver=solver,
            srm_cfg={},
            initial_stresses=None,
            interfaces=None,
            structural_elements=None,
            plastic_state=None,
            lightweight_trials=True,
            large_deformation_trials=False,
        )

        def solve_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            return solve_plane_strain_stage(
                mesh=mesh,
                materials=materials,
                boundary_conditions=cfg["boundary_conditions"],
                loads=cfg["loads"],
                stage_name=f"serial-FS{factor:g}",
                output_dir=None,
                solver=_srm_solver_with_retry_override(solver, solver_override),
                strength_factor=factor,
                postprocess_results=False,
            )

        serial = [
            fem2d_solver_module._srm_trial_record(factor, 1.0, solve_trial)
            for factor in (1.0, 1.1)
        ]
        stats: dict[str, Any] = {}
        process = _srm_evaluate_records_parallel(
            [1.0, 1.1],
            1.0,
            solve_trial,
            2,
            executor_kind="process",
            process_trial_spec=process_spec,
            cancellation_stats=stats,
        )

        self.assertEqual(stats.get("executor"), "process")
        self.assertFalse(stats.get("process_executor_fallback", False), stats)
        self.assertEqual([row["ok"] for row in serial], [row["ok"] for row in process])
        self.assertEqual([row["factor"] for row in serial], [row["factor"] for row in process])
        self.assertTrue(all(row.get("_parallel_executor") == "process" for row in process))
        for serial_row, process_row in zip(serial, process):
            serial_result = serial_row.get("result")
            process_result = process_row.get("result")
            self.assertIsInstance(serial_result, StageResult2D)
            self.assertIsInstance(process_result, StageResult2D)
            assert isinstance(serial_result, StageResult2D) and isinstance(process_result, StageResult2D)
            self.assertTrue(
                np.allclose(
                    serial_result.displacements,
                    process_result.displacements,
                    rtol=1.0e-11,
                    atol=1.0e-12,
                )
            )

    def test_srm_auto_passes_early_failure_policy_to_trials(self) -> None:
        overrides: list[Mapping[str, Any] | None] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            overrides.append(solver_override)
            if factor <= 1.0:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            raise FEM2DError(
                "srm early confirmed failure",
                diagnostics={
                    "trial_status": "srm_early_confirmed_failure",
                    "last_accepted_load_factor": 0.95,
                    "last_accepted_plastic_ratio": 0.75,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 7,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 9,
                    "max_cutbacks": 12,
                    "final_step_size": 1.0e-4,
                    "early_failure_stop": True,
                    "early_failure_score": 6,
                },
            )

        _result, _fos, trials, info = _run_srm_trial_search(
            [1.0, 1.2, 1.4],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "upper_branch": {"factor_max": 1.4, "factor_step": 0.2},
                "factor_tol": 0.05,
                "auto": {"enabled": True, "retry_suspect_failures": False, "early_failure_stop_enabled": True},
            },
            0.0,
            fake_trial,
        )

        self.assertTrue(overrides)
        self.assertTrue(all(isinstance(item, Mapping) and "_srm_early_failure_policy" in item for item in overrides))
        self.assertTrue(info["auto"]["early_failure_stop_enabled"])
        failed = next(row for row in trials if not row["ok"])
        self.assertEqual(failed["trial_status"], "srm_early_confirmed_failure")
        self.assertTrue(failed["early_failure_stop"])

    def test_srm_boundary_retry_extends_checkpoint_budget_and_uses_strict_tangent(self) -> None:
        checkpoint = fem2d_solver_module._IncrementContinuationCheckpoint(
            schema="geofem.srm_increment_checkpoint.v1",
            fingerprint="test",
            strength_factor=1.33125,
            source_stage_name="FS1.33125",
            source_status="increment_cutback_limit",
            target=0.95,
            next_step_size=1.0e-5,
            accepted_steps=20,
            cutbacks=15,
            log=(),
            plastic_state={},
            plastic_state_cache=None,
            displacement=np.zeros(2, dtype=float),
        )
        adjusted = _srm_solver_with_retry_override(
            {
                "newton": {"max_iter": 80},
                "increments": {
                    "steps": 20,
                    "max_cutbacks": 12,
                    "min_step": 1.0e-5,
                },
            },
            {
                "_srm_increment_checkpoint": checkpoint,
                "_srm_mc_strict_tangent": True,
                "_srm_retry_policy": {
                    "extra_cutbacks": 2,
                    "checkpoint_continuation_extra_cutbacks": 1,
                    "min_step_factor": 0.5,
                },
            },
        )

        self.assertIsInstance(adjusted, Mapping)
        assert isinstance(adjusted, Mapping)
        self.assertTrue(adjusted["_srm_mc_strict_tangent"])
        self.assertEqual(adjusted["increments"]["max_cutbacks"], 16)
        self.assertLess(adjusted["increments"]["min_step"], 1.0e-5)

    def test_srm_cost_aware_lookahead_limits_expensive_failure_side(self) -> None:
        heavy_initial = fem2d_solver_module._srm_cost_aware_lookahead_snapshot(
            {
                1.0: {
                    "factor": 1.0,
                    "ok": True,
                    "elapsed_seconds": 120.0,
                }
            },
            enabled=True,
            min_elapsed_seconds=30.0,
            failure_ratio_threshold=3.0,
            escalation_ratio_threshold=4.0,
            min_samples=2,
        )
        expensive_failure = (
            fem2d_solver_module._srm_cost_aware_lookahead_snapshot(
                {
                    1.0: {
                        "factor": 1.0,
                        "ok": True,
                        "elapsed_seconds": 20.0,
                    },
                    1.2: {
                        "factor": 1.2,
                        "ok": True,
                        "elapsed_seconds": 25.0,
                    },
                    1.4: {
                        "factor": 1.4,
                        "ok": False,
                        "elapsed_seconds": 180.0,
                    },
                },
                enabled=True,
                min_elapsed_seconds=30.0,
                failure_ratio_threshold=3.0,
                escalation_ratio_threshold=4.0,
                min_samples=2,
            )
        )

        self.assertEqual(heavy_initial["recommended_depth"], 2)
        self.assertEqual(
            heavy_initial["reason"],
            "expensive_initial_trial",
        )
        self.assertEqual(expensive_failure["recommended_depth"], 1)
        self.assertTrue(expensive_failure["failure_side_expensive"])
        self.assertGreater(
            expensive_failure["failure_to_stable_cost_ratio"],
            3.0,
        )

    def test_srm_event_driven_cost_shrink_stops_distant_trials(self) -> None:
        expensive_trial_started = threading.Event()

        def fake_trial(
            factor: float,
            *,
            solver_override: Mapping[str, Any] | None = None,
        ) -> StageResult2D:
            factor = round(float(factor), 6)
            if factor == 1.0:
                time.sleep(0.01)
                plastic = 0.0
            elif factor == 1.1:
                expensive_trial_started.set()
                time.sleep(0.08)
                plastic = 0.0
            else:
                self.assertTrue(expensive_trial_started.wait(timeout=1.0))
                for _ in range(200):
                    fem2d_solver_module._raise_if_solver_cancel_requested(
                        solver_override,
                        f"FS{factor:g}",
                        "increment_start",
                    )
                    time.sleep(0.002)
                plastic = 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                [],
                {"converged": True},
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.1, 1.2, 1.3],
            {
                "factors": [1.0, 1.1, 1.2, 1.3],
                "adaptive": True,
                "bracket_stride": 1,
                "factor_tol": 0.05,
                "max_bisection": 0,
                "parallel": {
                    "enabled": True,
                    "max_workers": 3,
                    "strategy": "lookahead",
                    "preserve_decision_order": True,
                    "decision_linked_cancellation": True,
                    "cost_aware_lookahead": True,
                    "event_driven_cost_cancellation": True,
                    "cost_aware_min_elapsed_seconds": 0.02,
                    "cost_aware_escalation_ratio": 3.0,
                    "cost_aware_min_samples": 2,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(fos, 1.1)
        self.assertEqual(info["stable_factor"], 1.1)
        self.assertEqual(info["failed_factor"], 1.2)
        self.assertEqual([row["factor"] for row in trials], [1.0, 1.1, 1.2])
        parallel = info["parallel"]
        self.assertGreaterEqual(parallel["event_driven_cost_shrink_count"], 1)
        self.assertGreaterEqual(
            parallel["event_driven_cost_cancel_candidate_count"], 1
        )
        self.assertGreaterEqual(parallel["decision_linked_safe_stop_count"], 1)

    def test_srm_verified_failure_requires_physical_evidence(self) -> None:
        settings = fem2d_solver_module._srm_auto_settings(
            {"search_mode": "auto", "auto": {"enabled": True}}
        )
        solver_only = {
            "failure_reason": "nonconvergence",
            "active_element_count": 100,
            "connected_plastic_cluster_size": 8,
            "plastic_cluster_spans_boundary": False,
            "plastic_ratio": 0.20,
            "cutback_count": 12,
            "max_cutbacks": 12,
        }
        physical = {
            **solver_only,
            "connected_plastic_cluster_size": 70,
            "plastic_cluster_spans_boundary": True,
            "plastic_ratio": 0.75,
        }

        solver_supported, solver_reason = (
            fem2d_solver_module._srm_verified_failure_has_physical_evidence(
                solver_only, settings
            )
        )
        physical_supported, physical_reason = (
            fem2d_solver_module._srm_verified_failure_has_physical_evidence(
                physical, settings
            )
        )

        self.assertFalse(solver_supported)
        self.assertEqual(
            solver_reason, "no_boundary_spanning_plastic_cluster"
        )
        self.assertTrue(physical_supported)
        self.assertEqual(physical_reason, "plastic_failure_evidence")

    def test_srm_checkpoint_residual_prediction_adds_only_productive_cutbacks(
        self,
    ) -> None:
        settings = fem2d_solver_module._srm_auto_settings(
            {"search_mode": "auto", "auto": {"enabled": True}}
        )
        decaying = fem2d_solver_module._srm_checkpoint_residual_prediction(
            {
                "last_accepted_residual_norm": 1.0e-4,
                "increment_log_tail": [
                    {"accepted": False, "residual_norm_final": 1.0e-1},
                    {"accepted": False, "residual_norm_final": 2.0e-2},
                    {"accepted": False, "residual_norm_final": 4.0e-3},
                ],
            },
            settings,
        )
        plateau = fem2d_solver_module._srm_checkpoint_residual_prediction(
            {
                "last_accepted_residual_norm": 1.0e-4,
                "increment_log_tail": [
                    {"accepted": False, "residual_norm_final": 1.0e-2},
                    {"accepted": False, "residual_norm_final": 1.1e-2},
                ],
            },
            settings,
        )

        self.assertEqual(decaying["reason"], "predictive_residual_decay")
        self.assertEqual(decaying["recommended_extra_cutbacks"], 4)
        self.assertEqual(plateau["reason"], "residual_plateau_or_growth")
        self.assertEqual(plateau["recommended_extra_cutbacks"], 1)

    def test_srm_solver_override_applies_adaptive_increment_policy(self) -> None:
        solver = {
            "increments": {"steps": 4, "max_cutbacks": 6, "min_step": 1.0e-5},
        }
        adjusted = _srm_solver_with_retry_override(
            solver,
            {
                "_srm_adaptive_increment_policy": {
                    "enabled": True,
                    "source_factor": 1.0,
                    "target_factor": 0.9,
                    "target_initial_step_factor": 0.5,
                    "min_initial_step_factor": 0.25,
                    "max_initial_step_factor": 1.0,
                    "max_steps_multiplier": 3.0,
                    "extra_cutbacks": 2,
                    "min_step_factor": 0.5,
                    "use_final_step_size": False,
                }
            },
        )

        self.assertIsInstance(adjusted, Mapping)
        assert isinstance(adjusted, Mapping)
        increments = adjusted["increments"]
        self.assertEqual(increments["steps"], 8)
        self.assertEqual(increments["max_cutbacks"], 8)
        self.assertAlmostEqual(increments["min_step"], 5.0e-6)
        self.assertTrue(adjusted["_srm_adaptive_increment_control"]["enabled"])
        self.assertEqual(adjusted["_srm_adaptive_increment_control"]["original_steps"], 4)
        self.assertEqual(adjusted["_srm_adaptive_increment_control"]["applied_steps"], 8)

    def test_srm_auto_adaptive_increment_uses_previous_failure_log(self) -> None:
        overrides: list[tuple[float, Mapping[str, Any] | None]] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            overrides.append((factor, solver_override))
            if factor >= 0.99:
                raise FEM2DError(
                    "increment cutback limit reached",
                    diagnostics={
                        "trial_status": "increment_cutback_limit",
                        "last_accepted_load_factor": 0.72,
                        "last_accepted_plastic_ratio": 0.55,
                        "active_element_count": 10,
                        "connected_plastic_cluster_size": 6,
                        "plastic_cluster_spans_boundary": True,
                        "cutback_count": 12,
                        "max_cutbacks": 12,
                        "final_step_size": 0.02,
                        "residual_reduction_ratio": 0.9,
                    },
                )
            return StageResult2D(
                f"FS{factor:g}",
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                [{"active": 1.0, "plastic": 0.0}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "lower_branch": {"factor_min": 0.8, "factor_step": 0.1},
                "upper_branch": {"factor_max": 1.2, "factor_step": 0.1},
                "bracket_stride": 1,
                "factor_tol": 0.05,
                "max_bisection": 0,
                "auto": {
                    "enabled": True,
                    "retry_suspect_failures": False,
                    "lower_projection_enabled": False,
                },
                "adaptive_increment_control": {
                    "enabled": True,
                    "min_last_load": 0.0,
                },
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(info["search_mode"], "auto")
        self.assertEqual(info["scan_direction"], "lower")
        self.assertEqual(fos, 0.9)
        self.assertEqual([row["factor"] for row in trials], [1.0, 0.9])
        lower_override = next(override for factor, override in overrides if factor == 0.9)
        self.assertIsInstance(lower_override, Mapping)
        assert isinstance(lower_override, Mapping)
        self.assertIn("_srm_adaptive_increment_policy", lower_override)
        policy = lower_override["_srm_adaptive_increment_policy"]
        self.assertEqual(policy["source_factor"], 1.0)
        self.assertEqual(policy["target_factor"], 0.9)
        self.assertLess(policy["target_initial_step_factor"], 1.0)
        self.assertIn("cutback_ratio", policy["reason"])
        lower_trial = trials[1]
        self.assertTrue(lower_trial["adaptive_increment_control"])
        self.assertEqual(lower_trial["adaptive_increment_source_factor"], 1.0)
        self.assertEqual(lower_trial["adaptive_increment_target_factor"], 0.9)

    def test_srm_warm_start_passes_previous_stable_displacement_to_next_factor(self) -> None:
        calls: list[tuple[float, np.ndarray | None]] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            factor = round(float(factor), 6)
            warm = solver_override.get("_srm_warm_start") if isinstance(solver_override, Mapping) else None
            calls.append((factor, None if not isinstance(warm, Mapping) else np.asarray(warm.get("displacement"), dtype=float).copy()))
            plastic = 0.0 if factor <= 1.1 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.full(4, factor, dtype=float),
                np.zeros(4, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                ["e1"],
                {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
            )

        factors = [1.0, 1.1, 1.2]
        _result, fos, trials, info = _run_srm_trial_search(
            factors,
            {
                "factors": factors,
                "adaptive": True,
                "bracket_stride": 1,
                "factor_tol": 0.05,
                "max_bisection": 0,
                "warm_start": {"enabled": True, "displacement_only": True, "max_factor_distance": 0.2},
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(fos, 1.1)
        self.assertEqual([factor for factor, _warm in calls], [1.0, 1.1, 1.2])
        self.assertIsNone(calls[0][1])
        self.assertTrue(np.allclose(calls[1][1], np.full(4, 1.0)))
        self.assertTrue(np.allclose(calls[2][1], np.full(4, 1.1)))
        self.assertEqual(info["warm_start"]["used_trial_count"], 2)
        self.assertEqual(info["warm_start"]["source_factors"], [1.0, 1.1])
        self.assertTrue(trials[1]["warm_start_used"])
        self.assertEqual(trials[1]["warm_start_source_factor"], 1.0)
        self.assertEqual(trials[1]["warm_start_target_factor"], 1.1)

        stage = StageResult2D(
            "srm",
            np.zeros(4, dtype=float),
            np.zeros(4, dtype=float),
            [],
            {},
            ["e1"],
            {"srm": {"factor_of_safety": fos, "trials": trials, **info}},
        )
        solve = SolveResult2D(Mesh2D([], np.zeros((0, 2), dtype=float), []), {}, [stage], Path("."))
        events = build_structured_analysis_log(solve)["events"]
        warm_event = next(row for row in events if row.get("event_type") == "srm_trial" and row.get("srm_factor") == 1.1)
        self.assertTrue(warm_event["warm_start_used"])
        self.assertEqual(warm_event["warm_start_source_factor"], 1.0)

    def test_srm_warm_start_reports_unsupported_solver_without_applying(self) -> None:
        calls: list[Mapping[str, Any] | None] = []

        def fake_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
            calls.append(solver_override)
            return StageResult2D(
                f"FS{factor:g}",
                np.full(2, factor, dtype=float),
                np.zeros(2, dtype=float),
                [{"active": 1.0, "plastic": 0.0 if factor <= 1.0 else 1.0}],
                {},
                ["e1"],
                {"converged": True},
            )

        factors = [1.0, 1.1, 1.2]
        _result, _fos, trials, info = _run_srm_trial_search(
            factors,
            {
                "factors": factors,
                "adaptive": True,
                "bracket_stride": 1,
                "factor_tol": 0.05,
                "max_bisection": 0,
                "warm_start": {"enabled": True},
            },
            0.0,
            fake_trial,
            warm_start_supported=False,
        )

        self.assertFalse(info["warm_start"]["enabled"])
        self.assertFalse(info["warm_start"]["supported"])
        self.assertEqual(info["warm_start"]["disabled_reason"], "warm_start_not_supported_for_this_srm_solver")
        self.assertFalse(any(isinstance(override, Mapping) and "_srm_warm_start" in override for override in calls))
        self.assertFalse(any(bool(row.get("warm_start_used", False)) for row in trials))

    def test_axisymmetric_srm_warm_start_passes_displacement_to_stage(self) -> None:
        calls: list[tuple[float, np.ndarray | None, bool]] = []

        def fake_axisymmetric_stage(**kwargs: Any) -> StageResult2D:
            factor = round(float(kwargs.get("strength_factor", 1.0)), 6)
            initial = kwargs.get("initial_displacement")
            postprocess = bool(kwargs.get("postprocess_results", True))
            calls.append((factor, None if initial is None else np.asarray(initial, dtype=float).copy(), postprocess))
            plastic = 0.0 if factor <= 1.1 else 1.0
            return StageResult2D(
                f"FS{factor:g}",
                np.full(4, factor, dtype=float),
                np.zeros(4, dtype=float),
                [{"active": 1.0, "plastic": plastic}],
                {},
                ["e1"],
                {"converged": True, "geometry": "axisymmetric", "residual_norm": 1.0e-8, "iterations": 2},
            )

        mesh = Mesh2D([], np.zeros((0, 2), dtype=float), [])
        with patch.object(fem2d_solver_module, "solve_axisymmetric_stage", side_effect=fake_axisymmetric_stage):
            stage = fem2d_solver_module.solve_axisymmetric_srm_stage(
                mesh=mesh,
                materials={},
                boundary_conditions=[],
                loads=[],
                stage_name="axis-srm",
                output_dir=None,
                solver={
                    "srm": {
                        "factors": [1.0, 1.1, 1.2],
                        "adaptive": True,
                        "bracket_stride": 1,
                        "factor_tol": 0.05,
                        "max_bisection": 0,
                        "failure_plastic_ratio": 0.0,
                        "warm_start": {"enabled": True, "displacement_only": True, "max_factor_distance": 0.2},
                    }
                },
            )

        search_calls = [row for row in calls if row[2] is False]
        self.assertEqual([row[0] for row in search_calls], [1.0, 1.1, 1.2])
        self.assertIsNone(search_calls[0][1])
        self.assertTrue(np.allclose(search_calls[1][1], np.full(4, 1.0)))
        self.assertTrue(np.allclose(search_calls[2][1], np.full(4, 1.1)))
        srm = stage.solver_info["srm"]
        self.assertTrue(srm["warm_start"]["enabled"])
        self.assertTrue(srm["warm_start"]["supported"])
        self.assertEqual(srm["warm_start"]["used_trial_count"], 2)
        self.assertEqual(srm["factor_of_safety"], 1.1)

    def test_srm_early_failure_cutback_decision_scores_log_metrics(self) -> None:
        topology = _SRMTopologyDiagnosticsCache(
            active_element_ids=("e1", "e2"),
            adjacency={"e1": ("e2",), "e2": ("e1",)},
            boundary_masks={"e1": 1, "e2": 2},
            min_det_j=1.0,
        )
        states = {
            "e1:0": PlasticState2D(np.array([0.02, 0.0, 0.0, 0.0], dtype=float), 0.02),
            "e2:0": PlasticState2D(np.array([0.03, 0.0, 0.0, 0.0], dtype=float), 0.03),
        }
        exc = FEM2DError(
            "newton stalled",
            diagnostics={
                "residual_reduction_ratio": 0.95,
                "line_search_reductions_total": 24,
                "displacement_increment_norm": 0.1,
            },
        )

        decision = _srm_early_failure_cutback_decision(
            {
                "enabled": True,
                "min_cutbacks": 4,
                "min_cutback_ratio": 0.75,
                "min_last_load": 0.90,
                "score_threshold": 5,
                "cluster_fraction": 0.50,
                "plastic_ratio": 0.50,
                "residual_reduction_min": 0.80,
                "line_search_min": 20,
            },
            last_load=0.95,
            attempted_load=1.0,
            cutbacks=8,
            max_cutbacks=12,
            state_current=states,
            state_current_cache=None,
            active_elements=["e1", "e2"],
            error=exc,
            topology_cache=topology,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertTrue(decision["early_failure_stop"])
        self.assertGreaterEqual(decision["early_failure_score"], decision["early_failure_score_threshold"])
        self.assertIn("qualified_plastic_boundary_span", decision["early_failure_reason"])

    def test_srm_early_failure_ignores_tiny_corner_spanning_cluster(self) -> None:
        active_ids = tuple(f"e{index}" for index in range(100))
        topology = _SRMTopologyDiagnosticsCache(
            active_element_ids=active_ids,
            adjacency={element_id: () for element_id in active_ids},
            boundary_masks={"e0": 1 | 4},
            min_det_j=1.0,
        )
        states = {"e0:0": PlasticState2D(np.array([0.02, 0.0, 0.0, 0.0], dtype=float), 0.02)}
        exc = FEM2DError("newton stalled", diagnostics={"residual_reduction_ratio": 0.95})

        decision = _srm_early_failure_cutback_decision(
            {
                "enabled": True,
                "min_cutbacks": 4,
                "min_cutback_ratio": 0.75,
                "min_last_load": 0.90,
                "score_threshold": 5,
                "spanning_cluster_fraction": 0.10,
                "cluster_fraction": 0.50,
                "plastic_ratio": 0.50,
                "residual_reduction_min": 0.80,
                "line_search_min": 20,
            },
            last_load=0.95,
            attempted_load=1.0,
            cutbacks=8,
            max_cutbacks=12,
            state_current=states,
            state_current_cache=None,
            active_elements=list(active_ids),
            error=exc,
            topology_cache=topology,
        )

        self.assertIsNone(decision)

    def test_srm_early_failure_stops_strong_low_load_coarse_collapse(self) -> None:
        active_ids = tuple(f"e{index}" for index in range(10))
        topology = _SRMTopologyDiagnosticsCache(
            active_element_ids=active_ids,
            adjacency={
                element_id: tuple(
                    neighbor
                    for neighbor in (
                        f"e{index - 1}" if index > 0 else "",
                        f"e{index + 1}" if index < 9 else "",
                    )
                    if neighbor
                )
                for index, element_id in enumerate(active_ids)
            },
            boundary_masks={"e0": 1, "e9": 2},
            min_det_j=1.0,
        )
        states = {
            f"{element_id}:0": PlasticState2D(
                np.array([0.02, 0.0, 0.0, 0.0], dtype=float), 0.02
            )
            for element_id in active_ids
        }
        exc = FEM2DError(
            "newton stalled",
            diagnostics={
                "residual_reduction_ratio": 0.97,
                "line_search_reductions_total": 40,
            },
        )

        decision = _srm_early_failure_cutback_decision(
            {
                "enabled": True,
                "min_cutbacks": 4,
                "min_cutback_ratio": 0.75,
                "min_last_load": 0.90,
                "line_search_min": 20,
                "strong_collapse_enabled": True,
                "strong_min_cutbacks": 4,
                "strong_min_cutback_ratio": 0.50,
                "strong_max_last_load": 0.85,
                "strong_cluster_fraction": 0.80,
                "strong_plastic_ratio": 0.70,
                "strong_residual_reduction_min": 0.90,
                "strong_require_boundary_span": True,
            },
            last_load=0.62,
            attempted_load=0.70,
            cutbacks=5,
            max_cutbacks=12,
            state_current=states,
            state_current_cache=None,
            active_elements=list(active_ids),
            error=exc,
            topology_cache=topology,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision["early_failure_class"], "strong_coarse_collapse")
        self.assertEqual(
            decision["early_failure_policy"],
            "srm_auto_strong_coarse_collapse",
        )
        self.assertIn("last_accepted_load_factor=0.62", decision["early_failure_reason"])

    def test_srm_trial_timing_reports_unattributed_time_and_coverage(self) -> None:
        from geofem_app.fem2d_solver import _srm_trial_timing_summary

        summary = _srm_trial_timing_summary(
            [
                {
                    "factor": 1.0,
                    "elapsed_seconds": 10.0,
                    "solver_elapsed_seconds": 8.0,
                    "unattributed_elapsed_seconds": 2.0,
                },
                {
                    "factor": 1.2,
                    "elapsed_seconds": 20.0,
                    "solver_elapsed_seconds": 15.0,
                    "unattributed_elapsed_seconds": 5.0,
                },
            ]
        )

        self.assertEqual(summary["schema"], "geofem.srm_trial_timing.v2")
        self.assertEqual(summary["total_unattributed_elapsed_seconds"], 7.0)
        self.assertAlmostEqual(summary["timing_coverage_ratio"], 23.0 / 30.0)

    def test_srm_auto_lower_projection_uses_last_accepted_load_factor(self) -> None:
        evaluated: list[float] = []

        def fake_trial(factor: float) -> StageResult2D:
            factor = round(float(factor), 6)
            evaluated.append(factor)
            if factor <= 0.70:
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": 0.0}],
                    {},
                    ["e1"],
                    {"converged": True, "residual_norm": 1.0e-8, "iterations": 2},
                )
            last_load = 0.62 if factor >= 0.99 else 0.97
            raise FEM2DError(
                "increment cutback limit reached",
                diagnostics={
                    "trial_status": "increment_cutback_limit",
                    "last_accepted_load_factor": last_load,
                    "last_accepted_plastic_ratio": 0.86,
                    "active_element_count": 10,
                    "connected_plastic_cluster_size": 8,
                    "plastic_cluster_spans_boundary": True,
                    "cutback_count": 12,
                    "max_cutbacks": 12,
                    "final_step_size": 1.0e-4,
                },
            )

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0],
            {
                "search_mode": "auto",
                "anchor_factor": 1.0,
                "lower_branch": {"factor_min": 0.5, "factor_step": 0.05},
                "upper_branch": {"factor_max": 1.2, "factor_step": 0.1},
                "bracket_stride": 5,
                "factor_tol": 0.01,
                "max_bisection": 5,
                "auto": {"enabled": True, "lower_projection_multipliers": [0.98, 1.10, 1.20]},
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(info["search_mode"], "auto")
        self.assertEqual(info["scan_direction"], "lower")
        self.assertTrue(info["lower_projection"]["used"])
        self.assertTrue(info["lower_projection"]["bracketed"])
        self.assertTrue(info["lower_projection"]["coarse_scan_skipped"])
        self.assertIn(0.682, evaluated)
        self.assertNotIn(0.75, evaluated)
        self.assertNotIn(0.5, evaluated)
        self.assertGreaterEqual(fos, 0.68)
        self.assertLessEqual(fos, 0.70)
        self.assertTrue(all("elapsed_seconds" in row for row in trials))
        self.assertEqual(trials[0]["estimated_fos_from_last_load"], 0.62)

    def test_srm_coarse_to_fine_builds_coarse_mesh_and_narrows_final_window(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        mesh = mesh_from_config(cfg)
        coarse_mesh, info = _srm_build_coarse_mesh(mesh, {"coarse_to_fine": {"enabled": True, "coarsening_factor": 2}})
        self.assertIsNotNone(coarse_mesh)
        self.assertTrue(info["used"])
        self.assertLess(info["coarse_element_count"], info["source_element_count"])

        factors = [round(1.0 + 0.1 * i, 3) for i in range(11)]
        narrowed = _srm_final_factors_from_coarse(
            factors,
            {"bracket_stable_factor": 1.2, "bracket_failed_factor": 1.3},
            {"factor_step": 0.1},
            {"margin_steps": 1},
        )
        self.assertIsNotNone(narrowed)
        self.assertIn(1.2, narrowed or [])
        self.assertIn(1.3, narrowed or [])
        self.assertLess(len(narrowed or []), len(factors))

    def test_srm_coarse_to_fine_runs_as_presearch_without_changing_final_mesh_result(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        cfg["mesh"]["nx"] = 4
        cfg["mesh"]["ny"] = 2
        cfg["loads"] = [{"type": "gravity", "gx": 0.0, "gy": -1.0, "scale": 1.0}]
        cfg["stages"] = [
            {
                "name": "srm-coarse-to-fine",
                "type": "srm",
                "srm": {
                    "factors": [1.0, 1.1, 1.2],
                    "failure_plastic_ratio": 0.0,
                    "coarse_to_fine": {"enabled": True, "coarsening_factor": 2},
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]

        srm = stage.solver_info["srm"]
        self.assertEqual(srm["factor_of_safety"], 1.2)
        self.assertTrue(srm["coarse_to_fine"]["enabled"])
        self.assertTrue(srm["coarse_to_fine"]["used"])
        self.assertLess(srm["coarse_to_fine"]["coarse_element_count"], srm["coarse_to_fine"]["source_element_count"])

    def test_srm_two_branch_parallel_prefetches_candidate_window(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_trial(factor: float) -> StageResult2D:
            nonlocal active, max_active
            factor = round(float(factor), 3)
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.02)
                plastic = 0.0 if factor <= 1.1 else 1.0
                return StageResult2D(
                    f"FS{factor:g}",
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    [{"active": 1.0, "plastic": plastic}],
                    {},
                    [],
                    {"converged": True},
                )
            finally:
                with lock:
                    active -= 1

        _result, fos, trials, info = _run_srm_trial_search(
            [1.0, 1.1, 1.2, 1.3],
            {
                "search_mode": "two_branch",
                "anchor_factor": 1.0,
                "lower_branch": {"factor_min": 0.8, "factor_step": 0.1},
                "upper_branch": {"factor_max": 1.3, "factor_step": 0.1},
                "factor_tol": 0.05,
                "max_bisection": 0,
                "parallel": {"enabled": True, "max_workers": 2},
            },
            0.0,
            fake_trial,
        )

        self.assertEqual(fos, 1.1)
        self.assertEqual([row["factor"] for row in trials], [1.0, 1.1, 1.2])
        self.assertTrue(info["bracketed"])
        self.assertEqual(info["parallel"]["max_workers"], 2)
        self.assertEqual(info["parallel"]["candidate_window_size"], 2)
        self.assertEqual(info["parallel"]["evaluated_trials"], 3)
        self.assertEqual(info["parallel"]["reported_trials"], 3)
        self.assertEqual(info["parallel"]["window_evaluated_trials"], 2)
        self.assertEqual(info["parallel"]["strategy"], "two_branch_candidate_windows")
        self.assertGreater(max_active, 1)
        self.assertLessEqual(max_active, 2)

    def test_geofeas_like_srm_benchmark_writes_integration_point_history(self) -> None:
        cfg = plane_strain_quad4_sample(integration="FULL")
        cfg["materials"]["soil"].update(
            {
                "model": "mohr_coulomb",
                "cohesion": 50.0,
                "friction_angle": 30.0,
                "dilation_angle": 0.0,
            }
        )
        cfg["steps"] = [{"name": "geofeas-like-srm", "type": "srm", "srm": {"factors": [1.0, 1.2], "failure_plastic_ratio": 0.95}}]
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
            with (stage.output_dir / "integration_point_stress.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(stage.solver_info["srm"]["factor_of_safety"], 1.2)
        self.assertEqual(len(stage.solver_info["srm"]["trials"]), 2)
        self.assertGreaterEqual(len(rows), len(result.mesh.elements) * 4)
        self.assertTrue(all(row["stage"] == "geofeas-like-srm" for row in rows))
        self.assertTrue(all(math.isfinite(float(row["q"])) for row in rows[:8]))

    def test_consolidation_pressure_diffusion_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [
                {
                    "name": "consolidation",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 100.0,
                        "dt": 0.1,
                        "steps": 2,
                        "storage": 1.0,
                        "permeability": 1.0,
                        "pressure_bcs": [{"set": "top", "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "pore_pressure.csv").exists())
        pressure = result.stages[0].pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        self.assertEqual(result.stages[0].solver_info["method"], "monolithic_up")
        info = result.stages[0].solver_info["consolidation"]
        self.assertEqual(info["unknowns"], 12)
        self.assertEqual(info["pressure_dof_count"], 4)
        self.assertEqual(len(info["step_history"]), 2)
        self.assertIn("flow_balance", info["step_history"][-1])
        step_row = info["step_history"][0]
        self.assertIn("coupled_assembly_elapsed_seconds", step_row)
        self.assertIn("reduced_matrix_elapsed_seconds", step_row)
        self.assertIn("linear_solve_elapsed_seconds", step_row)
        self.assertIn("elapsed_seconds", step_row)
        self.assertIn("history_storage_policy", info)
        self.assertIn("rollback_policy", info)
        self.assertTrue(info["step_cache"]["enabled"])
        self.assertTrue(all(row["step_cache_used"] for row in info["step_history"]))
        self.assertTrue(np.allclose(pressure[[2, 3]], 0.0))
        self.assertTrue(np.all((pressure[[0, 1]] > 0.0) & (pressure[[0, 1]] < 100.0)))

    def test_consolidation_step_cache_reuses_monolithic_lhs_and_reduced_matrix(self) -> None:
        clear_linear_factor_cache()
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 4,
                "ny": 2,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "solver": {"linear": {"cache_min_size": 0, "symbolic_cache_min_size": 0}},
            "steps": [
                {
                    "name": "consolidation-cache",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 100.0,
                        "dt": 0.1,
                        "steps": 3,
                        "storage": 1.0,
                        "permeability": 1.0,
                        "pressure_bcs": [{"set": "top", "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        info = result.stages[0].solver_info["consolidation"]["step_cache"]
        self.assertTrue(info["enabled"])
        self.assertTrue(info["pressure_lhs_cached"])
        self.assertTrue(info["monolithic_lhs_cached"])
        self.assertTrue(info["monolithic_lhs_direct_fill"]["enabled"])
        self.assertEqual(info["solves"], 3)
        self.assertEqual(info["monolithic_lhs_reuses"], 3)
        self.assertEqual(info["monolithic_lhs_direct_fill_reuses"], 3)
        self.assertEqual(info["reduced_cache_reuses"], 3)
        self.assertGreaterEqual(info["factor_cache_hits"], 2)
        hydraulic = info["hydraulic_assembly"]
        self.assertEqual(hydraulic["pressure_matrices"]["cache_kind"], "pressure_matrix_assembly_cache")
        self.assertEqual(hydraulic["biot_coupling"]["cache_kind"], "biot_coupling_assembly_cache")
        self.assertEqual(hydraulic["boundary_terms"]["cache_kind"], "pressure_boundary_term_cache")
        self.assertGreaterEqual(hydraulic["pressure_matrices"]["batched_elements"], 1)
        self.assertGreaterEqual(hydraulic["biot_coupling"]["batched_elements"], 1)

    def test_consolidation_gmres_uses_up_block_ilu_preconditioner(self) -> None:
        base_cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "bottom", "ux": 0.0, "uy": 0.0}],
            "steps": [
                {
                    "name": "consolidation-block-preconditioner",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 50.0,
                        "dt": 0.1,
                        "steps": 2,
                        "storage": 1.0,
                        "permeability": 1.0,
                        "pressure_bcs": [{"set": "top", "pressure": 0.0}],
                    },
                }
            ],
        }
        gmres_cfg = dict(base_cfg)
        gmres_cfg["solver"] = {
            "linear": {
                "method": "gmres",
                "preconditioner": "up_block_ilu",
                "tol_rel": 1.0e-11,
                "max_iter": 100,
                "ilu_drop_tol": 0.0,
            }
        }
        with tempfile.TemporaryDirectory() as direct_tmp, tempfile.TemporaryDirectory() as gmres_tmp:
            direct = solve_plane_strain_config(base_cfg, direct_tmp)
            gmres = solve_plane_strain_config(gmres_cfg, gmres_tmp)

        direct_stage = direct.stages[0]
        gmres_stage = gmres.stages[0]
        self.assertTrue(np.allclose(gmres_stage.displacements, direct_stage.displacements, rtol=1.0e-8, atol=1.0e-10))
        self.assertTrue(np.allclose(gmres_stage.pore_pressure, direct_stage.pore_pressure, rtol=1.0e-8, atol=1.0e-10))
        info = gmres_stage.solver_info["consolidation"]["step_cache"]
        self.assertEqual(info["linear_solver_method"], "gmres")
        self.assertGreaterEqual(info["preconditioner_reuses"], 1)
        self.assertEqual(info["linear_preconditioner"]["type"], "block_ilu")
        self.assertEqual(info["linear_preconditioner"]["block_count"], 2)

    def test_consolidation_updates_liquefaction_history_from_pore_pressure(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 4,
                "ny": 2,
                "element_type": "QUAD4",
                "material": "sand",
            },
            "materials": {
                "sand": {
                    "model": "bilinear_liquefaction",
                    "E": 10000.0,
                    "nu": 0.30,
                    "gamma": 18.0,
                    "friction_angle": 32.0,
                    "G0": 6000.0,
                    "gamma_ref": 0.001,
                    "liquefaction": {
                        "initial_effective_stress": 5.0,
                        "cyclic_resistance_ratio": 0.2,
                        "cyclic_stress_ratio": 0.18,
                        "generation_rate": 0.2,
                        "dissipation_rate": 0.1,
                        "cycles_per_step": 1.0,
                    },
                }
            },
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [
                {
                    "name": "liq-up",
                    "type": "consolidation",
                    "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            stage = result.stages[0]
            liq_csv = stage.output_dir / "liquefaction_state.csv"
            self.assertTrue(liq_csv.exists())
            self.assertTrue((stage.output_dir / "liquefaction_history.csv").exists())
            self.assertTrue((stage.output_dir / "liquefaction_ru_fl.svg").exists())
            self.assertTrue((stage.output_dir / "liquefaction_post.html").exists())
            with liq_csv.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertIn("liquefaction", stage.solver_info)
        self.assertIn("liquefaction_coupling", stage.solver_info["consolidation"])
        liq_coupling = stage.solver_info["consolidation"]["liquefaction_coupling"]
        self.assertGreater(liq_coupling["generation_source"], 0.0)
        self.assertGreaterEqual(liq_coupling["batched_elements"], 8)
        self.assertEqual(liq_coupling["fallback_elements"], 0)
        self.assertTrue(liq_coupling["direct_fill"]["enabled"])
        self.assertTrue(any(group["element_type"] == "QUAD4" for group in liq_coupling["batch_groups"]))
        self.assertGreater(stage.solver_info["consolidation"]["mass_balance_terms"]["liquefaction_generation_source"], 0.0)
        self.assertGreater(stage.solver_info["liquefaction"]["max_ru"], 0.0)
        self.assertGreater(float(rows[0]["ru"]), 0.0)
        self.assertIn("ru_generation_increment", rows[0])

    def test_axisymmetric_consolidation_batches_liquefaction_pressure_terms(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 4,
                "ny": 2,
                "element_type": "QUAD4",
                "material": "sand",
            },
            "materials": {
                "sand": {
                    "model": "bilinear_liquefaction",
                    "E": 10000.0,
                    "nu": 0.30,
                    "gamma": 18.0,
                    "friction_angle": 32.0,
                    "G0": 6000.0,
                    "gamma_ref": 0.001,
                    "liquefaction": {
                        "initial_effective_stress": 5.0,
                        "cyclic_resistance_ratio": 0.2,
                        "cyclic_stress_ratio": 0.18,
                        "generation_rate": 0.2,
                        "dissipation_rate": 0.1,
                        "cycles_per_step": 1.0,
                    },
                }
            },
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [
                {
                    "name": "axisym-liq-up",
                    "type": "consolidation",
                    "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        liq_coupling = stage.solver_info["consolidation"]["liquefaction_coupling"]
        self.assertTrue(liq_coupling["axisymmetric"])
        self.assertGreaterEqual(liq_coupling["batched_elements"], 8)
        self.assertEqual(liq_coupling["fallback_elements"], 0)
        self.assertTrue(liq_coupling["direct_fill"]["enabled"])
        self.assertGreater(liq_coupling["generation_source"], 0.0)

    def test_consolidation_flux_robin_mass_balance_diagnostics_run_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "all", "fixed": True}],
            "steps": [
                {
                    "name": "consolidation",
                    "type": "consolidation",
                    "hydro": {
                        "initial_pressure": 1.0,
                        "dt": 1.0,
                        "steps": 1,
                        "storage": 1.0,
                        "permeability": 0.0,
                        "pore_flux_bcs": [{"set": "bottom", "flux": 2.0}],
                        "pore_robin_bcs": [{"set": "top", "beta": 0.5, "pressure": -10.0, "seepage_face": True}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        pressure = result.stages[0].pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        info = result.stages[0].solver_info["consolidation"]
        self.assertAlmostEqual(info["boundary"]["flux_total"], 2.0)
        self.assertAlmostEqual(info["boundary"]["robin_conductance_total"], 0.5)
        self.assertEqual(info["seepage_active_edges"], 1)
        self.assertFalse(info["step_cache"]["enabled"])
        self.assertEqual(info["step_cache"]["reason"], "disabled_for_pressure_dependent_seepage_boundary")
        self.assertLess(info["mass_balance"], 1.0e-10)
        self.assertGreater(float(np.max(pressure)), 0.0)

    def test_consolidation_interface_hydraulic_transfer_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [0.0, 1.0], "3": [0.0, 0.0], "4": [0.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 1000.0,
                    "behavior": {"hydro": {"transfer": 2.0}},
                }
            ],
            "boundary_conditions": [{"nodes": ["1", "2", "3", "4"], "fixed": True}],
            "steps": [
                {
                    "name": "interface-transfer",
                    "type": "consolidation",
                    "hydro": {
                        "dt": 1.0,
                        "steps": 1,
                        "storage": 1.0,
                        "permeability": 0.0,
                        "pore_flux_bcs": [{"nodes": ["1", "2"], "flux": 1.0}],
                        "pressure_bcs": [{"nodes": ["3", "4"], "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        pressure = result.stages[0].pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        info = result.stages[0].solver_info["consolidation"]
        self.assertEqual(info["interface_transfer"]["count"], 1)
        self.assertAlmostEqual(info["interface_transfer"]["conductance_total"], 2.0)
        self.assertTrue(np.all(pressure[[0, 1]] > 0.0))
        self.assertTrue(np.allclose(pressure[[2, 3]], 0.0))

    def test_consolidation_mpc_lagrange_ties_slave_dof_without_penalty(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain", "fields": ["u", "p"]},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
            "steps": [
                {
                    "name": "up-lm",
                    "type": "consolidation",
                    "hydro": {"dt": 1.0, "steps": 1, "storage": 1.0, "permeability": 0.0, "pressure_bcs": [{"nodes": ["3", "4"], "pressure": 0.0}]},
                    "loads": [{"node": "3", "fy": -10.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=12)
        consolidation = stage.solver_info["consolidation"]
        self.assertTrue(consolidation["lagrange_linear_cache"]["enabled"])
        self.assertEqual(consolidation["lagrange_linear_cache"]["cache_kind"], "mpc_lagrange_linear_correction_cache")
        self.assertEqual(consolidation["step_cache"]["lagrange_linear_cache_builds"], 1)

    def test_static_increment_control_matches_single_step_solution(self) -> None:
        direct_cfg = plane_strain_quad4_sample(integration="B-bar")
        inc_cfg = plane_strain_quad4_sample(integration="B-bar")
        direct_cfg["steps"] = [{"name": "direct", "type": "static"}]
        inc_cfg["steps"] = [{"name": "incremented", "type": "static", "increments": {"steps": 3, "max_cutbacks": 2}}]
        with tempfile.TemporaryDirectory() as direct_tmp, tempfile.TemporaryDirectory() as inc_tmp:
            direct = solve_plane_strain_config(direct_cfg, direct_tmp)
            incremented = solve_plane_strain_config(inc_cfg, inc_tmp)
        self.assertTrue(np.allclose(incremented.stages[0].displacements, direct.stages[0].displacements, rtol=1.0e-10, atol=1.0e-12))
        info = incremented.stages[0].solver_info["increments"]
        self.assertEqual(info["accepted_steps"], 3)
        self.assertEqual(info["final_factor"], 1.0)
        self.assertTrue(info["step_cache_shared"])
        self.assertEqual(info["step_cache_kind"], "small_deformation_step_cache")
        self.assertTrue(info["step_cache"]["stiffness_pattern_cached"])
        self.assertTrue(all(row["step_cache_used"] for row in info["log"]))
        self.assertEqual(incremented.stages[0].solver_info["topology_cache"]["cache_kind"], "small_deformation_step_cache")

    def test_riks_load_factor_path_runs_in_2d_core(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg["loads"] = []
        cfg["steps"] = [
            {
                "name": "riks",
                "type": "riks",
                "riks": {"lambda_max": 2.0, "steps": 2},
                "loads": [{"edge": ["9", "18"], "ty": -50.0}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "riks_path.csv").exists())
        riks = result.stages[0].solver_info["riks"]
        self.assertEqual(result.stages[0].solver_info["method"], "arc_length")
        self.assertEqual(riks["lambda"], 2.0)
        self.assertEqual([row["lambda"] for row in riks["path"]], [1.0, 2.0])
        self.assertTrue(all(abs(row["constraint_residual"]) < 1.0e-10 for row in riks["path"]))
        self.assertEqual(riks["direction_flips"], 0)
        self.assertEqual(riks["negative_dlambda"], 0)
        self.assertEqual(riks["cutbacks"], 0)
        self.assertEqual(riks["predictor_flips"], 0)
        self.assertTrue(all("delta_lambda" in row for row in riks["path"]))
        self.assertTrue(all("postprocess_elapsed_seconds" in row for row in riks["path"]))
        self.assertTrue(all("linear_solve_elapsed_seconds" in row for row in riks["iteration_history"]))
        self.assertTrue(all("tangent_internal_assembly_elapsed_seconds" in row for row in riks["iteration_history"]))
        cache = riks["cache"]
        self.assertTrue(cache["step_cache"]["enabled"])
        self.assertTrue(cache["sparse_pattern_cached"])
        self.assertTrue(cache["combined_tangent_internal_assembly"])
        self.assertTrue(cache["reduced_matrix_cache"]["enabled"])
        self.assertGreaterEqual(cache["reduced_matrix_cache"]["hits"], 1)

    def test_riks_mpc_lagrange_ties_slave_dof_without_penalty(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"nodes": ["1", "4"], "fixed": True}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
            "steps": [
                {
                    "name": "riks-lm",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"node": "3", "fy": -10.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx3 = result.mesh.node_index["3"]
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertEqual(stage.solver_info["riks"]["lambda"], 2.0)
        self.assertAlmostEqual(stage.displacements[2 * idx2 + 1], stage.displacements[2 * idx3 + 1], places=12)
        cache = stage.solver_info["riks"]["cache"]
        self.assertTrue(cache["combined_tangent_internal_assembly"])
        performance = stage.solver_info["performance"]
        self.assertGreater(performance["assembly_elapsed_seconds"], 0.0)
        self.assertGreater(performance["linear_solve_elapsed_seconds"], 0.0)
        self.assertIn("lagrange_constraint_matrix_elapsed_seconds", performance)
        self.assertIn("lagrange_bmat_elapsed_seconds", performance)
        self.assertIn("lagrange_linear_solve_elapsed_seconds", performance)
        self.assertGreaterEqual(performance["io_report_elapsed_seconds"], 0.0)
        riks_profile = stage.solver_info["riks"]["profile"]
        self.assertIn("predictor_lagrange_bmat_elapsed_seconds", riks_profile)
        self.assertIn("stage_output_elapsed_seconds", riks_profile)

    def test_nonlinear_riks_uses_augmented_direct_fill_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 10.0, "friction_angle": 20.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0, "uy": 0.0}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [{"name": "riks-plastic", "type": "riks", "riks": {"lambda_max": 1.0, "steps": 2, "max_iter": 20}, "loads": [{"node": "3", "fy": -10.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        cache = stage.solver_info["riks"]["cache"]
        self.assertTrue(cache["combined_tangent_internal_assembly"])
        self.assertTrue(cache["augmented_matrix_cache"]["enabled"])
        self.assertEqual(cache["augmented_matrix_cache"]["builds"], 1)
        self.assertGreaterEqual(cache["augmented_matrix_cache"]["hits"], 1)
        self.assertGreaterEqual(cache["reduced_matrix_cache"]["hits"], 1)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_nonlinear_riks_mpc_lagrange_uses_direct_fill_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "static_plane_strain"},
            "mesh": {
                "nodes": {"1": [0.0, 0.0], "2": [1.0, 0.0], "3": [1.0, 1.0], "4": [0.0, 1.0]},
                "elements": [{"id": "e1", "type": "QUAD4", "nodes": ["1", "2", "3", "4"], "material": "soil"}],
            },
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 1.0, "friction_angle": 20.0}},
            "boundary_conditions": [{"nodes": ["1", "4"], "ux": 0.0, "uy": 0.0}],
            "mpc_constraints": [{"master": "2", "slave": "3", "dof": "uy", "coefficient": 1.0, "method": "lagrange"}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [{"name": "riks-lm-plastic", "type": "riks", "riks": {"lambda_max": 1.0, "steps": 2, "max_iter": 20}, "loads": [{"node": "3", "fy": -10.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        cache = stage.solver_info["riks"]["cache"]["lagrange_correction_cache"]
        self.assertTrue(cache["enabled"])
        self.assertEqual(cache["builds"], 1)
        self.assertGreaterEqual(cache["hits"], 1)
        self.assertEqual(cache["current"]["cache_kind"], "arc_length_lagrange_correction_cache")
        self.assertTrue(cache["current"]["constraint_matrix_cached"])
        self.assertTrue(cache["current"]["active_rows_cached"])
        self.assertTrue(cache["current"]["direct_fill"]["enabled"])
        linear_cache = stage.solver_info["riks"]["cache"]["lagrange_linear_cache"]
        self.assertTrue(linear_cache["enabled"])
        self.assertEqual(linear_cache["builds"], 1)
        self.assertGreaterEqual(linear_cache["hits"], 1)
        self.assertEqual(linear_cache["current"]["cache_kind"], "mpc_lagrange_linear_correction_cache")
        self.assertTrue(linear_cache["current"]["direct_fill"]["enabled"])
        performance = stage.solver_info["performance"]
        self.assertGreater(performance["lagrange_bmat_elapsed_seconds"], 0.0)
        self.assertGreater(performance["lagrange_linear_solve_elapsed_seconds"], 0.0)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_axisymmetric_riks_load_factor_path_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "steps": [
                {
                    "name": "axisym-riks",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"edge": ["3", "4"], "ty": -1.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "riks_path.csv").exists())
        stage = result.stages[0]
        riks = stage.solver_info["riks"]
        self.assertEqual(stage.solver_info["method"], "axisymmetric_arc_length")
        self.assertEqual(stage.solver_info["geometry"], "axisymmetric")
        self.assertEqual(riks["lambda"], 2.0)
        self.assertEqual([row["lambda"] for row in riks["path"]], [1.0, 2.0])
        self.assertTrue(all(abs(row["constraint_residual"]) < 1.0e-10 for row in riks["path"]))
        self.assertTrue(all("postprocess_elapsed_seconds" in row for row in riks["path"]))
        self.assertTrue(all("linear_solve_elapsed_seconds" in row for row in riks["iteration_history"]))
        self.assertTrue(all("tangent_internal_assembly_elapsed_seconds" in row for row in riks["iteration_history"]))
        self.assertEqual(riks["cutbacks"], 0)
        self.assertEqual(riks["predictor_flips"], 0)
        cache = riks["cache"]
        self.assertTrue(cache["step_cache"]["enabled"])
        self.assertTrue(cache["sparse_pattern_cached"])
        self.assertTrue(cache["combined_tangent_internal_assembly"])
        self.assertTrue(cache["reduced_matrix_cache"]["enabled"])
        self.assertGreaterEqual(cache["reduced_matrix_cache"]["hits"], 1)

    def test_axisymmetric_nonlinear_riks_uses_augmented_direct_fill_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {"generator": "rectangle", "x_range": [1.0, 2.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 1.0, "friction_angle": 20.0}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [{"name": "axisym-riks-plastic", "type": "riks", "riks": {"lambda_max": 1.0, "steps": 2, "max_iter": 20}, "loads": [{"edge": ["3", "4"], "ty": -10.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        cache = stage.solver_info["riks"]["cache"]
        self.assertTrue(cache["step_cache"]["enabled"])
        self.assertTrue(cache["sparse_pattern_cached"])
        self.assertTrue(cache["combined_tangent_internal_assembly"])
        self.assertTrue(cache["augmented_matrix_cache"]["enabled"])
        self.assertEqual(cache["augmented_matrix_cache"]["builds"], 1)
        self.assertGreaterEqual(cache["augmented_matrix_cache"]["hits"], 1)
        self.assertGreaterEqual(cache["reduced_matrix_cache"]["hits"], 1)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_axisymmetric_nonlinear_riks_mpc_lagrange_uses_direct_fill_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {"generator": "rectangle", "x_range": [1.0, 2.0], "y_range": [0.0, 1.0], "nx": 1, "ny": 1, "element_type": "QUAD4", "material": "soil"},
            "materials": {"soil": {"model": "drucker_prager", "E": 10000.0, "nu": 0.30, "cohesion": 1.0, "friction_angle": 20.0}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "mpc_constraints": [{"master": "2", "slave": "4", "dof": "ux", "coefficient": 1.0, "method": "lagrange"}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [{"name": "axisym-riks-lm-plastic", "type": "riks", "riks": {"lambda_max": 1.0, "steps": 2, "max_iter": 20}, "loads": [{"edge": ["3", "4"], "ty": -10.0}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        cache = stage.solver_info["riks"]["cache"]["lagrange_correction_cache"]
        self.assertTrue(cache["enabled"])
        self.assertEqual(cache["builds"], 1)
        self.assertGreaterEqual(cache["hits"], 1)
        self.assertEqual(cache["current"]["cache_kind"], "arc_length_lagrange_correction_cache")
        self.assertTrue(cache["current"]["constraint_matrix_cached"])
        self.assertTrue(cache["current"]["active_rows_cached"])
        self.assertTrue(cache["current"]["direct_fill"]["enabled"])
        linear_cache = stage.solver_info["riks"]["cache"]["lagrange_linear_cache"]
        self.assertTrue(linear_cache["enabled"])
        self.assertEqual(linear_cache["builds"], 1)
        self.assertGreaterEqual(linear_cache["hits"], 1)
        self.assertEqual(linear_cache["current"]["cache_kind"], "mpc_lagrange_linear_correction_cache")
        self.assertTrue(linear_cache["current"]["direct_fill"]["enabled"])
        self.assertTrue(np.all(np.isfinite(stage.displacements)))

    def test_axisymmetric_riks_with_interface_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [1.0, 1.0], "3": [1.0, 0.0], "4": [1.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [{"id": "joint", "minus_nodes": ["1", "2"], "plus_nodes": ["3", "4"], "kn": 1000.0, "kt": 500.0}],
            "boundary_conditions": [{"nodes": ["1", "2"], "fixed": True}, {"nodes": ["3", "4"], "uy": 0.0}],
            "steps": [
                {
                    "name": "axisym-riks-interface",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"nodes": ["3", "4"], "fx": 1.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        riks = stage.solver_info["riks"]
        self.assertEqual(stage.solver_info["method"], "axisymmetric_arc_length")
        self.assertEqual(riks["lambda"], 2.0)
        self.assertEqual([row["lambda"] for row in riks["path"]], [1.0, 2.0])
        self.assertTrue(stage.interface_results)
        self.assertTrue(all(row["geometry"] == "axisymmetric" for row in stage.interface_results))

    def test_axisymmetric_riks_couples_up_pressure_in_same_stage(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "steps": [
                {
                    "name": "axisym-riks-up",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"edge": ["3", "4"], "ty": -1.0}],
                    "hydro": {
                        "dt": 1.0,
                        "storage": 1.0,
                        "permeability": 0.1,
                        "pore_flux_bcs": [{"edge": ["3", "4"], "flux": 1.0}],
                        "pressure_bcs": [{"set": "bottom", "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
            self.assertTrue((result.stages[0].output_dir / "pore_pressure.csv").exists())
        stage = result.stages[0]
        self.assertEqual(stage.solver_info["method"], "axisymmetric_arc_length_up")
        self.assertAlmostEqual(stage.solver_info["riks"]["lambda"], 2.0)
        self.assertIsNotNone(stage.pore_pressure)
        assert stage.pore_pressure is not None
        self.assertGreater(float(np.max(stage.pore_pressure)), 0.0)
        info = stage.solver_info["consolidation"]
        self.assertTrue(info["coupled_with_riks"])
        self.assertLess(info["mass_balance"], 1.0e-8)
        self.assertTrue(all("pressure_residual_norm" in row for row in stage.solver_info["riks"]["path"]))
        riks_cache = stage.solver_info["riks"]["cache"]
        self.assertTrue(riks_cache["combined_tangent_internal_assembly"])
        self.assertTrue(riks_cache["axisymmetric_up_hydraulic_cache"]["enabled"])
        self.assertGreaterEqual(riks_cache["axisymmetric_up_hydraulic_cache"]["boundary_cache_reuses"], 1)
        self.assertEqual(riks_cache["axisymmetric_up_hydraulic_cache"]["boundary_terms"]["cache_kind"], "pressure_boundary_term_cache")
        self.assertTrue(riks_cache["axisymmetric_up_correction_cache"]["enabled"])
        self.assertGreaterEqual(riks_cache["axisymmetric_up_correction_cache"]["hits"], 1)
        self.assertTrue(info["correction_cache"]["enabled"])

    def test_axisymmetric_riks_up_lagrange_uses_fixed_pattern_correction_cache(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "mpc_constraints": [{"master": "2", "slave": "4", "dof": "ux", "coefficient": 1.0, "method": "lagrange"}],
            "solver": {"linear": {"cache_min_size": 1, "symbolic_cache_min_size": 1}},
            "steps": [
                {
                    "name": "axisym-riks-up-lagrange",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"edge": ["3", "4"], "ty": -1.0}],
                    "hydro": {
                        "dt": 1.0,
                        "storage": 1.0,
                        "permeability": 0.1,
                        "pore_flux_bcs": [{"edge": ["3", "4"], "flux": 1.0}],
                        "pressure_bcs": [{"set": "bottom", "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            stage = solve_plane_strain_config(cfg, tmp).stages[0]
        cache = stage.solver_info["riks"]["cache"]["axisymmetric_up_lagrange_correction_cache"]
        self.assertTrue(cache["enabled"])
        self.assertEqual(cache["builds"], 1)
        self.assertGreaterEqual(cache["hits"], 1)
        self.assertEqual(cache["current"]["cache_kind"], "axisymmetric_up_arc_length_lagrange_correction_cache")
        self.assertTrue(cache["current"]["direct_fill"]["enabled"])
        self.assertTrue(cache["current"]["constraint_matrix_cached"])
        self.assertTrue(cache["current"]["base_coupled_cache"]["coupling_blocks_cached"])
        self.assertEqual(stage.solver_info["consolidation"]["correction_cache"]["current"]["cache_kind"], cache["current"]["cache_kind"])
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-10)
        self.assertTrue(np.all(np.isfinite(stage.displacements)))
        self.assertIsNotNone(stage.pore_pressure)

    def test_axisymmetric_riks_interface_hydraulic_transfer_runs_in_same_stage(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static", "fields": ["u", "p"]},
            "mesh": {
                "nodes": {"1": [1.0, 0.0], "2": [1.0, 1.0], "3": [1.0, 0.0], "4": [1.0, 1.0]},
                "elements": [],
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "interfaces": [
                {
                    "id": "joint",
                    "minus_nodes": ["1", "2"],
                    "plus_nodes": ["3", "4"],
                    "kn": 1000.0,
                    "kt": 500.0,
                    "behavior": {"hydro": {"transfer": 2.0}},
                }
            ],
            "boundary_conditions": [{"nodes": ["1", "2"], "fixed": True}, {"nodes": ["3", "4"], "uy": 0.0}],
            "steps": [
                {
                    "name": "axisym-riks-interface-up",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"nodes": ["3", "4"], "fx": 1.0}],
                    "hydro": {
                        "dt": 1.0,
                        "storage": 1.0,
                        "permeability": 0.0,
                        "pore_flux_bcs": [{"nodes": ["1", "2"], "flux": 1.0}],
                        "pressure_bcs": [{"nodes": ["3", "4"], "pressure": 0.0}],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        pressure = stage.pore_pressure
        self.assertIsNotNone(pressure)
        assert pressure is not None
        self.assertEqual(stage.solver_info["method"], "axisymmetric_arc_length_up")
        self.assertEqual(stage.solver_info["riks"]["lambda"], 2.0)
        info = stage.solver_info["consolidation"]
        self.assertEqual(info["interface_transfer"]["count"], 1)
        self.assertAlmostEqual(info["interface_transfer"]["conductance_total"], 4.0 * math.pi)
        self.assertTrue(np.all(pressure[[0, 1]] > 0.0))
        self.assertTrue(np.allclose(pressure[[2, 3]], 0.0))

    def test_axisymmetric_riks_mpc_lagrange_runs_in_2d_core(self) -> None:
        cfg = {
            "analysis": {"dimension": "2D", "type": "axisymmetric_static"},
            "mesh": {
                "generator": "rectangle",
                "x_range": [1.0, 2.0],
                "y_range": [0.0, 1.0],
                "nx": 1,
                "ny": 1,
                "element_type": "QUAD4",
                "material": "soil",
            },
            "materials": {"soil": {"model": "elastic", "E": 10000.0, "nu": 0.30}},
            "boundary_conditions": [{"set": "left", "ux": 0.0}, {"set": "bottom", "uy": 0.0}],
            "mpc_constraints": [{"master": "2", "slave": "4", "dof": "ux", "coefficient": 1.0, "method": "lagrange"}],
            "steps": [
                {
                    "name": "axisym-riks-lm",
                    "type": "riks",
                    "riks": {"lambda_max": 2.0, "steps": 2},
                    "loads": [{"edge": ["3", "4"], "ty": -1.0}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = solve_plane_strain_config(cfg, tmp)
        stage = result.stages[0]
        idx2 = result.mesh.node_index["2"]
        idx4 = result.mesh.node_index["4"]
        self.assertEqual(stage.solver_info["mpc"]["applied_method"], "lagrange")
        self.assertLess(stage.solver_info["mpc"]["max_violation"], 1.0e-12)
        self.assertEqual(stage.solver_info["riks"]["lambda"], 2.0)
        self.assertAlmostEqual(stage.displacements[2 * idx2], stage.displacements[2 * idx4], places=12)
        cache = stage.solver_info["riks"]["cache"]
        self.assertTrue(cache["combined_tangent_internal_assembly"])

    def test_riks_internal_cutback_retries_before_failure(self) -> None:
        cfg = plane_strain_quad4_sample(integration="B-bar")
        cfg["loads"] = []
        cfg["steps"] = [
            {
                "name": "riks",
                "type": "riks",
                "riks": {"lambda_max": 1.0, "steps": 1, "max_iter": 0, "max_cutbacks": 1, "cutback_factor": 0.5},
                "loads": [{"edge": ["9", "18"], "ty": -50.0}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            original_metrics = fem2d_solver_module._riks_convergence_metrics

            def force_nonconvergence(**kwargs: object) -> dict[str, object]:
                metrics = dict(original_metrics(**kwargs))
                metrics["converged"] = False
                return metrics

            with patch.object(fem2d_solver_module, "_riks_convergence_metrics", side_effect=force_nonconvergence):
                with self.assertRaisesRegex(FEM2DError, "failed after 1 cutbacks"):
                    solve_plane_strain_config(cfg, tmp)

    def test_unsupported_material_still_fails_in_2d_core(self) -> None:
        cfg = plane_strain_quad4_sample()
        cfg["materials"]["soil"].update({"model": "cam_clay"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FEM2DError, "unsupported 2D core material"):
                solve_plane_strain_config(cfg, tmp)


if __name__ == "__main__":
    unittest.main()
