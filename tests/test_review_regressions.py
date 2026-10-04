"""Independent reproductions from the October 2026 numerical review."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import eye, csr_matrix

from geofem_app import fem2d_io as io, fem2d_materials as mat, fem2d_linear_solver as lin
from geofem_app.fem2d_types import Mesh2D, StageResult2D, FEM2DError, ElasticPlaneStrainMaterial
from geofem_app.fem2d_mesh import _generate_rectangle_mesh
from geofem_app.fem2d_constraints import collect_constraints
from geofem_app.fem2d_hydro import _water_level_pressure_specs, _collect_pressure_constraints


class ReviewRegressions(unittest.TestCase):
    def test_csv_identifiers_and_nonzero_results_roundtrip(self):
        mesh = Mesh2D(['1', '001', 'n1'], np.zeros((3, 2)), [])
        result = StageResult2D(name='test', displacements=np.arange(1., 7.),
                              reactions=np.arange(11., 17.), element_results=[],
                              constrained_dofs={0: .1, 3: .2, 5: .3},
                              active_elements=[], pore_pressure=np.array([7., 8., 9.]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            io.write_displacements_csv(mesh, result, path / 'displacements.csv')
            io.write_pore_pressure_csv(mesh, result, path / 'pore_pressure.csv')
            io.write_reactions_csv(mesh, result, path / 'reactions.csv')
            np.testing.assert_array_equal(io._stage_displacements_from_csv(mesh, path), result.displacements)
            np.testing.assert_array_equal(io._stage_pore_pressure_from_csv(mesh, path), result.pore_pressure)
            reactions, fixed = io._stage_reactions_from_csv(mesh, path)
            self.assertEqual(fixed, result.constrained_dofs)
            for dof in fixed:
                self.assertEqual(reactions[dof], result.reactions[dof])
            with (path / 'element_stress.csv').open('w', newline='') as f:
                csv.writer(f).writerows([['element_id', 'active', 'sxx'], ['001', '1', '12.5']])
            self.assertEqual(io._read_csv_dict_rows(path / 'element_stress.csv')[0],
                             {'element_id': '001', 'active': 1., 'sxx': 12.5})

    def test_tri6_boundary_sets_include_every_geometric_node_and_constraint(self):
        mesh = _generate_rectangle_mesh({'element_type': 'TRI6', 'nx': 2, 'ny': 2})
        for name, axis, value in [('left', 0, 0), ('right', 0, 1), ('bottom', 1, 0), ('top', 1, 1)]:
            expected = {nid for nid, xy in zip(mesh.node_ids, mesh.coords) if xy[axis] == value}
            self.assertEqual(set(mesh.node_sets[name]), expected)
            fixed = collect_constraints(mesh, [{'set': name, 'ux': .2, 'uy': -.1}])
            pressure = _collect_pressure_constraints(mesh, [{'set': name, 'pressure': 4.}])
            self.assertEqual(set(pressure), {mesh.node_index[n] for n in expected})
            self.assertEqual(len(fixed), 2 * len(expected))

    def test_csv_unknown_missing_and_duplicate_ids_are_diagnosed(self):
        mesh = Mesh2D(['1', '001'], np.zeros((2, 2)), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for ids in [['1', 'bad'], ['1'], ['1', '1']]:
                with (path / 'displacements.csv').open('w', newline='') as f:
                    csv.writer(f).writerows([['node_id', 'ux', 'uy'], *[[n, 1, 2] for n in ids]])
                with self.subTest(ids=ids), self.assertRaises(FEM2DError):
                    io._stage_displacements_from_csv(mesh, path)

    def test_explicit_invalid_water_level_target_fails(self):
        mesh = _generate_rectangle_mesh({})
        for key in ['water_level', 'water_levels', 'water_level_updates']:
            with self.subTest(key=key), self.assertRaises(FEM2DError):
                _water_level_pressure_specs(mesh, {key: {'set': 'typo', 'level': 2}}, {})

    def test_omitted_water_level_target_defaults_to_all(self):
        mesh = _generate_rectangle_mesh({})
        specs = _water_level_pressure_specs(mesh, {'water_level': {'level': 2}}, {})
        self.assertEqual({s['node'] for s in specs}, set(mesh.node_ids))

    def test_water_level_edges_apply_only_to_selected_boundary(self):
        mesh = _generate_rectangle_mesh({'element_type': 'TRI6'})
        for key in ['edge', 'edges']:
            specs = _water_level_pressure_specs(mesh, {'water_level': {key: 'bottom', 'level': 2}}, {})
            self.assertEqual({s['node'] for s in specs}, set(mesh.node_sets['bottom']))
            with self.assertRaises(FEM2DError):
                _water_level_pressure_specs(mesh, {'water_level': {key: 'typo', 'level': 2}}, {})

    def test_lu_factorization_failure_is_fem_error_and_releases_waiters(self):
        for settings in [{'cache_auto_disable_miss_streak': 0}, {'cache_auto_disable_miss_streak': 1}]:
            lin.clear_linear_factor_cache()
            with patch.object(lin, 'splu', side_effect=RuntimeError('singular')):
                with self.assertRaisesRegex(FEM2DError, 'review-stage'):
                    lin._solve_direct_scipy(eye(32, format='csr'), np.ones(32), 'review-stage', settings)
            self.assertFalse(lin._FACTOR_CACHE_BUILDING)
        lin.clear_linear_factor_cache()
        with self.assertRaises(FEM2DError):
            lin._solve_direct_scipy(csr_matrix((32, 32)), np.ones(32), 'singular-stage', {})

    def test_auto_disabled_lu_failure_uses_the_same_error_contract(self):
        lin.clear_linear_factor_cache()
        settings = {'cache_auto_disable_miss_streak': 1, 'cache_auto_reprobe_interval': 16}
        lin._solve_direct_scipy(eye(32, format='csr'), np.ones(32), 'populate', settings)
        self.assertTrue(any(s['disabled'] for s in lin._FACTOR_PATTERN_STATS.values()))
        with patch.object(lin, 'splu', side_effect=RuntimeError('singular')), self.assertRaisesRegex(FEM2DError, 'auto-disabled'):
            lin._solve_direct_scipy(2 * eye(32, format='csr'), np.ones(32), 'auto-disabled', settings)
        self.assertFalse(lin._FACTOR_CACHE_BUILDING)
        lin.clear_linear_factor_cache()

    def test_shared_hardening_active_system_and_tangent(self):
        # f_a = N_a sigma - c - H(kappa + sum(gamma)); derivative is H 11^T.
        coeff = mat._mc_plane_coeffs(np.pi / 6)
        cn = np.eye(3) * 1000
        trial = np.array([-100., 0., 0.])
        cterm = 20 * np.cos(np.pi / 6)
        values = coeff @ trial - cterm
        ids = np.argsort(values)[-2:]
        n = coeff[ids]
        h = 100.
        gamma = np.linalg.solve(n @ cn @ n.T + h * np.ones((2, 2)), values[ids])
        reference = trial - cn @ n.T @ gamma
        self.assertTrue(np.all(gamma > 0))
        np.testing.assert_allclose(n @ reference - cterm - h * gamma.sum(), 0, atol=1e-12)
        cache = mat._mc_python_candidate_matrix_cache(coeff.tobytes(), coeff.tobytes(), cn.tobytes(), h)
        np.testing.assert_allclose(cache[tuple(ids)][3], n @ cn @ n.T + h, atol=1e-12)
        padded = np.zeros(3, dtype=np.int64); padded[:2] = ids
        ok, corrected, multipliers, residual, metric = mat._mc_eval_active_candidate_numba(
            trial, values, coeff, coeff, cterm, cn, h, 0., padded, 2)
        self.assertTrue(ok)
        np.testing.assert_allclose(corrected, reference, atol=1e-11)
        np.testing.assert_allclose(residual[ids], 0., atol=1e-11)
        jac = np.eye(3) - cn @ n.T @ np.linalg.solve(n @ cn @ n.T + h, n)
        np.testing.assert_allclose(mat._mc_principal_jacobian(tuple(ids), coeff, coeff, cn, h), jac, atol=1e-12)
        ok, jit_jac = mat._mc_principal_jacobian_numba(padded, 2, coeff, coeff, cn, h)
        self.assertTrue(ok)
        np.testing.assert_allclose(jit_jac, jac, atol=1e-12)

    def test_public_hardening_return_satisfies_kkt_and_independent_difference(self):
        material = ElasticPlaneStrainMaterial('soil', 1000., 0., model='mohr_coulomb',
            cohesion=10., friction_angle=30., dilation_angle=30., hardening=100.)
        strain = np.array([-.1, 0., 0., 0.])
        update = mat.update_plane_strain_stress(material, strain)
        self.assertEqual(len(update.active_set), 2)
        coeff = mat._mc_plane_coeffs(np.pi / 6)
        values = coeff @ np.sort(update.stress[:3]) - 20 * np.cos(np.pi / 6) - 100 * update.kappa
        self.assertLessEqual(float(values.max()), 1e-10)
        self.assertGreater(update.kappa, 0)
        np.testing.assert_allclose(values[list(update.active_set)], 0, atol=1e-10)
        delta = 1e-8
        independent = np.column_stack([(mat.update_plane_strain_stress(material, strain + delta * e).stress - update.stress) / delta
                                       for e in np.eye(4)])
        np.testing.assert_allclose(mat.algorithmic_material_tangent(material, strain), independent, rtol=1e-6, atol=3e-5)

    def test_rankine_cap_tangents_match_public_stress_difference(self):
        material = ElasticPlaneStrainMaterial('soil', 1000., 0., model='mohr_coulomb',
            cohesion=10., friction_angle=30., dilation_angle=30., tension_cutoff=True,
            tensile_strength=0., mohr_coulomb_apex_policy='rankine_cap')
        strain = np.array([.1, -.1, 0., 0.])
        base = mat.update_plane_strain_stress(material, strain).stress
        delta = 1e-8
        reference = np.column_stack([(mat.update_plane_strain_stress(material, strain + delta * e).stress - base) / delta
                                     for e in np.eye(4)])
        for method in ['analytic', 'numerical']:
            np.testing.assert_allclose(mat.algorithmic_material_tangent(material, strain, method=method), reference,
                                       rtol=1e-6, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
