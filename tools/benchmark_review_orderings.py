"""Reproducible existing-ordering comparison, without changing solver defaults.

Run: python tools/benchmark_review_orderings.py --output runs/review-orderings.json
Only elastic QUAD4 matrices are exercised; results do not establish an SRM/plasticity speedup.
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
from geofem_app.fem2d import assemble_global_stiffness as assemble_stiffness, ElasticPlaneStrainMaterial
from geofem_app.fem2d_mesh import _generate_rectangle_mesh
from geofem_app.fem2d_constraints import collect_constraints
from geofem_app import fem2d_linear_solver as linear


def benchmark():
    rows = []
    profile = cProfile.Profile()
    with threadpool_limits(limits=1):
        for n in (12, 40, 80):
            mesh = _generate_rectangle_mesh({'nx': n, 'ny': n, 'element_type': 'QUAD4'})
            materials = {'soil': ElasticPlaneStrainMaterial('soil', 10000., .3)}
            matrix = profile.runcall(assemble_stiffness, mesh, materials)
            fixed = collect_constraints(mesh, [{'set': 'bottom', 'ux': 0, 'uy': 0}])
            free = np.array([i for i in range(matrix.shape[0]) if i not in fixed])
            matrix = matrix[free][:, free].tocsr()
            reference = np.sin(np.arange(len(free)) * .01) + 1
            rhs = matrix @ reference
            for name, settings in [
                ('rcm_natural', {}),
                ('colamd', {'cache_symbolic': False, 'permc_spec': 'COLAMD'}),
                ('mmd_at_plus_a', {'cache_symbolic': False, 'permc_spec': 'MMD_AT_PLUS_A'}),
            ]:
                settings = {**settings, 'cache_auto_disable_miss_streak': 0}
                times, warm = [], []
                fill = bytes_estimate = 0
                max_error = max_residual = 0.
                for repeat in range(5):
                    linear.clear_linear_factor_cache()
                    started = time.perf_counter()
                    solved, state, info = profile.runcall(linear._solve_direct_scipy, matrix, rhs, 'ordering-benchmark', settings)
                    times.append(time.perf_counter() - started)
                    entries = list(linear._FACTOR_CACHE.values())
                    if entries:
                        entry = entries[-1]
                        fill = entry.lu.L.nnz + entry.lu.U.nnz
                        bytes_estimate = entry.estimated_bytes
                    started = time.perf_counter()
                    solved_warm, warm_state, _ = linear._solve_direct_scipy(matrix, rhs, 'ordering-benchmark', settings)
                    warm.append(time.perf_counter() - started)
                    np.testing.assert_array_equal(solved_warm, solved)
                    assert warm_state == 'hit'
                    error = np.linalg.norm(solved-reference, ord=np.inf)
                    residual = np.linalg.norm(matrix @ solved-rhs) / np.linalg.norm(rhs)
                    assert error < 1e-7 and residual < 1e-10
                    max_error = max(max_error, float(error)); max_residual = max(max_residual, float(residual))
                rows.append({'nx': n, 'dof': len(free), 'nnz': matrix.nnz, 'ordering': name,
                             'cold_median_s': statistics.median(times), 'warm_lu_hit_median_s': statistics.median(warm),
                             'factor_nnz_l_plus_u': fill, 'factor_estimated_bytes': bytes_estimate,
                             'max_solution_error_inf': max_error, 'max_original_relative_residual': max_residual,
                             'all_successful': True, 'warm_identical': True})
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(15)
    return {'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'threads': 1, 'samples': 5, 'cold': 'factor/ordering caches cleared',
            'warm': 'same numeric LU hit; assembly/JIT excluded', 'cases': rows, 'profile': stream.getvalue()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'profile'}, indent=2))
