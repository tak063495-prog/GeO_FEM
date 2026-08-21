"""Compatibility facade for the split 2D FEM core.

The implementation is organized by responsibility in sibling modules:
config, mesh, materials, elements, interfaces, solver, I/O, and shared types.
Existing callers can continue importing from ``geofem_app.fem2d``.
"""

from __future__ import annotations

from .fem2d_config import *
from .fem2d_elements import *
from .fem2d_element_elastic_kernels import *
from .fem2d_element_elastic_post import *
from .fem2d_element_advanced_elastic_post import *
from .fem2d_element_advanced_strength_kernels import *
from .fem2d_element_j2dp_kernels import *
from .fem2d_element_mohr_coulomb_kernels import *
from .fem2d_element_numba_primitives import *
from .fem2d_element_tension_cutoff_kernels import *
from .fem2d_interfaces import *
from .fem2d_io import *
from .fem2d_large_deformation import *
from .fem2d_linear_solver import *
from .fem2d_materials import *
from .fem2d_mesh import *
from .fem2d_mpc import *
from .fem2d_performance_contract import *
from .fem2d_plastic_batch import *
from .fem2d_plastic_state_arrays import *
from .fem2d_solver import *
from .fem2d_solver_controls import *
from .fem2d_structural import *
from .fem2d_types import *
from .fem2d_utils import *
from .analysis_log import *
from .api_contracts import *
from .geofeas_public import *
from .geofeas_verification import *
from .html_report_utils import *
from .input_diagnostics import *
from .load_combinations import *
from .maintainability_audit import *
from .material_models import *
from .messages import *
from .mesh_quality import *
from .mesh_coupling import *
from .mesh_coupling_workflow import *
from .performance_monitor import *
from .pdf_writer import *
from .reduced_matrix_cache import *
from .result_viewer import *
from .sparse_assembly import *
from .standard_report import *
from .standard_benchmarks import *
from .startup_check import *
from .verification_benchmarks import *
from .vgflow2d import *

__all__ = sorted(name for name in globals() if not name.startswith("__"))
