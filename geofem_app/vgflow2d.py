"""VGFlow 2D public-information based seepage analysis substitute.

The commercial VGFlow 2D implementation is not public, so this module keeps a
clear boundary around the open substitute: a scalar total-head FEM solver with
van Genuchten/table unsaturated hydraulic properties and GeoFEAS-facing open
PRS/PTN style exports.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .fem2d_elements import integration_points, shape_functions
from .fem2d_linear_solver import solve_sparse_with_constraints
from .fem2d_mesh import _edge_length, _edge_lumped_weights, _pressure_edges, _target_nodes, mesh_from_config
from .fem2d_types import FEM2DError, Mesh2D
from .fem2d_utils import _ensure_list
from .hydro_exchange import (
    element_potential_points,
    pressure_head_from_total,
    shared_hydro_exchange_engine,
    total_head_from_pressure,
    waterline_points_from_total_head,
)
from .mesh_coupling_workflow import write_vgflow_geofeas_coupling_outputs
from .vgflow2d_alternative_spec import write_vgflow_alternative_spec
from .vgflow2d_boundary import collect_vgflow_boundary_diagnostics, write_vgflow_boundary_diagnostics
from .vgflow2d_cad import collect_vgflow_cad_import_diagnostics, write_vgflow_cad_import_outputs
from .vgflow2d_curve_package import write_vgflow_curve_package
from .vgflow2d_design import vgflow_design_template_catalog, write_vgflow_design_checks
from .vgflow2d_exchange import write_vgflow_exchange_outputs
from .vgflow2d_mesh import vgflow_mesh_template_catalog, write_vgflow_mesh_outputs
from .vgflow2d_post import vgflow_element_post_fields, write_vgflow_post_outputs
from .vgflow2d_pre import vgflow_pre_template_catalog, write_vgflow_pre_outputs
from .vgflow2d_project import read_vgflow_project_package, write_vgflow_project_package
from .vgflow2d_report import write_vgflow_report_bundle
from .vgflow2d_ui import write_vgflow_ui_profile_outputs
from .vgflow2d_kernels import (
    VGFLOW_MATERIAL_SATURATED,
    VGFLOW_MATERIAL_TABLE,
    VGFLOW_MATERIAL_VAN_GENUCHTEN,
    vgflow_table_state_array_numba,
    vgflow_table_state_numba,
    vgflow_quad4_assembly_triplets_numba,
    vgflow_quad8_assembly_triplets_numba,
    vgflow_tri3_assembly_triplets_numba,
    vgflow_tri6_assembly_triplets_numba,
    vgflow_water_state_array_numba,
    vgflow_water_state_numba,
)


VGFLOW_UNSATURATED_PUBLIC_PRESETS: dict[str, dict[str, Any]] = {
    "river_embankment_sandy": {
        "label": "River embankment sandy soil public substitute",
        "model": "van_genuchten",
        "alpha": 5.0,
        "n": 1.8,
        "theta_r": 0.08,
        "theta_s": 0.42,
        "note": "Open engineering starter value, not a commercial VGFlow2D built-in test value.",
    },
    "river_embankment_clayey": {
        "label": "River embankment clayey soil public substitute",
        "model": "van_genuchten",
        "alpha": 1.5,
        "n": 1.35,
        "theta_r": 0.12,
        "theta_s": 0.55,
        "note": "Open engineering starter value, not a commercial VGFlow2D built-in test value.",
    },
    "dam_core_low_permeability": {
        "label": "Dam core low-permeability zone public substitute",
        "model": "van_genuchten",
        "alpha": 1.5,
        "n": 3.0,
        "theta_r": 0.30,
        "theta_s": 0.70,
        "note": "Representative open value for guidance-style seepage examples.",
    },
    "random_zone": {
        "label": "Random zone public substitute",
        "model": "van_genuchten",
        "alpha": 5.0,
        "n": 1.5,
        "theta_r": 0.15,
        "theta_s": 0.60,
        "note": "Representative open value for guidance-style seepage examples.",
    },
    "foundation_rock": {
        "label": "Foundation rock public substitute",
        "model": "van_genuchten",
        "alpha": 5.0,
        "n": 4.0,
        "theta_r": 0.40,
        "theta_s": 0.80,
        "note": "Representative open value for guidance-style seepage examples.",
    },
}

_VGFLOW_UNSATURATED_PRESET_ALIASES = {
    "sandy": "river_embankment_sandy",
    "sand": "river_embankment_sandy",
    "clayey": "river_embankment_clayey",
    "clay": "river_embankment_clayey",
    "dam_core": "dam_core_low_permeability",
    "core": "dam_core_low_permeability",
    "random": "random_zone",
    "rock": "foundation_rock",
    "foundation": "foundation_rock",
}

_CURVE_FILE_CACHE: dict[tuple[str, int, int, str], list[dict[str, float]]] = {}


@dataclass(frozen=True)
class VgFlowMaterial:
    name: str
    kx: float
    ky: float
    specific_storage: float = 0.0
    angle_deg: float = 0.0
    unsaturated_model: str = "saturated"
    alpha: float = 1.0
    n: float = 2.0
    theta_r: float = 0.0
    theta_s: float = 1.0
    table: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class VgFlowStepResult:
    index: int
    time: float
    total_head: np.ndarray
    iteration_count: int
    residual_norm: float
    active_seepage_nodes: int


@dataclass(frozen=True)
class VgFlowSolveResult:
    output_dir: Path
    mesh: Mesh2D
    materials: dict[str, VgFlowMaterial]
    steps: list[VgFlowStepResult]
    warnings: list[str]
    manifest: dict[str, Any]


def is_vgflow2d_config(cfg: Mapping[str, Any]) -> bool:
    analysis = cfg.get("analysis", {})
    if not isinstance(analysis, Mapping):
        return False
    text = " ".join(str(analysis.get(key, "")) for key in ("type", "mode", "profile", "solver", "product")).lower()
    return any(token in text for token in ("vgflow", "richards", "seepage_flow", "saturated_unsaturated"))


def solve_vgflow2d_config(cfg: Mapping[str, Any], output_dir: str | Path) -> VgFlowSolveResult:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = mesh_from_config(cfg)
    materials = vgflow_materials_from_config(cfg)
    analysis = cfg.get("analysis", {}) if isinstance(cfg.get("analysis", {}), Mapping) else {}
    seepage = _vgflow_mapping(cfg)
    problem_type = _problem_type(analysis, seepage)
    gamma_w = float(seepage.get("gamma_w", seepage.get("water_unit_weight", 9.80665)) or 9.80665)
    transient = _is_transient(analysis, seepage)
    times = _analysis_times(seepage, transient)
    max_iter = int(seepage.get("max_iter", seepage.get("max_iterations", 30)) or 30)
    tolerance = float(seepage.get("tolerance", seepage.get("tol", 1.0e-8)) or 1.0e-8)
    head = _initial_total_head(mesh, seepage, problem_type)
    previous = head.copy()
    steps: list[VgFlowStepResult] = []
    warnings: list[str] = []

    for index, time in enumerate(times):
        dt = _step_dt(times, index, seepage)
        fixed_base = _head_constraints(mesh, seepage, time, problem_type)
        active_seepage: dict[int, float] = {}
        residual_norm = math.inf
        iterations = 0
        for iteration in range(1, max_iter + 1):
            active_seepage = _active_seepage_constraints(mesh, seepage, head, time, problem_type)
            fixed = {**fixed_base, **active_seepage}
            conductivity, capacity = _assemble_vgflow_matrices(mesh, materials, head, problem_type)
            rhs = _boundary_rhs(mesh, seepage, time)
            if transient:
                lhs = (capacity / dt + conductivity).tocsr()
                rhs = rhs + (capacity / dt) @ previous
            else:
                lhs = conductivity.tocsr()
            new_head = solve_sparse_with_constraints(lhs, rhs, fixed, stage_name=f"vgflow2d-{index}", solver={"linear": _linear_solver_cfg(seepage)})
            residual_norm = float(np.linalg.norm(new_head - head, ord=np.inf))
            head = new_head
            iterations = iteration
            if residual_norm <= tolerance:
                break
        else:
            warnings.append(f"VGFlow2D step {index}: nonlinear seepage iteration did not converge below {tolerance:g}")
        steps.append(VgFlowStepResult(index=index, time=float(time), total_head=head.copy(), iteration_count=iterations, residual_norm=residual_norm, active_seepage_nodes=len(active_seepage)))
        previous = head.copy()

    artifacts = _write_vgflow_outputs(out, mesh, materials, steps, problem_type, gamma_w, warnings, seepage)
    manifest = {
        "schema": "geofem.vgflow2d.public_substitute.v1",
        "profile": "VGFlow2D public-information based substitute",
        "problem_type": problem_type,
        "transient": transient,
        "step_count": len(steps),
        "node_count": len(mesh.node_ids),
        "element_count": len(mesh.elements),
        "material_count": len(materials),
        "features": [
            "richards_picard_total_head",
            "van_genuchten_unsaturated",
            "table_unsaturated",
            "known_head_boundary",
            "water_level_boundary",
            "pressure_head_boundary",
            "rainfall_flux_boundary",
            "seepage_face_switching",
            "boundary_curve_file_interchange",
            "open_boundary_curve_package_manifest",
            "shared_geofem_cad_import_engine",
            "boundary_rainfall_runoff_curve_alignment_diagnostics",
            "cad_raster_import_scale_diagnostics",
            "cad_raster_auto_strata_extraction",
            "open_vg2_surrogate_project_package",
            "transient_exchange_time_selection_package",
            "shared_hydro_exchange_engine",
            "public_unsaturated_presets",
            "pre_operation_log_templates",
            "ui_toolbar_context_modal_state_profile",
            "mesh_generation_plan_quality_diagnostics",
            "post_contours_vectors_flowlines_sections",
            "post_tables_units_copy_animation_flow_sign",
            "post_direct_avi_animation_export",
            "report_manifest_html_pdf",
            "report_public_ppf_print_profile_substitute",
            "alternative_spec_acceptance_profile",
            "design_checks_piping_boiling_courant_templates",
            "vgflow_geofeas_mesh_coupling_workflow",
            "vgflow_to_geofeas_stage_handoff",
            "coupled_material_layer_dictionary",
            "coupling_api_contract_and_benchmark",
            "open_prs_ptn_exports",
        ],
        "shared_engines": {
            "hydro_exchange": shared_hydro_exchange_engine(),
        },
        "artifacts": artifacts,
        "warnings": warnings,
    }
    (out / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return VgFlowSolveResult(out, mesh, materials, steps, warnings, manifest)


def vgflow_materials_from_config(cfg: Mapping[str, Any]) -> dict[str, VgFlowMaterial]:
    raw = cfg.get("materials", cfg.get("material", {}))
    if not isinstance(raw, Mapping) or not raw:
        raise FEM2DError("VGFlow2D analysis requires materials")
    return {str(name): _vgflow_material(str(name), value if isinstance(value, Mapping) else {}) for name, value in raw.items()}


def vgflow_unsaturated_public_catalog() -> list[dict[str, Any]]:
    return [{"id": preset_id, **preset} for preset_id, preset in sorted(VGFLOW_UNSATURATED_PUBLIC_PRESETS.items())]


def _vgflow_material(name: str, raw: Mapping[str, Any]) -> VgFlowMaterial:
    seep = raw.get("seepage", raw.get("hydraulic", raw))
    if not isinstance(seep, Mapping):
        seep = raw
    seep = _with_unsaturated_preset(seep)
    kx = _permeability_value(seep, ("kx", "Kx", "permeability_x", "horizontal_permeability"), "kx")
    ky = _permeability_value(seep, ("ky", "Ky", "permeability_y", "vertical_permeability"), "ky", default=kx)
    unsat = seep.get("unsaturated", seep.get("van_genuchten", seep.get("vg", {})))
    if not isinstance(unsat, Mapping):
        unsat = {}
    table = _unsaturated_table(unsat or seep)
    model = str(unsat.get("model", seep.get("unsaturated_model", "table" if table else "van_genuchten" if unsat else "saturated"))).lower().replace("-", "_")
    if model in {"vg", "van", "van_genuchten", "vangenuchten"}:
        model = "van_genuchten"
    return VgFlowMaterial(
        name=name,
        kx=kx,
        ky=ky,
        specific_storage=max(0.0, float(_pick(seep, ("specific_storage", "Ss", "ss", "storage"), 0.0) or 0.0)),
        angle_deg=float(_pick(seep, ("angle", "angle_deg", "inclination", "地層の傾斜角度"), 0.0) or 0.0),
        unsaturated_model=model,
        alpha=max(float(_pick(unsat, ("alpha", "α"), _pick(seep, ("alpha", "α"), 1.0)) or 1.0), np.finfo(float).eps),
        n=max(float(_pick(unsat, ("n",), _pick(seep, ("n",), 2.0)) or 2.0), 1.01),
        theta_r=float(_pick(unsat, ("theta_r", "residual_water_content", "θr"), _pick(seep, ("theta_r", "θr"), 0.0)) or 0.0),
        theta_s=float(_pick(unsat, ("theta_s", "saturated_water_content", "θs"), _pick(seep, ("theta_s", "θs"), 1.0)) or 1.0),
        table=table,
    )


def _permeability_value(raw: Mapping[str, Any], keys: tuple[str, ...], prefix: str, default: float | None = None) -> float:
    value = _pick(raw, keys, None)
    if value not in (None, ""):
        out = float(value)
    else:
        mantissa = _pick(raw, (f"{prefix}_mantissa", f"{prefix}_仮数", f"{prefix.upper()}_mantissa"), None)
        exponent = _pick(raw, (f"{prefix}_exponent", f"{prefix}_指数", f"{prefix.upper()}_exponent"), None)
        if mantissa not in (None, "") and exponent not in (None, ""):
            out = float(mantissa) * 10.0 ** float(exponent)
        elif default is not None:
            out = float(default)
        else:
            out = float(_pick(raw, ("permeability", "k", "K"), 1.0e-6) or 1.0e-6)
    if out < 0.0:
        raise FEM2DError(f"VGFlow2D material permeability must be non-negative: {prefix}")
    return out


def _unsaturated_table(raw: Mapping[str, Any]) -> tuple[tuple[float, float, float], ...]:
    rows = raw.get("table", raw.get("curve", raw.get("theta_psi_curve", [])))
    out: list[tuple[float, float, float]] = []
    for row in _ensure_list(rows):
        if not isinstance(row, Mapping):
            continue
        psi = float(_pick(row, ("pressure_head", "psi", "ψ", "suction_head"), 0.0) or 0.0)
        theta = float(_pick(row, ("theta", "water_content", "θ"), 1.0) or 1.0)
        kr = float(_pick(row, ("kr", "relative_permeability", "k_relative"), theta) or theta)
        out.append((psi, theta, max(0.0, min(1.0, kr))))
    return tuple(sorted(out, key=lambda item: item[0]))


def _with_unsaturated_preset(seep: Mapping[str, Any]) -> Mapping[str, Any]:
    preset_id = _pick(seep, ("unsaturated_preset", "test_value_preset", "standard_unsaturated"), None)
    unsat = seep.get("unsaturated", seep.get("van_genuchten", seep.get("vg", {})))
    if isinstance(unsat, Mapping) and preset_id in (None, ""):
        preset_id = _pick(unsat, ("preset", "unsaturated_preset", "standard", "standard_preset"), None)
    if preset_id in (None, ""):
        return seep
    preset_key = str(preset_id).strip().lower().replace("-", "_").replace(" ", "_")
    preset_key = _VGFLOW_UNSATURATED_PRESET_ALIASES.get(preset_key, preset_key)
    preset = VGFLOW_UNSATURATED_PUBLIC_PRESETS.get(preset_key)
    if preset is None:
        known = ", ".join(sorted(VGFLOW_UNSATURATED_PUBLIC_PRESETS))
        raise FEM2DError(f"unknown VGFlow2D unsaturated public preset: {preset_id!r}; known presets: {known}")
    preset_values = {key: value for key, value in preset.items() if key not in {"label", "note", "source"}}
    user_values = dict(unsat) if isinstance(unsat, Mapping) else {}
    for meta_key in ("preset", "unsaturated_preset", "standard", "standard_preset"):
        user_values.pop(meta_key, None)
    merged = {**preset_values, **user_values}
    out = dict(seep)
    out["unsaturated"] = merged
    return out


def _vgflow_mapping(cfg: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("vgflow", "vgflow2d", "seepage", "hydro"):
        value = cfg.get(key)
        if isinstance(value, Mapping):
            out.update(dict(value))
    for stage in _ensure_list(cfg.get("stages", cfg.get("steps", []))):
        if isinstance(stage, Mapping) and str(stage.get("type", "")).lower() in {"vgflow2d", "vgflow", "richards", "seepage_flow", "steady_seepage", "transient_seepage"}:
            out.update(dict(stage.get("vgflow", stage.get("seepage", stage))))
            break
    return out


def _problem_type(analysis: Mapping[str, Any], seepage: Mapping[str, Any]) -> str:
    raw = str(seepage.get("problem_type", analysis.get("problem_type", analysis.get("vgflow_problem", analysis.get("type", "vertical"))))).lower().replace("-", "_")
    if "axis" in raw:
        return "axisymmetric"
    if "horizontal" in raw or raw in {"plane", "plan"}:
        return "horizontal"
    return "vertical"


def _is_transient(analysis: Mapping[str, Any], seepage: Mapping[str, Any]) -> bool:
    text = " ".join(str(value) for value in (analysis.get("mode", ""), analysis.get("type", ""), seepage.get("mode", ""), seepage.get("analysis_mode", ""))).lower()
    return bool(seepage.get("transient", False)) or "transient" in text or "nonsteady" in text or "非定常" in text or int(seepage.get("steps", 1) or 1) > 1


def _analysis_times(seepage: Mapping[str, Any], transient: bool) -> list[float]:
    raw = seepage.get("times", seepage.get("time_points"))
    if raw is not None:
        values = [float(v) for v in _ensure_list(raw)]
        return sorted(set(values)) or [0.0]
    if not transient:
        return [float(seepage.get("time", 0.0) or 0.0)]
    dt = float(seepage.get("dt", seepage.get("time_step", 1.0)) or 1.0)
    steps = int(seepage.get("steps", seepage.get("n_steps", 1)) or 1)
    return [dt * (i + 1) for i in range(max(steps, 1))]


def _step_dt(times: list[float], index: int, seepage: Mapping[str, Any]) -> float:
    if index == 0:
        return max(float(seepage.get("dt", seepage.get("time_step", times[0] if times[0] > 0.0 else 1.0)) or 1.0), np.finfo(float).eps)
    return max(times[index] - times[index - 1], np.finfo(float).eps)


def _linear_solver_cfg(seepage: Mapping[str, Any]) -> dict[str, Any]:
    solver = seepage.get("solver", seepage.get("linear", {"method": "direct"}))
    if isinstance(solver, Mapping):
        return dict(solver.get("linear", solver))
    return {"method": str(solver)}


def _initial_total_head(mesh: Mesh2D, seepage: Mapping[str, Any], problem_type: str) -> np.ndarray:
    n = len(mesh.node_ids)
    if "initial_head" in seepage:
        return np.full(n, float(seepage["initial_head"]), dtype=float)
    if "initial_pressure_head" in seepage:
        pressure_head = float(seepage["initial_pressure_head"])
        return np.asarray([_total_head_from_pressure(mesh, nid, pressure_head, problem_type) for nid in mesh.node_ids], dtype=float)
    line = seepage.get("initial_wetting_surface", seepage.get("initial_phreatic_line", seepage.get("initial_waterline")))
    if line is not None:
        points = _xy_points(line)
        return np.asarray([_waterline_y(points, float(mesh.coords[i, 0])) for i in range(n)], dtype=float)
    return np.full(n, float(seepage.get("initial_water_level", seepage.get("water_level", 0.0)) or 0.0), dtype=float)


def _assemble_vgflow_matrices(mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], head: np.ndarray, problem_type: str) -> tuple[csr_matrix, csr_matrix]:
    fast = _assemble_vgflow_matrices_quad4_numba(mesh, materials, head, problem_type)
    if fast is None:
        fast = _assemble_vgflow_matrices_quad8_numba(mesh, materials, head, problem_type)
    if fast is None:
        fast = _assemble_vgflow_matrices_tri3_numba(mesh, materials, head, problem_type)
    if fast is None:
        fast = _assemble_vgflow_matrices_tri6_numba(mesh, materials, head, problem_type)
    if fast is not None:
        return fast
    return _assemble_vgflow_matrices_python(mesh, materials, head, problem_type)


def _assemble_vgflow_matrices_quad4_numba(mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], head: np.ndarray, problem_type: str) -> tuple[csr_matrix, csr_matrix] | None:
    active_elements = [element for element in mesh.elements if element.active]
    shape = (len(mesh.node_ids), len(mesh.node_ids))
    if not active_elements:
        return csr_matrix(shape), csr_matrix(shape)
    if any(element.type.upper() != "QUAD4" or len(element.nodes) != 4 for element in active_elements):
        return None
    material_names = list(materials)
    material_index = {name: i for i, name in enumerate(material_names)}
    model_codes: list[int] = []
    for name in material_names:
        code = _vgflow_material_model_code(materials[name])
        if code is None:
            return None
        model_codes.append(code)
    table_offsets, table_values = _vgflow_material_table_arrays([materials[name] for name in material_names])

    node_index = mesh.node_index
    connectivity = np.empty((len(active_elements), 4), dtype=np.int64)
    material_ids = np.empty(len(active_elements), dtype=np.int64)
    for i, element in enumerate(active_elements):
        connectivity[i, :] = [node_index[nid] for nid in element.nodes]
        material_ids[i] = material_index[element.material]

    kx = np.asarray([materials[name].kx for name in material_names], dtype=np.float64)
    ky = np.asarray([materials[name].ky for name in material_names], dtype=np.float64)
    specific_storage = np.asarray([materials[name].specific_storage for name in material_names], dtype=np.float64)
    angle_rad = np.radians(np.asarray([materials[name].angle_deg for name in material_names], dtype=np.float64))
    alpha = np.asarray([materials[name].alpha for name in material_names], dtype=np.float64)
    n_values = np.asarray([materials[name].n for name in material_names], dtype=np.float64)
    theta_r = np.asarray([materials[name].theta_r for name in material_names], dtype=np.float64)
    theta_s = np.asarray([materials[name].theta_s for name in material_names], dtype=np.float64)
    rows, cols, k_data, m_data, invalid, det = vgflow_quad4_assembly_triplets_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        connectivity,
        material_ids,
        np.ascontiguousarray(head, dtype=np.float64),
        kx,
        ky,
        specific_storage,
        angle_rad,
        np.asarray(model_codes, dtype=np.int64),
        table_offsets,
        table_values,
        alpha,
        n_values,
        theta_r,
        theta_s,
        _problem_type_is_horizontal(problem_type),
        problem_type == "axisymmetric",
    )
    if invalid >= 0:
        element = active_elements[int(invalid)]
        raise FEM2DError(f"{element.type}: detJ must be positive for VGFlow2D, got {det:.6e}")
    return coo_matrix((k_data, (rows, cols)), shape=shape).tocsr(), coo_matrix((m_data, (rows, cols)), shape=shape).tocsr()


def _assemble_vgflow_matrices_quad8_numba(mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], head: np.ndarray, problem_type: str) -> tuple[csr_matrix, csr_matrix] | None:
    active_elements = [element for element in mesh.elements if element.active]
    shape = (len(mesh.node_ids), len(mesh.node_ids))
    if not active_elements:
        return csr_matrix(shape), csr_matrix(shape)
    if any(element.type.upper() != "QUAD8" or len(element.nodes) != 8 for element in active_elements):
        return None
    material_names = list(materials)
    material_index = {name: i for i, name in enumerate(material_names)}
    model_codes: list[int] = []
    for name in material_names:
        code = _vgflow_material_model_code(materials[name])
        if code is None:
            return None
        model_codes.append(code)
    table_offsets, table_values = _vgflow_material_table_arrays([materials[name] for name in material_names])

    node_index = mesh.node_index
    connectivity = np.empty((len(active_elements), 8), dtype=np.int64)
    material_ids = np.empty(len(active_elements), dtype=np.int64)
    for i, element in enumerate(active_elements):
        connectivity[i, :] = [node_index[nid] for nid in element.nodes]
        material_ids[i] = material_index[element.material]

    kx = np.asarray([materials[name].kx for name in material_names], dtype=np.float64)
    ky = np.asarray([materials[name].ky for name in material_names], dtype=np.float64)
    specific_storage = np.asarray([materials[name].specific_storage for name in material_names], dtype=np.float64)
    angle_rad = np.radians(np.asarray([materials[name].angle_deg for name in material_names], dtype=np.float64))
    alpha = np.asarray([materials[name].alpha for name in material_names], dtype=np.float64)
    n_values = np.asarray([materials[name].n for name in material_names], dtype=np.float64)
    theta_r = np.asarray([materials[name].theta_r for name in material_names], dtype=np.float64)
    theta_s = np.asarray([materials[name].theta_s for name in material_names], dtype=np.float64)
    rows, cols, k_data, m_data, invalid, det = vgflow_quad8_assembly_triplets_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        connectivity,
        material_ids,
        np.ascontiguousarray(head, dtype=np.float64),
        kx,
        ky,
        specific_storage,
        angle_rad,
        np.asarray(model_codes, dtype=np.int64),
        table_offsets,
        table_values,
        alpha,
        n_values,
        theta_r,
        theta_s,
        _problem_type_is_horizontal(problem_type),
        problem_type == "axisymmetric",
    )
    if invalid >= 0:
        element = active_elements[int(invalid)]
        raise FEM2DError(f"{element.type}: detJ must be positive for VGFlow2D, got {det:.6e}")
    return coo_matrix((k_data, (rows, cols)), shape=shape).tocsr(), coo_matrix((m_data, (rows, cols)), shape=shape).tocsr()


def _assemble_vgflow_matrices_tri3_numba(mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], head: np.ndarray, problem_type: str) -> tuple[csr_matrix, csr_matrix] | None:
    return _assemble_vgflow_matrices_fixed_element_numba(mesh, materials, head, problem_type, "TRI3", 3, vgflow_tri3_assembly_triplets_numba)


def _assemble_vgflow_matrices_tri6_numba(mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], head: np.ndarray, problem_type: str) -> tuple[csr_matrix, csr_matrix] | None:
    return _assemble_vgflow_matrices_fixed_element_numba(mesh, materials, head, problem_type, "TRI6", 6, vgflow_tri6_assembly_triplets_numba)


def _assemble_vgflow_matrices_fixed_element_numba(
    mesh: Mesh2D,
    materials: Mapping[str, VgFlowMaterial],
    head: np.ndarray,
    problem_type: str,
    element_type: str,
    node_count: int,
    kernel: Any,
) -> tuple[csr_matrix, csr_matrix] | None:
    active_elements = [element for element in mesh.elements if element.active]
    shape = (len(mesh.node_ids), len(mesh.node_ids))
    if not active_elements:
        return csr_matrix(shape), csr_matrix(shape)
    if any(element.type.upper() != element_type or len(element.nodes) != node_count for element in active_elements):
        return None
    material_names = list(materials)
    material_index = {name: i for i, name in enumerate(material_names)}
    model_codes: list[int] = []
    for name in material_names:
        code = _vgflow_material_model_code(materials[name])
        if code is None:
            return None
        model_codes.append(code)
    table_offsets, table_values = _vgflow_material_table_arrays([materials[name] for name in material_names])

    node_index = mesh.node_index
    connectivity = np.empty((len(active_elements), node_count), dtype=np.int64)
    material_ids = np.empty(len(active_elements), dtype=np.int64)
    for i, element in enumerate(active_elements):
        connectivity[i, :] = [node_index[nid] for nid in element.nodes]
        material_ids[i] = material_index[element.material]

    kx = np.asarray([materials[name].kx for name in material_names], dtype=np.float64)
    ky = np.asarray([materials[name].ky for name in material_names], dtype=np.float64)
    specific_storage = np.asarray([materials[name].specific_storage for name in material_names], dtype=np.float64)
    angle_rad = np.radians(np.asarray([materials[name].angle_deg for name in material_names], dtype=np.float64))
    alpha = np.asarray([materials[name].alpha for name in material_names], dtype=np.float64)
    n_values = np.asarray([materials[name].n for name in material_names], dtype=np.float64)
    theta_r = np.asarray([materials[name].theta_r for name in material_names], dtype=np.float64)
    theta_s = np.asarray([materials[name].theta_s for name in material_names], dtype=np.float64)
    rows, cols, k_data, m_data, invalid, det = kernel(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        connectivity,
        material_ids,
        np.ascontiguousarray(head, dtype=np.float64),
        kx,
        ky,
        specific_storage,
        angle_rad,
        np.asarray(model_codes, dtype=np.int64),
        table_offsets,
        table_values,
        alpha,
        n_values,
        theta_r,
        theta_s,
        _problem_type_is_horizontal(problem_type),
        problem_type == "axisymmetric",
    )
    if invalid >= 0:
        element = active_elements[int(invalid)]
        raise FEM2DError(f"{element.type}: detJ must be positive for VGFlow2D, got {det:.6e}")
    return coo_matrix((k_data, (rows, cols)), shape=shape).tocsr(), coo_matrix((m_data, (rows, cols)), shape=shape).tocsr()


def _assemble_vgflow_matrices_python(mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], head: np.ndarray, problem_type: str) -> tuple[csr_matrix, csr_matrix]:
    rows: list[int] = []
    cols: list[int] = []
    k_data: list[float] = []
    m_data: list[float] = []
    node_index = mesh.node_index
    for element in mesh.elements:
        if not element.active:
            continue
        conn = [node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        material = materials[element.material]
        pressure_heads = np.asarray([_pressure_head(mesh, mesh.node_ids[idx], head[idx], problem_type) for idx in conn], dtype=float)
        avg_pressure = float(np.mean(pressure_heads))
        state = _water_state(material, avg_pressure)
        tensor = _permeability_tensor(material, state["kr"])
        nnode = len(conn)
        ke = np.zeros((nnode, nnode), dtype=float)
        me = np.zeros((nnode, nnode), dtype=float)
        for xi, eta, weight in integration_points(element.type, "FULL"):
            N, dN_dnatural = shape_functions(element.type, xi, eta)
            jac = dN_dnatural @ coords
            detJ = float(np.linalg.det(jac))
            if detJ <= 0.0:
                raise FEM2DError(f"{element.type}: detJ must be positive for VGFlow2D, got {detJ:.6e}")
            grad = np.linalg.inv(jac) @ dN_dnatural
            scale = detJ * weight
            if problem_type == "axisymmetric":
                radius = max(float(N @ coords[:, 0]), np.finfo(float).eps)
                scale *= 2.0 * math.pi * radius
            ke += grad.T @ tensor @ grad * scale
            storage = material.specific_storage + state["capacity"]
            me += storage * np.outer(N, N) * scale
        for a, row in enumerate(conn):
            for b, col in enumerate(conn):
                rows.append(row)
                cols.append(col)
                k_data.append(float(ke[a, b]))
                m_data.append(float(me[a, b]))
    shape = (len(mesh.node_ids), len(mesh.node_ids))
    return coo_matrix((k_data, (rows, cols)), shape=shape).tocsr(), coo_matrix((m_data, (rows, cols)), shape=shape).tocsr()


def _permeability_tensor(material: VgFlowMaterial, kr: float) -> np.ndarray:
    angle = math.radians(material.angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    local = np.diag([material.kx * kr, material.ky * kr])
    return rot @ local @ rot.T


def _vgflow_material_model_code(material: VgFlowMaterial) -> int | None:
    if material.unsaturated_model == "table" and material.table:
        return VGFLOW_MATERIAL_TABLE
    if material.unsaturated_model == "van_genuchten":
        return VGFLOW_MATERIAL_VAN_GENUCHTEN
    if material.unsaturated_model == "saturated" or not material.table:
        return VGFLOW_MATERIAL_SATURATED
    return None


def _vgflow_material_table_arrays(materials: list[VgFlowMaterial]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    rows: list[tuple[float, float, float]] = []
    for material in materials:
        rows.extend(material.table)
        offsets.append(len(rows))
    if rows:
        values = np.asarray(rows, dtype=np.float64)
    else:
        values = np.zeros((1, 3), dtype=np.float64)
    return np.asarray(offsets, dtype=np.int64), np.ascontiguousarray(values, dtype=np.float64)


def _problem_type_is_horizontal(problem_type: str) -> bool:
    return problem_type in {"horizontal", "plane", "plan"}


def _water_state(material: VgFlowMaterial, pressure_head: float) -> dict[str, float]:
    if material.unsaturated_model == "table" and material.table:
        table = np.ascontiguousarray(np.asarray(material.table, dtype=np.float64))
        psi, theta, saturation, kr, capacity = vgflow_table_state_numba(table, 0, table.shape[0], float(pressure_head), float(material.theta_r), float(material.theta_s))
        return {"pressure_head": psi, "theta": theta, "saturation": saturation, "kr": kr, "capacity": capacity}
    code = _vgflow_material_model_code(material)
    if code is None:
        code = VGFLOW_MATERIAL_SATURATED
    theta, saturation, kr, capacity = vgflow_water_state_numba(
        code,
        float(pressure_head),
        float(material.alpha),
        float(material.n),
        float(material.theta_r),
        float(material.theta_s),
    )
    return {"pressure_head": pressure_head, "theta": theta, "saturation": saturation, "kr": kr, "capacity": capacity}


def _pressure_head_array(mesh: Mesh2D, total_head: np.ndarray, problem_type: str) -> np.ndarray:
    head = np.asarray(total_head, dtype=float)
    if _problem_type_is_horizontal(problem_type):
        return head.copy()
    return head - mesh.coords[:, 1]


def _water_state_arrays(material: VgFlowMaterial, pressure_heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if material.unsaturated_model == "table" and material.table:
        _offsets, table_values = _vgflow_material_table_arrays([material])
        _psi, theta, saturation, _kr, _capacity = vgflow_table_state_array_numba(
            table_values,
            0,
            table_values.shape[0],
            np.ascontiguousarray(pressure_heads, dtype=np.float64),
            float(material.theta_r),
            float(material.theta_s),
        )
        return theta, saturation
    code = _vgflow_material_model_code(material)
    if code is None:
        code = VGFLOW_MATERIAL_SATURATED
    theta, saturation, _kr, _capacity = vgflow_water_state_array_numba(
        code,
        np.ascontiguousarray(pressure_heads, dtype=np.float64),
        float(material.alpha),
        float(material.n),
        float(material.theta_r),
        float(material.theta_s),
    )
    return theta, saturation


def _table_state(table: tuple[tuple[float, float, float], ...], pressure_head: float) -> tuple[float, float, float]:
    if pressure_head <= table[0][0]:
        return table[0]
    if pressure_head >= table[-1][0]:
        return table[-1]
    for left, right in zip(table[:-1], table[1:]):
        if left[0] <= pressure_head <= right[0]:
            t = (pressure_head - left[0]) / max(right[0] - left[0], np.finfo(float).eps)
            return pressure_head, left[1] + t * (right[1] - left[1]), left[2] + t * (right[2] - left[2])
    return table[-1]


def _table_capacity(table: tuple[tuple[float, float, float], ...], pressure_head: float) -> float:
    if len(table) < 2:
        return 0.0
    nearest = min(range(len(table) - 1), key=lambda i: abs(0.5 * (table[i][0] + table[i + 1][0]) - pressure_head))
    left, right = table[nearest], table[nearest + 1]
    return max(0.0, (right[1] - left[1]) / max(right[0] - left[0], np.finfo(float).eps))


def _theta_saturation(material: VgFlowMaterial, theta: float) -> float:
    return max(0.0, min(1.0, (theta - material.theta_r) / max(material.theta_s - material.theta_r, np.finfo(float).eps)))


def _head_constraints(mesh: Mesh2D, seepage: Mapping[str, Any], time: float, problem_type: str) -> dict[int, float]:
    specs: list[Any] = []
    for key in ("head_bcs", "known_head_bcs", "water_head_bcs", "fixed_head_bcs", "water_level_bcs", "pressure_head_bcs"):
        specs.extend(_ensure_list(seepage.get(key, [])))
    specs.extend([bc for bc in _ensure_list(seepage.get("boundary_conditions", seepage.get("bc", []))) if isinstance(bc, Mapping) and _bc_kind(bc) in {"head", "known_head", "water_level", "pressure_head"}])
    fixed: dict[int, float] = {}
    for raw in specs:
        if not isinstance(raw, Mapping):
            continue
        for nid in _nodes_from_spec(mesh, raw):
            fixed[mesh.node_index[nid]] = _constraint_head_value(mesh, nid, raw, time, problem_type)
    return fixed


def _constraint_head_value(mesh: Mesh2D, nid: str, spec: Mapping[str, Any], time: float, problem_type: str) -> float:
    if "pressure_head" in spec or "pressure_head_curve" in spec or _curve_file_from_spec(spec, "pressure_head") not in (None, ""):
        return _total_head_from_pressure(mesh, nid, _value_at_time(spec, "pressure_head", time), problem_type)
    if _bc_kind(spec) == "pressure_head":
        return _total_head_from_pressure(mesh, nid, _value_at_time(spec, "value", time), problem_type)
    for key in ("head", "total_head", "water_head", "water_level", "level", "value"):
        if key in spec or f"{key}_curve" in spec or _curve_file_from_spec(spec, key) not in (None, ""):
            return _value_at_time(spec, key, time)
    return 0.0


def _total_head_from_pressure_values(mesh: Mesh2D, node_indices: np.ndarray, pressure_head: float, problem_type: str) -> np.ndarray:
    if _problem_type_is_horizontal(problem_type):
        return np.full(node_indices.shape, float(pressure_head), dtype=float)
    return float(pressure_head) + mesh.coords[node_indices, 1]


def _active_seepage_constraints(mesh: Mesh2D, seepage: Mapping[str, Any], head: np.ndarray, time: float, problem_type: str) -> dict[int, float]:
    fixed: dict[int, float] = {}
    for spec in _ensure_list(seepage.get("seepage_faces", seepage.get("seepage_face_bcs", []))):
        if not isinstance(spec, Mapping):
            continue
        drain_head = _value_at_time(spec, "head", time, default=_value_at_time(spec, "water_level", time, default=0.0))
        nodes = _nodes_from_spec(mesh, spec)
        if not nodes:
            continue
        indices = np.asarray([mesh.node_index[nid] for nid in nodes], dtype=np.int64)
        if "pressure_head" in spec:
            limits = _total_head_from_pressure_values(mesh, indices, _value_at_time(spec, "pressure_head", time), problem_type)
        else:
            limits = np.full(indices.shape, drain_head, dtype=float)
        active = head[indices] > limits + 1.0e-12
        for idx, limit in zip(indices[active], limits[active]):
            fixed[int(idx)] = float(limit)
    return fixed


def _boundary_rhs(mesh: Mesh2D, seepage: Mapping[str, Any], time: float) -> np.ndarray:
    rhs = np.zeros(len(mesh.node_ids), dtype=float)
    flux_specs: list[Any] = []
    for key in ("flux_bcs", "pore_flux_bcs", "flow_bcs", "rainfall_bcs"):
        flux_specs.extend(_ensure_list(seepage.get(key, [])))
    if "rainfall" in seepage:
        rainfall = seepage["rainfall"]
        if isinstance(rainfall, Mapping):
            flux_specs.append({"set": rainfall.get("set", "top"), **rainfall})
        else:
            flux_specs.append({"set": "top", "rainfall": rainfall})
    for raw in flux_specs:
        if not isinstance(raw, Mapping):
            continue
        q = _flux_value(raw, time)
        _add_flux_edges_to_rhs(rhs, mesh, _pressure_edges(mesh, raw), q)
    for raw in _ensure_list(seepage.get("point_sources", seepage.get("point_source_bcs", []))):
        if not isinstance(raw, Mapping):
            continue
        value = _value_at_time(raw, "value", time, default=_value_at_time(raw, "flow", time, default=0.0))
        nodes = _nodes_from_spec(mesh, raw)
        if not nodes:
            continue
        indices = np.asarray([mesh.node_index[nid] for nid in nodes], dtype=np.int64)
        np.add.at(rhs, indices, value / max(len(indices), 1))
    return rhs


def _add_flux_edges_to_rhs(rhs: np.ndarray, mesh: Mesh2D, edges: list[tuple[str, ...]], q: float) -> None:
    node_index = mesh.node_index
    for node_count, weights in ((2, np.array([0.5, 0.5], dtype=float)), (3, np.array([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=float))):
        grouped = [edge for edge in edges if len(edge) == node_count]
        if not grouped:
            continue
        indices = np.asarray([[node_index[nid] for nid in edge] for edge in grouped], dtype=np.int64)
        points = mesh.coords[indices]
        lengths = np.linalg.norm(np.diff(points, axis=1), axis=2).sum(axis=1)
        if np.any(lengths <= 0.0):
            raise FEM2DError("edge length must be positive")
        values = float(q) * lengths[:, None] * weights[None, :]
        np.add.at(rhs, indices.ravel(), values.ravel())
    unsupported = [edge for edge in edges if len(edge) not in {2, 3}]
    if unsupported:
        _edge_lumped_weights(unsupported[0])


def _flux_value(spec: Mapping[str, Any], time: float) -> float:
    if "rainfall" in spec or _bc_kind(spec) == "rainfall":
        value = _value_at_time(spec, "rainfall", time, default=_value_at_time(spec, "value", time, default=0.0))
        unit = str(spec.get("unit", spec.get("rainfall_unit", ""))).lower().replace(" ", "")
        if unit in {"mm/hr", "mm/h", "mmhour"}:
            return value / 1000.0 / 3600.0
        if unit in {"mm/s"}:
            return value / 1000.0
        return value
    return _value_at_time(spec, "flux", time, default=_value_at_time(spec, "q", time, default=_value_at_time(spec, "value", time, default=0.0)))


def _nodes_from_spec(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[str]:
    try:
        return _target_nodes(mesh, spec)
    except FEM2DError:
        nodes: list[str] = []
        for edge in _pressure_edges(mesh, spec):
            nodes.extend(edge)
        return list(dict.fromkeys(nodes))


def _bc_kind(spec: Mapping[str, Any]) -> str:
    return str(spec.get("type", spec.get("kind", ""))).lower().strip().replace("-", "_")


def read_vgflow_curve_file(path: str | Path, *, value_field: str = "value") -> list[dict[str, float]]:
    curve_path = Path(path).expanduser()
    if not curve_path.exists():
        raise FEM2DError(f"VGFlow2D boundary curve file not found: {curve_path}")
    stat = curve_path.stat()
    cache_key = (str(curve_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size), value_field)
    cached = _CURVE_FILE_CACHE.get(cache_key)
    if cached is not None:
        return [dict(row) for row in cached]

    rows: list[dict[str, float]] = []
    header: list[str] | None = None
    with curve_path.open(encoding="utf-8-sig", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            tokens = _split_curve_line(line)
            if not tokens:
                continue
            if header is None and len(tokens) >= 2 and _is_float_token(tokens[0]) and _is_float_token(tokens[1]):
                rows.append({"time": float(tokens[0]), value_field: float(tokens[1])})
                continue
            if header is None:
                header = [_curve_field_name(token, value_field) for token in tokens]
                continue
            try:
                record = {header[i]: tokens[i] for i in range(min(len(header), len(tokens)))}
                rows.append(_curve_record_to_row(record, value_field))
            except (KeyError, ValueError) as exc:
                raise FEM2DError(f"invalid VGFlow2D boundary curve row at {curve_path}:{line_no}: {line.strip()}") from exc

    rows.sort(key=lambda row: row["time"])
    _CURVE_FILE_CACHE[cache_key] = [dict(row) for row in rows]
    return rows


def write_vgflow_curve_file(rows: Any, path: str | Path, *, value_field: str = "value") -> None:
    curve_path = Path(path)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", value_field])
        for raw in _ensure_list(rows):
            if not isinstance(raw, Mapping):
                continue
            time_value = float(raw.get("time", raw.get("t", 0.0)) or 0.0)
            if value_field in raw:
                value = raw[value_field]
            elif "value" in raw:
                value = raw["value"]
            else:
                value = next((val for key, val in raw.items() if key not in {"time", "t"}), 0.0)
            writer.writerow([time_value, float(value)])


def _split_curve_line(line: str) -> list[str]:
    text = line.strip()
    if not text or text.startswith("#"):
        return []
    for marker in ("#", "//"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if not text:
        return []
    for delimiter in (",", ";", "\t"):
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]
    return text.split()


def _curve_field_name(token: str, value_field: str) -> str:
    base = token.strip().lstrip("\ufeff").lower()
    for separator in ("(", "[", "{"):
        base = base.split(separator, 1)[0].strip()
    aliases = {
        "t": "time",
        "time": "time",
        "sec": "time",
        "s": "time",
        "hr": "time",
        "hour": "time",
        "h": "time",
        "\u6642\u523b": "time",
        "\u6642\u9593": "time",
        "head": "head",
        "h_total": "head",
        "total_head": "head",
        "\u6c34\u982d": "head",
        "\u5168\u6c34\u982d": "head",
        "level": "water_level",
        "water_level": "water_level",
        "\u6c34\u4f4d": "water_level",
        "pressure_head": "pressure_head",
        "psi": "pressure_head",
        "\u5727\u529b\u6c34\u982d": "pressure_head",
        "rain": "rainfall",
        "rainfall": "rainfall",
        "\u96e8\u91cf": "rainfall",
        "\u964d\u96e8": "rainfall",
        "\u964d\u96e8\u91cf": "rainfall",
        "flux": "flux",
        "q": "q",
        "flow": "flow",
        "\u6d41\u91cf": "flow",
        "\u96c6\u6c34\u91cf": "flow",
        "value": value_field,
    }
    return aliases.get(base, base)


def _curve_record_to_row(record: Mapping[str, str], value_field: str) -> dict[str, float]:
    if "time" not in record:
        raise KeyError("time")
    candidate_keys = [value_field, "value", "head", "water_level", "pressure_head", "rainfall", "flux", "q", "flow"]
    candidate_keys.extend(key for key in record if key not in {"time", *candidate_keys})
    for key in candidate_keys:
        if key in record and record[key] not in (None, ""):
            return {"time": float(record["time"]), value_field: float(record[key])}
    raise KeyError(value_field)


def _is_float_token(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _curve_pairs_for_key(spec: Mapping[str, Any], key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    curve_path = _curve_file_from_spec(spec, key)
    if curve_path not in (None, ""):
        pairs.extend((row["time"], row[key]) for row in read_vgflow_curve_file(curve_path, value_field=key))
    curve = spec.get(f"{key}_curve", spec.get("curve", spec.get("time_series")))
    if curve is not None:
        pairs.extend(_inline_curve_pairs(curve, key))
    return sorted(pairs)


def _curve_file_from_spec(spec: Mapping[str, Any], key: str) -> Any:
    for field in (f"{key}_curve_file", f"{key}_time_series_file", f"{key}_file", "curve_file", "time_series_file"):
        value = spec.get(field)
        if value not in (None, ""):
            return value
    if "file" in spec and any(marker in spec for marker in ("curve", "time_series", "curve_file", "time_series_file")):
        return spec["file"]
    return None


def _inline_curve_pairs(curve: Any, key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in _ensure_list(curve):
        if isinstance(row, Mapping):
            value = row.get(key, row.get("value", row.get("head", row.get("water_level", 0.0))))
            pairs.append((float(row.get("time", row.get("t", 0.0)) or 0.0), float(value or 0.0)))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            pairs.append((float(row[0]), float(row[1])))
    return pairs


def _value_at_time(spec: Mapping[str, Any], key: str, time: float, default: float | None = None) -> float:
    pairs = _curve_pairs_for_key(spec, key)
    if pairs:
        if time <= pairs[0][0]:
            return pairs[0][1]
        if time >= pairs[-1][0]:
            return pairs[-1][1]
        for left, right in zip(pairs[:-1], pairs[1:]):
            if left[0] <= time <= right[0]:
                t = (time - left[0]) / max(right[0] - left[0], np.finfo(float).eps)
                return left[1] + t * (right[1] - left[1])
    value = spec.get(key, default)
    if value is None:
        return 0.0
    return float(value)


def _pressure_head(mesh: Mesh2D, nid: str, total_head: float, problem_type: str) -> float:
    return pressure_head_from_total(mesh, nid, total_head, problem_type)


def _total_head_from_pressure(mesh: Mesh2D, nid: str, pressure_head: float, problem_type: str) -> float:
    return total_head_from_pressure(mesh, nid, pressure_head, problem_type)


def _write_vgflow_outputs(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, VgFlowMaterial],
    steps: list[VgFlowStepResult],
    problem_type: str,
    gamma_w: float,
    warnings: list[str],
    seepage: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "node_csv": str(out / "vgflow_node_results.csv"),
        "element_csv": str(out / "vgflow_element_results.csv"),
        "prs": str(out / "vgflow_waterline.PRS"),
        "ptn": str(out / "vgflow_potential.PTN"),
        "boundary_curves": str(out / "vgflow_boundary_curves.csv"),
        "html": str(out / "vgflow_post_index.html"),
    }
    _write_node_results(Path(paths["node_csv"]), mesh, materials, steps, problem_type, gamma_w)
    _write_element_results(Path(paths["element_csv"]), mesh, materials, steps, problem_type)
    _write_prs(Path(paths["prs"]), mesh, steps, problem_type)
    _write_ptn(Path(paths["ptn"]), mesh, materials, steps, problem_type)
    _write_boundary_curves(Path(paths["boundary_curves"]), seepage)
    _write_vgflow_html(Path(paths["html"]), mesh, steps, warnings, seepage)
    paths.update(write_vgflow_curve_package(out, seepage, read_vgflow_curve_file))
    paths.update(write_vgflow_exchange_outputs(out, mesh, steps, problem_type, seepage))
    paths.update(write_vgflow_boundary_diagnostics(out, mesh, materials, [step.time for step in steps], seepage, problem_type, read_vgflow_curve_file))
    paths.update(write_vgflow_cad_import_outputs(out, seepage))
    paths.update(write_vgflow_pre_outputs(out, mesh, seepage))
    paths.update(write_vgflow_mesh_outputs(out, mesh, materials, steps, problem_type, seepage))
    paths.update(write_vgflow_post_outputs(out, mesh, materials, steps, problem_type, gamma_w, seepage))
    paths.update(write_vgflow_ui_profile_outputs(out, mesh, seepage, paths))
    paths.update(write_vgflow_design_checks(out, mesh, materials, steps, problem_type, seepage))
    paths.update(write_vgflow_geofeas_coupling_outputs(out, mesh, materials, steps, problem_type, gamma_w, seepage, paths))
    paths.update(write_vgflow_report_bundle(out, mesh, materials, steps, problem_type, seepage, paths, warnings))
    paths.update(write_vgflow_alternative_spec(out, mesh, materials, steps, problem_type, seepage, paths, warnings))
    paths.update(write_vgflow_project_package(out, mesh, materials, steps, problem_type, seepage, paths))
    return paths


def _write_boundary_curves(path: Path, seepage: Mapping[str, Any]) -> None:
    fields = ["boundary_group", "boundary_index", "value_key", "source", "time", "value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, index, spec in _all_boundary_specs(seepage):
            for key in _boundary_curve_keys(spec):
                source = str(_curve_file_from_spec(spec, key) or "inline")
                for time_value, value in _curve_pairs_for_key(spec, key):
                    writer.writerow(
                        {
                            "boundary_group": group,
                            "boundary_index": index,
                            "value_key": key,
                            "source": source,
                            "time": time_value,
                            "value": value,
                        }
                    )


def _all_boundary_specs(seepage: Mapping[str, Any]) -> list[tuple[str, int, Mapping[str, Any]]]:
    specs: list[tuple[str, int, Mapping[str, Any]]] = []
    for group in (
        "known_head_bcs",
        "head_bcs",
        "water_level_bcs",
        "pressure_head_bcs",
        "flux_bcs",
        "pore_flux_bcs",
        "flow_bcs",
        "rainfall_bcs",
        "point_sources",
        "point_source_bcs",
        "seepage_faces",
    ):
        for index, raw in enumerate(_ensure_list(seepage.get(group, []))):
            if isinstance(raw, Mapping):
                specs.append((group, index, raw))
    if "rainfall" in seepage:
        rainfall = seepage["rainfall"]
        if isinstance(rainfall, Mapping):
            specs.append(("rainfall", 0, {"set": rainfall.get("set", "top"), **rainfall}))
        else:
            specs.append(("rainfall", 0, {"set": "top", "rainfall": rainfall}))
    return specs


def _boundary_curve_keys(spec: Mapping[str, Any]) -> list[str]:
    candidates = ("head", "water_level", "pressure_head", "rainfall", "flux", "q", "flow", "value")
    keys = [
        key
        for key in candidates
        if any(spec.get(field) is not None for field in (f"{key}_curve", f"{key}_curve_file", f"{key}_time_series_file", f"{key}_file"))
    ]
    if keys:
        return keys
    if not any(spec.get(field) is not None for field in ("curve", "time_series", "curve_file", "time_series_file")):
        return []
    kind = _bc_kind(spec)
    if kind == "rainfall" or "rainfall" in spec:
        return ["rainfall"]
    if any(key in spec for key in ("head", "total_head")) or kind in {"head", "known_head"}:
        return ["head"]
    if "water_level" in spec or kind == "water_level":
        return ["water_level"]
    if "pressure_head" in spec or kind == "pressure_head":
        return ["pressure_head"]
    if any(key in spec for key in ("flux", "q")) or kind in {"flux", "flow"}:
        return ["flux"]
    return ["value"]


def _write_node_results(path: Path, mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], steps: list[VgFlowStepResult], problem_type: str, gamma_w: float) -> None:
    fields = ["step", "time", "node_id", "x", "y", "total_head_m", "pressure_head_m", "pore_pressure_kpa", "saturation", "water_content"]
    default_material = next(iter(materials.values()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for step in steps:
            pressure_heads = _pressure_head_array(mesh, step.total_head, problem_type)
            theta, saturation = _water_state_arrays(default_material, pressure_heads)
            writer.writerows(
                (
                    step.index,
                    step.time,
                    nid,
                    float(mesh.coords[i, 0]),
                    float(mesh.coords[i, 1]),
                    float(step.total_head[i]),
                    float(pressure_heads[i]),
                    gamma_w * float(pressure_heads[i]),
                    float(saturation[i]),
                    float(theta[i]),
                )
                for i, nid in enumerate(mesh.node_ids)
            )


def _write_element_results(path: Path, mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], steps: list[VgFlowStepResult], problem_type: str) -> None:
    fields = ["step", "time", "element_id", "x", "y", "hydraulic_gradient_x", "hydraulic_gradient_y", "velocity_x_m_s", "velocity_y_m_s", "velocity_abs_m_s", "flow_rate_m3_s_per_m"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for step in steps:
            writer.writerows(
                (
                    row["step"],
                    row["time"],
                    row["element_id"],
                    row["x"],
                    row["y"],
                    row["hydraulic_gradient_x"],
                    row["hydraulic_gradient_y"],
                    row["velocity_x_m_s"],
                    row["velocity_y_m_s"],
                    row["velocity_abs_m_s"],
                    row["velocity_abs_m_s"],
                )
                for row in vgflow_element_post_fields(mesh, materials, step, problem_type)
            )


def _write_prs(path: Path, mesh: Mesh2D, steps: list[VgFlowStepResult], problem_type: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GeoFEM VGFlow2D public substitute PRS", "waterline pressure_head=0"])
        writer.writerow(["step", "time", "x", "y"])
        for step in steps:
            for x, y in waterline_points_from_total_head(mesh, step.total_head, problem_type):
                writer.writerow([step.index, step.time, x, y])


def _write_ptn(path: Path, mesh: Mesh2D, materials: Mapping[str, VgFlowMaterial], steps: list[VgFlowStepResult], problem_type: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GeoFEM VGFlow2D public substitute PTN", "element-center total_head"])
        writer.writerow(["step", "time", "element_id", "x", "y", "total_head_m"])
        for step in steps:
            for row in element_potential_points(mesh, step.total_head):
                writer.writerow([step.index, step.time, row["element_id"], row["x"], row["y"], row["total_head_m"]])


def _write_vgflow_html(path: Path, mesh: Mesh2D, steps: list[VgFlowStepResult], warnings: list[str], seepage: Mapping[str, Any]) -> None:
    rows = "".join(f"<tr><td>{step.index}</td><td>{step.time:g}</td><td>{step.iteration_count}</td><td>{step.residual_norm:.3e}</td><td>{step.active_seepage_nodes}</td></tr>" for step in steps)
    warn = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)
    path.write_text(
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\"><title>VGFlow2D substitute</title>"
        "<style>body{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style></head><body>"
        "<h1>VGFlow 2D public substitute</h1>"
        f"<p>nodes={len(mesh.node_ids)}, elements={len(mesh.elements)}, mode={html.escape(str(seepage.get('mode', '')))}</p>"
        f"<table><thead><tr><th>step</th><th>time</th><th>iterations</th><th>residual</th><th>active seepage nodes</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<ul>{warn}</ul>"
        "</body></html>",
        encoding="utf-8",
    )


def _xy_points(raw: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in _ensure_list(raw):
        if isinstance(item, Mapping):
            points.append((float(item.get("x", 0.0)), float(item.get("y", item.get("level", 0.0)))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
    if not points:
        raise FEM2DError("initial wetting surface requires at least one point")
    return sorted(points)


def _waterline_y(points: list[tuple[float, float]], x: float) -> float:
    if len(points) == 1 or x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for left, right in zip(points[:-1], points[1:]):
        if left[0] <= x <= right[0]:
            t = (x - left[0]) / max(right[0] - left[0], np.finfo(float).eps)
            return left[1] + t * (right[1] - left[1])
    return points[-1][1]


def _pick(raw: Mapping[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


__all__ = [
    "VgFlowMaterial",
    "VgFlowSolveResult",
    "VgFlowStepResult",
    "is_vgflow2d_config",
    "read_vgflow_curve_file",
    "read_vgflow_project_package",
    "solve_vgflow2d_config",
    "collect_vgflow_boundary_diagnostics",
    "collect_vgflow_cad_import_diagnostics",
    "vgflow_design_template_catalog",
    "vgflow_mesh_template_catalog",
    "vgflow_pre_template_catalog",
    "vgflow_unsaturated_public_catalog",
    "vgflow_materials_from_config",
    "write_vgflow_curve_file",
]
