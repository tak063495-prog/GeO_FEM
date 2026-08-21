"""Workflow artifacts for VGFlow2D and GeoFEAS mesh/result coupling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_mesh import mesh_from_config
from .fem2d_types import FEM2DError, Mesh2D
from .fem2d_utils import _ensure_list
from .hydro_exchange import pressure_head_from_total, waterline_points_from_total_head
from .mesh_coupling import (
    apply_scalar_projection_plan,
    build_scalar_projection_plan,
    build_quad8_pressure_load_report,
    downgrade_quad8_mesh_to_quad4,
    interpolate_quad4_nodal_values_to_quad8,
    project_scalar_values_between_meshes,
    upgrade_quad4_mesh_to_quad8,
    write_mesh_coupling_manifest,
    write_quad8_pressure_load_report,
)
from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


STAGE_FIELDS = ["stage_name", "step", "time", "load_case", "pressure_bc_count", "waterline_point_count", "status"]
HYDRO_BC_FIELDS = ["stage_name", "step", "time", "node_id", "x", "y", "total_head_m", "pressure_head_m", "pore_pressure"]
MATERIAL_FIELDS = ["layer_id", "stable_id", "vgflow_material", "geofeas_material", "vgflow_used", "geofeas_used", "status"]
POST_FIELDS = ["sample_id", "field", "x", "y", "vgflow_value", "geofeas_value", "difference", "abs_difference"]
PROFILE_FIELDS = ["section", "id", "label", "status", "details"]
BENCHMARK_FIELDS = ["metric", "value", "status"]
CONTRACT_FIELDS = ["artifact", "schema", "required_fields", "purpose"]


def build_coupling_ui_profile(
    vgflow_mesh: Mesh2D,
    geofeas_mesh: Mesh2D,
    *,
    artifacts: Mapping[str, str] | None = None,
    vgflow_signature: str | None = None,
    geofeas_signature: str | None = None,
) -> dict[str, Any]:
    common_elements = sorted({element.id for element in vgflow_mesh.elements} & {element.id for element in geofeas_mesh.elements})
    common_nodes = sorted(set(vgflow_mesh.node_ids) & set(geofeas_mesh.node_ids))
    source_box = _bbox(vgflow_mesh)
    target_box = _bbox(geofeas_mesh)
    signature_match = vgflow_signature == geofeas_signature if vgflow_signature and geofeas_signature else None
    return {
        "schema": "geofem.mesh_coupling.ui_profile.v1",
        "profile": "GUI-ready VGFlow2D QUAD4 / GeoFEAS QUAD8 coupling workflow substitute",
        "features": [
            "quad4_quad8_pair_template",
            "mesh_pair_comparison_view",
            "coupling_freshness_check",
            "handoff_artifact_buttons",
        ],
        "state": {
            "vgflow_element_types": _element_type_counts(vgflow_mesh),
            "geofeas_element_types": _element_type_counts(geofeas_mesh),
            "common_element_count": len(common_elements),
            "common_node_count": len(common_nodes),
            "bbox_match": _bbox_match(source_box, target_box),
            "freshness": "match" if signature_match is True else "unknown" if signature_match is None else "stale",
        },
        "template": {
            "id": "vgflow_quad4_geofeas_quad8_shared_cad_blocks",
            "label": "VGFlow QUAD4 / GeoFEAS QUAD8 common CAD-block coupling",
            "commands": [
                "generate_vgflow_quad4_mesh",
                "upgrade_to_geofeas_quad8_mesh",
                "compare_mesh_pair",
                "apply_selected_vgflow_times_to_geofeas_stages",
                "open_coupled_post_comparison",
            ],
        },
        "views": [
            {"id": "mesh_pair", "columns": ["element_id", "vgflow_type", "geofeas_type", "material", "status"]},
            {"id": "node_pair", "columns": ["node_id", "vgflow_xy", "geofeas_xy", "status"]},
            {"id": "freshness", "columns": ["artifact", "signature", "status"]},
        ],
        "diagnostics": {
            "vgflow_bbox": source_box,
            "geofeas_bbox": target_box,
            "missing_geofeas_elements": sorted({element.id for element in vgflow_mesh.elements} - {element.id for element in geofeas_mesh.elements}),
            "missing_vgflow_elements": sorted({element.id for element in geofeas_mesh.elements} - {element.id for element in vgflow_mesh.elements}),
        },
        "artifacts": dict(artifacts or {}),
    }


def build_geofeas_stage_handoff(
    vgflow_mesh: Mesh2D,
    geofeas_mesh: Mesh2D,
    steps: Sequence[Any],
    problem_type: str,
    *,
    selected_steps: Sequence[int] | None = None,
    selected_times: Sequence[float] | None = None,
    gamma_w: float = 9.80665,
    stage_prefix: str = "VGFlow",
    load_case_prefix: str = "VGFlowWater",
    existing_stages: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = _select_steps(steps, selected_steps=selected_steps, selected_times=selected_times)
    existing_by_name = {str(stage.get("name", stage.get("stage_name", ""))): stage for stage in (existing_stages or [])}
    stage_rows: list[dict[str, Any]] = []
    hydro_rows: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for step in selected:
        projected_head = _project_total_head(vgflow_mesh, geofeas_mesh, step.total_head)
        pressure_specs: list[dict[str, Any]] = []
        for nid in geofeas_mesh.node_ids:
            idx = geofeas_mesh.node_index[nid]
            total_head = float(projected_head[nid])
            pressure_head = pressure_head_from_total(geofeas_mesh, nid, total_head, problem_type)
            pore_pressure = float(gamma_w) * pressure_head
            xy = geofeas_mesh.coords[idx]
            pressure_specs.append({"node": nid, "pressure": pore_pressure, "total_head_m": total_head, "pressure_head_m": pressure_head})
            hydro_rows.append(
                {
                    "stage_name": f"{stage_prefix}_{int(step.index):04d}",
                    "step": int(step.index),
                    "time": float(step.time),
                    "node_id": nid,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "total_head_m": total_head,
                    "pressure_head_m": pressure_head,
                    "pore_pressure": pore_pressure,
                }
            )
        head_array = np.asarray([projected_head[nid] for nid in geofeas_mesh.node_ids], dtype=float)
        waterline = [{"x": x, "y": y} for x, y in waterline_points_from_total_head(geofeas_mesh, head_array, problem_type, sort_points=True)]
        stage_name = f"{stage_prefix}_{int(step.index):04d}"
        load_case = f"{load_case_prefix}_{int(step.index):04d}"
        status = "update" if stage_name in existing_by_name else "new"
        stage = {
            "name": stage_name,
            "type": "static",
            "time": float(step.time),
            "load_case": load_case,
            "hydro": {
                "pressure_bcs": pressure_specs,
                "water_level_points": waterline,
                "source": {"product": "VGFlow2D public substitute", "step": int(step.index), "time": float(step.time)},
            },
            "liquefaction": {
                "water_level_points": waterline,
                "source": "vgflow_handoff_waterline",
            },
        }
        stages.append(stage)
        stage_rows.append(
            {
                "stage_name": stage_name,
                "step": int(step.index),
                "time": float(step.time),
                "load_case": load_case,
                "pressure_bc_count": len(pressure_specs),
                "waterline_point_count": len(waterline),
                "status": status,
            }
        )
    return {
        "schema": "geofem.mesh_coupling.geofeas_stage_handoff.v1",
        "features": [
            "transient_vgflow_to_geofeas_stage_generation",
            "nodal_pore_pressure_bcs",
            "waterline_and_liquefaction_water_level",
            "existing_stage_diff_status",
        ],
        "selection": {
            "selected_steps": [int(step.index) for step in selected],
            "selected_times": [float(step.time) for step in selected],
        },
        "stage_rows": stage_rows,
        "hydro_rows": hydro_rows,
        "geofeas_stages": stages,
        "diagnostics": {
            "stage_count": len(stages),
            "pressure_bc_count": len(hydro_rows),
            "updated_stage_count": sum(1 for row in stage_rows if row["status"] == "update"),
            "new_stage_count": sum(1 for row in stage_rows if row["status"] == "new"),
        },
    }


def build_material_layer_dictionary(
    vgflow_mesh: Mesh2D,
    geofeas_mesh: Mesh2D,
    vgflow_materials: Mapping[str, Any],
    geofeas_materials: Mapping[str, Any],
    *,
    layer_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    vg_used = {element.material for element in vgflow_mesh.elements}
    gf_used = {element.material for element in geofeas_mesh.elements}
    rows: list[dict[str, Any]] = []
    if layer_map:
        layer_ids = [str(key) for key in layer_map]
    else:
        layer_ids = sorted(set(vgflow_materials) | set(geofeas_materials) | vg_used | gf_used)
    for layer_id in layer_ids:
        raw = layer_map.get(layer_id, {}) if layer_map else {}
        raw_map = raw if isinstance(raw, Mapping) else {}
        vg_name = str(raw_map.get("vgflow_material", raw_map.get("hydraulic_material", layer_id)))
        gf_name = str(raw_map.get("geofeas_material", raw_map.get("mechanical_material", layer_id)))
        missing: list[str] = []
        if vg_name not in vgflow_materials:
            missing.append("vgflow_material_definition")
        if gf_name not in geofeas_materials:
            missing.append("geofeas_material_definition")
        if vg_name not in vg_used:
            missing.append("vgflow_mesh_usage")
        if gf_name not in gf_used:
            missing.append("geofeas_mesh_usage")
        rows.append(
            {
                "layer_id": layer_id,
                "stable_id": str(raw_map.get("stable_id", f"layer:{layer_id}")),
                "vgflow_material": vg_name,
                "geofeas_material": gf_name,
                "vgflow_used": vg_name in vg_used,
                "geofeas_used": gf_name in gf_used,
                "status": "pass" if not missing else "missing:" + ";".join(missing),
            }
        )
    return {
        "schema": "geofem.mesh_coupling.material_layer_dictionary.v1",
        "features": ["hydraulic_mechanical_material_pairing", "stable_layer_ids", "mesh_usage_diagnostics"],
        "rows": rows,
        "diagnostics": {
            "layer_count": len(rows),
            "warning_count": sum(1 for row in rows if row["status"] != "pass"),
        },
    }


def build_coupled_post_comparison(
    vgflow_mesh: Mesh2D,
    geofeas_mesh: Mesh2D,
    vgflow_fields: Mapping[str, Mapping[str, float] | Sequence[float] | np.ndarray],
    geofeas_fields: Mapping[str, Mapping[str, float] | Sequence[float] | np.ndarray],
    *,
    field_pairs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    pairs = dict(field_pairs or {key: key for key in vgflow_fields if key in geofeas_fields})
    rows: list[dict[str, Any]] = []
    field_summary: list[dict[str, Any]] = []
    projection_plan = build_scalar_projection_plan(vgflow_mesh, geofeas_mesh)
    for vg_field, gf_field in pairs.items():
        if vg_field not in vgflow_fields or gf_field not in geofeas_fields:
            continue
        vg_projected, _manifest = apply_scalar_projection_plan(projection_plan, vgflow_fields[vg_field])
        gf_values = _field_map(geofeas_mesh, geofeas_fields[gf_field])
        diffs: list[float] = []
        for nid in geofeas_mesh.node_ids:
            xy = geofeas_mesh.coords[geofeas_mesh.node_index[nid]]
            vg_value = float(vg_projected[nid])
            gf_value = float(gf_values[nid])
            diff = gf_value - vg_value
            diffs.append(abs(diff))
            rows.append(
                {
                    "sample_id": nid,
                    "field": f"{vg_field}->{gf_field}",
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "vgflow_value": vg_value,
                    "geofeas_value": gf_value,
                    "difference": diff,
                    "abs_difference": abs(diff),
                }
            )
        field_summary.append({"field": f"{vg_field}->{gf_field}", "sample_count": len(geofeas_mesh.node_ids), "max_abs_difference": max(diffs, default=0.0), "mean_abs_difference": float(np.mean(diffs)) if diffs else 0.0})
    return {
        "schema": "geofem.mesh_coupling.coupled_post_comparison.v1",
        "features": ["same_coordinate_sampling_csv", "vgflow_geofeas_field_difference", "max_mean_difference_report", "reusable_projection_weight_cache"],
        "field_summary": field_summary,
        "samples": rows,
        "diagnostics": {
            "field_pair_count": len(field_summary),
            "sample_count": len(rows),
            "max_abs_difference": max((row["abs_difference"] for row in rows), default=0.0),
        },
    }


def build_minimum_mesh_coupling_benchmark() -> dict[str, Any]:
    cfg = {
        "mesh": {
            "generator": "rectangle",
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "nx": 1,
            "ny": 1,
            "element_type": "QUAD4",
            "material": "soil",
        }
    }
    quad4 = mesh_from_config(cfg)
    quad8, coupling_manifest = upgrade_quad4_mesh_to_quad8(quad4)
    values = {nid: float(quad4.coords[quad4.node_index[nid], 0] + quad4.coords[quad4.node_index[nid], 1]) for nid in quad4.node_ids}
    interpolated = interpolate_quad4_nodal_values_to_quad8(quad4, quad8, values)
    projected, projection_manifest = project_scalar_values_between_meshes(quad4, quad8, values, locations="both")
    pressure = {nid: 0.0 for nid in quad8.node_ids}
    for nid in quad8.node_sets.get("top", []):
        pressure[nid] = 10.0
    pressure_report = build_quad8_pressure_load_report(quad8, pressure)
    metrics = [
        {"metric": "area_match", "value": coupling_manifest["diagnostics"]["area_match"], "status": "pass" if coupling_manifest["diagnostics"]["area_match"] else "fail"},
        {"metric": "projection_fallback_count", "value": projection_manifest["diagnostics"]["fallback_count"], "status": "pass" if projection_manifest["diagnostics"]["fallback_count"] == 0 else "warning"},
        {"metric": "interpolated_node_count", "value": len(interpolated), "status": "pass"},
        {"metric": "pressure_conservation_error", "value": pressure_report["diagnostics"]["max_conservation_error"], "status": "pass" if pressure_report["diagnostics"]["max_conservation_error"] <= 1.0e-10 else "warning"},
        {"metric": "projection_location_count", "value": len(projected), "status": "pass"},
    ]
    return {
        "schema": "geofem.mesh_coupling.minimum_benchmark.v1",
        "features": ["quad4_quad8_minimum_benchmark", "node_ip_projection_check", "quadratic_pressure_load_conservation"],
        "metrics": metrics,
        "diagnostics": {"warning_count": sum(1 for row in metrics if row["status"] != "pass")},
    }


def mesh_coupling_api_contract() -> dict[str, Any]:
    artifacts = [
        ("mesh_coupling_manifest", "geofem.mesh_coupling.quad4_quad8.v1", "node_map;element_map;diagnostics", "QUAD4/QUAD8 same-topology mesh correspondence"),
        ("mesh_projection_manifest", "geofem.mesh_coupling.projection.v1", "projection_map;diagnostics", "non-matching mesh scalar transfer"),
        ("quad8_pressure_load_report", "geofem.mesh_coupling.quad8_pressure_load_report.v1", "pressure_loads;diagnostics", "quadratic boundary pressure load transfer"),
        ("stage_handoff", "geofem.mesh_coupling.geofeas_stage_handoff.v1", "geofeas_stages;hydro_rows", "VGFlow transient result to GeoFEAS stage generation"),
        ("material_layer_dictionary", "geofem.mesh_coupling.material_layer_dictionary.v1", "rows;diagnostics", "hydraulic/mechanical material and layer pairing"),
        ("coupled_post_comparison", "geofem.mesh_coupling.coupled_post_comparison.v1", "samples;field_summary", "same-coordinate Post comparison"),
    ]
    rows = [{"artifact": a, "schema": s, "required_fields": f, "purpose": p} for a, s, f, p in artifacts]
    return {
        "schema": "geofem.mesh_coupling.api_contract.v1",
        "features": ["schema_version_catalog", "required_field_catalog", "acceptance_gate_document"],
        "coordinate_system": "project XY in model length units",
        "sign_convention": "pore pressure positive in compression; water head in model length units",
        "artifacts": rows,
        "acceptance_gates": [
            "all JSON artifacts declare schema",
            "CSV files include documented required columns",
            "projection and pressure load reports include conservation/error diagnostics",
            "commercial samples can be compared by replacing open artifact readers without changing schemas",
        ],
    }


def write_mesh_coupling_ui_profile(output_dir: str | Path, profile: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_coupling_ui_profile") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": str(out / f"{prefix}.json"), "csv": str(out / f"{prefix}.csv"), "html": str(out / f"{prefix}.html")}
    write_json_artifact(paths["json"], profile)
    rows = _profile_rows(profile)
    write_dict_rows_csv(paths["csv"], rows, PROFILE_FIELDS)
    write_html_artifact(paths["html"], _simple_html("VGFlow2D / GeoFEAS Coupling UI Profile", profile.get("features", []), profile.get("state", {})))
    return paths


def write_geofeas_stage_handoff(output_dir: str | Path, handoff: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_stage_handoff") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out / f"{prefix}.json"),
        "stages_csv": str(out / f"{prefix}_stages.csv"),
        "hydro_bcs_csv": str(out / f"{prefix}_hydro_bcs.csv"),
        "html": str(out / f"{prefix}.html"),
    }
    write_json_artifact(paths["json"], handoff)
    write_dict_rows_csv(paths["stages_csv"], handoff.get("stage_rows", []), STAGE_FIELDS)
    write_dict_rows_csv(paths["hydro_bcs_csv"], handoff.get("hydro_rows", []), HYDRO_BC_FIELDS)
    write_html_artifact(paths["html"], _simple_html("VGFlow2D / GeoFEAS Stage Handoff", handoff.get("features", []), handoff.get("diagnostics", {})))
    return paths


def write_material_layer_dictionary(output_dir: str | Path, dictionary: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_material_layers") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": str(out / f"{prefix}.json"), "csv": str(out / f"{prefix}.csv"), "html": str(out / f"{prefix}.html")}
    write_json_artifact(paths["json"], dictionary)
    write_dict_rows_csv(paths["csv"], dictionary.get("rows", []), MATERIAL_FIELDS)
    write_html_artifact(paths["html"], _simple_html("VGFlow2D / GeoFEAS Material Layer Dictionary", dictionary.get("features", []), dictionary.get("diagnostics", {})))
    return paths


def write_coupled_post_comparison(output_dir: str | Path, comparison: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_post_comparison") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": str(out / f"{prefix}.json"), "csv": str(out / f"{prefix}.csv"), "html": str(out / f"{prefix}.html")}
    write_json_artifact(paths["json"], comparison)
    write_dict_rows_csv(paths["csv"], comparison.get("samples", []), POST_FIELDS)
    write_html_artifact(paths["html"], _simple_html("VGFlow2D / GeoFEAS Coupled Post Comparison", comparison.get("features", []), comparison.get("diagnostics", {})))
    return paths


def write_mesh_coupling_benchmark(output_dir: str | Path, benchmark: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_coupling_benchmark") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": str(out / f"{prefix}.json"), "csv": str(out / f"{prefix}.csv"), "html": str(out / f"{prefix}.html")}
    write_json_artifact(paths["json"], benchmark)
    write_dict_rows_csv(paths["csv"], benchmark.get("metrics", []), BENCHMARK_FIELDS)
    write_html_artifact(paths["html"], _simple_html("VGFlow2D / GeoFEAS Coupling Benchmark", benchmark.get("features", []), benchmark.get("diagnostics", {})))
    return paths


def write_mesh_coupling_api_contract(output_dir: str | Path, contract: Mapping[str, Any] | None = None, *, prefix: str = "vgflow_geofeas_coupling_api_contract") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(contract or mesh_coupling_api_contract())
    paths = {"json": str(out / f"{prefix}.json"), "csv": str(out / f"{prefix}.csv"), "html": str(out / f"{prefix}.html")}
    write_json_artifact(paths["json"], payload)
    write_dict_rows_csv(paths["csv"], payload.get("artifacts", []), CONTRACT_FIELDS)
    write_html_artifact(paths["html"], _simple_html("VGFlow2D / GeoFEAS Coupling API Contract", payload.get("features", []), {"artifact_count": len(payload.get("artifacts", []))}))
    return paths


def write_vgflow_geofeas_coupling_outputs(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    gamma_w: float,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> dict[str, str]:
    target_mesh, coupling_manifest = _geofeas_target_mesh(mesh)
    paths: dict[str, str] = {}
    paths.update(write_mesh_coupling_manifest(out, coupling_manifest))
    profile = build_coupling_ui_profile(mesh, target_mesh, artifacts=artifacts)
    paths.update({f"coupling_ui_{key}": value for key, value in write_mesh_coupling_ui_profile(out, profile).items()})
    exchange = seepage.get("exchange", seepage.get("geofeas_exchange", {}))
    exchange_cfg = exchange if isinstance(exchange, Mapping) else {}
    handoff = build_geofeas_stage_handoff(
        mesh,
        target_mesh,
        steps,
        problem_type,
        selected_steps=[int(value) for value in _ensure_list(exchange_cfg.get("selected_steps", exchange_cfg.get("steps", []))) if str(value) != ""],
        selected_times=[float(value) for value in _ensure_list(exchange_cfg.get("selected_times", exchange_cfg.get("times", []))) if str(value) != ""],
        gamma_w=gamma_w,
    )
    paths.update({f"stage_handoff_{key}": value for key, value in write_geofeas_stage_handoff(out, handoff).items()})
    material_dictionary = build_material_layer_dictionary(mesh, target_mesh, materials, materials, layer_map=seepage.get("material_layer_map") if isinstance(seepage.get("material_layer_map"), Mapping) else None)
    paths.update({f"material_layer_{key}": value for key, value in write_material_layer_dictionary(out, material_dictionary).items()})
    benchmark = build_minimum_mesh_coupling_benchmark()
    paths.update({f"coupling_benchmark_{key}": value for key, value in write_mesh_coupling_benchmark(out, benchmark).items()})
    paths.update({f"coupling_api_{key}": value for key, value in write_mesh_coupling_api_contract(out).items()})
    return paths


def _geofeas_target_mesh(mesh: Mesh2D) -> tuple[Mesh2D, dict[str, Any]]:
    element_types = {element.type for element in mesh.elements}
    if element_types == {"QUAD4"}:
        return upgrade_quad4_mesh_to_quad8(mesh)
    if element_types == {"QUAD8"}:
        hydraulic, manifest = downgrade_quad8_mesh_to_quad4(mesh)
        return mesh, manifest | {"hydraulic_mesh_summary": _mesh_small_summary(hydraulic)}
    return mesh, {
        "schema": "geofem.mesh_coupling.mixed_mesh_reference.v1",
        "mode": "mixed_reference",
        "features": ["mixed_mesh_passthrough"],
        "diagnostics": {"element_types": sorted(element_types), "area_match": True},
        "node_map": [{"source_node": nid, "target_node": nid, "kind": "identity", "weight": "1.0"} for nid in mesh.node_ids],
        "element_map": [{"source_element": element.id, "target_element": element.id, "source_type": element.type, "target_type": element.type, "material": element.material, "integration": element.integration} for element in mesh.elements],
    }


def _project_total_head(vgflow_mesh: Mesh2D, geofeas_mesh: Mesh2D, total_head: Sequence[float] | np.ndarray) -> dict[str, float]:
    if vgflow_mesh.node_ids == geofeas_mesh.node_ids and np.allclose(vgflow_mesh.coords, geofeas_mesh.coords):
        arr = np.asarray(total_head, dtype=float)
        return {nid: float(arr[vgflow_mesh.node_index[nid]]) for nid in vgflow_mesh.node_ids}
    if {element.type for element in vgflow_mesh.elements} == {"QUAD4"} and {element.type for element in geofeas_mesh.elements} == {"QUAD8"}:
        try:
            return interpolate_quad4_nodal_values_to_quad8(vgflow_mesh, geofeas_mesh, total_head)
        except FEM2DError:
            pass
    projected, _manifest = project_scalar_values_between_meshes(vgflow_mesh, geofeas_mesh, total_head)
    return projected


def _select_steps(steps: Sequence[Any], *, selected_steps: Sequence[int] | None, selected_times: Sequence[float] | None) -> list[Any]:
    if not steps:
        return []
    by_index = {int(step.index): step for step in steps}
    selected: list[Any] = []
    for index in selected_steps or []:
        if int(index) in by_index:
            selected.append(by_index[int(index)])
    for time_value in selected_times or []:
        nearest = min(steps, key=lambda step: abs(float(step.time) - float(time_value)))
        if nearest not in selected:
            selected.append(nearest)
    if not selected:
        selected = [steps[-1]]
    return sorted(selected, key=lambda step: int(step.index))


def _field_map(mesh: Mesh2D, values: Mapping[str, float] | Sequence[float] | np.ndarray) -> dict[str, float]:
    if isinstance(values, Mapping):
        return {str(key): float(value) for key, value in values.items()}
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != len(mesh.node_ids):
        raise FEM2DError("field array length must match mesh node count")
    return {nid: float(arr[mesh.node_index[nid]]) for nid in mesh.node_ids}


def _bbox(mesh: Mesh2D) -> dict[str, float]:
    mins = np.min(mesh.coords, axis=0)
    maxs = np.max(mesh.coords, axis=0)
    return {"xmin": float(mins[0]), "ymin": float(mins[1]), "xmax": float(maxs[0]), "ymax": float(maxs[1])}


def _bbox_match(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    return all(math.isclose(float(a[key]), float(b[key]), rel_tol=1.0e-12, abs_tol=1.0e-12) for key in ("xmin", "ymin", "xmax", "ymax"))


def _element_type_counts(mesh: Mesh2D) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in mesh.elements:
        counts[element.type] = counts.get(element.type, 0) + 1
    return counts


def _mesh_small_summary(mesh: Mesh2D) -> dict[str, Any]:
    return {"node_count": len(mesh.node_ids), "element_count": len(mesh.elements), "element_types": _element_type_counts(mesh)}


def _profile_rows(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {"section": "template", "id": profile.get("template", {}).get("id", ""), "label": profile.get("template", {}).get("label", ""), "status": "defined", "details": ";".join(profile.get("template", {}).get("commands", []))}
    ]
    for view in profile.get("views", []):
        rows.append({"section": "view", "id": view.get("id", ""), "label": view.get("id", ""), "status": "defined", "details": ";".join(view.get("columns", []))})
    for key, value in profile.get("state", {}).items():
        rows.append({"section": "state", "id": key, "label": key, "status": value, "details": value})
    return rows


def _simple_html(title: str, features: Sequence[Any], diagnostics: Mapping[str, Any]) -> str:
    rows = [["feature", feature] for feature in features] + [[key, value] for key, value in diagnostics.items()]
    return html_table_document(title=title, lead="Open public-substitute coupling artifact generated inside GeoFEM.", headers=["item", "value"], rows=rows)


__all__ = [
    "build_coupled_post_comparison",
    "build_coupling_ui_profile",
    "build_geofeas_stage_handoff",
    "build_material_layer_dictionary",
    "build_minimum_mesh_coupling_benchmark",
    "mesh_coupling_api_contract",
    "write_coupled_post_comparison",
    "write_geofeas_stage_handoff",
    "write_material_layer_dictionary",
    "write_mesh_coupling_api_contract",
    "write_mesh_coupling_benchmark",
    "write_mesh_coupling_ui_profile",
    "write_vgflow_geofeas_coupling_outputs",
]
