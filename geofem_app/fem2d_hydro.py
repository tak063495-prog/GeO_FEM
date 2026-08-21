"""Hydraulic input normalization helpers for 2D FEM stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fem2d_mesh import _target_nodes
from .fem2d_types import FEM2DError, Mesh2D, StageResult2D
from .fem2d_utils import _ensure_list

HYDRO_HELPER_FUNCTIONS = (
    "_stage_has_hydro_coupling",
    "_hydro_mapping",
    "_prepare_stage_hydro",
    "_merge_hydro_maps",
    "_stage_with_hydro",
    "_pore_pressure_from_hydro",
    "_seepage_pressure_specs",
    "_seepage_rows",
    "_select_time_rows",
    "_row_time_value",
    "_water_level_pressure_specs",
    "_normalize_pressure_spec",
    "_pressure_value_for_node",
    "_pressure_source_name",
    "_convert_pressure_unit",
    "_attach_stage_load_hydro_info",
    "_initial_pore_pressure",
    "_collect_pressure_constraints",
)


def hydro_helper_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.hydro_helpers.v1",
        "module": "geofem_app.fem2d_hydro",
        "function_count": len(HYDRO_HELPER_FUNCTIONS),
        "functions": list(HYDRO_HELPER_FUNCTIONS),
        "covered_surfaces": [
            "stage_hydro_merge",
            "seepage_pressure_sync",
            "water_level_pressure_conversion",
            "initial_pressure",
            "pressure_constraints",
            "hydro_solver_info",
        ],
    }


def _stage_has_hydro_coupling(stage_cfg: Mapping[str, Any]) -> bool:
    if _hydro_mapping(stage_cfg) is not None:
        return True
    fields = stage_cfg.get("fields", stage_cfg.get("unknowns", []))
    if isinstance(fields, str):
        fields = [fields]
    return any(str(field).lower().replace("_", "-") in {"p", "u-p", "pore-pressure", "pressure"} for field in _ensure_list(fields))

def _hydro_mapping(stage_cfg: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(stage_cfg, Mapping):
        return None
    for key in ("hydro", "consolidation", "up", "u_p"):
        value = stage_cfg.get(key)
        if isinstance(value, Mapping):
            return value
    if any(key in stage_cfg for key in ("pressure_bcs", "pore_pressure_bcs", "drainage", "pore_flux_bcs", "flux_bcs", "pore_robin_bcs", "robin_bcs", "storage", "specific_storage", "permeability", "k", "initial_pressure", "initial_pore_pressure", "seepage_csv", "seepage_results", "water_level", "water_levels", "water_level_updates")):
        return stage_cfg
    return None

def _prepare_stage_hydro(
    mesh: Mesh2D,
    cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    previous_pressure: np.ndarray | None,
    time: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hydro: dict[str, Any] = {}
    global_hydro = cfg.get("hydro", cfg.get("seepage", {}))
    if isinstance(global_hydro, Mapping):
        hydro.update(dict(global_hydro))
    stage_hydro = _hydro_mapping(stage_cfg)
    if isinstance(stage_hydro, Mapping):
        hydro = _merge_hydro_maps(hydro, stage_hydro)
    if not hydro:
        return {}, {}
    info: dict[str, Any] = {"time": time, "unit_conversion": [], "seepage_sync_count": 0, "water_level_update_count": 0}
    pressure_specs = _ensure_list(hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", [])))
    pressure_specs.extend(_seepage_pressure_specs(mesh, hydro, time, info))
    pressure_specs.extend(_water_level_pressure_specs(mesh, hydro, info))
    converted: list[dict[str, Any]] = []
    for spec in pressure_specs:
        if not isinstance(spec, Mapping):
            continue
        converted.extend(_normalize_pressure_spec(mesh, dict(spec), hydro, info))
    if converted:
        hydro["pressure_bcs"] = converted
    if previous_pressure is not None:
        info["previous_pressure_inherited"] = bool(hydro.get("inherit_previous_pressure", True))
    return hydro, info

def _merge_hydro_maps(base: Mapping[str, Any], local: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in local.items():
        if key in {"pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "flux_bcs", "pore_robin_bcs", "robin_bcs"}:
            merged[key] = _ensure_list(merged.get(key, [])) + _ensure_list(value)
        else:
            merged[key] = value
    return merged

def _stage_with_hydro(stage_cfg: Mapping[str, Any], hydro: Mapping[str, Any]) -> dict[str, Any]:
    stage = dict(stage_cfg)
    if hydro:
        stage["hydro"] = dict(hydro)
    return stage

def _pore_pressure_from_hydro(mesh: Mesh2D, hydro: Mapping[str, Any], previous_pressure: np.ndarray | None) -> np.ndarray | None:
    if not hydro:
        return None
    if previous_pressure is None and not any(key in hydro for key in ("initial_pressure", "initial_pore_pressure", "p0", "pressure_bcs", "pore_pressure_bcs", "water_level", "water_levels", "seepage_csv", "seepage_results")):
        return None
    p = _initial_pore_pressure(mesh, hydro, previous_pressure)
    for node, value in _collect_pressure_constraints(mesh, hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", []))).items():
        p[node] = value
    return p

def _seepage_pressure_specs(mesh: Mesh2D, hydro: Mapping[str, Any], time: float, info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _seepage_rows(hydro.get("seepage_csv", hydro.get("seepage_results", hydro.get("seepage_time_series"))))
    if not rows:
        return []
    selected = _select_time_rows(rows, time)
    specs: list[dict[str, Any]] = []
    for row in selected:
        nid = str(row.get("node_id", row.get("node", ""))).strip()
        if not nid or nid not in mesh.node_index:
            continue
        spec: dict[str, Any] = {"node": nid, "source": "seepage_sync", "time": float(row.get("time", row.get("t", time)) or time)}
        if row.get("pore_pressure", row.get("pressure", row.get("p"))) not in (None, ""):
            spec["pressure"] = float(row.get("pore_pressure", row.get("pressure", row.get("p"))))
        elif row.get("head", row.get("water_head")) not in (None, ""):
            spec["head"] = float(row.get("head", row.get("water_head")))
        elif row.get("water_level") not in (None, ""):
            spec["water_level"] = float(row.get("water_level"))
        specs.append(spec)
    info["seepage_sync_count"] = len(specs)
    return specs

def _seepage_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        from .geofeas_verification import import_external_seepage_results

        return import_external_seepage_results(raw)
    if isinstance(raw, Mapping):
        path = raw.get("file", raw.get("csv", raw.get("path")))
        if path:
            return _seepage_rows(path)
        raw = raw.get("rows", raw.get("data", []))
    rows: list[dict[str, Any]] = []
    for row in _ensure_list(raw):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows

def _select_time_rows(rows: list[dict[str, Any]], time: float) -> list[dict[str, Any]]:
    timed = [row for row in rows if row.get("time", row.get("t")) not in (None, "")]
    if not timed:
        return rows
    nearest_time = min((_row_time_value(row, 0.0) for row in timed), key=lambda t: abs(t - time))
    return [row for row in rows if abs(_row_time_value(row, nearest_time) - nearest_time) <= 1.0e-12]

def _row_time_value(row: Mapping[str, Any], default: float) -> float:
    value = row.get("time", row.get("t", default))
    if value in (None, ""):
        return default
    return float(value)

def _water_level_pressure_specs(mesh: Mesh2D, hydro: Mapping[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    raw = hydro.get("water_levels", hydro.get("water_level_updates", hydro.get("water_level")))
    if raw is None:
        return []
    if isinstance(raw, Mapping) and not any(key in raw for key in ("set", "node", "nodes", "edge", "edges", "water_level", "level")):
        specs = [{"set": key, "water_level": value} for key, value in raw.items()]
    else:
        specs = _ensure_list(raw)
    out: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        target = dict(spec)
        level = float(target.get("water_level", target.get("level", target.get("head", 0.0))) or 0.0)
        try:
            nodes = _target_nodes(mesh, target)
        except FEM2DError:
            nodes = list(mesh.node_ids)
        for nid in nodes:
            out.append({"node": nid, "water_level": level, "source": "water_level_update", **{k: v for k, v in target.items() if k not in {"node", "nodes", "set"}}})
    info["water_level_update_count"] = len(out)
    return out

def _normalize_pressure_spec(mesh: Mesh2D, spec: dict[str, Any], hydro: Mapping[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        nodes = _target_nodes(mesh, spec)
    except FEM2DError:
        nodes = []
    if nodes and ("water_level" in spec or "head" in spec or "water_head" in spec or "pressure_head" in spec):
        out: list[dict[str, Any]] = []
        for nid in nodes:
            item = dict(spec)
            item.pop("nodes", None)
            item.pop("set", None)
            item["node"] = nid
            item["pressure"] = _pressure_value_for_node(mesh, nid, spec, hydro)
            item["converted_from"] = _pressure_source_name(spec)
            out.append(item)
            info["unit_conversion"].append({"node": nid, "from": item["converted_from"], "pressure": item["pressure"]})
        return out
    if "pressure" not in spec and "p" not in spec and "value" not in spec:
        spec["pressure"] = _pressure_value_for_node(mesh, nodes[0] if nodes else mesh.node_ids[0], spec, hydro)
        spec["converted_from"] = _pressure_source_name(spec)
        info["unit_conversion"].append({"target": spec.get("node", spec.get("set", "")), "from": spec["converted_from"], "pressure": spec["pressure"]})
    else:
        spec["pressure"] = _convert_pressure_unit(float(spec.get("pressure", spec.get("p", spec.get("value", 0.0))) or 0.0), str(spec.get("unit", hydro.get("pressure_unit", ""))))
    return [spec]

def _pressure_value_for_node(mesh: Mesh2D, nid: str, spec: Mapping[str, Any], hydro: Mapping[str, Any]) -> float:
    gamma_w = float(spec.get("gamma_w", hydro.get("gamma_w", hydro.get("water_unit_weight", 9.80665))) or 9.80665)
    y = float(mesh.coords[mesh.node_index[nid], 1])
    if "pressure_head" in spec:
        return gamma_w * float(spec["pressure_head"])
    if "water_level" in spec:
        head = float(spec["water_level"]) - y
        if not bool(spec.get("allow_negative_pressure", False)):
            head = max(head, 0.0)
        return gamma_w * head
    if "head" in spec or "water_head" in spec:
        datum = float(spec.get("datum", hydro.get("head_datum", 0.0)) or 0.0)
        head = float(spec.get("head", spec.get("water_head", 0.0))) - (y - datum)
        if not bool(spec.get("allow_negative_pressure", False)):
            head = max(head, 0.0)
        return gamma_w * head
    return _convert_pressure_unit(float(spec.get("pressure", spec.get("p", spec.get("value", 0.0))) or 0.0), str(spec.get("unit", hydro.get("pressure_unit", ""))))

def _pressure_source_name(spec: Mapping[str, Any]) -> str:
    for key in ("pressure_head", "water_level", "head", "water_head"):
        if key in spec:
            return key
    return "pressure"

def _convert_pressure_unit(value: float, unit: str) -> float:
    text = unit.lower().replace(" ", "")
    if text in {"pa"}:
        return value / 1000.0
    if text in {"mpa"}:
        return value * 1000.0
    if text in {"m", "mh2o", "m-water", "mwater"}:
        return value * 9.80665
    return value

def _attach_stage_load_hydro_info(result: StageResult2D, load_info: Mapping[str, Any], hydro_info: Mapping[str, Any]) -> None:
    if load_info:
        result.solver_info["load_processing"] = dict(load_info)
        if isinstance(load_info.get("seismic"), Mapping):
            result.solver_info["seismic"] = dict(load_info["seismic"])
    if hydro_info:
        result.solver_info["hydro_sync"] = dict(hydro_info)

def _initial_pore_pressure(mesh: Mesh2D, hydro: Mapping[str, Any], previous_pressure: np.ndarray | None) -> np.ndarray:
    if previous_pressure is not None:
        return np.asarray(previous_pressure, dtype=float).copy()
    raw = hydro.get("initial_pressure", hydro.get("initial_pore_pressure", hydro.get("p0", 0.0)))
    if isinstance(raw, Mapping):
        p = np.zeros(len(mesh.node_ids), dtype=float)
        for key, value in raw.items():
            if str(key) in mesh.node_sets:
                for nid in mesh.node_sets[str(key)]:
                    p[mesh.node_index[nid]] = float(value)
            elif str(key) in mesh.node_ids:
                p[mesh.node_index[str(key)]] = float(value)
            else:
                raise FEM2DError(f"unknown pressure initial target '{key}'")
        return p
    return np.full(len(mesh.node_ids), float(raw), dtype=float)

def _collect_pressure_constraints(mesh: Mesh2D, pressure_bcs: Any) -> dict[int, float]:
    fixed: dict[int, float] = {}
    for bc in _ensure_list(pressure_bcs):
        if not isinstance(bc, Mapping):
            raise FEM2DError("each pressure boundary condition must be a mapping")
        value = float(bc.get("pressure", bc.get("p", bc.get("value", 0.0))))
        for nid in _target_nodes(mesh, bc):
            fixed[mesh.node_index[nid]] = value
    return fixed
