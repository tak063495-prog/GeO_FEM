"""Pre-operation templates for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fem2d_types import Mesh2D
from .fem2d_utils import _ensure_list


VGFLOW_MESH_MODES: dict[str, dict[str, Any]] = {
    "auto_mixed": {
        "label": "Auto mixed mesh",
        "allowed_elements": ["TRI3", "QUAD4"],
        "notes": "Public substitute for VGFlow 2D auto mixed mesh operation.",
    },
    "quadrilateral_only": {
        "label": "Quadrilateral only",
        "allowed_elements": ["QUAD4", "QUAD8"],
        "notes": "Public substitute for VGFlow 2D quadrilateral mesh operation.",
    },
    "triangular_only": {
        "label": "Triangular only",
        "allowed_elements": ["TRI3", "TRI6"],
        "notes": "Public substitute for VGFlow 2D triangular mesh operation.",
    },
    "semi_auto": {
        "label": "Semi-auto",
        "allowed_elements": ["TRI3", "QUAD4", "QUAD8"],
        "notes": "Public substitute for VGFlow 2D semi-auto mesh operation.",
    },
}

_MESH_MODE_ALIASES = {
    "auto": "auto_mixed",
    "auto_mixed": "auto_mixed",
    "mixed": "auto_mixed",
    "オート混合": "auto_mixed",
    "quad": "quadrilateral_only",
    "quadrilateral": "quadrilateral_only",
    "quadrilateral_only": "quadrilateral_only",
    "四角形のみ": "quadrilateral_only",
    "tri": "triangular_only",
    "triangular": "triangular_only",
    "triangular_only": "triangular_only",
    "三角形のみ": "triangular_only",
    "semi": "semi_auto",
    "semi_auto": "semi_auto",
    "semiauto": "semi_auto",
    "セミオート": "semi_auto",
}


def vgflow_pre_template_catalog() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in sorted(VGFLOW_MESH_MODES.items())]


def write_vgflow_pre_outputs(out: Path, mesh: Mesh2D, seepage: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "pre_operation_log_json": str(out / "vgflow_pre_operation_log.json"),
        "pre_operation_log_csv": str(out / "vgflow_pre_operation_log.csv"),
        "pre_state": str(out / "vgflow_pre_state.json"),
        "pre_templates": str(out / "vgflow_pre_templates.json"),
    }
    pre = _pre_cfg(seepage)
    state = _pre_state(mesh, seepage, pre)
    operation_log = _operation_log(state, pre, seepage)
    payload = {
        "schema": "geofem.vgflow2d.pre_operation_log.public_substitute.v1",
        "workflow": str(pre.get("workflow", "vgflow2d_public_pre")),
        "mesh_mode": state["mesh_mode"],
        "operation_log": operation_log,
    }
    Path(paths["pre_operation_log_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_operation_csv(Path(paths["pre_operation_log_csv"]), operation_log)
    Path(paths["pre_state"]).write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    Path(paths["pre_templates"]).write_text(json.dumps({"mesh_modes": vgflow_pre_template_catalog()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def _pre_cfg(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("pre", "vgflow_pre", "preprocess", "pre_operation"):
        value = seepage.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _pre_state(mesh: Mesh2D, seepage: Mapping[str, Any], pre: Mapping[str, Any]) -> dict[str, Any]:
    raw_mode = pre.get("mesh_mode", seepage.get("mesh_mode", seepage.get("vgflow_mesh_mode", "auto_mixed")))
    mesh_mode = _normalize_mesh_mode(raw_mode)
    lines = _limited_lines(pre.get("straight_lines", pre.get("lines", [])))
    block_ops = _block_operations(pre)
    grid = _grid_state(pre)
    snap = _snap_state(pre, grid)
    return {
        "mesh_mode": mesh_mode,
        "mesh_mode_label": VGFLOW_MESH_MODES[mesh_mode]["label"],
        "analysis_type": str(seepage.get("analysis_type", "saturated_unsaturated_seepage")),
        "analysis_mode": str(seepage.get("mode", seepage.get("analysis_mode", "steady"))),
        "problem_type": str(seepage.get("problem_type", "vertical")),
        "grid": grid,
        "snap": snap,
        "straight_lines": lines,
        "straight_line_count": len(lines),
        "straight_line_limit": 50,
        "node_corrections": _ensure_list(pre.get("node_corrections", [])),
        "origin": pre.get("origin", grid.get("origin", [0.0, 0.0])),
        "block_operations": block_ops,
        "mesh_summary": {
            "node_count": len(mesh.node_ids),
            "element_count": len(mesh.elements),
            "node_set_count": len(mesh.node_sets),
            "element_types": _element_type_counts(mesh),
        },
        "diagnostics": _pre_diagnostics(mesh_mode, lines, block_ops),
    }


def _operation_log(state: Mapping[str, Any], pre: Mapping[str, Any], seepage: Mapping[str, Any]) -> list[dict[str, Any]]:
    log = [
        _op(1, "file", "new_project", "create new VGFlow 2D public substitute model", "empty model initialized"),
        _op(2, "toolbar", "analysis_type", f"select analysis type: {state['analysis_type']}", "analysis type stored"),
        _op(3, "toolbar", "analysis_mode", f"select analysis mode: {state['analysis_mode']}", "steady/transient mode stored"),
        _op(4, "toolbar", "mesh_mode", f"select mesh mode: {state['mesh_mode_label']}", "mesh mode stored"),
        _op(5, "pre", "grid_snap", f"set grid={state['grid']} snap={state['snap']}", "grid and snap state stored"),
    ]
    next_step = len(log) + 1
    if state["straight_lines"]:
        log.append(_op(next_step, "pre", "straight_line_registration", f"register {state['straight_line_count']} straight/horizontal lines", "line registry stored"))
        next_step += 1
    if state["node_corrections"]:
        log.append(_op(next_step, "pre", "node_coordinate_correction", "apply right-click style node coordinate corrections", "node correction list stored"))
        next_step += 1
    for block in state["block_operations"]:
        log.append(_op(next_step, "pre", "block_selection", f"{block['mode']} selection for {block['name']} then {block['action']}", "block operation stored"))
        next_step += 1
    log.extend(
        [
            _op(next_step, "mesh", "mesh_split", f"execute mesh split using {state['mesh_mode_label']}", "mesh summary stored"),
            _op(next_step + 1, "analysis", "material_and_boundary_definition", "define seepage materials and boundary conditions", "solver input ready"),
            _op(next_step + 2, "analysis", "run", "execute saturated/unsaturated seepage analysis", "result artifacts written"),
        ]
    )
    if bool(pre.get("mesh_regeneration_clears_conditions", True)):
        log.insert(-2, _op(next_step + 1, "mesh", "condition_reset_prompt", "mesh split invalidates previous analysis conditions and prompts redefinition", "reset guidance stored"))
        for index, row in enumerate(log, start=1):
            row["step"] = index
    return log


def _op(step: int, tab: str, command: str, action: str, expected: str) -> dict[str, Any]:
    return {"step": step, "tab": tab, "command": command, "action": action, "expected": expected}


def _write_operation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["step", "tab", "command", "action", "expected"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _normalize_mesh_mode(value: Any) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _MESH_MODE_ALIASES.get(key, key if key in VGFLOW_MESH_MODES else "auto_mixed")


def _grid_state(pre: Mapping[str, Any]) -> dict[str, Any]:
    raw = pre.get("grid", {})
    grid = dict(raw) if isinstance(raw, Mapping) else {}
    spacing = grid.get("spacing", pre.get("grid_spacing", pre.get("grid_size", 1.0)))
    return {
        "enabled": bool(grid.get("enabled", pre.get("grid_enabled", True))),
        "origin": _xy_pair(grid.get("origin", pre.get("origin", [0.0, 0.0]))),
        "spacing": float(spacing or 1.0),
    }


def _snap_state(pre: Mapping[str, Any], grid: Mapping[str, Any]) -> dict[str, Any]:
    raw = pre.get("snap", {})
    snap = dict(raw) if isinstance(raw, Mapping) else {}
    return {
        "enabled": bool(snap.get("enabled", pre.get("snap_enabled", True))),
        "mode": str(snap.get("mode", pre.get("snap_mode", "grid"))),
        "grid_size": float(snap.get("grid_size", grid.get("spacing", 1.0)) or 1.0),
    }


def _limited_lines(raw: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for index, item in enumerate(_ensure_list(raw)[:50], start=1):
        if isinstance(item, Mapping):
            points = item.get("points", [item.get("start", [0.0, 0.0]), item.get("end", [1.0, 0.0])])
            lines.append({"id": str(item.get("id", item.get("name", f"L{index}"))), "points": [_xy_pair(point) for point in _ensure_list(points)[:2]], "horizontal": bool(item.get("horizontal", False))})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append({"id": f"L{index}", "points": [_xy_pair(item[0]), _xy_pair(item[1])], "horizontal": False})
    return lines


def _block_operations(pre: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = pre.get("block_selection", pre.get("block_operations", pre.get("blocks", [])))
    operations: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        iterable = [{"name": key, **(value if isinstance(value, Mapping) else {})} for key, value in raw.items()]
    else:
        iterable = _ensure_list(raw)
    for index, item in enumerate(iterable, start=1):
        if not isinstance(item, Mapping):
            continue
        operations.append(
            {
                "name": str(item.get("name", item.get("id", f"B{index}"))),
                "mode": str(item.get("mode", item.get("selection_mode", "box"))),
                "action": str(item.get("action", "auto_block")),
                "box": item.get("box", {"x_range": item.get("x_range", []), "y_range": item.get("y_range", [])}),
                "hatching_check": bool(item.get("hatching_check", item.get("hatch", True))),
                "release_selection": bool(item.get("release_selection", True)),
            }
        )
    return operations


def _xy_pair(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        return [float(value.get("x", 0.0)), float(value.get("y", 0.0))]
    seq = _ensure_list(value)
    if len(seq) >= 2:
        return [float(seq[0]), float(seq[1])]
    return [0.0, 0.0]


def _element_type_counts(mesh: Mesh2D) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in mesh.elements:
        counts[element.type] = counts.get(element.type, 0) + 1
    return counts


def _pre_diagnostics(mesh_mode: str, lines: Sequence[Mapping[str, Any]], blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    diagnostics = []
    if len(lines) >= 50:
        diagnostics.append({"level": "warning", "code": "straight_line_limit", "message": "straight line registry reached the public substitute limit of 50"})
    if mesh_mode == "semi_auto" and not blocks:
        diagnostics.append({"level": "info", "code": "semi_auto_blocks", "message": "semi-auto mode benefits from explicit block selection operations"})
    return diagnostics


__all__ = ["vgflow_pre_template_catalog", "write_vgflow_pre_outputs"]
