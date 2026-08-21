"""Geometry/CAD GUI operation controller functions split from MainWindow.

The MainWindow keeps widgets, project state, scene ownership, and notifications.
This module owns geometry table synchronization, CAD dimension constraints,
curve-control/Boolean table operations, and overlap repair actions.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

import yaml

from geofem_app.gui.cad_worker import solve_dimension_constraints_snapshot


GEOMETRY_CONTROLLER_METHODS = (
    'populate_geometry_tables',
    'add_geometry_line_row',
    'add_geometry_tunnel_row',
    'add_geometry_layer_row',
    'add_geometry_annotation_row',
    'add_geometry_dimension_row',
    'remove_selected_geometry_rows',
    'remove_selected_cad_text_rows',
    'load_selected_geometry_point_to_editor',
    'apply_geometry_point_absolute_edit',
    'nudge_selected_geometry_point',
    'join_selected_geometry_endpoint_to_nearest',
    'add_closure_segment_from_selected_endpoint',
    'set_selected_geometry_lines_purpose',
    'refresh_cad_repair_diagnostics',
    'populate_cad_repair_table',
    'select_cad_repair_candidate',
    'cad_repair_diagnostics_snapshot',
    '_selected_geometry_endpoint',
    '_selected_geometry_line_target',
    '_move_geometry_point_target',
    'set_all_cad_layers_visible',
    'set_selected_cad_layers_locked',
    'apply_dimension_constraints',
    'apply_dimension_constraints_async',
    '_start_dimension_constraints_job',
    '_cad_dimension_constraints_finished',
    '_dimension_constraints_from_table',
    '_dimension_constraints_from_geometry_dimensions',
    '_parse_dimension_constraint',
    '_dimension_target_value',
    '_solve_dimension_constraints',
    '_cad_constraint_point_refs',
    '_cluster_cad_constraint_refs',
    '_nearest_cad_constraint_group',
    '_cad_kernel_constraints_from_geometry',
    '_cad_constraint_group_lookup',
    '_cad_group_id_from_label',
    '_cad_constraint_required_slots',
    '_refs_near_point',
    '_refs_centroid',
    '_move_refs',
    '_project_dimension_constraint',
    '_dimension_constraint_tolerance',
    '_update_dimension_geometry_after_constraint',
    '_resample_curve_region_points',
    '_mutable_curve_groups',
    'add_geometry_curve_row',
    'populate_geometry_curve_tables',
    '_add_curve_control_row',
    'populate_curve_control_point_table',
    'apply_curve_control_point_table',
    '_apply_curve_control_point_table_to_curve_rows',
    'populate_geometry_boolean_table',
    'apply_curve_boolean_panel',
    'rebuild_geometry_boolean_graph',
    '_regions_with_curve_table_data',
    '_curve_spec_from_table_row',
    '_curve_boundary_specs',
    '_curve_hole_specs',
    '_curve_parameters_yaml',
    '_default_curve_parameters',
    '_sample_curve_chain_points',
    '_sample_curve_spec',
    '_boolean_graph_target',
    '_active_cad_boolean_graph',
    '_refresh_boolean_operation_combo',
    'apply_geometry_panel',
    'refresh_cad_overlap_edges',
    'set_selected_cad_overlap_state',
    'repair_selected_cad_overlap_edges',
    '_selected_boolean_operation_name',
    'select_cad_boolean_operation_from_table',
    'store_selected_boolean_operation_as_manual_repair',
    'clear_manual_boolean_repair',
    '_iter_cad_overlap_edges',
    '_cad_boolean_graphs',
)


def geometry_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.geometry_controller.v1",
        "method_count": len(GEOMETRY_CONTROLLER_METHODS),
        "methods": list(GEOMETRY_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner geometry/CAD widgets, dispatches CAD constraint jobs, and syncs curve/Boolean repair state; MainWindow delegates geometry input-domain actions",
        "covered_surfaces": ["geometry_tables", "cad_dimension_constraints", "curve_controls", "cad_boolean", "cad_overlap_repair", "cad_repair_diagnostics"],
    }


_QT: dict[str, Any] = {}


def _bind_qt(qt: Mapping[str, Any]) -> None:
    global QTableWidgetItem, QMessageBox, Qt, QtCallableRunner
    if not qt:
        return
    _QT.update(dict(qt))
    QTableWidgetItem = _QT.get("QTableWidgetItem")
    QMessageBox = _QT.get("QMessageBox")
    Qt = _QT.get("Qt")
    QtCallableRunner = _QT.get("QtCallableRunner")


def populate_geometry_tables(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    line_table = getattr(self, "geometry_line_table", None)
    tunnel_table = getattr(self, "geometry_tunnel_table", None)
    if line_table is None or tunnel_table is None:
        return
    geometry = self._mapping(self.cfg.get("geometry", {}))
    layer_table = getattr(self, "geometry_layer_table", None)
    if layer_table is not None:
        layer_table.setRowCount(0)
        layers = geometry.get("layers", [])
        if isinstance(layers, list) and layers:
            for raw in layers:
                if not isinstance(raw, Mapping):
                    continue
                self.add_geometry_layer_row(
                    visible=raw.get("visible", raw.get("on", raw.get("active", True))),
                    locked=raw.get("locked", raw.get("lock", False)),
                    name=str(raw.get("name", raw.get("id", ""))),
                    color=str(raw.get("color", "")),
                    linetype=str(raw.get("linetype", raw.get("line_type", ""))),
                    lineweight=str(raw.get("lineweight", raw.get("weight", ""))),
                    opacity=str(raw.get("opacity", "")),
                    source=str(raw.get("source", raw.get("source_id", ""))),
                )
        else:
            layer_names = set()
            for collection in ("lines", "tunnels", "regions", "annotations", "dimensions", "hatches"):
                for raw in geometry.get(collection, []) if isinstance(geometry.get(collection, []), list) else []:
                    if isinstance(raw, Mapping) and str(raw.get("layer", "")).strip():
                        layer_names.add(str(raw.get("layer")).strip())
            for name in sorted(layer_names):
                self.add_geometry_layer_row(name=name, visible=True)
    line_table.setRowCount(0)
    for raw in geometry.get("lines", []) if isinstance(geometry.get("lines", []), list) else []:
        if not isinstance(raw, Mapping):
            continue
        try:
            x1, y1 = self._xy_pair(raw.get("start", raw.get("p1", [0.0, 0.0])))
            x2, y2 = self._xy_pair(raw.get("end", raw.get("p2", [0.0, 0.0])))
        except ValueError:
            continue
        self.add_geometry_line_row(
            line_id=str(raw.get("id", "")),
            purpose=str(raw.get("purpose", "model")),
            layer=str(raw.get("layer", "")),
            x1=str(x1),
            y1=str(y1),
            x2=str(x2),
            y2=str(y2),
        )
    tunnel_table.setRowCount(0)
    for raw in geometry.get("tunnels", []) if isinstance(geometry.get("tunnels", []), list) else []:
        if not isinstance(raw, Mapping):
            continue
        try:
            cx, cy = self._xy_pair(raw.get("center", [0.0, 0.0]))
            radius = float(raw.get("radius", 0.0))
            segments = int(raw.get("segments", 24))
        except (TypeError, ValueError):
            continue
        self.add_geometry_tunnel_row(
            tunnel_id=str(raw.get("id", "")),
            layer=str(raw.get("layer", "")),
            cx=str(cx),
            cy=str(cy),
            radius=str(radius),
            segments=str(segments),
        )
    editor = getattr(self, "geometry_regions_editor", None)
    if editor is not None:
        regions = geometry.get("regions", [])
        editor.setPlainText(yaml.safe_dump(regions if isinstance(regions, list) else [], allow_unicode=True, sort_keys=False))
    annotation_table = getattr(self, "geometry_annotation_table", None)
    if annotation_table is not None:
        annotation_table.setRowCount(0)
        for raw in geometry.get("annotations", []) if isinstance(geometry.get("annotations", []), list) else []:
            if not isinstance(raw, Mapping):
                continue
            try:
                x, y = self._xy_pair(raw.get("point", raw.get("position", [0.0, 0.0])))
            except ValueError:
                continue
            self.add_geometry_annotation_row(
                annotation_id=str(raw.get("id", "")),
                layer=str(raw.get("layer", "")),
                x=f"{x:g}",
                y=f"{y:g}",
                text=str(raw.get("text", "")),
            )
    dimension_table = getattr(self, "geometry_dimension_table", None)
    if dimension_table is not None:
        dimension_table.setRowCount(0)
        for raw in geometry.get("dimensions", []) if isinstance(geometry.get("dimensions", []), list) else []:
            if not isinstance(raw, Mapping):
                continue
            try:
                x1, y1 = self._xy_pair(raw.get("start", raw.get("p1", [0.0, 0.0])))
                x2, y2 = self._xy_pair(raw.get("end", raw.get("p2", [0.0, 0.0])))
            except ValueError:
                continue
            self.add_geometry_dimension_row(
                dimension_id=str(raw.get("id", "")),
                layer=str(raw.get("layer", "")),
                x1=f"{x1:g}",
                y1=f"{y1:g}",
                x2=f"{x2:g}",
                y2=f"{y2:g}",
                text=str(raw.get("text", "")),
                constraint=str(raw.get("constraint", raw.get("constraint_type", ""))),
                locked=raw.get("locked", raw.get("lock", False)),
            )
    self.populate_geometry_curve_tables()
    self.populate_curve_control_point_table()
    self.populate_geometry_boolean_table()
    self.refresh_cad_overlap_edges()
    self.refresh_cad_repair_diagnostics(write=False)

def add_geometry_line_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_line_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"line_id": f"line_{row + 1}", "purpose": "model", "layer": "", "x1": "0.0", "y1": "0.0", "x2": "10.0", "y2": "0.0"}
    defaults.update(values)
    for col, key in enumerate(["line_id", "purpose", "layer", "x1", "y1", "x2", "y2"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def add_geometry_tunnel_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_tunnel_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"tunnel_id": f"tunnel_{row + 1}", "layer": "", "cx": "5.0", "cy": "1.0", "radius": "0.5", "segments": "24"}
    defaults.update(values)
    for col, key in enumerate(["tunnel_id", "layer", "cx", "cy", "radius", "segments"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def add_geometry_layer_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_layer_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"visible": True, "locked": False, "name": f"Layer{row + 1}", "color": "", "linetype": "", "lineweight": "", "opacity": "", "source": "gui"}
    defaults.update(values)
    for col, key in enumerate(["visible", "locked", "name", "color", "linetype", "lineweight", "opacity", "source"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def add_geometry_annotation_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_annotation_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"annotation_id": f"note_{row + 1}", "layer": "", "x": "0.0", "y": "0.0", "text": "注記"}
    defaults.update(values)
    for col, key in enumerate(["annotation_id", "layer", "x", "y", "text"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def add_geometry_dimension_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_dimension_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"dimension_id": f"dim_{row + 1}", "layer": "", "x1": "0.0", "y1": "0.0", "x2": "1.0", "y2": "0.0", "text": "", "constraint": "", "locked": False}
    defaults.update(values)
    for col, key in enumerate(["dimension_id", "layer", "x1", "y1", "x2", "y2", "text", "constraint", "locked"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def remove_selected_geometry_rows(owner: Any, qt: Mapping[str, Any], table: QTableWidget) -> None:
    self = owner
    _bind_qt(qt)
    before = table.rowCount()
    self.remove_selected_rows(table)
    if table.rowCount() == before:
        return
    if hasattr(self, "_geometry_detail_tables_dirty"):
        self._geometry_detail_tables_dirty = True
    source_tables = (
        getattr(self, "geometry_line_table", None),
        getattr(self, "geometry_tunnel_table", None),
        getattr(self, "geometry_layer_table", None),
        getattr(self, "geometry_annotation_table", None),
        getattr(self, "geometry_dimension_table", None),
        getattr(self, "geometry_curve_table", None),
        getattr(self, "geometry_curve_control_table", None),
        getattr(self, "geometry_overlap_table", None),
    )
    if any(table is source_table for source_table in source_tables) and hasattr(self, "apply_geometry_panel"):
        if self.apply_geometry_panel(after_change=False) is False:
            return
        if hasattr(self, "_geometry_detail_tables_dirty"):
            self._geometry_detail_tables_dirty = False
        if hasattr(self, "_after_form_change"):
            self._after_form_change("形状/CAD表の行削除を作図パレットへ反映しました")

def remove_selected_cad_text_rows(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    for table in (self.geometry_annotation_table, self.geometry_dimension_table):
        if table.selectedIndexes():
            self.remove_selected_rows(table)

def _selected_geometry_endpoint(owner: Any, qt: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
    self = owner
    _bind_qt(qt)
    for item in self.scene.selectedItems():
        data = item.data(0)
        if not isinstance(data, Mapping):
            continue
        kind = str(data.get("kind", ""))
        if kind == "geometry_endpoint":
            line = self._geometry_line_by_id(str(data.get("id", "")))
            if isinstance(line, dict):
                endpoint = "start" if str(data.get("endpoint", "end")) == "start" else "end"
                return line, endpoint
    return None

def _geometry_point_widgets(owner: Any, *, control_scope: str = "auto") -> dict[str, Any]:
    scope = str(control_scope or "auto").strip().lower()
    if scope == "model":
        return {
            "target": getattr(owner, "model_geometry_point_target", None),
            "x": getattr(owner, "model_geometry_point_x", None),
            "y": getattr(owner, "model_geometry_point_y", None),
            "dx": getattr(owner, "model_geometry_point_dx", None),
            "dy": getattr(owner, "model_geometry_point_dy", None),
            "step": getattr(owner, "model_geometry_nudge_step", None),
        }
    return {
        "target": getattr(owner, "geometry_point_target", None),
        "x": getattr(owner, "geometry_point_x", None),
        "y": getattr(owner, "geometry_point_y", None),
        "dx": getattr(owner, "geometry_point_dx", None),
        "dy": getattr(owner, "geometry_point_dy", None),
        "step": getattr(owner, "geometry_nudge_step", None),
    }

def _selected_geometry_line_target(owner: Any, qt: Mapping[str, Any], control_scope: str = "auto") -> tuple[dict[str, Any], str] | None:
    self = owner
    _bind_qt(qt)
    target_combo = _geometry_point_widgets(self, control_scope=control_scope).get("target")
    target = str(target_combo.currentData() if target_combo is not None else "selected")
    endpoint = self._selected_geometry_endpoint()
    if target == "selected":
        return endpoint
    line = self._selected_geometry_line()
    if not isinstance(line, dict) and endpoint is not None:
        line = endpoint[0]
    if not isinstance(line, dict):
        return None
    if target not in {"start", "end", "both"}:
        target = endpoint[1] if endpoint is not None and endpoint[0] is line else "start"
    return line, target

def load_selected_geometry_point_to_editor(owner: Any, qt: Mapping[str, Any], control_scope: str = "auto") -> None:
    self = owner
    _bind_qt(qt)
    widgets = _geometry_point_widgets(self, control_scope=control_scope)
    selected = self._selected_geometry_line_target(control_scope=control_scope)
    if selected is None:
        QMessageBox.information(self, "GeoFEM", "モデルビューで線または端点を選択してください。")
        return
    line, target = selected
    try:
        x1, y1 = self._xy_pair(line.get("start", [0.0, 0.0]))
        x2, y2 = self._xy_pair(line.get("end", [0.0, 0.0]))
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if target == "end":
        x, y = x2, y2
    elif target == "both":
        x, y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    else:
        x, y = x1, y1
    if widgets.get("x") is not None:
        widgets["x"].setText(f"{x:.12g}")
    if widgets.get("y") is not None:
        widgets["y"].setText(f"{y:.12g}")
    self.statusBar().showMessage(f"座標を読込みました: {line.get('id', '')} {target} ({x:.6g}, {y:.6g})")

def apply_geometry_point_absolute_edit(owner: Any, qt: Mapping[str, Any], control_scope: str = "auto") -> None:
    self = owner
    _bind_qt(qt)
    widgets = _geometry_point_widgets(self, control_scope=control_scope)
    selected = self._selected_geometry_line_target(control_scope=control_scope)
    if selected is None:
        QMessageBox.information(self, "GeoFEM", "モデルビューで線または端点を選択してください。")
        return
    try:
        x_widget = widgets.get("x")
        y_widget = widgets.get("y")
        x = self._float_text(x_widget.text() if x_widget is not None else "", "X")
        y = self._float_text(y_widget.text() if y_widget is not None else "", "Y")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if self._move_geometry_point_target(selected[0], selected[1], absolute=(x, y)):
        self._after_form_change("形状端点の座標を微修正しました")

def nudge_selected_geometry_point(owner: Any, qt: Mapping[str, Any], axis: str = "", sign: float = 1.0, control_scope: str = "auto") -> None:
    self = owner
    _bind_qt(qt)
    widgets = _geometry_point_widgets(self, control_scope=control_scope)
    selected = self._selected_geometry_line_target(control_scope=control_scope)
    if selected is None:
        QMessageBox.information(self, "GeoFEM", "モデルビューで線または端点を選択してください。")
        return
    try:
        if axis == "x":
            step_widget = widgets.get("step")
            dx, dy = self._float_text(step_widget.text() if step_widget is not None else "", "微修正量") * float(sign), 0.0
        elif axis == "y":
            step_widget = widgets.get("step")
            dx, dy = 0.0, self._float_text(step_widget.text() if step_widget is not None else "", "微修正量") * float(sign)
        else:
            dx_widget = widgets.get("dx")
            dy_widget = widgets.get("dy")
            dx = self._float_text(dx_widget.text() if dx_widget is not None else "", "dX")
            dy = self._float_text(dy_widget.text() if dy_widget is not None else "", "dY")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if self._move_geometry_point_target(selected[0], selected[1], delta=(dx, dy)):
        self._after_form_change("形状端点を相対移動しました")

def _geometry_repair_tolerance(owner: Any) -> float:
    geometry = owner._mapping(owner.cfg.get("geometry", {}))
    for key in ("repair_tolerance", "closure_repair_tolerance", "gap_tolerance"):
        try:
            value = float(geometry.get(key, ""))
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            return value
    try:
        grid = float(owner.snap_grid_size.text())
    except Exception:
        grid = 1.0
    try:
        snap = float(owner._snap_tolerance_model())
    except Exception:
        snap = 1.0e-6
    return max(grid * 0.25, snap, 1.0e-7)

def _cad_diagnostic_line_records(owner: Any) -> list[dict[str, Any]]:
    geometry = owner._mapping(owner.cfg.get("geometry", {}))
    records: list[dict[str, Any]] = []
    lines = geometry.get("lines", [])
    if not isinstance(lines, list):
        return records
    for index, raw in enumerate(lines):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("purpose", "model")).strip().lower() == "helper":
            continue
        try:
            x1, y1 = owner._xy_pair(raw.get("start", raw.get("p1", [0.0, 0.0])))
            x2, y2 = owner._xy_pair(raw.get("end", raw.get("p2", [0.0, 0.0])))
        except ValueError:
            continue
        line_id = str(raw.get("id", f"line_{index + 1}")) or f"line_{index + 1}"
        records.append(
            {
                "id": line_id,
                "index": index,
                "line": raw,
                "start": (float(x1), float(y1)),
                "end": (float(x2), float(y2)),
                "length": math.hypot(float(x2) - float(x1), float(y2) - float(y1)),
            }
        )
    return records

def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def _segments_equivalent(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> bool:
    return (
        _distance(left["start"], right["start"]) <= tolerance
        and _distance(left["end"], right["end"]) <= tolerance
    ) or (
        _distance(left["start"], right["end"]) <= tolerance
        and _distance(left["end"], right["start"]) <= tolerance
    )

def _segments_share_endpoint(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> bool:
    return any(
        _distance(a, b) <= tolerance
        for a in (left["start"], left["end"])
        for b in (right["start"], right["end"])
    )

def cad_repair_diagnostics_snapshot(owner: Any, qt: Mapping[str, Any], *, line_limit: int | None = None) -> dict[str, Any]:
    self = owner
    _bind_qt(qt)
    geometry = self._mapping(self.cfg.get("geometry", {}))
    tolerance = _geometry_repair_tolerance(self)
    try:
        closure_tolerance = float(geometry.get("closure_tolerance", ""))
    except (TypeError, ValueError):
        closure_tolerance = max(tolerance * 1.0e-4, 1.0e-7)
    try:
        short_tolerance = float(geometry.get("short_line_tolerance", ""))
    except (TypeError, ValueError):
        short_tolerance = max(tolerance * 0.05, closure_tolerance * 10.0, 1.0e-9)
    if line_limit is None:
        try:
            line_limit = int(float(geometry.get("cad_auto_diagnostic_line_limit", 1000)))
        except (TypeError, ValueError):
            line_limit = 1000
    records = _cad_diagnostic_line_records(self)
    candidates: list[dict[str, Any]] = []
    repair_line_ids: set[str] = set()
    repair_endpoint_keys: set[tuple[str, str]] = set()

    def add_candidate(kind: str, severity: str, line_ids: list[str], message: str, action: str, **extra: Any) -> None:
        row = {
            "kind": kind,
            "severity": severity,
            "line_ids": [str(value) for value in line_ids if str(value)],
            "message": message,
            "action": action,
        }
        row.update(extra)
        candidates.append(row)
        for line_id in row["line_ids"]:
            repair_line_ids.add(line_id)
        for key in row.get("endpoint_keys", []):
            if isinstance(key, (list, tuple)) and len(key) == 2:
                repair_endpoint_keys.add((str(key[0]), str(key[1])))

    for record in records:
        if float(record["length"]) <= short_tolerance:
            add_candidate(
                "short_line",
                "WARN",
                [str(record["id"])],
                f"極短線分です: {record['id']} 長さ={record['length']:.6g}",
                "削除、延長、または近接端点へ結合してください。",
                distance=float(record["length"]),
                endpoint_keys=[(str(record["id"]), "start"), (str(record["id"]), "end")],
            )

    open_keys = sorted(self._geometry_open_endpoint_key_set())
    for line_id, endpoint in open_keys:
        add_candidate(
            "open_endpoint",
            "ERROR",
            [line_id],
            f"未閉合端点です: {line_id}.{endpoint}",
            "端点結合、閉合線、延長、補助線化で閉合してください。",
            endpoint=line_id + "." + endpoint,
            endpoint_keys=[(line_id, endpoint)],
        )

    skipped_pair_checks = len(records) > max(0, int(line_limit))
    if not skipped_pair_checks:
        from geofem_app.mesh_generation import segment_intersection

        endpoints: list[tuple[str, str, tuple[float, float]]] = []
        for record in records:
            endpoints.append((str(record["id"]), "start", record["start"]))
            endpoints.append((str(record["id"]), "end", record["end"]))
        for left_index, (left_id, left_endpoint, left_point) in enumerate(endpoints):
            for right_id, right_endpoint, right_point in endpoints[left_index + 1 :]:
                if left_id == right_id:
                    continue
                gap = _distance(left_point, right_point)
                if closure_tolerance < gap <= tolerance:
                    add_candidate(
                        "endpoint_gap",
                        "WARN",
                        [left_id, right_id],
                        f"端点ギャップです: {left_id}.{left_endpoint} - {right_id}.{right_endpoint} 距離={gap:.6g}",
                        "端点結合またはスナップで同一点へ寄せてください。",
                        distance=gap,
                        endpoint_keys=[(left_id, left_endpoint), (right_id, right_endpoint)],
                    )
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                if _segments_equivalent(left, right, tolerance):
                    add_candidate(
                        "duplicate_line",
                        "WARN",
                        [str(left["id"]), str(right["id"])],
                        f"重複線分です: {left['id']} / {right['id']}",
                        "どちらかを削除、または片方を補助線化してください。",
                    )
                    continue
                hit = segment_intersection(left["start"], left["end"], right["start"], right["end"])
                if hit is None:
                    continue
                t, u, hx, hy = hit
                if _segments_share_endpoint(left, right, tolerance):
                    continue
                hit_point = (float(hx), float(hy))
                if all(_distance(point, hit_point) > tolerance for point in (left["start"], left["end"], right["start"], right["end"])):
                    add_candidate(
                        "self_intersection",
                        "ERROR",
                        [str(left["id"]), str(right["id"])],
                        f"自己交差候補です: {left['id']} / {right['id']} @ ({hx:.6g}, {hy:.6g})",
                        "交点で分割、トリム、または不要線分を削除してください。",
                        point=[float(hx), float(hy)],
                        parameters=[float(t), float(u)],
                    )
    else:
        add_candidate(
            "diagnostic_skipped",
            "INFO",
            [],
            f"線分数が多いためペア診断を省略しました: {len(records)}本",
            "cad_auto_diagnostic_line_limitを上げるか、範囲を絞って診断してください。",
        )

    summary = {
        "line_count": len(records),
        "candidate_count": len(candidates),
        "error_count": sum(1 for row in candidates if str(row.get("severity", "")).upper() == "ERROR"),
        "warning_count": sum(1 for row in candidates if str(row.get("severity", "")).upper() == "WARN"),
        "repair_line_ids": sorted(repair_line_ids),
        "repair_endpoint_keys": [[line_id, endpoint] for line_id, endpoint in sorted(repair_endpoint_keys)],
        "pair_checks_skipped": skipped_pair_checks,
        "tolerance": tolerance,
        "closure_tolerance": closure_tolerance,
        "short_line_tolerance": short_tolerance,
    }
    return {
        "schema": "geofem.gui.cad_repair_diagnostics.v1",
        "summary": summary,
        "candidates": candidates,
    }

def populate_cad_repair_table(owner: Any, qt: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_repair_table", None)
    if table is None:
        return
    if diagnostics is None:
        diagnostics = cad_repair_diagnostics_snapshot(self, qt)
    candidates = diagnostics.get("candidates", []) if isinstance(diagnostics, Mapping) else []
    table.setRowCount(0)
    if not isinstance(candidates, list):
        return
    for row_data in candidates:
        if not isinstance(row_data, Mapping):
            continue
        row = table.rowCount()
        table.insertRow(row)
        values = [
            str(row_data.get("severity", "")),
            str(row_data.get("kind", "")),
            ",".join(str(value) for value in row_data.get("line_ids", [])),
            str(row_data.get("endpoint", "")),
            f"{float(row_data.get('distance', 0.0)):.6g}" if row_data.get("distance") not in (None, "") else "",
            str(row_data.get("action", "")),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, dict(row_data))
            table.setItem(row, col, item)

def refresh_cad_repair_diagnostics(owner: Any, qt: Mapping[str, Any], *_args: Any, write: bool = True) -> dict[str, Any]:
    self = owner
    _bind_qt(qt)
    diagnostics = cad_repair_diagnostics_snapshot(self, qt)
    summary = diagnostics.get("summary", {}) if isinstance(diagnostics, Mapping) else {}
    self._geometry_repair_line_ids = set(str(value) for value in summary.get("repair_line_ids", []))
    self._geometry_repair_endpoint_keys = {
        (str(value[0]), str(value[1]))
        for value in summary.get("repair_endpoint_keys", [])
        if isinstance(value, (list, tuple)) and len(value) == 2
    }
    if write:
        geometry = self._geometry_cfg()
        geometry["cad_repair_diagnostics"] = diagnostics
    self.populate_cad_repair_table(diagnostics)
    if write:
        self._after_form_change(
            f"CAD修復診断: ERROR {summary.get('error_count', 0)} / WARN {summary.get('warning_count', 0)} / 候補 {summary.get('candidate_count', 0)}"
        )
    return diagnostics

def select_cad_repair_candidate(owner: Any, qt: Mapping[str, Any], row_index: int | None = None, _col: int | None = None) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_repair_table", None)
    if table is None:
        return
    row = int(row_index if row_index is not None and row_index >= 0 else table.currentRow())
    if row < 0 or row >= table.rowCount():
        return
    item = table.item(row, 0)
    payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
    if not isinstance(payload, Mapping):
        return
    line_ids = {str(value) for value in payload.get("line_ids", [])}
    endpoint_keys = {
        (str(value[0]), str(value[1]))
        for value in payload.get("endpoint_keys", [])
        if isinstance(value, (list, tuple)) and len(value) == 2
    }
    timer = getattr(self, "preview_update_timer", None)
    if timer is not None and timer.isActive() and hasattr(self, "_run_pending_preview_update"):
        timer.stop()
        self._run_pending_preview_update()
    self.scene.clearSelection()
    selected_items = []
    for scene_item in self.scene.items():
        data = scene_item.data(0)
        if not isinstance(data, Mapping):
            continue
        kind = str(data.get("kind", ""))
        line_id = str(data.get("id", ""))
        matched = kind == "geometry_line" and line_id in line_ids
        if kind == "geometry_endpoint":
            endpoint = str(data.get("endpoint", ""))
            matched = matched or (line_id, endpoint) in endpoint_keys
        if matched:
            scene_item.setSelected(True)
            selected_items.append(scene_item)
    if selected_items:
        rect = selected_items[0].sceneBoundingRect()
        for scene_item in selected_items[1:]:
            rect = rect.united(scene_item.sceneBoundingRect())
        self.view.fitInView(rect.adjusted(-30.0, -30.0, 30.0, 30.0), Qt.AspectRatioMode.KeepAspectRatio)
        self.statusBar().showMessage(f"CAD修復候補を選択しました: {payload.get('kind', '')}")

def _line_endpoint(owner: Any, line: Mapping[str, Any], endpoint: str) -> tuple[float, float]:
    key = "start" if endpoint == "start" else "end"
    fallback = "p1" if key == "start" else "p2"
    return owner._xy_pair(line.get(key, line.get(fallback, [0.0, 0.0])))

def _line_endpoint_records(owner: Any, *, include_helper: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = owner._geometry_cfg().get("lines", [])
    if not isinstance(lines, list):
        return records
    for line in lines:
        if not isinstance(line, dict):
            continue
        purpose = str(line.get("purpose", "model")).strip().lower()
        if purpose == "helper" and not include_helper:
            continue
        line_id = str(line.get("id", ""))
        if not line_id:
            continue
        for endpoint in ("start", "end"):
            try:
                x, y = _line_endpoint(owner, line, endpoint)
            except ValueError:
                continue
            records.append({"line": line, "line_id": line_id, "endpoint": endpoint, "x": x, "y": y})
    return records

def _nearest_repair_endpoint(
    owner: Any,
    line: Mapping[str, Any],
    endpoint: str,
    *,
    open_only: bool = False,
    tolerance: float | None = None,
) -> dict[str, Any] | None:
    x, y = _line_endpoint(owner, line, endpoint)
    limit = _geometry_repair_tolerance(owner) if tolerance is None else max(float(tolerance), 1.0e-12)
    open_keys = owner._geometry_open_endpoint_key_set() if open_only else set()
    best: tuple[float, dict[str, Any]] | None = None
    for record in _line_endpoint_records(owner):
        if record["line"] is line and record["endpoint"] == endpoint:
            continue
        if record["line"] is line:
            continue
        if open_only and (str(record["line_id"]), str(record["endpoint"])) not in open_keys:
            continue
        distance = math.hypot(x - float(record["x"]), y - float(record["y"]))
        if distance <= 1.0e-12 or distance > limit:
            continue
        if best is None or distance < best[0]:
            best = (distance, record)
    return dict(best[1]) | {"distance": best[0]} if best is not None else None

def join_selected_geometry_endpoint_to_nearest(owner: Any, qt: Mapping[str, Any], tolerance: float | None = None) -> None:
    self = owner
    _bind_qt(qt)
    selected = self._selected_geometry_endpoint() or self._selected_geometry_line_target()
    if selected is None:
        QMessageBox.information(self, "GeoFEM", "結合する線分端点を選択してください。")
        return
    line, endpoint = selected
    if endpoint == "both":
        endpoint = "end"
    nearest = _nearest_repair_endpoint(self, line, endpoint, tolerance=tolerance)
    if nearest is None:
        self.statusBar().showMessage("結合できる近接端点が許容距離内にありません。")
        return
    target = (float(nearest["x"]), float(nearest["y"]))
    if self._move_geometry_point_target(line, endpoint, absolute=target):
        self._after_form_change(
            f"端点結合: {line.get('id', '')}.{endpoint} -> {nearest.get('line_id', '')}.{nearest.get('endpoint', '')}"
        )

def add_closure_segment_from_selected_endpoint(owner: Any, qt: Mapping[str, Any], tolerance: float | None = None) -> None:
    self = owner
    _bind_qt(qt)
    selected = self._selected_geometry_endpoint() or self._selected_geometry_line_target()
    if selected is None:
        QMessageBox.information(self, "GeoFEM", "閉合線を追加する始点端点を選択してください。")
        return
    line, endpoint = selected
    if endpoint == "both":
        endpoint = "end"
    nearest = _nearest_repair_endpoint(self, line, endpoint, open_only=True, tolerance=tolerance)
    if nearest is None:
        self.statusBar().showMessage("閉合補助に使える未閉合端点が許容距離内にありません。")
        return
    x1, y1 = _line_endpoint(self, line, endpoint)
    x2, y2 = float(nearest["x"]), float(nearest["y"])
    lines = self._geometry_cfg().setdefault("lines", [])
    if not isinstance(lines, list):
        lines = []
        self._geometry_cfg()["lines"] = lines
    lines.append(
        {
            "id": self._next_id("line", lines),
            "purpose": "model",
            "start": [float(x1), float(y1)],
            "end": [x2, y2],
            "source": "closure_helper",
        }
    )
    self._after_form_change(
        f"閉合補助線を追加しました: {line.get('id', '')}.{endpoint} -> {nearest.get('line_id', '')}.{nearest.get('endpoint', '')}"
    )

def _selected_geometry_line_ids(owner: Any) -> list[str]:
    ids: list[str] = []
    for item in owner.scene.selectedItems():
        data = item.data(0)
        if isinstance(data, Mapping) and data.get("kind") in {"geometry_line", "geometry_endpoint"} and data.get("id") is not None:
            ids.append(str(data.get("id")))
    table = getattr(owner, "geometry_line_table", None)
    if table is not None:
        for index in table.selectedIndexes():
            item = table.item(index.row(), 0)
            if item is not None and item.text():
                ids.append(str(item.text()))
    seen: set[str] = set()
    deduped: list[str] = []
    for line_id in ids:
        if line_id and line_id not in seen:
            seen.add(line_id)
            deduped.append(line_id)
    return deduped

def set_selected_geometry_lines_purpose(owner: Any, qt: Mapping[str, Any], purpose: str = "helper") -> None:
    self = owner
    _bind_qt(qt)
    purpose = "helper" if str(purpose).strip().lower() == "helper" else "model"
    selected_ids = _selected_geometry_line_ids(self)
    if not selected_ids:
        QMessageBox.information(self, "GeoFEM", "補助線/分割線へ切り替える線分を選択してください。")
        return
    changed = 0
    for line_id in selected_ids:
        line = self._geometry_line_by_id(line_id)
        if isinstance(line, dict) and str(line.get("purpose", "model")).strip().lower() != purpose:
            line["purpose"] = purpose
            changed += 1
    if changed == 0:
        self.statusBar().showMessage("選択線分の種別はすでに指定状態です。")
        return
    label = "補助線化" if purpose == "helper" else "メッシュ分割線化"
    self._after_form_change(f"{label}: {changed}件")

def _move_geometry_point_target(
    owner: Any,
    qt: Mapping[str, Any],
    line: dict[str, Any],
    target: str,
    *,
    absolute: tuple[float, float] | None = None,
    delta: tuple[float, float] | None = None,
) -> bool:
    self = owner
    _bind_qt(qt)
    try:
        x1, y1 = self._xy_pair(line.get("start", [0.0, 0.0]))
        x2, y2 = self._xy_pair(line.get("end", [0.0, 0.0]))
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return False
    if absolute is not None:
        ax, ay = absolute
        if target == "both":
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            dx, dy = ax - cx, ay - cy
        elif target == "end":
            dx, dy = ax - x2, ay - y2
        else:
            dx, dy = ax - x1, ay - y1
    else:
        dx, dy = delta or (0.0, 0.0)
    if target in {"start", "end"} and hasattr(self, "_coincident_geometry_endpoint_refs") and hasattr(self, "_move_geometry_endpoint_refs"):
        base_x, base_y = (x1, y1) if target == "start" else (x2, y2)
        refs = self._coincident_geometry_endpoint_refs(line, target)
        if refs and self._move_geometry_endpoint_refs(refs, base_x + dx, base_y + dy):
            return True
    if target in {"start", "both"}:
        line["start"] = [float(x1 + dx), float(y1 + dy)]
    if target in {"end", "both"}:
        line["end"] = [float(x2 + dx), float(y2 + dy)]
    if target not in {"start", "end", "both"}:
        line["start"] = [float(x1 + dx), float(y1 + dy)]
    return True

def set_all_cad_layers_visible(owner: Any, qt: Mapping[str, Any], visible: bool) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_layer_table
    for row in range(table.rowCount()):
        table.setItem(row, 0, QTableWidgetItem(str(visible)))
    self.apply_geometry_panel()

def set_selected_cad_layers_locked(owner: Any, qt: Mapping[str, Any], locked: bool) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_layer_table
    rows = sorted({index.row() for index in table.selectedIndexes()})
    if not rows:
        rows = list(range(table.rowCount()))
    for row in rows:
        table.setItem(row, 1, QTableWidgetItem(str(locked)))
    self.apply_geometry_panel()

def apply_dimension_constraints(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    try:
        constraints = self._dimension_constraints_from_table()
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"dimension constraint failed: {exc}")
        return
    if self.apply_geometry_panel(after_change=False) is False:
        return
    geometry = self._geometry_cfg()
    if constraints:
        geometry["dimension_constraints"] = constraints
        self._solve_dimension_constraints(geometry, constraints)
    else:
        geometry.pop("dimension_constraints", None)
        geometry.pop("dimension_solver", None)
    self._after_form_change("dimension constraints solved and applied to CAD geometry")

def apply_dimension_constraints_async(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    if self._cad_job_id:
        self.statusBar().showMessage("CAD処理ジョブを実行中です。")
        return
    try:
        constraints = self._dimension_constraints_from_table()
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"dimension constraint failed: {exc}")
        return
    if self.apply_geometry_panel(after_change=False) is False:
        return
    geometry = self._geometry_cfg()
    if not constraints:
        geometry.pop("dimension_constraints", None)
        geometry.pop("dimension_solver", None)
        self._after_form_change("dimension constraints cleared")
        return
    self._start_dimension_constraints_job(geometry, constraints, "dimension constraints solved and applied to CAD geometry")

def _start_dimension_constraints_job(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any], constraints: list[dict[str, Any]], message: str) -> bool:
    self = owner
    _bind_qt(qt)
    if self._cad_job_id:
        self.statusBar().showMessage("CAD処理ジョブを実行中です。")
        return False
    geometry_snapshot = copy.deepcopy(dict(self._mapping(geometry)))
    job_id = self.gui_jobs.start_job(
        "cad_geometry",
        target=str(self.current_input or "current-config"),
        metadata={"operation": "dimension_constraints", "constraint_count": len(constraints)},
    )
    self._cad_job_id = job_id
    self._cad_dimension_completion_message = message
    runner = QtCallableRunner(
        job_id,
        lambda: solve_dimension_constraints_snapshot(type(self), geometry_snapshot, constraints),
    )
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._cad_dimension_constraints_finished)
    runner.signals.failed.connect(self._cad_job_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage("CAD寸法拘束をバックグラウンド解法中...")
    return True

def _cad_dimension_constraints_finished(owner: Any, qt: Mapping[str, Any], job_id: str, result_obj: Any) -> None:
    self = owner
    _bind_qt(qt)
    if job_id != self._cad_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    result = dict(result_obj or {})
    diagnostics = self._mapping(result.get("diagnostics", {}))
    constraint_count = int(result.get("constraint_count", 0) or 0)
    max_residual = float(diagnostics.get("max_abs_residual", 0.0) or 0.0)
    self._complete_gui_worker_job(job_id, status="finished", message=f"{constraint_count} constraints")
    self._cad_job_id = ""
    message = self._cad_dimension_completion_message or "dimension constraints solved and applied to CAD geometry"
    self._cad_dimension_completion_message = ""
    self.cfg["geometry"] = dict(self._mapping(result.get("geometry", {})))
    self._after_form_change(message)
    self.append_log(f"[GUI] CAD寸法拘束解法完了: constraints={constraint_count}, max_residual={max_residual:.6g}")

def _dimension_constraints_from_table(owner: Any, qt: Mapping[str, Any]) -> list[dict[str, Any]]:
    self = owner
    _bind_qt(qt)
    constraints: list[dict[str, Any]] = []
    table = self.geometry_dimension_table
    for row in range(table.rowCount()):
        dimension_id = self._table_text(table, row, 0).strip() or f"dim_{row + 1}"
        x1 = self._float_text(self._table_text(table, row, 2), "dimension x1")
        y1 = self._float_text(self._table_text(table, row, 3), "dimension y1")
        x2 = self._float_text(self._table_text(table, row, 4), "dimension x2")
        y2 = self._float_text(self._table_text(table, row, 5), "dimension y2")
        text = self._table_text(table, row, 6).strip()
        constraint = self._table_text(table, row, 7).strip()
        locked = self._bool_text(self._table_text(table, row, 8), False)
        if not constraint and not locked:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        target = self._dimension_target_value(text, length)
        constraint_info = self._parse_dimension_constraint(constraint or "length", text)
        constraints.append(
            {
                "id": dimension_id,
                "type": constraint_info["type"],
                "locked": locked,
                "start": [x1, y1],
                "end": [x2, y2],
                "value": target,
                "measured": length,
                "source": "gui_dimension_table",
                **{key: value for key, value in constraint_info.items() if key != "type"},
            }
        )
        if not text:
            table.setItem(row, 6, QTableWidgetItem(f"{length:.6g}"))
    return constraints

def _dimension_constraints_from_geometry_dimensions(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    self = owner
    _bind_qt(qt)
    constraints: list[dict[str, Any]] = []
    dimensions = geometry.get("dimensions", [])
    if not isinstance(dimensions, list):
        return constraints
    for index, raw in enumerate(dimensions, start=1):
        if not isinstance(raw, Mapping):
            continue
        constraint = str(raw.get("constraint", raw.get("constraint_type", ""))).strip()
        locked = self._bool_text(str(raw.get("locked", raw.get("lock", False))), False)
        if not constraint and not locked:
            continue
        try:
            x1, y1 = self._xy_pair(raw.get("start", raw.get("p1", [0.0, 0.0])))
            x2, y2 = self._xy_pair(raw.get("end", raw.get("p2", [0.0, 0.0])))
        except ValueError:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        constraint_info = self._parse_dimension_constraint(constraint or "length", str(raw.get("text", "")))
        constraints.append(
            {
                "id": str(raw.get("id", f"dim_{index}")),
                "type": constraint_info["type"],
                "locked": locked,
                "start": [x1, y1],
                "end": [x2, y2],
                "value": self._dimension_target_value(str(raw.get("text", "")), length),
                "measured": length,
                "source": "gui_dimension_drag",
                **{key: value for key, value in constraint_info.items() if key != "type"},
            }
        )
    return constraints

def _parse_dimension_constraint(owner: Any, qt: Mapping[str, Any], raw_constraint: str, text: str = "") -> dict[str, Any]:
    self = owner
    _bind_qt(qt)
    raw = str(raw_constraint or "length").strip()
    lower = raw.lower()
    reference = ""
    if ":" in lower:
        head, tail = raw.split(":", 1)
        lower = head.strip().lower()
        reference = tail.strip()
    elif "@" in lower:
        head, tail = raw.split("@", 1)
        lower = head.strip().lower()
        reference = tail.strip()
    angle_match = re.match(r"angle\s*[=:]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", lower)
    if angle_match:
        return {"type": "angle", "angle_degrees": float(angle_match.group(1))}
    if lower in {"distance", "dim", "dimension", "locked"}:
        lower = "length"
    elif lower in {"h", "horiz"}:
        lower = "horizontal"
    elif lower in {"v", "vert"}:
        lower = "vertical"
    elif lower in {"perp", "orthogonal"}:
        lower = "perpendicular"
    elif lower in {"equal", "same_length", "equal-length"}:
        lower = "equal_length"
    info: dict[str, Any] = {"type": lower or "length"}
    if reference:
        info["reference"] = reference
    if info["type"] == "angle":
        info["angle_degrees"] = self._dimension_target_value(text, 0.0)
    return info

def _dimension_target_value(owner: Any, qt: Mapping[str, Any], text: str, fallback: float) -> float:
    self = owner
    _bind_qt(qt)
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(text or ""))
    if match is None:
        return float(fallback)
    try:
        return float(match.group(0))
    except ValueError:
        return float(fallback)

def _solve_dimension_constraints(owner: Any, qt: Mapping[str, Any], geometry: dict[str, Any], constraints: list[dict[str, Any]]) -> None:
    self = owner
    _bind_qt(qt)
    refs = self._cad_constraint_point_refs(geometry)
    tolerance = self._dimension_constraint_tolerance(geometry)
    groups = self._cluster_cad_constraint_refs(refs, tolerance)
    point_specs = {
        group["id"]: {"x": group["x"], "y": group["y"], "locked": group["locked"]}
        for group in groups
    }
    unresolved: list[dict[str, Any]] = []
    solver_constraints: list[dict[str, Any]] = []
    endpoints_by_dimension: dict[str, tuple[str, str]] = {}
    for constraint in constraints:
        try:
            x1, y1 = self._xy_pair(constraint.get("start", [0.0, 0.0]))
            x2, y2 = self._xy_pair(constraint.get("end", [0.0, 0.0]))
            target = float(constraint.get("value", math.hypot(x2 - x1, y2 - y1)))
        except (TypeError, ValueError):
            unresolved.append({"id": constraint.get("id", ""), "reason": "invalid constraint coordinates"})
            continue
        start_group = self._nearest_cad_constraint_group(groups, x1, y1, tolerance)
        end_group = self._nearest_cad_constraint_group(groups, x2, y2, tolerance)
        if start_group is None or end_group is None:
            unresolved.append({"id": constraint.get("id", ""), "reason": "no linked CAD point", "start_group": start_group is not None, "end_group": end_group is not None})
            continue
        constraint_id = str(constraint.get("id", ""))
        endpoints_by_dimension[constraint_id] = (str(start_group["id"]), str(end_group["id"]))
        solver_constraint = {
            "id": constraint_id,
            "type": constraint.get("type", "length"),
            "p1": str(start_group["id"]),
            "p2": str(end_group["id"]),
            "value": target,
        }
        if constraint.get("reference"):
            solver_constraint["reference"] = constraint.get("reference")
        if constraint.get("angle_degrees") is not None:
            solver_constraint["angle_degrees"] = constraint.get("angle_degrees")
        solver_constraints.append(solver_constraint)
    for solver_constraint in solver_constraints:
        reference = str(solver_constraint.get("reference", "")).strip()
        if reference and reference in endpoints_by_dimension:
            solver_constraint["reference_p1"], solver_constraint["reference_p2"] = endpoints_by_dimension[reference]
        elif reference:
            unresolved.append({"id": solver_constraint.get("id", ""), "reason": "reference dimension not found", "reference": reference})
    explicit_constraints = self._cad_kernel_constraints_from_geometry(geometry, groups, endpoints_by_dimension, unresolved)
    solver_constraints.extend(explicit_constraints)
    if solver_constraints and point_specs:
        from geofem_app.cad_constraints import solve_cad_constraints

        result = solve_cad_constraints(point_specs, solver_constraints, tolerance=tolerance, max_iterations=200)
        solved_points = self._mapping(result.get("points", {}))
        for group in groups:
            coord = solved_points.get(str(group["id"]))
            if not isinstance(coord, list) or len(coord) < 2:
                continue
            for ref in group["refs"]:
                if ref.get("locked"):
                    continue
                ref["set"]((float(coord[0]), float(coord[1])))
                ref["x"] = float(coord[0])
                ref["y"] = float(coord[1])
            group["x"] = float(coord[0])
            group["y"] = float(coord[1])
        for constraint in solver_constraints:
            p1 = solved_points.get(str(constraint.get("p1")))
            p2 = solved_points.get(str(constraint.get("p2")))
            if isinstance(p1, list) and len(p1) >= 2 and isinstance(p2, list) and len(p2) >= 2:
                self._update_dimension_geometry_after_constraint(
                    geometry,
                    str(constraint.get("id", "")),
                    (float(p1[0]), float(p1[1])),
                    (float(p2[0]), float(p2[1])),
                    float(constraint.get("value", 0.0)),
                )
        diagnostics = dict(self._mapping(result.get("diagnostics", {})))
    else:
        diagnostics = {
            "engine": "cad_constraint_least_squares",
            "status": "no_constraints",
            "point_count": len(point_specs),
            "constraint_scalar_count": 0,
            "max_abs_residual": 0.0,
        }
    self._resample_curve_region_points(geometry)
    diagnostics["tolerance"] = tolerance
    diagnostics["constraint_count"] = len(solver_constraints)
    diagnostics["explicit_constraint_count"] = len(explicit_constraints)
    diagnostics["unresolved"] = unresolved
    geometry["dimension_solver"] = diagnostics

def _cad_constraint_point_refs(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any]) -> list[dict[str, Any]]:
    self = owner
    _bind_qt(qt)
    refs: list[dict[str, Any]] = []
    layer_map = self._cad_layer_map(geometry)

    def add_ref(raw_point: Any, setter: Any, *, raw: Mapping[str, Any], label: str, locked: bool = False) -> None:
        try:
            x, y = self._xy_pair(raw_point)
        except ValueError:
            return
        refs.append({"x": x, "y": y, "set": setter, "raw": raw, "label": label, "locked": locked or self._cad_layer_locked(raw, layer_map)})

    for raw in geometry.get("lines", []) if isinstance(geometry.get("lines", []), list) else []:
        if not isinstance(raw, dict):
            continue
        for key in ("start", "end"):
            add_ref(raw.get(key, raw.get("p1" if key == "start" else "p2", [0.0, 0.0])), lambda value, raw=raw, key=key: raw.__setitem__(key, [float(value[0]), float(value[1])]), raw=raw, label=f"line:{raw.get('id', '')}:{key}")

    for raw in geometry.get("tunnels", []) if isinstance(geometry.get("tunnels", []), list) else []:
        if not isinstance(raw, dict):
            continue
        add_ref(raw.get("center", [0.0, 0.0]), lambda value, raw=raw: raw.__setitem__("center", [float(value[0]), float(value[1])]), raw=raw, label=f"tunnel:{raw.get('id', '')}:center")

    regions = geometry.get("regions", [])
    if isinstance(regions, list):
        for region_index, region in enumerate(regions, start=1):
            if not isinstance(region, dict):
                continue
            region_locked = self._cad_layer_locked(region, layer_map)
            points = region.get("points", [])
            if isinstance(points, list):
                for point_index, _point in enumerate(points):
                    add_ref(points[point_index], lambda value, points=points, point_index=point_index: points.__setitem__(point_index, [float(value[0]), float(value[1])]), raw=region, label=f"region:{region_index}:point:{point_index + 1}", locked=region_locked)
            for group_role, specs in self._mutable_curve_groups(region):
                for curve_index, spec in enumerate(specs, start=1):
                    if not isinstance(spec, dict):
                        continue
                    locked = region_locked or self._cad_layer_locked(spec, layer_map)
                    key = "control_points" if isinstance(spec.get("control_points"), list) else ("points" if isinstance(spec.get("points"), list) else "")
                    if key:
                        controls = spec.get(key)
                        if isinstance(controls, list):
                            for point_index, _point in enumerate(controls):
                                add_ref(controls[point_index], lambda value, controls=controls, point_index=point_index: controls.__setitem__(point_index, [float(value[0]), float(value[1])]), raw=spec, label=f"curve:{region_index}:{group_role}:{curve_index}:control:{point_index + 1}", locked=locked)
                    for role in ("start", "end", "center"):
                        if role in spec:
                            add_ref(spec.get(role), lambda value, spec=spec, role=role: spec.__setitem__(role, [float(value[0]), float(value[1])]), raw=spec, label=f"curve:{region_index}:{group_role}:{curve_index}:{role}", locked=locked)
    return refs

def _cluster_cad_constraint_refs(owner: Any, qt: Mapping[str, Any], refs: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    self = owner
    _bind_qt(qt)
    groups: list[dict[str, Any]] = []
    for ref in refs:
        x = float(ref.get("x", 0.0))
        y = float(ref.get("y", 0.0))
        match: dict[str, Any] | None = None
        for group in groups:
            if math.hypot(float(group["x"]) - x, float(group["y"]) - y) <= tolerance:
                match = group
                break
        if match is None:
            match = {"id": f"g{len(groups) + 1}", "x": x, "y": y, "refs": [], "locked": False}
            groups.append(match)
        match["refs"].append(ref)
        count = len(match["refs"])
        match["x"] = (float(match["x"]) * (count - 1) + x) / count
        match["y"] = (float(match["y"]) * (count - 1) + y) / count
        match["locked"] = bool(match.get("locked", False) or ref.get("locked", False))
    return groups

def _nearest_cad_constraint_group(owner: Any, qt: Mapping[str, Any], groups: list[dict[str, Any]], x: float, y: float, tolerance: float) -> dict[str, Any] | None:
    self = owner
    _bind_qt(qt)
    best: tuple[float, dict[str, Any]] | None = None
    for group in groups:
        dist = math.hypot(float(group.get("x", 0.0)) - x, float(group.get("y", 0.0)) - y)
        if dist <= tolerance and (best is None or dist < best[0]):
            best = (dist, group)
    return None if best is None else best[1]

def _cad_kernel_constraints_from_geometry(
    owner: Any,
    qt: Mapping[str, Any],
    geometry: Mapping[str, Any],
    groups: list[dict[str, Any]],
    endpoints_by_dimension: Mapping[str, tuple[str, str]],
    unresolved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    self = owner
    _bind_qt(qt)
    raw_constraints: list[Any] = []
    for key in ("cad_constraints", "advanced_constraints", "curve_constraints"):
        raw_value = geometry.get(key, [])
        if isinstance(raw_value, list):
            raw_constraints.extend(raw_value)
    if not raw_constraints:
        return []
    lookup = self._cad_constraint_group_lookup(groups)
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_constraints, start=1):
        if not isinstance(raw, Mapping):
            unresolved.append({"id": f"cad_constraint_{index}", "reason": "constraint is not a mapping"})
            continue
        constraint_id = str(raw.get("id", f"cad_constraint_{index}"))
        constraint_type = str(raw.get("type", raw.get("kind", ""))).strip() or "tangent"
        spec: dict[str, Any] = {"id": constraint_id, "type": constraint_type}
        point_values = self._ensure_list(raw.get("points", raw.get("point_labels", [])))
        reference_values = self._ensure_list(raw.get("reference_points", raw.get("reference_labels", [])))
        for key, aliases in {
            "p1": ("p1", "start", "center", "center_id"),
            "p2": ("p2", "end", "reference_center", "reference_center_id"),
            "p3": ("p3", "third", "next"),
            "reference_p1": ("reference_p1", "ref_p1", "reference_start"),
            "reference_p2": ("reference_p2", "ref_p2", "reference_end"),
            "reference_p3": ("reference_p3", "ref_p3", "reference_third", "reference_next"),
        }.items():
            value = next((raw.get(alias) for alias in aliases if raw.get(alias) is not None), None)
            if value is not None:
                spec[key] = value
        for key, values, offset in (("p", point_values, 0), ("reference_p", reference_values, 0)):
            for local_index, value in enumerate(values, start=1):
                spec[f"{key}{local_index + offset}"] = value
        if "dimension" in raw and str(raw.get("dimension")) in endpoints_by_dimension:
            spec["p1"], spec["p2"] = endpoints_by_dimension[str(raw.get("dimension"))]
        if "reference_dimension" in raw and str(raw.get("reference_dimension")) in endpoints_by_dimension:
            spec["reference_p1"], spec["reference_p2"] = endpoints_by_dimension[str(raw.get("reference_dimension"))]
        for key in ("p1", "p2", "p3", "reference_p1", "reference_p2", "reference_p3"):
            if key not in spec:
                continue
            group_id = self._cad_group_id_from_label(spec[key], lookup)
            if group_id is None:
                unresolved.append({"id": constraint_id, "reason": "CAD constraint point not found", "point": str(spec[key]), "slot": key})
            else:
                spec[key] = group_id
        if "value" in raw:
            spec["value"] = raw.get("value")
        if "angle_degrees" in raw:
            spec["angle_degrees"] = raw.get("angle_degrees")
        required = self._cad_constraint_required_slots(constraint_type)
        missing = [key for key in required if key not in spec]
        if missing:
            unresolved.append({"id": constraint_id, "reason": "missing CAD constraint points", "missing": missing})
            continue
        out.append(spec)
    return out

def _cad_constraint_group_lookup(owner: Any, qt: Mapping[str, Any], groups: list[dict[str, Any]]) -> dict[str, str]:
    self = owner
    _bind_qt(qt)
    lookup: dict[str, str] = {}
    for group in groups:
        group_id = str(group.get("id", ""))
        if not group_id:
            continue
        lookup[group_id] = group_id
        lookup[group_id.lower()] = group_id
        for ref in self._ensure_list(group.get("refs", [])):
            if not isinstance(ref, Mapping):
                continue
            label = str(ref.get("label", "")).strip()
            if label:
                lookup[label] = group_id
                lookup[label.lower()] = group_id
    return lookup

def _cad_group_id_from_label(owner: Any, qt: Mapping[str, Any], value: Any, lookup: Mapping[str, str]) -> str | None:
    self = owner
    _bind_qt(qt)
    if value is None:
        return None
    text = str(value).strip()
    if text in lookup:
        return lookup[text]
    lower = text.lower()
    return lookup.get(lower)

def _cad_constraint_required_slots(owner: Any, qt: Mapping[str, Any], constraint_type: str) -> list[str]:
    self = owner
    _bind_qt(qt)
    ctype = str(constraint_type or "").strip().lower().replace("-", "_")
    if ctype in {"curvature", "curvature_continuity", "g2", "c2", "nurbs_curvature"}:
        return ["p1", "p2", "p3", "reference_p1", "reference_p2", "reference_p3"]
    if ctype in {"parallel", "perpendicular", "equal_length", "same_length", "tangent", "g1", "c1", "tangency"}:
        return ["p1", "p2", "reference_p1", "reference_p2"]
    return ["p1", "p2"]

def _refs_near_point(owner: Any, qt: Mapping[str, Any], refs: list[dict[str, Any]], x: float, y: float, tolerance: float) -> list[dict[str, Any]]:
    self = owner
    _bind_qt(qt)
    return [ref for ref in refs if math.hypot(float(ref["x"]) - x, float(ref["y"]) - y) <= tolerance]

def _refs_centroid(owner: Any, qt: Mapping[str, Any], refs: list[dict[str, Any]]) -> tuple[float, float]:
    self = owner
    _bind_qt(qt)
    if not refs:
        return 0.0, 0.0
    return sum(float(ref["x"]) for ref in refs) / len(refs), sum(float(ref["y"]) for ref in refs) / len(refs)

def _move_refs(owner: Any, qt: Mapping[str, Any], refs: list[dict[str, Any]], x: float, y: float) -> bool:
    self = owner
    _bind_qt(qt)
    changed = False
    for ref in refs:
        if ref.get("locked"):
            continue
        ref["set"]((x, y))
        ref["x"] = float(x)
        ref["y"] = float(y)
        changed = True
    return changed

def _project_dimension_constraint(
    owner: Any,
    qt: Mapping[str, Any],
    start_refs: list[dict[str, Any]],
    end_refs: list[dict[str, Any]],
    start_hint: tuple[float, float],
    end_hint: tuple[float, float],
    target: float,
    constraint_type: str,
) -> bool:
    self = owner
    _bind_qt(qt)
    ctype = str(constraint_type or "length").strip().lower()
    sx, sy = self._refs_centroid(start_refs)
    ex, ey = self._refs_centroid(end_refs)
    dx = ex - sx
    dy = ey - sy
    if "horizontal" in ctype:
        direction = 1.0 if dx >= 0.0 else -1.0
        nx, ny = sx + direction * abs(target), sy
    elif "vertical" in ctype:
        direction = 1.0 if dy >= 0.0 else -1.0
        nx, ny = sx, sy + direction * abs(target)
    else:
        length = math.hypot(dx, dy)
        if length <= 1.0e-14:
            hx = end_hint[0] - start_hint[0]
            hy = end_hint[1] - start_hint[1]
            length = math.hypot(hx, hy)
            dx, dy = (hx, hy) if length > 1.0e-14 else (1.0, 0.0)
            length = math.hypot(dx, dy)
        nx = sx + dx / length * target
        ny = sy + dy / length * target
    if any(not ref.get("locked") for ref in end_refs):
        return self._move_refs(end_refs, nx, ny)
    if any(not ref.get("locked") for ref in start_refs):
        return self._move_refs(start_refs, ex - (nx - sx), ey - (ny - sy))
    return False

def _dimension_constraint_tolerance(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any]) -> float:
    self = owner
    _bind_qt(qt)
    bbox = self._geometry_bbox(geometry)
    if bbox is None:
        return 1.0e-7
    xmin, xmax, ymin, ymax = bbox
    return max(max(xmax - xmin, ymax - ymin, 1.0) * 1.0e-7, 1.0e-8)

def _update_dimension_geometry_after_constraint(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any], dimension_id: str, start: tuple[float, float], end: tuple[float, float], target: float) -> None:
    self = owner
    _bind_qt(qt)
    dimensions = geometry.get("dimensions", [])
    if not isinstance(dimensions, list):
        return
    for raw in dimensions:
        if not isinstance(raw, dict) or str(raw.get("id", "")) != dimension_id:
            continue
        raw["start"] = [float(start[0]), float(start[1])]
        raw["end"] = [float(end[0]), float(end[1])]
        if not str(raw.get("text", "")).strip():
            raw["text"] = f"{target:.6g}"
        return

def _resample_curve_region_points(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    regions = geometry.get("regions", [])
    if not isinstance(regions, list):
        return
    for region in regions:
        if not isinstance(region, dict):
            continue
        boundary = [spec for _role, specs in self._mutable_curve_groups(region) if _role == "boundary" for spec in specs if isinstance(spec, Mapping)]
        if not boundary:
            continue
        sampled = self._sample_curve_chain_points(boundary)
        if len(sampled) >= 3:
            region["points"] = sampled

def _mutable_curve_groups(owner: Any, qt: Mapping[str, Any], region: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    self = owner
    _bind_qt(qt)
    groups: list[tuple[str, list[Any]]] = []
    for key in ("curve_boundary", "boundary", "segments", "curves", "edges"):
        value = region.get(key)
        if isinstance(value, list) and any(isinstance(item, Mapping) and self._curve_mapping_has_parameters(item) for item in value):
            groups.append(("boundary", value))
            break
        if isinstance(value, Mapping) and self._curve_mapping_has_parameters(value):
            region[key] = [dict(value)]
            groups.append(("boundary", region[key]))
            break
    holes = region.get("curve_holes", region.get("holes", region.get("islands", [])))
    if isinstance(holes, list):
        for index, hole in enumerate(holes, start=1):
            if isinstance(hole, list):
                groups.append((f"hole_{index}", hole))
            elif isinstance(hole, Mapping) and self._curve_mapping_has_parameters(hole):
                holes[index - 1] = [dict(hole)]
                groups.append((f"hole_{index}", holes[index - 1]))
    return groups

def add_geometry_curve_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_curve_table
    row = table.rowCount()
    table.insertRow(row)
    curve_type = str(values.get("curve_type", values.get("type", "line"))).strip().lower() or "line"
    params = values.get("parameters", values.get("params"))
    if params is None:
        params = self._default_curve_parameters(curve_type)
    if isinstance(params, Mapping):
        params = yaml.safe_dump(dict(params), allow_unicode=True, sort_keys=False).strip()
    defaults = {
        "region": values.get("region", "1"),
        "index": values.get("index", str(row + 1)),
        "curve_type": curve_type,
        "role": values.get("role", "boundary"),
        "parameters": params,
        "segments": values.get("segments", ""),
    }
    for col, key in enumerate(["region", "index", "curve_type", "role", "parameters", "segments"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def populate_geometry_curve_tables(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_curve_table", None)
    if table is None:
        return
    table.setRowCount(0)
    geometry = self._mapping(self.cfg.get("geometry", {}))
    regions = geometry.get("regions", [])
    if not isinstance(regions, list):
        return
    for region_index, raw_region in enumerate(regions, start=1):
        if not isinstance(raw_region, Mapping):
            continue
        for curve_index, spec in enumerate(self._curve_boundary_specs(raw_region), start=1):
            self.add_geometry_curve_row(
                region=str(region_index),
                index=str(curve_index),
                curve_type=str(spec.get("type", spec.get("kind", "line"))),
                role="boundary",
                parameters=self._curve_parameters_yaml(spec),
                segments=str(spec.get("segments", "")),
            )
        for hole_index, hole in enumerate(self._curve_hole_specs(raw_region), start=1):
            for curve_index, spec in enumerate(hole, start=1):
                self.add_geometry_curve_row(
                    region=str(region_index),
                    index=str(curve_index),
                    curve_type=str(spec.get("type", spec.get("kind", "line"))),
                    role=f"hole_{hole_index}",
                    parameters=self._curve_parameters_yaml(spec),
                    segments=str(spec.get("segments", "")),
                )
    self.populate_curve_control_point_table()

def _add_curve_control_row(owner: Any, qt: Mapping[str, Any], *, curve_row: int, point: int, role: str, x: float, y: float, weight: Any = "", locked: Any = False) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_curve_control_table
    row = table.rowCount()
    table.insertRow(row)
    values = [curve_row, point, role, f"{float(x):.12g}", f"{float(y):.12g}", "" if weight is None else str(weight), str(locked)]
    for col, value in enumerate(values):
        table.setItem(row, col, QTableWidgetItem(str(value)))

def populate_curve_control_point_table(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_curve_control_table", None)
    curve_table = getattr(self, "geometry_curve_table", None)
    if table is None or curve_table is None:
        return
    table.setRowCount(0)
    for curve_row in range(curve_table.rowCount()):
        try:
            spec = self._curve_spec_from_table_row(curve_row)
        except Exception:
            continue
        controls = spec.get("control_points", spec.get("points"))
        weights = self._ensure_list(spec.get("weights", []))
        if isinstance(controls, list) and controls:
            for point_index, point in enumerate(controls, start=1):
                try:
                    x, y = self._xy_pair(point)
                except ValueError:
                    continue
                weight = weights[point_index - 1] if point_index - 1 < len(weights) else ""
                self._add_curve_control_row(curve_row=curve_row + 1, point=point_index, role="control", x=x, y=y, weight=weight, locked=False)
            continue
        for role in ("start", "end", "center"):
            if role not in spec:
                continue
            try:
                x, y = self._xy_pair(spec.get(role))
            except ValueError:
                continue
            weight = spec.get("radius", "") if role == "center" else ""
            self._add_curve_control_row(curve_row=curve_row + 1, point=1 if role == "start" else (2 if role == "end" else 0), role=role, x=x, y=y, weight=weight, locked=False)

def apply_curve_control_point_table(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    try:
        self._apply_curve_control_point_table_to_curve_rows()
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"control point edit failed: {exc}")
        return
    self.apply_curve_boolean_panel()

def _apply_curve_control_point_table_to_curve_rows(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_curve_control_table", None)
    curve_table = getattr(self, "geometry_curve_table", None)
    if table is None or curve_table is None or table.rowCount() == 0:
        return
    specs_by_row: dict[int, dict[str, Any]] = {}
    for row in range(table.rowCount()):
        curve_row = int(self._float_text(self._table_text(table, row, 0).strip() or "0", "curve_row")) - 1
        if curve_row < 0 or curve_row >= curve_table.rowCount():
            continue
        if self._bool_text(self._table_text(table, row, 6), False):
            continue
        spec = specs_by_row.setdefault(curve_row, self._curve_spec_from_table_row(curve_row))
        point_index = int(self._float_text(self._table_text(table, row, 1).strip() or "0", "point")) - 1
        role = self._table_text(table, row, 2).strip().lower() or "control"
        x = self._float_text(self._table_text(table, row, 3), "control x")
        y = self._float_text(self._table_text(table, row, 4), "control y")
        weight_text = self._table_text(table, row, 5).strip()
        if role == "control":
            key = "control_points" if "control_points" in spec or "points" not in spec else "points"
            points = [list(self._xy_pair(point)) for point in self._ensure_list(spec.get(key, []))]
            if point_index < 0:
                continue
            while len(points) <= point_index:
                points.append([0.0, 0.0])
            points[point_index] = [x, y]
            spec[key] = points
            if weight_text:
                weights = [float(value) for value in self._ensure_list(spec.get("weights", []))]
                while len(weights) <= point_index:
                    weights.append(1.0)
                weights[point_index] = self._float_text(weight_text, "control weight")
                spec["weights"] = weights
        elif role in {"start", "end", "center"}:
            spec[role] = [x, y]
            if role == "center" and weight_text:
                spec["radius"] = self._float_text(weight_text, "arc radius")
        else:
            spec[role] = [x, y]
    for curve_row, spec in specs_by_row.items():
        curve_table.setItem(curve_row, 4, QTableWidgetItem(self._curve_parameters_yaml(spec)))
        if "segments" in spec:
            curve_table.setItem(curve_row, 5, QTableWidgetItem(str(spec.get("segments", ""))))

def populate_geometry_boolean_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_boolean_table", None)
    if table is None:
        return
    table.setRowCount(0)
    graph = self._active_cad_boolean_graph()
    geometry = self._mapping(self.cfg.get("geometry", {}))
    mesh = self._mapping(self.cfg.get("mesh", {}))
    cad_boolean = self._mapping(geometry.get("cad_boolean", mesh.get("cad_boolean", {})))
    selected = str(cad_boolean.get("selected_operation", ""))
    manual_operation = str(cad_boolean.get("manual_selected_operation", ""))
    visible_operations = set(str(value) for value in self._ensure_list(cad_boolean.get("visible_operations", [])))
    operations = self._mapping(graph.get("boolean_operations", {})) if graph else {}
    if not operations:
        return
    try:
        from geofem_app.analytic_boolean import operation_loop_area
    except Exception:
        operation_loop_area = None
    for name, raw in operations.items():
        if not isinstance(raw, Mapping):
            continue
        row = table.rowCount()
        table.insertRow(row)
        area = ""
        if operation_loop_area is not None:
            try:
                area = f"{operation_loop_area(graph, operation=str(name), target=graph.get('target')):.6g}"
            except Exception:
                area = ""
        values = [
            str(name),
            str(raw.get("edge_count", "")),
            str(raw.get("loop_count", len(raw.get("loops", [])) if isinstance(raw.get("loops", []), list) else "")),
            area,
            "yes" if str(name) == selected else "",
            "yes" if str(name) == manual_operation else "",
            "yes" if not visible_operations or str(name) in visible_operations else "",
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col < 6:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, col, item)
    self._refresh_boolean_operation_combo(operations, selected)

def apply_curve_boolean_panel(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    try:
        geometry = dict(self._mapping(self.cfg.get("geometry", {})))
        regions = geometry.get("regions", [])
        if not isinstance(regions, list):
            regions = []
        self._apply_curve_control_point_table_to_curve_rows()
        geometry["regions"] = self._regions_with_curve_table_data(regions)
        expression = self.geometry_boolean_expression.text().strip()
        if expression:
            geometry["boolean_expression"] = expression
            self._mesh_cfg()["boolean_expression"] = expression
        else:
            geometry.pop("boolean_expression", None)
        cad_boolean = dict(self._mapping(geometry.get("cad_boolean", {})))
        selected_operation = self._combo_value(self.geometry_boolean_operation).strip()
        if selected_operation:
            cad_boolean["selected_operation"] = selected_operation
        if cad_boolean:
            geometry["cad_boolean"] = cad_boolean
        self.cfg["geometry"] = geometry
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"曲線/Boolean入力が不正です: {exc}")
        return
    self._after_form_change("解析曲線/Boolean表を反映しました")

def rebuild_geometry_boolean_graph(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    try:
        geometry = dict(self._mapping(self.cfg.get("geometry", {})))
        regions = geometry.get("regions", [])
        if not isinstance(regions, list):
            regions = []
        self._apply_curve_control_point_table_to_curve_rows()
        geometry["regions"] = self._regions_with_curve_table_data(regions)
        expression = self.geometry_boolean_expression.text().strip()
        if expression:
            geometry["boolean_expression"] = expression
            self._mesh_cfg()["boolean_expression"] = expression
        from geofem_app.analytic_boolean import build_analytic_curve_boolean_graph, classify_graph_boolean_operations
        from geofem_app.mesh_generation import _curve_chain_specs_from_region

        curve_regions = [_curve_chain_specs_from_region(raw) for raw in geometry["regions"] if isinstance(raw, Mapping)]
        curve_regions = [chain for chain in curve_regions if chain]
        if not curve_regions:
            raise ValueError("curve_boundaryを持つ領域がありません。")
        target = self._boolean_graph_target()
        graph = classify_graph_boolean_operations(
            build_analytic_curve_boolean_graph(curve_regions, target=target, tol=max(target * 1.0e-9, 1.0e-10)),
            expression=expression or None,
        )
        selected_operation = self._combo_value(self.geometry_boolean_operation).strip()
        if selected_operation == "expression" and "expression" not in self._mapping(graph.get("boolean_operations", {})):
            selected_operation = "union"
        if not selected_operation:
            selected_operation = "expression" if expression else "union"
        cad_boolean = dict(self._mapping(geometry.get("cad_boolean", {})))
        cad_boolean.update(
            {
                "engine": "analytic_curve_graph_winding_containment",
                "area_selection_engine": graph.get("selection_engine", "analytic_winding_containment"),
                "selected_operation": selected_operation,
                "boolean_expression": expression,
                "analytic_curve_graph": graph,
                "trim_edge_overrides": dict(self._mapping(geometry.get("cad_overlap_edge_states", {}))),
            }
        )
        geometry["cad_boolean"] = cad_boolean
        self.cfg["geometry"] = geometry
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"Boolean graphを作成できません: {exc}")
        return
    self._syncing_yaml = True
    self.yaml_editor.setPlainText(yaml.safe_dump(self.cfg, allow_unicode=True, sort_keys=False))
    self._syncing_yaml = False
    self.populate_geometry_boolean_table()
    self.refresh_cad_overlap_edges()
    self.update_preview()
    self.run_model_check()
    self.append_log("[GUI] 解析曲線Boolean graphを更新しました")

def _regions_with_curve_table_data(owner: Any, qt: Mapping[str, Any], regions: list[Any]) -> list[Any]:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_curve_table", None)
    if table is None or table.rowCount() == 0:
        return regions
    out: list[Any] = [dict(raw) if isinstance(raw, Mapping) else raw for raw in regions]
    grouped: dict[int, dict[str, list[tuple[int, dict[str, Any]]]]] = {}
    for row in range(table.rowCount()):
        region_text = self._table_text(table, row, 0).strip() or "1"
        index_text = self._table_text(table, row, 1).strip() or str(row + 1)
        try:
            region_index = max(1, int(float(region_text)))
            curve_index = max(1, int(float(index_text)))
        except ValueError as exc:
            raise ValueError(f"曲線表 {row + 1}: region/indexは数値で指定してください。") from exc
        role = self._table_text(table, row, 3).strip().lower() or "boundary"
        spec = self._curve_spec_from_table_row(row)
        grouped.setdefault(region_index, {}).setdefault(role, []).append((curve_index, spec))
    while len(out) < max(grouped):
        out.append({"id": f"region_{len(out) + 1}", "points": []})
    for region_index, roles in grouped.items():
        region = dict(self._mapping(out[region_index - 1]))
        boundary = [spec for _idx, spec in sorted(roles.get("boundary", []), key=lambda item: item[0])]
        if boundary:
            region["curve_boundary"] = boundary
            sampled = self._sample_curve_chain_points(boundary)
            if len(sampled) >= 3:
                region["points"] = sampled
        holes: list[list[dict[str, Any]]] = []
        for role, items in sorted(roles.items()):
            if not role.startswith("hole"):
                continue
            holes.append([spec for _idx, spec in sorted(items, key=lambda item: item[0])])
        if holes:
            region["curve_holes"] = holes
        elif "curve_holes" in region:
            region.pop("curve_holes", None)
        out[region_index - 1] = region
    return out

def _curve_spec_from_table_row(owner: Any, qt: Mapping[str, Any], row: int) -> dict[str, Any]:
    self = owner
    _bind_qt(qt)
    curve_type = self._table_text(self.geometry_curve_table, row, 2).strip().lower() or "line"
    params = self._yaml_mapping_text(self._table_text(self.geometry_curve_table, row, 4), "curve parameters YAML")
    spec = dict(params)
    spec["type"] = curve_type
    segments = self._table_text(self.geometry_curve_table, row, 5).strip()
    if segments:
        spec["segments"] = int(self._float_text(segments, "curve segments"))
    return spec

def _curve_boundary_specs(owner: Any, qt: Mapping[str, Any], region: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    self = owner
    _bind_qt(qt)
    for key in ("curve_boundary", "boundary", "segments", "curves", "edges"):
        value = region.get(key)
        if isinstance(value, Mapping) and self._curve_mapping_has_parameters(value):
            return [value]
        if isinstance(value, list):
            specs = [item for item in value if isinstance(item, Mapping) and self._curve_mapping_has_parameters(item)]
            if specs:
                return specs
    return []

def _curve_hole_specs(owner: Any, qt: Mapping[str, Any], region: Mapping[str, Any]) -> list[list[Mapping[str, Any]]]:
    self = owner
    _bind_qt(qt)
    raw_holes = region.get("curve_holes", region.get("holes", region.get("islands", [])))
    chains: list[list[Mapping[str, Any]]] = []
    if not isinstance(raw_holes, list):
        return chains
    for raw in raw_holes:
        if isinstance(raw, Mapping) and self._curve_mapping_has_parameters(raw):
            chains.append([raw])
        elif isinstance(raw, list):
            specs = [item for item in raw if isinstance(item, Mapping) and self._curve_mapping_has_parameters(item)]
            if specs:
                chains.append(specs)
    return chains

def _curve_parameters_yaml(owner: Any, qt: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    self = owner
    _bind_qt(qt)
    params = {str(key): value for key, value in spec.items() if str(key) not in {"type", "kind", "curve_type", "segments"}}
    return yaml.safe_dump(params, allow_unicode=True, sort_keys=False).strip() if params else ""

def _default_curve_parameters(owner: Any, qt: Mapping[str, Any], curve_type: str) -> dict[str, Any]:
    self = owner
    _bind_qt(qt)
    if curve_type in {"arc", "circle"}:
        return {"center": [0.0, 0.0], "radius": 1.0, "start_angle": 0.0, "end_angle": 90.0, "closed": curve_type == "circle"}
    if curve_type == "bezier":
        return {"control_points": [[0.0, 0.0], [0.5, 0.8], [1.0, 0.0]]}
    if curve_type in {"nurbs", "nurbs_curve"}:
        return {
            "control_points": [[0.0, 0.0], [0.4, 0.8], [0.8, 0.8], [1.2, 0.0]],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "knots": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "degree": 3,
        }
    return {"start": [0.0, 0.0], "end": [1.0, 0.0]}

def _sample_curve_chain_points(owner: Any, qt: Mapping[str, Any], specs: list[Mapping[str, Any]]) -> list[list[float]]:
    self = owner
    _bind_qt(qt)
    points: list[list[float]] = []
    for spec in specs:
        for x, y in self._sample_curve_spec(spec, target=self._boolean_graph_target()):
            if points and math.hypot(points[-1][0] - x, points[-1][1] - y) <= 1.0e-9:
                continue
            points.append([float(x), float(y)])
    if len(points) > 2 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= 1.0e-9:
        points.pop()
    return points

def _sample_curve_spec(owner: Any, qt: Mapping[str, Any], spec: Mapping[str, Any], *, target: float) -> list[tuple[float, float]]:
    self = owner
    _bind_qt(qt)
    try:
        from geofem_app.analytic_boolean import eval_curve
        from geofem_app.mesh_generation import _normalize_curve_spec

        normalized = _normalize_curve_spec(spec)
        if normalized is None:
            return []
        segments = int(spec.get("segments", max(8, min(64, int(math.ceil(1.0 / max(target, 1.0e-6))) * 4))))
        segments = max(2, min(128, segments))
        return [eval_curve(normalized, i / segments) for i in range(segments + 1)]
    except Exception:
        try:
            return [self._xy_pair(spec.get("start", [0.0, 0.0])), self._xy_pair(spec.get("end", [0.0, 0.0]))]
        except ValueError:
            return []

def _boolean_graph_target(owner: Any, qt: Mapping[str, Any]) -> float:
    self = owner
    _bind_qt(qt)
    mesh = self._mapping(self.cfg.get("mesh", {}))
    for key in ("target_size", "division_width"):
        try:
            value = float(mesh.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    try:
        return max(abs(float(self.mesh_x1.text()) - float(self.mesh_x0.text())) / max(int(float(self.mesh_nx.text())), 1), 1.0e-6)
    except Exception:
        return 1.0

def _active_cad_boolean_graph(owner: Any, qt: Mapping[str, Any]) -> Mapping[str, Any]:
    self = owner
    _bind_qt(qt)
    geometry = self._mapping(self.cfg.get("geometry", {}))
    mesh = self._mapping(self.cfg.get("mesh", {}))
    for source in (
        self._mapping(geometry.get("cad_boolean", {})).get("analytic_curve_graph"),
        geometry.get("analytic_curve_graph"),
        self._mapping(mesh.get("cad_boolean", {})).get("analytic_curve_graph"),
    ):
        if isinstance(source, Mapping) and isinstance(source.get("edges"), list):
            return source
    return {}

def _refresh_boolean_operation_combo(owner: Any, qt: Mapping[str, Any], operations: Mapping[str, Any], selected: str) -> None:
    self = owner
    _bind_qt(qt)
    combo = getattr(self, "geometry_boolean_operation", None)
    if combo is None:
        return
    current = selected or str(combo.currentText() or "")
    combo.blockSignals(True)
    combo.clear()
    for name in operations:
        combo.addItem(str(name))
    if not operations:
        combo.addItems(["union", "intersection", "expression"])
    if current:
        self._set_combo(combo, current)
    combo.blockSignals(False)

def apply_geometry_panel(owner: Any, qt: Mapping[str, Any], *_args: Any, after_change: bool = True) -> bool:
    self = owner
    _bind_qt(qt)
    geometry = dict(self._mapping(self.cfg.get("geometry", {})))
    lines: list[dict[str, Any]] = []
    tunnels: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    overlap_states: dict[str, str] = {}
    existing_lines_by_id = {
        str(raw.get("id")): dict(raw)
        for raw in geometry.get("lines", [])
        if isinstance(raw, Mapping) and raw.get("id") is not None
    }
    existing_tunnels_by_id = {
        str(raw.get("id")): dict(raw)
        for raw in geometry.get("tunnels", [])
        if isinstance(raw, Mapping) and raw.get("id") is not None
    }
    try:
        for row in range(self.geometry_layer_table.rowCount()):
            name = self._table_text(self.geometry_layer_table, row, 2).strip()
            if not name:
                continue
            layer: dict[str, Any] = {
                "name": name,
                "visible": self._bool_text(self._table_text(self.geometry_layer_table, row, 0), True),
                "locked": self._bool_text(self._table_text(self.geometry_layer_table, row, 1), False),
            }
            color = self._table_text(self.geometry_layer_table, row, 3).strip()
            linetype = self._table_text(self.geometry_layer_table, row, 4).strip()
            lineweight = self._table_text(self.geometry_layer_table, row, 5).strip()
            opacity = self._table_text(self.geometry_layer_table, row, 6).strip()
            source = self._table_text(self.geometry_layer_table, row, 7).strip()
            if color:
                layer["color"] = color
            if linetype:
                layer["linetype"] = linetype
            if lineweight:
                layer["lineweight"] = self._float_text(lineweight, "layer lineweight")
            if opacity:
                layer["opacity"] = self._float_text(opacity, "layer opacity")
            if source:
                layer["source"] = source
            layers.append(layer)
        for row in range(self.geometry_line_table.rowCount()):
            line_id = self._table_text(self.geometry_line_table, row, 0).strip() or f"line_{row + 1}"
            purpose = self._table_text(self.geometry_line_table, row, 1).strip() or "model"
            layer = self._table_text(self.geometry_line_table, row, 2).strip()
            x1 = self._float_text(self._table_text(self.geometry_line_table, row, 3), "x1")
            y1 = self._float_text(self._table_text(self.geometry_line_table, row, 4), "y1")
            x2 = self._float_text(self._table_text(self.geometry_line_table, row, 5), "x2")
            y2 = self._float_text(self._table_text(self.geometry_line_table, row, 6), "y2")
            line = dict(existing_lines_by_id.get(line_id, {}))
            line.update({"id": line_id, "purpose": purpose, "start": [x1, y1], "end": [x2, y2]})
            if layer:
                line["layer"] = layer
            else:
                line.pop("layer", None)
            lines.append(line)
        for row in range(self.geometry_tunnel_table.rowCount()):
            tunnel_id = self._table_text(self.geometry_tunnel_table, row, 0).strip() or f"tunnel_{row + 1}"
            layer = self._table_text(self.geometry_tunnel_table, row, 1).strip()
            cx = self._float_text(self._table_text(self.geometry_tunnel_table, row, 2), "cx")
            cy = self._float_text(self._table_text(self.geometry_tunnel_table, row, 3), "cy")
            radius = self._float_text(self._table_text(self.geometry_tunnel_table, row, 4), "radius")
            segments = int(self._float_text(self._table_text(self.geometry_tunnel_table, row, 5), "segments"))
            if radius <= 0.0:
                raise ValueError("tunnel radius must be positive")
            tunnel = dict(existing_tunnels_by_id.get(tunnel_id, {}))
            tunnel.update({"id": tunnel_id, "center": [cx, cy], "radius": radius, "segments": max(8, segments)})
            if layer:
                tunnel["layer"] = layer
            else:
                tunnel.pop("layer", None)
            tunnels.append(tunnel)
        for row in range(self.geometry_annotation_table.rowCount()):
            annotation_id = self._table_text(self.geometry_annotation_table, row, 0).strip() or f"note_{row + 1}"
            layer = self._table_text(self.geometry_annotation_table, row, 1).strip()
            x = self._float_text(self._table_text(self.geometry_annotation_table, row, 2), "annotation x")
            y = self._float_text(self._table_text(self.geometry_annotation_table, row, 3), "annotation y")
            text = self._table_text(self.geometry_annotation_table, row, 4)
            if not text.strip():
                continue
            annotation = {"id": annotation_id, "point": [x, y], "text": text}
            if layer:
                annotation["layer"] = layer
            annotations.append(annotation)
        for row in range(self.geometry_dimension_table.rowCount()):
            dimension_id = self._table_text(self.geometry_dimension_table, row, 0).strip() or f"dim_{row + 1}"
            layer = self._table_text(self.geometry_dimension_table, row, 1).strip()
            x1 = self._float_text(self._table_text(self.geometry_dimension_table, row, 2), "dimension x1")
            y1 = self._float_text(self._table_text(self.geometry_dimension_table, row, 3), "dimension y1")
            x2 = self._float_text(self._table_text(self.geometry_dimension_table, row, 4), "dimension x2")
            y2 = self._float_text(self._table_text(self.geometry_dimension_table, row, 5), "dimension y2")
            text = self._table_text(self.geometry_dimension_table, row, 6).strip()
            constraint = self._table_text(self.geometry_dimension_table, row, 7).strip()
            locked = self._bool_text(self._table_text(self.geometry_dimension_table, row, 8), False)
            dimension = {"id": dimension_id, "start": [x1, y1], "end": [x2, y2]}
            if layer:
                dimension["layer"] = layer
            if text:
                dimension["text"] = text
            if constraint:
                dimension["constraint"] = constraint
            if locked:
                dimension["locked"] = True
            dimensions.append(dimension)
        overlap_table = getattr(self, "geometry_overlap_table", None)
        if overlap_table is not None:
            for row in range(overlap_table.rowCount()):
                edge_id = self._table_text(overlap_table, row, 0).strip()
                state = self._table_text(overlap_table, row, 5).strip()
                if edge_id and state:
                    overlap_states[edge_id] = state
        regions = yaml.safe_load(self.geometry_regions_editor.toPlainText()) if self.geometry_regions_editor.toPlainText().strip() else []
        if regions is None:
            regions = []
        if not isinstance(regions, list):
            raise ValueError("regions YAMLはリストで入力してください。")
        if getattr(self, "geometry_curve_table", None) is not None and self.geometry_curve_table.rowCount() > 0:
            self._apply_curve_control_point_table_to_curve_rows()
            regions = self._regions_with_curve_table_data(regions)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"形状表の入力が不正です: {exc}")
        return False
    geometry["lines"] = lines
    geometry["tunnels"] = tunnels
    geometry["regions"] = regions
    geometry["layers"] = layers
    geometry["annotations"] = annotations
    geometry["dimensions"] = dimensions
    if overlap_states:
        geometry["cad_overlap_edge_states"] = overlap_states
        cad_boolean = dict(self._mapping(geometry.get("cad_boolean", {})))
        cad_boolean["trim_edge_overrides"] = dict(overlap_states)
        geometry["cad_boolean"] = cad_boolean
    else:
        geometry.pop("cad_overlap_edge_states", None)
    expression = self.geometry_boolean_expression.text().strip() if hasattr(self, "geometry_boolean_expression") else ""
    if expression:
        geometry["boolean_expression"] = expression
        self._mesh_cfg()["boolean_expression"] = expression
    else:
        geometry.pop("boolean_expression", None)
    self.cfg["geometry"] = geometry
    if not after_change:
        return True
    self._after_form_change("形状/CAD表を反映しました")

    return True

def refresh_cad_overlap_edges(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_overlap_table", None)
    if table is None:
        return
    table.setRowCount(0)
    geometry = self._geometry_cfg()
    cad_boolean = self._mapping(geometry.get("cad_boolean", {}))
    states = dict(self._mapping(cad_boolean.get("trim_edge_overrides", {})))
    states.update(dict(self._mapping(geometry.get("cad_overlap_edge_states", {}))))
    for edge, _graph in self._iter_cad_overlap_edges():
        row = table.rowCount()
        table.insertRow(row)
        edge_id = str(edge.get("id", row + 1))
        values = [
            edge_id,
            str(edge.get("curve_id", edge.get("curve", ""))),
            ",".join(str(item) for item in self._ensure_list(edge.get("coincident_edge_ids", []))),
            str(edge.get("region", "")),
            str(edge.get("overlap_role", edge.get("boolean_role", ""))),
            str(states.get(edge_id, edge.get("ui_state", "active"))),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if col < 5:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, col, item)

def set_selected_cad_overlap_state(owner: Any, qt: Mapping[str, Any], state: str) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_overlap_table
    rows = sorted({index.row() for index in table.selectedIndexes()})
    if not rows:
        return
    geometry = self._geometry_cfg()
    states = geometry.setdefault("cad_overlap_edge_states", {})
    if not isinstance(states, dict):
        states = {}
        geometry["cad_overlap_edge_states"] = states
    cad_boolean = geometry.setdefault("cad_boolean", {})
    if not isinstance(cad_boolean, dict):
        cad_boolean = {}
        geometry["cad_boolean"] = cad_boolean
    trim_overrides = cad_boolean.setdefault("trim_edge_overrides", {})
    if not isinstance(trim_overrides, dict):
        trim_overrides = {}
        cad_boolean["trim_edge_overrides"] = trim_overrides
    for row in rows:
        edge_id = self._table_text(table, row, 0).strip()
        if edge_id:
            states[edge_id] = state
            trim_overrides[edge_id] = state
            table.setItem(row, 5, QTableWidgetItem(state))
    self._after_form_change(f"重複edgeを{state}にしました")

def repair_selected_cad_overlap_edges(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    table = self.geometry_overlap_table
    rows = sorted({index.row() for index in table.selectedIndexes()})
    if not rows:
        rows = list(range(table.rowCount()))
    if not rows:
        return
    geometry = self._geometry_cfg()
    cad_boolean = geometry.setdefault("cad_boolean", {})
    if not isinstance(cad_boolean, dict):
        cad_boolean = {}
        geometry["cad_boolean"] = cad_boolean
    repairs = cad_boolean.setdefault("manual_edge_repairs", {})
    if not isinstance(repairs, dict):
        repairs = {}
        cad_boolean["manual_edge_repairs"] = repairs
    states = geometry.setdefault("cad_overlap_edge_states", {})
    if not isinstance(states, dict):
        states = {}
        geometry["cad_overlap_edge_states"] = states
    trim_overrides = cad_boolean.setdefault("trim_edge_overrides", {})
    if not isinstance(trim_overrides, dict):
        trim_overrides = {}
        cad_boolean["trim_edge_overrides"] = trim_overrides
    for row in rows:
        edge_id = self._table_text(table, row, 0).strip()
        if not edge_id:
            continue
        state = "repaired_prefer_primary"
        states[edge_id] = state
        trim_overrides[edge_id] = state
        repairs[edge_id] = {
            "action": "prefer_primary_trim",
            "coincident": self._table_text(table, row, 2).strip(),
            "region": self._table_text(table, row, 3).strip(),
            "role": self._table_text(table, row, 4).strip(),
            "source": "gui_overlap_repair",
        }
        table.setItem(row, 5, QTableWidgetItem(state))
    self._after_form_change("CAD overlap edges marked for manual trim repair")

def _selected_boolean_operation_name(owner: Any, qt: Mapping[str, Any]) -> str:
    self = owner
    _bind_qt(qt)
    table = getattr(self, "geometry_boolean_table", None)
    if table is not None:
        rows = sorted({index.row() for index in table.selectedIndexes()})
        if rows:
            value = self._table_text(table, rows[0], 0).strip()
            if value:
                return value
    combo = getattr(self, "geometry_boolean_operation", None)
    return combo.currentText().strip() if combo is not None else ""

def select_cad_boolean_operation_from_table(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    operation = self._selected_boolean_operation_name()
    if not operation:
        return
    geometry = self._geometry_cfg()
    cad_boolean = geometry.setdefault("cad_boolean", {})
    if not isinstance(cad_boolean, dict):
        cad_boolean = {}
        geometry["cad_boolean"] = cad_boolean
    cad_boolean["selected_operation"] = operation
    mesh = self._mesh_cfg()
    mesh_cad = mesh.setdefault("cad_boolean", {})
    if isinstance(mesh_cad, dict):
        mesh_cad["selected_operation"] = operation
    if hasattr(self, "geometry_boolean_operation"):
        self._set_combo(self.geometry_boolean_operation, operation)
    self.populate_geometry_boolean_table()
    self.update_preview()
    self._after_form_change(f"CAD Boolean operation selected: {operation}")

def store_selected_boolean_operation_as_manual_repair(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    graph = self._active_cad_boolean_graph()
    operation = self._selected_boolean_operation_name()
    operations = self._mapping(graph.get("boolean_operations", {})) if graph else {}
    operation_data = self._mapping(operations.get(operation, {}))
    if not operation or not operation_data:
        return
    geometry = self._geometry_cfg()
    cad_boolean = geometry.setdefault("cad_boolean", {})
    if not isinstance(cad_boolean, dict):
        cad_boolean = {}
        geometry["cad_boolean"] = cad_boolean
    manual = cad_boolean.setdefault("manual_loop_overrides", {})
    if not isinstance(manual, dict):
        manual = {}
        cad_boolean["manual_loop_overrides"] = manual
    loops = [dict(loop) for loop in self._ensure_list(operation_data.get("loops", [])) if isinstance(loop, Mapping)]
    manual[operation] = {"operation": operation, "loops": loops, "source": "gui_boolean_table", "edge_count": operation_data.get("edge_count", 0)}
    cad_boolean["manual_selected_operation"] = operation
    cad_boolean["selected_operation"] = operation
    cad_boolean["visible_operations"] = [operation]
    self.populate_geometry_boolean_table()
    self.update_preview()
    self._after_form_change(f"CAD Boolean manual loop override stored: {operation}")

def clear_manual_boolean_repair(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    _bind_qt(qt)
    geometry = self._geometry_cfg()
    cad_boolean = geometry.get("cad_boolean", {})
    if isinstance(cad_boolean, dict):
        cad_boolean.pop("manual_loop_overrides", None)
        cad_boolean.pop("manual_selected_operation", None)
        cad_boolean.pop("visible_operations", None)
    self.populate_geometry_boolean_table()
    self.update_preview()
    self._after_form_change("CAD Boolean manual loop overrides cleared")

def _iter_cad_overlap_edges(owner: Any, qt: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    self = owner
    _bind_qt(qt)
    out: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for graph in self._cad_boolean_graphs():
        edges = graph.get("edges", [])
        if not isinstance(edges, list):
            continue
        for edge in edges:
            if isinstance(edge, Mapping) and (edge.get("coincident_edge_ids") or edge.get("overlap_span_ids") or edge.get("overlap_role")):
                out.append((edge, graph))
    return out

def _cad_boolean_graphs(owner: Any, qt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    self = owner
    _bind_qt(qt)
    graphs: list[Mapping[str, Any]] = []

    def add_graph(value: Any) -> None:
        if isinstance(value, Mapping) and isinstance(value.get("edges"), list):
            graphs.append(value)

    geometry = self._mapping(self.cfg.get("geometry", {}))
    mesh = self._mapping(self.cfg.get("mesh", {}))
    add_graph(geometry.get("analytic_curve_graph"))
    add_graph(self._mapping(geometry.get("cad_boolean", {})).get("analytic_curve_graph"))
    add_graph(self._mapping(mesh.get("cad_boolean", {})).get("analytic_curve_graph"))
    for source in (geometry.get("boolean_operations"), mesh.get("boolean_operations")):
        if isinstance(source, Mapping):
            add_graph(source.get("analytic_curve_graph"))
    return graphs


__all__ = [
    "GEOMETRY_CONTROLLER_METHODS",
    "geometry_controller_contract",
    *GEOMETRY_CONTROLLER_METHODS,
]
