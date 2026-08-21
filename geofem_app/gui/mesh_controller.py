"""Mesh GUI operation controller functions split from MainWindow.

The MainWindow remains the owner of Qt widgets, project state, background job
controller, and notifications.  This module owns mesh-panel synchronization,
mesh-control tables, quality-repair workflows, and element-library operations.
"""

from __future__ import annotations

import copy
from datetime import datetime
import math
from typing import Any, Mapping

import yaml

from geofem_app.gui.mesh_worker import collect_mesh_quality_violations_snapshot, compare_mesh_quality_improvements_snapshot


MESH_CONTROLLER_METHODS = (
    "apply_mesh_panel",
    "add_mesh_refinement_row",
    "add_mesh_control_point_row",
    "add_mesh_split_line_row",
    "add_mesh_size_map_row",
    "add_mesh_block_row",
    "remove_selected_mesh_control_rows",
    "add_selected_point_refinement",
    "add_selected_mesh_split_line",
    "split_selected_mesh_block",
    "populate_mesh_control_tables",
    "_mesh_control_data_from_tables",
    "apply_mesh_controls_panel",
    "populate_mesh_quality_violation_table",
    "populate_mesh_quality_violation_table_async",
    "_mesh_quality_violations_finished",
    "_apply_mesh_quality_violations",
    "_selected_quality_violation_ids",
    "select_mesh_quality_violations",
    "repair_selected_mesh_quality_violations",
    "compare_mesh_quality_improvements",
    "compare_mesh_quality_improvements_async",
    "_mesh_quality_improvements_finished",
    "_mesh_quality_failed",
    "_apply_mesh_quality_improvement_candidates",
    "_append_mesh_quality_improvement_row",
    "apply_selected_mesh_quality_improvement",
    "_selected_mesh_quality_improvement_methods",
    "_mesh_dict_from_fem_mesh",
    "_format_float",
    "add_element_library_preset",
    "add_element_library_row",
    "remove_selected_element_library_rows",
    "populate_element_library_table",
    "apply_element_library_panel",
)


def mesh_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.mesh_controller.v1",
        "method_count": len(MESH_CONTROLLER_METHODS),
        "methods": list(MESH_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner mesh widgets/config and dispatches mesh-quality background jobs; MainWindow delegates mesh input-domain actions",
        "covered_surfaces": ["mesh_panel", "mesh_control_tables", "mesh_quality", "element_library"],
    }


def apply_mesh_panel(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    try:
        existing = dict(self._mapping(self.cfg.get("mesh", {})))
        mesh = {
            "generator": "rectangle",
            "x_range": [float(self.mesh_x0.text()), float(self.mesh_x1.text())],
            "y_range": [float(self.mesh_y0.text()), float(self.mesh_y1.text())],
            "nx": int(float(self.mesh_nx.text())),
            "ny": int(float(self.mesh_ny.text())),
            "element_type": self.mesh_type.currentText(),
            "integration": self.mesh_integration.currentText(),
            "material": self.mesh_material.text().strip() or "soil",
        }
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"メッシュ入力が不正です: {exc}")
        return
    boolean_expression = self.mesh_boolean_expression.text().strip()
    if boolean_expression:
        mesh["boolean_expression"] = boolean_expression
    try:
        mesh.update(self._mesh_control_data_from_tables())
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"メッシュ制御入力が不正です: {exc}")
        return
    for key in (
        "mode",
        "division_width",
        "target_size",
        "node_sets",
        "element_sets",
        "auto_mesh_confirmed_at",
        "cad_boolean",
        "mesh_quality",
        "quality_violations",
        "quality_repairs",
    ):
        if key in existing:
            mesh[key] = existing[key]
    self.cfg["mesh"] = mesh
    if hasattr(self, "mark_mesh_rebuilt_for_current_geometry"):
        self.mark_mesh_rebuilt_for_current_geometry()
    self._after_form_change("メッシュを反映しました")


def add_mesh_refinement_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.mesh_refinement_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"id": f"refine_{row + 1}", "cx": "0.0", "cy": "0.0", "radius": "1.0", "factor": "2.0"}
    defaults.update(values)
    for col, key in enumerate(["id", "cx", "cy", "radius", "factor"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_mesh_control_point_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.mesh_control_point_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"id": f"cp_{row + 1}", "x": "0.0", "y": "0.0", "target_size": "1.0", "tag": ""}
    defaults.update(values)
    for col, key in enumerate(["id", "x", "y", "target_size", "tag"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_mesh_split_line_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.mesh_split_line_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"id": f"split_{row + 1}", "x1": "0.0", "y1": "0.0", "x2": "1.0", "y2": "0.0", "target_size": "", "locked": True}
    defaults.update(values)
    for col, key in enumerate(["id", "x1", "y1", "x2", "y2", "target_size", "locked"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_mesh_size_map_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.mesh_size_map_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"id": f"size_{row + 1}", "x": "0.0", "y": "0.0", "radius": "1.0", "target_size": "0.5", "grading": "1.4"}
    defaults.update(values)
    for col, key in enumerate(["id", "x", "y", "radius", "target_size", "grading"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_mesh_block_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.mesh_block_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"id": f"block_{row + 1}", "name": f"block_{row + 1}", "element_set": "", "active": True, "split_hint": "", "extra": ""}
    defaults.update(values)
    extra = defaults.get("extra", "")
    if isinstance(extra, Mapping):
        extra = yaml.safe_dump(dict(extra), allow_unicode=True, sort_keys=False).strip()
    for col, key in enumerate(["id", "name", "element_set", "active", "split_hint", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(extra if key == "extra" else defaults.get(key, ""))))


def remove_selected_mesh_control_rows(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    for table in (self.mesh_refinement_table, self.mesh_control_point_table, self.mesh_split_line_table, self.mesh_size_map_table, self.mesh_block_table, self.mesh_quality_violation_table):
        if table.selectedIndexes():
            self.remove_selected_rows(table)


def add_selected_point_refinement(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    selected = self._selected_model_points()
    if selected:
        x, y = selected[0]
    else:
        bbox = self._geometry_bbox(self._mapping(self.cfg.get("geometry", {})))
        if bbox is None:
            QMessageBox.information(self, "GeoFEM", "モデル点または形状を選択してください。")
            return
        x = (bbox[0] + bbox[1]) * 0.5
        y = (bbox[2] + bbox[3]) * 0.5
    self.add_mesh_refinement_row(cx=f"{x:g}", cy=f"{y:g}", radius="1.0", factor="2.0")


def add_selected_mesh_split_line(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    points = self._selected_model_points()
    if len(points) < 2:
        edge = self._selected_mesh_edge_points()
        if edge is not None:
            points = [edge[0], edge[1]]
    if len(points) < 2:
        bbox = self._geometry_bbox(self._mapping(self.cfg.get("geometry", {})))
        if bbox is None:
            QMessageBox.information(self, "GeoFEM", "split lineに使う節点/辺/形状を選択してください。")
            return
        x0, x1, y0, y1 = bbox
        xm = (x0 + x1) * 0.5
        points = [(xm, y0), (xm, y1)]
    (x1, y1), (x2, y2) = points[0], points[1]
    self.add_mesh_split_line_row(x1=f"{x1:g}", y1=f"{y1:g}", x2=f"{x2:g}", y2=f"{y2:g}", target_size="", locked=True)


def split_selected_mesh_block(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"mesh block split failed: {exc}")
        return
    selected = self._selected_element_ids()
    if not selected:
        selected = [element.id for element in mesh.elements]
    coords: list[tuple[float, float]] = []
    for element in mesh.elements:
        if element.id not in selected:
            continue
        corner_count = 3 if element.type.startswith("TRI") else 4
        for nid in element.nodes[:corner_count]:
            idx = mesh.node_index[nid]
            coords.append((float(mesh.coords[idx, 0]), float(mesh.coords[idx, 1])))
    if not coords:
        return
    xs = [x for x, _y in coords]
    ys = [y for _x, y in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    vertical = (x1 - x0) >= (y1 - y0)
    if vertical:
        xm = (x0 + x1) * 0.5
        self.add_mesh_split_line_row(id=f"split_block_{self.mesh_split_line_table.rowCount() + 1}", x1=f"{xm:g}", y1=f"{y0:g}", x2=f"{xm:g}", y2=f"{y1:g}", target_size="", locked=True)
        split_hint = "vertical"
    else:
        ym = (y0 + y1) * 0.5
        self.add_mesh_split_line_row(id=f"split_block_{self.mesh_split_line_table.rowCount() + 1}", x1=f"{x0:g}", y1=f"{ym:g}", x2=f"{x1:g}", y2=f"{ym:g}", target_size="", locked=True)
        split_hint = "horizontal"
    block_id = f"block_{self.mesh_block_table.rowCount() + 1}"
    self.add_mesh_block_row(
        id=block_id,
        name=block_id,
        element_set=",".join(sorted(selected, key=self._natural_sort_key)),
        active=True,
        split_hint=split_hint,
        extra={"interactive_split": True, "bbox": [x0, x1, y0, y1]},
    )
    self.apply_mesh_controls_panel()


def populate_mesh_control_tables(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    if not hasattr(self, "mesh_refinement_table"):
        return
    mesh = self._mapping(self.cfg.get("mesh", {}))
    self.mesh_refinement_table.setRowCount(0)
    raw_refinements = mesh.get("refinements", mesh.get("local_refinements", []))
    if isinstance(raw_refinements, list):
        for index, raw in enumerate(raw_refinements, start=1):
            if not isinstance(raw, Mapping):
                continue
            try:
                cx, cy = self._xy_pair(raw.get("center", raw.get("point", [0.0, 0.0])))
            except ValueError:
                continue
            self.add_mesh_refinement_row(
                id=str(raw.get("id", f"refine_{index}")),
                cx=f"{cx:g}",
                cy=f"{cy:g}",
                radius=str(raw.get("radius", raw.get("r", ""))),
                factor=str(raw.get("factor", raw.get("refinement", ""))),
            )
    self.mesh_control_point_table.setRowCount(0)
    raw_points = mesh.get("control_points", mesh.get("mesh_control_points", []))
    if isinstance(raw_points, list):
        for index, raw in enumerate(raw_points, start=1):
            if not isinstance(raw, Mapping):
                continue
            try:
                x, y = self._xy_pair(raw.get("point", raw.get("position", [raw.get("x", 0.0), raw.get("y", 0.0)])))
            except ValueError:
                continue
            self.add_mesh_control_point_row(
                id=str(raw.get("id", f"cp_{index}")),
                x=f"{x:g}",
                y=f"{y:g}",
                target_size=str(raw.get("target_size", raw.get("size", ""))),
                tag=str(raw.get("tag", raw.get("name", ""))),
            )
    self.mesh_split_line_table.setRowCount(0)
    raw_split_lines = mesh.get("split_lines", mesh.get("split_line_constraints", []))
    if isinstance(raw_split_lines, list):
        for index, raw in enumerate(raw_split_lines, start=1):
            if not isinstance(raw, Mapping):
                continue
            try:
                start = raw.get("start", raw.get("p1", [raw.get("x1", 0.0), raw.get("y1", 0.0)]))
                end = raw.get("end", raw.get("p2", [raw.get("x2", 0.0), raw.get("y2", 0.0)]))
                x1, y1 = self._xy_pair(start)
                x2, y2 = self._xy_pair(end)
            except ValueError:
                continue
            target_size = raw.get("target_size", raw.get("size", ""))
            self.add_mesh_split_line_row(
                id=str(raw.get("id", f"split_{index}")),
                x1=f"{x1:g}",
                y1=f"{y1:g}",
                x2=f"{x2:g}",
                y2=f"{y2:g}",
                target_size="" if target_size is None else str(target_size),
                locked=raw.get("locked", True),
            )
    self.mesh_size_map_table.setRowCount(0)
    raw_size_map = mesh.get("size_map", mesh.get("local_size_map", []))
    if isinstance(raw_size_map, list):
        for index, raw in enumerate(raw_size_map, start=1):
            if not isinstance(raw, Mapping):
                continue
            try:
                x, y = self._xy_pair(raw.get("center", raw.get("point", [raw.get("x", 0.0), raw.get("y", 0.0)])))
            except ValueError:
                continue
            self.add_mesh_size_map_row(
                id=str(raw.get("id", f"size_{index}")),
                x=f"{x:g}",
                y=f"{y:g}",
                radius=str(raw.get("radius", raw.get("r", ""))),
                target_size=str(raw.get("target_size", raw.get("size", ""))),
                grading=str(raw.get("grading", raw.get("growth", ""))),
            )
    self.mesh_block_table.setRowCount(0)
    blocks = mesh.get("blocks", {})
    if isinstance(blocks, Mapping):
        for key, raw in blocks.items():
            block = dict(self._mapping(raw))
            elements = block.get("elements", block.get("element_set", ""))
            element_set = ",".join(str(item) for item in elements) if isinstance(elements, list) else str(elements or "")
            self.add_mesh_block_row(
                id=str(block.get("id", key)),
                name=str(block.get("name", key)),
                element_set=element_set,
                active=block.get("active", True),
                split_hint=str(block.get("split_hint", block.get("division", ""))),
                extra=self._element_extra_yaml(block, {"id", "name", "elements", "element_set", "active", "split_hint", "division"}),
            )
    elif isinstance(blocks, list):
        for index, raw in enumerate(blocks, start=1):
            if not isinstance(raw, Mapping):
                continue
            self.add_mesh_block_row(
                id=str(raw.get("id", f"block_{index}")),
                name=str(raw.get("name", raw.get("id", f"block_{index}"))),
                element_set=str(raw.get("element_set", "")),
                active=raw.get("active", True),
                split_hint=str(raw.get("split_hint", raw.get("division", ""))),
                extra=self._element_extra_yaml(raw, {"id", "name", "element_set", "active", "split_hint", "division"}),
            )
    if getattr(self, "_startup_model_refresh_deferred", False):
        self._mesh_quality_table_dirty = True
    else:
        self.populate_mesh_quality_violation_table()


def _mesh_control_data_from_tables(owner: Any, qt: Mapping[str, Any], *_args: Any) -> dict[str, Any]:
    self = owner
    data: dict[str, Any] = {}
    refinements: list[dict[str, Any]] = []
    for row in range(self.mesh_refinement_table.rowCount()):
        refine_id = self._table_text(self.mesh_refinement_table, row, 0).strip() or f"refine_{row + 1}"
        cx = self._float_text(self._table_text(self.mesh_refinement_table, row, 1), "refinement cx")
        cy = self._float_text(self._table_text(self.mesh_refinement_table, row, 2), "refinement cy")
        radius = self._float_text(self._table_text(self.mesh_refinement_table, row, 3), "refinement radius")
        factor = self._float_text(self._table_text(self.mesh_refinement_table, row, 4), "refinement factor")
        if radius <= 0.0 or factor <= 1.0:
            raise ValueError("局所細分はradius>0, factor>1で指定してください。")
        refinements.append({"id": refine_id, "center": [cx, cy], "radius": radius, "factor": factor})
    if refinements:
        data["refinements"] = refinements

    control_points: list[dict[str, Any]] = []
    for row in range(self.mesh_control_point_table.rowCount()):
        point_id = self._table_text(self.mesh_control_point_table, row, 0).strip() or f"cp_{row + 1}"
        x = self._float_text(self._table_text(self.mesh_control_point_table, row, 1), "control point x")
        y = self._float_text(self._table_text(self.mesh_control_point_table, row, 2), "control point y")
        size_text = self._table_text(self.mesh_control_point_table, row, 3).strip()
        tag = self._table_text(self.mesh_control_point_table, row, 4).strip()
        point: dict[str, Any] = {"id": point_id, "point": [x, y]}
        if size_text:
            target_size = self._float_text(size_text, "control point target_size")
            if target_size <= 0.0:
                raise ValueError("制御点target_sizeは正値で指定してください。")
            point["target_size"] = target_size
        if tag:
            point["tag"] = tag
        control_points.append(point)
    if control_points:
        data["control_points"] = control_points

    split_lines: list[dict[str, Any]] = []
    for row in range(self.mesh_split_line_table.rowCount()):
        split_id = self._table_text(self.mesh_split_line_table, row, 0).strip() or f"split_{row + 1}"
        x1 = self._float_text(self._table_text(self.mesh_split_line_table, row, 1), "split line x1")
        y1 = self._float_text(self._table_text(self.mesh_split_line_table, row, 2), "split line y1")
        x2 = self._float_text(self._table_text(self.mesh_split_line_table, row, 3), "split line x2")
        y2 = self._float_text(self._table_text(self.mesh_split_line_table, row, 4), "split line y2")
        if math.hypot(x2 - x1, y2 - y1) <= 1.0e-12:
            raise ValueError("split line length must be positive")
        target_text = self._table_text(self.mesh_split_line_table, row, 5).strip()
        split = {"id": split_id, "type": "split_line", "start": [x1, y1], "end": [x2, y2], "locked": self._bool_text(self._table_text(self.mesh_split_line_table, row, 6), True)}
        if target_text:
            target_size = self._float_text(target_text, "split line target_size")
            if target_size <= 0.0:
                raise ValueError("split line target_size must be positive")
            split["target_size"] = target_size
        split_lines.append(split)
    if split_lines:
        data["split_lines"] = split_lines

    size_map: list[dict[str, Any]] = []
    for row in range(self.mesh_size_map_table.rowCount()):
        size_id = self._table_text(self.mesh_size_map_table, row, 0).strip() or f"size_{row + 1}"
        x = self._float_text(self._table_text(self.mesh_size_map_table, row, 1), "size map x")
        y = self._float_text(self._table_text(self.mesh_size_map_table, row, 2), "size map y")
        radius = self._float_text(self._table_text(self.mesh_size_map_table, row, 3), "size map radius")
        target_size = self._float_text(self._table_text(self.mesh_size_map_table, row, 4), "size map target_size")
        grading = self._float_text(self._table_text(self.mesh_size_map_table, row, 5), "size map grading")
        if radius <= 0.0 or target_size <= 0.0 or grading < 1.0:
            raise ValueError("size map requires radius>0, target_size>0, grading>=1")
        size_map.append({"id": size_id, "center": [x, y], "radius": radius, "target_size": target_size, "grading": grading})
    if size_map:
        data["size_map"] = size_map

    blocks: dict[str, dict[str, Any]] = {}
    for row in range(self.mesh_block_table.rowCount()):
        block_id = self._table_text(self.mesh_block_table, row, 0).strip() or f"block_{row + 1}"
        name = self._table_text(self.mesh_block_table, row, 1).strip() or block_id
        element_set = self._table_text(self.mesh_block_table, row, 2).strip()
        active = self._bool_text(self._table_text(self.mesh_block_table, row, 3), True)
        split_hint = self._table_text(self.mesh_block_table, row, 4).strip()
        extra = self._yaml_mapping_text(self._table_text(self.mesh_block_table, row, 5), "mesh block extra YAML")
        block = dict(extra)
        block["id"] = block_id
        block["name"] = name
        block["active"] = active
        if element_set:
            parts = [part.strip() for part in element_set.replace(";", ",").split(",") if part.strip()]
            block["elements" if len(parts) != 1 or parts[0].lower() != "all" else "element_set"] = parts if len(parts) != 1 or parts[0].lower() != "all" else "all"
        if split_hint:
            block["split_hint"] = split_hint
        blocks[name] = block
    if blocks:
        data["blocks"] = blocks
    return data


def apply_mesh_controls_panel(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    try:
        data = self._mesh_control_data_from_tables()
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"メッシュ制御入力が不正です: {exc}")
        return
    mesh = dict(self._mapping(self.cfg.get("mesh", {})))
    for key in ("refinements", "control_points", "split_lines", "size_map", "blocks"):
        if key in data:
            mesh[key] = data[key]
        else:
            mesh.pop(key, None)
    self.cfg["mesh"] = mesh
    self._after_form_change("メッシュ制御を反映しました")


def populate_mesh_quality_violation_table(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    table = getattr(self, "mesh_quality_violation_table", None)
    if table is None:
        return
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
        violations = self._mesh_quality_violations(mesh)
    except Exception:
        violations = []
    self._apply_mesh_quality_violations(violations)


def populate_mesh_quality_violation_table_async(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QtCallableRunner = qt["QtCallableRunner"]
    if self._mesh_quality_job_id:
        self.statusBar().showMessage("メッシュ品質ジョブを実行中です。")
        return
    job_id = self.gui_jobs.start_job("mesh_quality", target=str(self.current_input or "current-config"), metadata={"operation": "violations"})
    self._mesh_quality_job_id = job_id
    cfg_snapshot = copy.deepcopy(self.cfg)
    runner = QtCallableRunner(job_id, lambda: collect_mesh_quality_violations_snapshot(cfg_snapshot))
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._mesh_quality_violations_finished)
    runner.signals.failed.connect(self._mesh_quality_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage("メッシュ品質違反をバックグラウンド抽出中...")


def _mesh_quality_violations_finished(owner: Any, qt: Mapping[str, Any], job_id: str, violations_obj: Any) -> None:
    self = owner
    if job_id != self._mesh_quality_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    violations = [dict(item) for item in list(violations_obj or []) if isinstance(item, Mapping)]
    self._complete_gui_worker_job(job_id, status="finished", message=f"{len(violations)} violations")
    self._mesh_quality_job_id = ""
    self._apply_mesh_quality_violations(violations)
    self.append_log(f"[GUI] メッシュ品質違反抽出完了: {len(violations)}件")


def _apply_mesh_quality_violations(owner: Any, qt: Mapping[str, Any], violations: list[Mapping[str, Any]]) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = getattr(self, "mesh_quality_violation_table", None)
    if table is None:
        return
    table.setRowCount(0)
    for row, item in enumerate(violations):
        table.insertRow(row)
        values = [
            item.get("element", ""),
            item.get("severity", ""),
            f"{float(item.get('area', 0.0)):.6g}",
            f"{float(item.get('min_angle', 0.0)):.6g}",
            f"{float(item.get('aspect', 0.0)):.6g}",
            f"{float(item.get('skew', 0.0)):.6g}",
            item.get("reason", ""),
            item.get("repair", ""),
        ]
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(str(value)))


def _selected_quality_violation_ids(owner: Any, qt: Mapping[str, Any], *_args: Any) -> list[str]:
    self = owner
    table = getattr(self, "mesh_quality_violation_table", None)
    if table is None:
        return []
    rows = sorted({item.row() for item in table.selectedItems()})
    if not rows:
        rows = list(range(table.rowCount()))
    ids: list[str] = []
    for row in rows:
        element_id = self._table_text(table, row, 0).strip()
        if element_id:
            ids.append(element_id)
    return sorted(set(ids), key=self._natural_sort_key)


def select_mesh_quality_violations(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    ids = set(self._selected_quality_violation_ids())
    self.scene.clearSelection()
    if not ids:
        return
    node_ids: set[str] = set()
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
        for element in mesh.elements:
            if str(element.id) in ids:
                node_ids.update(str(nid) for nid in element.nodes)
    except Exception:
        node_ids = set()
    for item in self.scene.items():
        data = item.data(0)
        if not isinstance(data, dict):
            continue
        element_id = str(data.get("element", data.get("id", "")))
        if data.get("kind") in {"element", "mesh_quality_violation"} and element_id in ids:
            item.setSelected(True)
        elif data.get("kind") == "node" and str(data.get("id", "")) in node_ids:
            item.setSelected(True)
    self.statusBar().showMessage(f"品質違反を選択しました: 要素 {len(ids)} / 節点 {len(node_ids)}")


def repair_selected_mesh_quality_violations(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    ids = self._selected_quality_violation_ids()
    if not ids:
        QMessageBox.information(self, "GeoFEM", "mesh quality violation rows are not available")
        return
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"mesh quality repair failed: {exc}")
        return
    mesh_cfg = self._mesh_cfg()
    base = self._mesh_base_target(mesh_cfg)
    refinements = self._list_value(mesh_cfg, "refinements")
    size_map = self._list_value(mesh_cfg, "size_map")
    control_points = self._list_value(mesh_cfg, "control_points")
    split_lines = self._list_value(mesh_cfg, "split_lines")
    repairs = self._list_value(mesh_cfg, "quality_repairs")
    element_lookup = {str(element.id): element for element in mesh.elements}
    added = 0

    def append_unique(target: list[Any], item: dict[str, Any]) -> bool:
        item_id = str(item.get("id", ""))
        if item_id and any(isinstance(raw, Mapping) and str(raw.get("id", "")) == item_id for raw in target):
            return False
        target.append(item)
        return True

    def split_line_for(coords: list[tuple[float, float]], element_id: str, target_size: float) -> dict[str, Any] | None:
        if len(coords) < 3:
            return None
        edges = [(index, math.hypot(coords[(index + 1) % len(coords)][0] - coords[index][0], coords[(index + 1) % len(coords)][1] - coords[index][1])) for index in range(len(coords))]
        if not edges:
            return None
        edge_index, length = max(edges, key=lambda item: item[1])
        if length <= max(target_size * 1.5, 1.0e-12):
            return None
        a = coords[edge_index]
        b = coords[(edge_index + 1) % len(coords)]
        start = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
        if len(coords) == 4:
            c = coords[(edge_index + 2) % 4]
            d = coords[(edge_index + 3) % 4]
            end = [(c[0] + d[0]) * 0.5, (c[1] + d[1]) * 0.5]
        else:
            c = coords[(edge_index + 2) % len(coords)]
            end = [c[0], c[1]]
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= 1.0e-12:
            return None
        return {
            "id": f"split_quality_{element_id}_{len(split_lines) + 1}",
            "type": "split_line",
            "start": [round(float(start[0]), 12), round(float(start[1]), 12)],
            "end": [round(float(end[0]), 12), round(float(end[1]), 12)],
            "target_size": float(target_size),
            "locked": True,
            "source": "quality_violation",
            "element": element_id,
        }

    for element_id in ids:
        element = element_lookup.get(str(element_id))
        if element is None:
            continue
        corner_count = 3 if element.type.startswith("TRI") else 4
        coords = [(float(mesh.coords[mesh.node_index[nid], 0]), float(mesh.coords[mesh.node_index[nid], 1])) for nid in element.nodes[:corner_count]]
        if len(coords) < 3:
            continue
        cx = sum(x for x, _y in coords) / len(coords)
        cy = sum(y for _x, y in coords) / len(coords)
        radius = max((math.hypot(x - cx, y - cy) for x, y in coords), default=base) * 1.25
        radius = max(radius, base * 0.75)
        target_size = max(base * 0.5, 1.0e-9)
        repair_id = f"quality_repair_{element_id}_{len(repairs) + 1}"
        if append_unique(refinements, {"id": repair_id, "center": [cx, cy], "radius": radius, "factor": 2.0, "source": "quality_violation", "element": element_id}):
            added += 1
        append_unique(size_map, {"id": f"size_{repair_id}", "center": [cx, cy], "radius": radius, "target_size": target_size, "grading": 1.35, "source": "quality_violation", "element": element_id})
        append_unique(
            control_points,
            {
                "id": f"cp_{repair_id}",
                "point": [cx, cy],
                "target_size": target_size,
                "radius": radius,
                "tag": "quality_violation",
                "source": "quality_violation",
                "element": element_id,
            },
        )
        split = split_line_for(coords, str(element_id), target_size)
        if split is not None:
            append_unique(split_lines, split)
        append_unique(
            repairs,
            {
                "id": repair_id,
                "element": element_id,
                "action": "local_refinement_and_point_redistribution",
                "center": [cx, cy],
                "radius": radius,
                "target_size": target_size,
            },
        )
    if added:
        mesh_cfg["requires_rebuild"] = True
        mesh_cfg["partial_rebuild_required"] = True
        mesh_cfg["stale_reason"] = "メッシュ品質修復制御: 再構成待ち"
        self._after_form_change(f"メッシュ品質修復制御を追加しました: {added} 箇所")
    else:
        QMessageBox.information(self, "GeoFEM", "no selected quality violations could be repaired")


def compare_mesh_quality_improvements(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    table = getattr(self, "mesh_quality_improvement_table", None)
    if table is None:
        return
    table.setRowCount(0)
    self.mesh_quality_improvement_candidates = []
    try:
        from geofem_app.fem2d import mesh_from_config
        from geofem_app.mesh_generation import improve_mesh_quality

        mesh_obj = mesh_from_config(self.cfg)
        nodes, elements = self._mesh_dict_from_fem_mesh(mesh_obj)
        min_area, min_angle, max_aspect, max_skew = self._mesh_quality_threshold_values()
        iterations = max(1, int(float(self.mesh_quality_iterations.text() or "5")))
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"mesh quality comparison failed: {exc}")
        return
    selected = self._selected_quality_violation_ids()
    if not selected:
        selected = self._selected_element_ids()
    methods = self._selected_mesh_quality_improvement_methods()
    for method in methods:
        try:
            new_nodes, new_elements, report = improve_mesh_quality(
                nodes,
                elements,
                method=method,
                min_area=min_area,
                min_angle_deg=min_angle,
                max_aspect_ratio=max_aspect,
                max_skew=max_skew,
                iterations=iterations,
                selected_elements=selected,
            )
        except Exception as exc:
            report = {"method": method, "status": str(exc), "before": {}, "after": {}, "before_violation_count": "", "after_violation_count": "", "changed_nodes": 0, "changed_elements": 0, "score_delta": -math.inf}
            new_nodes, new_elements = nodes, elements
        self.mesh_quality_improvement_candidates.append({"nodes": new_nodes, "elements": new_elements, "report": report})
        self._append_mesh_quality_improvement_row(report)
    if self.mesh_quality_improvement_candidates and table.rowCount() > 0:
        table.selectRow(0)
    self.statusBar().showMessage(f"mesh quality improvement candidates: {len(self.mesh_quality_improvement_candidates)}")


def compare_mesh_quality_improvements_async(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    QtCallableRunner = qt["QtCallableRunner"]
    table = getattr(self, "mesh_quality_improvement_table", None)
    if table is None:
        return
    if self._mesh_quality_job_id:
        self.statusBar().showMessage("メッシュ品質ジョブを実行中です。")
        return
    try:
        min_area, min_angle, max_aspect, max_skew = self._mesh_quality_threshold_values()
        iterations = max(1, int(float(self.mesh_quality_iterations.text() or "5")))
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"mesh quality comparison failed: {exc}")
        return
    selected = self._selected_quality_violation_ids()
    if not selected:
        selected = self._selected_element_ids()
    methods = self._selected_mesh_quality_improvement_methods()
    table.setRowCount(0)
    self.mesh_quality_improvement_candidates = []
    job_id = self.gui_jobs.start_job("mesh_quality", target=str(self.current_input or "current-config"), metadata={"operation": "improvements", "methods": methods})
    self._mesh_quality_job_id = job_id
    cfg_snapshot = copy.deepcopy(self.cfg)
    runner = QtCallableRunner(
        job_id,
        lambda: compare_mesh_quality_improvements_snapshot(
            cfg_snapshot,
            methods=methods,
            iterations=iterations,
            thresholds=(min_area, min_angle, max_aspect, max_skew),
            selected_elements=selected,
        ),
    )
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._mesh_quality_improvements_finished)
    runner.signals.failed.connect(self._mesh_quality_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage("メッシュ品質改善候補をバックグラウンド比較中...")


def _mesh_quality_improvements_finished(owner: Any, qt: Mapping[str, Any], job_id: str, candidates_obj: Any) -> None:
    self = owner
    if job_id != self._mesh_quality_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    candidates = [dict(item) for item in list(candidates_obj or []) if isinstance(item, Mapping)]
    self._complete_gui_worker_job(job_id, status="finished", message=f"{len(candidates)} candidates")
    self._mesh_quality_job_id = ""
    self._apply_mesh_quality_improvement_candidates(candidates)
    self.append_log(f"[GUI] メッシュ品質改善比較完了: {len(candidates)}件")


def _mesh_quality_failed(owner: Any, qt: Mapping[str, Any], job_id: str, message: str) -> None:
    self = owner
    if job_id != self._mesh_quality_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    manifest = self._write_gui_worker_failure_manifest(job_id, message)
    self._complete_gui_worker_job(job_id, status="failed", message=message)
    self._mesh_quality_job_id = ""
    suffix = f" manifest={manifest}" if manifest is not None else ""
    self.statusBar().showMessage(f"メッシュ品質ジョブ失敗: {message}")
    self.append_log(f"[GUI] メッシュ品質ジョブ失敗: {message}{suffix}")


def _apply_mesh_quality_improvement_candidates(owner: Any, qt: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> None:
    self = owner
    table = getattr(self, "mesh_quality_improvement_table", None)
    if table is None:
        return
    table.setRowCount(0)
    self.mesh_quality_improvement_candidates = []
    for candidate in candidates:
        report = self._mapping(candidate.get("report", {}))
        self.mesh_quality_improvement_candidates.append(
            {
                "nodes": dict(self._mapping(candidate.get("nodes", {}))),
                "elements": list(self._ensure_list(candidate.get("elements", []))),
                "report": dict(report),
            }
        )
        self._append_mesh_quality_improvement_row(report)
    if self.mesh_quality_improvement_candidates and table.rowCount() > 0:
        table.selectRow(0)
    self.statusBar().showMessage(f"mesh quality improvement candidates: {len(self.mesh_quality_improvement_candidates)}")


def _append_mesh_quality_improvement_row(owner: Any, qt: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.mesh_quality_improvement_table
    row = table.rowCount()
    table.insertRow(row)
    before = self._mapping(report.get("before", {}))
    after = self._mapping(report.get("after", {}))
    values = [
        report.get("method", ""),
        report.get("before_violation_count", ""),
        report.get("after_violation_count", ""),
        self._format_float(before.get("min_angle_deg", "")),
        self._format_float(after.get("min_angle_deg", "")),
        self._format_float(before.get("max_aspect_ratio", "")),
        self._format_float(after.get("max_aspect_ratio", "")),
        report.get("changed_nodes", 0),
        report.get("changed_elements", 0),
        report.get("status", ""),
    ]
    for col, value in enumerate(values):
        table.setItem(row, col, QTableWidgetItem(str(value)))


def apply_selected_mesh_quality_improvement(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    if not self.mesh_quality_improvement_candidates:
        self.compare_mesh_quality_improvements()
    if not self.mesh_quality_improvement_candidates:
        return
    table = self.mesh_quality_improvement_table
    selected_rows = sorted({item.row() for item in table.selectedItems()})
    if selected_rows:
        index = max(0, min(selected_rows[0], len(self.mesh_quality_improvement_candidates) - 1))
    else:
        index = min(
            range(len(self.mesh_quality_improvement_candidates)),
            key=lambda idx: (
                int(self.mesh_quality_improvement_candidates[idx]["report"].get("after_violation_count", 10**9) or 10**9),
                -float(self.mesh_quality_improvement_candidates[idx]["report"].get("score_delta", -math.inf)),
            ),
        )
    candidate = self.mesh_quality_improvement_candidates[index]
    mesh_cfg = self._mesh_cfg()
    mesh_cfg["nodes"] = candidate["nodes"]
    mesh_cfg["elements"] = candidate["elements"]
    node_sets = mesh_cfg.get("node_sets")
    if isinstance(node_sets, dict):
        node_sets["all"] = list(candidate["nodes"])
        for key, values in list(node_sets.items()):
            if key == "all":
                continue
            node_sets[key] = [str(nid) for nid in self._ensure_list(values) if str(nid) in candidate["nodes"]]
    element_ids = [str(element.get("id", "")) for element in candidate["elements"]]
    element_sets = mesh_cfg.get("element_sets")
    if not isinstance(element_sets, dict):
        element_sets = {}
        mesh_cfg["element_sets"] = element_sets
    element_sets["all"] = element_ids
    valid_elements = set(element_ids)
    for key, values in list(element_sets.items()):
        if key == "all":
            continue
        element_sets[key] = [str(eid) for eid in self._ensure_list(values) if str(eid) in valid_elements]
    history = self._list_value(mesh_cfg, "quality_improvement_history")
    history.append({"at": datetime.now().isoformat(timespec="seconds"), **dict(candidate["report"])})
    self.mesh_quality_improvement_candidates = []
    self._after_form_change(f"mesh quality improvement applied: {candidate['report'].get('method', '')}")


def _selected_mesh_quality_improvement_methods(owner: Any, qt: Mapping[str, Any], *_args: Any) -> list[str]:
    self = owner
    method = str(self.mesh_quality_method.currentData() or "all")
    if method == "all":
        return ["laplace", "node_optimize", "local_remesh", "quad_topology"]
    return [method]


def _mesh_dict_from_fem_mesh(owner: Any, qt: Mapping[str, Any], mesh: Any) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    self = owner
    nodes = {
        str(nid): [float(mesh.coords[index, 0]), float(mesh.coords[index, 1])]
        for index, nid in enumerate(mesh.node_ids)
    }
    elements: list[dict[str, Any]] = []
    for element in mesh.elements:
        item = {
            "id": str(element.id),
            "type": str(element.type),
            "nodes": [str(nid) for nid in element.nodes],
            "material": str(element.material),
            "integration": str(element.integration),
        }
        if not bool(getattr(element, "active", True)):
            item["active"] = False
        elements.append(item)
    return nodes, elements


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def add_element_library_preset(owner: Any, qt: Mapping[str, Any], kind: str) -> None:
    self = owner
    presets = {
        "beam": {
            "id": "beam_1",
            "type": "BEAM2",
            "nodes": "1,2",
            "material": "",
            "section": "section_name=RC_RECT_1M, E=30000.0, theory=euler, rotational_spring_i=1.0e12, rotational_spring_j=1.0e12",
            "behavior": "beam, rotational_dof, semi_rigid_ready",
            "active": True,
            "extra": {"solver_status": "solver_connected_2d", "analysis_connection": "2d_structural_line", "gui_element": "beam"},
        },
        "bar": {
            "id": "bar_1",
            "type": "BAR2",
            "nodes": "1,2",
            "material": "bar_steel",
            "section": "A=1.0",
            "behavior": "axial",
            "active": True,
            "extra": {"solver_status": "solver_connected_2d", "analysis_connection": "2d_structural_line", "gui_element": "bar"},
        },
        "spring": {
            "id": "spring_1",
            "type": "SPRING2",
            "nodes": "1,2",
            "material": "",
            "section": "kx=1000.0, ky=1000.0",
            "behavior": "linear",
            "active": True,
            "extra": {"solver_status": "solver_connected_2d", "analysis_connection": "2d_structural_line", "gui_element": "spring"},
        },
        "bilinear_spring": {
            "id": "bilinear_spring_1",
            "type": "AXIAL_SPRING2",
            "nodes": "1,2",
            "material": "",
            "section": "kx=1000.0, hysteresis_model=BILINEAR_STANDARD, yield_force=100.0, post_yield_stiffness=100.0",
            "behavior": "bilinear",
            "active": True,
            "extra": {"solver_status": "solver_connected_2d", "analysis_connection": "2d_structural_line", "gui_element": "bilinear_spring"},
        },
        "axial_spring": {
            "id": "axial_spring_1",
            "type": "AXIAL_SPRING2",
            "nodes": "1,2",
            "material": "",
            "section": "kx=1000.0",
            "behavior": "axial",
            "active": True,
            "extra": {"solver_status": "solver_connected_2d", "analysis_connection": "2d_structural_line", "gui_element": "axial_spring"},
        },
        "shear_spring": {
            "id": "shear_spring_1",
            "type": "SHEAR_SPRING2",
            "nodes": "1,2",
            "material": "",
            "section": "ky=1000.0",
            "behavior": "shear",
            "active": True,
            "extra": {"solver_status": "solver_connected_2d", "analysis_connection": "2d_structural_line", "gui_element": "shear_spring"},
        },
        "joint": {
            "id": "joint_1",
            "type": "JOINT",
            "nodes": "1,2|3,4",
            "material": "",
            "section": "kn=1000.0, kt=500.0, friction=0.3, cohesion=0.0, roughness=0.0, dilatancy_angle=0.0, roughness_degradation=0.0, residual_roughness_ratio=0.2",
            "behavior": "friction,no_tension",
            "active": True,
            "extra": {"material_model": "mohr_coulomb", "friction": 0.3, "cohesion": 0.0, "roughness": 0.0, "dilatancy_angle": 0.0, "roughness_degradation": 0.0, "residual_roughness_ratio": 0.2, "hydraulic_transfer": 0.0, "state_output": True, "slip_state_output": True},
        },
    }
    spec = presets.get(kind, presets["beam"])
    self.add_element_library_row(**spec)


def add_element_library_row(owner: Any, qt: Mapping[str, Any], *_args: Any, **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.element_library_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"id": "element_1", "type": "BEAM2", "nodes": "1,2", "material": "", "section": "", "behavior": "", "active": True, "extra": ""}
    defaults.update(values)
    extra = defaults.get("extra", "")
    if isinstance(extra, Mapping):
        extra = yaml.safe_dump(dict(extra), allow_unicode=True, sort_keys=False).strip()
    for col, key in enumerate(["id", "type", "nodes", "material", "section", "behavior", "active", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(extra if key == "extra" else defaults.get(key, ""))))


def remove_selected_element_library_rows(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    rows = sorted({index.row() for index in self.element_library_table.selectedIndexes()}, reverse=True)
    for row in rows:
        self.element_library_table.removeRow(row)


def populate_element_library_table(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    table = getattr(self, "element_library_table", None)
    if table is None:
        return
    table.setRowCount(0)
    for raw in self._ensure_list(self.cfg.get("structural_elements", [])):
        if not isinstance(raw, Mapping):
            continue
        section = self._key_value_text(self._mapping(raw.get("section", raw.get("stiffness", {}))))
        self.add_element_library_row(
            id=raw.get("id", ""),
            type=raw.get("type", raw.get("element_type", "BEAM2")),
            nodes=self._nodes_text(raw.get("nodes", [])),
            material=raw.get("material", raw.get("property", "")),
            section=section,
            behavior=self._behavior_text(raw.get("behavior", "")),
            active=raw.get("active", True),
            extra=self._element_extra_yaml(raw, {"id", "type", "element_type", "nodes", "material", "property", "section", "stiffness", "behavior", "active"}),
        )
    for raw in self._ensure_list(self.cfg.get("interfaces", self.cfg.get("interface_elements", []))):
        if not isinstance(raw, Mapping):
            continue
        minus = raw.get("minus_nodes", raw.get("nodes_minus"))
        plus = raw.get("plus_nodes", raw.get("nodes_plus"))
        if minus is not None and plus is not None:
            nodes = f"{self._nodes_text(minus)}|{self._nodes_text(plus)}"
        else:
            nodes = self._nodes_text(raw.get("nodes", []))
        stiffness = self._key_value_text({k: raw.get(k) for k in ("kn", "kt", "thickness", "friction", "cohesion", "hydraulic_transfer") if raw.get(k) is not None})
        behavior = dict(self._mapping(raw.get("behavior", {})))
        if raw.get("no_tension", False):
            behavior["no_tension"] = True
        self.add_element_library_row(
            id=raw.get("id", ""),
            type="JOINT",
            nodes=nodes,
            material="",
            section=stiffness,
            behavior=self._behavior_text(behavior),
            active=raw.get("active", True),
            extra=self._element_extra_yaml(raw, {"id", "nodes", "minus_nodes", "nodes_minus", "plus_nodes", "nodes_plus", "kn", "kt", "thickness", "friction", "cohesion", "hydraulic_transfer", "behavior", "no_tension", "active"}),
        )


def apply_element_library_panel(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    structural: list[dict[str, Any]] = []
    interfaces: list[dict[str, Any]] = []
    try:
        for row in range(self.element_library_table.rowCount()):
            element_id = self._table_text(self.element_library_table, row, 0).strip() or f"library_{row + 1}"
            element_type = self._table_text(self.element_library_table, row, 1).strip() or "BEAM2"
            nodes_text = self._table_text(self.element_library_table, row, 2).strip()
            material = self._table_text(self.element_library_table, row, 3).strip()
            section = self._parse_key_value_text(self._table_text(self.element_library_table, row, 4))
            behavior_text = self._table_text(self.element_library_table, row, 5).strip()
            active = self._bool_text(self._table_text(self.element_library_table, row, 6), True)
            extra = self._yaml_mapping_text(self._table_text(self.element_library_table, row, 7), "element library extra YAML")
            kind = element_type.lower().replace("-", "_")
            if kind in {"joint", "interface", "interface2d", "joint2"}:
                interface = dict(extra)
                interface["id"] = element_id
                if "|" in nodes_text:
                    minus_text, plus_text = nodes_text.split("|", 1)
                    interface["minus_nodes"] = self._parse_nodes_text(minus_text)
                    interface["plus_nodes"] = self._parse_nodes_text(plus_text)
                elif nodes_text:
                    interface["nodes"] = self._parse_nodes_text(nodes_text)
                interface["kn"] = float(section.get("kn", interface.get("kn", 1000.0)))
                interface["kt"] = float(section.get("kt", interface.get("kt", 500.0)))
                for key in ("thickness", "friction", "cohesion", "roughness", "dilatancy_angle", "dilation_angle", "psi", "roughness_degradation", "jrc_degradation", "residual_roughness_ratio", "roughness_residual_ratio", "hydraulic_transfer"):
                    if key in section:
                        target_key = "dilatancy_angle" if key in {"dilation_angle", "psi"} else ("roughness_degradation" if key == "jrc_degradation" else ("residual_roughness_ratio" if key == "roughness_residual_ratio" else key))
                        interface[target_key] = float(section[key])
                behavior = dict(self._mapping(interface.get("behavior", {})))
                behavior.update(self._behavior_mapping(behavior_text))
                if behavior:
                    interface["behavior"] = behavior
                if behavior.get("no_tension", False):
                    interface["no_tension"] = True
                if interface.get("friction", 0.0) or interface.get("cohesion", 0.0):
                    interface.setdefault("material_model", "mohr_coulomb")
                interface.setdefault("state_output", True)
                interface.setdefault("slip_state_output", True)
                interface["active"] = active
                interfaces.append(interface)
            else:
                element = dict(extra)
                element["id"] = element_id
                element["type"] = element_type.upper()
                element["nodes"] = self._parse_nodes_text(nodes_text)
                if material:
                    element["material"] = material
                if kind.startswith("spring"):
                    element["stiffness"] = section
                else:
                    element["section"] = section
                if behavior_text:
                    element["behavior"] = behavior_text
                element["active"] = active
                element.setdefault("solver_status", "solver_connected_2d")
                element.setdefault("analysis_connection", "2d_structural_line")
                structural.append(element)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if structural:
        self.cfg["structural_elements"] = structural
    else:
        self.cfg.pop("structural_elements", None)
    if interfaces:
        self.cfg["interfaces"] = interfaces
    else:
        self.cfg.pop("interfaces", None)
    self._after_form_change("要素ライブラリを反映しました")


__all__ = [
    "MESH_CONTROLLER_METHODS",
    "mesh_controller_contract",
    *MESH_CONTROLLER_METHODS,
]
