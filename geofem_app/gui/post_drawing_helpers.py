"""Low-level Post drawing helpers split from MainWindow.

The owner keeps the scene, widgets, and result state; this module owns contour,
vector, stage-difference, SRM slip candidate, legend, and colormap drawing logic.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

POST_DRAWING_HELPER_METHODS = (
    "_display_element_values",
    "_element_result_brush",
    "_draw_result_overlays",
    "_draw_node_result_overlay",
    "_add_non_overlapping_text",
    "_draw_contour_polylines",
    "_draw_stage_overlays",
    "_stage_diff_visual_sets",
    "_draw_stage_diff_overlay",
    "_draw_arrow",
    "_draw_slip_surface_overlay",
    "_draw_srm_slip_candidates_overlay",
    "_draw_distribution_plot",
    "_draw_result_legend",
    "_contour_color",
    "_result_colormap_changed",
    "_current_colormap_name",
    "_legend_title_text",
    "_legend_bounds",
    "_optional_float",
    "_colormap_stops",
    "_gradient_color",
)


def post_drawing_helper_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.post_drawing_helpers.v1",
        "method_count": len(POST_DRAWING_HELPER_METHODS),
        "methods": list(POST_DRAWING_HELPER_METHODS),
        "owner_boundary": "helpers draw into owner scene and read owner post/result widgets; MainWindow delegates drawing helpers",
    }


def _display_element_values(owner: Any, qt: Mapping[str, Any], mesh: Any | None = None) -> dict[str, float]:
    self = owner
    if not self.result_element_values:
        return {}
    values = {str(eid): float(value) for eid, value in self.result_element_values.items()}
    if self.compare_element_values:
        values = {eid: value - float(self.compare_element_values.get(eid, 0.0)) for eid, value in values.items()}
    smooth = getattr(self, "smooth_contours", None)
    if smooth is None or not smooth.isChecked() or mesh is None:
        return values
    node_values: dict[str, list[float]] = {}
    for element in mesh.elements:
        if element.id not in values:
            continue
        for nid in element.nodes:
            node_values.setdefault(str(nid), []).append(values[element.id])
    averaged_nodes = {nid: sum(vals) / len(vals) for nid, vals in node_values.items() if vals}
    smoothed: dict[str, float] = {}
    for element in mesh.elements:
        samples = [averaged_nodes[str(nid)] for nid in element.nodes if str(nid) in averaged_nodes]
        if samples:
            smoothed[element.id] = sum(samples) / len(samples)
    values.update(smoothed)
    return values

def _element_result_brush(owner: Any, qt: Mapping[str, Any], element_id: str, default: QBrush) -> QBrush:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    if self.post_mode not in {"contour", "plastic", "srm"} or not self.result_element_values:
        return default
    if self.post_mode == "srm":
        fl_check = getattr(self, "srm_show_fl_contour", None)
        if fl_check is not None and not fl_check.isChecked():
            return default
    display_values = getattr(self, "_post_element_display_values", {}) or self._display_element_values()
    value = display_values.get(str(element_id))
    if value is None:
        return default
    if self.post_mode == "plastic":
        return QBrush(QColor("#f4a261") if value > 0.0 else QColor("#e9f2ff"))
    values = list(display_values.values())
    vmin, vmax = self._legend_bounds(values)
    clip = getattr(self, "clip_contours_to_legend", None)
    if clip is not None and clip.isChecked() and vmax > vmin:
        value = max(vmin, min(vmax, value))
    return QBrush(self._contour_color(value, vmin, vmax, 150))

def _draw_result_overlays(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    QPointF = qt["QPointF"]
    QPolygonF = qt["QPolygonF"]
    if self.post_mode == "plastic" and self.result_element_values:
        self._draw_slip_surface_overlay(mesh, scale, ox, oy)
    if self.post_mode == "srm" and self.srm_slip_candidates:
        show_candidates = getattr(self, "srm_show_slip_candidates", None)
        if show_candidates is None or show_candidates.isChecked():
            self._draw_srm_slip_candidates_overlay(mesh, scale, ox, oy)
    if self.post_mode == "node_contour" and self.result_node_values:
        self._draw_node_result_overlay(mesh, scale, ox, oy)
    if self.post_mode not in {"contour", "node_contour", "plastic", "srm", "deformed", "vector"} or not self.result_displacements:
        return
    try:
        factor = float(self.deformation_scale.text())
    except ValueError:
        factor = 1.0
    pen = QPen(QColor("#dc3545"))
    pen.setWidth(2)
    vector_pen = QPen(QColor("#7b2cbf"))
    vector_pen.setWidth(2)
    node_index = mesh.node_index
    edge_map = {
        "TRI3": [(0, 1), (1, 2), (2, 0)],
        "TRI6": [(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)],
        "QUAD4": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "QUAD8": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 3), (3, 7), (7, 0)],
    }

    def deformed_xy(nid: str) -> tuple[float, float]:
        idx = node_index[nid]
        ux, uy = self.result_displacements.get(nid, (0.0, 0.0))
        x = (mesh.coords[idx, 0] + factor * ux) * scale + ox
        y = oy - (mesh.coords[idx, 1] + factor * uy) * scale
        return float(x), float(y)

    element_display_values = getattr(self, "_post_element_display_values", {}) or self._display_element_values(mesh)
    element_value_list = list(element_display_values.values())
    node_value_list = list(self.result_node_values.values()) if self.result_node_values else []
    element_vmin, element_vmax = self._legend_bounds(element_value_list) if element_value_list else (0.0, 1.0)
    node_vmin, node_vmax = self._legend_bounds(node_value_list) if node_value_list else (0.0, 1.0)

    def deformed_brush(element: Any) -> QBrush:
        if self.post_mode == "plastic":
            value = element_display_values.get(str(element.id), 0.0)
            return QBrush(QColor(244, 162, 97, 170) if value > 0.0 else QColor(233, 242, 255, 70))
        if self.post_mode in {"contour", "srm"} and element_value_list:
            value = element_display_values.get(str(element.id))
            if value is not None:
                return QBrush(self._contour_color(value, element_vmin, element_vmax, 185))
        if self.post_mode == "node_contour" and node_value_list:
            samples = [self.result_node_values[str(nid)] for nid in element.nodes if str(nid) in self.result_node_values]
            if samples:
                value = sum(samples) / len(samples)
                return QBrush(self._contour_color(value, node_vmin, node_vmax, 170))
        return QBrush(QColor(220, 53, 69, 34))

    overlay = getattr(self, "show_deformed_overlay", None)
    active_helper = getattr(self, "_deformed_result_overlay_active", None)
    draw_deformed_mesh = active_helper() if callable(active_helper) else (self.post_mode == "deformed" or overlay is None or overlay.isChecked())
    if draw_deformed_mesh:
        for element in mesh.elements:
            corner_count = 3 if element.type.startswith("TRI") else 4
            polygon_points = []
            for nid in element.nodes[:corner_count]:
                x, y = deformed_xy(nid)
                polygon_points.append(QPointF(x, y))
            if len(polygon_points) >= 3:
                poly_item = self.scene.addPolygon(QPolygonF(polygon_points), QPen(QColor(220, 53, 69, 95)), deformed_brush(element))
                poly_item.setData(0, {"kind": "deformed_element", "element": element.id, "scale": factor, "component": self.post_component})
                poly_item.setToolTip(f"deformed element {element.id} / {self.post_component} / scale={factor:g}")
            for a, b in edge_map[element.type]:
                na = element.nodes[a]
                nb = element.nodes[b]
                ax, ay = deformed_xy(na)
                bx, by = deformed_xy(nb)
                edge_item = self.scene.addLine(ax, ay, bx, by, pen)
                edge_item.setData(0, {"kind": "deformed_edge", "element": element.id, "scale": factor})
                edge_item.setToolTip(f"deformed element {element.id} / scale={factor:g}")
    if self.post_mode not in {"deformed", "vector"}:
        return
    max_u = max((math.hypot(ux, uy) for ux, uy in self.result_displacements.values()), default=0.0)
    if max_u > 0.0 and factor > 0.0:
        arrow_scale = factor * scale
        vector_rows: list[tuple[str, float, float]] = []
        for nid, (uxv, uyv) in self.result_displacements.items():
            if nid not in node_index:
                continue
            if self.post_mode == "vector" or math.hypot(uxv, uyv) >= max_u * 0.15:
                vector_rows.append((nid, uxv, uyv))
        policy = self._current_display_quality_policy or self._display_quality_policy(mesh)
        max_vectors = max(1, int(policy.max_vectors))
        stride = max(1, math.ceil(len(vector_rows) / max_vectors)) if vector_rows else 1
        for index, (nid, uxv, uyv) in enumerate(vector_rows):
            if index % stride != 0:
                continue
            idx = node_index[nid]
            x = float(mesh.coords[idx, 0] * scale + ox)
            y = float(oy - mesh.coords[idx, 1] * scale)
            if self.post_mode == "vector" or math.hypot(uxv, uyv) >= max_u * 0.15:
                self._draw_arrow(x, y, x + uxv * arrow_scale, y - uyv * arrow_scale, vector_pen)

def _draw_node_result_overlay(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    values = list(self.result_node_values.values())
    if not values:
        return
    vmin, vmax = self._legend_bounds(values)
    pen = QPen(QColor("#343a40"))
    for i, nid in enumerate(mesh.node_ids):
        if nid not in self.result_node_values:
            continue
        value = self.result_node_values[nid]
        brush = QBrush(self._contour_color(value, vmin, vmax, 210))
        x = float(mesh.coords[i, 0] * scale + ox)
        y = float(oy - mesh.coords[i, 1] * scale)
        item = self.scene.addEllipse(x - 5.5, y - 5.5, 11.0, 11.0, pen, brush)
        item.setToolTip(f"{self.post_component} node {nid}: {value:.6g}")
        item.setData(0, {"kind": "node", "id": nid})

def _add_non_overlapping_text(owner: Any, qt: Mapping[str, Any], text: str, x: float, y: float, color: QColor, data: Mapping[str, Any] | None = None) -> QGraphicsTextItem | None:
    self = owner
    QColor = qt["QColor"]
    QGraphicsTextItem = qt["QGraphicsTextItem"]
    offsets = [(0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (0.0, 12.0), (0.0, -12.0), (14.0, 10.0), (-14.0, 10.0), (14.0, -10.0), (-14.0, -10.0)]
    for dx, dy in offsets:
        item = self.scene.addText(text)
        item.setDefaultTextColor(color)
        item.setPos(x + dx, y + dy)
        rect = item.sceneBoundingRect().adjusted(-2.0, -2.0, 2.0, 2.0)
        if any(rect.intersects(occupied) for occupied in getattr(self, "_post_label_bounds", [])):
            self.scene.removeItem(item)
            continue
        self._post_label_bounds.append(rect)
        if data is not None:
            item.setData(0, dict(data))
        return item
    return None

def _draw_contour_polylines(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    Qt = qt["Qt"]
    toggle = getattr(self, "show_contour_lines", None)
    if toggle is not None and not toggle.isChecked():
        return
    if self.post_mode not in {"contour", "srm"} or not self.result_element_values:
        return
    values = getattr(self, "_post_element_display_values", {}) or self._display_element_values(mesh)
    if not values:
        return
    node_samples: dict[str, list[float]] = {}
    for element in mesh.elements:
        if element.id not in values:
            continue
        corner_count = 3 if element.type.startswith("TRI") else 4
        for nid in element.nodes[:corner_count]:
            node_samples.setdefault(str(nid), []).append(float(values[element.id]))
    node_values = {nid: sum(samples) / len(samples) for nid, samples in node_samples.items() if samples}
    finite = [value for value in node_values.values() if math.isfinite(value)]
    if len(finite) < 2:
        return
    vmin, vmax = self._legend_bounds(finite)
    if vmax <= vmin:
        return
    policy = self._current_display_quality_policy or self._display_quality_policy(mesh)
    try:
        level_count = max(2, min(30, int(float(self.contour_level_count.text()))))
    except Exception:
        level_count = 7
    level_count = max(2, min(level_count, int(policy.contour_level_count)))
    levels = [vmin + (vmax - vmin) * i / (level_count + 1.0) for i in range(1, level_count + 1)]
    curve_mode = getattr(self, "contour_interpolation", None) is not None and "曲線" in self.contour_interpolation.currentText()
    try:
        curve_segments = max(2, min(16, int(float(self.contour_curve_segments.text()))))
    except Exception:
        curve_segments = 4
    curve_segments = max(2, min(curve_segments, int(policy.contour_curve_segments)))
    node_index = mesh.node_index
    pen = QPen(QColor("#212529"))
    pen.setWidth(1)
    pen.setStyle(Qt.PenStyle.SolidLine)
    label_every = 0

    def interpolate(a: str, b: str, level: float) -> tuple[float, float] | None:
        va = node_values.get(a)
        vb = node_values.get(b)
        if va is None or vb is None:
            return None
        if (level < min(va, vb)) or (level > max(va, vb)) or abs(vb - va) <= 1.0e-30:
            return None
        t = (level - va) / (vb - va)
        ia = node_index[a]
        ib = node_index[b]
        x = float((mesh.coords[ia, 0] * (1.0 - t) + mesh.coords[ib, 0] * t) * scale + ox)
        y = float(oy - (mesh.coords[ia, 1] * (1.0 - t) + mesh.coords[ib, 1] * t) * scale)
        return x, y

    for element in mesh.elements:
        corner_count = 3 if element.type.startswith("TRI") else 4
        corners = [str(nid) for nid in element.nodes[:corner_count]]
        edges = list(zip(corners, corners[1:] + corners[:1]))
        center_x = 0.0
        center_y = 0.0
        for nid in corners:
            idx = node_index[nid]
            center_x += float(mesh.coords[idx, 0] * scale + ox)
            center_y += float(oy - mesh.coords[idx, 1] * scale)
        center_x /= max(len(corners), 1)
        center_y /= max(len(corners), 1)
        for level in levels:
            points = []
            for a, b in edges:
                point = interpolate(a, b, level)
                if point is not None:
                    points.append(point)
            if len(points) < 2:
                continue
            for a, b in zip(points[0::2], points[1::2]):
                if curve_mode:
                    mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
                    control = (mid[0] * 0.82 + center_x * 0.18, mid[1] * 0.82 + center_y * 0.18)
                    last = a
                    for segment in range(1, curve_segments + 1):
                        t = segment / curve_segments
                        x = (1.0 - t) ** 2 * a[0] + 2.0 * (1.0 - t) * t * control[0] + t**2 * b[0]
                        y = (1.0 - t) ** 2 * a[1] + 2.0 * (1.0 - t) * t * control[1] + t**2 * b[1]
                        item = self.scene.addLine(last[0], last[1], x, y, pen)
                        item.setData(0, {"kind": "contour_line", "level": level, "element": element.id, "curve": True})
                        last = (x, y)
                else:
                    item = self.scene.addLine(a[0], a[1], b[0], b[1], pen)
                    item.setData(0, {"kind": "contour_line", "level": level, "element": element.id, "curve": False})
                if (
                    getattr(self, "show_contour_labels", None) is not None
                    and self.show_contour_labels.isChecked()
                    and policy.draw_contour_labels
                    and label_every % 5 == 0
                ):
                    self._add_non_overlapping_text(
                        f"{level:.3g}",
                        0.5 * (a[0] + b[0]) + 3.0,
                        0.5 * (a[1] + b[1]) - 12.0,
                        QColor("#212529"),
                        {"kind": "contour_line_label", "level": level, "element": element.id},
                    )
                label_every += 1

def _draw_stage_overlays(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    QPointF = qt["QPointF"]
    QPolygonF = qt["QPolygonF"]
    row = self._selected_stage_row()
    stages = self._stages()
    if row is None or row < 0 or row >= len(stages):
        return
    stage = stages[row]
    node_index = mesh.node_index

    def map_xy(nid: str) -> tuple[float, float]:
        idx = node_index[nid]
        return float(mesh.coords[idx, 0] * scale + ox), float(oy - mesh.coords[idx, 1] * scale)

    element_pen = QPen(QColor("#2a9d8f"))
    element_pen.setWidth(3)
    node_pen = QPen(QColor("#0d6efd"))
    node_brush = QBrush(QColor("#0d6efd"))
    load_pen = QPen(QColor("#c1121f"))
    load_pen.setWidth(2)
    hydro_pen = QPen(QColor("#0aa2c0"))
    hydro_pen.setWidth(2)

    for prop in self._ensure_list(stage.get("element_properties", [])):
        if not isinstance(prop, Mapping):
            continue
        targets = self._element_targets_for_spec(mesh, prop)
        for element in mesh.elements:
            if element.id not in targets:
                continue
            points = []
            corner_count = 3 if element.type.startswith("TRI") else 4
            for nid in element.nodes[:corner_count]:
                x, y = map_xy(nid)
                points.append(QPointF(x, y))
            item = self.scene.addPolygon(QPolygonF(points), element_pen)
            item.setToolTip(f"stage material: {prop.get('material', '')}")

    merged_bcs = list(self._ensure_list(self.cfg.get("boundary_conditions", self.cfg.get("bc", []))))
    merged_bcs.extend(self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))))
    for bc in merged_bcs:
        if not isinstance(bc, Mapping):
            continue
        for nid in self._node_targets_for_spec(mesh, bc):
            x, y = map_xy(nid)
            self.scene.addRect(x - 4.0, y - 4.0, 8.0, 8.0, node_pen, node_brush)

    stage_loads = list(self._ensure_list(self.cfg.get("loads", []))) + list(self._ensure_list(stage.get("loads", [])))
    nodal_vectors: list[tuple[str, float, float]] = []
    edge_vectors: list[tuple[tuple[str, str], float, float]] = []
    for load in stage_loads:
        if not isinstance(load, Mapping):
            continue
        tx = float(load.get("tx", load.get("qx", 0.0)) or 0.0)
        ty = float(load.get("ty", load.get("qy", 0.0)) or 0.0)
        fx = float(load.get("fx", load.get("px", 0.0)) or 0.0)
        fy = float(load.get("fy", load.get("py", 0.0)) or 0.0)
        if "edge" in load or "edges" in load:
            for edge in self._edge_targets_for_spec(mesh, load):
                edge_vectors.append((edge, tx, ty))
        else:
            for nid in self._node_targets_for_spec(mesh, load):
                nodal_vectors.append((nid, fx, fy))
    max_load = max([math.hypot(vx, vy) for _nid, vx, vy in nodal_vectors] + [math.hypot(vx, vy) for _edge, vx, vy in edge_vectors] + [1.0])
    arrow_scale = 36.0 / max_load
    for nid, fx, fy in nodal_vectors:
        x, y = map_xy(nid)
        self._draw_arrow(x, y, x + fx * arrow_scale, y - fy * arrow_scale, load_pen)
    for edge, tx, ty in edge_vectors:
        p0 = map_xy(edge[0])
        p1 = map_xy(edge[1])
        mx = 0.5 * (p0[0] + p1[0])
        my = 0.5 * (p0[1] + p1[1])
        self.scene.addLine(p0[0], p0[1], p1[0], p1[1], load_pen)
        self._draw_arrow(mx, my, mx + tx * arrow_scale, my - ty * arrow_scale, load_pen)

    hydro = stage.get("hydro", stage.get("consolidation", {}))
    if isinstance(hydro, Mapping):
        for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs"):
            for spec in self._ensure_list(hydro.get(key, [])):
                if not isinstance(spec, Mapping):
                    continue
                for edge in self._edge_targets_for_spec(mesh, spec):
                    p0 = map_xy(edge[0])
                    p1 = map_xy(edge[1])
                    self.scene.addLine(p0[0], p0[1], p1[0], p1[1], hydro_pen)
                for nid in self._node_targets_for_spec(mesh, spec):
                    x, y = map_xy(nid)
                    self.scene.addEllipse(x - 4.0, y - 4.0, 8.0, 8.0, hydro_pen, QBrush(QColor("#cff4fc")))

def _stage_diff_visual_sets(owner: Any, qt: Mapping[str, Any], mesh: Any) -> dict[str, set[Any]]:
    self = owner
    row = self._selected_stage_row()
    stages = self._stages()
    if row is None or row < 0 or row >= len(stages):
        return {"deactivated": set(), "reactivated": set(), "material": set(), "nodes": set(), "edges": set()}
    stage = stages[row]
    before = self._active_elements_after_stage_index(mesh, row - 1)
    after = self._active_elements_after_stage_index(mesh, row)
    material: set[str] = set()
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for prop in self._ensure_list(stage.get("element_properties", [])):
        if isinstance(prop, Mapping):
            material.update(self._element_targets_for_spec(mesh, prop))
    for bc in self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))):
        if isinstance(bc, Mapping):
            nodes.update(self._node_targets_for_spec(mesh, bc))
    for load in self._ensure_list(stage.get("loads", [])):
        if not isinstance(load, Mapping):
            continue
        nodes.update(self._node_targets_for_spec(mesh, load))
        edges.update(self._edge_targets_for_spec(mesh, load))
    hydro = stage.get("hydro", stage.get("consolidation", {}))
    if isinstance(hydro, Mapping):
        for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs"):
            for spec in self._ensure_list(hydro.get(key, [])):
                if isinstance(spec, Mapping):
                    nodes.update(self._node_targets_for_spec(mesh, spec))
                    edges.update(self._edge_targets_for_spec(mesh, spec))
    return {
        "deactivated": before - after,
        "reactivated": after - before,
        "material": material,
        "nodes": nodes,
        "edges": edges,
    }

def _draw_stage_diff_overlay(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    QPointF = qt["QPointF"]
    QPolygonF = qt["QPolygonF"]
    Qt = qt["Qt"]
    toggle = getattr(self, "show_stage_diff_overlay", None)
    if toggle is not None and not toggle.isChecked():
        return
    visual = self._stage_diff_visual_sets(mesh)
    if not any(visual.values()):
        return
    node_index = mesh.node_index

    def map_xy(nid: str) -> tuple[float, float]:
        idx = node_index[nid]
        return float(mesh.coords[idx, 0] * scale + ox), float(oy - mesh.coords[idx, 1] * scale)

    pens = {
        "deactivated": QPen(QColor("#d00000")),
        "reactivated": QPen(QColor("#198754")),
        "material": QPen(QColor("#f77f00")),
    }
    brushes = {
        "deactivated": QBrush(QColor(208, 0, 0, 55)),
        "reactivated": QBrush(QColor(25, 135, 84, 45)),
        "material": QBrush(QColor(247, 127, 0, 50)),
    }
    for pen in pens.values():
        pen.setWidth(3)
        pen.setStyle(Qt.PenStyle.DashLine)
    for element in mesh.elements:
        category = ""
        if element.id in visual["deactivated"]:
            category = "deactivated"
        elif element.id in visual["reactivated"]:
            category = "reactivated"
        elif element.id in visual["material"]:
            category = "material"
        if not category:
            continue
        corner_count = 3 if element.type.startswith("TRI") else 4
        points = [QPointF(*map_xy(nid)) for nid in element.nodes[:corner_count]]
        item = self.scene.addPolygon(QPolygonF(points), pens[category], brushes[category])
        item.setToolTip(f"stage diff: {category} element {element.id}")
        item.setData(0, {"kind": "stage_diff", "category": category, "element": element.id})
    node_pen = QPen(QColor("#6610f2"))
    node_pen.setWidth(2)
    node_brush = QBrush(QColor(102, 16, 242, 120))
    for nid in visual["nodes"]:
        if nid not in node_index:
            continue
        x, y = map_xy(nid)
        item = self.scene.addRect(x - 6.0, y - 6.0, 12.0, 12.0, node_pen, node_brush)
        item.setToolTip(f"stage diff: boundary/load/hydro node {nid}")
        item.setData(0, {"kind": "stage_diff", "category": "node_condition", "node": nid})
    edge_pen = QPen(QColor("#6610f2"))
    edge_pen.setWidth(4)
    edge_pen.setStyle(Qt.PenStyle.DotLine)
    for edge in visual["edges"]:
        if edge[0] not in node_index or edge[1] not in node_index:
            continue
        p0 = map_xy(edge[0])
        p1 = map_xy(edge[1])
        item = self.scene.addLine(p0[0], p0[1], p1[0], p1[1], edge_pen)
        item.setToolTip(f"stage diff: boundary/load/hydro edge {edge[0]}-{edge[1]}")
        item.setData(0, {"kind": "stage_diff", "category": "edge_condition", "edge": list(edge)})

def _draw_arrow(owner: Any, qt: Mapping[str, Any], x0: float, y0: float, x1: float, y1: float, pen: Any) -> None:
    self = owner
    QPointF = qt["QPointF"]
    self.scene.addLine(x0, y0, x1, y1, pen)
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1.0e-12:
        return
    ux = dx / length
    uy = dy / length
    size = 7.0
    left = QPointF(x1 - size * (ux + 0.45 * uy), y1 - size * (uy - 0.45 * ux))
    right = QPointF(x1 - size * (ux - 0.45 * uy), y1 - size * (uy + 0.45 * ux))
    self.scene.addLine(x1, y1, left.x(), left.y(), pen)
    self.scene.addLine(x1, y1, right.x(), right.y(), pen)

def _draw_slip_surface_overlay(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    import numpy as np

    node_index = mesh.node_index
    centers: list[tuple[float, float]] = []
    for element in mesh.elements:
        if self.result_element_values.get(element.id, 0.0) <= 0.0:
            continue
        pts = mesh.coords[[node_index[nid] for nid in element.nodes[: 3 if element.type.startswith("TRI") else 4]]]
        centers.append((float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))))
    if len(centers) < 2:
        return
    centers.sort(key=lambda item: (item[0], item[1]))
    pen = QPen(QColor("#c1121f"))
    pen.setWidth(3)
    for a, b in zip(centers, centers[1:]):
        self.scene.addLine(a[0] * scale + ox, oy - a[1] * scale, b[0] * scale + ox, oy - b[1] * scale, pen)

def _draw_srm_slip_candidates_overlay(owner: Any, qt: Mapping[str, Any], mesh: Any, scale: float, ox: float, oy: float) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    Qt = qt["Qt"]
    colors = ["#c1121f", "#f77f00", "#7b2cbf", "#198754"]
    for index, candidate in enumerate(self.srm_slip_candidates[:8], start=1):
        points = candidate.get("points", [])
        if not isinstance(points, list) or not points:
            continue
        color = QColor(colors[(index - 1) % len(colors)])
        pen = QPen(color)
        pen.setWidth(4 if index == 1 else 2)
        if index > 1:
            pen.setStyle(Qt.PenStyle.DashLine)
        screen_points: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = float(point[0]) * scale + ox
                y = oy - float(point[1]) * scale
            except (TypeError, ValueError):
                continue
            screen_points.append((x, y))
        if not screen_points:
            continue
        tip = (
            f"SRMすべり面候補 {index}\n"
            f"type={candidate.get('type', '')} optimized={candidate.get('optimized', '')}\n"
            f"elements={','.join(str(eid) for eid in candidate.get('elements', []))}\n"
            f"min_FL={candidate.get('min_fl', '')} score={candidate.get('score', '')}"
        )
        if len(screen_points) == 1:
            x, y = screen_points[0]
            item = self.scene.addEllipse(x - 7.0, y - 7.0, 14.0, 14.0, pen, QBrush(color))
            item.setToolTip(tip)
            item.setData(0, {"kind": "srm_slip_candidate", "rank": index})
        else:
            for a, b in zip(screen_points, screen_points[1:]):
                item = self.scene.addLine(a[0], a[1], b[0], b[1], pen)
                item.setToolTip(tip)
                item.setData(0, {"kind": "srm_slip_candidate", "rank": index})
        label = self.scene.addText(str(index))
        label.setDefaultTextColor(color)
        label.setPos(screen_points[0][0] + 4.0, screen_points[0][1] - 18.0)

def _draw_distribution_plot(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    if self.post_mode != "distribution" or len(self.result_distribution) < 2:
        return
    rect = self.scene.itemsBoundingRect()
    x0 = rect.right() + 22.0
    y0 = rect.bottom() - 165.0
    width = 220.0
    height = 130.0
    xs = [item[0] for item in self.result_distribution]
    ys = [item[1] for item in self.result_distribution]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(xmax - xmin) <= 1.0e-30:
        xmax = xmin + 1.0
    if abs(ymax - ymin) <= 1.0e-30:
        ymax = ymin + 1.0
    axis_pen = QPen(QColor("#495057"))
    curve_pen = QPen(QColor("#0d6efd"))
    curve_pen.setWidth(2)
    self.scene.addRect(x0, y0, width, height, axis_pen, QBrush(QColor(255, 255, 255, 215)))
    last: tuple[float, float] | None = None
    for xval, yval in self.result_distribution:
        px = x0 + (xval - xmin) / (xmax - xmin) * width
        py = y0 + height - (yval - ymin) / (ymax - ymin) * height
        if last is not None:
            self.scene.addLine(last[0], last[1], px, py, curve_pen)
        last = (px, py)
    title = self.scene.addText(f"分布図: {self.post_component}")
    title.setDefaultTextColor(QColor("#212529"))
    title.setPos(x0, y0 - 24.0)
    hi = self.scene.addText(f"{ymax:.4g}")
    hi.setDefaultTextColor(QColor("#212529"))
    hi.setPos(x0 + width + 6.0, y0 - 4.0)
    lo = self.scene.addText(f"{ymin:.4g}")
    lo.setDefaultTextColor(QColor("#212529"))
    lo.setPos(x0 + width + 6.0, y0 + height - 16.0)
    xmin_label = self.scene.addText(f"{xmin:.4g}")
    xmin_label.setDefaultTextColor(QColor("#212529"))
    xmin_label.setPos(x0, y0 + height + 2.0)
    xmax_label = self.scene.addText(f"{xmax:.4g}")
    xmax_label.setDefaultTextColor(QColor("#212529"))
    xmax_label.setPos(x0 + width - 42.0, y0 + height + 2.0)

def _draw_result_legend(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QPen = qt["QPen"]
    Qt = qt["Qt"]
    if self.post_mode == "node_contour":
        values = list(self.result_node_values.values())
    elif self.post_mode in {"contour", "plastic", "srm"}:
        values = list((getattr(self, "_post_element_display_values", {}) or self._display_element_values()).values())
    else:
        return
    if not values:
        return
    rect = self.scene.itemsBoundingRect()
    x0 = rect.right() + 20.0
    y0 = rect.top() + 20.0
    if self.post_mode == "plastic":
        self.scene.addRect(x0, y0, 18.0, 18.0, QPen(QColor("#9c6644")), QBrush(QColor("#f4a261")))
        label = self.scene.addText("plastic / SRM slip band")
        label.setDefaultTextColor(QColor("#333333"))
        label.setPos(x0 + 24.0, y0 - 2.0)
        return
    vmin, vmax = self._legend_bounds(values)
    title_text = self._legend_title_text() or f"{self.post_component}{' diff' if self.compare_element_values else ''}"
    title = self.scene.addText(title_text)
    title.setDefaultTextColor(QColor("#333333"))
    title.setPos(x0, y0 - 22.0)
    if self._is_axisymmetric_analysis():
        note = self.scene.addText("axisym r-z: x=r, y=z, ring measure 2*pi*r")
        note.setDefaultTextColor(QColor("#0d6efd"))
        note.setPos(x0, y0 - 42.0)
        note.setData(0, {"kind": "axisymmetric_legend_note", "component": self.post_component})
    for i in range(10):
        t = i / 9.0
        self.scene.addRect(x0, y0 + (9 - i) * 12.0, 24.0, 12.0, QPen(Qt.PenStyle.NoPen), QBrush(self._contour_color(vmin + (vmax - vmin) * t, vmin, vmax, 180)))
    hi = self.scene.addText(f"{vmax:.4g}")
    hi.setDefaultTextColor(QColor("#333333"))
    hi.setPos(x0 + 30.0, y0 - 3.0)
    lo = self.scene.addText(f"{vmin:.4g}")
    lo.setDefaultTextColor(QColor("#333333"))
    lo.setPos(x0 + 30.0, y0 + 106.0)
    if self.post_component in {"FL", "safety_factor", "factor_of_safety"} and vmin < 1.0 < vmax:
        ycrit = y0 + (1.0 - (1.0 - vmin) / (vmax - vmin)) * 108.0
        self.scene.addLine(x0 - 3.0, ycrit, x0 + 28.0, ycrit, QPen(QColor("#c1121f")))
        crit = self.scene.addText("FL=1")
        crit.setDefaultTextColor(QColor("#c1121f"))
        crit.setPos(x0 + 30.0, ycrit - 10.0)
    if self.post_mode == "srm":
        slip_pen = QPen(QColor("#c1121f"))
        slip_pen.setWidth(4)
        self.scene.addLine(x0, y0 + 136.0, x0 + 26.0, y0 + 136.0, slip_pen)
        slip = self.scene.addText("SRMすべり面候補")
        slip.setDefaultTextColor(QColor("#333333"))
        slip.setPos(x0 + 31.0, y0 + 125.0)

def _contour_color(owner: Any, qt: Mapping[str, Any], value: float, vmin: float, vmax: float, alpha: int) -> QColor:
    self = owner
    QColor = qt["QColor"]
    t = 0.5 if abs(vmax - vmin) < 1.0e-30 else (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    cmap = self._current_colormap_name()
    if cmap.lower().startswith("safety") and vmin < 1.0 < vmax:
        if value < 1.0:
            t = 0.5 * max(0.0, min(1.0, (value - vmin) / (1.0 - vmin)))
        else:
            t = 0.5 + 0.5 * max(0.0, min(1.0, (value - 1.0) / (vmax - 1.0)))
    return self._gradient_color(self._colormap_stops(cmap), t, alpha)

def _result_colormap_changed(owner: Any, qt: Mapping[str, Any], text: str) -> None:
    self = owner
    self.result_colormap_name = text.strip() or "Geo blue-red"
    self.update_preview()

def _current_colormap_name(owner: Any, qt: Mapping[str, Any]) -> str:
    self = owner
    combo = getattr(self, "result_colormap", None)
    if combo is None:
        return self.result_colormap_name
    return combo.currentText().strip() or self.result_colormap_name

def _legend_title_text(owner: Any, qt: Mapping[str, Any]) -> str:
    self = owner
    edit = getattr(self, "legend_title_edit", None)
    return edit.text().strip() if edit is not None else ""

def _legend_bounds(owner: Any, qt: Mapping[str, Any], values: list[float]) -> tuple[float, float]:
    self = owner
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    vmin = min(finite)
    vmax = max(finite)
    min_edit = getattr(self, "legend_min_edit", None)
    max_edit = getattr(self, "legend_max_edit", None)
    user_min = self._optional_float(min_edit.text()) if min_edit is not None else None
    user_max = self._optional_float(max_edit.text()) if max_edit is not None else None
    if user_min is not None:
        vmin = user_min
    if user_max is not None:
        vmax = user_max
    if abs(vmax - vmin) <= 1.0e-30:
        vmax = vmin + 1.0
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    return vmin, vmax

def _optional_float(owner: Any, qt: Mapping[str, Any], text: str) -> float | None:
    self = owner
    raw = text.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None

def _colormap_stops(owner: Any, qt: Mapping[str, Any], name: str) -> list[QColor]:
    self = owner
    QColor = qt["QColor"]
    raw = name.strip()
    if raw.lower().startswith("custom:"):
        colors = [QColor(part.strip()) for part in raw.split(":", 1)[1].split(",") if part.strip()]
        valid = [color for color in colors if color.isValid()]
        if len(valid) >= 2:
            return valid
    key = raw.lower().replace("_", " ").replace("-", " ")
    if "viridis" in key:
        return [QColor("#440154"), QColor("#31688e"), QColor("#35b779"), QColor("#fde725")]
    if "terrain" in key:
        return [QColor("#225ea8"), QColor("#41b6c4"), QColor("#a1d99b"), QColor("#ffffcc"), QColor("#b15928")]
    if "gray" in key or "grey" in key:
        return [QColor("#f8f9fa"), QColor("#6c757d"), QColor("#212529")]
    if "safety" in key:
        return [QColor("#b02a37"), QColor("#ffc107"), QColor("#2a9d8f"), QColor("#0d6efd")]
    return [QColor("#313695"), QColor("#74add1"), QColor("#ffffbf"), QColor("#f46d43"), QColor("#a50026")]

def _gradient_color(owner: Any, qt: Mapping[str, Any], stops: list[QColor], t: float, alpha: int) -> QColor:
    self = owner
    QColor = qt["QColor"]
    if not stops:
        return QColor(120, 120, 120, alpha)
    if len(stops) == 1:
        color = QColor(stops[0])
        color.setAlpha(alpha)
        return color
    t = max(0.0, min(1.0, t))
    scaled = t * (len(stops) - 1)
    index = min(int(math.floor(scaled)), len(stops) - 2)
    local = scaled - index
    a = stops[index]
    b = stops[index + 1]
    color = QColor(
        int(a.red() * (1.0 - local) + b.red() * local),
        int(a.green() * (1.0 - local) + b.green() * local),
        int(a.blue() * (1.0 - local) + b.blue() * local),
        alpha,
    )
    return color

__all__ = [
    "POST_DRAWING_HELPER_METHODS",
    "post_drawing_helper_contract",
    *POST_DRAWING_HELPER_METHODS,
]
