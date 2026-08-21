"""Mesh auto-generation and drag controller functions split from MainWindow.

MainWindow remains responsible for widget ownership, scene ownership, and generic
notifications.  This module owns GeoFEAS-like mesh generation commands,
background auto-mesh completion handling, and interactive mesh-control dragging.
"""

from __future__ import annotations

import copy
from datetime import datetime
import math
from typing import Any, Mapping

from geofem_app.gui.mesh_worker import generate_auto_geometry_mesh_snapshot


MESH_AUTO_CONTROLLER_METHODS = (
    "set_auto_mixed_mesh_mode",
    "set_mesh_division_width",
    "add_mesh_refinement",
    "confirm_mesh_generation",
    "apply_auto_geometry_mesh_async",
    "_auto_mesh_finished",
    "_auto_mesh_failed",
    "_auto_geometry_mesh_message",
    "apply_auto_geometry_mesh",
    "apply_delaunay_geometry_mesh",
    "begin_mesh_control_drag",
    "update_mesh_control_drag",
    "_set_mesh_control_from_drag",
    "begin_mesh_node_drag",
    "update_mesh_node_drag",
    "_set_mesh_node_from_drag",
    "_selected_mesh_edge_points",
)


def mesh_auto_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.mesh_auto_controller.v1",
        "method_count": len(MESH_AUTO_CONTROLLER_METHODS),
        "methods": list(MESH_AUTO_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner mesh config, dispatches auto-mesh jobs, and updates mesh-control drag state; MainWindow delegates mesh auto/drag actions",
        "covered_surfaces": ["auto_mesh_generation", "mesh_control_drag", "mesh_node_drag", "mesh_edge_selection"],
    }


def set_auto_mixed_mesh_mode(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    mesh_cfg = self._mesh_cfg()
    mesh_cfg["mode"] = "auto_mixed"
    mesh_cfg.setdefault("generator", "rectangle")
    self._after_form_change("オートメッシュ(混合)を設定しました")


def set_mesh_division_width(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    width, ok = QInputDialog.getDouble(self, "分割幅の設定", "分割幅", 1.0, 1.0e-9, 1.0e9, 6)
    if not ok:
        return
    mesh_cfg = self._mesh_cfg()
    mesh_cfg["division_width"] = width
    mesh_cfg["target_size"] = width
    self._after_form_change("分割幅を設定しました")


def add_mesh_refinement(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    text, ok = QInputDialog.getText(self, "局所細分", "cx,cy,radius,細分倍率", text="5,1,1,2")
    if not ok:
        return
    try:
        cx, cy, radius, factor = self._parse_float_csv(text, 4, "局所細分")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if radius <= 0.0 or factor <= 1.0:
        QMessageBox.warning(self, "GeoFEM", "radiusは正、細分倍率は1より大きい値にしてください。")
        return
    mesh_cfg = self._mesh_cfg()
    refinements = self._list_value(mesh_cfg, "refinements")
    refinements.append({"center": [cx, cy], "radius": radius, "factor": factor})
    self._after_form_change("局所細分を追加しました")


def confirm_mesh_generation(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    mesh_cfg = self._mesh_cfg()
    mesh_cfg["auto_mesh_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
    geometry = self._mapping(self.cfg.get("geometry", {}))
    has_geometry = any(isinstance(geometry.get(key), list) and geometry.get(key) for key in ("regions", "lines", "tunnels"))
    if str(mesh_cfg.get("mode", "")).lower() == "auto_mixed" and has_geometry:
        self.apply_auto_geometry_mesh_async()
        return
    self.apply_mesh_panel()
    self.append_log("[GeoFEAS操作] メッシュ分割-確認を実行しました")


def apply_auto_geometry_mesh_async(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QtCallableRunner = qt["QtCallableRunner"]
    if self._auto_mesh_job_id:
        self.statusBar().showMessage("オートメッシュ生成ジョブを実行中です。")
        return
    cfg_snapshot = copy.deepcopy(self.cfg)
    mesh_cfg = self._mapping(cfg_snapshot.get("mesh", {}))
    geometry = self._mapping(cfg_snapshot.get("geometry", {}))
    requested_type = self.mesh_type.currentText().upper()
    material = self.mesh_material.text().strip() or str(mesh_cfg.get("material", "soil"))
    integration = self.mesh_integration.currentText()
    boolean_expression = self.mesh_boolean_expression.text().strip() or str(mesh_cfg.get("boolean_expression", geometry.get("boolean_expression", "")) or "").strip()
    job_id = self.gui_jobs.start_job(
        "auto_mesh",
        target=str(self.current_input or "current-config"),
        metadata={"requested_type": requested_type, "material": material},
    )
    self._auto_mesh_job_id = job_id
    self._auto_mesh_snapshot = self._cfg_snapshot()
    runner = QtCallableRunner(
        job_id,
        lambda: generate_auto_geometry_mesh_snapshot(
            type(self),
            cfg_snapshot,
            requested_type=requested_type,
            material=material,
            integration=integration,
            boolean_expression=boolean_expression,
            mesh_x0=self.mesh_x0.text(),
            mesh_x1=self.mesh_x1.text(),
            mesh_nx=self.mesh_nx.text(),
        ),
    )
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._auto_mesh_finished)
    runner.signals.failed.connect(self._auto_mesh_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage("形状からのオートメッシュをバックグラウンド生成中...")


def _auto_mesh_finished(owner: Any, qt: Mapping[str, Any], job_id: str, result_obj: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    if job_id != self._auto_mesh_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    result = dict(result_obj or {})
    mesh_result = dict(self._mapping(result.get("mesh", {})))
    node_count = len(mesh_result.get("nodes", {})) if isinstance(mesh_result.get("nodes", {}), Mapping) else 0
    element_count = len(mesh_result.get("elements", [])) if isinstance(mesh_result.get("elements", []), list) else 0
    self._complete_gui_worker_job(job_id, status="finished", message=f"{node_count} nodes / {element_count} elements")
    self._auto_mesh_job_id = ""
    if self._auto_mesh_snapshot and self._auto_mesh_snapshot != self._cfg_snapshot():
        self._auto_mesh_snapshot = ""
        QMessageBox.warning(self, "GeoFEM", "オートメッシュ生成中に入力が変更されました。結果は適用せず、再実行してください。")
        return
    self._auto_mesh_snapshot = ""
    self.cfg["mesh"] = mesh_result
    if hasattr(self, "mark_mesh_rebuilt_for_current_geometry"):
        self.mark_mesh_rebuilt_for_current_geometry()
    self._after_form_change(self._auto_geometry_mesh_message(mesh_result))
    self.append_log("[GeoFEAS操作] 形状からオートメッシュを生成しました")


def _auto_mesh_failed(owner: Any, qt: Mapping[str, Any], job_id: str, message: str) -> None:
    self = owner
    if job_id != self._auto_mesh_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    manifest = self._write_gui_worker_failure_manifest(job_id, message)
    self._complete_gui_worker_job(job_id, status="failed", message=message)
    self._auto_mesh_job_id = ""
    self._auto_mesh_snapshot = ""
    suffix = f" manifest={manifest}" if manifest is not None else ""
    self.statusBar().showMessage(f"オートメッシュ生成ジョブ失敗: {message}")
    self.append_log(f"[GUI] オートメッシュ生成ジョブ失敗: {message}{suffix}")


def _auto_geometry_mesh_message(owner: Any, qt: Mapping[str, Any], mesh_result: Mapping[str, Any]) -> str:
    self = owner
    nodes = mesh_result.get("nodes", {})
    elements = mesh_result.get("elements", [])
    node_count = len(nodes) if isinstance(nodes, Mapping) else 0
    element_count = len(elements) if isinstance(elements, list) else 0
    if str(mesh_result.get("mode", "")).lower() == "auto_mixed_delaunay":
        return f"Delaunay三角メッシュを生成しました: 節点 {node_count} / 要素 {element_count}"
    quality = self._mapping(mesh_result.get("mesh_quality", {}))
    if quality:
        return (
            f"四角形優先メッシュを生成しました: 節点 {node_count} / 要素 {element_count} "
            f"(quad={quality.get('quad_count', 0)}, fallback_tri={quality.get('fallback_tri_count', 0)}, "
            f"recombined={quality.get('recombined_quad_count', 0)}, "
            f"mapped={quality.get('mapped_paving_quad_count', 0)}, "
            f"rectilinear={quality.get('rectilinear_paving_quad_count', 0)}, "
            f"regions={quality.get('rectilinear_paving_region_count', 0)}, "
            f"ring={quality.get('quadrilateral_ring_paving_quad_count', 0)}, "
            f"polyhole={quality.get('polygon_hole_paving_quad_count', 0)}, "
            f"circle={quality.get('circular_tunnel_paving_quad_count', 0)}, "
            f"multi_circle={quality.get('circular_tunnel_count', 0)}, "
            f"polygon={quality.get('polygon_quad_paving_quad_count', 0)}, "
            f"poly_regions={quality.get('polygon_quad_paving_region_count', 0)}, "
            f"curved={quality.get('parametric_curved_quad_paving_quad_count', quality.get('parametric_curve_hole_paving_quad_count', quality.get('parametric_curve_hole_boundary_grid_quad_count', quality.get('parametric_curved_outer_block_quad_count', 0))))}, "
            f"holes={quality.get('polygon_hole_count', 0)}, "
            f"nonstar_holes={quality.get('polygon_hole_non_star_count', 0)}, "
            f"tri_split={quality.get('subdivided_triangle_count', 0)}, "
            f"projected={quality.get('boundary_projected_node_count', 0)}, "
            f"layer_lines={quality.get('boundary_layer_added_x_count', 0) + quality.get('boundary_layer_added_y_count', 0)}, "
            f"curve_seg={quality.get('curved_boundary_segment_count', 0)})"
        )
    return f"形状からメッシュを生成しました: 節点 {node_count} / 要素 {element_count}"


def apply_auto_geometry_mesh(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    geometry = self._mapping(self.cfg.get("geometry", {}))
    bbox = self._geometry_bbox(geometry)
    if bbox is None:
        raise ValueError("線、閉領域、トンネルのいずれかを登録してください。")
    x0, x1, y0, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("形状の範囲が不正です。")
    old_mesh = dict(self._mapping(self.cfg.get("mesh", {})))
    target = float(old_mesh.get("target_size", old_mesh.get("division_width", 0.0)) or 0.0)
    if target <= 0.0:
        try:
            target = max(abs(float(self.mesh_x1.text()) - float(self.mesh_x0.text())) / max(int(float(self.mesh_nx.text())), 1), 1.0e-6)
        except ValueError:
            target = max((x1 - x0), (y1 - y0)) / 12.0
    nx = max(1, int(math.ceil((x1 - x0) / target)))
    ny = max(1, int(math.ceil((y1 - y0) / target)))
    requested_type = self.mesh_type.currentText().upper()
    material = self.mesh_material.text().strip() or str(old_mesh.get("material", "soil"))
    integration = self.mesh_integration.currentText()
    regions = self._geometry_regions(geometry)
    tunnels = self._geometry_tunnels(geometry)
    polygon_holes = self._geometry_polygon_holes(geometry)
    curved_regions = self._geometry_curved_regions(geometry)
    refinement_specs = self._mesh_refinement_specs(old_mesh)
    boolean_expression = self.mesh_boolean_expression.text().strip() or str(old_mesh.get("boolean_expression", geometry.get("boolean_expression", "")) or "").strip()
    region_ids = _geometry_region_ids(geometry)
    region_settings = _region_settings_with_geometry_materials(geometry, region_ids, old_mesh.get("region_settings", old_mesh.get("shape_mesh_settings", {})))
    from geofem_app.mesh_generation import generate_per_region_mapped_mesh, generate_quad_dominant_mesh

    mesh_result = generate_per_region_mapped_mesh(
        bbox=bbox,
        target=target,
        regions=regions,
        tunnels=tunnels,
        material=material,
        integration=integration,
        requested_type=requested_type,
        refinements=refinement_specs,
        polygon_holes=polygon_holes,
        region_ids=region_ids,
        region_settings=region_settings,
    )
    if mesh_result is not None:
        if region_settings:
            mesh_result["region_settings"] = dict(region_settings) if isinstance(region_settings, Mapping) else list(region_settings)
    elif requested_type.startswith("TRI"):
        self.apply_delaunay_geometry_mesh(geometry, bbox, target, requested_type)
        return
    else:
        mesh_result = generate_quad_dominant_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinement_specs,
            polygon_holes=polygon_holes,
            curved_regions=curved_regions,
            boolean_expression=boolean_expression or None,
        )

    if boolean_expression:
        mesh_result["boolean_expression"] = boolean_expression
    for key in ("refinements", "control_points", "split_lines", "size_map", "blocks", "quality_repairs", "region_settings"):
        if key in old_mesh:
            mesh_result[key] = old_mesh[key]
    for key in ("requires_rebuild", "partial_rebuild_required", "region_rebuild_required", "deleted_region_meshes", "stale_reason"):
        mesh_result.pop(key, None)
    mesh_result["auto_mesh_confirmed_at"] = datetime.now().isoformat(timespec="seconds")
    self.cfg["mesh"] = mesh_result
    if hasattr(self, "mark_mesh_rebuilt_for_current_geometry"):
        self.mark_mesh_rebuilt_for_current_geometry()
    quality = self._mapping(mesh_result.get("mesh_quality", {}))
    self._after_form_change(
        f"四角形優先メッシュを生成しました: 節点 {len(mesh_result['nodes'])} / 要素 {len(mesh_result['elements'])} "
        f"(quad={quality.get('quad_count', 0)}, fallback_tri={quality.get('fallback_tri_count', 0)}, "
        f"recombined={quality.get('recombined_quad_count', 0)}, "
        f"mapped={quality.get('mapped_paving_quad_count', 0)}, "
        f"rectilinear={quality.get('rectilinear_paving_quad_count', 0)}, "
        f"regions={quality.get('rectilinear_paving_region_count', 0)}, "
        f"ring={quality.get('quadrilateral_ring_paving_quad_count', 0)}, "
        f"polyhole={quality.get('polygon_hole_paving_quad_count', 0)}, "
        f"circle={quality.get('circular_tunnel_paving_quad_count', 0)}, "
        f"multi_circle={quality.get('circular_tunnel_count', 0)}, "
        f"polygon={quality.get('polygon_quad_paving_quad_count', 0)}, "
        f"poly_regions={quality.get('polygon_quad_paving_region_count', 0)}, "
        f"curved={quality.get('parametric_curved_quad_paving_quad_count', quality.get('parametric_curve_hole_paving_quad_count', quality.get('parametric_curve_hole_boundary_grid_quad_count', quality.get('parametric_curved_outer_block_quad_count', 0))))}, "
        f"holes={quality.get('polygon_hole_count', 0)}, "
        f"nonstar_holes={quality.get('polygon_hole_non_star_count', 0)}, "
        f"tri_split={quality.get('subdivided_triangle_count', 0)}, "
        f"projected={quality.get('boundary_projected_node_count', 0)}, "
        f"layer_lines={quality.get('boundary_layer_added_x_count', 0) + quality.get('boundary_layer_added_y_count', 0)}, "
        f"curve_seg={quality.get('curved_boundary_segment_count', 0)})"
    )
    return

    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            nid = str(len(nodes) + 1)
            nodes[nid] = [x, y]
            node_ids[(i, j)] = nid

    elements: list[dict[str, Any]] = []
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(regions))}
    tunnel_excluded: list[str] = []
    for j in range(ny):
        for i in range(nx):
            cx = x0 + (x1 - x0) * (i + 0.5) / nx
            cy = y0 + (y1 - y0) * (j + 0.5) / ny
            inside_region = True if not regions else any(self._point_in_polygon(cx, cy, region) for region in regions)
            inside_tunnel = any(math.hypot(cx - tx, cy - ty) <= tr for tx, ty, tr in tunnels)
            if not inside_region or inside_tunnel:
                tunnel_excluded.append(f"cell_{i}_{j}")
                continue
            n00 = node_ids[(i, j)]
            n10 = node_ids[(i + 1, j)]
            n11 = node_ids[(i + 1, j + 1)]
            n01 = node_ids[(i, j + 1)]
            if etype == "TRI3":
                created = [
                    {"id": str(len(elements) + 1), "type": "TRI3", "nodes": [n00, n10, n11], "material": material, "integration": integration},
                    {"id": str(len(elements) + 2), "type": "TRI3", "nodes": [n00, n11, n01], "material": material, "integration": integration},
                ]
            else:
                created = [{"id": str(len(elements) + 1), "type": "QUAD4", "nodes": [n00, n10, n11, n01], "material": material, "integration": integration}]
            for element in created:
                elements.append(element)
                for ridx, region in enumerate(regions):
                    if self._point_in_polygon(cx, cy, region):
                        region_sets[f"region_{ridx + 1}"].append(element["id"])
    if not elements:
        raise ValueError("有効な要素が生成されませんでした。閉領域やトンネル径を確認してください。")

    node_sets = {
        "left": [node_ids[(0, j)] for j in range(ny + 1)],
        "right": [node_ids[(nx, j)] for j in range(ny + 1)],
        "bottom": [node_ids[(i, 0)] for i in range(nx + 1)],
        "top": [node_ids[(i, ny)] for i in range(nx + 1)],
        "all": list(nodes),
    }
    element_sets = {"all": [element["id"] for element in elements]}
    element_sets.update({key: value for key, value in region_sets.items() if value})
    self.cfg["mesh"] = {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": element_sets,
        "mode": "auto_mixed",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": tunnel_excluded,
        "auto_mesh_confirmed_at": datetime.now().isoformat(timespec="seconds"),
    }
    self._after_form_change(f"形状からメッシュを生成しました: 節点 {len(nodes)} / 要素 {len(elements)}")


def apply_delaunay_geometry_mesh(owner: Any, qt: Mapping[str, Any], geometry: Mapping[str, Any], bbox: tuple[float, float, float, float], target: float, requested_type: str) -> None:
    self = owner
    import numpy as np
    from scipy.spatial import Delaunay

    x0, x1, y0, y1 = bbox
    material = self.mesh_material.text().strip() or str(self._mapping(self.cfg.get("mesh", {})).get("material", "soil"))
    integration = self.mesh_integration.currentText()
    regions = self._geometry_regions(geometry)
    tunnels = self._geometry_tunnels(geometry)
    mesh_options = self._mapping(self.cfg.get("mesh", {}))
    region_ids = _geometry_region_ids(geometry)
    region_settings = _region_settings_with_geometry_materials(geometry, region_ids, mesh_options.get("region_settings", mesh_options.get("shape_mesh_settings", {})))
    refinements = self._mesh_refinements(mesh_options)
    split_lines = self._mesh_split_lines(mesh_options)
    points: list[tuple[float, float]] = []
    boundary_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def add_point(x: float, y: float) -> None:
        points.append((float(x), float(y)))

    def add_segment(a: tuple[float, float], b: tuple[float, float]) -> None:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(length / target)))
        for i in range(n + 1):
            t = i / n
            add_point(a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)

    if regions:
        for region in regions:
            for a, b in zip(region, region[1:] + region[:1]):
                add_segment(a, b)
                boundary_segments.append((a, b))
    else:
        rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for a, b in zip(rect, rect[1:] + rect[:1]):
            add_segment(a, b)
            boundary_segments.append((a, b))

    for cx, cy, radius in tunnels:
        segments = max(16, int(math.ceil(2.0 * math.pi * radius / max(target, 1.0e-9))))
        ring: list[tuple[float, float]] = []
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            point = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            ring.append(point)
            add_point(*point)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            boundary_segments.append((a, b))

    for start, end, local_target in split_lines:
        local = local_target if local_target is not None and local_target > 0.0 else target
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1.0e-12:
            continue
        n = max(1, int(math.ceil(length / max(local, 1.0e-9))))
        for i in range(n + 1):
            t = i / n
            add_point(start[0] * (1.0 - t) + end[0] * t, start[1] * (1.0 - t) + end[1] * t)
        boundary_segments.append((start, end))

    nx = max(1, int(math.ceil((x1 - x0) / target)))
    ny = max(1, int(math.ceil((y1 - y0) / target)))
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * (i + 0.5 * (j % 2)) / max(nx, 1)
            if x > x1:
                x = x1
            inside_region = True if not regions else any(self._point_in_polygon(x, y, region) for region in regions)
            inside_tunnel = any(math.hypot(x - tx, y - ty) <= tr for tx, ty, tr in tunnels)
            if inside_region and not inside_tunnel:
                add_point(x, y)

    for cx, cy, radius, factor in refinements:
        local = max(target / factor, target * 0.05)
        rx0 = max(x0, cx - radius)
        rx1 = min(x1, cx + radius)
        ry0 = max(y0, cy - radius)
        ry1 = min(y1, cy + radius)
        rnx = max(1, int(math.ceil((rx1 - rx0) / local)))
        rny = max(1, int(math.ceil((ry1 - ry0) / local)))
        for j in range(rny + 1):
            y = ry0 + (ry1 - ry0) * j / rny
            for i in range(rnx + 1):
                x = rx0 + (rx1 - rx0) * i / rnx
                if math.hypot(x - cx, y - cy) > radius:
                    continue
                inside_region = True if not regions else any(self._point_in_polygon(x, y, region) for region in regions)
                inside_tunnel = any(math.hypot(x - tx, y - ty) <= tr for tx, ty, tr in tunnels)
                if inside_region and not inside_tunnel:
                    add_point(x, y)

    unique: dict[tuple[int, int], tuple[float, float]] = {}
    q = max(target, 1.0) * 1.0e-9
    for x, y in points:
        unique[(int(round(x / q)), int(round(y / q)))] = (x, y)
    point_array = np.asarray(list(unique.values()), dtype=float)
    if point_array.shape[0] < 3:
        raise ValueError("Delaunayメッシュには3点以上が必要です。")
    tri = Delaunay(point_array)
    kept: list[tuple[int, int, int, float, float]] = []
    for simplex in tri.simplices:
        pts = point_array[simplex]
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        inside_region = True if not regions else any(self._point_in_polygon(cx, cy, region) for region in regions)
        inside_tunnel = any(math.hypot(cx - tx, cy - ty) <= tr for tx, ty, tr in tunnels)
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        area = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
        crosses_boundary = self._triangle_crosses_boundary([(float(p[0]), float(p[1])) for p in pts], boundary_segments)
        if inside_region and not inside_tunnel and area > 1.0e-18 and not crosses_boundary:
            kept.append((int(simplex[0]), int(simplex[1]), int(simplex[2]), cx, cy))
    if not kept:
        raise ValueError("有効なDelaunay三角形が生成されませんでした。")
    point_array = self._smooth_delaunay_points(point_array, kept, boundary_segments, regions, tunnels, iterations=int(mesh_options.get("smooth_iterations", 2)))

    used = sorted({idx for simplex in kept for idx in simplex[:3]})
    id_by_index = {idx: str(i + 1) for i, idx in enumerate(used)}
    nodes = {id_by_index[idx]: [float(point_array[idx, 0]), float(point_array[idx, 1])] for idx in used}
    elements: list[dict[str, Any]] = []
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(regions))}
    for simplex in kept:
        eid = str(len(elements) + 1)
        region_index = next((ridx for ridx, region in enumerate(regions) if self._point_in_polygon(simplex[3], simplex[4], region)), None)
        local_material = material
        if region_index is not None and isinstance(region_settings, Mapping):
            region_id = region_ids[region_index] if region_index < len(region_ids) else f"region_{region_index + 1}"
            for candidate in _region_setting_key_candidates(region_id, region_ids):
                setting = region_settings.get(candidate)
                if isinstance(setting, Mapping) and str(setting.get("material", "") or "").strip():
                    local_material = str(setting.get("material"))
                    break
        element = {
            "id": eid,
            "type": "TRI3",
            "nodes": [id_by_index[simplex[0]], id_by_index[simplex[1]], id_by_index[simplex[2]]],
            "material": local_material,
            "integration": integration,
        }
        if region_index is not None:
            element["region"] = region_index + 1
            element["region_id"] = region_ids[region_index] if region_index < len(region_ids) else f"region_{region_index + 1}"
        elements.append(element)
        if region_index is not None:
            region_sets[f"region_{region_index + 1}"].append(eid)

    tol = max(target * 0.5, 1.0e-9)
    node_sets = {
        "left": [nid for nid, xy in nodes.items() if abs(xy[0] - x0) <= tol],
        "right": [nid for nid, xy in nodes.items() if abs(xy[0] - x1) <= tol],
        "bottom": [nid for nid, xy in nodes.items() if abs(xy[1] - y0) <= tol],
        "top": [nid for nid, xy in nodes.items() if abs(xy[1] - y1) <= tol],
        "all": list(nodes),
    }
    element_sets = {"all": [element["id"] for element in elements]}
    element_sets.update({key: value for key, value in region_sets.items() if value})
    from geofem_app.mesh_generation import with_requested_element_order

    mesh_result = {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": element_sets,
        "mode": "auto_mixed_delaunay",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "refinements": [{"center": [cx, cy], "radius": radius, "factor": factor} for cx, cy, radius, factor in refinements],
        "auto_mesh_confirmed_at": datetime.now().isoformat(timespec="seconds"),
    }
    if region_settings:
        mesh_result["region_settings"] = dict(region_settings) if isinstance(region_settings, Mapping) else list(region_settings)
    mesh_result = with_requested_element_order(mesh_result, requested_type)
    for key in ("refinements", "control_points", "split_lines", "size_map", "blocks", "quality_repairs", "region_settings"):
        if key in mesh_options:
            mesh_result[key] = mesh_options[key]
    self.cfg["mesh"] = mesh_result
    if hasattr(self, "mark_mesh_rebuilt_for_current_geometry"):
        self.mark_mesh_rebuilt_for_current_geometry()
    self._after_form_change(f"Delaunay三角メッシュを生成しました: 節点 {len(nodes)} / 要素 {len(elements)}")


def _region_setting_key_candidates(region_key: str, region_ids: list[str]) -> list[str]:
    region_key = str(region_key)
    candidates = [region_key]
    if region_key in region_ids:
        candidates.append(f"region_{region_ids.index(region_key) + 1}")
    elif region_key.startswith("region_"):
        suffix = region_key.removeprefix("region_")
        if suffix.isdigit():
            index = int(suffix) - 1
            if 0 <= index < len(region_ids):
                candidates.append(region_ids[index])
        else:
            candidates.append(suffix)
    elif region_key.isdigit():
        candidates.append(f"region_{region_key}")
        index = int(region_key) - 1
        if 0 <= index < len(region_ids):
            candidates.append(region_ids[index])
    return list(dict.fromkeys(candidates))


def _region_settings_with_geometry_materials(
    geometry: Mapping[str, Any],
    region_ids: list[str],
    raw_settings: Mapping[str, Any] | list[Any] | None,
) -> Mapping[str, Any] | list[Any]:
    if isinstance(raw_settings, list):
        settings: dict[str, Any] = {
            f"region_{index}": dict(raw)
            for index, raw in enumerate(raw_settings, start=1)
            if isinstance(raw, Mapping)
        }
    elif isinstance(raw_settings, Mapping):
        settings = {str(key): dict(value) if isinstance(value, Mapping) else value for key, value in raw_settings.items()}
    else:
        settings = {}
    raw_regions = geometry.get("regions", [])
    if not isinstance(raw_regions, list):
        return settings
    for index, raw in enumerate(raw_regions, start=1):
        if not isinstance(raw, Mapping):
            continue
        material = str(raw.get("material", raw.get("mat", "")) or "").strip()
        if not material:
            continue
        region_id = str(raw.get("id", f"region_{index}") or f"region_{index}")
        for candidate in _region_setting_key_candidates(region_id, region_ids):
            current = settings.get(candidate)
            setting = dict(current) if isinstance(current, Mapping) else {}
            setting["material"] = material
            settings[candidate] = setting
    return settings


def _geometry_region_ids(geometry: Mapping[str, Any]) -> list[str]:
    raw_regions = geometry.get("regions", [])
    ids: list[str] = []
    if not isinstance(raw_regions, list):
        return ids
    for index, raw in enumerate(raw_regions, start=1):
        if isinstance(raw, Mapping):
            ids.append(str(raw.get("id", f"region_{index}") or f"region_{index}"))
        else:
            ids.append(f"region_{index}")
    return ids


def begin_mesh_control_drag(owner: Any, qt: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    self = owner
    QGraphicsView = qt["QGraphicsView"]
    self.mesh_control_drag = dict(data)
    self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
    self.statusBar().showMessage(f"mesh control drag: {data.get('kind', '')}")


def update_mesh_control_drag(owner: Any, qt: Mapping[str, Any], scene_point: QPointF, *, final: bool) -> None:
    self = owner
    if self.mesh_control_drag is None:
        return
    x, y = self._scene_to_model(scene_point)
    x, y, _snap_label = self._snap_model_point(x, y)
    changed = self._set_mesh_control_from_drag(dict(self.mesh_control_drag), x, y)
    if final:
        self.mesh_control_drag = None
        self._apply_selection_drag_mode()
        if changed:
            self._after_form_change("mesh control drag edit applied")
        else:
            self.update_preview()
    elif changed:
        self.update_preview()


def _set_mesh_control_from_drag(owner: Any, qt: Mapping[str, Any], data: Mapping[str, Any], x: float, y: float) -> bool:
    self = owner
    kind = str(data.get("kind", ""))
    mesh = self._mesh_cfg()
    x = round(float(x), 12)
    y = round(float(y), 12)
    if kind == "mesh_control_point":
        raw_points = mesh.get("control_points", mesh.get("mesh_control_points", []))
        if not isinstance(raw_points, list):
            raw_points = []
            mesh["control_points"] = raw_points
        target_id = str(data.get("id", ""))
        try:
            target_index = int(data.get("index", -1))
        except (TypeError, ValueError):
            target_index = -1
        for index, raw in enumerate(raw_points):
            if not isinstance(raw, dict):
                continue
            if index == target_index or (target_id and str(raw.get("id", index + 1)) == target_id):
                raw["point"] = [x, y]
                return True
        return False
    if kind == "mesh_refinement":
        raw_refinements = mesh.get("refinements", mesh.get("local_refinements", []))
        if not isinstance(raw_refinements, list):
            return False
        target = data.get("id")
        target_text = str(target or "")
        target_index = -1
        try:
            if data.get("index") is not None:
                target_index = int(data.get("index"))
            elif target is not None:
                target_index = int(float(target)) - 1
        except (TypeError, ValueError):
            target_index = -1
        for index, raw in enumerate(raw_refinements):
            if not isinstance(raw, dict):
                continue
            if index == target_index or (target_text and str(raw.get("id", "")) == target_text):
                raw["center"] = [x, y]
                return True
        return False
    if kind == "mesh_size_map":
        raw_size_map = mesh.get("size_map", mesh.get("local_size_map", []))
        if not isinstance(raw_size_map, list):
            return False
        target_id = str(data.get("id", ""))
        try:
            target_index = int(data.get("index", -1))
        except (TypeError, ValueError):
            target_index = -1
        for index, raw in enumerate(raw_size_map):
            if not isinstance(raw, dict):
                continue
            if index == target_index or (target_id and str(raw.get("id", index + 1)) == target_id):
                raw["center"] = [x, y]
                return True
        return False
    if kind == "mesh_split_line":
        raw_split_lines = mesh.get("split_lines", mesh.get("split_line_constraints", []))
        if not isinstance(raw_split_lines, list):
            return False
        target = data.get("id")
        target_text = str(target or "")
        target_index = -1
        try:
            if data.get("index") is not None:
                target_index = int(data.get("index"))
            elif target is not None:
                target_index = int(float(target)) - 1
        except (TypeError, ValueError):
            target_index = -1
        start = data.get("start", [x, y])
        end = data.get("end", [x, y])
        try:
            sx, sy = self._xy_pair(start)
            ex, ey = self._xy_pair(end)
        except (TypeError, ValueError):
            sx = ex = float(x)
            sy = ey = float(y)
        dx = x - (sx + ex) * 0.5
        dy = y - (sy + ey) * 0.5
        for index, raw in enumerate(raw_split_lines):
            if not isinstance(raw, dict):
                continue
            if index == target_index or (target_text and str(raw.get("id", "")) == target_text):
                try:
                    raw_start = raw.get("start", raw.get("p1", [raw.get("x1", sx), raw.get("y1", sy)]))
                    raw_end = raw.get("end", raw.get("p2", [raw.get("x2", ex), raw.get("y2", ey)]))
                    rsx, rsy = self._xy_pair(raw_start)
                    rex, rey = self._xy_pair(raw_end)
                except (TypeError, ValueError):
                    rsx, rsy, rex, rey = sx, sy, ex, ey
                raw["start"] = [round(float(rsx + dx), 12), round(float(rsy + dy), 12)]
                raw["end"] = [round(float(rex + dx), 12), round(float(rey + dy), 12)]
                return True
        return False
    return False


def begin_mesh_node_drag(owner: Any, qt: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    self = owner
    QGraphicsView = qt["QGraphicsView"]
    node_id = str(data.get("id", "")).strip()
    nodes = self._mapping(self._mesh_cfg().get("nodes", {}))
    if not node_id or node_id not in nodes:
        self.statusBar().showMessage("移動するメッシュ節点を選択してください。")
        return
    try:
        start = self._xy_pair(nodes[node_id])
    except (TypeError, ValueError):
        start = (0.0, 0.0)
    self.mesh_node_drag = {
        "id": node_id,
        "start": [float(start[0]), float(start[1])],
        "moved": False,
        "preview_scale": float(getattr(self, "preview_scale", 1.0) or 1.0),
        "preview_ox": float(getattr(self, "preview_ox", 0.0) or 0.0),
        "preview_oy": float(getattr(self, "preview_oy", 0.0) or 0.0),
    }
    self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
    self.statusBar().showMessage(f"メッシュ節点ドラッグ: {node_id}")


def update_mesh_node_drag(owner: Any, qt: Mapping[str, Any], scene_point: QPointF, *, final: bool) -> None:
    self = owner
    if getattr(self, "mesh_node_drag", None) is None:
        return
    if final and not bool(self.mesh_node_drag.get("moved", False)):
        self.mesh_node_drag = None
        self._apply_selection_drag_mode()
        return
    if final and bool(self.mesh_node_drag.get("moved", False)):
        data = dict(self.mesh_node_drag)
        self.mesh_node_drag = None
        self._apply_selection_drag_mode()
        mesh = self._mesh_cfg()
        moves = self._list_value(mesh, "manual_node_moves")
        node_id = str(data.get("id", ""))
        nodes = self._mapping(mesh.get("nodes", {}))
        end = list(nodes.get(node_id, [])) if node_id in nodes else []
        moves.append(
            {
                "node": node_id,
                "from": list(data.get("start", [])),
                "to": end,
                "source": "gui_mesh_node_drag",
                "at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self._after_form_change(f"メッシュ節点 {node_id} を移動しました")
        return
    scale = float(self.mesh_node_drag.get("preview_scale", getattr(self, "preview_scale", 1.0)) or 1.0)
    if abs(scale) <= 1.0e-30:
        scale = 1.0
    ox = float(self.mesh_node_drag.get("preview_ox", getattr(self, "preview_ox", 0.0)) or 0.0)
    oy = float(self.mesh_node_drag.get("preview_oy", getattr(self, "preview_oy", 0.0)) or 0.0)
    x, y = (float(scene_point.x()) - ox) / scale, (oy - float(scene_point.y())) / scale
    x, y, _snap_label = self._snap_model_point(x, y)
    changed = self._set_mesh_node_from_drag(dict(self.mesh_node_drag), x, y)
    if changed:
        self.mesh_node_drag["moved"] = True
        self.request_preview_update(reset_view=False, reason="mesh node drag")


def _set_mesh_node_from_drag(owner: Any, qt: Mapping[str, Any], data: Mapping[str, Any], x: float, y: float) -> bool:
    self = owner
    node_id = str(data.get("id", "")).strip()
    if not node_id:
        return False
    mesh = self._mesh_cfg()
    nodes = mesh.get("nodes", {})
    x = round(float(x), 12)
    y = round(float(y), 12)
    if isinstance(nodes, dict):
        if node_id not in nodes:
            return False
        old = nodes.get(node_id)
        try:
            ox, oy = self._xy_pair(old)
        except (TypeError, ValueError):
            ox = oy = math.nan
        if math.isfinite(ox) and math.isfinite(oy) and abs(ox - x) <= 1.0e-12 and abs(oy - y) <= 1.0e-12:
            return False
        nodes[node_id] = [x, y]
        return True
    if isinstance(nodes, list):
        for raw in nodes:
            if not isinstance(raw, dict) or str(raw.get("id", "")) != node_id:
                continue
            try:
                ox, oy = self._xy_pair(raw.get("point", raw.get("coords", raw.get("xy", [raw.get("x", 0.0), raw.get("y", 0.0)]))))
            except (TypeError, ValueError):
                ox = oy = math.nan
            if math.isfinite(ox) and math.isfinite(oy) and abs(ox - x) <= 1.0e-12 and abs(oy - y) <= 1.0e-12:
                return False
            raw["point"] = [x, y]
            raw["x"] = x
            raw["y"] = y
            return True
    return False


def _selected_mesh_edge_points(owner: Any, qt: Mapping[str, Any], *_args: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    self = owner
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        return None
    for item in self.scene.selectedItems():
        data = item.data(0)
        if not isinstance(data, dict) or data.get("kind") != "edge":
            continue
        nodes = data.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) < 2:
            continue
        n1 = str(nodes[0])
        n2 = str(nodes[1])
        if n1 not in mesh.node_index or n2 not in mesh.node_index:
            continue
        i1 = mesh.node_index[n1]
        i2 = mesh.node_index[n2]
        return (
            (float(mesh.coords[i1, 0]), float(mesh.coords[i1, 1])),
            (float(mesh.coords[i2, 0]), float(mesh.coords[i2, 1])),
        )
    return None


__all__ = [
    "MESH_AUTO_CONTROLLER_METHODS",
    "mesh_auto_controller_contract",
    *MESH_AUTO_CONTROLLER_METHODS,
]
