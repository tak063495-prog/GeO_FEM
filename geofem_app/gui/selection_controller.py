"""Model-view selection and snap controller functions split from MainWindow.

MainWindow keeps widget ownership, project state, scene ownership, and job
lifecycle. This module owns selection modes, condition/expression selection,
selection history/named selections, and model-view snap point resolution.
"""

from __future__ import annotations

import ast
import math
from datetime import datetime
from typing import Any, Mapping

import yaml


SELECTION_CONTROLLER_METHODS = (
    'set_operation_mode',
    'set_draw_mode',
    '_apply_selection_drag_mode',
    'set_selection_mode',
    'begin_lasso_selection',
    'extend_lasso_selection',
    'finish_lasso_selection',
    '_current_selection_operation',
    '_selection_mode_help_text',
    '_update_selection_help',
    'show_mode_help',
    'build_selection_expression_from_gui',
    'clear_preview_selection',
    'invert_preview_selection',
    '_preview_select_items',
    'select_by_filter_dialog',
    'select_by_filter',
    'select_by_expression',
    '_eval_selection_expression',
    '_compare_selection_values',
    '_current_stage_condition_targets',
    '_record_selection_history',
    '_refresh_selection_history_table',
    'restore_selection_history',
    'undo_selection',
    'redo_selection',
    'save_named_selection',
    '_refresh_named_selection_table',
    'restore_named_selection_by_row',
    'restore_named_selection',
    'compare_named_selection_dialog',
    'compare_named_selections',
    'save_selection_comparison',
    '_refresh_selection_compare_table',
    'save_current_selection_as_set',
    '_scene_to_model',
    '_snap_tolerance_model',
    '_snap_model_point',
    '_snap_candidates',
    '_selected_element_ids',
    '_selected_model_points',
    '_selected_geometry_line',
    '_geometry_line_by_id',
    '_mesh_node_points',
    '_geometry_snap_points',
    '_intersection_snap_points',
    '_model_points',
    '_nearest_node_id',
)

_QT_SYMBOLS = (
    "QBrush",
    "QColor",
    "QGraphicsItem",
    "QGraphicsView",
    "QInputDialog",
    "QMessageBox",
    "QPointF",
    "QPolygonF",
    "QTableWidgetItem",
    "Qt",
)

CAD_DRAW_MODES = {"line", "helper", "region", "polyline", "rectangle", "circle", "arc", "curve", "point"}


def selection_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.selection_controller.v1",
        "method_count": len(SELECTION_CONTROLLER_METHODS),
        "methods": list(SELECTION_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner selection state and reads model geometry/mesh snapshots; MainWindow delegates model-view selection and snap actions",
        "covered_surfaces": [
            "selection_modes",
            "filter_expression_selection",
            "selection_history",
            "named_selection",
            "selection_sets",
            "snap_helpers",
            "selection_targets",
        ],
    }


def _bind_qt(qt: Mapping[str, Any]) -> None:
    for name in _QT_SYMBOLS:
        if name in qt:
            globals()[name] = qt[name]


def set_operation_mode(owner: Any, qt: Mapping[str, Any], mode: str) -> None:
    _bind_qt(qt)
    self = owner
    if "選択" in mode:
        self.set_draw_mode("select")
    self.statusBar().showMessage(mode)
    self.append_log(f"[GeoFEAS操作] {mode}")
def set_draw_mode(owner: Any, qt: Mapping[str, Any], mode: str) -> None:
    _bind_qt(qt)
    self = owner
    self.draw_mode = mode
    self.draw_start = None
    self.cad_draw_points = []
    if mode != "region":
        self.region_points = []
    if hasattr(self, "clear_cad_lightweight_preview"):
        self.clear_cad_lightweight_preview()
    if mode in CAD_DRAW_MODES:
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        labels = {
            "line": "直線作図",
            "helper": "補助線作図",
            "region": "閉領域作図",
            "polyline": "折れ線作図",
            "rectangle": "矩形作図",
            "circle": "円作図",
            "arc": "円弧作図",
            "curve": "曲線作図",
            "point": "点作図",
        }
        label = labels.get(mode, "作図")
        detail = "右クリック/Enterで確定、Escで取消できます。" if mode in {"region", "polyline", "curve"} else "モデルビューをクリックしてください。"
        self.statusBar().showMessage(f"{label}: {detail}")
    else:
        self._apply_selection_drag_mode()
        self.statusBar().showMessage("選択モード")
    self._update_selection_help()
    if hasattr(self, "_refresh_auxiliary_selection_context"):
        self._refresh_auxiliary_selection_context()
    if hasattr(self, "_refresh_cad_tool_button_state"):
        self._refresh_cad_tool_button_state()
    if hasattr(self, "_refresh_mesh_tool_button_state"):
        self._refresh_mesh_tool_button_state()
def _apply_selection_drag_mode(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    if self.draw_mode in CAD_DRAW_MODES:
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
        return
    if self.selection_mode == "rectangle":
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
    elif self.selection_mode == "lasso":
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
    else:
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
def set_selection_mode(owner: Any, qt: Mapping[str, Any], mode: str) -> None:
    _bind_qt(qt)
    self = owner
    self.draw_mode = "select"
    self.draw_start = None
    self.region_points = []
    if mode not in {"normal", "rectangle", "lasso"}:
        mode = "normal"
    self.selection_mode = mode
    self.lasso_points = []
    self._apply_selection_drag_mode()
    labels = {"normal": "通常選択", "rectangle": "矩形選択", "lasso": "投げ縄選択"}
    self.statusBar().showMessage(labels[mode])
    self._update_selection_help()
    if hasattr(self, "_refresh_auxiliary_selection_context"):
        self._refresh_auxiliary_selection_context()
    if hasattr(self, "_refresh_cad_tool_button_state"):
        self._refresh_cad_tool_button_state()
    if hasattr(self, "_refresh_mesh_tool_button_state"):
        self._refresh_mesh_tool_button_state()
def begin_lasso_selection(owner: Any, qt: Mapping[str, Any], scene_point: QPointF) -> None:
    _bind_qt(qt)
    self = owner
    self.lasso_points = [QPointF(scene_point)]
    self.statusBar().showMessage("投げ縄選択: ドラッグして範囲を囲んでください。")
def extend_lasso_selection(owner: Any, qt: Mapping[str, Any], scene_point: QPointF) -> None:
    _bind_qt(qt)
    self = owner
    if not self.lasso_points:
        return
    last = self.lasso_points[-1]
    if math.hypot(last.x() - scene_point.x(), last.y() - scene_point.y()) >= 3.0:
        self.lasso_points.append(QPointF(scene_point))
def finish_lasso_selection(owner: Any, qt: Mapping[str, Any], scene_point: QPointF | None = None) -> None:
    _bind_qt(qt)
    self = owner
    if scene_point is not None:
        self.extend_lasso_selection(scene_point)
    points = list(self.lasso_points)
    self.lasso_points = []
    if len(points) < 3:
        return
    polygon = QPolygonF(points)
    self._preview_select_items(
        lambda item, data: polygon.containsPoint(item.sceneBoundingRect().center(), Qt.FillRule.OddEvenFill),
        mode=self._current_selection_operation(),
    )
    self._record_selection_history("投げ縄選択")
    self.statusBar().showMessage("投げ縄選択を適用しました。")
def _current_selection_operation(owner: Any, qt: Mapping[str, Any]) -> str:
    _bind_qt(qt)
    self = owner
    combo = getattr(self, "selection_operation", None)
    if combo is None:
        return "replace"
    mode = combo.currentData()
    return str(mode or "replace")
def _selection_mode_help_text(owner: Any, qt: Mapping[str, Any]) -> str:
    _bind_qt(qt)
    self = owner
    if self.draw_mode in CAD_DRAW_MODES:
        return "作図モード: 左クリックで点を入力します。右クリック/Enterで確定、Escで取消できます。スナップと長さ/角度指定は下部入力欄と連動します。"
    if self.selection_mode == "rectangle":
        return "矩形選択: ドラッグ矩形内の節点/辺/要素を選択します。選択操作で追加/解除/反転を切替できます。"
    if self.selection_mode == "lasso":
        return "投げ縄選択: 左ドラッグで囲んだ範囲を選択します。細かい対象は条件式フィルタで絞り込めます。"
    return "通常選択: クリックで節点/辺/要素を選択、ドラッグでパンします。Ctrl+Alt+Z/Yで選択を元に戻す/やり直す。"
def _update_selection_help(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    label = getattr(self, "selection_help_label", None)
    if label is not None:
        label.setText(self._selection_mode_help_text())
def show_mode_help(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    text = self._selection_mode_help_text()
    self.statusBar().showMessage(text)
    QMessageBox.information(self, "GeoFEM 選択ヘルプ", text)
def build_selection_expression_from_gui(owner: Any, qt: Mapping[str, Any]) -> str:
    _bind_qt(qt)
    self = owner
    field = self.selection_expr_field.currentText() if getattr(self, "selection_expr_field", None) is not None else "kind"
    op = self.selection_expr_operator.currentText() if getattr(self, "selection_expr_operator", None) is not None else "=="
    raw_value = self.selection_expr_value.text().strip() if getattr(self, "selection_expr_value", None) is not None else ""
    if field in {"sets", "blocks"} and op in {"in", "not in"}:
        expression = f"{raw_value!r} {op} {field}"
    elif field in {"stage_active", "stage_inactive"}:
        expression = f"{field} == {str(raw_value).lower() not in {'false', '0', 'no'}}"
    elif op in {"in", "not in"}:
        values = [part.strip() for part in raw_value.split(",") if part.strip()]
        expression = f"{field} {op} {values!r}"
    else:
        try:
            numeric = float(raw_value)
            value_repr = str(numeric)
        except ValueError:
            value_repr = repr(raw_value)
        expression = f"{field} {op} {value_repr}"
    preview = getattr(self, "selection_expr_preview", None)
    if preview is not None:
        preview.setText(expression)
    return expression
def clear_preview_selection(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    self.scene.clearSelection()
    self._record_selection_history("選択解除")
def invert_preview_selection(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    for item in self.scene.items():
        data = item.data(0)
        if isinstance(data, Mapping) and data.get("kind") in {"node", "edge", "element", "geometry_line", "geometry_region", "geometry_tunnel"}:
            item.setSelected(not item.isSelected())
    self._record_selection_history("選択反転")
def _preview_select_items(owner: Any, qt: Mapping[str, Any], predicate: Any, *, mode: str = "replace") -> int:
    _bind_qt(qt)
    self = owner
    if mode == "replace":
        self.scene.clearSelection()
    changed = 0
    for item in self.scene.items():
        data = item.data(0)
        if not isinstance(data, Mapping):
            continue
        if data.get("kind") not in {"node", "edge", "element", "geometry_line", "geometry_region", "geometry_tunnel"}:
            continue
        if not predicate(item, data):
            continue
        if mode == "remove":
            item.setSelected(False)
        elif mode == "toggle":
            item.setSelected(not item.isSelected())
        else:
            item.setSelected(True)
        changed += 1
    return changed
def select_by_filter_dialog(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    labels = {
        "節点ID": "nodes",
        "要素ID": "elements",
        "辺ノード": "edges",
        "材料": "material",
        "ブロック": "block",
        "現在ステージ有効要素": "stage_active",
        "現在ステージ無効要素": "stage_inactive",
        "表示セット": "set",
        "ステージ条件": "stage_condition",
        "条件式": "expression",
    }
    label, ok = QInputDialog.getItem(self, "条件選択", "対象", list(labels), 0, False)
    if not ok:
        return
    default = "" if labels[label] in {"stage_active", "stage_inactive", "set", "stage_condition"} else "all"
    if labels[label] == "expression":
        default = 'kind == "element" and material == "soil"'
    value, ok = QInputDialog.getText(self, "条件選択", "値（all またはカンマ区切り）", text=default)
    if not ok:
        return
    self.select_by_filter(kind=labels[label], value=value, mode=self._current_selection_operation())
def select_by_filter(owner: Any, qt: Mapping[str, Any], *, kind: str, value: str | None = None, mode: str = "replace") -> int:
    _bind_qt(qt)
    self = owner
    value_text = (value or "").strip()
    if kind == "expression":
        return self.select_by_expression(value_text, mode=mode)
    wanted = {part.strip() for part in value_text.split(",") if part.strip()}
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        mesh = None
    active: set[str] = set()
    inactive: set[str] = set()
    if mesh is not None and kind in {"stage_active", "stage_inactive"}:
        active, _label = self._stage_visual_state(mesh)
        inactive = {element.id for element in mesh.elements} - active
    material_map: dict[str, str] = {}
    if mesh is not None:
        material_map = {str(element.id): str(element.material) for element in mesh.elements}
    set_nodes: set[str] = set()
    set_elements: set[str] = set()
    block_elements: set[str] = set()
    if mesh is not None and kind == "set":
        if value_text:
            mesh_cfg = self._mesh_cfg()
            node_sets = self._mapping(mesh_cfg.get("node_sets", {}))
            element_sets = self._mapping(mesh_cfg.get("element_sets", {}))
            for name in wanted or set(node_sets) | set(element_sets):
                set_nodes.update(str(nid) for nid in self._ensure_list(node_sets.get(name, [])))
                set_elements.update(str(eid) for eid in self._ensure_list(element_sets.get(name, [])))
        else:
            set_nodes, set_elements, _label = self._selected_set_targets(mesh)
    if mesh is not None and kind == "block":
        mesh_cfg = self._mesh_cfg()
        blocks = self._mapping(mesh_cfg.get("blocks", {}))
        names = wanted or set(blocks)
        for name in names:
            raw = blocks.get(name)
            if not isinstance(raw, Mapping):
                continue
            block_elements.update(str(eid) for eid in self._ensure_list(raw.get("elements", raw.get("element_ids", []))))
        if not block_elements:
            element_sets = self._mapping(mesh_cfg.get("element_sets", {}))
            for name in names:
                block_elements.update(str(eid) for eid in self._ensure_list(element_sets.get(name, [])))
    stage_targets = self._current_stage_condition_targets(mesh) if kind == "stage_condition" and mesh is not None else {"nodes": set(), "edges": set(), "elements": set()}

    def matches(_item: QGraphicsItem, data: Mapping[str, Any]) -> bool:
        item_kind = str(data.get("kind", ""))
        ident = str(data.get("id", ""))
        if kind == "nodes":
            return item_kind == "node" and (not wanted or "all" in wanted or ident in wanted)
        if kind == "elements":
            return item_kind == "element" and (not wanted or "all" in wanted or ident in wanted)
        if kind == "edges":
            edge_nodes = [str(nid) for nid in self._ensure_list(data.get("nodes", []))]
            edge_text = "-".join(edge_nodes)
            return item_kind == "edge" and (not wanted or "all" in wanted or ident in wanted or edge_text in wanted)
        if kind == "material":
            return item_kind == "element" and (not wanted or material_map.get(ident) in wanted)
        if kind == "block":
            return item_kind == "element" and ident in block_elements
        if kind == "stage_active":
            return item_kind == "element" and ident in active
        if kind == "stage_inactive":
            return item_kind == "element" and ident in inactive
        if kind == "set":
            return (item_kind == "node" and ident in set_nodes) or (item_kind == "element" and ident in set_elements)
        if kind == "stage_condition":
            if item_kind == "node" and ident in stage_targets["nodes"]:
                return True
            if item_kind == "element" and ident in stage_targets["elements"]:
                return True
            if item_kind == "edge":
                edge_nodes = tuple(str(nid) for nid in self._ensure_list(data.get("nodes", [])))
                return edge_nodes in stage_targets["edges"] or tuple(reversed(edge_nodes)) in stage_targets["edges"]
        return False

    count = self._preview_select_items(matches, mode=mode)
    self._record_selection_history(f"条件選択:{kind}:{value_text or 'current'}")
    self.statusBar().showMessage(f"条件選択 {count}件")
    return count
def select_by_expression(owner: Any, qt: Mapping[str, Any], expression: str, *, mode: str = "replace") -> int:
    _bind_qt(qt)
    self = owner
    expression = expression.strip()
    if not expression:
        return 0
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        QMessageBox.warning(self, "GeoFEM", f"条件式を読めません: {exc}")
        return 0
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        mesh = None
    material_map = {str(element.id): str(element.material) for element in mesh.elements} if mesh is not None else {}
    type_map = {str(element.id): str(element.type) for element in mesh.elements} if mesh is not None else {}
    active: set[str] = set()
    if mesh is not None:
        active, _label = self._stage_visual_state(mesh)
    set_membership: dict[str, list[str]] = {}
    mesh_cfg = self._mesh_cfg()
    for set_kind, raw_sets in (("node", self._mapping(mesh_cfg.get("node_sets", {}))), ("element", self._mapping(mesh_cfg.get("element_sets", {})))):
        for set_name, values in raw_sets.items():
            for ident in self._ensure_list(values):
                set_membership.setdefault(f"{set_kind}:{ident}", []).append(str(set_name))
    block_membership: dict[str, list[str]] = {}
    for block_name, raw in self._mapping(mesh_cfg.get("blocks", {})).items():
        if isinstance(raw, Mapping):
            for eid in self._ensure_list(raw.get("elements", raw.get("element_ids", []))):
                block_membership.setdefault(str(eid), []).append(str(block_name))
    node_xy: dict[str, tuple[float, float]] = {}
    if mesh is not None:
        for index, nid in enumerate(mesh.node_ids):
            node_xy[str(nid)] = (float(mesh.coords[index, 0]), float(mesh.coords[index, 1]))

    def context(data: Mapping[str, Any], selected: bool) -> dict[str, Any]:
        item_kind = str(data.get("kind", ""))
        ident = str(data.get("id", ""))
        x = y = None
        if item_kind == "node" and ident in node_xy:
            x, y = node_xy[ident]
        return {
            "kind": item_kind,
            "id": ident,
            "material": material_map.get(ident, str(data.get("material", ""))),
            "type": type_map.get(ident, str(data.get("type", ""))),
            "stage_active": item_kind == "element" and ident in active,
            "stage_inactive": item_kind == "element" and mesh is not None and ident not in active,
            "selected": selected,
            "sets": set_membership.get(f"{item_kind}:{ident}", []),
            "blocks": block_membership.get(ident, []),
            "x": x,
            "y": y,
        }

    try:
        count = self._preview_select_items(lambda item, data: bool(self._eval_selection_expression(parsed.body, context(data, item.isSelected()))), mode=mode)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"条件式が不正です: {exc}")
        return 0
    self._record_selection_history(f"条件式:{expression}")
    self.statusBar().showMessage(f"条件式選択 {count}件")
    return count
def _eval_selection_expression(owner: Any, qt: Mapping[str, Any], node: ast.AST, ctx: Mapping[str, Any]) -> Any:
    _bind_qt(qt)
    self = owner
    if isinstance(node, ast.BoolOp):
        values = [bool(self._eval_selection_expression(value, ctx)) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ValueError("and/or以外の論理演算は使えません。")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(self._eval_selection_expression(node.operand, ctx))
    if isinstance(node, ast.Compare):
        left = self._eval_selection_expression(node.left, ctx)
        for op, comparator in zip(node.ops, node.comparators):
            right = self._eval_selection_expression(comparator, ctx)
            ok = self._compare_selection_values(left, op, right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise ValueError(f"未知の変数 '{node.id}'")
        return ctx[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [self._eval_selection_expression(item, ctx) for item in node.elts]
    raise ValueError("条件式に使えるのは and/or/not, 比較, in, リスト, 文字列, 数値だけです。")
def _compare_selection_values(owner: Any, qt: Mapping[str, Any], left: Any, op: ast.cmpop, right: Any) -> bool:
    _bind_qt(qt)
    self = owner
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    if isinstance(op, ast.Gt):
        return left is not None and right is not None and left > right
    if isinstance(op, ast.GtE):
        return left is not None and right is not None and left >= right
    if isinstance(op, ast.Lt):
        return left is not None and right is not None and left < right
    if isinstance(op, ast.LtE):
        return left is not None and right is not None and left <= right
    raise ValueError("未対応の比較演算子です。")
def _current_stage_condition_targets(owner: Any, qt: Mapping[str, Any], mesh: Any) -> dict[str, set[Any]]:
    _bind_qt(qt)
    self = owner
    row = self._selected_stage_row()
    if row is None:
        return {"nodes": set(), "edges": set(), "elements": set()}
    stages = self._stages()
    if not (0 <= row < len(stages)):
        return {"nodes": set(), "edges": set(), "elements": set()}
    stage = stages[row]
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    elements = self._stage_element_targets(mesh, stage)
    for raw in self._ensure_list(stage.get("boundary_conditions", [])):
        if isinstance(raw, Mapping):
            nodes.update(str(nid) for nid in self._node_targets_for_spec(mesh, raw))
    for raw in self._ensure_list(stage.get("loads", [])):
        if not isinstance(raw, Mapping):
            continue
        nodes.update(str(nid) for nid in self._node_targets_for_spec(mesh, raw))
        edges.update(self._edge_targets_for_spec(mesh, raw))
    hydro_specs: list[Any] = []
    hydro = stage.get("hydro", stage.get("consolidation", stage.get("hydraulic_conditions", [])))
    if isinstance(hydro, Mapping):
        for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs"):
            hydro_specs.extend(self._ensure_list(hydro.get(key, [])))
    else:
        hydro_specs.extend(self._ensure_list(hydro))
    for raw in hydro_specs:
        if not isinstance(raw, Mapping):
            continue
        nodes.update(str(nid) for nid in self._node_targets_for_spec(mesh, raw))
        edges.update(self._edge_targets_for_spec(mesh, raw))
    for raw in self._ensure_list(stage.get("mpc_constraints", [])):
        if isinstance(raw, Mapping):
            nodes.update(str(nid) for nid in self._ensure_list(raw.get("nodes", [])))
    return {"nodes": nodes, "edges": edges, "elements": elements}
def _record_selection_history(owner: Any, qt: Mapping[str, Any], label: str) -> None:
    _bind_qt(qt)
    self = owner
    entities = self._selected_preview_entities()
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "nodes": sorted(entities["nodes"], key=self._natural_sort_key),
        "elements": sorted(entities["elements"], key=self._natural_sort_key),
        "edges": [list(edge) for edge in entities["edges"]],
    }
    if self.selection_history_index < len(self.selection_history) - 1:
        self.selection_history = self.selection_history[: self.selection_history_index + 1]
    self.selection_history.append(entry)
    self.selection_history = self.selection_history[-30:]
    self.selection_history_index = len(self.selection_history) - 1
    self.cfg["selection_history"] = list(self.selection_history)
    self._refresh_selection_history_table()
def _refresh_selection_history_table(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    label = getattr(self, "selection_history_label", None)
    if label is not None:
        if getattr(self, "gui_locale", "ja") == "en":
            label.setText(f"Selection History {len(self.selection_history)}")
        else:
            label.setText(f"選択履歴 {len(self.selection_history)}")
    table = getattr(self, "selection_history_table", None)
    if table is None:
        return
    table.blockSignals(True)
    table.setRowCount(len(self.selection_history))
    for row, entry in enumerate(self.selection_history):
        values = [
            str(row + 1),
            str(entry.get("label", "")),
            str(len(self._ensure_list(entry.get("nodes", [])))),
            str(len(self._ensure_list(entry.get("edges", [])))),
            str(len(self._ensure_list(entry.get("elements", [])))),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if row == self.selection_history_index:
                item.setBackground(QBrush(QColor("#fff3cd")))
            table.setItem(row, col, item)
    table.blockSignals(False)
    if 0 <= self.selection_history_index < table.rowCount():
        table.selectRow(self.selection_history_index)
def restore_selection_history(owner: Any, qt: Mapping[str, Any], index: int) -> None:
    _bind_qt(qt)
    self = owner
    if not (0 <= index < len(self.selection_history)):
        return
    entry = self.selection_history[index]
    self.scene.clearSelection()
    nodes = {str(value) for value in self._ensure_list(entry.get("nodes", []))}
    elements = {str(value) for value in self._ensure_list(entry.get("elements", []))}
    edge_keys = {tuple(str(item) for item in self._ensure_list(edge)[:2]) for edge in self._ensure_list(entry.get("edges", []))}
    for item in self.scene.items():
        data = item.data(0)
        if not isinstance(data, Mapping):
            continue
        kind = data.get("kind")
        ident = str(data.get("id", ""))
        if kind == "node" and ident in nodes:
            item.setSelected(True)
        elif kind == "element" and ident in elements:
            item.setSelected(True)
        elif kind == "edge":
            edge_nodes = tuple(str(nid) for nid in self._ensure_list(data.get("nodes", []))[:2])
            if edge_nodes in edge_keys or tuple(reversed(edge_nodes)) in edge_keys:
                item.setSelected(True)
    self.selection_history_index = index
    self._refresh_selection_history_table()
    self.statusBar().showMessage(f"選択履歴を復元: {entry.get('label', '')}")
def undo_selection(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    if self.selection_history_index <= 0:
        return
    self.restore_selection_history(self.selection_history_index - 1)
def redo_selection(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    if self.selection_history_index >= len(self.selection_history) - 1:
        return
    self.restore_selection_history(self.selection_history_index + 1)
def save_named_selection(owner: Any, qt: Mapping[str, Any], name: str | None = None) -> None:
    _bind_qt(qt)
    self = owner
    entities = self._selected_preview_entities()
    if not entities["nodes"] and not entities["edges"] and not entities["elements"]:
        QMessageBox.information(self, "GeoFEM", "保存する選択がありません。")
        return
    if name is None:
        name, ok = QInputDialog.getText(self, "名前付き選択保存", "名前", text=f"selection_{len(self._mapping(self.cfg.get('named_selections', {}))) + 1}")
        if not ok:
            return
    name = str(name).strip()
    if not name:
        return
    named = self.cfg.setdefault("named_selections", {})
    if not isinstance(named, dict):
        named = {}
        self.cfg["named_selections"] = named
    named[name] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "nodes": sorted(entities["nodes"], key=self._natural_sort_key),
        "edges": [list(edge) for edge in entities["edges"]],
        "elements": sorted(entities["elements"], key=self._natural_sort_key),
    }
    self._refresh_named_selection_table()
    self._after_form_change(f"名前付き選択を保存しました: {name}")
def _refresh_named_selection_table(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    table = getattr(self, "named_selection_table", None)
    if table is None:
        return
    named = self._mapping(self.cfg.get("named_selections", {}))
    table.setRowCount(len(named))
    for row, name in enumerate(sorted(named)):
        entry = self._mapping(named.get(name, {}))
        values = [
            name,
            str(entry.get("time", "")),
            str(len(self._ensure_list(entry.get("nodes", [])))),
            str(len(self._ensure_list(entry.get("edges", [])))),
            str(len(self._ensure_list(entry.get("elements", [])))),
        ]
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(value))
def restore_named_selection_by_row(owner: Any, qt: Mapping[str, Any], row: int) -> None:
    _bind_qt(qt)
    self = owner
    table = getattr(self, "named_selection_table", None)
    if table is None or row < 0:
        return
    item = table.item(row, 0)
    if item is not None:
        self.restore_named_selection(item.text())
def restore_named_selection(owner: Any, qt: Mapping[str, Any], name: str) -> None:
    _bind_qt(qt)
    self = owner
    entry = self._mapping(self._mapping(self.cfg.get("named_selections", {})).get(name, {}))
    if not entry:
        return
    self.selection_history.append(
        {
            "time": str(entry.get("time", "")),
            "label": f"名前付き:{name}",
            "nodes": list(self._ensure_list(entry.get("nodes", []))),
            "edges": list(self._ensure_list(entry.get("edges", []))),
            "elements": list(self._ensure_list(entry.get("elements", []))),
        }
    )
    self.selection_history_index = len(self.selection_history) - 1
    self.restore_selection_history(self.selection_history_index)
def compare_named_selection_dialog(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    names = sorted(self._mapping(self.cfg.get("named_selections", {})))
    if len(names) < 2:
        QMessageBox.information(self, "GeoFEM", "比較する名前付き選択が2つ以上必要です。")
        return
    left, ok = QInputDialog.getItem(self, "選択比較", "A", names, 0, False)
    if not ok:
        return
    right, ok = QInputDialog.getItem(self, "選択比較", "B", names, 1 if len(names) > 1 else 0, False)
    if not ok:
        return
    result = self.compare_named_selections(left, right)
    self.result_view.setPlainText(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    self.tabs.setCurrentWidget(self.result_view)
def compare_named_selections(owner: Any, qt: Mapping[str, Any], left: str, right: str) -> dict[str, Any]:
    _bind_qt(qt)
    self = owner
    named = self._mapping(self.cfg.get("named_selections", {}))
    a = self._mapping(named.get(left, {}))
    b = self._mapping(named.get(right, {}))
    result: dict[str, Any] = {"left": left, "right": right}
    for key in ("nodes", "elements"):
        aset = {str(value) for value in self._ensure_list(a.get(key, []))}
        bset = {str(value) for value in self._ensure_list(b.get(key, []))}
        result[key] = {
            "only_left": sorted(aset - bset, key=self._natural_sort_key),
            "only_right": sorted(bset - aset, key=self._natural_sort_key),
            "common": sorted(aset & bset, key=self._natural_sort_key),
        }
    aedges = {tuple(str(item) for item in self._ensure_list(edge)[:2]) for edge in self._ensure_list(a.get("edges", []))}
    bedges = {tuple(str(item) for item in self._ensure_list(edge)[:2]) for edge in self._ensure_list(b.get("edges", []))}
    result["edges"] = {
        "only_left": [list(edge) for edge in sorted(aedges - bedges)],
        "only_right": [list(edge) for edge in sorted(bedges - aedges)],
        "common": [list(edge) for edge in sorted(aedges & bedges)],
    }
    return result
def save_selection_comparison(owner: Any, qt: Mapping[str, Any], left: str, right: str, name: str | None = None) -> dict[str, Any]:
    _bind_qt(qt)
    self = owner
    result = self.compare_named_selections(left, right)
    if name is None:
        name = f"{left}_vs_{right}"
    comparisons = self.cfg.setdefault("selection_comparisons", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
        self.cfg["selection_comparisons"] = comparisons
    result["time"] = datetime.now().isoformat(timespec="seconds")
    comparisons[str(name)] = result
    self._refresh_selection_compare_table()
    self._after_form_change(f"選択比較を保存しました: {name}")
    return result
def _refresh_selection_compare_table(owner: Any, qt: Mapping[str, Any]) -> None:
    _bind_qt(qt)
    self = owner
    table = getattr(self, "selection_compare_table", None)
    if table is None:
        return
    comparisons = self._mapping(self.cfg.get("selection_comparisons", {}))
    table.setRowCount(len(comparisons))
    for row, name in enumerate(sorted(comparisons)):
        raw = self._mapping(comparisons.get(name, {}))
        nodes = self._mapping(raw.get("nodes", {}))
        elements = self._mapping(raw.get("elements", {}))
        values = [
            name,
            str(raw.get("left", "")),
            str(raw.get("right", "")),
            str(len(self._ensure_list(nodes.get("only_left", [])))),
            str(len(self._ensure_list(nodes.get("only_right", [])))),
            str(len(self._ensure_list(elements.get("only_left", [])))),
            str(len(self._ensure_list(elements.get("only_right", [])))),
            str(raw.get("time", "")),
        ]
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(value))
def save_current_selection_as_set(owner: Any, qt: Mapping[str, Any], name: str | None = None) -> None:
    _bind_qt(qt)
    self = owner
    entities = self._selected_preview_entities()
    if not entities["nodes"] and not entities["elements"]:
        QMessageBox.information(self, "GeoFEM", "登録する節点または要素を選択してください。")
        return
    if name is None:
        default = f"selection_{len(self.selection_history) + 1}"
        name, ok = QInputDialog.getText(self, "選択set登録", "set名", text=default)
        if not ok:
            return
    name = str(name).strip()
    if not name:
        return
    mesh_cfg = self._mesh_cfg()
    if entities["nodes"]:
        node_sets = mesh_cfg.setdefault("node_sets", {})
        if not isinstance(node_sets, dict):
            node_sets = {}
            mesh_cfg["node_sets"] = node_sets
        node_sets[name] = sorted(entities["nodes"], key=self._natural_sort_key)
    if entities["elements"]:
        element_sets = mesh_cfg.setdefault("element_sets", {})
        if not isinstance(element_sets, dict):
            element_sets = {}
            mesh_cfg["element_sets"] = element_sets
        element_sets[name] = sorted(entities["elements"], key=self._natural_sort_key)
    self._record_selection_history(f"set登録:{name}")
    self._after_form_change(f"選択set '{name}' を登録しました")
def _scene_to_model(owner: Any, qt: Mapping[str, Any], point: QPointF) -> tuple[float, float]:
    _bind_qt(qt)
    self = owner
    scale = self.preview_scale if abs(self.preview_scale) > 1.0e-30 else 1.0
    return (float(point.x()) - self.preview_ox) / scale, (self.preview_oy - float(point.y())) / scale
def _snap_tolerance_model(owner: Any, qt: Mapping[str, Any]) -> float:
    _bind_qt(qt)
    self = owner
    scale = self.preview_scale if abs(self.preview_scale) > 1.0e-30 else 1.0
    return max(8.0 / scale, 1.0e-8)
def _snap_model_point(owner: Any, qt: Mapping[str, Any], x: float, y: float) -> tuple[float, float, str | None]:
    _bind_qt(qt)
    self = owner
    if not self.snap_enabled.isChecked():
        return x, y, None
    snap_kind = str(self.snap_type.currentData() or "all") if hasattr(self, "snap_type") else "all"
    if snap_kind == "grid":
        try:
            grid = float(self.snap_grid_size.text())
        except (AttributeError, ValueError):
            grid = 1.0
        if grid > 0.0:
            return round(x / grid) * grid, round(y / grid) * grid, f"grid:{grid:g}"
    tolerance = self._snap_tolerance_model()
    best: tuple[float, str, float, float] | None = None
    points = self._snap_candidates(snap_kind)
    for label, px, py in points:
        dist = math.hypot(x - px, y - py)
        if dist <= tolerance and (best is None or dist < best[0]):
            best = (dist, label, px, py)
    if best is None:
        return x, y, None
    return best[2], best[3], best[1]
def _snap_candidates(owner: Any, qt: Mapping[str, Any], kind: str) -> list[tuple[str, float, float]]:
    _bind_qt(qt)
    self = owner
    try:
        if kind == "nodes":
            return self._mesh_node_points()
        if kind == "geometry":
            return self._geometry_snap_points()
        if kind == "intersections":
            return self._intersection_snap_points()
        points = self._mesh_node_points()
        points.extend(self._geometry_snap_points())
        points.extend(self._intersection_snap_points())
        return points
    except Exception:
        return []
def _selected_element_ids(owner: Any, qt: Mapping[str, Any]) -> list[str]:
    _bind_qt(qt)
    self = owner
    ids: list[str] = []
    for item in self.scene.selectedItems():
        data = item.data(0)
        if isinstance(data, dict) and data.get("kind") == "element":
            ids.append(str(data.get("id")))
    return ids
def _selected_model_points(owner: Any, qt: Mapping[str, Any]) -> list[tuple[float, float]]:
    _bind_qt(qt)
    self = owner
    points: list[tuple[float, float]] = []
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        mesh = None
    node_lookup: dict[str, tuple[float, float]] = {}
    if mesh is not None:
        node_lookup = {
            str(nid): (float(mesh.coords[index, 0]), float(mesh.coords[index, 1]))
            for index, nid in enumerate(mesh.node_ids)
        }
    geometry = self._geometry_cfg()
    lines = {str(raw.get("id", "")): raw for raw in geometry.get("lines", []) if isinstance(raw, Mapping)}
    for item in self.scene.selectedItems():
        data = item.data(0)
        if not isinstance(data, dict):
            continue
        kind = data.get("kind")
        if kind == "node" and str(data.get("id")) in node_lookup:
            points.append(node_lookup[str(data.get("id"))])
        elif kind == "geometry_endpoint":
            raw = lines.get(str(data.get("id")))
            if raw is None:
                continue
            key = "start" if data.get("endpoint") == "start" else "end"
            try:
                points.append(self._xy_pair(raw.get(key, raw.get("p1" if key == "start" else "p2", [0.0, 0.0]))))
            except ValueError:
                continue
        elif kind == "mesh_control_point":
            for raw in self._ensure_list(self._mapping(self.cfg.get("mesh", {})).get("control_points", [])):
                if isinstance(raw, Mapping) and str(raw.get("id")) == str(data.get("id")):
                    try:
                        points.append(self._xy_pair(raw.get("point", [0.0, 0.0])))
                    except ValueError:
                        pass
                    break
    return points
def _selected_geometry_line(owner: Any, qt: Mapping[str, Any]) -> dict[str, Any] | None:
    _bind_qt(qt)
    self = owner
    selected_ids: list[str] = []
    for item in self.scene.selectedItems():
        data = item.data(0)
        if isinstance(data, dict) and data.get("kind") == "geometry_line" and data.get("id") is not None:
            selected_ids.append(str(data.get("id")))
    if not selected_ids:
        return None
    return self._geometry_line_by_id(selected_ids[0])
def _geometry_line_by_id(owner: Any, qt: Mapping[str, Any], line_id: str) -> dict[str, Any] | None:
    _bind_qt(qt)
    self = owner
    lines = self._geometry_cfg().get("lines", [])
    if not isinstance(lines, list):
        return None
    for line in lines:
        if isinstance(line, dict) and str(line.get("id")) == line_id:
            return line
    return None
def _mesh_node_points(owner: Any, qt: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    _bind_qt(qt)
    self = owner
    from geofem_app.fem2d import mesh_from_config

    mesh = mesh_from_config(self.cfg)
    return [(f"node:{nid}", float(mesh.coords[i, 0]), float(mesh.coords[i, 1])) for i, nid in enumerate(mesh.node_ids)]
def _geometry_snap_points(owner: Any, qt: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    _bind_qt(qt)
    self = owner
    points: list[tuple[str, float, float]] = []
    geometry = self._mapping(self.cfg.get("geometry", {}))
    lines = geometry.get("lines", [])
    if isinstance(lines, list):
        for raw in lines:
            if not isinstance(raw, Mapping):
                continue
            try:
                x1, y1 = self._xy_pair(raw.get("start", raw.get("p1", [0.0, 0.0])))
                x2, y2 = self._xy_pair(raw.get("end", raw.get("p2", [0.0, 0.0])))
            except ValueError:
                continue
            lid = str(raw.get("id", "line"))
            points.append((f"{lid}:start", x1, y1))
            points.append((f"{lid}:end", x2, y2))
    regions = geometry.get("regions", [])
    if isinstance(regions, list):
        for index, raw in enumerate(regions, start=1):
            if not isinstance(raw, Mapping):
                continue
            for pindex, point in enumerate(self._ensure_list(raw.get("points", [])), start=1):
                try:
                    px, py = self._xy_pair(point)
                except ValueError:
                    continue
                points.append((f"region_{index}:{pindex}", px, py))
    tunnels = geometry.get("tunnels", [])
    if isinstance(tunnels, list):
        for raw in tunnels:
            if not isinstance(raw, Mapping):
                continue
            try:
                cx, cy = self._xy_pair(raw.get("center", [0.0, 0.0]))
                radius = float(raw.get("radius", 0.0))
                segments = max(8, int(raw.get("segments", 24)))
            except (TypeError, ValueError):
                continue
            tid = str(raw.get("id", "tunnel"))
            points.append((f"{tid}:center", cx, cy))
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                points.append((f"{tid}:{i + 1}", cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    for raw in geometry.get("points", []) if isinstance(geometry.get("points", []), list) else []:
        if not isinstance(raw, Mapping):
            continue
        try:
            x, y = self._xy_pair(raw.get("point", raw.get("position", [raw.get("x", 0.0), raw.get("y", 0.0)])))
        except ValueError:
            continue
        points.append((f"point:{raw.get('id', len(points) + 1)}", x, y))
    for raw in geometry.get("annotations", []) if isinstance(geometry.get("annotations", []), list) else []:
        if not isinstance(raw, Mapping):
            continue
        try:
            x, y = self._xy_pair(raw.get("point", raw.get("position", [0.0, 0.0])))
        except ValueError:
            continue
        points.append((f"note:{raw.get('id', len(points) + 1)}", x, y))
    for raw in self._ensure_list(self._mapping(self.cfg.get("mesh", {})).get("control_points", [])):
        if not isinstance(raw, Mapping):
            continue
        try:
            x, y = self._xy_pair(raw.get("point", raw.get("position", [raw.get("x", 0.0), raw.get("y", 0.0)])))
        except ValueError:
            continue
        points.append((f"control:{raw.get('id', len(points) + 1)}", x, y))
    return points
def _intersection_snap_points(owner: Any, qt: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    _bind_qt(qt)
    self = owner
    geometry = self._mapping(self.cfg.get("geometry", {}))
    raw_lines = geometry.get("lines", [])
    segments: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    if isinstance(raw_lines, list):
        for raw in raw_lines:
            if not isinstance(raw, Mapping):
                continue
            try:
                start = self._xy_pair(raw.get("start", raw.get("p1", [0.0, 0.0])))
                end = self._xy_pair(raw.get("end", raw.get("p2", [0.0, 0.0])))
            except ValueError:
                continue
            segments.append((str(raw.get("id", f"line_{len(segments) + 1}")), start, end))
    points: list[tuple[str, float, float]] = []
    for i, (lid_a, a0, a1) in enumerate(segments):
        for lid_b, b0, b1 in segments[i + 1 :]:
            hit = self._segment_intersection(a0, a1, b0, b1)
            if hit is None:
                continue
            ta, tb, x, y = hit
            if -1.0e-9 <= ta <= 1.0 + 1.0e-9 and -1.0e-9 <= tb <= 1.0 + 1.0e-9:
                points.append((f"intersection:{lid_a}-{lid_b}", x, y))
    return points
def _model_points(owner: Any, qt: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    _bind_qt(qt)
    self = owner
    points = self._mesh_node_points()
    points.extend(self._geometry_snap_points())
    points.extend(self._intersection_snap_points())
    return points
def _nearest_node_id(owner: Any, qt: Mapping[str, Any], mesh: Any, x: float, y: float) -> str:
    _bind_qt(qt)
    self = owner
    best_id = ""
    best_dist = math.inf
    for i, nid in enumerate(mesh.node_ids):
        dx = float(mesh.coords[i, 0]) - x
        dy = float(mesh.coords[i, 1]) - y
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_id = str(nid)
    if not best_id:
        raise ValueError("mesh has no nodes")
    return best_id

