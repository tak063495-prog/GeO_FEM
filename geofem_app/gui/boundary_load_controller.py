"""Boundary and load GUI operation controller functions split from MainWindow.

The owner keeps widgets, configuration, and selection state; this module owns
Boundary/Load table synchronization, selection-to-condition actions, load cases,
and axisymmetric boundary/load/hydro presets.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import yaml

BOUNDARY_LOAD_CONTROLLER_METHODS = (
    "apply_axisymmetric_standard_presets",
    "_append_axisymmetric_boundary_conditions",
    "apply_axisymmetric_boundary_preset",
    "apply_axisymmetric_load_preset",
    "apply_axisymmetric_hydro_preset",
    "add_boundary_row",
    "remove_selected_boundary_rows",
    "apply_boundary_conditions_panel",
    "add_selected_boundary_condition",
    "add_selected_hydro_boundary_condition",
    "add_selected_mpc_constraints",
    "register_selected_nodes_as_set",
    "_boundary_condition_from_selection",
    "_append_boundary_condition_to_scope",
    "_ensure_up_analysis_fields",
    "_optional_line_float",
    "populate_boundary_table",
    "populate_boundary_condition_tree",
    "select_boundary_condition_tree_item",
    "add_load_case_row",
    "remove_selected_load_case_rows",
    "apply_load_cases_panel",
    "populate_load_case_table",
    "_load_cases",
    "_refresh_load_case_selector",
    "_selected_load_case_name",
    "_attach_load_case",
    "add_load_row",
    "remove_selected_load_rows",
    "apply_loads_panel",
    "add_selected_nodal_load_condition",
    "add_selected_body_load_condition",
    "add_selected_distributed_load_condition",
    "add_panel_earthquake_load",
    "_append_load_to_scope",
    "populate_load_table",
    "add_fix_boundary_condition",
    "add_support_preset",
    "add_prescribed_displacement",
    "add_mpc_constraint",
    "add_nodal_load",
    "add_edge_load",
    "add_gravity_load",
    "add_nodal_pore_pressure",
    "add_hydro_boundary",
    "_loads_from_table",
    "_add_boundary_mapping_to_table",
    "_add_load_mapping_to_table",
)


def boundary_load_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.boundary_load_controller.v1",
        "method_count": len(BOUNDARY_LOAD_CONTROLLER_METHODS),
        "methods": list(BOUNDARY_LOAD_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner cfg/widgets for Boundary and Load operations; MainWindow delegates input-domain actions",
    }


def apply_axisymmetric_standard_presets(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    self._ensure_axisymmetric_analysis()
    try:
        self._axisymmetric_reference_sets(update=True)
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"Axisymmetric standard preset failed: {exc}")
        return
    scope_widget = getattr(self, "boundary_batch_scope", None)
    scope = str(scope_widget.currentData() if scope_widget is not None else "global") or "global"
    self._append_axisymmetric_boundary_conditions(
        [
            {"set": "axisymmetric_axis", "ux": 0.0, "support_type": "axisymmetric_symmetry_axis", "coordinate_system": "axisymmetric_rz", "source": "axisym_gui_preset"},
            {"set": "axisymmetric_bottom", "uy": 0.0, "support_type": "axisymmetric_axial_roller", "coordinate_system": "axisymmetric_rz", "source": "axisym_gui_preset"},
        ],
        scope,
        "Axisymmetric standard boundary presets applied",
    )

def _append_axisymmetric_boundary_conditions(owner: Any, qt: Mapping[str, Any], bcs: list[dict[str, Any]], scope: str, message: str) -> None:
    self = owner
    if not bcs:
        return
    if scope == "stage":
        stages, row, stage = self._selected_stage_context()
        self._list_value(stage, "boundary_conditions").extend(bcs)
        stage.setdefault("selection_history", []).append({"kind": "axisymmetric_boundary_preset", "count": len(bcs), "at": datetime.now().isoformat(timespec="seconds")})
        stages[row] = stage
        self.cfg["stages"] = stages
        self._after_form_change(message)
        self.stage_table.selectRow(row)
        return
    existing = self.cfg.setdefault("boundary_conditions", [])
    if not isinstance(existing, list):
        existing = []
        self.cfg["boundary_conditions"] = existing
    existing.extend(bcs)
    self._after_form_change(message)

def apply_axisymmetric_boundary_preset(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    preset = str(self.axisym_boundary_preset.currentData() or "symmetry_axis")
    try:
        self._ensure_axisymmetric_analysis()
        self._axisymmetric_reference_sets(update=True)
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"Axisymmetric boundary preset failed: {exc}")
        return
    base = {"coordinate_system": "axisymmetric_rz", "source": "axisym_gui_preset"}
    if preset == "bottom_axial_roller":
        bcs = [dict(base, set="axisymmetric_bottom", uy=0.0, support_type="axisymmetric_axial_roller")]
    elif preset == "outer_radial_roller":
        bcs = [dict(base, set="axisymmetric_outer_radius", ux=0.0, support_type="axisymmetric_radial_roller")]
    elif preset == "minimal_support":
        bcs = [
            dict(base, set="axisymmetric_axis", ux=0.0, support_type="axisymmetric_symmetry_axis"),
            dict(base, set="axisymmetric_bottom", uy=0.0, support_type="axisymmetric_axial_roller"),
        ]
    else:
        bcs = [dict(base, set="axisymmetric_axis", ux=0.0, support_type="axisymmetric_symmetry_axis")]
    scope = str(self.boundary_batch_scope.currentData() or "global") if hasattr(self, "boundary_batch_scope") else "global"
    self._append_axisymmetric_boundary_conditions(bcs, scope, f"Axisymmetric boundary preset applied: {preset}")

def apply_axisymmetric_load_preset(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    preset = str(self.axisym_load_preset.currentData() or "outer_radial_pressure")
    try:
        self._ensure_axisymmetric_analysis()
        self._axisymmetric_reference_sets(update=True)
        value = self._float_text(self.axisym_load_value.text().strip() or "0.0", "axisym load value")
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"Axisymmetric load preset failed: {exc}")
        return
    base = {"coordinate_system": "axisymmetric_rz", "axisymmetric_measure": "2*pi*r", "source": "axisym_gui_preset"}
    if preset == "top_axial_surcharge":
        load = dict(base, edges="axisymmetric_top", tx=0.0, ty=-abs(value), component="axial_surface_load")
    elif preset == "self_weight":
        load = dict(base, type="gravity", gx=0.0, gy=-1.0, scale=value, component="axisymmetric_self_weight")
    elif preset == "outer_ring_force":
        load = dict(base, set="axisymmetric_outer_radius", fx=value, fy=0.0, component="radial_ring_nodal_force", axisymmetric_total_ring_force=True)
    else:
        load = dict(base, edges="axisymmetric_outer_radius", tx=-abs(value), ty=0.0, component="radial_surface_pressure")
    load = self._attach_load_case(load)
    scope = str(self.load_batch_scope.currentData() or "global") if hasattr(self, "load_batch_scope") else "global"
    self._append_load_to_scope(load, scope, f"Axisymmetric load preset applied: {preset}")

def apply_axisymmetric_hydro_preset(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    preset = str(self.axisym_hydro_preset.currentData() or "top_drained")
    try:
        self._ensure_axisymmetric_analysis()
        self._axisymmetric_reference_sets(update=True)
        value = self._float_text(self.axisym_hydro_value.text().strip() or "0.0", "axisym hydro value")
        beta = self._float_text(self.axisym_hydro_beta.text().strip() or "1.0", "axisym hydro beta")
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"Axisymmetric hydro preset failed: {exc}")
        return
    stages, row, stage = self._selected_stage_context()
    if str(stage.get("type", "")).lower() not in {"consolidation", "u-p", "u_p", "up", "coupled_consolidation", "riks", "arc_length", "arclength"}:
        stage["type"] = "consolidation"
    self._ensure_up_analysis_fields()
    hydro = stage.setdefault("hydro", {})
    if not isinstance(hydro, dict):
        hydro = {}
        stage["hydro"] = hydro
    base = {"coordinate_system": "axisymmetric_rz", "axisymmetric_measure": "2*pi*r", "source": "axisym_gui_preset"}
    if preset == "outer_pressure":
        self._list_value(hydro, "pressure_bcs").append(dict(base, set="axisymmetric_outer_radius", pressure=value, component="outer_fixed_pore_pressure"))
    elif preset == "axis_no_flux":
        self._list_value(hydro, "pore_flux_bcs").append(dict(base, edges="axisymmetric_axis", flux=value, component="axis_no_flow"))
    elif preset == "outer_robin":
        self._list_value(hydro, "pore_robin_bcs").append(dict(base, edges="axisymmetric_outer_radius", beta=max(beta, 0.0), pressure=value, component="outer_robin_drain"))
    else:
        self._list_value(hydro, "pressure_bcs").append(dict(base, set="axisymmetric_top", pressure=value, component="top_drained_pressure"))
    stage.setdefault("selection_history", []).append({"kind": "axisymmetric_hydro_preset", "preset": preset, "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"Axisymmetric hydro preset applied: {preset}")
    self.stage_table.selectRow(row)

def add_boundary_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.boundary_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"target": "left", "ux": "0.0", "uy": "0.0", "extra": ""}
    defaults.update(values)
    for col, key in enumerate(["target", "ux", "uy", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def remove_selected_boundary_rows(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    rows = sorted({index.row() for index in self.boundary_table.selectedIndexes()}, reverse=True)
    for row in rows:
        self.boundary_table.removeRow(row)

def apply_boundary_conditions_panel(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    boundary_conditions: list[dict[str, Any]] = []
    try:
        for row in range(self.boundary_table.rowCount()):
            target = self._table_text(self.boundary_table, row, 0).strip()
            ux_text = self._table_text(self.boundary_table, row, 1).strip()
            uy_text = self._table_text(self.boundary_table, row, 2).strip()
            extra = self._yaml_mapping_text(self._table_text(self.boundary_table, row, 3), "boundary extra YAML")
            if not target and not extra:
                continue
            bc = dict(extra)
            if target:
                bc.update(self._target_node_spec(target))
            if ux_text:
                bc["ux"] = self._float_text(ux_text, "ux")
            if uy_text:
                bc["uy"] = self._float_text(uy_text, "uy")
            if not any(key in bc for key in ("ux", "uy", "fixed", "dof", "dofs")):
                raise ValueError(f"境界条件 {row + 1}: ux/uyまたはfixed/dofを指定してください。")
            boundary_conditions.append(bc)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    self.cfg["boundary_conditions"] = boundary_conditions
    self._after_form_change("境界条件表を反映しました")

def add_selected_boundary_condition(owner: Any, qt: Mapping[str, Any], *_args: Any, scope: str | None = None) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    nodes = self._selected_nodes_for_stage_action()
    if not nodes:
        QMessageBox.information(self, "GeoFEM", "モデルビューで節点、辺、または要素を選択してください。")
        return
    restore_payload = {"nodes": list(nodes)}
    try:
        bc = self._boundary_condition_from_selection(nodes)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    target_scope = scope or str(self.boundary_batch_scope.currentData() or "global")
    self._append_boundary_condition_to_scope(bc, target_scope, f"選択節点 {len(nodes)} 件へ境界条件を追加しました")
    self._select_preview_payload(restore_payload)

def add_selected_hydro_boundary_condition(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    kind = str(self.boundary_hydro_kind.currentData() or self.boundary_hydro_kind.currentText() or "pressure")
    try:
        value = self._float_text(self.boundary_hydro_value.text().strip() or "0.0", "水理値")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    edges = self._selected_edges_for_stage_action()
    nodes = self._selected_nodes_for_stage_action(include_elements=True)
    selected_entities = self._selected_preview_entities()
    restore_payload = {
        "nodes": sorted(selected_entities["nodes"], key=self._natural_sort_key),
        "edges": list(selected_entities["edges"]),
        "elements": sorted(selected_entities["elements"], key=self._natural_sort_key),
    }
    if kind == "node_pressure":
        if not nodes:
            QMessageBox.information(self, "GeoFEM", "節点水圧にする節点を選択してください。")
            return
        spec: dict[str, Any] = {"nodes": nodes, "pressure": value, "source": "gui_selection"}
        target_key = "pressure_bcs"
        count = len(nodes)
    else:
        if not edges:
            QMessageBox.information(self, "GeoFEM", "水理境界にする辺または要素境界を選択してください。")
            return
        edge_list = [[a, b] for a, b in edges]
        spec = {"edges": edge_list, "source": "gui_selection"}
        if kind == "flux":
            spec["flux"] = value
            target_key = "pore_flux_bcs"
        elif kind == "robin":
            spec["beta"] = max(value, 0.0)
            spec.setdefault("pressure", 0.0)
            target_key = "pore_robin_bcs"
        else:
            spec["pressure"] = value
            target_key = "pressure_bcs"
        count = len(edges)
    stages, row, stage = self._selected_stage_context()
    if str(stage.get("type", "")).lower() not in {"consolidation", "u-p", "u_p", "up", "coupled_consolidation", "riks", "arc_length", "arclength"}:
        stage["type"] = "consolidation"
    self._ensure_up_analysis_fields()
    hydro = stage.setdefault("hydro", {})
    if not isinstance(hydro, dict):
        hydro = {}
        stage["hydro"] = hydro
    self._list_value(hydro, target_key).append(spec)
    stage.setdefault("selection_history", []).append({"kind": "hydro_boundary", "condition": kind, "count": count, "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択から水理境界 {kind} を {count} 件追加しました")
    self.stage_table.selectRow(row)
    self._select_preview_payload(restore_payload)

def add_selected_mpc_constraints(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    nodes = self._selected_nodes_for_stage_action()
    if len(nodes) < 2:
        QMessageBox.information(self, "GeoFEM", "MPCにはmasterとslaveになる2節点以上を選択してください。")
        return
    restore_payload = {"nodes": list(nodes)}
    master = self.boundary_mpc_master.text().strip() or nodes[0]
    if master not in nodes:
        nodes = [master] + nodes
    slaves = [node for node in nodes if node != master]
    if not slaves:
        QMessageBox.information(self, "GeoFEM", "master以外のslave節点を選択してください。")
        return
    try:
        coefficient = self._float_text(self.boundary_mpc_coefficient.text().strip() or "1.0", "MPC coefficient")
        value = self._float_text(self.boundary_mpc_value.text().strip() or "0.0", "MPC value")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    dof = self._combo_value(self.boundary_mpc_dof, "ux").strip() or "ux"
    method = self._combo_value(self.boundary_mpc_method, "lagrange").strip() or "lagrange"
    stages, row, stage = self._selected_stage_context()
    mpcs = self._list_value(stage, "mpc_constraints")
    for slave in slaves:
        mpcs.append(
            {
                "master": master,
                "slave": slave,
                "dof": dof,
                "coefficient": coefficient,
                "value": value,
                "method": method,
                "source": "gui_selection",
            }
        )
    stage.setdefault("selection_history", []).append({"kind": "mpc", "master": master, "slave_count": len(slaves), "dof": dof, "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択節点からMPC {len(slaves)} 件を追加しました")
    self.stage_table.selectRow(row)
    self._select_preview_payload(restore_payload)

def register_selected_nodes_as_set(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    nodes = self._selected_nodes_for_stage_action()
    if not nodes:
        QMessageBox.information(self, "GeoFEM", "node setにする節点、辺、または要素を選択してください。")
        return
    default = f"selected_nodes_{len(self._mapping(self._mesh_cfg().get('node_sets', {}))) + 1}"
    name, ok = QInputDialog.getText(self, "選択節点→set登録", "node set名", text=default)
    if not ok or not name.strip():
        return
    mesh_cfg = self._mesh_cfg()
    node_sets = mesh_cfg.setdefault("node_sets", {})
    if not isinstance(node_sets, dict):
        node_sets = {}
        mesh_cfg["node_sets"] = node_sets
    node_sets[name.strip()] = nodes
    self._after_form_change(f"選択節点 {len(nodes)} 件をnode set '{name.strip()}' に登録しました")

def _boundary_condition_from_selection(owner: Any, qt: Mapping[str, Any], nodes: list[str]) -> dict[str, Any]:
    self = owner
    kind = str(self.boundary_batch_kind.currentData() or self.boundary_batch_kind.currentText() or "fixed")
    bc: dict[str, Any] = {"nodes": nodes, "source": "gui_selection", "support_type": kind}
    if kind in {"fixed", "pin"}:
        bc["fixed"] = True
    elif kind == "roller_y":
        bc["uy"] = 0.0
    elif kind == "roller_x":
        bc["ux"] = 0.0
    elif kind == "prescribed":
        ux = self._optional_line_float(self.boundary_batch_ux.text(), "ux")
        uy = self._optional_line_float(self.boundary_batch_uy.text(), "uy")
        if ux is None and uy is None:
            raise ValueError("強制変位はuxまたはuyの少なくとも一方を入力してください。")
        if ux is not None:
            bc["ux"] = ux
        if uy is not None:
            bc["uy"] = uy
    else:
        bc["fixed"] = True
    return bc

def _append_boundary_condition_to_scope(owner: Any, qt: Mapping[str, Any], bc: dict[str, Any], scope: str, message: str) -> None:
    self = owner
    if scope == "stage":
        stages, row, stage = self._selected_stage_context()
        self._list_value(stage, "boundary_conditions").append(bc)
        stage.setdefault("selection_history", []).append(
            {
                "kind": "boundary_condition",
                "node_count": len(self._ensure_list(bc.get("nodes", bc.get("node", [])))),
                "at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        stages[row] = stage
        self.cfg["stages"] = stages
        self._after_form_change(message)
        self.stage_table.selectRow(row)
        return
    bcs = self.cfg.setdefault("boundary_conditions", [])
    if not isinstance(bcs, list):
        bcs = []
        self.cfg["boundary_conditions"] = bcs
    bcs.append(bc)
    self._after_form_change(message)

def _ensure_up_analysis_fields(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    analysis = dict(self._mapping(self.cfg.get("analysis", {})))
    fields = {str(field).lower() for field in self._ensure_list(analysis.get("fields", ["u"]))}
    fields.update({"u", "p"})
    analysis["fields"] = sorted(fields)
    self.cfg["analysis"] = analysis

def _optional_line_float(owner: Any, qt: Mapping[str, Any], text: str, label: str) -> float | None:
    self = owner
    raw = text.strip()
    if not raw:
        return None
    return self._float_text(raw, label)

def populate_boundary_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    table = getattr(self, "boundary_table", None)
    if table is None:
        return
    table.setRowCount(0)
    for raw in self._ensure_list(self.cfg.get("boundary_conditions", self.cfg.get("bc", []))):
        if not isinstance(raw, Mapping):
            continue
        target = self._node_spec_text(raw)
        ux = ""
        uy = ""
        if bool(raw.get("fixed", False)):
            ux = "0.0"
            uy = "0.0"
        if "ux" in raw and raw.get("ux") is not None:
            ux = str(raw.get("ux"))
        if "uy" in raw and raw.get("uy") is not None:
            uy = str(raw.get("uy"))
        if str(raw.get("dof", "")).lower() == "ux":
            ux = str(raw.get("value", 0.0))
        elif str(raw.get("dof", "")).lower() == "uy":
            uy = str(raw.get("value", 0.0))
        exclude = {"node", "nodes", "set", "all", "ux", "uy", "fixed"}
        if str(raw.get("dof", "")).lower() in {"ux", "uy"}:
            exclude.update({"dof", "value"})
        extra = {k: v for k, v in raw.items() if k not in exclude}
        if bool(raw.get("fixed", False)):
            extra.pop("fixed", None)
        self.add_boundary_row(
            target=target,
            ux=ux,
            uy=uy,
            extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip(),
        )

def populate_boundary_condition_tree(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QTreeWidgetItem = qt["QTreeWidgetItem"]
    Qt = qt["Qt"]
    tree = getattr(self, "boundary_condition_tree", None)
    if tree is None:
        return
    tree.blockSignals(True)
    tree.clear()
    tree.setColumnCount(2)
    tree.setHeaderLabels(["境界条件", "対象節点"])
    for spec in _boundary_condition_tree_specs(self):
        nodes = list(spec.get("nodes", []))
        edges = list(spec.get("edges", []))
        target_count = f"{len(nodes)}節点"
        if edges:
            target_count = f"{target_count} / {len(edges)}辺"
        item = QTreeWidgetItem([str(spec.get("label", "境界条件")), target_count])
        item.setData(
            0,
            Qt.ItemDataRole.UserRole,
            {
                "kind": "boundary_condition",
                "condition_kind": spec.get("condition_kind", ""),
                "source": spec.get("source", {}),
                "nodes": nodes,
                "edges": edges,
            },
        )
        item.setToolTip(0, str(spec.get("tooltip", "")))
        for node in nodes:
            child = QTreeWidgetItem([f"節点 {node}", ""])
            child.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": "boundary_condition_node",
                    "condition_kind": spec.get("condition_kind", ""),
                    "source": spec.get("source", {}),
                    "nodes": [node],
                    "remove_nodes": [node],
                },
            )
            item.addChild(child)
        for edge in edges:
            edge_nodes = [str(edge[0]), str(edge[1])]
            child = QTreeWidgetItem([f"辺 {edge_nodes[0]}-{edge_nodes[1]}", ""])
            child.setData(
                0,
                Qt.ItemDataRole.UserRole,
                {
                    "kind": "boundary_condition_edge",
                    "condition_kind": spec.get("condition_kind", ""),
                    "source": spec.get("source", {}),
                    "nodes": edge_nodes,
                    "edges": [edge],
                    "remove_edges": [edge],
                },
            )
            item.addChild(child)
        if not nodes and not edges:
            child = QTreeWidgetItem(["対象未解決", str(spec.get("target", ""))])
            item.addChild(child)
        tree.addTopLevelItem(item)
        item.setExpanded(True)
    tree.resizeColumnToContents(0)
    tree.blockSignals(False)

def select_boundary_condition_tree_item(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    Qt = qt["Qt"]
    tree = getattr(self, "boundary_condition_tree", None)
    if tree is None:
        return
    item = tree.currentItem()
    if item is None:
        return
    payload = item.data(0, Qt.ItemDataRole.UserRole)
    if not isinstance(payload, Mapping):
        return
    nodes = [str(node) for node in self._ensure_list(payload.get("nodes", [])) if str(node)]
    if nodes:
        self._select_preview_payload({"nodes": nodes})
        self.statusBar().showMessage(f"境界条件ツリーから節点 {len(nodes)} 件を選択しました")

def _boundary_condition_tree_specs(owner: Any) -> list[dict[str, Any]]:
    self = owner
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        mesh = None
    specs: list[dict[str, Any]] = []

    def add_condition(scope: str, index: int, raw: Mapping[str, Any], *, kind: str = "境界", source: Mapping[str, Any] | None = None) -> None:
        nodes = _nodes_for_condition(self, mesh, raw)
        edges = _edges_for_condition(self, mesh, raw)
        label = f"{scope} {kind}{index}: {_boundary_condition_label(self, raw)}"
        target = self._node_spec_text(raw) if hasattr(self, "_node_spec_text") else ""
        condition_kind = ""
        if hasattr(self, "_boundary_condition_kind_from_spec"):
            condition_kind = self._boundary_condition_kind_from_spec(raw, hydro_key=str((source or {}).get("hydro_key", "")))
        specs.append(
            {
                "label": label,
                "nodes": nodes,
                "edges": edges,
                "target": target,
                "source": dict(source or {}),
                "condition_kind": condition_kind,
                "tooltip": yaml.safe_dump(dict(raw), allow_unicode=True, sort_keys=False).strip(),
            }
        )

    for index, raw in enumerate(self._ensure_list(self.cfg.get("boundary_conditions", self.cfg.get("bc", []))), start=1):
        if isinstance(raw, Mapping):
            add_condition("全体", index, raw, source={"scope": "global_bc", "index": index - 1})
    row = self._selected_stage_row()
    stages = self._stages()
    if row is not None and 0 <= row < len(stages):
        stage = stages[row]
        stage_name = str(stage.get("name", f"Stage-{row + 1}"))
        for index, raw in enumerate(self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))), start=1):
            if isinstance(raw, Mapping):
                add_condition(stage_name, index, raw, source={"scope": "stage_bc", "stage_index": row, "index": index - 1})
        hydro = stage.get("hydro", stage.get("consolidation", {}))
        if isinstance(hydro, Mapping):
            hydro_index = 1
            for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs"):
                for list_index, raw in enumerate(self._ensure_list(hydro.get(key, [])), start=1):
                    if isinstance(raw, Mapping):
                        add_condition(
                            stage_name,
                            hydro_index,
                            raw,
                            kind=f"水理 {key}",
                            source={"scope": "hydro", "stage_index": row, "hydro_key": key, "index": list_index - 1},
                        )
                        hydro_index += 1
        for index, raw in enumerate(self._ensure_list(stage.get("mpc_constraints", stage.get("mpc", []))), start=1):
            if isinstance(raw, Mapping):
                add_condition(stage_name, index, raw, kind="MPC", source={"scope": "stage_mpc", "stage_index": row, "index": index - 1})
    return specs

def _nodes_for_condition(owner: Any, mesh: Any, raw: Mapping[str, Any]) -> list[str]:
    self = owner
    mpc_nodes = {
        str(raw.get(key)).strip()
        for key in ("master", "slave", "master_node", "slave_node")
        if raw.get(key) is not None and str(raw.get(key)).strip()
    }
    if mesh is not None:
        nodes = list(self._node_targets_for_spec(mesh, raw))
        for edge in self._edge_targets_for_spec(mesh, raw):
            nodes.extend([str(edge[0]), str(edge[1])])
        nodes.extend(mpc_nodes)
        return sorted(set(nodes), key=self._natural_sort_key)
    nodes = {str(value) for value in self._ensure_list(raw.get("nodes", raw.get("node", []))) if value is not None}
    nodes.update(mpc_nodes)
    return sorted(nodes, key=self._natural_sort_key)

def _edges_for_condition(owner: Any, mesh: Any, raw: Mapping[str, Any]) -> list[tuple[str, str]]:
    self = owner
    if mesh is None:
        return []
    return list(self._edge_targets_for_spec(mesh, raw))

def _boundary_condition_label(owner: Any, raw: Mapping[str, Any]) -> str:
    self = owner
    support = str(raw.get("support_type", "") or "").strip()
    if support:
        return support
    if bool(raw.get("fixed", False)):
        return "fixed ux=uy=0"
    values = []
    for key in ("ux", "uy"):
        if key in raw and raw.get(key) is not None:
            values.append(f"{key}={raw.get(key)}")
    if values:
        return ", ".join(values)
    if "pressure" in raw:
        return f"pressure={raw.get('pressure')}"
    if "flux" in raw:
        return f"flux={raw.get('flux')}"
    if "beta" in raw:
        return f"robin beta={raw.get('beta')}"
    target = self._node_spec_text(raw) if hasattr(self, "_node_spec_text") else ""
    return target or "boundary"

def add_load_case_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.load_case_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"name": "LC1", "case_type": "static", "scale": "1.0", "active": "true", "description": ""}
    defaults.update(values)
    for col, key in enumerate(["name", "case_type", "scale", "active", "description"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def remove_selected_load_case_rows(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    rows = sorted({index.row() for index in self.load_case_table.selectedIndexes()}, reverse=True)
    for row in rows:
        self.load_case_table.removeRow(row)

def apply_load_cases_panel(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    cases: list[dict[str, Any]] = []
    try:
        for row in range(self.load_case_table.rowCount()):
            name = self._table_text(self.load_case_table, row, 0).strip()
            if not name:
                continue
            cases.append(
                {
                    "name": name,
                    "type": self._table_text(self.load_case_table, row, 1).strip() or "static",
                    "scale": self._float_text(self._table_text(self.load_case_table, row, 2).strip() or "1.0", "load case scale"),
                    "active": self._bool_text(self._table_text(self.load_case_table, row, 3).strip() or "true", True),
                    "description": self._table_text(self.load_case_table, row, 4).strip(),
                }
            )
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    self.cfg["load_cases"] = cases
    self._after_form_change("荷重ケースを反映しました")

def populate_load_case_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    table = getattr(self, "load_case_table", None)
    if table is None:
        return
    current_case = self._selected_load_case_name()
    table.setRowCount(0)
    cases = self._load_cases()
    if not cases:
        cases = [{"name": "LC1", "type": "static", "scale": 1.0, "active": True, "description": "default"}]
        self.cfg["load_cases"] = cases
    for case in cases:
        self.add_load_case_row(
            name=str(case.get("name", "")),
            case_type=str(case.get("type", "static")),
            scale=str(case.get("scale", 1.0)),
            active=str(case.get("active", True)).lower(),
            description=str(case.get("description", "")),
        )
    self._refresh_load_case_selector(current_case)

def _load_cases(owner: Any, qt: Mapping[str, Any]) -> list[dict[str, Any]]:
    self = owner
    raw = self.cfg.get("load_cases", [])
    if isinstance(raw, Mapping):
        return [dict(value, name=str(key)) if isinstance(value, Mapping) else {"name": str(key), "type": str(value)} for key, value in raw.items()]
    if not isinstance(raw, list):
        return []
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(raw, start=1):
        if isinstance(case, Mapping):
            data = dict(case)
            data.setdefault("name", f"LC{index}")
            cases.append(data)
        else:
            cases.append({"name": str(case), "type": "static", "scale": 1.0, "active": True, "description": ""})
    return cases

def _refresh_load_case_selector(owner: Any, qt: Mapping[str, Any], preferred: str = "") -> None:
    self = owner
    combo = getattr(self, "load_case_selector", None)
    if combo is None:
        return
    current = preferred or combo.currentText().strip()
    names = [str(case.get("name", "")) for case in self._load_cases() if str(case.get("name", "")).strip()]
    if not names:
        names = ["LC1"]
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(names)
    if current and current in names:
        combo.setCurrentText(current)
    elif names:
        combo.setCurrentIndex(0)
    combo.blockSignals(False)

def _selected_load_case_name(owner: Any, qt: Mapping[str, Any]) -> str:
    self = owner
    combo = getattr(self, "load_case_selector", None)
    if combo is None:
        return ""
    return combo.currentText().strip()

def _attach_load_case(owner: Any, qt: Mapping[str, Any], load: dict[str, Any]) -> dict[str, Any]:
    self = owner
    case_name = self._selected_load_case_name()
    if case_name:
        load = dict(load)
        load["load_case"] = case_name
    return load

def add_load_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.load_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"load_type": "node", "target": "right", "fx": "0.0", "fy": "-10.0", "tx": "", "ty": "", "scale": "", "extra": ""}
    defaults.update(values)
    for col, key in enumerate(["load_type", "target", "fx", "fy", "tx", "ty", "scale", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))

def remove_selected_load_rows(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    rows = sorted({index.row() for index in self.load_table.selectedIndexes()}, reverse=True)
    for row in rows:
        self.load_table.removeRow(row)

def apply_loads_panel(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    loads: list[dict[str, Any]] = []
    try:
        for row in range(self.load_table.rowCount()):
            load_type = self._table_text(self.load_table, row, 0).strip().lower() or "node"
            target = self._table_text(self.load_table, row, 1).strip()
            fx_text = self._table_text(self.load_table, row, 2).strip()
            fy_text = self._table_text(self.load_table, row, 3).strip()
            tx_text = self._table_text(self.load_table, row, 4).strip()
            ty_text = self._table_text(self.load_table, row, 5).strip()
            scale_text = self._table_text(self.load_table, row, 6).strip()
            extra = self._yaml_mapping_text(self._table_text(self.load_table, row, 7), "load extra YAML")
            if not target and load_type not in {"gravity", "self_weight", "body"} and not extra:
                continue
            load = dict(extra)
            if load_type in {"gravity", "self_weight", "body"}:
                kind = "body" if load_type == "body" or any(key in load for key in ("bx", "by", "body_fx", "body_fy", "body_x", "body_y")) else "gravity"
                load.update({"type": kind, "gx": float(load.get("gx", 0.0)), "gy": float(load.get("gy", -1.0))})
                if target:
                    load.update(self._target_body_load_spec(target))
                load["scale"] = self._float_text(scale_text or str(load.get("scale", 1.0)), "scale")
            elif load_type in {"edge", "edges", "traction", "distributed"}:
                if target:
                    load.update(self._target_load_edge_spec(target))
                elif not ("edge" in load or "edges" in load):
                    raise ValueError(f"荷重 {row + 1}: 辺荷重のtargetを指定してください。")
                if tx_text:
                    load["tx"] = self._float_text(tx_text, "tx")
                if ty_text:
                    load["ty"] = self._float_text(ty_text, "ty")
                if not any(key in load for key in ("tx", "ty", "qx", "qy", "tx1", "ty1", "tx2", "ty2", "qx1", "qy1", "qx2", "qy2")):
                    raise ValueError(f"荷重 {row + 1}: txまたはtyを指定してください。")
            else:
                if target:
                    load.update(self._target_node_spec(target))
                elif not any(key in load for key in ("node", "nodes", "set", "all")):
                    raise ValueError(f"荷重 {row + 1}: 節点荷重のtargetを指定してください。")
                if fx_text:
                    load["fx"] = self._float_text(fx_text, "fx")
                if fy_text:
                    load["fy"] = self._float_text(fy_text, "fy")
                if not any(key in load for key in ("fx", "fy", "px", "py")):
                    raise ValueError(f"荷重 {row + 1}: fxまたはfyを指定してください。")
            loads.append(load)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    self.cfg["loads"] = loads
    self._after_form_change("荷重表を反映しました")

def add_selected_nodal_load_condition(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    nodes = self._selected_nodes_for_stage_action()
    if not nodes:
        QMessageBox.information(self, "GeoFEM", "節点荷重を与える節点、辺、または要素を選択してください。")
        return
    restore_payload = {"nodes": list(nodes)}
    try:
        fx = self._float_text(self.load_fx.text().strip() or "0.0", "fx")
        fy = self._float_text(self.load_fy.text().strip() or "0.0", "fy")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    load = self._attach_load_case({"nodes": nodes, "fx": fx, "fy": fy, "source": "gui_selection"})
    scope = str(self.load_batch_scope.currentData() or "global")
    self._append_load_to_scope(load, scope, f"選択節点 {len(nodes)} 件へ節点荷重を追加しました")
    self._select_preview_payload(restore_payload)

def add_selected_body_load_condition(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    selected_entities = self._selected_preview_entities()
    restore_payload = {
        "nodes": sorted(selected_entities["nodes"], key=self._natural_sort_key),
        "edges": list(selected_entities["edges"]),
        "elements": sorted(selected_entities["elements"], key=self._natural_sort_key),
    }
    combo = getattr(self, "load_body_material", None)
    raw_material = ""
    if combo is not None:
        data = combo.currentData()
        if data is not None and str(data) != "__selected__":
            raw_material = str(data).strip()
        elif str(combo.currentText()).strip() != "選択要素の材料":
            raw_material = str(combo.currentText()).strip()
    material_names: list[str] = []
    if raw_material:
        material_names = [raw_material]
    elif selected_entities["elements"]:
        try:
            from geofem_app.fem2d import mesh_from_config

            mesh = mesh_from_config(self.cfg)
            selected = {str(eid) for eid in selected_entities["elements"]}
            material_names = sorted(
                {str(element.material) for element in mesh.elements if str(element.id) in selected and str(element.material).strip()},
                key=self._natural_sort_key,
            )
        except Exception:
            material_names = []
    if not material_names:
        QMessageBox.information(self, "GeoFEM", "体積力を与える要素を選択するか、材料名を指定してください。")
        return
    try:
        bx = self._float_text(self.load_body_bx.text().strip() or "0.0", "bx")
        by = self._float_text(self.load_body_by.text().strip() or "0.0", "by")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    target: dict[str, Any] = {"material": material_names[0]} if len(material_names) == 1 else {"materials": material_names}
    load = self._attach_load_case({"type": "body", **target, "bx": bx, "by": by, "scale": 1.0, "source": "gui_selection_material_body"})
    scope = str(self.load_batch_scope.currentData() or "global")
    self._append_load_to_scope(load, scope, f"材料 {', '.join(material_names)} へ体積力を追加しました")
    self._select_preview_payload(restore_payload)

def add_selected_distributed_load_condition(owner: Any, qt: Mapping[str, Any], *_args: Any, distribution: str | None = None) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    edges = self._selected_edges_for_stage_action()
    if not edges:
        QMessageBox.information(self, "GeoFEM", "分布荷重を与える辺または要素境界を選択してください。")
        return
    selected_entities = self._selected_preview_entities()
    restore_payload = {
        "nodes": sorted(selected_entities["nodes"], key=self._natural_sort_key),
        "edges": list(selected_entities["edges"]),
        "elements": sorted(selected_entities["elements"], key=self._natural_sort_key),
    }
    mode = str(distribution or "").strip().lower()
    if not mode and getattr(self, "load_surface_distribution", None) is not None:
        mode = str(self.load_surface_distribution.currentData() or self.load_surface_distribution.currentText() or "uniform").strip().lower()
    try:
        tx = self._float_text(self.load_tx.text().strip() or "0.0", "tx")
        ty = self._float_text(self.load_ty.text().strip() or "0.0", "ty")
        if mode in {"linear", "varying", "gradient", "偏分布"}:
            tx_end = self._float_text(self.load_tx_end.text().strip() or str(tx), "終端tx")
            ty_end = self._float_text(self.load_ty_end.text().strip() or str(ty), "終端ty")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    if mode in {"linear", "varying", "gradient", "偏分布"}:
        load = {
            "edges": [[a, b] for a, b in edges],
            "distribution": "linear",
            "tx1": tx,
            "ty1": ty,
            "tx2": tx_end,
            "ty2": ty_end,
            "source": "gui_selection_surface",
        }
        message = f"選択辺 {len(edges)} 件へ偏分布面荷重を追加しました"
    else:
        load = {"edges": [[a, b] for a, b in edges], "tx": tx, "ty": ty, "source": "gui_selection_surface"}
        message = f"選択辺 {len(edges)} 件へ等分布面荷重を追加しました"
    load = self._attach_load_case(load)
    scope = str(self.load_batch_scope.currentData() or "global")
    self._append_load_to_scope(load, scope, message)
    self._select_preview_payload(restore_payload)

def add_panel_earthquake_load(owner: Any, qt: Mapping[str, Any], *_args: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    try:
        kh = self._float_text(self.load_seismic_kh.text().strip() or "0.0", "kh")
        kv = self._float_text(self.load_seismic_kv.text().strip() or "0.0", "kv")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    direction = self.load_seismic_direction.currentText().strip() or "+X"
    gx = 0.0
    gy = kv
    if direction == "+X":
        gx = abs(kh)
    elif direction == "-X":
        gx = -abs(kh)
    elif direction == "+Y":
        gy = kv + abs(kh)
    elif direction == "-Y":
        gy = kv - abs(kh)
    load = self._attach_load_case(
        {
            "type": "gravity",
            "gx": gx,
            "gy": gy,
            "scale": 1.0,
            "seismic": {"method": "pseudo_static", "kh": kh, "kv": kv, "direction": direction},
            "source": "gui_load_panel",
        }
    )
    scope = str(self.load_batch_scope.currentData() or "stage")
    self._append_load_to_scope(load, scope, f"疑似静的地震荷重 kh={kh:g}, kv={kv:g}, dir={direction} を追加しました")

def _append_load_to_scope(owner: Any, qt: Mapping[str, Any], load: dict[str, Any], scope: str, message: str) -> None:
    self = owner
    if scope == "stage":
        stages, row, stage = self._selected_stage_context()
        self._list_value(stage, "loads").append(load)
        stage.setdefault("selection_history", []).append(
            {
                "kind": "load",
                "load_case": load.get("load_case", ""),
                "source": load.get("source", ""),
                "at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        stages[row] = stage
        self.cfg["stages"] = stages
        self._after_form_change(message)
        self.stage_table.selectRow(row)
        return
    loads = self.cfg.setdefault("loads", [])
    if not isinstance(loads, list):
        loads = []
        self.cfg["loads"] = loads
    loads.append(load)
    self._after_form_change(message)

def populate_load_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    table = getattr(self, "load_table", None)
    if table is None:
        return
    table.setRowCount(0)
    for raw in self._ensure_list(self.cfg.get("loads", [])):
        if not isinstance(raw, Mapping):
            continue
        ltype = str(raw.get("type", "")).lower().strip()
        if ltype in {"gravity", "self_weight", "body"} or bool(raw.get("self_weight", False)):
            extra = {
                k: v
                for k, v in raw.items()
                if k not in {"type", "self_weight", "scale", "material", "materials", "element", "elements", "element_set", "set", "all"}
            }
            self.add_load_row(
                load_type="body" if ltype == "body" or any(key in raw for key in ("bx", "by", "body_fx", "body_fy", "body_x", "body_y")) else "gravity",
                target=self._body_spec_text(raw),
                fx="",
                fy="",
                tx="",
                ty="",
                scale=str(raw.get("scale", 1.0)),
                extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip(),
            )
            continue
        if "edge" in raw or "edges" in raw:
            extra = {k: v for k, v in raw.items() if k not in {"edge", "edges", "tx", "ty", "qx", "qy", "tx1", "ty1", "tx2", "ty2", "qx1", "qy1", "qx2", "qy2"}}
            self.add_load_row(
                load_type="edge",
                target=self._edge_spec_text(raw),
                fx="",
                fy="",
                tx=str(raw.get("tx", raw.get("qx", raw.get("tx1", raw.get("qx1", ""))))),
                ty=str(raw.get("ty", raw.get("qy", raw.get("ty1", raw.get("qy1", ""))))),
                scale="",
                extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip(),
            )
            continue
        extra = {k: v for k, v in raw.items() if k not in {"node", "nodes", "set", "all", "fx", "fy", "px", "py"}}
        self.add_load_row(
            load_type="node",
            target=self._node_spec_text(raw),
            fx=str(raw.get("fx", raw.get("px", ""))),
            fy=str(raw.get("fy", raw.get("py", ""))),
            tx="",
            ty="",
            scale="",
            extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip(),
        )

def add_fix_boundary_condition(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    stage = self._last_stage()
    target, ok = QInputDialog.getText(self, "節点自由度拘束", "対象(set名 または node IDs)", text="left")
    if not ok or not target.strip():
        return
    dof_text, ok = QInputDialog.getItem(self, "節点自由度拘束", "拘束", ["固定(ux,uy)", "ux", "uy"], 0, False)
    if not ok:
        return
    bc = self._target_node_spec(target)
    if dof_text == "固定(ux,uy)":
        bc["fixed"] = True
    else:
        bc[dof_text] = 0.0
    self._list_value(stage, "boundary_conditions").append(bc)
    self._after_form_change("節点自由度拘束を追加しました")

def add_support_preset(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    stage = self._last_stage()
    kind, ok = QInputDialog.getItem(self, "ローラ/ピン支点", "支点種別", ["水平ローラ(uy=0)", "鉛直ローラ(ux=0)", "ピン(ux=uy=0)"], 0, False)
    if not ok:
        return
    target, ok = QInputDialog.getText(self, "ローラ/ピン支点", "対象(set名 または node IDs)", text="bottom")
    if not ok or not target.strip():
        return
    bc = self._target_node_spec(target)
    if kind.startswith("水平"):
        bc["uy"] = 0.0
    elif kind.startswith("鉛直"):
        bc["ux"] = 0.0
    else:
        bc["fixed"] = True
    self._list_value(stage, "boundary_conditions").append(bc)
    self._after_form_change("ローラ/ピン支点を追加しました")

def add_prescribed_displacement(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    stage = self._last_stage()
    target, ok = QInputDialog.getText(self, "強制変位", "対象(set名 または node IDs)", text="right")
    if not ok or not target.strip():
        return
    dof, ok = QInputDialog.getItem(self, "強制変位", "DOF", ["ux", "uy"], 0, False)
    if not ok:
        return
    value, ok = QInputDialog.getDouble(self, "強制変位", "変位値", 0.0, -1.0e12, 1.0e12, 8)
    if not ok:
        return
    bc = self._target_node_spec(target)
    bc[dof] = value
    self._list_value(stage, "boundary_conditions").append(bc)
    self._after_form_change("強制変位を追加しました")

def add_mpc_constraint(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    stage = self._last_stage()
    text, ok = QInputDialog.getText(self, "MPC拘束", "master,slave,dof,係数", text="1,2,ux,1.0")
    if not ok:
        return
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        QMessageBox.warning(self, "GeoFEM", "master,slave,dof,係数 の形式で入力してください。")
        return
    try:
        coeff = float(parts[3])
    except ValueError:
        QMessageBox.warning(self, "GeoFEM", "係数は数値で入力してください。")
        return
    self._list_value(stage, "mpc_constraints").append({"master": parts[0], "slave": parts[1], "dof": parts[2], "coefficient": coeff, "method": "elimination"})
    self._after_form_change("MPC拘束を記録しました")

def add_nodal_load(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    stage = self._last_stage()
    text, ok = QInputDialog.getText(self, "節点集中荷重", "node,fx,fy", text="1,0,-10")
    if not ok:
        return
    try:
        node, fx, fy = self._parse_node_force(text)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    self._list_value(stage, "loads").append({"node": node, "fx": fx, "fy": fy})
    self._after_form_change("節点集中荷重を追加しました")

def add_edge_load(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    stage = self._last_stage()
    text, ok = QInputDialog.getText(self, "座標系分布荷重", "edge/set,tx,ty", text="top,0,-10")
    if not ok:
        return
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        QMessageBox.warning(self, "GeoFEM", "edge/set,tx,ty の形式で入力してください。")
        return
    target, tx_text, ty_text = parts
    try:
        tx = float(tx_text)
        ty = float(ty_text)
    except ValueError:
        QMessageBox.warning(self, "GeoFEM", "tx,tyは数値で入力してください。")
        return
    load: dict[str, Any] = {"tx": tx, "ty": ty}
    edge_nodes = [item.strip() for item in target.replace(" ", ",").split(",") if item.strip()]
    if len(edge_nodes) == 2:
        load["edge"] = edge_nodes
    else:
        load["edges"] = target
    self._list_value(stage, "loads").append(load)
    self._after_form_change("座標系分布荷重を追加しました")

def add_gravity_load(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    stage = self._last_stage()
    scale, ok = QInputDialog.getDouble(self, "自重", "倍率", 1.0, -1.0e6, 1.0e6, 6)
    if not ok:
        return
    self._list_value(stage, "loads").append({"type": "gravity", "gx": 0.0, "gy": -1.0, "scale": scale})
    self._after_form_change("自重を追加しました")

def add_nodal_pore_pressure(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    stage = self._last_stage()
    if str(stage.get("type", "")).lower() not in {"consolidation", "u-p", "u_p", "up", "coupled_consolidation"}:
        stage["type"] = "consolidation"
    target, ok = QInputDialog.getText(self, "節点水圧", "対象(set名 または node IDs)", text="top")
    if not ok or not target.strip():
        return
    value, ok = QInputDialog.getDouble(self, "節点水圧", "水圧", 0.0, -1.0e12, 1.0e12, 6)
    if not ok:
        return
    hydro = stage.setdefault("hydro", {})
    if not isinstance(hydro, dict):
        hydro = {}
        stage["hydro"] = hydro
    spec = self._target_node_spec(target)
    spec["pressure"] = value
    self._list_value(hydro, "pressure_bcs").append(spec)
    self._after_form_change("節点水圧を追加しました")

def add_hydro_boundary(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    stage = self._last_stage()
    if str(stage.get("type", "")).lower() not in {"consolidation", "u-p", "u_p", "up", "coupled_consolidation"}:
        stage["type"] = "consolidation"
    kind, ok = QInputDialog.getItem(self, "水位/水圧条件", "条件種別", ["pressure", "flux", "robin"], 0, False)
    if not ok:
        return
    target, ok = QInputDialog.getText(self, "水位/水圧条件", "対象 edge set または node1,node2", text="top")
    if not ok or not target.strip():
        return
    value, ok = QInputDialog.getDouble(self, "水位/水圧条件", "値", 0.0, -1.0e12, 1.0e12, 6)
    if not ok:
        return
    hydro = stage.setdefault("hydro", {})
    if not isinstance(hydro, dict):
        hydro = {}
        stage["hydro"] = hydro
    spec = self._target_edge_spec(target)
    if kind == "pressure":
        spec["pressure"] = value
        self._list_value(hydro, "pressure_bcs").append(spec)
    elif kind == "flux":
        spec["flux"] = value
        self._list_value(hydro, "pore_flux_bcs").append(spec)
    else:
        spec["beta"] = max(value, 0.0)
        spec["pressure"] = 0.0
        self._list_value(hydro, "pore_robin_bcs").append(spec)
    self._after_form_change("水位/水圧条件を追加しました")

def _loads_from_table(owner: Any, qt: Mapping[str, Any], table: QTableWidget) -> list[dict[str, Any]]:
    self = owner
    QTableWidget = qt["QTableWidget"]
    loads: list[dict[str, Any]] = []
    for row in range(table.rowCount()):
        load_type = self._table_text(table, row, 0).strip().lower() or "node"
        target = self._table_text(table, row, 1).strip()
        fx_text = self._table_text(table, row, 2).strip()
        fy_text = self._table_text(table, row, 3).strip()
        tx_text = self._table_text(table, row, 4).strip()
        ty_text = self._table_text(table, row, 5).strip()
        scale_text = self._table_text(table, row, 6).strip()
        extra = self._yaml_mapping_text(self._table_text(table, row, 7), "load extra YAML")
        if not target and load_type not in {"gravity", "self_weight", "body"} and not extra:
            continue
        load = dict(extra)
        if load_type in {"gravity", "self_weight", "body"}:
            kind = "body" if load_type == "body" or any(key in load for key in ("bx", "by", "body_fx", "body_fy", "body_x", "body_y")) else "gravity"
            load.update({"type": kind, "gx": float(load.get("gx", 0.0)), "gy": float(load.get("gy", -1.0))})
            if target:
                load.update(self._target_body_load_spec(target))
            load["scale"] = self._float_text(scale_text or str(load.get("scale", 1.0)), "scale")
        elif load_type in {"edge", "edges", "traction", "distributed"}:
            if target:
                load.update(self._target_load_edge_spec(target))
            elif not ("edge" in load or "edges" in load):
                raise ValueError(f"荷重 {row + 1}: 辺荷重のtargetを指定してください。")
            if tx_text:
                load["tx"] = self._float_text(tx_text, "tx")
            if ty_text:
                load["ty"] = self._float_text(ty_text, "ty")
            if not any(key in load for key in ("tx", "ty", "qx", "qy", "tx1", "ty1", "tx2", "ty2", "qx1", "qy1", "qx2", "qy2")):
                raise ValueError(f"荷重 {row + 1}: txまたはtyを指定してください。")
        else:
            if target:
                load.update(self._target_node_spec(target))
            elif not any(key in load for key in ("node", "nodes", "set", "all")):
                raise ValueError(f"荷重 {row + 1}: 節点荷重のtargetを指定してください。")
            if fx_text:
                load["fx"] = self._float_text(fx_text, "fx")
            if fy_text:
                load["fy"] = self._float_text(fy_text, "fy")
            if not any(key in load for key in ("fx", "fy", "px", "py")):
                raise ValueError(f"荷重 {row + 1}: fxまたはfyを指定してください。")
        loads.append(load)
    return loads

def _add_boundary_mapping_to_table(owner: Any, qt: Mapping[str, Any], table: QTableWidget, raw: Mapping[str, Any], add_row: Any) -> None:
    self = owner
    QTableWidget = qt["QTableWidget"]
    target = self._node_spec_text(raw)
    ux = ""
    uy = ""
    if bool(raw.get("fixed", False)):
        ux = "0.0"
        uy = "0.0"
    if "ux" in raw and raw.get("ux") is not None:
        ux = str(raw.get("ux"))
    if "uy" in raw and raw.get("uy") is not None:
        uy = str(raw.get("uy"))
    if str(raw.get("dof", "")).lower() == "ux":
        ux = str(raw.get("value", 0.0))
    elif str(raw.get("dof", "")).lower() == "uy":
        uy = str(raw.get("value", 0.0))
    exclude = {"node", "nodes", "set", "all", "ux", "uy", "fixed"}
    if str(raw.get("dof", "")).lower() in {"ux", "uy"}:
        exclude.update({"dof", "value"})
    extra = {k: v for k, v in raw.items() if k not in exclude}
    if bool(raw.get("fixed", False)):
        extra.pop("fixed", None)
    add_row(target=target, ux=ux, uy=uy, extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())

def _add_load_mapping_to_table(owner: Any, qt: Mapping[str, Any], table: QTableWidget, raw: Mapping[str, Any], add_row: Any) -> None:
    self = owner
    QTableWidget = qt["QTableWidget"]
    ltype = str(raw.get("type", "")).lower().strip()
    if ltype in {"gravity", "self_weight", "body"} or bool(raw.get("self_weight", False)):
        extra = {k: v for k, v in raw.items() if k not in {"type", "self_weight", "scale"}}
        add_row(load_type="gravity", target="", fx="", fy="", tx="", ty="", scale=str(raw.get("scale", 1.0)), extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
        return
    if "edge" in raw or "edges" in raw:
        extra = {k: v for k, v in raw.items() if k not in {"edge", "edges", "tx", "ty", "qx", "qy"}}
        add_row(load_type="edge", target=self._edge_spec_text(raw), fx="", fy="", tx=str(raw.get("tx", raw.get("qx", ""))), ty=str(raw.get("ty", raw.get("qy", ""))), scale="", extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
        return
    extra = {k: v for k, v in raw.items() if k not in {"node", "nodes", "set", "all", "fx", "fy", "px", "py"}}
    add_row(load_type="node", target=self._node_spec_text(raw), fx=str(raw.get("fx", raw.get("px", ""))), fy=str(raw.get("fy", raw.get("py", ""))), tx="", ty="", scale="", extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())

__all__ = [
    "BOUNDARY_LOAD_CONTROLLER_METHODS",
    "boundary_load_controller_contract",
    *BOUNDARY_LOAD_CONTROLLER_METHODS,
]
