"""Stage operation controller functions split from MainWindow.

These functions operate on the MainWindow owner but keep stage table,
diff approval, template, and guidance logic outside the GUI shell.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import getpass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

STAGE_CONTROLLER_METHODS = (
    "add_stage",
    "copy_selected_stage",
    "move_selected_stage",
    "delete_selected_stage",
    "apply_stage_table",
    "populate_stage_detail_tables",
    "_clear_stage_detail_form",
    "_populate_stage_detail_form",
    "_populate_stage_construction_table",
    "populate_stage_change_table",
    "add_stage_construction_row",
    "add_stage_change_row",
    "stage_recommended_defaults",
    "refresh_stage_recommendation_label",
    "apply_stage_recommended_defaults",
    "_apply_stage_detail_form_values",
    "_construction_events_from_table",
    "apply_stage_change_table",
    "add_stage_material_row",
    "add_stage_boundary_row",
    "add_stage_load_row",
    "add_stage_hydro_row",
    "add_stage_mpc_row",
    "apply_stage_detail_tables",
    "apply_selected_elements_to_stage_set",
    "add_selected_elements_to_stage_death",
    "add_selected_elements_to_stage_birth",
    "add_selected_elements_to_stage_material_change",
    "add_selected_prescribed_displacement",
    "add_selected_edge_load",
    "add_pseudo_static_earthquake_load",
    "add_selected_hydro_coupling",
    "_stage_difference_rows",
    "refresh_stage_difference_table",
    "_stage_diff_approval_key",
    "_stage_diff_approval_status",
    "_stage_diff_is_locked",
    "_selected_stage_diff_key",
    "_append_stage_approval_history",
    "approve_selected_stage_difference",
    "approve_stage_difference",
    "reject_selected_stage_difference",
    "reject_stage_difference",
    "reapprove_selected_stage_difference",
    "reapprove_stage_difference",
    "stage_approval_history",
    "compare_stage_approval_history",
    "refresh_stage_approval_history_table",
    "apply_stage_construction_template",
    "repair_selected_stage_difference",
    "repair_stage_difference_row",
    "repair_stage_difference_cell",
    "_stage_template_library",
    "_refresh_stage_template_combo",
    "apply_stage_template_from_library",
    "save_stage_template_library",
    "load_stage_template_library",
    "collect_stage_cumulative_conflicts",
    "refresh_stage_conflict_table",
    "repair_selected_stage_conflict",
    "apply_stage_conflict_repair_suggestions",
    "repair_stage_conflict",
    "refresh_stage_cross_compare_table",
    "refresh_stage_guidance",
    "stage_guidance_steps",
    "refresh_stage_wizard_table",
    "show_stage_difference",
)

def stage_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.stage_controller.v1",
        "method_count": len(STAGE_CONTROLLER_METHODS),
        "methods": list(STAGE_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner cfg/widgets; MainWindow delegates stage operations",
    }


def _notify_information(owner: Any, message_box: Any, message: str) -> None:
    if hasattr(owner, "notify_user"):
        owner.notify_user(message)
    else:
        message_box.information(owner, "GeoFEM", message)


def _stage_type_family(stage_type: str) -> str:
    stype = str(stage_type or "static").strip().lower().replace("-", "_")
    if stype in {"geostatic", "k0", "initial_stress"}:
        return "geostatic"
    if stype in {"excavation", "death", "deactivate", "remove"}:
        return "excavation"
    if stype in {"consolidation", "u_p", "up", "u-p", "coupled_consolidation"}:
        return "consolidation"
    if stype in {"srm", "safety_factor", "strength_reduction"}:
        return "srm"
    if stype in {"riks", "arc_length", "arclength"}:
        return "riks"
    if stype in {"large_deformation", "large_displacement", "finite_deformation", "updated_lagrangian"}:
        return "large_deformation"
    return "static"


def _first_material_name(owner: Any, fallback: str = "soil") -> str:
    edit = getattr(owner, "mesh_material", None)
    if edit is not None:
        text = edit.text().strip()
        if text:
            return text
    materials = owner.cfg.get("materials", {}) if isinstance(getattr(owner, "cfg", {}), Mapping) else {}
    if isinstance(materials, Mapping) and materials:
        return str(next(iter(materials)))
    return fallback


def _analysis_deformation_mode(owner: Any) -> str:
    cfg = getattr(owner, "cfg", {})
    analysis = cfg.get("analysis", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(analysis, Mapping):
        return "small_deformation"
    raw = str(analysis.get("deformation_mode", analysis.get("geometry_mode", "small_deformation"))).lower().strip().replace("-", "_")
    return "large_deformation" if raw in {"large", "large_deformation", "large_displacement", "finite_deformation", "updated_lagrangian"} else "small_deformation"


def stage_recommended_defaults(owner: Any, qt: Mapping[str, Any], stage_type: str | None = None) -> dict[str, Any]:
    self = owner
    combo = getattr(self, "stage_detail_type", None)
    selected_type = stage_type or (self._combo_value(combo, "static") if combo is not None else "static")
    family = _stage_type_family(str(selected_type))
    base: dict[str, Any] = {
        "stage_type": str(selected_type or "static"),
        "family": family,
        "line_edits": {"stage_detail_scale": "1.0", "stage_detail_increments": "1"},
        "combo_edits": {},
        "clear_line_edits": [],
        "solver": {"max_iter": 20, "tolerance": 1.0e-6},
        "summary": "推奨値: 静的ステージは荷重倍率 1.0、増分数 1、Newton最大20回、収束許容値 1.0e-6 を初期値にします。",
    }
    if family == "geostatic":
        base.update(
            {
                "line_edits": {
                    "stage_detail_k0": "0.5",
                    "stage_detail_surface_y": "0.0",
                    "stage_detail_gx": "0.0",
                    "stage_detail_gy": "-9.80665",
                    "stage_detail_scale": "1.0",
                    "stage_detail_increments": "1",
                },
                "combo_edits": {"stage_detail_apply_gravity": "true"},
                "clear_line_edits": [
                    "stage_detail_stress_release",
                    "stage_detail_dt",
                    "stage_detail_steps",
                    "stage_detail_storage",
                    "stage_detail_permeability",
                    "stage_detail_biot_alpha",
                    "stage_detail_srm_start",
                    "stage_detail_srm_end",
                    "stage_detail_srm_step",
                    "stage_detail_srm_failure_ratio",
                    "stage_detail_riks_arc",
                    "stage_detail_riks_steps",
                ],
                "solver": {"max_iter": 30, "tolerance": 1.0e-6},
                "summary": "推奨値: K0/初期応力は K0=0.5、地表Y=0.0、重力Y=-9.80665、増分数1を初期値にします。",
            }
        )
    elif family == "excavation":
        base.update(
            {
                "line_edits": {
                    "stage_detail_target": "all",
                    "stage_detail_stress_release": "1.0",
                    "stage_detail_increments": "4",
                },
                "clear_line_edits": [
                    "stage_detail_dt",
                    "stage_detail_steps",
                    "stage_detail_storage",
                    "stage_detail_permeability",
                    "stage_detail_biot_alpha",
                    "stage_detail_srm_start",
                    "stage_detail_srm_end",
                    "stage_detail_srm_step",
                    "stage_detail_srm_failure_ratio",
                    "stage_detail_riks_arc",
                    "stage_detail_riks_steps",
                ],
                "solver": {"max_iter": 30, "tolerance": 1.0e-6, "cutback": True},
                "summary": "推奨値: 掘削/deathは対象 set=all、応力解放率 1.0、増分数4、カットバック有効を初期値にします。対象setは必ず案件に合わせて確認してください。",
            }
        )
    elif family == "consolidation":
        base.update(
            {
                "line_edits": {
                    "stage_detail_dt": "1.0",
                    "stage_detail_steps": "1",
                    "stage_detail_storage": "1.0e-5",
                    "stage_detail_permeability": "1.0e-6",
                    "stage_detail_biot_alpha": "1.0",
                    "stage_detail_increments": "4",
                },
                "clear_line_edits": [
                    "stage_detail_stress_release",
                    "stage_detail_srm_start",
                    "stage_detail_srm_end",
                    "stage_detail_srm_step",
                    "stage_detail_srm_failure_ratio",
                    "stage_detail_riks_arc",
                    "stage_detail_riks_steps",
                ],
                "solver": {"max_iter": 40, "tolerance": 1.0e-6, "cutback": True},
                "summary": "推奨値: 圧密/u-pは dt=1.0、steps=1、storage=1.0e-5、透水係数=1.0e-6、Biot alpha=1.0、増分数4を初期値にします。",
            }
        )
    elif family == "srm":
        base.update(
            {
                "line_edits": {
                    "stage_detail_srm_start": "1.0",
                    "stage_detail_srm_end": "2.0",
                    "stage_detail_srm_step": "0.05",
                    "stage_detail_srm_failure_ratio": "0.95",
                    "stage_detail_increments": "1",
                },
                "clear_line_edits": [
                    "stage_detail_stress_release",
                    "stage_detail_dt",
                    "stage_detail_steps",
                    "stage_detail_storage",
                    "stage_detail_permeability",
                    "stage_detail_biot_alpha",
                    "stage_detail_riks_arc",
                    "stage_detail_riks_steps",
                ],
                "solver": {"max_iter": 60, "tolerance": 1.0e-6, "cutback": True},
                "summary": "推奨値: SRMは低減係数 1.0→2.0、刻み0.05、Newton最大60回、カットバック有効を初期値にします。",
            }
        )
    elif family == "riks":
        base.update(
            {
                "line_edits": {
                    "stage_detail_riks_arc": "0.1",
                    "stage_detail_riks_steps": "12",
                    "stage_detail_increments": "12",
                },
                "clear_line_edits": [
                    "stage_detail_stress_release",
                    "stage_detail_dt",
                    "stage_detail_steps",
                    "stage_detail_storage",
                    "stage_detail_permeability",
                    "stage_detail_biot_alpha",
                    "stage_detail_srm_start",
                    "stage_detail_srm_end",
                    "stage_detail_srm_step",
                    "stage_detail_srm_failure_ratio",
                ],
                "solver": {"max_iter": 40, "tolerance": 1.0e-6, "line_search": True},
                "summary": "推奨値: Riks/弧長法は arc length=0.1、最大12ステップ、Newton最大40回、ラインサーチ有効を初期値にします。",
            }
        )
    elif family == "large_deformation":
        base.update(
            {
                "line_edits": {"stage_detail_increments": "8"},
                "solver": {
                    "max_iter": 40,
                    "tolerance": 1.0e-6,
                    "line_search": True,
                    "large_deformation": {"enabled": True, "steps": 8, "backend": "auto"},
                },
                "summary": "推奨値: 大変形はUpdated Lagrangian形状更新、8分割、Newton最大40回、ラインサーチ有効を初期値にします。",
            }
        )
    if getattr(self, "gui_locale", "ja") == "en":
        english_summaries = {
            "static": "Defaults: static stages use load scale 1.0, one increment, Newton max_iter 20, and tolerance 1.0e-6.",
            "geostatic": "Defaults: geostatic/K0 stages use K0=0.5, surface Y=0.0, gravity gy=-9.80665, and one increment.",
            "excavation": "Defaults: excavation/death stages use target set=all, stress release 1.0, four increments, and cutback enabled.",
            "consolidation": "Defaults: consolidation/u-p stages use dt=1.0, steps=1, storage=1.0e-5, permeability=1.0e-6, Biot alpha=1.0, and four increments.",
            "srm": "Defaults: SRM sweeps strength-reduction factor 1.0 to 2.0 by 0.05 with Newton max_iter 60 and cutback enabled.",
            "riks": "Defaults: Riks/arc-length stages use arc length 0.1, max steps 12, Newton max_iter 40, and line search enabled.",
            "large_deformation": "Defaults: large deformation uses updated-Lagrangian geometry updates, 8 steps, Newton max_iter 40, and line search enabled.",
        }
        base["summary"] = english_summaries.get(family, english_summaries["static"])
    return base


def refresh_stage_recommendation_label(owner: Any, qt: Mapping[str, Any]) -> dict[str, Any]:
    self = owner
    defaults = self.stage_recommended_defaults()
    label = getattr(self, "stage_recommendation_label", None)
    if label is not None:
        label.setText(str(defaults.get("summary", "")))
    return defaults


def apply_stage_recommended_defaults(owner: Any, qt: Mapping[str, Any], force: bool = False, stage_type: str | None = None) -> dict[str, Any]:
    self = owner
    if stage_type and getattr(self, "stage_detail_type", None) is not None:
        self._set_combo(self.stage_detail_type, stage_type)
    defaults = self.stage_recommended_defaults(stage_type)
    if force:
        for attr in defaults.get("clear_line_edits", []):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.clear()
    for attr, value in defaults.get("line_edits", {}).items():
        widget = getattr(self, attr, None)
        if widget is None:
            continue
        if force or not widget.text().strip():
            widget.setText(str(value))
    for attr, value in defaults.get("combo_edits", {}).items():
        combo = getattr(self, attr, None)
        if combo is not None:
            self._set_combo(combo, str(value))
    solver = getattr(self, "stage_detail_solver", None)
    if solver is not None and (force or not solver.toPlainText().strip()):
        solver.setPlainText(yaml.safe_dump(defaults.get("solver", {}), allow_unicode=True, sort_keys=False).strip())
    label = getattr(self, "stage_recommendation_label", None)
    if label is not None:
        label.setText(str(defaults.get("summary", "")))
    return defaults


def add_stage(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    name, ok = QInputDialog.getText(self, "ステージ追加", "ステージ名", text=f"Stage-{len(self._stages()) + 1}")
    if not ok or not name.strip():
        return
    stage_options = ["static", "large_deformation", "geostatic", "k0", "excavation", "death", "deactivate", "srm", "safety_factor", "consolidation", "u-p", "riks", "arc_length"]
    default_index = 1 if _analysis_deformation_mode(self) == "large_deformation" else 0
    stage_type, ok = QInputDialog.getItem(
        self,
        "ステージ追加",
        "解析状態",
        stage_options,
        default_index,
        False,
    )
    if not ok:
        return
    stage: dict[str, Any] = {"name": name.strip(), "type": stage_type}
    deformation_mode = _analysis_deformation_mode(self)
    stage["deformation_mode"] = deformation_mode
    if stage_type == "geostatic":
        stage.update({"apply_gravity": True, "k0": 0.5})
    elif stage_type in {"excavation", "death", "deactivate"}:
        stage.update({"set": "all", "stress_release": 1.0, "increments": 4, "solver": {"max_iter": 30, "tolerance": 1.0e-6, "cutback": True}})
    elif stage_type == "consolidation":
        stage.update({"increments": 4, "solver": {"max_iter": 40, "tolerance": 1.0e-6, "cutback": True}})
        stage["hydro"] = {"dt": 1.0, "steps": 1, "storage": 1.0e-5, "permeability": 1.0e-6, "biot_alpha": 1.0}
    elif stage_type in {"riks", "arc_length"}:
        stage.update({"increments": 12, "solver": {"max_iter": 40, "tolerance": 1.0e-6, "line_search": True}})
        stage["arc_length"] = {"arc_length": 0.1, "steps": 12}
    elif stage_type in {"srm", "safety_factor"}:
        stage.update({"increments": 1, "solver": {"max_iter": 60, "tolerance": 1.0e-6, "cutback": True}})
        stage["srm"] = {"search_mode": "adaptive_bracket", "anchor_factor": 1.0, "bracket_stride": 5, "factor_tol": 0.01, "max_bisection": 6, "factor_start": 1.0, "factor_max": 2.0, "factor_step": 0.05, "failure_plastic_ratio": 0.95}
    elif stage_type in {"large_deformation", "large_displacement", "finite_deformation", "updated_lagrangian"}:
        stage.update({"increments": 8, "solver": {"max_iter": 40, "tolerance": 1.0e-6, "line_search": True, "large_deformation": {"enabled": True, "steps": 8, "backend": "auto"}}})
    self._stages().append(stage)
    self._after_form_change(f"ステージ '{stage['name']}' を追加しました")


def copy_selected_stage(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    stages = self._stages()
    row = self._selected_stage_row()
    if row is None or row >= len(stages):
        _notify_information(self, QMessageBox, "コピーするステージを選択してください。")
        return
    copied = json.loads(json.dumps(stages[row], ensure_ascii=False))
    copied["name"] = f"{copied.get('name', f'Stage-{row + 1}')}-copy"
    stages.insert(row + 1, copied)
    self._after_form_change("ステージをコピーしました")


def move_selected_stage(owner: Any, qt: Mapping[str, Any], delta: int) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    stages = self._stages()
    row = self._selected_stage_row()
    if row is None:
        _notify_information(self, QMessageBox, "移動するステージを選択してください。")
        return
    new_row = row + delta
    if new_row < 0 or new_row >= len(stages):
        return
    stages[row], stages[new_row] = stages[new_row], stages[row]
    self._after_form_change("ステージ順序を変更しました")
    self.stage_table.selectRow(new_row)


def delete_selected_stage(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    stages = self._stages()
    row = self._selected_stage_row()
    if row is None or row >= len(stages):
        _notify_information(self, QMessageBox, "削除するステージを選択してください。")
        return
    stages.pop(row)
    self._after_form_change("ステージを削除しました")


def apply_stage_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    stages = self._stages()
    new_stages: list[dict[str, Any]] = []
    self.clear_table_cell_errors(self.stage_table)
    try:
        for row in range(self.stage_table.rowCount()):
            base = dict(stages[row]) if row < len(stages) else {}
            name = self._table_text(self.stage_table, row, 1).strip() or f"Stage-{row + 1}"
            stage_type = self._table_text(self.stage_table, row, 2).strip() or "static"
            target = self._table_text(self.stage_table, row, 3).strip()
            release = self._table_text(self.stage_table, row, 4).strip()
            base["name"] = name
            base["type"] = stage_type
            for key in ("set", "element_set", "elementSet", "element", "elements", "all"):
                base.pop(key, None)
            if target:
                base.update(self._target_element_spec(target))
            if release:
                base["stress_release"] = self._float_table_cell(self.stage_table, row, 4, "stress_release")
            else:
                base.pop("stress_release", None)
            new_stages.append(base)
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"ステージ表の入力が不正です: {exc}")
        return
    self.cfg["stages"] = new_stages
    self._after_form_change("ステージ表を反映しました")


@contextmanager
def _quiet_stage_detail_population(owner: Any):
    widget_names = (
        "stage_material_table",
        "stage_boundary_table",
        "stage_load_table",
        "stage_hydro_table",
        "stage_mpc_table",
        "stage_construction_table",
        "stage_diff_table",
        "stage_detail_name",
        "stage_detail_type",
        "stage_detail_target",
        "stage_detail_stress_release",
        "stage_detail_apply_gravity",
        "stage_detail_k0",
        "stage_detail_surface_y",
        "stage_detail_gx",
        "stage_detail_gy",
        "stage_detail_scale",
        "stage_detail_dt",
        "stage_detail_steps",
        "stage_detail_storage",
        "stage_detail_permeability",
        "stage_detail_biot_alpha",
        "stage_detail_srm_start",
        "stage_detail_srm_end",
        "stage_detail_srm_step",
        "stage_detail_srm_failure_ratio",
        "stage_detail_riks_arc",
        "stage_detail_riks_steps",
        "stage_detail_increments",
        "stage_detail_solver",
    )
    states: list[tuple[Any, bool, bool]] = []
    for name in widget_names:
        widget = getattr(owner, name, None)
        if widget is None:
            continue
        signals_blocked = bool(widget.signalsBlocked())
        updates_enabled = bool(widget.updatesEnabled())
        states.append((widget, signals_blocked, updates_enabled))
        widget.blockSignals(True)
        widget.setUpdatesEnabled(False)
    try:
        yield
    finally:
        for widget, signals_blocked, updates_enabled in reversed(states):
            widget.setUpdatesEnabled(updates_enabled)
            widget.blockSignals(signals_blocked)
            if updates_enabled:
                widget.update()


def populate_stage_detail_tables(owner: Any, qt: Mapping[str, Any]) -> None:
    with _quiet_stage_detail_population(owner):
        _populate_stage_detail_tables_unblocked(owner, qt)


def _populate_stage_detail_tables_unblocked(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    for table_name in ("stage_material_table", "stage_boundary_table", "stage_load_table", "stage_hydro_table", "stage_mpc_table", "stage_construction_table"):
        table = getattr(self, table_name, None)
        if table is not None:
            table.setRowCount(0)
    self._clear_stage_detail_form()
    row = self._selected_stage_row()
    stages = self._stages()
    if row is None or row < 0 or row >= len(stages):
        return
    stage = stages[row]
    self._populate_stage_detail_form(stage, row)
    self._populate_stage_construction_table(stage)
    for raw in self._ensure_list(stage.get("element_properties", [])):
        if not isinstance(raw, Mapping):
            continue
        target = self._element_spec_text(raw)
        material = str(raw.get("material", ""))
        extra = {k: v for k, v in raw.items() if k not in {"set", "element_set", "elementSet", "element", "elements", "all", "material"}}
        self.add_stage_material_row(target=target, material=material, extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
    for raw in self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))):
        if isinstance(raw, Mapping):
            self._add_boundary_mapping_to_table(self.stage_boundary_table, raw, self.add_stage_boundary_row)
    for raw in self._ensure_list(stage.get("loads", [])):
        if isinstance(raw, Mapping):
            self._add_load_mapping_to_table(self.stage_load_table, raw, self.add_stage_load_row)
    hydro = stage.get("hydro", stage.get("consolidation", {}))
    if isinstance(hydro, Mapping):
        for raw in self._ensure_list(hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", []))):
            if isinstance(raw, Mapping):
                target = self._edge_spec_text(raw) or self._node_spec_text(raw)
                value = raw.get("pressure", raw.get("value", 0.0))
                extra = {k: v for k, v in raw.items() if k not in {"edge", "edges", "node", "nodes", "set", "all", "pressure", "value"}}
                kind = "node_pressure" if any(key in raw for key in ("node", "nodes")) and not any(key in raw for key in ("edge", "edges")) else "pressure"
                self.add_stage_hydro_row(kind=kind, target=target, value=str(value), extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
        for raw in self._ensure_list(hydro.get("pore_flux_bcs", [])):
            if isinstance(raw, Mapping):
                target = self._edge_spec_text(raw) or self._node_spec_text(raw)
                value = raw.get("flux", raw.get("value", 0.0))
                extra = {k: v for k, v in raw.items() if k not in {"edge", "edges", "node", "nodes", "set", "all", "flux", "value"}}
                self.add_stage_hydro_row(kind="flux", target=target, value=str(value), extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
        for raw in self._ensure_list(hydro.get("pore_robin_bcs", [])):
            if isinstance(raw, Mapping):
                target = self._edge_spec_text(raw) or self._node_spec_text(raw)
                value = raw.get("beta", raw.get("value", 0.0))
                extra = {k: v for k, v in raw.items() if k not in {"edge", "edges", "node", "nodes", "set", "all", "beta", "value"}}
                self.add_stage_hydro_row(kind="robin", target=target, value=str(value), extra="" if not extra else yaml.safe_dump(extra, allow_unicode=True, sort_keys=False).strip())
    for raw in self._ensure_list(stage.get("mpc_constraints", stage.get("mpc", []))):
        if not isinstance(raw, Mapping):
            continue
        self.add_stage_mpc_row(
            master=str(raw.get("master", raw.get("master_node", ""))),
            slave=str(raw.get("slave", raw.get("slave_node", ""))),
            dof=str(raw.get("dof", raw.get("component", "ux"))),
            coefficient=str(raw.get("coefficient", raw.get("coef", 1.0))),
            value=str(raw.get("value", raw.get("offset", 0.0))),
            method=str(raw.get("method", "elimination")),
        )
    self.refresh_stage_difference_table()


def _clear_stage_detail_form(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    for attr in (
        "stage_detail_name",
        "stage_detail_target",
        "stage_detail_stress_release",
        "stage_detail_k0",
        "stage_detail_surface_y",
        "stage_detail_gx",
        "stage_detail_gy",
        "stage_detail_scale",
        "stage_detail_dt",
        "stage_detail_steps",
        "stage_detail_storage",
        "stage_detail_permeability",
        "stage_detail_biot_alpha",
        "stage_detail_srm_start",
        "stage_detail_srm_end",
        "stage_detail_srm_step",
        "stage_detail_srm_failure_ratio",
        "stage_detail_riks_arc",
        "stage_detail_riks_steps",
        "stage_detail_increments",
    ):
        widget = getattr(self, attr, None)
        if widget is not None:
            widget.setText("")
    if hasattr(self, "stage_detail_solver"):
        self.stage_detail_solver.setPlainText("")


def _populate_stage_detail_form(owner: Any, qt: Mapping[str, Any], stage: Mapping[str, Any], row: int) -> None:
    self = owner
    self.stage_detail_name.setText(str(stage.get("name", f"Stage-{row + 1}")))
    self._set_combo(self.stage_detail_type, str(stage.get("type", "static")))
    self.stage_detail_target.setText(self._element_spec_text(stage))
    self.stage_detail_stress_release.setText("" if stage.get("stress_release") is None else str(stage.get("stress_release")))
    self._set_combo(self.stage_detail_apply_gravity, "true" if self._bool_text(str(stage.get("apply_gravity", True)), True) else "false")
    self.stage_detail_k0.setText("" if stage.get("k0", stage.get("K0")) is None else str(stage.get("k0", stage.get("K0"))))
    self.stage_detail_surface_y.setText("" if stage.get("surface_y", stage.get("ground_y")) is None else str(stage.get("surface_y", stage.get("ground_y"))))
    self.stage_detail_gx.setText("" if stage.get("gx") is None else str(stage.get("gx")))
    self.stage_detail_gy.setText("" if stage.get("gy") is None else str(stage.get("gy")))
    self.stage_detail_scale.setText("" if stage.get("scale") is None else str(stage.get("scale")))
    hydro = self._mapping(stage.get("hydro", stage.get("consolidation", {})))
    self.stage_detail_dt.setText("" if hydro.get("dt", hydro.get("time_step")) is None else str(hydro.get("dt", hydro.get("time_step"))))
    self.stage_detail_steps.setText("" if hydro.get("steps", hydro.get("n_steps")) is None else str(hydro.get("steps", hydro.get("n_steps"))))
    self.stage_detail_storage.setText("" if hydro.get("storage", hydro.get("specific_storage")) is None else str(hydro.get("storage", hydro.get("specific_storage"))))
    self.stage_detail_permeability.setText("" if hydro.get("permeability", hydro.get("k")) is None else str(hydro.get("permeability", hydro.get("k"))))
    self.stage_detail_biot_alpha.setText("" if hydro.get("biot_alpha", hydro.get("alpha")) is None else str(hydro.get("biot_alpha", hydro.get("alpha"))))
    srm = self._mapping(stage.get("srm", stage))
    srm_start = srm.get("factor_start", srm.get("start_factor", srm.get("start", srm.get("fs_start"))))
    srm_end = srm.get("factor_max", srm.get("end_factor", srm.get("max_factor", srm.get("end", srm.get("fs_max")))))
    srm_step = srm.get("factor_step", srm.get("step", srm.get("fs_step")))
    srm_failure = srm.get("failure_plastic_ratio", srm.get("plastic_ratio_limit"))
    self.stage_detail_srm_start.setText("" if srm_start is None else str(srm_start))
    self.stage_detail_srm_end.setText("" if srm_end is None else str(srm_end))
    self.stage_detail_srm_step.setText("" if srm_step is None else str(srm_step))
    if hasattr(self, "stage_detail_srm_failure_ratio"):
        self.stage_detail_srm_failure_ratio.setText("" if srm_failure is None else str(srm_failure))
    riks = self._mapping(stage.get("riks", stage.get("arc_length", {})))
    self.stage_detail_riks_arc.setText("" if riks.get("arc_length", riks.get("ds")) is None else str(riks.get("arc_length", riks.get("ds"))))
    self.stage_detail_riks_steps.setText("" if riks.get("steps", riks.get("max_steps")) is None else str(riks.get("steps", riks.get("max_steps"))))
    self.stage_detail_increments.setText("" if stage.get("increments", stage.get("increment")) is None else str(stage.get("increments", stage.get("increment"))))
    solver = stage.get("solver", {})
    self.stage_detail_solver.setPlainText(yaml.safe_dump(solver, allow_unicode=True, sort_keys=False).strip() if isinstance(solver, Mapping) and solver else "")
    self.apply_stage_recommended_defaults(force=False)
    if hasattr(self, "_refresh_stage_inspector_visibility"):
        self._refresh_stage_inspector_visibility()


def _populate_stage_construction_table(owner: Any, qt: Mapping[str, Any], stage: Mapping[str, Any]) -> None:
    self = owner
    events = self._ensure_list(stage.get("construction_events", stage.get("construction", [])))
    if events:
        for raw in events:
            if not isinstance(raw, Mapping):
                continue
            self.add_stage_construction_row(
                action=str(raw.get("action", raw.get("type", "excavation"))),
                target=self._element_spec_text(raw),
                stress_release=str(raw.get("stress_release", "")),
                reactivate=raw.get("reactivate", raw.get("re_active", "")),
                material=str(raw.get("material", "")),
                extra=self._element_extra_yaml(raw, {"action", "type", "target", "set", "element_set", "elementSet", "element", "elements", "all", "stress_release", "reactivate", "re_active", "material"}),
            )
        return
    stype = str(stage.get("type", "")).lower().strip()
    if stype in {"excavation", "death", "deactivate"} and self._element_spec_text(stage):
        self.add_stage_construction_row(action=stype, target=self._element_spec_text(stage), stress_release=str(stage.get("stress_release", "")))


def _stage_change_stage_label(stage: Mapping[str, Any], index: int) -> str:
    return str(stage.get("name", f"Stage-{index + 1}"))


def _stage_change_display_value(owner: Any, column: int, value: Any) -> str:
    raw = "" if value is None else str(value)
    if getattr(owner, "_show_internal_representation", lambda: True)():
        return raw
    key = raw.strip().lower()
    english = str(getattr(owner, "gui_locale", "ja")).startswith("en")
    labels = {
        1: {
            "death": ("無効化", "Deactivate"),
            "birth": ("再有効化", "Reactivate"),
            "material": ("材料変更", "Material Change"),
            "材料": ("材料変更", "Material Change"),
            "boundary": ("境界変更", "Boundary Change"),
            "境界": ("境界変更", "Boundary Change"),
            "load": ("荷重変更", "Load Change"),
            "荷重": ("荷重変更", "Load Change"),
        },
        2: {"all": ("全対象", "All Targets")},
        3: {
            "death": ("無効化", "Deactivate"),
            "excavation": ("掘削・無効化", "Excavate / Deactivate"),
            "deactivate": ("無効化", "Deactivate"),
            "reactivate": ("再有効化", "Reactivate"),
            "birth": ("再有効化", "Reactivate"),
            "material": ("材料変更", "Material Change"),
            "fixed": ("固定境界", "Fixed Boundary"),
            "node": ("節点荷重", "Nodal Load"),
            "edge": ("分布荷重", "Distributed Load"),
            "gravity": ("自重", "Self Weight"),
        },
    }
    pair = labels.get(column, {}).get(key)
    if pair is None:
        return raw
    return pair[1] if english else pair[0]


def _stage_change_raw_cell(owner: Any, qt: Mapping[str, Any], table: Any, row: int, column: int) -> str:
    item = table.item(row, column)
    if item is None:
        return ""
    visible = str(item.text() or "").strip()
    raw = item.data(qt["Qt"].ItemDataRole.UserRole)
    if raw is None:
        return visible
    raw_text = str(raw).strip()
    return raw_text if visible == _stage_change_display_value(owner, column, raw_text) else visible


def _stage_change_target_text(owner: Any, spec: Mapping[str, Any]) -> str:
    return owner._element_spec_text(spec) or owner._node_spec_text(spec) or owner._edge_spec_text(spec)


def populate_stage_change_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = getattr(self, "stage_change_table", None)
    if table is None:
        return
    table.blockSignals(True)
    table.setRowCount(0)

    def append_row(stage_label: str, kind: str, target: str = "", action: str = "", material: str = "", ux: Any = "", uy: Any = "", fx: Any = "", fy: Any = "", tx: Any = "", ty: Any = "", release: Any = "", extra: Mapping[str, Any] | str | None = None) -> None:
        row = table.rowCount()
        table.insertRow(row)
        if isinstance(extra, Mapping):
            extra_text = yaml.safe_dump(dict(extra), allow_unicode=True, sort_keys=False).strip()
        else:
            extra_text = "" if extra is None else str(extra)
        values = [stage_label, kind, target, action, material, ux, uy, fx, fy, tx, ty, release, extra_text]
        for col, value in enumerate(values):
            raw_text = "" if value is None else str(value)
            item = QTableWidgetItem(_stage_change_display_value(self, col, raw_text))
            item.setData(qt["Qt"].ItemDataRole.UserRole, raw_text)
            table.setItem(row, col, item)

    for index, stage in enumerate(self._stages()):
        stage_label = _stage_change_stage_label(stage, index)
        seen_primary_death = False
        for raw in self._ensure_list(stage.get("construction_events", stage.get("construction", []))):
            if not isinstance(raw, Mapping):
                continue
            action = str(raw.get("action", raw.get("type", "")) or "").strip().lower()
            target = self._element_spec_text(raw)
            extra = {
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "action",
                    "type",
                    "target",
                    "set",
                    "element_set",
                    "elementSet",
                    "element",
                    "elements",
                    "all",
                    "stress_release",
                    "reactivate",
                    "re_active",
                    "material",
                }
            }
            if action in {"excavation", "death", "deactivate", "remove"}:
                seen_primary_death = True
                append_row(stage_label, "Death", target, action or "death", "", "", "", "", "", "", "", raw.get("stress_release", ""), extra)
            elif action in {"reactivate", "activate", "birth"}:
                append_row(stage_label, "Birth", target, action or "reactivate", "", "", "", "", "", "", "", "", extra)
            elif action in {"material", "property", "element_property"}:
                append_row(stage_label, "材料", target, action or "material", raw.get("material", ""), "", "", "", "", "", "", "", extra)
            else:
                append_row(stage_label, "要素", target, action or "event", raw.get("material", ""), "", "", "", "", "", "", raw.get("stress_release", ""), extra)
        stype = str(stage.get("type", "")).lower().strip()
        if not seen_primary_death and stype in {"excavation", "death", "deactivate"} and self._element_spec_text(stage):
            append_row(stage_label, "Death", self._element_spec_text(stage), stype, "", "", "", "", "", "", "", stage.get("stress_release", ""))
        for raw in self._ensure_list(stage.get("element_properties", [])):
            if not isinstance(raw, Mapping):
                continue
            extra = {k: v for k, v in raw.items() if k not in {"set", "element_set", "elementSet", "element", "elements", "all", "material"}}
            append_row(stage_label, "材料", self._element_spec_text(raw), "material", raw.get("material", ""), extra=extra)
        for raw in self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))):
            if not isinstance(raw, Mapping):
                continue
            extra = {k: v for k, v in raw.items() if k not in {"node", "nodes", "set", "all", "ux", "uy"}}
            append_row(stage_label, "境界", self._node_spec_text(raw), str(raw.get("kind", raw.get("type", ""))), "", raw.get("ux", ""), raw.get("uy", ""), extra=extra)
        for raw in self._ensure_list(stage.get("loads", [])):
            if not isinstance(raw, Mapping):
                continue
            extra = {k: v for k, v in raw.items() if k not in {"type", "node", "nodes", "edge", "edges", "element", "elements", "set", "all", "fx", "fy", "tx", "ty", "scale"}}
            target = self._node_spec_text(raw) or self._edge_spec_text(raw) or self._element_spec_text(raw)
            append_row(stage_label, "荷重", target, str(raw.get("type", "")), "", "", "", raw.get("fx", ""), raw.get("fy", ""), raw.get("tx", ""), raw.get("ty", ""), "", extra)
    table.blockSignals(False)


def add_stage_construction_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.stage_construction_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"action": "excavation", "target": "all", "stress_release": "", "reactivate": "", "material": "", "extra": ""}
    defaults.update(values)
    action = str(defaults.get("action", "excavation") or "excavation").strip().lower()
    defaults["action"] = action
    if not str(defaults.get("target", "")).strip():
        defaults["target"] = "all"
    if action in {"excavation", "death", "deactivate", "remove"}:
        if not str(defaults.get("stress_release", "")).strip():
            defaults["stress_release"] = "1.0"
    elif action in {"reactivate", "activate"}:
        defaults["stress_release"] = str(values.get("stress_release", "") or "")
        if defaults.get("reactivate") in {"", None, False}:
            defaults["reactivate"] = "true"
    elif action in {"material", "property", "element_property"}:
        defaults["stress_release"] = str(values.get("stress_release", "") or "")
        if not str(defaults.get("material", "")).strip():
            defaults["material"] = _first_material_name(self)
    if defaults.get("reactivate") is True:
        defaults["reactivate"] = "true"
    elif defaults.get("reactivate") is False:
        defaults["reactivate"] = ""
    extra = defaults.get("extra", "")
    if isinstance(extra, Mapping):
        extra = yaml.safe_dump(dict(extra), allow_unicode=True, sort_keys=False).strip()
    for col, key in enumerate(["action", "target", "stress_release", "reactivate", "material", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(extra if key == "extra" else defaults.get(key, ""))))


def add_stage_change_row(owner: Any, qt: Mapping[str, Any], kind: str = "death") -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = getattr(self, "stage_change_table", None)
    if table is None:
        return
    stages, row_index, stage = self._selected_stage_context()
    stage_label = str(stage.get("name", f"Stage-{row_index + 1}"))
    normalized = str(kind or "death").strip().lower()
    presets = {
        "death": ["Death", "all", "death", "", "", "", "", "", "", "", "", "1.0", ""],
        "birth": ["Birth", "all", "reactivate", "", "", "", "", "", "", "", "", "", ""],
        "material": ["材料", "all", "material", _first_material_name(self), "", "", "", "", "", "", "", "", ""],
        "boundary": ["境界", "left", "fixed", "", "0.0", "0.0", "", "", "", "", "", "", ""],
        "load": ["荷重", "right", "node", "", "", "", "0.0", "-10.0", "", "", "", "", ""],
    }
    values = presets.get(normalized, presets["death"])
    row = table.rowCount()
    table.insertRow(row)
    for col, value in enumerate([stage_label, *values]):
        raw_text = str(value)
        item = QTableWidgetItem(_stage_change_display_value(self, col, raw_text))
        item.setData(qt["Qt"].ItemDataRole.UserRole, raw_text)
        table.setItem(row, col, item)
    table.selectRow(row)


def _stage_index_from_change_label(owner: Any, stages: list[dict[str, Any]], label: str) -> int:
    text = str(label or "").strip()
    selected = owner._selected_stage_row()
    default_index = selected if isinstance(selected, int) and 0 <= selected < len(stages) else 0
    if not text:
        if not stages:
            stages.append({"name": "Stage-1", "type": "static"})
        return default_index
    for index, stage in enumerate(stages):
        name = str(stage.get("name", f"Stage-{index + 1}") or f"Stage-{index + 1}")
        if text == name or text.lower() == name.lower() or text == str(index + 1):
            return index
    lowered = text.lower()
    if lowered.startswith("stage-"):
        try:
            index = int(lowered.split("-", 1)[1]) - 1
        except Exception:
            index = -1
        if index >= 0:
            while len(stages) <= index:
                stages.append({"name": f"Stage-{len(stages) + 1}", "type": "static"})
            return index
    stages.append({"name": text, "type": "static"})
    return len(stages) - 1


def _stage_change_float(owner: Any, table: Any, row: int, col: int, label: str) -> float | None:
    text = owner._table_text(table, row, col).strip()
    if not text:
        return None
    return owner._float_table_cell(table, row, col, label)


def apply_stage_change_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    table = getattr(self, "stage_change_table", None)
    if table is None:
        return
    stages = [dict(stage) if isinstance(stage, Mapping) else {"name": f"Stage-{index + 1}", "type": "static"} for index, stage in enumerate(self._stages())]
    if not stages:
        stages.append({"name": "Stage-1", "type": "static"})
    for stage in stages:
        for key in ("construction_events", "construction", "element_properties", "boundary_conditions", "bc", "loads"):
            stage.pop(key, None)
    self.clear_table_cell_errors(table)
    selected_row_after_apply = self._selected_stage_row()
    try:
        for row in range(table.rowCount()):
            raw_values = [_stage_change_raw_cell(self, qt, table, row, col) for col in range(13)]
            if not any(raw_values):
                continue
            stage_index = _stage_index_from_change_label(self, stages, raw_values[0])
            stage = stages[stage_index]
            selected_row_after_apply = stage_index
            kind = raw_values[1]
            target = raw_values[2]
            action = raw_values[3]
            material = raw_values[4]
            release = raw_values[11]
            extra = self._yaml_mapping_text(raw_values[12], f"stage change row {row + 1} extra YAML")
            kind_lower = kind.lower()
            action_lower = action.lower()
            has_boundary_values = bool(raw_values[5] or raw_values[6])
            has_node_load_values = bool(raw_values[7] or raw_values[8])
            has_edge_load_values = bool(raw_values[9] or raw_values[10])
            is_death = kind_lower in {"death", "excavation", "deactivate", "remove", "掘削"} or action_lower in {"death", "excavation", "deactivate", "remove"}
            is_birth = kind_lower in {"birth", "reactivate", "activate", "復旧", "再有効化"} or action_lower in {"birth", "reactivate", "activate"}
            is_material = "material" in kind_lower or "材料" in kind or action_lower in {"material", "property", "element_property"}
            is_boundary = "boundary" in kind_lower or "境界" in kind or has_boundary_values
            is_load = "load" in kind_lower or "荷重" in kind or has_node_load_values or has_edge_load_values or action_lower in {"node", "edge", "body", "gravity", "traction"}
            if is_death or is_birth:
                event = dict(extra)
                event["action"] = action_lower or ("reactivate" if is_birth else "death")
                if target:
                    event.update(self._target_element_spec(target))
                else:
                    event["all"] = True
                if is_birth:
                    event.setdefault("reactivate", True)
                if release:
                    event["stress_release"] = self._float_table_cell(table, row, 11, "stress_release")
                self._list_value(stage, "construction_events").append(event)
                if is_death:
                    stage["type"] = "excavation" if event["action"] == "excavation" else "death"
                    for target_key in ("set", "element_set", "elementSet", "element", "elements", "all"):
                        stage.pop(target_key, None)
                    for target_key in ("set", "element_set", "elementSet", "element", "elements", "all"):
                        if target_key in event:
                            stage[target_key] = event[target_key]
                    if "stress_release" in event:
                        stage["stress_release"] = event["stress_release"]
                continue
            if is_material:
                if not material:
                    raise ValueError(f"stage change row {row + 1}: 材料名を指定してください。")
                prop = dict(extra)
                if target:
                    prop.update(self._target_element_spec(target))
                else:
                    prop["all"] = True
                prop["material"] = material
                self._list_value(stage, "element_properties").append(prop)
                continue
            if is_boundary:
                bc = dict(extra)
                if target:
                    bc.update(self._target_node_spec(target))
                ux = _stage_change_float(self, table, row, 5, "ux")
                uy = _stage_change_float(self, table, row, 6, "uy")
                if ux is not None:
                    bc["ux"] = ux
                if uy is not None:
                    bc["uy"] = uy
                if action and action_lower not in {"boundary", "bc"}:
                    bc.setdefault("type", action)
                if not any(key in bc for key in ("ux", "uy", "fixed", "dof", "dofs", "type")):
                    raise ValueError(f"stage change row {row + 1}: 境界条件のux/uyまたは種類を指定してください。")
                self._list_value(stage, "boundary_conditions").append(bc)
                continue
            if is_load:
                load = dict(extra)
                if has_edge_load_values:
                    load.update(self._target_load_edge_spec(target) if target else {"edges": "all"})
                    tx = _stage_change_float(self, table, row, 9, "tx")
                    ty = _stage_change_float(self, table, row, 10, "ty")
                    if tx is not None:
                        load["tx"] = tx
                    if ty is not None:
                        load["ty"] = ty
                    load.setdefault("type", action or "edge")
                elif action_lower in {"body", "gravity"}:
                    if target:
                        load.update(self._target_body_load_spec(target))
                    fx = _stage_change_float(self, table, row, 7, "fx")
                    fy = _stage_change_float(self, table, row, 8, "fy")
                    if fx is not None:
                        load["gx"] = fx
                    if fy is not None:
                        load["gy"] = fy
                    load.setdefault("type", "gravity")
                else:
                    if target:
                        load.update(self._target_node_spec(target))
                    fx = _stage_change_float(self, table, row, 7, "fx")
                    fy = _stage_change_float(self, table, row, 8, "fy")
                    if fx is not None:
                        load["fx"] = fx
                    if fy is not None:
                        load["fy"] = fy
                    load.setdefault("type", action or "node")
                self._list_value(stage, "loads").append(load)
                continue
            raise ValueError(f"stage change row {row + 1}: 区分を 無効化/再有効化/材料/境界/荷重 のいずれかで指定してください。")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"ステージ変更表の入力が不正です: {exc}")
        return
    self.cfg["stages"] = stages
    self._after_form_change("ステージ変更表を反映しました")
    if isinstance(selected_row_after_apply, int) and getattr(self, "stage_table", None) is not None:
        self.stage_table.selectRow(max(0, min(selected_row_after_apply, self.stage_table.rowCount() - 1)))


def _apply_stage_detail_form_values(owner: Any, qt: Mapping[str, Any], stage: dict[str, Any]) -> None:
    self = owner
    name = self.stage_detail_name.text().strip()
    if name:
        stage["name"] = name
    stage_type = self._combo_value(self.stage_detail_type, "static").strip() or "static"
    stage["type"] = stage_type
    for key in ("set", "element_set", "elementSet", "element", "elements", "all"):
        stage.pop(key, None)
    target = self.stage_detail_target.text().strip()
    if target:
        stage.update(self._target_element_spec(target))
    release = self.stage_detail_stress_release.text().strip()
    if release:
        stage["stress_release"] = self._float_text(release, "stress_release")
    else:
        stage.pop("stress_release", None)
    stage["apply_gravity"] = self._bool_text(self._combo_value(self.stage_detail_apply_gravity, "true"), True)
    for attr, key, label in [
        ("stage_detail_k0", "k0", "k0"),
        ("stage_detail_surface_y", "surface_y", "surface_y"),
        ("stage_detail_gx", "gx", "gx"),
        ("stage_detail_gy", "gy", "gy"),
        ("stage_detail_scale", "scale", "scale"),
    ]:
        text = getattr(self, attr).text().strip()
        if text:
            stage[key] = self._float_text(text, label)
        else:
            stage.pop(key, None)
    hydro = dict(self._mapping(stage.get("hydro", stage.get("consolidation", {}))))
    for attr, key, label, integer in [
        ("stage_detail_dt", "dt", "dt", False),
        ("stage_detail_steps", "steps", "steps", True),
        ("stage_detail_storage", "storage", "storage", False),
        ("stage_detail_permeability", "permeability", "permeability", False),
        ("stage_detail_biot_alpha", "biot_alpha", "biot_alpha", False),
    ]:
        text = getattr(self, attr).text().strip()
        if text:
            value = self._float_text(text, label)
            hydro[key] = int(value) if integer else value
        else:
            hydro.pop(key, None)
    if hydro:
        stage["hydro"] = hydro
    elif str(stage_type).lower() in {"consolidation", "u-p", "u_p", "up", "coupled_consolidation"}:
        stage["hydro"] = {"dt": 1.0, "steps": 1}
    else:
        stage.pop("hydro", None)
    srm_values: dict[str, Any] = {}
    for attr, key, label in [
        ("stage_detail_srm_start", "factor_start", "srm start"),
        ("stage_detail_srm_end", "factor_max", "srm end"),
        ("stage_detail_srm_step", "factor_step", "srm step"),
        ("stage_detail_srm_failure_ratio", "failure_plastic_ratio", "srm failure ratio"),
    ]:
        if not hasattr(self, attr):
            continue
        text = getattr(self, attr).text().strip()
        if text:
            srm_values[key] = self._float_text(text, label)
    if srm_values:
        stage["srm"] = {**dict(self._mapping(stage.get("srm", {}))), **srm_values}
    elif "srm" in stage and str(stage_type).lower() not in {"srm", "safety_factor"}:
        stage.pop("srm", None)
    riks_values: dict[str, Any] = {}
    arc_text = self.stage_detail_riks_arc.text().strip()
    if arc_text:
        riks_values["arc_length"] = self._float_text(arc_text, "riks arc_length")
    steps_text = self.stage_detail_riks_steps.text().strip()
    if steps_text:
        riks_values["steps"] = int(self._float_text(steps_text, "riks steps"))
    if riks_values:
        existing_riks = dict(self._mapping(stage.get("arc_length", stage.get("riks", {}))))
        stage["arc_length"] = {**existing_riks, **riks_values}
    elif "arc_length" in stage and str(stage_type).lower() not in {"riks", "arc_length", "arclength"}:
        stage.pop("arc_length", None)
    increments = self.stage_detail_increments.text().strip()
    if increments:
        stage["increments"] = int(self._float_text(increments, "increments"))
    else:
        stage.pop("increments", None)
    solver_text = self.stage_detail_solver.toPlainText().strip()
    if solver_text:
        stage["solver"] = self._yaml_mapping_text(solver_text, "stage solver YAML")
    else:
        stage.pop("solver", None)


def _construction_events_from_table(owner: Any, qt: Mapping[str, Any], stage: dict[str, Any], element_properties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    self = owner
    events: list[dict[str, Any]] = []
    first_deactivation: dict[str, Any] | None = None
    for row in range(self.stage_construction_table.rowCount()):
        action = self._table_text(self.stage_construction_table, row, 0).strip().lower() or "excavation"
        target = self._table_text(self.stage_construction_table, row, 1).strip()
        release_text = self._table_text(self.stage_construction_table, row, 2).strip()
        reactivate_text = self._table_text(self.stage_construction_table, row, 3).strip()
        material = self._table_text(self.stage_construction_table, row, 4).strip()
        extra = self._yaml_mapping_text(self._table_text(self.stage_construction_table, row, 5), "construction extra YAML")
        if not target and not material and not extra:
            continue
        event = dict(extra)
        event["action"] = action
        if target:
            event.update(self._target_element_spec(target))
        if release_text:
            event["stress_release"] = self._float_text(release_text, "construction stress_release")
        if reactivate_text:
            event["reactivate"] = self._bool_text(reactivate_text, False)
        if material:
            event["material"] = material
        events.append(event)
        if action in {"excavation", "death", "deactivate", "remove"} and first_deactivation is None:
            first_deactivation = event
        if action in {"material", "property", "element_property"} and material:
            prop = {k: v for k, v in event.items() if k not in {"action", "stress_release", "reactivate"}}
            element_properties.append(prop)
    if first_deactivation is not None:
        stage["type"] = "excavation" if first_deactivation.get("action") == "excavation" else "death"
        for key in ("set", "element_set", "elementSet", "element", "elements", "all"):
            stage.pop(key, None)
        for key in ("set", "element_set", "elementSet", "element", "elements", "all"):
            if key in first_deactivation:
                stage[key] = first_deactivation[key]
        if "stress_release" in first_deactivation:
            stage["stress_release"] = first_deactivation["stress_release"]
    return events


def add_stage_material_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.stage_material_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"target": "all", "material": "soil", "extra": ""}
    defaults.update(values)
    for col, key in enumerate(["target", "material", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_stage_boundary_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.stage_boundary_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"target": "left", "ux": "0.0", "uy": "0.0", "extra": ""}
    defaults.update(values)
    for col, key in enumerate(["target", "ux", "uy", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_stage_load_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.stage_load_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"load_type": "node", "target": "right", "fx": "0.0", "fy": "-10.0", "tx": "", "ty": "", "scale": "", "extra": ""}
    defaults.update(values)
    for col, key in enumerate(["load_type", "target", "fx", "fy", "tx", "ty", "scale", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_stage_hydro_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.stage_hydro_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"kind": "pressure", "target": "top", "value": "0.0", "extra": ""}
    defaults.update(values)
    for col, key in enumerate(["kind", "target", "value", "extra"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def add_stage_mpc_row(owner: Any, qt: Mapping[str, Any], **values: Any) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = self.stage_mpc_table
    row = table.rowCount()
    table.insertRow(row)
    defaults = {"master": "1", "slave": "2", "dof": "ux", "coefficient": "1.0", "value": "0.0", "method": "elimination"}
    defaults.update(values)
    for col, key in enumerate(["master", "slave", "dof", "coefficient", "value", "method"]):
        table.setItem(row, col, QTableWidgetItem(str(defaults.get(key, ""))))


def apply_stage_detail_tables(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    row = self._selected_stage_row()
    stages = self._stages()
    if row is None or row < 0 or row >= len(stages):
        _notify_information(self, QMessageBox, "詳細を編集するステージを選択してください。")
        return
    stage = dict(stages[row])
    self.clear_table_cell_errors(
        self.stage_material_table,
        self.stage_boundary_table,
        self.stage_load_table,
        self.stage_hydro_table,
        self.stage_mpc_table,
        self.stage_construction_table,
    )
    try:
        self._apply_stage_detail_form_values(stage)
        element_properties: list[dict[str, Any]] = []
        for table_row in range(self.stage_material_table.rowCount()):
            target = self._table_text(self.stage_material_table, table_row, 0).strip()
            material = self._table_text(self.stage_material_table, table_row, 1).strip()
            extra = self._yaml_mapping_text(self._table_text(self.stage_material_table, table_row, 2), "stage material extra YAML")
            if not target and not material and not extra:
                continue
            prop = dict(extra)
            if target:
                prop.update(self._target_element_spec(target))
            if material:
                prop["material"] = material
            if not prop.get("material"):
                raise ValueError(f"材料変更 {table_row + 1}: materialを指定してください。")
            element_properties.append(prop)
        construction_events = self._construction_events_from_table(stage, element_properties)

        boundary_conditions: list[dict[str, Any]] = []
        for table_row in range(self.stage_boundary_table.rowCount()):
            target = self._table_text(self.stage_boundary_table, table_row, 0).strip()
            ux_text = self._table_text(self.stage_boundary_table, table_row, 1).strip()
            uy_text = self._table_text(self.stage_boundary_table, table_row, 2).strip()
            extra = self._yaml_mapping_text(self._table_text(self.stage_boundary_table, table_row, 3), "stage boundary extra YAML")
            if not target and not extra:
                continue
            bc = dict(extra)
            if target:
                bc.update(self._target_node_spec(target))
            if ux_text:
                bc["ux"] = self._float_table_cell(self.stage_boundary_table, table_row, 1, "ux")
            if uy_text:
                bc["uy"] = self._float_table_cell(self.stage_boundary_table, table_row, 2, "uy")
            if not any(key in bc for key in ("ux", "uy", "fixed", "dof", "dofs")):
                raise ValueError(f"境界 {table_row + 1}: ux/uyまたはfixed/dofを指定してください。")
            boundary_conditions.append(bc)

        loads = self._loads_from_table(self.stage_load_table)

        hydro: dict[str, Any] = dict(self._mapping(stage.get("hydro", stage.get("consolidation", {}))))
        for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs"):
            hydro.pop(key, None)
        for table_row in range(self.stage_hydro_table.rowCount()):
            kind = self._table_text(self.stage_hydro_table, table_row, 0).strip().lower() or "pressure"
            target = self._table_text(self.stage_hydro_table, table_row, 1).strip()
            value_text = self._table_text(self.stage_hydro_table, table_row, 2).strip()
            extra = self._yaml_mapping_text(self._table_text(self.stage_hydro_table, table_row, 3), "stage hydro extra YAML")
            if not target and not extra:
                continue
            spec = dict(extra)
            value = self._float_table_cell(self.stage_hydro_table, table_row, 2, "hydro value", fallback=str(spec.get("value", 0.0)))
            if kind in {"node_pressure", "pore_pressure_node"}:
                if target:
                    spec.update(self._target_node_spec(target))
                spec["pressure"] = value
                self._list_value(hydro, "pressure_bcs").append(spec)
            elif kind in {"flux", "pore_flux"}:
                if target:
                    spec.update(self._target_edge_spec(target))
                spec["flux"] = value
                self._list_value(hydro, "pore_flux_bcs").append(spec)
            elif kind in {"robin", "pore_robin"}:
                if target:
                    spec.update(self._target_edge_spec(target))
                spec["beta"] = value
                spec.setdefault("pressure", 0.0)
                self._list_value(hydro, "pore_robin_bcs").append(spec)
            else:
                if target:
                    spec.update(self._target_edge_spec(target))
                spec["pressure"] = value
                self._list_value(hydro, "pressure_bcs").append(spec)

        mpcs: list[dict[str, Any]] = []
        for table_row in range(self.stage_mpc_table.rowCount()):
            master = self._table_text(self.stage_mpc_table, table_row, 0).strip()
            slave = self._table_text(self.stage_mpc_table, table_row, 1).strip()
            dof = self._table_text(self.stage_mpc_table, table_row, 2).strip() or "ux"
            coefficient = self._table_text(self.stage_mpc_table, table_row, 3).strip() or "1.0"
            value = self._table_text(self.stage_mpc_table, table_row, 4).strip() or "0.0"
            method = self._table_text(self.stage_mpc_table, table_row, 5).strip() or "elimination"
            if not master and not slave:
                continue
            if not master or not slave:
                raise ValueError(f"MPC {table_row + 1}: master/slaveを指定してください。")
            mpcs.append(
                {
                    "master": master,
                    "slave": slave,
                    "dof": dof,
                    "coefficient": self._float_table_cell(self.stage_mpc_table, table_row, 3, "coefficient"),
                    "value": self._float_table_cell(self.stage_mpc_table, table_row, 4, "value"),
                    "method": method,
                }
            )
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", f"ステージ詳細の入力が不正です: {exc}")
        return
    if construction_events:
        stage["construction_events"] = construction_events
    else:
        stage.pop("construction_events", None)
        stage.pop("construction", None)
    if element_properties:
        stage["element_properties"] = element_properties
    else:
        stage.pop("element_properties", None)
    if boundary_conditions:
        stage["boundary_conditions"] = boundary_conditions
    else:
        stage.pop("boundary_conditions", None)
        stage.pop("bc", None)
    if loads:
        stage["loads"] = loads
    else:
        stage.pop("loads", None)
    if any(hydro.get(key) for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs")) or hydro:
        stage["hydro"] = hydro
    else:
        stage.pop("hydro", None)
    if mpcs:
        stage["mpc_constraints"] = mpcs
    else:
        stage.pop("mpc_constraints", None)
        stage.pop("mpc", None)
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change("選択ステージ詳細を反映しました")
    self.stage_table.selectRow(row)
    self.refresh_stage_difference_table()


def apply_selected_elements_to_stage_set(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    entities = self._selected_preview_entities()
    element_ids = sorted(entities["elements"], key=self._natural_sort_key)
    if not element_ids:
        _notify_information(self, QMessageBox, "モデルビューで要素を選択してください。")
        return
    stages, row, stage = self._selected_stage_context()
    default_name = f"{stage.get('name', f'Stage-{row + 1}')}_selection".replace(" ", "_")
    set_name, ok = QInputDialog.getText(self, "選択要素→施工set", "element set名", text=default_name)
    if not ok or not set_name.strip():
        return
    mesh_cfg = self._mesh_cfg()
    element_sets = mesh_cfg.setdefault("element_sets", {})
    if not isinstance(element_sets, dict):
        element_sets = {}
        mesh_cfg["element_sets"] = element_sets
    element_sets[set_name.strip()] = element_ids
    stage["set"] = set_name.strip()
    stage.setdefault("selection_history", []).append({"kind": "elements", "count": len(element_ids), "set": set_name.strip(), "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択要素 {len(element_ids)} 件を施工set '{set_name.strip()}' に反映しました")
    self.stage_table.selectRow(row)


def _selected_stage_element_ids(owner: Any) -> list[str]:
    entities = owner._selected_preview_entities()
    return sorted((str(element_id) for element_id in entities.get("elements", set())), key=owner._natural_sort_key)


def _merge_selected_stage_death_target(owner: Any, stage: dict[str, Any], element_ids: list[str]) -> None:
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(owner.cfg)
        existing = set(owner._stage_element_targets(mesh, stage))
    except Exception:
        existing = set()
    merged = set(element_ids)
    if not bool(stage.get("all", False)):
        merged.update(existing)
    for target_key in ("set", "element_set", "elementSet", "element", "elements", "all"):
        stage.pop(target_key, None)
    stage["elements"] = sorted(merged, key=owner._natural_sort_key)


def add_selected_elements_to_stage_death(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    element_ids = _selected_stage_element_ids(self)
    if not element_ids:
        _notify_information(self, QMessageBox, "モデルビューで無効化する要素を選択してください。")
        return
    stages, row, stage = self._selected_stage_context()
    release_text = str(getattr(self, "stage_detail_stress_release", None).text() if getattr(self, "stage_detail_stress_release", None) is not None else "")
    if not release_text.strip():
        release_text = str(stage.get("stress_release", "1.0") or "1.0")
    try:
        release = float(release_text)
    except ValueError:
        release = 1.0
    event = {"action": "death", "elements": element_ids, "stress_release": release}
    self._list_value(stage, "construction_events").append(event)
    stage["type"] = "death"
    stage["stress_release"] = release
    _merge_selected_stage_death_target(self, stage, element_ids)
    stage.setdefault("selection_history", []).append({"kind": "stage_death", "element_count": len(element_ids), "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択要素 {len(element_ids)} 件をDeathに追加しました")
    self.stage_table.selectRow(row)


def add_selected_elements_to_stage_birth(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    element_ids = _selected_stage_element_ids(self)
    if not element_ids:
        _notify_information(self, QMessageBox, "モデルビューで再有効化する要素を選択してください。")
        return
    stages, row, stage = self._selected_stage_context()
    event = {"action": "reactivate", "elements": element_ids, "reactivate": True}
    self._list_value(stage, "construction_events").append(event)
    stage.setdefault("selection_history", []).append({"kind": "stage_birth", "element_count": len(element_ids), "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択要素 {len(element_ids)} 件をBirthに追加しました")
    self.stage_table.selectRow(row)


def add_selected_elements_to_stage_material_change(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    element_ids = _selected_stage_element_ids(self)
    if not element_ids:
        _notify_information(self, QMessageBox, "モデルビューで材料変更対象の要素を選択してください。")
        return
    materials = self._material_names()
    if not materials:
        QMessageBox.warning(self, "GeoFEM", "材料を先に定義してください。")
        return
    material, ok = QInputDialog.getItem(self, "ステージ材料変更", "材料", materials, 0, False)
    if not ok or not str(material).strip():
        return
    stages, row, stage = self._selected_stage_context()
    prop = {"elements": element_ids, "material": str(material).strip()}
    self._list_value(stage, "element_properties").append(prop)
    stage.setdefault("selection_history", []).append({"kind": "stage_material", "element_count": len(element_ids), "material": str(material).strip(), "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択要素 {len(element_ids)} 件へステージ材料 {material} を追加しました")
    self.stage_table.selectRow(row)


def add_selected_prescribed_displacement(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    nodes = self._selected_nodes_for_stage_action()
    if not nodes:
        _notify_information(self, QMessageBox, "モデルビューで節点または辺を選択してください。")
        return
    text, ok = QInputDialog.getText(self, "一括強制変位", "ux,uy（空欄は未指定）", text="0.0,")
    if not ok:
        return
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2 or (not parts[0] and not parts[1]):
        QMessageBox.warning(self, "GeoFEM", "ux,uy の2項目で入力してください。例: 0.0, または , -0.01")
        return
    bc: dict[str, Any] = {"nodes": nodes}
    try:
        if parts[0]:
            bc["ux"] = float(parts[0])
        if parts[1]:
            bc["uy"] = float(parts[1])
    except ValueError:
        QMessageBox.warning(self, "GeoFEM", "ux/uy は数値で入力してください。")
        return
    stages, row, stage = self._selected_stage_context()
    self._list_value(stage, "boundary_conditions").append(bc)
    stage.setdefault("selection_history", []).append({"kind": "prescribed_displacement", "node_count": len(nodes), "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択節点 {len(nodes)} 件へ一括強制変位を追加しました")
    self.stage_table.selectRow(row)


def add_selected_edge_load(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    edges = self._selected_edges_for_stage_action()
    if not edges:
        _notify_information(self, QMessageBox, "モデルビューで辺または要素を選択してください。")
        return
    text, ok = QInputDialog.getText(self, "選択辺→分布荷重", "tx,ty", text="0.0,-10.0")
    if not ok:
        return
    try:
        tx, ty = self._parse_float_csv(text, 2, "tx,ty")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    stages, row, stage = self._selected_stage_context()
    load = {"edges": [[a, b] for a, b in edges], "tx": tx, "ty": ty}
    self._list_value(stage, "loads").append(load)
    stage.setdefault("selection_history", []).append({"kind": "edge_load", "edge_count": len(edges), "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"選択辺 {len(edges)} 件へ分布荷重を追加しました")
    self.stage_table.selectRow(row)


def add_pseudo_static_earthquake_load(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    text, ok = QInputDialog.getText(self, "地震荷重", "水平震度kh, 鉛直震度kv", text="0.10,0.0")
    if not ok:
        return
    try:
        kh, kv = self._parse_float_csv(text, 2, "kh,kv")
    except ValueError as exc:
        QMessageBox.warning(self, "GeoFEM", str(exc))
        return
    stages, row, stage = self._selected_stage_context()
    load = {"type": "gravity", "gx": kh, "gy": kv, "scale": 1.0, "seismic": {"method": "pseudo_static", "kh": kh, "kv": kv}}
    self._list_value(stage, "loads").append(load)
    stage.setdefault("selection_history", []).append({"kind": "earthquake", "kh": kh, "kv": kv, "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"地震荷重 kh={kh:g}, kv={kv:g} を追加しました")
    self.stage_table.selectRow(row)


def add_selected_hydro_coupling(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QInputDialog = qt["QInputDialog"]
    QMessageBox = qt["QMessageBox"]
    edges = self._selected_edges_for_stage_action()
    nodes = self._selected_nodes_for_stage_action(include_elements=False)
    if not edges and not nodes:
        _notify_information(self, QMessageBox, "モデルビューで水理境界にする辺または節点を選択してください。")
        return
    options = ["pressure", "flux", "robin"] if edges else ["node_pressure"]
    kind, ok = QInputDialog.getItem(self, "水圧連携", "水理条件", options, 0, False)
    if not ok:
        return
    value, ok = QInputDialog.getDouble(self, "水圧連携", "値", 0.0, -1.0e12, 1.0e12, 6)
    if not ok:
        return
    stages, row, stage = self._selected_stage_context()
    if str(stage.get("type", "")).lower() not in {"consolidation", "u-p", "u_p", "up", "coupled_consolidation"}:
        stage["type"] = "consolidation"
    analysis = dict(self._mapping(self.cfg.get("analysis", {})))
    fields = {str(field).lower() for field in self._ensure_list(analysis.get("fields", ["u"]))}
    fields.update({"u", "p"})
    analysis["fields"] = sorted(fields)
    self.cfg["analysis"] = analysis
    hydro = stage.setdefault("hydro", {})
    if not isinstance(hydro, dict):
        hydro = {}
        stage["hydro"] = hydro
    if kind == "node_pressure":
        spec = {"nodes": nodes, "pressure": value, "source": "gui_selection"}
        self._list_value(hydro, "pressure_bcs").append(spec)
        count = len(nodes)
    elif kind == "flux":
        spec = {"edges": [[a, b] for a, b in edges], "flux": value, "source": "gui_selection"}
        self._list_value(hydro, "pore_flux_bcs").append(spec)
        count = len(edges)
    elif kind == "robin":
        spec = {"edges": [[a, b] for a, b in edges], "beta": max(value, 0.0), "pressure": 0.0, "source": "gui_selection"}
        self._list_value(hydro, "pore_robin_bcs").append(spec)
        count = len(edges)
    else:
        spec = {"edges": [[a, b] for a, b in edges], "pressure": value, "source": "gui_selection"}
        self._list_value(hydro, "pressure_bcs").append(spec)
        count = len(edges)
    stage.setdefault("selection_history", []).append({"kind": "hydro", "condition": kind, "count": count, "at": datetime.now().isoformat(timespec="seconds")})
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"水圧連携 {kind} を {count} 件追加しました")
    self.stage_table.selectRow(row)


def _stage_difference_rows(owner: Any, qt: Mapping[str, Any], row: int | None = None) -> list[tuple[str, str, str, str]]:
    self = owner
    stages = self._stages()
    if not stages:
        return []
    if row is None:
        row = self._selected_stage_row()
    if row is None:
        row = len(stages) - 1
    row = max(0, min(row, len(stages) - 1))
    previous = stages[row - 1] if row > 0 else {}
    current = stages[row]
    rows: list[tuple[str, str, str, str]] = []
    labels = {
        "type": "解析種別",
        "set": "施工対象",
        "element_set": "施工対象",
        "elements": "施工対象",
        "element": "施工対象",
        "all": "施工対象",
        "stress_release": "応力解放",
        "construction_events": "施工/death",
        "element_properties": "材料変更",
        "boundary_conditions": "境界変更",
        "loads": "荷重変更",
        "hydro": "水理条件",
        "consolidation": "水理条件",
        "mpc_constraints": "MPC",
    }
    for key in sorted(set(previous) | set(current)):
        if key in {"name", "selection_history"}:
            continue
        before = previous.get(key)
        after = current.get(key)
        if self._stable_json(before) == self._stable_json(after):
            continue
        rows.append((labels.get(key, key), key, self._short_json(before), self._short_json(after)))
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
        before_active = self._active_elements_after_stage_index(mesh, row - 1)
        after_active = self._active_elements_after_stage_index(mesh, row)
        removed = sorted(before_active - after_active, key=self._natural_sort_key)
        added = sorted(after_active - before_active, key=self._natural_sort_key)
        if removed:
            rows.append(("施工/death", "deactivated elements", "-", ", ".join(removed[:80])))
        if added:
            rows.append(("施工/death", "reactivated elements", "-", ", ".join(added[:80])))
    except Exception:
        pass
    return rows


def refresh_stage_difference_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    Qt = qt["Qt"]
    table = getattr(self, "stage_diff_table", None)
    if table is None:
        return
    rows = self._stage_difference_rows()
    table.setRowCount(len(rows))
    for row_index, row_values in enumerate(rows):
        approval = self._stage_diff_approval_status(row_values[1])
        for col, value in enumerate((*row_values, approval)):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, row_values[1])
            table.setItem(row_index, col, item)
    self.update_preview()


def _stage_diff_approval_key(owner: Any, qt: Mapping[str, Any], diff_key: str) -> str:
    self = owner
    row = self._selected_stage_row()
    stage_name = ""
    stages = self._stages()
    if row is not None and 0 <= row < len(stages):
        stage_name = str(stages[row].get("name", f"Stage-{row + 1}"))
    return f"{row}:{stage_name}:{diff_key}"


def _stage_diff_approval_status(owner: Any, qt: Mapping[str, Any], diff_key: str) -> str:
    self = owner
    approvals = self._mapping(self.cfg.get("stage_diff_approvals", {}))
    record = approvals.get(self._stage_diff_approval_key(str(diff_key)))
    if isinstance(record, Mapping):
        lock = " locked" if record.get("locked") else ""
        return f"{record.get('status', '承認済')} {record.get('approver', '')} {record.get('time', '')}{lock}".strip()
    return "未承認"


def _stage_diff_is_locked(owner: Any, qt: Mapping[str, Any], diff_key: str) -> bool:
    self = owner
    approvals = self._mapping(self.cfg.get("stage_diff_approvals", {}))
    record = approvals.get(self._stage_diff_approval_key(str(diff_key)))
    return isinstance(record, Mapping) and bool(record.get("locked"))


def _selected_stage_diff_key(owner: Any, qt: Mapping[str, Any]) -> str | None:
    self = owner
    Qt = qt["Qt"]
    table = getattr(self, "stage_diff_table", None)
    if table is None or table.currentRow() < 0:
        return None
    item = table.item(table.currentRow(), 1)
    if item is None:
        return None
    return str(item.data(Qt.ItemDataRole.UserRole) or item.text())


def _append_stage_approval_history(owner: Any, qt: Mapping[str, Any], diff_key: str, action: str, record: Mapping[str, Any]) -> None:
    self = owner
    history = self.cfg.setdefault("stage_diff_approval_history", [])
    if not isinstance(history, list):
        history = []
        self.cfg["stage_diff_approval_history"] = history
    approval_key = self._stage_diff_approval_key(diff_key)
    history.append(
        {
            "approval_key": approval_key,
            "stage": approval_key.split(":", 2)[:2],
            "diff": diff_key,
            "action": action,
            "status": record.get("status", action),
            "approver": record.get("approver", ""),
            "note": record.get("note", ""),
            "time": record.get("time", datetime.now().isoformat(timespec="seconds")),
            "locked": bool(record.get("locked")),
        }
    )
    self.write_audit_event(f"stage_{action}", approval_key, dict(record))


def approve_selected_stage_difference(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    diff_key = self._selected_stage_diff_key()
    if diff_key is None:
        _notify_information(self, QMessageBox, "承認する差分行を選択してください。")
        return
    self.approve_stage_difference(
        diff_key,
        note=self.stage_approval_note.text().strip(),
        approver=self.stage_approval_user.text().strip() or getpass.getuser(),
        locked=self.stage_approval_lock.isChecked(),
    )


def approve_stage_difference(owner: Any, qt: Mapping[str, Any], diff_key: str, note: str = "", approver: str | None = None, locked: bool = True) -> None:
    self = owner
    approvals = self.cfg.setdefault("stage_diff_approvals", {})
    if not isinstance(approvals, dict):
        approvals = {}
        self.cfg["stage_diff_approvals"] = approvals
    approval_key = self._stage_diff_approval_key(diff_key)
    action = "reapprove" if approval_key in approvals else "approve"
    record = {
        "status": "approved",
        "approver": approver or getpass.getuser(),
        "note": note,
        "time": datetime.now().isoformat(timespec="seconds"),
        "locked": bool(locked),
    }
    approvals[approval_key] = record
    self._append_stage_approval_history(diff_key, action, record)
    self._syncing_yaml = True
    self.yaml_editor.setPlainText(yaml.safe_dump(self.cfg, allow_unicode=True, sort_keys=False))
    self._syncing_yaml = False
    self.refresh_stage_difference_table()
    self.refresh_stage_approval_history_table(activate=False)


def reject_selected_stage_difference(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    diff_key = self._selected_stage_diff_key()
    if diff_key is None:
        _notify_information(self, QMessageBox, "差戻しする差分行を選択してください。")
        return
    self.reject_stage_difference(
        diff_key,
        note=self.stage_approval_note.text().strip(),
        approver=self.stage_approval_user.text().strip() or getpass.getuser(),
    )


def reject_stage_difference(owner: Any, qt: Mapping[str, Any], diff_key: str, note: str = "", approver: str | None = None, locked: bool = False) -> None:
    self = owner
    approvals = self.cfg.setdefault("stage_diff_approvals", {})
    if not isinstance(approvals, dict):
        approvals = {}
        self.cfg["stage_diff_approvals"] = approvals
    record = {
        "status": "rejected",
        "approver": approver or getpass.getuser(),
        "note": note,
        "time": datetime.now().isoformat(timespec="seconds"),
        "locked": bool(locked),
    }
    approvals[self._stage_diff_approval_key(diff_key)] = record
    self._append_stage_approval_history(diff_key, "reject", record)
    self._syncing_yaml = True
    self.yaml_editor.setPlainText(yaml.safe_dump(self.cfg, allow_unicode=True, sort_keys=False))
    self._syncing_yaml = False
    self.refresh_stage_difference_table()
    self.refresh_stage_approval_history_table(activate=False)


def reapprove_selected_stage_difference(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    diff_key = self._selected_stage_diff_key()
    if diff_key is None:
        _notify_information(self, QMessageBox, "再承認する差分行を選択してください。")
        return
    self.reapprove_stage_difference(
        diff_key,
        note=self.stage_approval_note.text().strip(),
        approver=self.stage_approval_user.text().strip() or getpass.getuser(),
        locked=self.stage_approval_lock.isChecked(),
    )


def reapprove_stage_difference(owner: Any, qt: Mapping[str, Any], diff_key: str, note: str = "", approver: str | None = None, locked: bool = True) -> None:
    self = owner
    self.approve_stage_difference(diff_key, note=note, approver=approver, locked=locked)


def stage_approval_history(owner: Any, qt: Mapping[str, Any], diff_key: str | None = None) -> list[dict[str, Any]]:
    self = owner
    rows = [dict(row) for row in self._ensure_list(self.cfg.get("stage_diff_approval_history", [])) if isinstance(row, Mapping)]
    if diff_key is None:
        return rows
    approval_key = self._stage_diff_approval_key(diff_key)
    return [row for row in rows if row.get("approval_key") == approval_key or row.get("diff") == diff_key]


def compare_stage_approval_history(owner: Any, qt: Mapping[str, Any], diff_key: str | None = None) -> dict[str, Any]:
    self = owner
    rows = self.stage_approval_history(diff_key)
    if not rows:
        return {"ok": False, "reason": "no approval history", "diff": diff_key}
    statuses = [str(row.get("status", row.get("action", ""))) for row in rows]
    actors = sorted({str(row.get("approver", "")) for row in rows if row.get("approver")})
    result = {
        "ok": True,
        "diff": diff_key,
        "events": len(rows),
        "first": rows[0],
        "latest": rows[-1],
        "statuses": statuses,
        "actors": actors,
        "changed_status": len(set(statuses)) > 1,
    }
    self.write_audit_event("stage_approval_history_compare", str(diff_key or "all"), result)
    return result


def refresh_stage_approval_history_table(owner: Any, qt: Mapping[str, Any], activate: bool = True) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = getattr(self, "stage_approval_history_table", None)
    if table is None:
        return
    rows = self.stage_approval_history()
    table.setRowCount(len(rows))
    for row, event in enumerate(rows):
        stage = event.get("stage", "")
        if isinstance(stage, list):
            stage = ":".join(map(str, stage))
        values = [
            event.get("time", ""),
            event.get("action", ""),
            stage,
            event.get("diff", ""),
            event.get("approver", ""),
            event.get("locked", ""),
            event.get("note", ""),
        ]
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(str(value)))
    if activate:
        self.tabs.setCurrentWidget(table)


def apply_stage_construction_template(owner: Any, qt: Mapping[str, Any], template: str) -> None:
    self = owner
    stages, row, stage = self._selected_stage_context()
    target = self.stage_detail_target.text().strip() or self._element_spec_text(stage) or "all"
    if template == "excavation":
        stage["type"] = "death"
        stage.update(self._target_element_spec(target))
        stage.setdefault("increments", 4)
        stage.setdefault("solver", {"max_iter": 30, "tolerance": 1.0e-6, "cutback": True})
        stage.setdefault("stress_release", self._float_text(self.stage_detail_stress_release.text().strip() or "1.0", "stress_release"))
        self.add_stage_construction_row(action="excavation", target=target, stress_release=str(stage.get("stress_release", 1.0)))
    elif template == "boundary_change":
        bcs = self._list_value(stage, "boundary_conditions")
        if not bcs:
            bcs.append({"set": "bottom", "uy": 0.0, "template": "boundary_change"})
        stage.setdefault("template", {})["boundary_change"] = True
    elif template == "hydro_change":
        stage["type"] = "consolidation"
        hydro = stage.setdefault("hydro", {})
        if isinstance(hydro, dict) and not hydro.get("pressure_bcs"):
            hydro.setdefault("dt", 1.0)
            hydro.setdefault("steps", 1)
            hydro.setdefault("storage", 1.0e-5)
            hydro.setdefault("permeability", 1.0e-6)
            hydro.setdefault("biot_alpha", 1.0)
            hydro["pressure_bcs"] = [{"set": "top", "pressure": 0.0, "template": "hydro_change"}]
        stage.setdefault("increments", 4)
        stage.setdefault("solver", {"max_iter": 40, "tolerance": 1.0e-6, "cutback": True})
        stage.setdefault("template", {})["hydro_change"] = True
    else:
        return
    stages[row] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"施工テンプレートを適用しました: {template}")
    self.stage_table.selectRow(row)


def repair_selected_stage_difference(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    table = getattr(self, "stage_diff_table", None)
    if table is None or table.currentRow() < 0:
        _notify_information(self, QMessageBox, "修復する差分行を選択してください。")
        return
    self.repair_stage_difference_row(table.currentRow())


def repair_stage_difference_row(owner: Any, qt: Mapping[str, Any], row_index: int) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    Qt = qt["Qt"]
    stages = self._stages()
    row = self._selected_stage_row()
    if row is None or not (0 <= row < len(stages)):
        return
    previous = stages[row - 1] if row > 0 else {}
    current = dict(stages[row])
    key_item = self.stage_diff_table.item(row_index, 1) if getattr(self, "stage_diff_table", None) is not None else None
    key = str(key_item.data(Qt.ItemDataRole.UserRole) if key_item is not None and key_item.data(Qt.ItemDataRole.UserRole) else (key_item.text() if key_item else ""))
    if key and self._stage_diff_is_locked(key):
        QMessageBox.warning(self, "GeoFEM", f"承認ロック済みの差分は修復できません: {key}")
        return
    if key in {"deactivated elements", "reactivated elements"}:
        current["type"] = "static"
        for target_key in ("set", "element_set", "elementSet", "element", "elements", "all", "stress_release"):
            current.pop(target_key, None)
    elif key:
        if key in previous:
            current[key] = previous[key]
        else:
            current.pop(key, None)
    stages[row] = current
    self.cfg["stages"] = stages
    self._after_form_change(f"差分を前ステージに戻しました: {key}")
    self.stage_table.selectRow(row)


def repair_stage_difference_cell(owner: Any, qt: Mapping[str, Any], row_index: int, col: int) -> None:
    self = owner
    if col in {1, 2, 3}:
        self.repair_stage_difference_row(row_index)


def _stage_template_library(owner: Any, qt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    self = owner
    defaults: dict[str, dict[str, Any]] = {
        "掘削:応力解放": {"type": "death", "stress_release": 1.0, "construction_events": [{"action": "excavation", "set": "all", "stress_release": 1.0}]},
        "圧密:水位更新": {"type": "consolidation", "hydro": {"pressure_bcs": [{"set": "top", "pressure": 0.0}], "pore_flux_bcs": []}},
        "境界:支保切替": {"type": "static", "boundary_conditions": [{"set": "bottom", "uy": 0.0}]},
        "SRM:安全率": {"type": "srm", "srm": {"search_mode": "adaptive_bracket", "anchor_factor": 1.0, "bracket_stride": 5, "factor_tol": 0.01, "max_bisection": 6, "factor_start": 1.0, "factor_max": 2.0, "factor_step": 0.05, "failure_plastic_ratio": 0.95}},
        "道路土工:段階掘削": {"type": "death", "set": "excavation_block", "stress_release": 0.5, "loads": [{"type": "gravity", "scale": 1.0}]},
        "河川耐震:水圧更新": {"type": "consolidation", "hydro": {"pressure_bcs": [{"set": "water_side", "pressure": 0.0}], "pore_robin_bcs": [{"set": "drain", "beta": 1.0, "pressure": 0.0}]}},
        "斜面安定:SRM照査": {"type": "srm", "srm": {"search_mode": "adaptive_bracket", "anchor_factor": 1.0, "bracket_stride": 5, "factor_tol": 0.01, "max_bisection": 6, "factor_start": 1.0, "factor_max": 3.0, "factor_step": 0.02, "failure_plastic_ratio": 0.95}, "solver": {"max_iter": 60, "cutback": True}},
        "土留め:支保工発生": {"type": "static", "boundary_conditions": [{"set": "wall_support", "ux": 0.0}], "loads": [{"type": "earth_pressure", "set": "wall"}]},
        "NATM:上半掘削": {"type": "death", "set": "top_heading", "stress_release": 0.4, "construction_events": [{"action": "excavation", "set": "top_heading", "stress_release": 0.4}]},
        "NATM:下半掘削": {"type": "death", "set": "bench", "stress_release": 0.6, "construction_events": [{"action": "excavation", "set": "bench", "stress_release": 0.6}]},
        "盛土:層別載荷": {"type": "static", "loads": [{"type": "gravity", "scale": 1.0}], "element_properties": [{"set": "embankment_lift", "material": "fill"}]},
        "軟弱地盤:排水境界": {"type": "consolidation", "hydro": {"pore_flux_bcs": [{"set": "drain", "flux": 0.0}], "pore_robin_bcs": [{"set": "vertical_drain", "beta": 1.0, "pressure": 0.0}]}},
        "港湾:水位低下": {"type": "consolidation", "hydro": {"pressure_bcs": [{"set": "sea_side", "pressure": -10.0}]}},
        "鉄道:列車荷重": {"type": "static", "loads": [{"set": "rail_load", "fy": -80.0, "load_case": "train"}]},
    }
    defaults.update(
        {
            "Road earthwork: staged embankment": {
                "type": "static",
                "standard": "road_earthwork",
                "business": "embankment",
                "loads": [{"type": "gravity", "scale": 1.0}],
                "element_properties": [{"set": "lift", "material": "fill"}],
                "construction_events": [{"action": "activate", "set": "lift"}],
            },
            "Road earthwork: excavation with release": {
                "type": "death",
                "standard": "road_earthwork",
                "business": "excavation",
                "set": "excavation_block",
                "stress_release": 0.5,
                "construction_events": [{"action": "excavation", "set": "excavation_block", "stress_release": 0.5}],
            },
            "Railway: train surcharge": {
                "type": "static",
                "standard": "railway",
                "business": "track_support",
                "loads": [{"set": "rail_load", "fy": -80.0, "load_case": "train"}],
            },
            "Port: tidal water pressure": {
                "type": "consolidation",
                "standard": "port",
                "business": "waterfront",
                "hydro": {"pressure_bcs": [{"set": "sea_side", "pressure": -10.0}], "pore_flux_bcs": []},
            },
            "Tunnel NATM: top heading": {
                "type": "death",
                "standard": "tunnel",
                "business": "natm",
                "set": "top_heading",
                "stress_release": 0.4,
                "construction_events": [{"action": "excavation", "set": "top_heading", "stress_release": 0.4}],
            },
            "Tunnel NATM: bench": {
                "type": "death",
                "standard": "tunnel",
                "business": "natm",
                "set": "bench",
                "stress_release": 0.6,
                "construction_events": [{"action": "excavation", "set": "bench", "stress_release": 0.6}],
            },
            "Building excavation: retaining support": {
                "type": "static",
                "standard": "building",
                "business": "retaining_excavation",
                "boundary_conditions": [{"set": "wall_support", "ux": 0.0}],
                "loads": [{"type": "earth_pressure", "set": "wall"}],
            },
            "Dam: rapid drawdown": {
                "type": "consolidation",
                "standard": "dam",
                "business": "reservoir",
                "hydro": {"pressure_bcs": [{"set": "reservoir_side", "pressure": 0.0}], "pore_robin_bcs": [{"set": "drain", "beta": 1.0, "pressure": 0.0}]},
            },
            "Soft ground: vertical drain consolidation": {
                "type": "consolidation",
                "standard": "soft_ground",
                "business": "improvement",
                "hydro": {"pore_flux_bcs": [{"set": "drain", "flux": 0.0}], "pore_robin_bcs": [{"set": "vertical_drain", "beta": 1.0, "pressure": 0.0}]},
            },
            "Landslide: SRM verification": {
                "type": "srm",
                "standard": "slope",
                "business": "stability",
                "srm": {"search_mode": "adaptive_bracket", "anchor_factor": 1.0, "bracket_stride": 5, "factor_tol": 0.01, "max_bisection": 6, "factor_start": 1.0, "factor_max": 3.0, "factor_step": 0.02, "failure_plastic_ratio": 0.95},
                "solver": {"max_iter": 60, "cutback": True},
            },
        }
    )
    try:
        from geofem_app.geofeas_public import workflow_catalog

        workflow_defaults = {
            "tunnel_excavation": {"type": "death", "set": "excavation_block", "stress_release": 0.5},
            "retaining_excavation": {"type": "death", "set": "excavation_block", "stress_release": 0.5},
            "river_liquefaction_h19": {"type": "consolidation", "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1}},
            "river_liquefaction_h28": {"type": "consolidation", "hydro": {"initial_pressure": 0.0, "dt": 1.0, "steps": 1}},
            "seepage_pressure": {"type": "consolidation", "hydro": {"seepage_csv": "", "water_levels": []}},
            "axisymmetric": {"type": "static"},
            "srm_slope": {"type": "srm", "srm": {"search_mode": "adaptive_bracket", "anchor_factor": 1.0, "bracket_stride": 5, "factor_tol": 0.01, "max_bisection": 6, "factor_start": 1.0, "factor_max": 2.0, "factor_step": 0.05, "failure_plastic_ratio": 0.95}},
        }
        for item in workflow_catalog():
            workflow = str(item.get("id", ""))
            if not workflow:
                continue
            spec = dict(workflow_defaults.get(workflow, {"type": "static"}))
            spec["geofeas_workflow"] = workflow
            spec.setdefault("post", {"geofeas_style": True})
            spec["operation_log"] = item.get("stage_pattern", [])
            defaults[f"GeoFEAS public: {item.get('title', workflow)}"] = spec
    except Exception:
        pass
    bundled = Path(__file__).resolve().parent.parent / "templates" / "stage_templates.yaml"
    if bundled.exists():
        try:
            data = yaml.safe_load(bundled.read_text(encoding="utf-8")) or {}
            bundled_library = data.get("stage_template_library", data)
            if isinstance(bundled_library, Mapping):
                for name, value in bundled_library.items():
                    if isinstance(value, Mapping):
                        defaults[str(name)] = dict(value)
        except Exception:
            pass
    custom = self._mapping(self.cfg.get("stage_template_library", {}))
    for name, value in custom.items():
        if isinstance(value, Mapping):
            defaults[str(name)] = dict(value)
    return defaults


def _stage_template_display_name(name: str, value: Mapping[str, Any], index: int, locale: str) -> str:
    text = str(name)
    if locale != "en":
        return text
    translations = {
        "掘削:応力解放": "Excavation: stress release",
        "圧密:水位更新": "Consolidation: water level update",
        "境界:支保切替": "Boundary: support switch",
        "SRM:安全率": "SRM: safety factor",
        "道路土工:段階掘削": "Road earthwork: staged excavation",
        "河川耐震:水圧更新": "River seismic: water pressure update",
        "斜面安定:SRM照査": "Slope stability: SRM verification",
        "土留め:支保工発生": "Retaining wall: support activation",
        "NATM:上半掘削": "NATM: top heading excavation",
        "NATM:下半掘削": "NATM: bench excavation",
        "盛土:層別載荷": "Embankment: staged loading",
        "軟弱地盤:排水境界": "Soft ground: drainage boundary",
        "港湾:水位低下": "Port: water drawdown",
        "鉄道:列車荷重": "Railway: train load",
    }
    if text in translations:
        return translations[text]
    if not any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in text):
        return text
    stage_type = str(value.get("type", "static") if isinstance(value, Mapping) else "static")
    standard = str(value.get("standard", "") if isinstance(value, Mapping) else "").replace("_", " ")
    business = str(value.get("business", "") if isinstance(value, Mapping) else "").replace("_", " ")
    suffix = " / ".join(part for part in (standard, business, stage_type) if part)
    return f"Template {index + 1}" + (f": {suffix}" if suffix else "")


def _refresh_stage_template_combo(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    combo = getattr(self, "stage_template_combo", None)
    if combo is None:
        return
    current = combo.currentData()
    if current is None:
        current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    library = self._stage_template_library()
    for index, name in enumerate(sorted(library)):
        combo.addItem(_stage_template_display_name(name, library.get(name, {}), index, getattr(self, "gui_locale", "ja")), name)
    if current:
        selected = combo.findData(current)
        if selected < 0:
            selected = combo.findText(str(current))
        if selected >= 0:
            combo.setCurrentIndex(selected)
    combo.blockSignals(False)


def apply_stage_template_from_library(owner: Any, qt: Mapping[str, Any], name: str) -> None:
    self = owner
    library = self._stage_template_library()
    template = library.get(str(name))
    if not template:
        return
    stages, row, stage = self._selected_stage_context()
    merged = dict(stage)
    for key, value in template.items():
        if key == "name":
            continue
        merged[key] = value
    stages[row] = merged
    self.cfg["stages"] = stages
    self._after_form_change(f"業務別施工テンプレートを適用しました: {name}")
    self.stage_table.selectRow(row)


def save_stage_template_library(owner: Any, qt: Mapping[str, Any], path: str | Path | None = None) -> Path | None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    if path is None:
        default = self.project_root / "stage_template_library.yaml"
        selected, _ = QFileDialog.getSaveFileName(self, "施工テンプレート保存", str(default), "YAML (*.yaml *.yml)")
        if not selected:
            return None
        path = selected
    out = Path(path)
    data = {"stage_template_library": self._stage_template_library()}
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    self.statusBar().showMessage(f"施工テンプレートを保存しました: {out}")
    return out


def load_stage_template_library(owner: Any, qt: Mapping[str, Any], path: str | Path | None = None) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    if path is None:
        selected, _ = QFileDialog.getOpenFileName(self, "施工テンプレート読込", str(self.project_root), "YAML (*.yaml *.yml)")
        if not selected:
            return
        path = selected
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    library = data.get("stage_template_library", data)
    if not isinstance(library, Mapping):
        QMessageBox.warning(self, "GeoFEM", "施工テンプレートYAMLの形式が不正です。")
        return
    self.cfg["stage_template_library"] = {str(key): dict(value) for key, value in library.items() if isinstance(value, Mapping)}
    self._refresh_stage_template_combo()
    self._after_form_change("施工テンプレートを読込みました")


def collect_stage_cumulative_conflicts(owner: Any, qt: Mapping[str, Any]) -> list[dict[str, Any]]:
    self = owner
    conflicts: list[dict[str, Any]] = []
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        mesh = None
    active = {element.id for element in mesh.elements if element.active} if mesh is not None else set()
    bc_values: dict[tuple[str, str], tuple[float, int]] = {}
    hydro_values: dict[str, tuple[float, int]] = {}
    for index, stage in enumerate(self._stages()):
        stype = str(stage.get("type", "")).lower().strip()
        targets = self._stage_element_targets(mesh, stage) if mesh is not None else set()
        if stype in {"death", "excavation", "deactivate"}:
            repeated = targets - active
            for eid in sorted(repeated, key=self._natural_sort_key):
                conflicts.append({"stage": index, "category": "death", "target": eid, "detail": "既に非アクティブな要素を再度deathしています。", "repair": "remove_target", "suggestion": "death対象から除外、またはreactivateイベントへ変更"})
            active.difference_update(targets)
        for prop in self._ensure_list(stage.get("element_properties", [])):
            if not isinstance(prop, Mapping) or mesh is None:
                continue
            inactive = self._element_targets_for_spec(mesh, prop) - active
            for eid in sorted(inactive, key=self._natural_sort_key):
                conflicts.append({"stage": index, "category": "material", "target": eid, "detail": "非アクティブ要素への材料変更です。", "repair": "remove_material", "suggestion": "材料変更行を削除、または材料変更ステージをdeath前へ移動"})
        if mesh is not None:
            for bc in self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))):
                if not isinstance(bc, Mapping):
                    continue
                for nid in self._node_targets_for_spec(mesh, bc):
                    for dof in ("ux", "uy"):
                        if dof not in bc:
                            continue
                        value = float(bc[dof])
                        key = (nid, dof)
                        if key in bc_values and abs(bc_values[key][0] - value) > 1.0e-12:
                            conflicts.append({"stage": index, "category": "bc", "target": f"{nid}.{dof}", "detail": "累積拘束値が前ステージと矛盾します。", "repair": "remove_bc", "suggestion": "同一DOFの後続拘束を削除、または前ステージ拘束を解除"})
                        bc_values[key] = (value, index)
        hydro = stage.get("hydro", stage.get("consolidation", {}))
        if mesh is not None and isinstance(hydro, Mapping):
            for spec in self._ensure_list(hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", []))):
                if not isinstance(spec, Mapping):
                    continue
                value = float(spec.get("pressure", spec.get("value", 0.0)) or 0.0)
                targets_text = [f"node:{nid}" for nid in self._node_targets_for_spec(mesh, spec)] + [f"edge:{a}-{b}" for a, b in self._edge_targets_for_spec(mesh, spec)]
                for target in targets_text:
                    if target in hydro_values and abs(hydro_values[target][0] - value) > 1.0e-12:
                        conflicts.append({"stage": index, "category": "hydro", "target": target, "detail": "累積水圧値が前ステージと矛盾します。", "repair": "remove_hydro", "suggestion": "後続水圧を水位更新テンプレートとして明示、または重複水圧を削除"})
                    hydro_values[target] = (value, index)
    return conflicts


def refresh_stage_conflict_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    Qt = qt["Qt"]
    table = getattr(self, "stage_conflict_table", None)
    if table is None:
        return
    conflicts = self.collect_stage_cumulative_conflicts()
    table.setRowCount(len(conflicts))
    for row, conflict in enumerate(conflicts):
        values = [conflict.get("stage", ""), conflict.get("category", ""), conflict.get("target", ""), conflict.get("detail", ""), conflict.get("repair", ""), conflict.get("suggestion", "")]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, conflict)
            table.setItem(row, col, item)


def repair_selected_stage_conflict(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    Qt = qt["Qt"]
    table = getattr(self, "stage_conflict_table", None)
    if table is None or table.currentRow() < 0:
        _notify_information(self, QMessageBox, "修復する矛盾行を選択してください。")
        return
    item = table.item(table.currentRow(), 0)
    conflict = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
    if isinstance(conflict, Mapping):
        self.repair_stage_conflict(conflict)


def apply_stage_conflict_repair_suggestions(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    conflicts = self.collect_stage_cumulative_conflicts()
    for conflict in conflicts:
        self.repair_stage_conflict(conflict)
    self.refresh_stage_conflict_table()


def repair_stage_conflict(owner: Any, qt: Mapping[str, Any], conflict: Mapping[str, Any]) -> None:
    self = owner
    stages = self._stages()
    index = int(conflict.get("stage", -1))
    if not (0 <= index < len(stages)):
        return
    stage = dict(stages[index])
    repair = str(conflict.get("repair", ""))
    target = str(conflict.get("target", ""))
    if repair == "remove_target":
        ids = {value for value in self._ensure_list(stage.get("elements", [])) if str(value) != target}
        if ids:
            stage["elements"] = sorted(str(value) for value in ids)
        else:
            for key in ("elements", "element", "set", "element_set", "all"):
                stage.pop(key, None)
            stage["type"] = "static"
    elif repair == "remove_material":
        stage["element_properties"] = [prop for prop in self._ensure_list(stage.get("element_properties", [])) if not (isinstance(prop, Mapping) and target in {str(v) for v in self._ensure_list(prop.get("elements", prop.get("element", [])))})]
    elif repair == "remove_bc":
        node = target.split(".")[0]
        stage["boundary_conditions"] = [bc for bc in self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))) if not (isinstance(bc, Mapping) and node in {str(nid) for nid in self._ensure_list(bc.get("nodes", bc.get("node", [])))})]
    elif repair == "remove_hydro":
        hydro = dict(self._mapping(stage.get("hydro", stage.get("consolidation", {}))))
        for key in ("pressure_bcs", "pore_pressure_bcs"):
            hydro[key] = []
        stage["hydro"] = hydro
    stages[index] = stage
    self.cfg["stages"] = stages
    self._after_form_change(f"累積矛盾を修復しました: {target}")
    self.refresh_stage_conflict_table()


_STAGE_COMPARE_HEADERS = ["ステージ", "種別", "要素", "材料", "境界", "荷重", "水理/MPC"]
_STAGE_COMPARE_CHANGE_COLOR = "#fff3cd"


def _stage_compare_stages_until(stages: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    if index < 0:
        return []
    return stages[: index + 1]


def _stage_compare_list(owner: Any, stage: Mapping[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        if key in stage:
            return owner._ensure_list(stage.get(key))
    return []


def _stage_compare_hydro_entries(owner: Any, stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    hydro = owner._mapping(stage.get("hydro", stage.get("consolidation", {})))
    if not hydro:
        hydro = owner._mapping(stage.get("consolidation", {}))
    entries: list[dict[str, Any]] = []
    for key in ("pressure_bcs", "pore_pressure_bcs", "pore_flux_bcs", "pore_robin_bcs", "drainage", "flux_bcs", "robin_bcs"):
        value = hydro.get(key, [])
        if isinstance(value, list):
            for item in value:
                entries.append({"kind": key, "value": item})
        elif value:
            entries.append({"kind": key, "value": value})
    return entries


def _stage_compare_active_elements(owner: Any, mesh: Any, stages: list[dict[str, Any]], index: int) -> list[str] | list[dict[str, Any]]:
    active = {element.id for element in mesh.elements if element.active} if mesh is not None else set()
    if mesh is None:
        events: list[dict[str, Any]] = []
        for stage_index, stage in enumerate(_stage_compare_stages_until(stages, index)):
            stype = str(stage.get("type", "")).lower().strip()
            target = owner._element_spec_text(stage)
            if stype in {"excavation", "death", "deactivate", "remove", "activate", "birth", "reactivate"} or target:
                events.append({"stage": stage_index, "type": stype, "target": target})
            for raw in _stage_compare_list(owner, stage, "construction_events", "construction"):
                if isinstance(raw, Mapping):
                    events.append({"stage": stage_index, **dict(raw)})
                else:
                    events.append({"stage": stage_index, "value": raw})
        return events

    for stage in _stage_compare_stages_until(stages, index):
        stype = str(stage.get("type", "")).lower().strip()
        if stype in {"excavation", "death", "deactivate", "remove"}:
            active.difference_update(owner._stage_element_targets(mesh, stage))
        elif stype in {"activate", "birth", "reactivate"}:
            active.update(owner._stage_element_targets(mesh, stage))
        for raw in _stage_compare_list(owner, stage, "construction_events", "construction"):
            if not isinstance(raw, Mapping):
                continue
            action = str(raw.get("action", raw.get("type", ""))).lower().strip()
            targets = owner._stage_element_targets(mesh, raw)
            if action in {"excavation", "death", "deactivate", "remove"}:
                active.difference_update(targets)
            elif action in {"activate", "birth", "reactivate"}:
                active.update(targets)
    return sorted(active, key=owner._natural_sort_key)


def _stage_compare_material_state(owner: Any, mesh: Any, stages: list[dict[str, Any]], index: int) -> list[Any]:
    if mesh is None:
        entries: list[Any] = []
        for stage in _stage_compare_stages_until(stages, index):
            entries.extend(_stage_compare_list(owner, stage, "element_properties"))
        return entries
    state = {element.id: element.material for element in mesh.elements}
    for stage in _stage_compare_stages_until(stages, index):
        for prop in _stage_compare_list(owner, stage, "element_properties"):
            if not isinstance(prop, Mapping):
                continue
            material = prop.get("material")
            if material in (None, ""):
                continue
            for eid in owner._element_targets_for_spec(mesh, prop):
                state[str(eid)] = str(material)
    return [{"element": eid, "material": state[eid]} for eid in sorted(state, key=owner._natural_sort_key)]


def _stage_compare_boundary_state(owner: Any, mesh: Any, stages: list[dict[str, Any]], index: int) -> list[Any]:
    if mesh is None:
        entries: list[Any] = []
        for stage in _stage_compare_stages_until(stages, index):
            entries.extend(_stage_compare_list(owner, stage, "boundary_conditions", "bc"))
        return entries
    state: dict[tuple[str, str], Any] = {}
    fallback: list[Any] = []
    selector_keys = {"node", "nodes", "set", "edge", "edges", "element", "elements", "all", "source"}
    for stage in _stage_compare_stages_until(stages, index):
        for bc in _stage_compare_list(owner, stage, "boundary_conditions", "bc"):
            if not isinstance(bc, Mapping):
                fallback.append(bc)
                continue
            nodes = owner._node_targets_for_spec(mesh, bc)
            value_keys = [key for key in bc if key not in selector_keys]
            if not nodes or not value_keys:
                fallback.append(dict(bc))
                continue
            for node in nodes:
                for key in value_keys:
                    state[(str(node), str(key))] = bc.get(key)
    rows = [{"node": node, "dof": dof, "value": value} for (node, dof), value in sorted(state.items(), key=lambda item: (owner._natural_sort_key(item[0][0]), item[0][1]))]
    if fallback:
        rows.append({"unresolved": fallback})
    return rows


def _stage_compare_load_state(owner: Any, stages: list[dict[str, Any]], index: int) -> list[Any]:
    entries: list[Any] = []
    for stage in _stage_compare_stages_until(stages, index):
        entries.extend(_stage_compare_list(owner, stage, "loads"))
    return entries


def _stage_compare_hydro_mpc_state(owner: Any, stages: list[dict[str, Any]], index: int) -> list[Any]:
    entries: list[Any] = []
    for stage in _stage_compare_stages_until(stages, index):
        entries.extend(_stage_compare_hydro_entries(owner, stage))
        for mpc in _stage_compare_list(owner, stage, "mpc_constraints", "mpc"):
            entries.append({"kind": "mpc", "value": mpc})
    return entries


def _stage_compare_states(owner: Any, mesh: Any, stages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    return {
        "type": str(stages[index].get("type", "static")) if 0 <= index < len(stages) else "static",
        "elements": _stage_compare_active_elements(owner, mesh, stages, index),
        "materials": _stage_compare_material_state(owner, mesh, stages, index),
        "boundary": _stage_compare_boundary_state(owner, mesh, stages, index),
        "loads": _stage_compare_load_state(owner, stages, index),
        "hydro_mpc": _stage_compare_hydro_mpc_state(owner, stages, index),
    }


def _stage_compare_cell_text(owner: Any, mesh: Any, stage: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> str:
    if key == "type":
        return str(after.get("type", "static") or "static")
    if key == "elements":
        if mesh is not None:
            before_set = {str(value) for value in owner._ensure_list(before.get("elements", []))}
            after_set = {str(value) for value in owner._ensure_list(after.get("elements", []))}
            removed = before_set - after_set
            added = after_set - before_set
            parts = [f"有効 {len(after_set)}"]
            if removed:
                parts.append(f"Death {len(removed)}")
            if added:
                parts.append(f"Birth {len(added)}")
            event_count = len(_stage_compare_list(owner, stage, "construction_events", "construction"))
            if event_count and not (removed or added):
                parts.append(f"施工 {event_count}")
            return " / ".join(parts)
        event_count = len(_stage_compare_list(owner, stage, "construction_events", "construction"))
        target = owner._element_spec_text(stage)
        if event_count or target:
            return f"施工 {event_count}" + (f" / 対象 {target}" if target else "")
        return "-"
    if key == "materials":
        current = len(_stage_compare_list(owner, stage, "element_properties"))
        total = len(after.get("materials", [])) if mesh is None else len({item.get("material") for item in after.get("materials", []) if isinstance(item, Mapping)})
        if current:
            return f"変更 {current} / 材料種 {total}"
        return f"材料種 {total}" if total else "-"
    if key == "boundary":
        current = len(_stage_compare_list(owner, stage, "boundary_conditions", "bc"))
        total = len(after.get("boundary", []))
        if current:
            return f"変更 {current} / DOF {total}"
        return f"DOF {total}" if total else "-"
    if key == "loads":
        current = len(_stage_compare_list(owner, stage, "loads"))
        total = len(after.get("loads", []))
        if current:
            return f"追加 {current} / 累積 {total}"
        return f"累積 {total}" if total else "-"
    if key == "hydro_mpc":
        current = len(_stage_compare_hydro_entries(owner, stage)) + len(_stage_compare_list(owner, stage, "mpc_constraints", "mpc"))
        total = len(after.get("hydro_mpc", []))
        if current:
            return f"変更 {current} / 累積 {total}"
        return f"累積 {total}" if total else "-"
    return "-"


def refresh_stage_cross_compare_table(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    QColor = qt.get("QColor")
    Qt = qt["Qt"]
    table = getattr(self, "stage_compare_table", None)
    if table is None:
        return
    stages = self._stages()
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
    except Exception:
        mesh = None
    table.setColumnCount(len(_STAGE_COMPARE_HEADERS))
    table.setHorizontalHeaderLabels(_STAGE_COMPARE_HEADERS)
    table.setRowCount(len(stages))
    for index, stage in enumerate(stages):
        before = _stage_compare_states(self, mesh, stages, index - 1)
        after = _stage_compare_states(self, mesh, stages, index)
        row_values = [
            str(stage.get("name", f"Stage-{index + 1}")),
            _stage_compare_cell_text(self, mesh, stage, before, after, "type"),
            _stage_compare_cell_text(self, mesh, stage, before, after, "elements"),
            _stage_compare_cell_text(self, mesh, stage, before, after, "materials"),
            _stage_compare_cell_text(self, mesh, stage, before, after, "boundary"),
            _stage_compare_cell_text(self, mesh, stage, before, after, "loads"),
            _stage_compare_cell_text(self, mesh, stage, before, after, "hydro_mpc"),
        ]
        keys = ["stage", "type", "elements", "materials", "boundary", "loads", "hydro_mpc"]
        for col, value in enumerate(row_values):
            key = keys[col]
            item = QTableWidgetItem(str(value))
            if key == "stage":
                changed = False
                before_value = ""
                after_value = str(stage.get("name", f"Stage-{index + 1}"))
            else:
                before_value = before.get(key)
                after_value = after.get(key)
                changed = self._stable_json(before_value) != self._stable_json(after_value)
            item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "stage": index,
                    "category": key,
                    "changed": changed,
                    "before": before_value,
                    "after": after_value,
                },
            )
            if QColor is not None and changed:
                item.setBackground(QColor(_STAGE_COMPARE_CHANGE_COLOR))
            if hasattr(Qt, "AlignmentFlag"):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if hasattr(Qt, "ItemFlag"):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if key != "stage":
                item.setToolTip(
                    (
                        "前ステージから変更あり\n"
                        f"前: {self._short_json(before_value, 240)}\n"
                        f"後: {self._short_json(after_value, 240)}"
                    )
                    if changed
                    else "前ステージから変更なし"
                )
            table.setItem(index, col, item)


def refresh_stage_guidance(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    label = getattr(self, "stage_guidance_label", None)
    if label is None:
        return
    row = self._selected_stage_row()
    stages = self._stages()
    if row is None or not (0 <= row < len(stages)):
        label.setText("ステージを選択するとGeoFEAS風の入力ガイダンスを表示します。")
        return
    stage = stages[row]
    stype = str(stage.get("type", "static")).lower()
    hints = [f"{stage.get('name', f'Stage-{row + 1}')}: {stype}"]
    workflow = str(stage.get("geofeas_workflow", "") or "").strip()
    if workflow:
        hints.append(f"GeoFEAS public workflow: {workflow}")
    if stype in {"death", "excavation", "deactivate"}:
        hints.append("掘削/death: 対象要素set、応力解放率、必要なら材料変更と水圧条件を確認してください。")
    elif stype in {"consolidation", "u-p", "up"}:
        hints.append("u-p/圧密: 水理境界、dt/steps、storage/permeability、排水/非排水境界を確認してください。")
    elif stype in {"geostatic", "k0"}:
        hints.append("Geostatic/K0: K0、地表面Y、自重方向、初期拘束を確認してください。")
    elif stype in {"srm", "safety_factor"}:
        hints.append("SRM: 強度低減範囲、収束設定、PostのFL/すべり面候補を確認してください。")
    else:
        hints.append("静的ステージ: 境界、荷重、材料変更、MPCの差分を確認してください。")
    if not self._ensure_list(stage.get("boundary_conditions", stage.get("bc", []))) and row == 0:
        hints.append("初期ステージに拘束がありません。剛体モードチェックを推奨します。")
    label.setText(" / ".join(hints))
    self.refresh_stage_wizard_table(row)


def stage_guidance_steps(owner: Any, qt: Mapping[str, Any], row: int | None = None) -> list[tuple[str, str]]:
    self = owner
    stages = self._stages()
    if row is None:
        row = self._selected_stage_row()
    if row is None or not (0 <= row < len(stages)):
        return []
    stage = stages[row]
    stype = str(stage.get("type", "static")).lower()
    workflow = str(stage.get("geofeas_workflow", "") or "").strip()
    if workflow:
        try:
            from geofem_app.geofeas_public import public_workflow_operation_log

            return [
                (
                    str(item.get("step", index + 1)),
                    f"{item.get('tab', '')}: {item.get('action', '')} -> {item.get('expected', '')}",
                )
                for index, item in enumerate(public_workflow_operation_log(workflow))
            ]
        except Exception:
            pass
    steps = [("1", "対象要素/節点/辺を画面選択またはsetで確認する"), ("2", "前ステージとの差分表を確認する")]
    if stype in {"death", "excavation", "deactivate"}:
        steps.extend([("3", "掘削対象と応力解放率を設定する"), ("4", "材料変更/境界/荷重/水理の差分を確認する")])
    elif stype in {"consolidation", "u-p", "up"}:
        steps.extend([("3", "水理境界と圧密パラメータを設定する"), ("4", "水圧/流量/Robin条件の累積矛盾を診断する")])
    elif stype in {"srm", "safety_factor"}:
        steps.extend([("3", "強度低減範囲と収束条件を設定する"), ("4", "PostでFL図とすべり面候補を確認する")])
    else:
        steps.extend([("3", "境界/荷重/MPCを設定する"), ("4", "モデルチェックで拘束と荷重の整合を確認する")])
    steps.append(("5", "全ステージ横断比較と累積矛盾診断を実行する"))
    return steps


def refresh_stage_wizard_table(owner: Any, qt: Mapping[str, Any], row: int | None = None) -> None:
    self = owner
    QTableWidgetItem = qt["QTableWidgetItem"]
    table = getattr(self, "stage_wizard_table", None)
    if table is None:
        return
    steps = self.stage_guidance_steps(row)
    table.setRowCount(len(steps))
    for index, (step, text) in enumerate(steps):
        for col, value in enumerate([step, text, "未確認"]):
            table.setItem(index, col, QTableWidgetItem(value))


def show_stage_difference(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    stages = self._stages()
    if not stages:
        _notify_information(self, QMessageBox, "ステージがありません。")
        return
    row = self._selected_stage_row()
    if row is None:
        row = len(stages) - 1
    row = max(0, min(row, len(stages) - 1))
    previous = stages[row - 1] if row > 0 else {}
    current = stages[row]
    lines = [
        f"施工ステップ差分: {previous.get('name', '<start>')} -> {current.get('name', f'Stage-{row + 1}')}",
        "",
    ]
    keys = sorted(set(previous) | set(current))
    for key in keys:
        if self._stable_json(previous.get(key)) != self._stable_json(current.get(key)):
            lines.append(f"- {key}: {self._short_json(previous.get(key))} -> {self._short_json(current.get(key))}")
    try:
        from geofem_app.fem2d import mesh_from_config

        mesh = mesh_from_config(self.cfg)
        before_active = self._active_elements_after_stage_index(mesh, row - 1)
        after_active = self._active_elements_after_stage_index(mesh, row)
        removed = sorted(before_active - after_active, key=self._natural_sort_key)
        added = sorted(after_active - before_active, key=self._natural_sort_key)
        lines.extend(
            [
                "",
                f"active elements: {len(before_active)} -> {len(after_active)}",
                f"deactivated: {len(removed)} {', '.join(removed[:20])}",
                f"reactivated: {len(added)} {', '.join(added[:20])}",
            ]
        )
    except Exception as exc:
        lines.append("")
        lines.append(f"active element diff unavailable: {exc}")
    text = "\n".join(lines)
    self.result_view.setPlainText(text)
    self.results_summary.setText(text.replace("\n", " / ")[:800])
    self.refresh_stage_difference_table()
    self.show_stage_workspace_tab(1)


__all__ = [
    "STAGE_CONTROLLER_METHODS",
    "stage_controller_contract",
    "add_stage",
    "copy_selected_stage",
    "move_selected_stage",
    "delete_selected_stage",
    "apply_stage_table",
    "populate_stage_detail_tables",
    "populate_stage_change_table",
    "_clear_stage_detail_form",
    "_populate_stage_detail_form",
    "_populate_stage_construction_table",
    "add_stage_construction_row",
    "add_stage_change_row",
    "stage_recommended_defaults",
    "refresh_stage_recommendation_label",
    "apply_stage_recommended_defaults",
    "_apply_stage_detail_form_values",
    "_construction_events_from_table",
    "apply_stage_change_table",
    "add_stage_material_row",
    "add_stage_boundary_row",
    "add_stage_load_row",
    "add_stage_hydro_row",
    "add_stage_mpc_row",
    "apply_stage_detail_tables",
    "apply_selected_elements_to_stage_set",
    "add_selected_elements_to_stage_death",
    "add_selected_elements_to_stage_birth",
    "add_selected_elements_to_stage_material_change",
    "add_selected_prescribed_displacement",
    "add_selected_edge_load",
    "add_pseudo_static_earthquake_load",
    "add_selected_hydro_coupling",
    "_stage_difference_rows",
    "refresh_stage_difference_table",
    "_stage_diff_approval_key",
    "_stage_diff_approval_status",
    "_stage_diff_is_locked",
    "_selected_stage_diff_key",
    "_append_stage_approval_history",
    "approve_selected_stage_difference",
    "approve_stage_difference",
    "reject_selected_stage_difference",
    "reject_stage_difference",
    "reapprove_selected_stage_difference",
    "reapprove_stage_difference",
    "stage_approval_history",
    "compare_stage_approval_history",
    "refresh_stage_approval_history_table",
    "apply_stage_construction_template",
    "repair_selected_stage_difference",
    "repair_stage_difference_row",
    "repair_stage_difference_cell",
    "_stage_template_library",
    "_refresh_stage_template_combo",
    "apply_stage_template_from_library",
    "save_stage_template_library",
    "load_stage_template_library",
    "collect_stage_cumulative_conflicts",
    "refresh_stage_conflict_table",
    "repair_selected_stage_conflict",
    "apply_stage_conflict_repair_suggestions",
    "repair_stage_conflict",
    "refresh_stage_cross_compare_table",
    "refresh_stage_guidance",
    "stage_guidance_steps",
    "refresh_stage_wizard_table",
    "show_stage_difference",
]
