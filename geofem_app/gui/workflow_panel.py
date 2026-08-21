"""Workflow panel construction and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from geofem_app.gui.i18n import DEFAULT_GUI_LOCALE, gui_message
from geofem_app.gui.presentation_labels import friendly_input_reference
from geofem_app.gui.workflow_guidance import build_workflow_guidance


WORKFLOW_BUTTON_MIN_HEIGHT = 36
WORKFLOW_RIBBON_MIN_HEIGHT = 44
WORKFLOW_RIBBON_MAX_HEIGHT = 52
WORKFLOW_RIBBON_BUTTON_MIN_HEIGHT = 30
WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT = 32
WORKFLOW_RIBBON_BACK_BUTTON_WIDTH = 72
WORKFLOW_RIBBON_JUMP_BUTTON_WIDTH = 104
WORKFLOW_RIBBON_FOCUS_BUTTON_WIDTH = 116
WORKFLOW_RIBBON_NAV_SPACING = 6
WORKFLOW_RIBBON_NAV_WIDTH = WORKFLOW_RIBBON_BACK_BUTTON_WIDTH + WORKFLOW_RIBBON_JUMP_BUTTON_WIDTH + WORKFLOW_RIBBON_NAV_SPACING
WORKFLOW_GUIDANCE_TABLE_MAX_HEIGHT = 220
WORKFLOW_COMPACT_COLUMNS = ("#", "step", "status", "next_action")
EXECUTION_AND_LATER_STEP_IDS = {"solver", "results", "report"}
RESULT_LOCK_ALLOWED_STEP_IDS = {"solver", "results", "report"}


def _display_input_reference(owner: Any, value: Any) -> str:
    if hasattr(owner, "_present_input_reference"):
        return str(owner._present_input_reference(value))
    return friendly_input_reference(value, locale=_owner_locale(owner))


@dataclass(frozen=True)
class WorkflowButtonSpec:
    label: str
    callback: str
    args: tuple[Any, ...] = ()


@dataclass(frozen=True)
class WorkflowGroupSpec:
    title: str
    buttons: tuple[WorkflowButtonSpec, ...]


WORKFLOW_PANEL_GROUPS: tuple[WorkflowGroupSpec, ...] = (
    WorkflowGroupSpec(
        "1. モデル作成",
        (
            WorkflowButtonSpec("選択モード", "set_operation_mode", ("モデル作成: 選択モード",)),
            WorkflowButtonSpec("直線を登録", "add_geometry_line"),
            WorkflowButtonSpec("直線をマウス作成", "set_draw_mode", ("line",)),
            WorkflowButtonSpec("正多角形/円形トンネル", "add_tunnel_geometry"),
            WorkflowButtonSpec("矩形選択モード", "set_operation_mode", ("モデル作成: 矩形選択モード",)),
            WorkflowButtonSpec("閉領域をマウス作成", "set_draw_mode", ("region",)),
            WorkflowButtonSpec("閉領域を確定", "finish_region_drawing"),
            WorkflowButtonSpec("補助線", "add_helper_line"),
            WorkflowButtonSpec("補助線をマウス作成", "set_draw_mode", ("helper",)),
            WorkflowButtonSpec("選択線を編集", "edit_selected_geometry_line"),
            WorkflowButtonSpec("線交点で分割", "split_lines_at_intersections_async"),
            WorkflowButtonSpec("選択線をトリム", "trim_selected_geometry_line"),
            WorkflowButtonSpec("選択線を延長", "extend_selected_geometry_line"),
            WorkflowButtonSpec("DXF/SXF/GF1線形読込", "import_cad_lines"),
            WorkflowButtonSpec("浸透CSV水圧読込", "import_pore_pressure_csv"),
            WorkflowButtonSpec("選択形状を削除", "delete_selected_geometry"),
            WorkflowButtonSpec("近接点のチェック", "run_near_point_check"),
            WorkflowButtonSpec("決定: 自動ブロック化", "confirm_geometry_blocks"),
        ),
    ),
    WorkflowGroupSpec(
        "2. メッシュ分割",
        (
            WorkflowButtonSpec("選択モード", "set_operation_mode", ("メッシュ分割: 選択モード",)),
            WorkflowButtonSpec("ブロック指定", "assign_selected_block"),
            WorkflowButtonSpec("ブロック解除", "release_selected_block"),
            WorkflowButtonSpec("オートメッシュ(混合)", "set_auto_mixed_mesh_mode"),
            WorkflowButtonSpec("分割幅を設定", "set_mesh_division_width"),
            WorkflowButtonSpec("局所細分", "add_mesh_refinement"),
            WorkflowButtonSpec("確認: メッシュ生成", "confirm_mesh_generation"),
        ),
    ),
    WorkflowGroupSpec(
        "3. ステージ設定",
        (
            WorkflowButtonSpec("追加", "add_stage"),
            WorkflowButtonSpec("要素プロパティ", "add_stage_material_change"),
            WorkflowButtonSpec("節点自由度拘束", "add_fix_boundary_condition"),
            WorkflowButtonSpec("ローラ/ピン支点", "add_support_preset"),
            WorkflowButtonSpec("強制変位", "add_prescribed_displacement"),
            WorkflowButtonSpec("MPC拘束", "add_mpc_constraint"),
            WorkflowButtonSpec("節点集中荷重", "add_nodal_load"),
            WorkflowButtonSpec("座標系分布荷重", "add_edge_load"),
            WorkflowButtonSpec("自重", "add_gravity_load"),
            WorkflowButtonSpec("節点水圧", "add_nodal_pore_pressure"),
            WorkflowButtonSpec("水位/水圧条件", "add_hydro_boundary"),
            WorkflowButtonSpec("応力解放率", "set_stress_release"),
            WorkflowButtonSpec("編集/確定", "_after_form_change", ("ステージ設定を確定しました",)),
        ),
    ),
    WorkflowGroupSpec(
        "4-7. 解析・Post・計算書",
        (
            WorkflowButtonSpec("解析実行", "run_solver"),
            WorkflowButtonSpec("結果確認", "show_last_run_path"),
            WorkflowButtonSpec("計算書作成", "show_report_note"),
            WorkflowButtonSpec("データ保存", "save_input"),
        ),
    ),
)


def workflow_panel_action_groups() -> list[dict[str, Any]]:
    """Return a UI-independent description of the workflow panel controls."""

    return [
        {
            "title": group.title,
            "buttons": [
                {"label": button.label, "callback": button.callback, "args": list(button.args)}
                for button in group.buttons
            ],
        }
        for group in WORKFLOW_PANEL_GROUPS
    ]


def workflow_panel_layout_contract() -> dict[str, Any]:
    """Return layout guarantees for the top wizard and detailed action panel."""

    return {
        "schema": "geofem.gui.workflow_panel_layout.v1",
        "primary_placement": "model_top_left_wizard",
        "details_placement": "workflow_diagnostics_panel",
        "scrollable": False,
        "button_min_height": WORKFLOW_BUTTON_MIN_HEIGHT,
        "ribbon_min_height": WORKFLOW_RIBBON_MIN_HEIGHT,
        "ribbon_max_height": WORKFLOW_RIBBON_MAX_HEIGHT,
        "ribbon_button_min_height": WORKFLOW_RIBBON_BUTTON_MIN_HEIGHT,
        "ribbon_back_button_width": WORKFLOW_RIBBON_BACK_BUTTON_WIDTH,
        "ribbon_jump_button_width": WORKFLOW_RIBBON_JUMP_BUTTON_WIDTH,
        "ribbon_focus_button_width": WORKFLOW_RIBBON_FOCUS_BUTTON_WIDTH,
        "ribbon_navigation_width": WORKFLOW_RIBBON_NAV_WIDTH,
        "ribbon_navigation_fixed": True,
        "button_role_property": "workflow",
        "compact_guidance_columns": list(WORKFLOW_COMPACT_COLUMNS),
        "guidance_table_max_height": WORKFLOW_GUIDANCE_TABLE_MAX_HEIGHT,
        "summary_band": False,
        "next_action_band": True,
        "previous_action_button": False,
        "canonical_navigation": "workflow_ribbon",
        "diagnostic_navigation_aliases_visible": False,
        "selected_detail_label": True,
        "diagnostics_are_secondary": True,
        "vertical_compression": "responsive",
        "compact_vertical_compression": "single_row_summary",
        "expanded_vertical_compression": "two_row_top_grid",
    }


def build_workflow_ribbon(owner: Any, qt: Mapping[str, Any]) -> Any:
    """Build the always-visible workflow wizard above the model view."""

    QWidget = qt["QWidget"]
    QHBoxLayout = qt["QHBoxLayout"]
    QVBoxLayout = qt["QVBoxLayout"]
    QGridLayout = qt["QGridLayout"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QSizePolicy = qt["QSizePolicy"]

    ribbon = QWidget()
    ribbon.setProperty("role", "workflowRibbon")
    ribbon.setProperty("placement", "model_top_left")
    ribbon.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    ribbon.setMinimumHeight(104)
    ribbon.setMaximumHeight(116)
    layout = QVBoxLayout(ribbon)
    layout.setContentsMargins(6, 4, 6, 4)
    layout.setSpacing(4)

    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.setSpacing(6)
    owner.workflow_ribbon_title_label = QLabel(_workflow_text(owner, "作業ウィザード", "Workflow"))
    owner.workflow_ribbon_title_label.setMaximumHeight(WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT)
    header_row.addWidget(owner.workflow_ribbon_title_label)

    owner.workflow_ribbon_reference_label = QLabel()
    owner.workflow_ribbon_reference_label.setProperty("informationRole", "warning")
    owner.workflow_ribbon_reference_label.setMaximumHeight(WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT)
    owner.workflow_ribbon_reference_label.setVisible(False)
    header_row.addWidget(owner.workflow_ribbon_reference_label)

    owner.workflow_ribbon_progress_label = QLabel()
    owner.workflow_ribbon_progress_label.setVisible(False)

    owner.workflow_ribbon_next_label = QLabel()
    owner.workflow_ribbon_next_label.setWordWrap(False)
    owner.workflow_ribbon_next_label.setMaximumHeight(WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT)
    owner.workflow_ribbon_next_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    header_row.addWidget(owner.workflow_ribbon_next_label, 1)

    owner.workflow_ribbon_focus_button = QPushButton(_workflow_text(owner, "表示を広げる", "Expand View"))
    _prepare_ribbon_nav_button(owner.workflow_ribbon_focus_button, QSizePolicy, WORKFLOW_RIBBON_FOCUS_BUTTON_WIDTH)
    owner.workflow_ribbon_focus_button.setProperty("role", "workspace_focus_button")
    owner.workflow_ribbon_focus_button.setCheckable(True)
    owner.workflow_ribbon_focus_button.setAccessibleName(_workflow_text(owner, "モデル・結果表示を広げる", "Expand model or result view"))
    owner.workflow_ribbon_focus_button.setToolTip(
        _workflow_text(
            owner,
            "モデル・結果表示を中央いっぱいに広げます。",
            "Expand the model or result view across the workspace.",
        )
    )
    owner.workflow_ribbon_focus_button.toggled.connect(owner.set_workspace_focus_mode)
    header_row.addWidget(owner.workflow_ribbon_focus_button)

    owner.workflow_ribbon_overview_button = QPushButton(_workflow_text(owner, "工程一覧", "All Steps"))
    _prepare_ribbon_nav_button(owner.workflow_ribbon_overview_button, QSizePolicy, 84)
    owner.workflow_ribbon_overview_button.setProperty("role", "workflow_ribbon_button")
    owner.workflow_ribbon_overview_button.clicked.connect(
        lambda _checked=False: owner._activate_panel("workflow")
    )
    header_row.addWidget(owner.workflow_ribbon_overview_button)

    nav_container = QWidget()
    nav_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    nav_container.setFixedWidth(WORKFLOW_RIBBON_NAV_WIDTH)
    nav_container.setMinimumHeight(WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT)
    nav_container.setMaximumHeight(WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT)
    nav_layout = QHBoxLayout(nav_container)
    nav_layout.setContentsMargins(0, 0, 0, 0)
    nav_layout.setSpacing(WORKFLOW_RIBBON_NAV_SPACING)

    owner.workflow_ribbon_back_button = QPushButton(_workflow_text(owner, "戻る", "Back"))
    _prepare_ribbon_nav_button(owner.workflow_ribbon_back_button, QSizePolicy, WORKFLOW_RIBBON_BACK_BUTTON_WIDTH)
    owner.workflow_ribbon_back_button.setProperty("role", "workflow_ribbon_button")
    owner.workflow_ribbon_back_button.clicked.connect(lambda _checked=False: jump_to_previous_workflow_step(owner, qt))
    nav_layout.addWidget(owner.workflow_ribbon_back_button)
    owner.workflow_ribbon_jump_button = QPushButton(_workflow_text(owner, "次へ移動", "Jump"))
    _prepare_ribbon_nav_button(owner.workflow_ribbon_jump_button, QSizePolicy, WORKFLOW_RIBBON_JUMP_BUTTON_WIDTH)
    owner.workflow_ribbon_jump_button.setProperty("role", "workflow_ribbon_button")
    owner.workflow_ribbon_jump_button.clicked.connect(lambda _checked=False: jump_to_next_workflow_step(owner, qt))
    nav_layout.addWidget(owner.workflow_ribbon_jump_button)
    owner.workflow_ribbon_navigation = nav_container
    header_row.addWidget(nav_container)
    layout.addLayout(header_row)

    step_container = QWidget()
    owner.workflow_ribbon_step_container = step_container
    owner.workflow_ribbon_steps_layout = QGridLayout(step_container)
    owner.workflow_ribbon_steps_layout.setContentsMargins(0, 0, 0, 0)
    owner.workflow_ribbon_steps_layout.setHorizontalSpacing(4)
    owner.workflow_ribbon_steps_layout.setVerticalSpacing(4)
    owner.workflow_ribbon_step_columns = 6
    layout.addWidget(step_container)
    owner.workflow_ribbon_step_scroll = None
    owner.workflow_ribbon_step_buttons = []
    owner.workflow_ribbon = ribbon
    _refresh_workflow_from_current_state(owner, qt)
    compact = bool(getattr(owner, "_is_compact_gui", lambda: True)())
    set_workflow_ribbon_compact(owner, compact)
    return ribbon


def set_workflow_ribbon_compact(owner: Any, compact: bool) -> None:
    """Switch between a single-row summary and the full step grid."""

    ribbon = getattr(owner, "workflow_ribbon", None)
    step_container = getattr(owner, "workflow_ribbon_step_container", None)
    overview = getattr(owner, "workflow_ribbon_overview_button", None)
    if ribbon is None or step_container is None:
        return
    compact = bool(compact)
    owner.workflow_ribbon_compact = compact
    step_container.setVisible(not compact)
    if compact:
        ribbon.setMinimumHeight(WORKFLOW_RIBBON_MIN_HEIGHT)
        ribbon.setMaximumHeight(WORKFLOW_RIBBON_MAX_HEIGHT)
    else:
        ribbon.setMinimumHeight(104)
        ribbon.setMaximumHeight(116)
    if overview is not None:
        overview.setText(_workflow_text(owner, "工程一覧", "All Steps"))
        overview.setVisible(compact and int(getattr(owner, "width", lambda: 0)()) >= 1180)


def build_workflow_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    """Build the detailed workflow QWidget while keeping MainWindow free of layout detail."""

    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QLabel = qt["QLabel"]
    QTableWidget = qt["QTableWidget"]
    QHeaderView = qt["QHeaderView"]
    QPushButton = qt["QPushButton"]
    QGroupBox = qt["QGroupBox"]
    QScrollArea = qt["QScrollArea"]
    QSizePolicy = qt["QSizePolicy"]
    Qt = qt["Qt"]

    page = QWidget()
    outer_layout = QVBoxLayout(page)
    outer_layout.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)
    summary_group = QGroupBox("ワークフロー診断")
    summary_layout = QVBoxLayout(summary_group)
    owner.workflow_progress_label = QLabel()
    owner.workflow_progress_label.setWordWrap(True)
    owner.workflow_progress_label.setProperty("informationRole", "primary")
    owner.workflow_progress_label.setVisible(False)
    owner.workflow_next_step_label = QLabel()
    owner.workflow_next_step_label.setWordWrap(True)
    owner.workflow_next_step_label.setProperty("informationRole", "primary")
    summary_layout.addWidget(owner.workflow_next_step_label)
    owner.workflow_missing_label = QLabel()
    owner.workflow_missing_label.setWordWrap(True)
    owner.workflow_missing_label.setProperty("informationRole", "warning")
    summary_layout.addWidget(owner.workflow_missing_label)
    owner.workflow_back_button = QPushButton(_workflow_text(owner, "前の工程へ戻る", "Back to Previous Step"), summary_group)
    _prepare_workflow_button(owner.workflow_back_button, QSizePolicy)
    owner.workflow_back_button.clicked.connect(lambda _checked=False: jump_to_previous_workflow_step(owner, qt))
    owner.workflow_back_button.setProperty("navigationAliasOnly", True)
    owner.workflow_back_button.setVisible(False)
    owner.workflow_back_button.setEnabled(False)
    owner.workflow_jump_button = QPushButton(_workflow_text(owner, "次の工程へ移動", "Jump to Next Step"), summary_group)
    _prepare_workflow_button(owner.workflow_jump_button, QSizePolicy)
    owner.workflow_jump_button.clicked.connect(lambda _checked=False: jump_to_next_workflow_step(owner, qt))
    owner.workflow_jump_button.setProperty("navigationAliasOnly", True)
    owner.workflow_jump_button.setVisible(False)
    owner.workflow_jump_button.setEnabled(False)
    layout.addWidget(summary_group)

    owner.workflow_guidance_table = QTableWidget(0, len(WORKFLOW_COMPACT_COLUMNS))
    _set_compact_table_headers(owner.workflow_guidance_table, locale=_owner_locale(owner))
    owner.workflow_guidance_table.setMaximumHeight(WORKFLOW_GUIDANCE_TABLE_MAX_HEIGHT)
    owner.workflow_guidance_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
    owner.workflow_guidance_table.doubleClicked.connect(lambda _index: owner.jump_to_workflow_guidance_row())
    owner.workflow_guidance_table.itemSelectionChanged.connect(lambda: _update_selected_step_detail(owner, qt))
    layout.addWidget(owner.workflow_guidance_table)

    owner.workflow_selected_detail_label = QLabel()
    owner.workflow_selected_detail_label.setWordWrap(True)
    owner.workflow_selected_detail_label.setProperty("informationRole", "detail")
    layout.addWidget(owner.workflow_selected_detail_label)

    owner.workflow_refresh_button = QPushButton(gui_message("workflow.refresh_button", locale=_owner_locale(owner)))
    _prepare_workflow_button(owner.workflow_refresh_button, QSizePolicy)
    owner.workflow_refresh_button.clicked.connect(owner.refresh_workflow_guidance)
    layout.addWidget(owner.workflow_refresh_button)

    shortcut_title = QLabel("操作ショートカット")
    shortcut_title.setProperty("informationRole", "primary")
    layout.addWidget(shortcut_title)
    for group_spec in WORKFLOW_PANEL_GROUPS:
        group = QGroupBox(group_spec.title)
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(6)
        for button_spec in group_spec.buttons:
            button = QPushButton(button_spec.label)
            _prepare_workflow_button(button, QSizePolicy)
            button.setToolTip(button_spec.label)
            button.clicked.connect(_button_callback(owner, button_spec))
            group_layout.addWidget(button)
        layout.addWidget(group)
    layout.addStretch(1)
    scroll.setWidget(content)
    outer_layout.addWidget(scroll)
    owner.workflow_scroll_area = scroll
    owner.refresh_workflow_guidance()
    return page


def refresh_workflow_guidance_view(owner: Any, qt: Mapping[str, Any]) -> None:
    _refresh_workflow_from_current_state(owner, qt)


def jump_to_next_workflow_step(owner: Any, qt: Mapping[str, Any]) -> None:
    guidance = getattr(owner, "workflow_guidance", None)
    if not isinstance(guidance, Mapping):
        guidance = _build_owner_workflow_guidance(owner)
        owner.workflow_guidance = guidance
    step = _adjacent_workflow_step(owner, guidance, 1)
    if step is None:
        status_bar = owner.statusBar() if hasattr(owner, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(_workflow_text(owner, "最後の表示作業です。", "Already at the last visible step."))
        return
    if _execution_gate_blocks(owner, step, guidance):
        return
    _activate_workflow_step(owner, step)


def jump_to_previous_workflow_step(owner: Any, qt: Mapping[str, Any]) -> None:
    guidance = getattr(owner, "workflow_guidance", None)
    if not isinstance(guidance, Mapping):
        guidance = _build_owner_workflow_guidance(owner)
        owner.workflow_guidance = guidance
    step = _adjacent_workflow_step(owner, guidance, -1)
    if step is None:
        status_bar = owner.statusBar() if hasattr(owner, "statusBar") else None
        if status_bar is not None:
            status_bar.showMessage(_workflow_text(owner, "先頭の工程です。", "Already at the first step."))
        return
    _activate_workflow_step(owner, step)


def _refresh_workflow_from_current_state(owner: Any, qt: Mapping[str, Any]) -> None:
    locale = _owner_locale(owner)
    guidance = _build_owner_workflow_guidance(owner)
    owner.workflow_guidance = guidance
    ratio = float(guidance.get("completion_ratio", 0.0)) * 100.0
    next_step = _adjacent_workflow_step(owner, guidance, 1) or guidance.get("next_step") or {}
    next_label = str(next_step.get("label", gui_message("workflow.default_next_label", locale=locale)))
    next_action = str(next_step.get("next_action", gui_message("workflow.default_next_action", locale=locale)))
    progress_text = gui_message("workflow.progress_ratio", locale=locale, ratio=ratio)
    progress_label = getattr(owner, "workflow_progress_label", None)
    if progress_label is not None:
        progress_label.setText(gui_message("workflow.progress", locale=locale, ratio=ratio, next_label=next_label, next_action=next_action))
    next_label_widget = getattr(owner, "workflow_next_step_label", None)
    if next_label_widget is not None:
        next_label_widget.setText(f"{_workflow_text(owner, '次に行う作業', 'Next')}: {next_label} - {next_action}")
    missing_label = getattr(owner, "workflow_missing_label", None)
    if missing_label is not None:
        missing_paths = [_display_input_reference(owner, item) for item in next_step.get("missing_paths", [])] if isinstance(next_step, Mapping) else []
        missing_label.setText(f"{gui_message('workflow.table.missing', locale=locale)}: {', '.join(missing_paths) if missing_paths else _workflow_text(owner, 'なし', 'None')}")
    refresh_button = getattr(owner, "workflow_refresh_button", None)
    if refresh_button is not None:
        refresh_button.setText(gui_message("workflow.refresh_button", locale=locale))
    jump_button = getattr(owner, "workflow_jump_button", None)
    if jump_button is not None:
        jump_button.setText(_workflow_text(owner, "次の工程へ移動", "Jump to Next Step"))
    back_button = getattr(owner, "workflow_back_button", None)
    if back_button is not None:
        back_button.setText(_workflow_text(owner, "前の工程へ戻る", "Back to Previous Step"))
        back_button.setEnabled(
            not bool(back_button.property("navigationAliasOnly"))
            and _adjacent_workflow_step(owner, guidance, -1) is not None
        )

    _sync_workflow_ribbon(owner, qt, guidance, progress_text, next_label, next_action, locale)
    if getattr(owner, "workflow_guidance_table", None) is not None:
        _populate_guidance_table(owner, qt, guidance, locale)
        _select_guidance_row(owner, qt, guidance)
        _update_selected_step_detail(owner, qt)


def jump_to_workflow_guidance_row(owner: Any, qt: Mapping[str, Any]) -> None:
    table = getattr(owner, "workflow_guidance_table", None)
    if table is None:
        return
    row = table.currentRow()
    if row < 0:
        jump_to_next_workflow_step(owner, qt)
        return
    item = table.item(row, 0)
    Qt = qt["Qt"]
    step = item.data(Qt.ItemDataRole.UserRole) if item is not None else {}
    _activate_workflow_step(owner, step)


def _activate_workflow_step(owner: Any, step: Mapping[str, Any] | Any) -> None:
    panel = str(step.get("panel", "")) if isinstance(step, Mapping) else ""
    if not panel:
        return
    guidance = getattr(owner, "workflow_guidance", None)
    if _execution_gate_blocks(owner, step, guidance if isinstance(guidance, Mapping) else None):
        return
    if hasattr(owner, "_activate_panel"):
        owner._activate_panel(panel)
    elif panel in owner.panel_pages:
        owner.panel_stack.setCurrentWidget(owner.panel_pages[panel])
    if isinstance(step, Mapping):
        owner.statusBar().showMessage(
            gui_message(
                "workflow.jump_status",
                locale=_owner_locale(owner),
                label=step.get("label", panel),
                next_action=step.get("next_action", ""),
            )
        )


def _build_owner_workflow_guidance(owner: Any) -> Mapping[str, Any]:
    locale = _owner_locale(owner)
    if hasattr(owner, "_active_result_dir"):
        result_dir = owner._active_result_dir()
    else:
        result_dir = owner.last_run_dir / "results" if owner.last_run_dir is not None else None
    return build_workflow_guidance(owner.cfg, result_dir=result_dir, locale=locale)


def _workflow_step_index(owner: Any, guidance: Mapping[str, Any]) -> int:
    steps = [row for row in guidance.get("steps", []) if isinstance(row, Mapping)]
    if not steps:
        return -1
    current_panel = str(getattr(owner, "current_panel_key", "") or "")
    for index, step in enumerate(steps):
        if str(step.get("panel", "")) == current_panel or str(step.get("id", "")) == current_panel:
            return index
    return -1


def _adjacent_workflow_step(owner: Any, guidance: Mapping[str, Any], direction: int) -> Mapping[str, Any] | None:
    steps = [row for row in guidance.get("steps", []) if isinstance(row, Mapping)]
    if not steps:
        return None
    current_index = _workflow_step_index(owner, guidance)
    if current_index < 0:
        return steps[0] if direction > 0 else None
    target_index = current_index + (1 if direction > 0 else -1)
    if target_index < 0 or target_index >= len(steps):
        return None
    return steps[target_index]


def _sync_workflow_ribbon(
    owner: Any,
    qt: Mapping[str, Any],
    guidance: Mapping[str, Any],
    progress_text: str,
    next_label: str,
    next_action: str,
    locale: str,
) -> None:
    steps = [row for row in guidance.get("steps", []) if isinstance(row, Mapping)]
    current_index = _workflow_step_index(owner, guidance)
    current_step = steps[current_index] if 0 <= current_index < len(steps) else {}
    current_label = str(current_step.get("label", "")) if isinstance(current_step, Mapping) else ""
    title = getattr(owner, "workflow_ribbon_title_label", None)
    if title is not None:
        if bool(getattr(owner, "workflow_ribbon_compact", False)):
            prefix = _workflow_text(owner, "工程", "Step")
            position = f"{current_index + 1}/{len(steps)}" if current_index >= 0 and steps else f"-/{len(steps)}"
            title.setText(f"{prefix} {position}: {current_label or '-'}")
        else:
            title.setText(_workflow_text(owner, "作業ウィザード", "Workflow"))
    progress = getattr(owner, "workflow_ribbon_progress_label", None)
    if progress is not None:
        progress.setText(progress_text)
    next_widget = getattr(owner, "workflow_ribbon_next_label", None)
    if next_widget is not None:
        next_widget.setText(f"{_workflow_text(owner, '次', 'Next')}: {next_label}")
        next_widget.setToolTip(f"{next_label} - {next_action}")
    refresh = getattr(owner, "workflow_ribbon_refresh_button", None)
    if refresh is not None:
        refresh.setText(gui_message("workflow.refresh_button", locale=locale))
    jump = getattr(owner, "workflow_ribbon_jump_button", None)
    if jump is not None:
        jump.setText(_workflow_text(owner, "次へ移動", "Jump"))
    back = getattr(owner, "workflow_ribbon_back_button", None)
    if back is not None:
        back.setText(_workflow_text(owner, "戻る", "Back"))
        back.setEnabled(_adjacent_workflow_step(owner, guidance, -1) is not None)
    overview = getattr(owner, "workflow_ribbon_overview_button", None)
    if overview is not None:
        overview.setText(_workflow_text(owner, "工程一覧", "All Steps"))
    _ensure_ribbon_step_buttons(owner, qt, steps)
    display_next = _adjacent_workflow_step(owner, guidance, 1)
    next_id = str(display_next.get("id", "")) if isinstance(display_next, Mapping) else ""
    current_panel = str(getattr(owner, "current_panel_key", "") or "")
    blockers = _input_issue_blockers(owner, guidance)
    for index, (button, step) in enumerate(zip(getattr(owner, "workflow_ribbon_step_buttons", []), steps), start=1):
        is_current = current_panel and current_panel in {str(step.get("id", "")), str(step.get("panel", ""))}
        has_issue = _step_has_input_issue(owner, step, blockers)
        result_locked = _result_navigation_locked(owner)
        result_allowed = (not result_locked) or _result_lock_allows_step(owner, step)
        if not result_allowed:
            status = "locked"
        elif is_current:
            status = "current_missing" if step.get("required") and not step.get("completed") else ("current_issue" if has_issue else "current")
        else:
            status = "issue" if has_issue else ("ok" if step.get("completed") else ("next" if str(step.get("id", "")) == next_id else ("required" if step.get("required") else "optional")))
        button.setText(f"{index} {_compact_ribbon_step_label(step, owner)}")
        tooltip = _step_tooltip(owner, step, locale, blockers)
        if not result_allowed:
            tooltip = f"{_result_lock_message(owner)}\n{tooltip}"
        button.setToolTip(tooltip)
        button.setEnabled(result_allowed)
        button.setProperty("workflowStatus", status)
        button.setStyleSheet(_ribbon_button_style(status))


def _ensure_ribbon_step_buttons(owner: Any, qt: Mapping[str, Any], steps: list[Mapping[str, Any]]) -> None:
    layout = getattr(owner, "workflow_ribbon_steps_layout", None)
    if layout is None:
        return
    buttons = list(getattr(owner, "workflow_ribbon_step_buttons", []))
    if len(buttons) == len(steps):
        return
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
    QPushButton = qt["QPushButton"]
    QSizePolicy = qt["QSizePolicy"]
    buttons = []
    columns = max(1, int(getattr(owner, "workflow_ribbon_step_columns", 1) or 1))
    for index, step in enumerate(steps):
        button = QPushButton()
        button.setMinimumHeight(24)
        button.setMaximumHeight(28)
        button.setMinimumWidth(0)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setProperty("role", "workflow_step")
        target_step = dict(step)
        button.clicked.connect(lambda _checked=False, row=target_step: _activate_workflow_step(owner, row))
        if columns > 1:
            layout.addWidget(button, index // columns, index % columns)
        else:
            layout.addWidget(button)
        buttons.append(button)
    if columns <= 1 and hasattr(layout, "addStretch"):
        layout.addStretch(1)
    owner.workflow_ribbon_step_buttons = buttons


def _ribbon_button_style(status: str) -> str:
    if status == "current":
        return "QPushButton { background: #e8f1ff; border: 2px solid #2563eb; color: #111827; font-weight: 700; }"
    if status in {"current_missing", "current_issue"}:
        return "QPushButton { background: #fff1f2; border: 2px solid #dc2626; color: #991b1b; font-weight: 700; }"
    if status == "issue":
        return "QPushButton { background: #fff1f2; border: 1px solid #dc2626; color: #991b1b; font-weight: 700; }"
    if status == "ok":
        return "QPushButton { background: #e2f6e5; border: 1px solid #78b983; }"
    if status == "next":
        return "QPushButton { background: #fff0d6; border: 2px solid #d18a00; font-weight: 600; }"
    if status == "required":
        return "QPushButton { background: #fff7ed; border: 1px solid #dc2626; color: #991b1b; }"
    if status == "locked":
        return "QPushButton { background: #f3f4f6; border: 1px solid #9ca3af; color: #6b7280; }"
    return "QPushButton { background: #f7f9fc; border: 1px solid #c8d0dc; }"


def _execution_gate_blocks(owner: Any, step: Mapping[str, Any] | Any, guidance: Mapping[str, Any] | None = None) -> bool:
    step_id = str(step.get("id", "")) if isinstance(step, Mapping) else ""
    panel = str(step.get("panel", "")) if isinstance(step, Mapping) else ""
    target_step = step_id or panel
    if _result_lock_blocks(owner, step, show=True):
        return True
    if target_step == "mesh":
        if hasattr(owner, "analysis_reference_mode_active"):
            try:
                if bool(owner.analysis_reference_mode_active()):
                    return False
            except Exception:
                pass
        blockers = []
        if hasattr(owner, "input_execution_blockers"):
            blockers = owner.input_execution_blockers(guidance)
        geometry_blockers = [item for item in blockers if isinstance(item, Mapping) and str(item.get("source", "")) == "geometry_closure"]
        if not geometry_blockers:
            return False
        if hasattr(owner, "show_input_execution_blockers"):
            owner.show_input_execution_blockers(str(step.get("label", panel or step_id)) if isinstance(step, Mapping) else "")
        return True
    if step_id not in EXECUTION_AND_LATER_STEP_IDS and panel not in EXECUTION_AND_LATER_STEP_IDS:
        return False
    if target_step in {"results", "report"}:
        return _result_navigation_gate_blocks(owner, target_step)
    if target_step == "solver":
        return False
    blockers = []
    if hasattr(owner, "input_execution_blockers"):
        blockers = owner.input_execution_blockers(guidance)
    if not blockers:
        return False
    if hasattr(owner, "show_input_execution_blockers"):
        owner.show_input_execution_blockers(str(step.get("label", panel or step_id)) if isinstance(step, Mapping) else "")
    return True


def _result_navigation_gate_blocks(owner: Any, step_id: str) -> bool:
    if step_id not in {"results", "report"}:
        return False
    if hasattr(owner, "_active_result_dir"):
        results_dir = owner._active_result_dir()
    else:
        run_dir = getattr(owner, "last_run_dir", None)
        results_dir = run_dir / "results" if run_dir is not None else None
    if results_dir is None or not results_dir.exists():
        message = _workflow_text(owner, "解析結果がまだありません。解析実行後に結果へ移動できます。", "No analysis results are available yet. Run the analysis before opening results.")
        _show_workflow_navigation_message(owner, message, "results")
        return True
    if step_id == "results":
        has_result = any((results_dir / name).exists() for name in ("summary.json", "failure_report.json"))
        if not has_result:
            message = _workflow_text(owner, "summary.json または failure_report.json がまだありません。解析完了後に結果へ移動できます。", "summary.json or failure_report.json is not available yet.")
            _show_workflow_navigation_message(owner, message, "results")
            return True
        return False
    has_result = any((results_dir / name).exists() for name in ("summary.json", "failure_report.json"))
    has_report = any((results_dir / name).exists() for name in ("calculation_report.html", "standard_report.html", "gui_report.html", "report_manifest.json", "calculation_report_manifest.json"))
    if not has_report and not has_result:
        message = _workflow_text(owner, "帳票ファイルがまだありません。解析完了後、または帳票作成後に移動できます。", "No report artifact is available yet.")
        _show_workflow_navigation_message(owner, message, "report")
        return True
    return False


def _result_navigation_locked(owner: Any) -> bool:
    if hasattr(owner, "result_navigation_locked"):
        try:
            return bool(owner.result_navigation_locked())
        except Exception:
            return False
    return bool(
        getattr(owner, "last_run_dir", None) is not None
        or getattr(owner, "_loaded_result_summary_path", None) is not None
    )


def _result_lock_allows_step(owner: Any, step: Mapping[str, Any] | Any) -> bool:
    step_id = str(step.get("id", "")) if isinstance(step, Mapping) else ""
    panel = str(step.get("panel", "")) if isinstance(step, Mapping) else ""
    if hasattr(owner, "result_navigation_target_allowed"):
        try:
            return bool(owner.result_navigation_target_allowed(panel or step_id))
        except Exception:
            pass
    return (step_id or panel) in RESULT_LOCK_ALLOWED_STEP_IDS or panel in RESULT_LOCK_ALLOWED_STEP_IDS


def _result_lock_message(owner: Any) -> str:
    if hasattr(owner, "result_navigation_lock_message"):
        try:
            return str(owner.result_navigation_lock_message())
        except Exception:
            pass
    return _workflow_text(owner, "解析結果が残っているため、結果リセットまで9 実行、10 結果、11 帳票のみ移動できます。", "Analysis results are present. Until results are reset, only Run, Results, and Report are available.")


def _result_lock_blocks(owner: Any, step: Mapping[str, Any] | Any, *, show: bool = False) -> bool:
    if not _result_navigation_locked(owner) or _result_lock_allows_step(owner, step):
        return False
    if show:
        label = str(step.get("label", step.get("panel", ""))) if isinstance(step, Mapping) else ""
        if hasattr(owner, "show_result_navigation_locked_message"):
            owner.show_result_navigation_locked_message(label)
        else:
            _show_workflow_navigation_message(owner, _result_lock_message(owner), "results")
    return True


def _show_workflow_navigation_message(owner: Any, message: str, panel: str) -> None:
    status_bar = owner.statusBar() if hasattr(owner, "statusBar") else None
    if status_bar is not None:
        status_bar.showMessage(message)
    label = getattr(owner, "results_summary" if panel == "results" else "report_summary", None)
    if label is not None:
        label.setText(message)


def _input_issue_blockers(owner: Any, guidance: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not hasattr(owner, "input_execution_blockers"):
        return []
    try:
        return [item for item in owner.input_execution_blockers(guidance) if isinstance(item, Mapping)]
    except Exception:
        return []


def _step_has_input_issue(owner: Any, step: Mapping[str, Any], blockers: list[Mapping[str, Any]]) -> bool:
    step_id = str(step.get("id", ""))
    panel = str(step.get("panel", ""))
    if hasattr(owner, "workflow_step_has_input_issue"):
        try:
            return bool(owner.workflow_step_has_input_issue(step_id, panel, blockers))
        except Exception:
            return False
    return any(str(item.get("id", "")) == step_id or str(item.get("panel", "")) == panel for item in blockers)


def _step_issue_lines(owner: Any, step: Mapping[str, Any], blockers: list[Mapping[str, Any]], *, limit: int = 5) -> list[str]:
    step_id = str(step.get("id", ""))
    panel = str(step.get("panel", ""))
    if hasattr(owner, "workflow_step_issue_lines"):
        try:
            return [str(line) for line in owner.workflow_step_issue_lines(step_id, panel, blockers, limit=limit)][:limit]
        except Exception:
            return []
    lines = []
    for blocker in blockers:
        if str(blocker.get("id", "")) == step_id or str(blocker.get("panel", "")) == panel:
            missing = [_display_input_reference(owner, value) for value in blocker.get("missing_paths", [])] if isinstance(blocker.get("missing_paths"), list) else []
            lines.append(", ".join(missing) if missing else str(blocker.get("label", step_id)))
    return lines[:limit]


def _step_completion_line(owner: Any, step: Mapping[str, Any], blockers: list[Mapping[str, Any]], *, include_ok: bool = True) -> str:
    step_id = str(step.get("id", ""))
    panel = str(step.get("panel", ""))
    if hasattr(owner, "workflow_step_completion_line"):
        try:
            return str(owner.workflow_step_completion_line(step_id, panel, blockers, include_ok=include_ok) or "")
        except Exception:
            return ""
    return ""


def _populate_guidance_table(owner: Any, qt: Mapping[str, Any], guidance: Mapping[str, Any], locale: str) -> None:
    QTableWidgetItem = qt["QTableWidgetItem"]
    Qt = qt["Qt"]
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]

    table = owner.workflow_guidance_table
    steps = list(guidance.get("steps", []))
    current_panel = str(getattr(owner, "current_panel_key", "") or "")
    blockers = _input_issue_blockers(owner, guidance)
    table.blockSignals(True)
    table.setColumnCount(len(WORKFLOW_COMPACT_COLUMNS))
    _set_compact_table_headers(table, locale=locale)
    table.setRowCount(len(steps))
    for row, step in enumerate(steps):
        values = [
            str(row + 1),
            str(step.get("label", "")),
            str(step.get("status_label", "")),
            str(step.get("next_action", "")),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setData(Qt.ItemDataRole.UserRole, step)
            item.setToolTip(_step_tooltip(owner, step, locale, blockers))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            has_issue = _step_has_input_issue(owner, step, blockers) if isinstance(step, Mapping) else False
            if current_panel and current_panel in {str(step.get("id", "")), str(step.get("panel", ""))} and has_issue:
                item.setBackground(QBrush(QColor(255, 241, 242)))
            elif has_issue:
                item.setBackground(QBrush(QColor(255, 241, 242)))
            elif current_panel and current_panel in {str(step.get("id", "")), str(step.get("panel", ""))}:
                item.setBackground(QBrush(QColor(232, 241, 255)))
            elif step.get("completed"):
                item.setBackground(QBrush(QColor(226, 246, 229)))
            elif step.get("required"):
                item.setBackground(QBrush(QColor(255, 238, 224)))
            table.setItem(row, col, item)
    table.blockSignals(False)
    table.resizeRowsToContents()


def _set_compact_table_headers(table: Any, *, locale: str) -> None:
    table.setHorizontalHeaderLabels(
        [
            "#",
            gui_message("workflow.table.step", locale=locale),
            gui_message("workflow.table.status", locale=locale),
            gui_message("workflow.table.next", locale=locale),
        ]
    )


def _select_guidance_row(owner: Any, qt: Mapping[str, Any], guidance: Mapping[str, Any]) -> None:
    table = getattr(owner, "workflow_guidance_table", None)
    if table is None or table.rowCount() <= 0:
        return
    Qt = qt["Qt"]
    next_step = guidance.get("next_step") or {}
    next_id = str(next_step.get("id", "")) if isinstance(next_step, Mapping) else ""
    current_panel = str(getattr(owner, "current_panel_key", "") or "")
    target_row = 0
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        step = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(step, Mapping) and current_panel and current_panel in {str(step.get("id", "")), str(step.get("panel", ""))}:
            target_row = row
            break
        if isinstance(step, Mapping) and str(step.get("id", "")) == next_id:
            target_row = row
            break
    table.selectRow(target_row)


def _update_selected_step_detail(owner: Any, qt: Mapping[str, Any]) -> None:
    label = getattr(owner, "workflow_selected_detail_label", None)
    table = getattr(owner, "workflow_guidance_table", None)
    if label is None or table is None:
        return
    row = table.currentRow()
    if row < 0:
        label.setText("")
        return
    item = table.item(row, 0)
    Qt = qt["Qt"]
    step = item.data(Qt.ItemDataRole.UserRole) if item is not None else {}
    if not isinstance(step, Mapping):
        label.setText("")
        return
    missing = ", ".join(_display_input_reference(owner, value) for value in step.get("missing_paths", [])) or _workflow_text(owner, "なし", "None")
    blockers = _input_issue_blockers(owner, getattr(owner, "workflow_guidance", {}) or {})
    completion_text = _step_completion_line(owner, step, blockers, include_ok=True) or _workflow_text(owner, "なし", "None")
    issue_lines = _step_issue_lines(owner, step, blockers, limit=4)
    issue_text = "\n".join(issue_lines) if issue_lines else _workflow_text(owner, "なし", "None")
    label.setText(
        f"{step.get('label', '')}: {step.get('detail', '')}\n"
        f"{gui_message('workflow.table.next', locale=_owner_locale(owner))}: {step.get('next_action', '')}\n"
        f"{gui_message('workflow.table.missing', locale=_owner_locale(owner))}: {missing}\n"
        f"{_workflow_text(owner, '完了条件', 'Completion')}: {completion_text}\n"
        f"{_workflow_text(owner, '入力課題', 'Input issues')}: {issue_text}"
    )


def _step_tooltip(owner: Any, step: Mapping[str, Any], locale: str, blockers: list[Mapping[str, Any]] | None = None) -> str:
    required = gui_message("workflow.required", locale=locale) if step.get("required") else gui_message("workflow.optional", locale=locale)
    missing = ", ".join(_display_input_reference(owner, item) for item in step.get("missing_paths", [])) or "-"
    lines = [
        f"{step.get('label', '')} / {required}",
        f"{gui_message('workflow.table.status', locale=locale)}: {step.get('status_label', '')}",
        f"{gui_message('workflow.table.missing', locale=locale)}: {missing}",
        f"{gui_message('workflow.table.next', locale=locale)}: {step.get('next_action', '')}",
    ]
    completion = _step_completion_line(owner, step, blockers or [], include_ok=True)
    if completion:
        lines.append(f"{_workflow_text(owner, '完了条件', 'Completion')}: {completion}")
    issue_lines = _step_issue_lines(owner, step, blockers or [], limit=5)
    if issue_lines:
        lines.append(f"{_workflow_text(owner, '入力課題', 'Input issues')}:")
        lines.extend(f"- {line}" for line in issue_lines)
    return "\n".join(lines)


def _button_callback(owner: Any, spec: WorkflowButtonSpec) -> Any:
    def _handler(_checked: bool = False) -> None:
        callback = getattr(owner, spec.callback)
        callback(*spec.args)

    return _handler


def _prepare_workflow_button(button: Any, QSizePolicy: Any) -> None:
    button.setMinimumHeight(WORKFLOW_BUTTON_MIN_HEIGHT)
    button.setProperty("role", "workflow")
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def _prepare_ribbon_nav_button(button: Any, QSizePolicy: Any, width: int) -> None:
    button.setMinimumHeight(WORKFLOW_RIBBON_BUTTON_MIN_HEIGHT)
    button.setMaximumHeight(WORKFLOW_RIBBON_BUTTON_MAX_HEIGHT)
    button.setFixedWidth(width)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def _owner_locale(owner: Any) -> str:
    return str(getattr(owner, "gui_locale", DEFAULT_GUI_LOCALE) or DEFAULT_GUI_LOCALE)


def _workflow_text(owner: Any, ja: str, en: str) -> str:
    return en if _owner_locale(owner).startswith("en") else ja


def _compact_ribbon_step_label(step: Mapping[str, Any], owner: Any) -> str:
    step_id = str(step.get("id", ""))
    if _owner_locale(owner).startswith("en"):
        labels = {
            "analysis": "Analysis",
            "geometry": "Geo",
            "mesh": "Mesh",
            "materials": "Mat",
            "boundary_conditions": "BC",
            "loads": "Load",
            "stages": "Stage",
            "model_check": "Check",
            "solver": "Run",
            "results": "Post",
            "report": "Report",
        }
    else:
        labels = {
            "analysis": "解析",
            "geometry": "形状",
            "mesh": "メッシュ",
            "materials": "材料",
            "boundary_conditions": "境界",
            "loads": "荷重",
            "stages": "ステージ",
            "model_check": "確認",
            "solver": "実行",
            "results": "結果",
            "report": "帳票",
        }
    return labels.get(step_id, str(step.get("label", "")))
