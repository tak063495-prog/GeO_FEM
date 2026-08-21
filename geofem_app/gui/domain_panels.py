"""Domain-specific GUI panel builders split from MainWindow.

The functions in this module own widget layout and presentation wiring.
MainWindow remains the owner of state, callbacks, and solver interactions.
"""
from __future__ import annotations

import getpass
from typing import Any, Mapping

from geofem_app.gui.help_system import documentation_payload
from geofem_app.gui.presentation_labels import populate_labeled_combo

DOMAIN_PANEL_KEYS = (
    "mesh",
    "geometry",
    "external",
    "materials",
    "boundary_conditions",
    "loads",
    "stages",
    "results",
    "report",
)


def domain_panel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.domain_panels.v1",
        "panel_keys": list(DOMAIN_PANEL_KEYS),
        "builder_count": len(DOMAIN_PANEL_KEYS),
        "owner_boundary": "builders create widgets; owner provides state and callbacks",
    }


def _edit_undo_redo_buttons(owner: Any, QPushButton: Any) -> list[Any]:
    buttons: list[Any] = []
    for label, callback, icon_role, tooltip in (
        ("Undo", owner.undo_edit, "undo", "直前の編集を元に戻します。履歴は最大10件です。"),
        ("Redo", owner.redo_edit, "redo", "Undoで戻した編集をやり直します。履歴は最大10件です。"),
    ):
        button = QPushButton(label)
        button.clicked.connect(callback)
        if hasattr(owner, "_apply_button_icon"):
            owner._apply_button_icon(button, icon_role)
        button.setToolTip(tooltip)
        buttons.append(button)
    return buttons


def _set_operation_rule(button: Any, rule: str) -> Any:
    button.setProperty("operationRule", rule)
    return button


def build_mesh_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QPushButton = qt["QPushButton"]
    QTableWidget = qt["QTableWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    form = QFormLayout()
    self.mesh_generator = QComboBox()
    populate_labeled_combo(self.mesh_generator, "mesh_generator", ["rectangle"], locale=getattr(self, "gui_locale", "ja"))
    self.mesh_x0 = QLineEdit()
    self.mesh_x1 = QLineEdit()
    self.mesh_y0 = QLineEdit()
    self.mesh_y1 = QLineEdit()
    self.mesh_nx = QLineEdit()
    self.mesh_ny = QLineEdit()
    self.mesh_type = QComboBox()
    self.mesh_type.addItems(["QUAD4", "QUAD8", "TRI3", "TRI6"])
    self.mesh_integration = QComboBox()
    self.mesh_integration.addItems(["B-bar", "FULL", "SRI"])
    self.mesh_material = QLineEdit()
    self.mesh_boolean_expression = QLineEdit()
    self.mesh_boolean_expression.setPlaceholderText("A-B-C, (A|B)&C")
    for label, widget in [
        ("生成", self.mesh_generator),
        ("X最小", self.mesh_x0),
        ("X最大", self.mesh_x1),
        ("Y最小", self.mesh_y0),
        ("Y最大", self.mesh_y1),
        ("分割 nx", self.mesh_nx),
        ("分割 ny", self.mesh_ny),
        ("要素", self.mesh_type),
        ("積分", self.mesh_integration),
        ("材料", self.mesh_material),
    ]:
        form.addRow(label, widget)
    form.addRow("Boolean expression", self.mesh_boolean_expression)
    layout.addLayout(form)
    apply_btn = QPushButton("矩形メッシュを反映")
    apply_btn.clicked.connect(self.apply_mesh_panel)
    layout.addWidget(apply_btn)
    mesh_buttons = [
        ("自動ブロック化", self.confirm_geometry_blocks),
        ("ブロック指定", self.assign_selected_block),
        ("ブロック解除", self.release_selected_block),
        ("オートメッシュ", self.set_auto_mixed_mesh_mode),
        ("分割幅", self.set_mesh_division_width),
        ("局所細分", self.add_mesh_refinement),
        ("メッシュ生成", self.confirm_mesh_generation),
    ]
    mesh_action_buttons = []
    for label, callback in mesh_buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        mesh_action_buttons.append(button)
    self._add_panel_button_rows(layout, mesh_action_buttons, columns=2)

    mesh_control_box = QGroupBox("メッシュ制御点/ブロック分割")
    mesh_control_layout = QVBoxLayout(mesh_control_box)
    self.mesh_refinement_table = QTableWidget(0, 5)
    self.mesh_refinement_table.setHorizontalHeaderLabels(["id", "cx", "cy", "radius", "factor"])
    self.mesh_control_point_table = QTableWidget(0, 5)
    self.mesh_control_point_table.setHorizontalHeaderLabels(["id", "x", "y", "target_size", "tag"])
    self.mesh_split_line_table = QTableWidget(0, 7)
    self.mesh_split_line_table.setHorizontalHeaderLabels(["id", "x1", "y1", "x2", "y2", "target_size", "locked"])
    self.mesh_size_map_table = QTableWidget(0, 6)
    self.mesh_size_map_table.setHorizontalHeaderLabels(["id", "x", "y", "radius", "target_size", "grading"])
    self.mesh_block_table = QTableWidget(0, 6)
    self.mesh_block_table.setHorizontalHeaderLabels(["id", "name", "element_set", "active", "split_hint", "extra YAML"])
    self.mesh_quality_violation_table = QTableWidget(0, 8)
    self.mesh_quality_violation_table.setHorizontalHeaderLabels(["element", "severity", "area", "min_angle", "aspect", "skew", "reason", "repair"])
    self.mesh_quality_improvement_table = QTableWidget(0, 10)
    self.mesh_quality_improvement_table.setHorizontalHeaderLabels(["method", "before_bad", "after_bad", "before_angle", "after_angle", "before_aspect", "after_aspect", "nodes", "elements", "status"])
    self.mesh_quality_method = QComboBox()
    for label, value in [
        ("Laplace smoothing", "laplace"),
        ("Node optimization", "node_optimize"),
        ("Local remesh", "local_remesh"),
        ("Quad topology", "quad_topology"),
        ("Compare all", "all"),
    ]:
        self.mesh_quality_method.addItem(label, value)
    self.mesh_quality_method.setCurrentIndex(self.mesh_quality_method.findData("all"))
    self.mesh_quality_iterations = QLineEdit("5")
    for table in (self.mesh_refinement_table, self.mesh_control_point_table, self.mesh_split_line_table, self.mesh_size_map_table, self.mesh_block_table, self.mesh_quality_violation_table, self.mesh_quality_improvement_table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    mesh_control_layout.addWidget(QLabel("局所細分"))
    mesh_control_layout.addWidget(self.mesh_refinement_table)
    mesh_control_layout.addWidget(QLabel("制御点"))
    mesh_control_layout.addWidget(self.mesh_control_point_table)
    mesh_control_layout.addWidget(QLabel("split line constraints"))
    mesh_control_layout.addWidget(self.mesh_split_line_table)
    mesh_control_layout.addWidget(QLabel("local size map"))
    mesh_control_layout.addWidget(self.mesh_size_map_table)
    mesh_control_layout.addWidget(QLabel("ブロック分割"))
    mesh_control_layout.addWidget(self.mesh_block_table)
    mesh_control_layout.addWidget(QLabel("quality threshold violations"))
    mesh_control_layout.addWidget(self.mesh_quality_violation_table)
    quality_improvement_controls = QHBoxLayout()
    quality_improvement_controls.addWidget(QLabel("quality improvement"))
    quality_improvement_controls.addWidget(self.mesh_quality_method)
    quality_improvement_controls.addWidget(QLabel("iterations"))
    quality_improvement_controls.addWidget(self.mesh_quality_iterations)
    mesh_control_layout.addLayout(quality_improvement_controls)
    mesh_control_layout.addWidget(self.mesh_quality_improvement_table)
    mesh_control_buttons = []
    for label, callback in [
        ("細分追加", lambda _checked=False: self.add_mesh_refinement_row()),
        ("選択点→細分", self.add_selected_point_refinement),
        ("制御点追加", lambda _checked=False: self.add_mesh_control_point_row()),
        ("Split line追加", lambda _checked=False: self.add_mesh_split_line_row()),
        ("選択→Split line", self.add_selected_mesh_split_line),
        ("Size map追加", lambda _checked=False: self.add_mesh_size_map_row()),
        ("ブロック追加", lambda _checked=False: self.add_mesh_block_row()),
        ("選択ブロック分割", self.split_selected_mesh_block),
        ("選択行削除", self.remove_selected_mesh_control_rows),
        ("制御を反映", self.apply_mesh_controls_panel),
        ("品質違反抽出", self.populate_mesh_quality_violation_table_async),
        ("違反選択", self.select_mesh_quality_violations),
        ("違反修復(細分/再配置)", self.repair_selected_mesh_quality_violations),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        mesh_control_buttons.append(button)
    for label, callback in [
        ("改善比較", self.compare_mesh_quality_improvements_async),
        ("選択改善適用", self.apply_selected_mesh_quality_improvement),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        mesh_control_buttons.append(button)
    self._add_panel_button_rows(mesh_control_layout, mesh_control_buttons, columns=2)
    layout.addWidget(mesh_control_box)

    element_box = QGroupBox("GeoFEAS要素ライブラリ")
    element_layout = QVBoxLayout(element_box)
    self.element_library_table = QTableWidget(0, 8)
    self.element_library_table.setHorizontalHeaderLabels(["id", "type", "nodes", "material/property", "section/stiffness", "behavior", "active", "extra YAML"])
    self.element_library_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    element_layout.addWidget(self.element_library_table)
    element_buttons = [
        ("梁", lambda _checked=False: self.add_element_library_preset("beam")),
        ("棒", lambda _checked=False: self.add_element_library_preset("bar")),
        ("バネ", lambda _checked=False: self.add_element_library_preset("spring")),
        ("Axial spring", lambda _checked=False: self.add_element_library_preset("axial_spring")),
        ("Shear spring", lambda _checked=False: self.add_element_library_preset("shear_spring")),
        ("Bilinear spring", lambda _checked=False: self.add_element_library_preset("bilinear_spring")),
        ("ジョイント", lambda _checked=False: self.add_element_library_preset("joint")),
        ("削除", self.remove_selected_element_library_rows),
        ("要素ライブラリを反映", self.apply_element_library_panel),
    ]
    element_action_buttons = []
    for label, callback in element_buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        element_action_buttons.append(button)
    self._add_panel_button_rows(element_layout, element_action_buttons, columns=2)
    layout.addWidget(element_box)
    self.mesh_summary = QLabel()
    self.mesh_summary.setWordWrap(True)
    layout.addWidget(self.mesh_summary)
    layout.addStretch(1)
    return page

def build_geometry_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGroupBox = qt["QGroupBox"]
    QHeaderView = qt["QHeaderView"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QPushButton = qt["QPushButton"]
    QTableWidget = qt["QTableWidget"]
    QTreeWidget = qt["QTreeWidget"]
    Qt = qt["Qt"]
    QVBoxLayout = qt["QVBoxLayout"]
    QSizePolicy = qt.get("QSizePolicy")
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel("CAD的なモデル作成データを表で編集します。線・補助線・トンネルはモデルビューの作図操作とも同期します。"))

    draw_box = QGroupBox("作図パレット")
    draw_layout = QVBoxLayout(draw_box)
    draw_layout.addWidget(QLabel("CAD読込後の線も同じ形状データとして扱うため、端点ドラッグ・表編集・座標微修正で編集できます。ホイールで拡大縮小、中ボタンドラッグで移動します。"))
    self.geometry_closure_summary = QLabel("閉合診断: 未確認")
    self.geometry_closure_summary.setWordWrap(True)
    draw_layout.addWidget(self.geometry_closure_summary)

    self._add_cad_tool_button_grid(draw_layout, self._cad_palette_buttons(), columns=16)
    repair_box = QGroupBox("CAD修復候補")
    repair_layout = QVBoxLayout(repair_box)
    self.geometry_repair_table = QTableWidget(0, 6)
    self.geometry_repair_table.setHorizontalHeaderLabels(["重要度", "種類", "線分", "端点", "距離", "推奨操作"])
    self.geometry_repair_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.geometry_repair_table.setMinimumHeight(120)
    self.geometry_repair_table.cellDoubleClicked.connect(self.select_cad_repair_candidate)
    repair_layout.addWidget(self.geometry_repair_table)
    repair_buttons = []
    for label, callback in [
        ("診断更新", lambda _checked=False: self.refresh_cad_repair_diagnostics()),
        ("選択/ズーム", lambda _checked=False: self.select_cad_repair_candidate()),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        repair_buttons.append(button)
    self._add_panel_button_rows(repair_layout, repair_buttons, columns=2)
    draw_layout.addWidget(repair_box)
    layout.addWidget(draw_box)

    coordinate_box = QGroupBox("座標微修正")
    coordinate_layout = QVBoxLayout(coordinate_box)
    coordinate_form = QFormLayout()
    self.geometry_point_target = QComboBox()
    self.geometry_point_target.addItem("選択端点", "selected")
    self.geometry_point_target.addItem("線の始点", "start")
    self.geometry_point_target.addItem("線の終点", "end")
    self.geometry_point_target.addItem("線全体", "both")
    self.geometry_point_x = QLineEdit()
    self.geometry_point_y = QLineEdit()
    self.geometry_point_dx = QLineEdit("0.0")
    self.geometry_point_dy = QLineEdit("0.0")
    self.geometry_nudge_step = QLineEdit("0.01")
    coordinate_form.addRow("対象", self.geometry_point_target)
    xy_row = QWidget()
    xy_layout = QHBoxLayout(xy_row)
    xy_layout.setContentsMargins(0, 0, 0, 0)
    xy_layout.addWidget(QLabel("X"))
    xy_layout.addWidget(self.geometry_point_x)
    xy_layout.addWidget(QLabel("Y"))
    xy_layout.addWidget(self.geometry_point_y)
    coordinate_form.addRow("絶対座標", xy_row)
    dxy_row = QWidget()
    dxy_layout = QHBoxLayout(dxy_row)
    dxy_layout.setContentsMargins(0, 0, 0, 0)
    dxy_layout.addWidget(QLabel("dX"))
    dxy_layout.addWidget(self.geometry_point_dx)
    dxy_layout.addWidget(QLabel("dY"))
    dxy_layout.addWidget(self.geometry_point_dy)
    coordinate_form.addRow("相対移動", dxy_row)
    coordinate_form.addRow("微修正量", self.geometry_nudge_step)
    coordinate_layout.addLayout(coordinate_form)
    coordinate_buttons = []
    for label, callback in [
        ("選択点読込", self.load_selected_geometry_point_to_editor),
        ("座標へ移動", self.apply_geometry_point_absolute_edit),
        ("相対移動", lambda _checked=False: self.nudge_selected_geometry_point()),
        ("-X", lambda _checked=False: self.nudge_selected_geometry_point(axis="x", sign=-1.0)),
        ("+X", lambda _checked=False: self.nudge_selected_geometry_point(axis="x", sign=1.0)),
        ("-Y", lambda _checked=False: self.nudge_selected_geometry_point(axis="y", sign=-1.0)),
        ("+Y", lambda _checked=False: self.nudge_selected_geometry_point(axis="y", sign=1.0)),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        coordinate_buttons.append(button)
    self._add_uniform_panel_button_grid(coordinate_layout, coordinate_buttons, columns=7, min_width=94)
    layout.addWidget(coordinate_box)

    self.geometry_line_table = QTableWidget(0, 7)
    self.geometry_line_table.setHorizontalHeaderLabels(["id", "purpose", "layer", "x1", "y1", "x2", "y2"])
    self.geometry_line_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(QLabel("線/補助線"))
    layout.addWidget(self.geometry_line_table)
    line_buttons = [
        ("線を追加", lambda _checked=False: self.add_geometry_line_row(purpose="model")),
        ("補助線を追加", lambda _checked=False: self.add_geometry_line_row(purpose="helper")),
        ("選択行削除", lambda _checked=False: self.remove_selected_geometry_rows(self.geometry_line_table)),
        ("交点分割", self.split_lines_at_intersections_async),
        ("選択線トリム", self.trim_selected_geometry_line),
        ("選択線延長", self.extend_selected_geometry_line),
    ]
    line_action_buttons = []
    for label, callback in line_buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        line_action_buttons.append(button)
    self._add_panel_button_rows(layout, line_action_buttons, columns=2)

    self.geometry_tunnel_table = QTableWidget(0, 6)
    self.geometry_tunnel_table.setHorizontalHeaderLabels(["id", "layer", "cx", "cy", "radius", "segments"])
    self.geometry_tunnel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(QLabel("トンネル/穴"))
    layout.addWidget(self.geometry_tunnel_table)
    tunnel_buttons = [
        ("トンネル追加", self.add_geometry_tunnel_row),
        ("選択行削除", lambda _checked=False: self.remove_selected_geometry_rows(self.geometry_tunnel_table)),
        ("CAD/SXF/GF1読込", self.import_cad_lines),
        ("近接点チェック", self.run_near_point_check),
    ]
    tunnel_action_buttons = []
    for label, callback in tunnel_buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        tunnel_action_buttons.append(button)
    self._add_panel_button_rows(layout, tunnel_action_buttons, columns=2)

    layer_box = QGroupBox("CADレイヤ/スナップ表示")
    layer_layout = QVBoxLayout(layer_box)
    self.geometry_layer_table = QTableWidget(0, 8)
    self.geometry_layer_table.setHorizontalHeaderLabels(["visible", "locked", "name", "color", "linetype", "lineweight", "opacity", "source"])
    self.geometry_layer_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layer_layout.addWidget(self.geometry_layer_table)
    layer_buttons = []
    for label, callback in [
        ("レイヤ追加", lambda _checked=False: self.add_geometry_layer_row()),
        ("選択削除", lambda _checked=False: self.remove_selected_geometry_rows(self.geometry_layer_table)),
        ("全表示", lambda _checked=False: self.set_all_cad_layers_visible(True)),
        ("全非表示", lambda _checked=False: self.set_all_cad_layers_visible(False)),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layer_buttons.append(button)
    lock_layer_btn = QPushButton("Lock")
    lock_layer_btn.clicked.connect(lambda _checked=False: self.set_selected_cad_layers_locked(True))
    unlock_layer_btn = QPushButton("Unlock")
    unlock_layer_btn.clicked.connect(lambda _checked=False: self.set_selected_cad_layers_locked(False))
    layer_buttons.extend([lock_layer_btn, unlock_layer_btn])
    self._add_panel_button_rows(layer_layout, layer_buttons, columns=2)
    layout.addWidget(layer_box)

    annotation_box = QGroupBox("寸法/注記")
    annotation_layout = QVBoxLayout(annotation_box)
    self.geometry_annotation_table = QTableWidget(0, 5)
    self.geometry_annotation_table.setHorizontalHeaderLabels(["id", "layer", "x", "y", "text"])
    self.geometry_dimension_table = QTableWidget(0, 9)
    self.geometry_dimension_table.setHorizontalHeaderLabels(["id", "layer", "x1", "y1", "x2", "y2", "text", "constraint", "locked"])
    for table in (self.geometry_annotation_table, self.geometry_dimension_table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    annotation_layout.addWidget(QLabel("注記"))
    annotation_layout.addWidget(self.geometry_annotation_table)
    annotation_layout.addWidget(QLabel("寸法"))
    annotation_layout.addWidget(self.geometry_dimension_table)
    annotation_buttons = []
    for label, callback in [
        ("注記追加", lambda _checked=False: self.add_geometry_annotation_row()),
        ("寸法追加", lambda _checked=False: self.add_geometry_dimension_row()),
        ("選択削除", self.remove_selected_cad_text_rows),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        annotation_buttons.append(button)
    apply_dims_btn = QPushButton("Apply dimensions")
    apply_dims_btn.clicked.connect(self.apply_dimension_constraints_async)
    annotation_buttons.append(apply_dims_btn)
    self._add_panel_button_rows(annotation_layout, annotation_buttons, columns=2)
    layout.addWidget(annotation_box)

    overlap_box = QGroupBox("NURBS/曲線重複edge")
    overlap_layout = QVBoxLayout(overlap_box)
    self.geometry_overlap_table = QTableWidget(0, 6)
    self.geometry_overlap_table.setHorizontalHeaderLabels(["id", "curve", "coincident", "region", "role", "state"])
    self.geometry_overlap_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    overlap_layout.addWidget(self.geometry_overlap_table)
    overlap_buttons = []
    for label, callback in [
        ("重複edge再表示", self.refresh_cad_overlap_edges),
        ("選択edge抑止", lambda _checked=False: self.set_selected_cad_overlap_state("suppressed")),
        ("選択edge有効", lambda _checked=False: self.set_selected_cad_overlap_state("active")),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        overlap_buttons.append(button)
    self._add_panel_button_rows(overlap_layout, overlap_buttons, columns=2)
    layout.addWidget(overlap_box)

    curve_box = QGroupBox("解析曲線/Boolean split graph")
    curve_layout = QVBoxLayout(curve_box)
    curve_form = QFormLayout()
    self.geometry_boolean_expression = QLineEdit()
    self.geometry_boolean_expression.setPlaceholderText("A-B-C, (A|B)&C")
    self.geometry_boolean_operation = QComboBox()
    populate_labeled_combo(
        self.geometry_boolean_operation,
        "geometry_boolean",
        ["union", "intersection", "expression"],
        locale=getattr(self, "gui_locale", "ja"),
    )
    curve_form.addRow("Boolean expression", self.geometry_boolean_expression)
    curve_form.addRow("mesh operation", self.geometry_boolean_operation)
    curve_layout.addLayout(curve_form)
    self.geometry_curve_table = QTableWidget(0, 6)
    self.geometry_curve_table.setHorizontalHeaderLabels(["region", "index", "type", "role", "parameters YAML", "segments"])
    self.geometry_boolean_table = QTableWidget(0, 7)
    self.geometry_boolean_table.setHorizontalHeaderLabels(["operation", "edges", "loops", "area", "selected", "manual", "visible"])
    for table in (self.geometry_curve_table, self.geometry_boolean_table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    curve_layout.addWidget(QLabel("曲線境界"))
    curve_layout.addWidget(self.geometry_curve_table)
    self.geometry_curve_control_table = QTableWidget(0, 7)
    self.geometry_curve_control_table.setHorizontalHeaderLabels(["curve_row", "point", "role", "x", "y", "weight/radius", "locked"])
    self.geometry_curve_control_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    curve_layout.addWidget(QLabel("NURBS/Arc/Bezier control points"))
    curve_layout.addWidget(self.geometry_curve_control_table)
    curve_buttons = []
    control_extract_btn = QPushButton("Control extract")
    control_extract_btn.clicked.connect(self.populate_curve_control_point_table)
    curve_buttons.append(control_extract_btn)
    control_apply_btn = QPushButton("Control apply")
    control_apply_btn.clicked.connect(self.apply_curve_control_point_table)
    curve_buttons.append(control_apply_btn)
    for label, callback in [
        ("曲線表抽出", self.populate_geometry_curve_tables),
        ("Line追加", lambda _checked=False: self.add_geometry_curve_row(curve_type="line")),
        ("Arc追加", lambda _checked=False: self.add_geometry_curve_row(curve_type="arc")),
        ("Bezier追加", lambda _checked=False: self.add_geometry_curve_row(curve_type="bezier")),
        ("NURBS追加", lambda _checked=False: self.add_geometry_curve_row(curve_type="nurbs")),
        ("選択削除", lambda _checked=False: self.remove_selected_geometry_rows(self.geometry_curve_table)),
        ("曲線表を反映", self.apply_curve_boolean_panel),
        ("Boolean graph更新", self.rebuild_geometry_boolean_graph),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        curve_buttons.append(button)
    self._add_panel_button_rows(curve_layout, curve_buttons, columns=2)
    curve_layout.addWidget(QLabel("Boolean演算診断"))
    curve_layout.addWidget(self.geometry_boolean_table)
    boolean_buttons = []
    for label, callback in [
        ("操作選択", self.select_cad_boolean_operation_from_table),
        ("手動保存", self.store_selected_boolean_operation_as_manual_repair),
        ("手動消去", self.clear_manual_boolean_repair),
        ("重複修復", self.repair_selected_cad_overlap_edges),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        boolean_buttons.append(button)
    self._add_panel_button_rows(curve_layout, boolean_buttons, columns=2)
    layout.addWidget(curve_box)

    layout.addWidget(QLabel("閉領域 regions YAML"))
    self.geometry_regions_editor = QPlainTextEdit()
    self.geometry_regions_editor.setMaximumHeight(120)
    layout.addWidget(self.geometry_regions_editor)
    apply_btn = QPushButton("形状表を反映")
    apply_btn.clicked.connect(self.apply_geometry_panel)
    layout.addWidget(apply_btn)
    return page

def build_external_link_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    intro = QLabel("浸透流・地形・水位線・GF1/CADなど外部データをPre入力へ取り込みます。")
    intro.setWordWrap(True)
    layout.addWidget(intro)
    self.external_summary = QLabel()
    self.external_summary.setWordWrap(True)
    buttons = [
        ("浸透流CSVから節点水圧", self.import_pore_pressure_csv),
        ("水位線CSVから形状線", self.import_waterline_csv),
        ("地形/境界線CSV・CAD読込", self.import_cad_lines),
        ("GF1/DXF/SXFメタ情報確認", self.show_cad_import_summary),
        ("SXF/P21再エクスポート", self.export_geometry_sxf),
        ("SXF/P21往復検証", self.validate_sxf_roundtrip_dialog),
        ("DWG変換器検証", self.validate_dwg_converter_dialog),
    ]
    for label, callback in buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
    layout.addWidget(self.external_summary)
    layout.addStretch(1)
    return page

def build_materials_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGroupBox = qt["QGroupBox"]
    QHeaderView = qt["QHeaderView"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QPushButton = qt["QPushButton"]
    QScrollArea = qt["QScrollArea"]
    QTableWidget = qt["QTableWidget"]
    QTabWidget = qt["QTabWidget"]
    QTreeWidget = qt["QTreeWidget"]
    Qt = qt["Qt"]
    QVBoxLayout = qt["QVBoxLayout"]
    QSizePolicy = qt.get("QSizePolicy")
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    self.material_tabs = QTabWidget()
    self.material_tabs.setDocumentMode(True)

    material_list_page = QWidget()
    material_list_layout = QVBoxLayout(material_list_page)
    material_list_layout.setContentsMargins(6, 6, 6, 6)
    material_list_layout.setSpacing(6)
    material_list_status = QLabel("材料一覧")
    material_list_status.setProperty("informationRole", "primary")

    library_box = QGroupBox("GeoFEAS材料ライブラリ")
    library_form = QFormLayout(library_box)
    self.material_library_name = QLineEdit("soil_elastic")
    self.material_library_model = QComboBox()
    self.material_library_model.setMinimumWidth(280)
    self.material_library_model.setMinimumContentsLength(28)
    if QSizePolicy is not None:
        self.material_library_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    try:
        self.material_library_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.material_library_model.view().setMinimumWidth(360)
    except Exception:
        pass
    for label, key in [
        ("線形弾性", "elastic_linear"),
        ("直交異方性弾性", "elastic_orthotropic"),
        ("粘性土非排水弾性", "elastic_undrained"),
        ("K0/自重初期応力弾性", "elastic_k0"),
        ("非線形弾性 Hardin-Drnevich", "nonlinear_elastic_hardin_drnevich"),
        ("非線形弾性 Duncan-Chang", "nonlinear_elastic_duncan_chang"),
        ("非線形弾性 Ramberg-Osgood", "nonlinear_elastic_ramberg_osgood"),
        ("UW-Clay", "uw_clay"),
        ("Pastor-Zienkiewicz sand", "pastor_zienkiewicz_sand"),
        ("Pastor-Zienkiewicz clay", "pastor_zienkiewicz_clay"),
        ("弾完全塑性 Mohr-Coulomb", "perfect_plastic_mohr_coulomb"),
        ("弾完全塑性 Drucker-Prager", "perfect_plastic_drucker_prager"),
        ("弾完全塑性 Von-Mises/J2", "perfect_plastic_von_mises"),
        ("弾塑性 MC 硬化", "elastoplastic_mohr_coulomb_hardening"),
        ("弾塑性 DP 硬化", "elastoplastic_drucker_prager_hardening"),
        ("No-Tension", "no_tension"),
        ("バイリニア液状化", "bilinear_liquefaction"),
    ]:
        self.material_library_model.addItem(label, key)
    self.material_library_model.currentIndexChanged.connect(self.material_library_model_changed)
    self.material_library_E = QLineEdit("50000.0")
    self.material_library_nu = QLineEdit("0.3")
    self.material_library_gamma = QLineEdit("18.0")
    self.material_library_cohesion = QLineEdit("0.0")
    self.material_library_phi = QLineEdit("0.0")
    self.material_library_psi = QLineEdit("0.0")
    self.material_library_yield = QLineEdit("0.0")
    self.material_library_hardening = QLineEdit("0.0")
    self.material_library_k0 = QLineEdit("")
    self.material_library_ft = QLineEdit("")
    self.material_library_g0 = QLineEdit("")
    self.material_library_gamma_ref = QLineEdit("")
    self.material_library_liq_crr = QLineEdit("")
    self.material_library_extra = QPlainTextEdit()
    self.material_library_extra.setMinimumHeight(56)
    self.material_library_extra.setMaximumHeight(105)
    self.material_test_curve = QPlainTextEdit()
    self.material_test_curve.setPlaceholderText("gamma,G または gamma,tau をCSVで入力")
    self.material_test_curve.setMinimumHeight(56)
    self.material_test_curve.setMaximumHeight(80)
    self.material_river_n = QLineEdit("")
    self.material_river_fc = QLineEdit("")
    self.material_river_sigma_v = QLineEdit("")
    self.material_river_sigma_v_eff = QLineEdit("")
    self.material_river_csr = QLineEdit("")
    for label, widget in [
        ("name", self.material_library_name),
        ("model", self.material_library_model),
        ("E", self.material_library_E),
        ("nu", self.material_library_nu),
        ("gamma", self.material_library_gamma),
        ("cohesion", self.material_library_cohesion),
        ("phi", self.material_library_phi),
        ("psi", self.material_library_psi),
        ("yield", self.material_library_yield),
        ("hardening", self.material_library_hardening),
        ("k0", self.material_library_k0),
        ("No-Tension ft", self.material_library_ft),
        ("G0 / 初期せん断剛性", self.material_library_g0),
        ("gamma_ref", self.material_library_gamma_ref),
        ("液状化 CRR/RL20", self.material_library_liq_crr),
        ("詳細YAML", self.material_library_extra),
        ("材料試験カーブ", self.material_test_curve),
        ("河川耐震 N値", self.material_river_n),
        ("河川耐震 Fc(%)", self.material_river_fc),
        ("σv", self.material_river_sigma_v),
        ("σv'", self.material_river_sigma_v_eff),
        ("CSR", self.material_river_csr),
    ]:
        library_form.addRow(label, widget)
    estimate_curve_btn = QPushButton("試験カーブから定数推定")
    estimate_curve_btn.clicked.connect(self.estimate_material_constants_from_curve)
    library_form.addRow(estimate_curve_btn)
    river_btn = QPushButton("河川耐震指針パラメータ反映")
    river_btn.clicked.connect(self.apply_river_seismic_parameters)
    library_form.addRow(river_btn)
    add_library_btn = QPushButton("専用フォームから材料追加")
    add_library_btn.clicked.connect(self.add_material_from_library)
    library_form.addRow(add_library_btn)

    self.material_table = QTableWidget(0, 14)
    self.material_table.setHorizontalHeaderLabels([
        "材料名",
        "モデル",
        "E(ヤング率)",
        "ν(ポアソン比)",
        "γ(単位体積重量)",
        "粘着力 c",
        "内部摩擦角 φ",
        "ダイレイタンシー角 ψ",
        "降伏応力",
        "硬化係数",
        "K0",
        "引張カット",
        "引張強度 ft",
        "追加YAML",
    ])
    material_header = self.material_table.horizontalHeader()
    material_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    material_header.setStretchLastSection(False)
    for column, width in enumerate((150, 260, 110, 110, 125, 110, 125, 145, 110, 110, 90, 110, 110, 220)):
        self.material_table.setColumnWidth(column, width)
    self.material_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    self.material_table.setWordWrap(False)
    self.material_table.setAlternatingRowColors(True)
    self.material_table.setMinimumHeight(220)
    self.material_table.setProperty("compactColumns", "0,1,2,3,4,5,6")
    for column in range(7, self.material_table.columnCount()):
        self.material_table.setColumnHidden(column, True)
    material_list_layout.addWidget(self.material_table, 1)

    library_page = QWidget()
    library_page_layout = QVBoxLayout(library_page)
    library_page_layout.setContentsMargins(0, 0, 0, 0)
    library_scroll = QScrollArea()
    library_scroll.setWidgetResizable(True)
    library_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    library_scroll.setWidget(library_box)
    library_page_layout.addWidget(library_scroll)
    self.material_library_scroll = library_scroll
    self.material_tabs.addTab(material_list_page, "材料一覧")
    self.material_tabs.addTab(library_page, "材料ライブラリ")
    layout.addWidget(self.material_tabs, 1)
    add_btn = QPushButton("追加")
    add_btn.clicked.connect(self.add_material_row)
    elastic_btn = QPushButton("弾性")
    elastic_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("elastic"))
    mc_btn = QPushButton("MC")
    mc_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("mohr_coulomb"))
    dp_btn = QPushButton("DP")
    dp_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("drucker_prager"))
    j2_btn = QPushButton("J2")
    j2_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("von_mises"))
    no_tension_btn = QPushButton("No-Tension")
    no_tension_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("no_tension"))
    nonlinear_btn = QPushButton("非線形弾性")
    nonlinear_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("nonlinear_elastic_hardin_drnevich"))
    liquefaction_btn = QPushButton("液状化")
    liquefaction_btn.clicked.connect(lambda _checked=False: self.add_material_preset_row("bilinear_liquefaction"))
    remove_btn = QPushButton("削除")
    remove_btn.clicked.connect(self.remove_selected_material_rows)
    apply_btn = QPushButton("材料を反映")
    apply_btn.clicked.connect(self.apply_materials_panel)
    advanced_columns_btn = QPushButton("全項目")
    advanced_columns_btn.setCheckable(True)
    advanced_columns_btn.setToolTip("強度・硬化・引張・追加YAML列の表示を切り替えます。")

    def set_advanced_columns_visible(visible: bool) -> None:
        for column in range(7, self.material_table.columnCount()):
            self.material_table.setColumnHidden(column, not visible)
        if str(getattr(self, "gui_locale", "ja")) == "en":
            advanced_columns_btn.setText("Basic Fields" if visible else "All Fields")
        else:
            advanced_columns_btn.setText("基本項目" if visible else "全項目")

    advanced_columns_btn.toggled.connect(set_advanced_columns_visible)
    self.material_advanced_columns_button = advanced_columns_btn
    self.set_material_advanced_columns_visible = set_advanced_columns_visible
    material_header_row = QHBoxLayout()
    material_header_row.addWidget(material_list_status)
    material_header_row.addStretch(1)
    material_header_row.addWidget(advanced_columns_btn)
    material_list_layout.insertLayout(0, material_header_row)
    undo_btn, redo_btn = _edit_undo_redo_buttons(self, QPushButton)
    self.material_action_buttons = [
        add_btn,
        remove_btn,
        mc_btn,
        dp_btn,
        j2_btn,
        no_tension_btn,
        elastic_btn,
        nonlinear_btn,
        liquefaction_btn,
        apply_btn,
        undo_btn,
        redo_btn,
    ]
    self.material_library_model_changed()
    return page

def build_boundary_conditions_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QPushButton = qt["QPushButton"]
    QTableWidget = qt["QTableWidget"]
    QTreeWidget = qt["QTreeWidget"]
    Qt = qt["Qt"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel("境界条件を表で編集します。複雑なMPC等は下のYAML詳細で保持・編集できます。"))
    self.boundary_table = QTableWidget(0, 4)
    self.boundary_table.setHorizontalHeaderLabels(["対象(節点/セット)", "ux", "uy", "追加YAML"])
    self.boundary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(self.boundary_table)
    boundary_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("fixed", "固定", lambda _checked=False: self.add_boundary_row(target="left", ux="0.0", uy="0.0")),
        ("roller_y", "水平ローラ", lambda _checked=False: self.add_boundary_row(target="bottom", ux="", uy="0.0")),
        ("roller_x", "鉛直ローラ", lambda _checked=False: self.add_boundary_row(target="left", ux="0.0", uy="")),
        ("prescribed", "強制変位", lambda _checked=False: self.add_boundary_row(target="right", ux="0.0", uy="")),
        ("delete", "削除", self.remove_selected_boundary_rows),
        ("apply", "表を反映", self.apply_boundary_conditions_panel),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        boundary_buttons[key] = button
    boundary_undo_btn, boundary_redo_btn = _edit_undo_redo_buttons(self, QPushButton)

    selection_box = QGroupBox("画面選択から一括境界を作成")
    selection_layout = QVBoxLayout(selection_box)
    self.boundary_selection_summary = QLabel("モデルビューで節点・辺・要素を複数選択してから操作します。")
    self.boundary_selection_summary.setWordWrap(True)
    selection_layout.addWidget(self.boundary_selection_summary)
    selection_form = QFormLayout()
    self.boundary_batch_scope = QComboBox()
    self.boundary_batch_scope.addItem("全体BC", "global")
    self.boundary_batch_scope.addItem("選択ステージ", "stage")
    self.boundary_batch_kind = QComboBox()
    for label, data in [
        ("固定(ux=uy=0)", "fixed"),
        ("水平ローラ(uy=0)", "roller_y"),
        ("鉛直ローラ(ux=0)", "roller_x"),
        ("ピン(ux=uy=0)", "pin"),
        ("強制変位", "prescribed"),
    ]:
        self.boundary_batch_kind.addItem(label, data)
    self.boundary_batch_ux = QLineEdit()
    self.boundary_batch_ux.setPlaceholderText("強制変位ux。空欄なら未指定")
    self.boundary_batch_uy = QLineEdit()
    self.boundary_batch_uy.setPlaceholderText("強制変位uy。空欄なら未指定")
    self.boundary_hydro_kind = QComboBox()
    for label, data in [("辺水圧", "pressure"), ("節点水圧", "node_pressure"), ("流量", "flux"), ("Robin", "robin")]:
        self.boundary_hydro_kind.addItem(label, data)
    if hasattr(self, "_refresh_operation_button_enabled_states"):
        self.boundary_hydro_kind.currentIndexChanged.connect(lambda _index: self._refresh_operation_button_enabled_states())
    self.boundary_hydro_value = QLineEdit("0.0")
    self.boundary_mpc_master = QLineEdit()
    self.boundary_mpc_master.setPlaceholderText("空欄なら選択節点の先頭")
    self.boundary_mpc_dof = QComboBox()
    populate_labeled_combo(self.boundary_mpc_dof, "mpc_dof", ["ux", "uy"], locale=getattr(self, "gui_locale", "ja"))
    self.boundary_mpc_method = QComboBox()
    populate_labeled_combo(
        self.boundary_mpc_method,
        "mpc_method",
        ["lagrange", "elimination"],
        locale=getattr(self, "gui_locale", "ja"),
    )
    self.boundary_mpc_coefficient = QLineEdit("1.0")
    self.boundary_mpc_value = QLineEdit("0.0")
    selection_form.addRow("反映先", self.boundary_batch_scope)
    selection_form.addRow("支点/変位", self.boundary_batch_kind)
    selection_form.addRow("ux", self.boundary_batch_ux)
    selection_form.addRow("uy", self.boundary_batch_uy)
    selection_form.addRow("水理条件", self.boundary_hydro_kind)
    selection_form.addRow("水理値", self.boundary_hydro_value)
    selection_form.addRow("MPC master", self.boundary_mpc_master)
    selection_form.addRow("MPC dof", self.boundary_mpc_dof)
    selection_form.addRow("MPC method", self.boundary_mpc_method)
    selection_form.addRow("MPC coefficient", self.boundary_mpc_coefficient)
    selection_form.addRow("MPC value", self.boundary_mpc_value)
    selection_layout.addLayout(selection_form)
    boundary_selection_buttons: dict[str, Any] = {}
    for key, label, callback, rule in [
        ("selected_bc", "選択節点→支点/変位", self.add_selected_boundary_condition, "boundary_nodes_edges"),
        ("selected_hydro", "選択辺/要素→水理境界", self.add_selected_hydro_boundary_condition, "boundary_hydro"),
        ("selected_mpc", "選択節点→MPC", self.add_selected_mpc_constraints, "boundary_mpc"),
        ("selected_set", "選択節点→set登録", self.register_selected_nodes_as_set, "boundary_nodes_edges"),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        _set_operation_rule(button, rule)
        boundary_selection_buttons[key] = button
    self.boundary_action_button_rows = [
        [boundary_buttons["fixed"], boundary_buttons["prescribed"]],
        [boundary_buttons["roller_y"], boundary_buttons["roller_x"]],
        [boundary_buttons["delete"], boundary_buttons["apply"]],
        [boundary_selection_buttons["selected_bc"], boundary_selection_buttons["selected_hydro"]],
        [boundary_selection_buttons["selected_mpc"], boundary_selection_buttons["selected_set"]],
        [boundary_undo_btn, boundary_redo_btn],
    ]
    layout.addWidget(selection_box)

    tree_box = QGroupBox("境界条件ツリー")
    self.boundary_condition_tree_box = tree_box
    tree_layout = QVBoxLayout(tree_box)
    self.boundary_condition_tree = QTreeWidget()
    self.boundary_condition_tree.setColumnCount(2)
    self.boundary_condition_tree.setHeaderLabels(["境界条件", "対象節点"])
    self.boundary_condition_tree.setMinimumHeight(160)
    self.boundary_condition_tree.itemSelectionChanged.connect(self.select_boundary_condition_tree_item)
    self.boundary_condition_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    self.boundary_condition_tree.customContextMenuRequested.connect(self.show_boundary_condition_tree_context_menu)
    tree_layout.addWidget(self.boundary_condition_tree)
    tree_box.setVisible(False)

    axisym_box = QGroupBox("Axisymmetric boundary / hydraulic presets")
    self.boundary_axisym_box = axisym_box
    axisym_layout = QVBoxLayout(axisym_box)
    axisym_form = QFormLayout()
    self.axisym_boundary_preset = QComboBox()
    for label, data in [
        ("r=0 symmetry axis: ux=0", "symmetry_axis"),
        ("bottom axial roller: uz=0", "bottom_axial_roller"),
        ("outer radial roller: ur=0", "outer_radial_roller"),
        ("minimal axisym support", "minimal_support"),
    ]:
        self.axisym_boundary_preset.addItem(label, data)
    self.axisym_hydro_preset = QComboBox()
    for label, data in [
        ("top drained pressure p=0", "top_drained"),
        ("outer radius fixed pore pressure", "outer_pressure"),
        ("axis no-flow flux=0", "axis_no_flux"),
        ("outer radius Robin drain", "outer_robin"),
    ]:
        self.axisym_hydro_preset.addItem(label, data)
    self.axisym_hydro_value = QLineEdit("0.0")
    self.axisym_hydro_beta = QLineEdit("1.0")
    axisym_form.addRow("BC preset", self.axisym_boundary_preset)
    axisym_form.addRow("Hydro preset", self.axisym_hydro_preset)
    axisym_form.addRow("Hydro pressure/flux", self.axisym_hydro_value)
    axisym_form.addRow("Robin beta", self.axisym_hydro_beta)
    axisym_layout.addLayout(axisym_form)
    axisym_controls = QHBoxLayout()
    axisym_bc_btn = QPushButton("Apply axisym BC")
    axisym_bc_btn.clicked.connect(self.apply_axisymmetric_boundary_preset)
    axisym_hydro_btn = QPushButton("Apply axisym hydro")
    axisym_hydro_btn.clicked.connect(self.apply_axisymmetric_hydro_preset)
    axisym_controls.addWidget(axisym_bc_btn)
    axisym_controls.addWidget(axisym_hydro_btn)
    axisym_controls.addStretch(1)
    axisym_layout.addLayout(axisym_controls)
    axisym_box.setVisible(False)
    self.boundary_conditions_editor = QPlainTextEdit()
    self.boundary_conditions_editor.setVisible(False)
    return page

def build_loads_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGroupBox = qt["QGroupBox"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QPushButton = qt["QPushButton"]
    QTableWidget = qt["QTableWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    QGridLayout = qt.get("QGridLayout")
    QSizePolicy = qt.get("QSizePolicy")
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel("荷重を節点/辺/自重の表で編集します。高度な荷重は下のYAML詳細で保持・編集できます。"))
    case_box = QGroupBox("荷重ケース")
    case_layout = QVBoxLayout(case_box)
    self.load_case_table = QTableWidget(0, 5)
    self.load_case_table.setHorizontalHeaderLabels(["ケース", "種別", "倍率", "有効", "説明"])
    self.load_case_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    case_layout.addWidget(self.load_case_table)
    load_case_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("static", "静的ケース追加", lambda _checked=False: self.add_load_case_row(name=f"LC{self.load_case_table.rowCount() + 1}", case_type="static", scale="1.0", active="true")),
        ("earthquake", "地震ケース追加", lambda _checked=False: self.add_load_case_row(name=f"EQ{self.load_case_table.rowCount() + 1}", case_type="earthquake", scale="1.0", active="true")),
        ("delete", "ケース削除", self.remove_selected_load_case_rows),
        ("apply", "ケース反映", self.apply_load_cases_panel),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        load_case_buttons[key] = button
    layout.addWidget(case_box)
    self.load_table = QTableWidget(0, 8)
    self.load_table.setHorizontalHeaderLabels(["種別", "対象(節点/セット/辺)", "fx", "fy", "tx", "ty", "倍率", "追加YAML"])
    self.load_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    layout.addWidget(self.load_table)
    load_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("node", "節点集中", lambda _checked=False: self.add_load_row(load_type="node", target="right", fx="0.0", fy="-10.0")),
        ("edge", "分布荷重", lambda _checked=False: self.add_load_row(load_type="edge", target="top", tx="0.0", ty="-10.0")),
        ("gravity", "自重", lambda _checked=False: self.add_load_row(load_type="gravity", scale="1.0")),
        ("delete", "荷重削除", self.remove_selected_load_rows),
        ("apply", "荷重表を反映", self.apply_loads_panel),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        load_buttons[key] = button
    load_undo_btn, load_redo_btn = _edit_undo_redo_buttons(self, QPushButton)

    selection_box = QGroupBox("画面選択から一括荷重を作成")
    selection_layout = QVBoxLayout(selection_box)
    self.load_selection_summary = QLabel("モデルビューで節点・辺・要素を複数選択してから操作します。")
    self.load_selection_summary.setWordWrap(True)
    selection_layout.addWidget(self.load_selection_summary)
    self.load_batch_scope = QComboBox()
    self.load_batch_scope.addItem("全体荷重", "global")
    self.load_batch_scope.addItem("選択ステージ", "stage")
    self.load_case_selector = QComboBox()
    self.load_case_selector.setEditable(True)
    self.load_body_material = QComboBox()
    self.load_body_material.setEditable(True)
    self.load_body_material.addItem("選択要素の材料", "__selected__")
    if hasattr(self, "_refresh_operation_button_enabled_states"):
        self.load_body_material.currentIndexChanged.connect(lambda _index: self._refresh_operation_button_enabled_states())
    self.load_body_bx = QLineEdit("0.0")
    self.load_body_by = QLineEdit("-18.0")
    self.load_surface_distribution = QComboBox()
    self.load_surface_distribution.addItem("等分布", "uniform")
    self.load_surface_distribution.addItem("偏分布", "linear")
    self.load_fx = QLineEdit("0.0")
    self.load_fy = QLineEdit("-10.0")
    self.load_tx = QLineEdit("0.0")
    self.load_ty = QLineEdit("-10.0")
    self.load_tx_end = QLineEdit("0.0")
    self.load_ty_end = QLineEdit("-10.0")
    self.load_seismic_kh = QLineEdit("0.10")
    self.load_seismic_kv = QLineEdit("0.0")
    self.load_seismic_direction = QComboBox()
    self.load_seismic_direction.addItems(["+X", "-X", "+Y", "-Y"])
    if QGridLayout is not None:
        selection_form = QGridLayout()
        selection_form.setContentsMargins(0, 0, 0, 0)
        selection_form.setHorizontalSpacing(8)
        selection_form.setVerticalSpacing(6)
        for value_column in (1, 3, 5, 7):
            selection_form.setColumnStretch(value_column, 1)

        def add_load_field(row: int, pair_column: int, label_text: str, widget: Any) -> None:
            label = QLabel(label_text)
            label.setWordWrap(True)
            widget.setProperty("loadBatchGridColumns", 8)
            widget.setProperty("loadBatchGridRow", row)
            widget.setProperty("loadBatchGridPairColumn", pair_column)
            widget.setProperty("loadBatchGridLabel", label_text)
            if QSizePolicy is not None:
                widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            selection_form.addWidget(label, row, pair_column * 2)
            selection_form.addWidget(widget, row, pair_column * 2 + 1)

        load_field_rows = [
            [
                ("反映先", self.load_batch_scope),
                ("荷重ケース", self.load_case_selector),
                ("体積力材料", self.load_body_material),
                ("面荷重分布", self.load_surface_distribution),
            ],
            [
                ("体積力 bx", self.load_body_bx),
                ("体積力 by", self.load_body_by),
                ("節点 fx", self.load_fx),
                ("節点 fy", self.load_fy),
            ],
            [
                ("分布 tx", self.load_tx),
                ("分布 ty", self.load_ty),
                ("終端 tx", self.load_tx_end),
                ("終端 ty", self.load_ty_end),
            ],
            [
                ("水平震度 kh", self.load_seismic_kh),
                ("鉛直震度 kv", self.load_seismic_kv),
                ("地震方向", self.load_seismic_direction),
            ],
        ]
        for row_index, fields in enumerate(load_field_rows):
            for pair_column, (label_text, widget) in enumerate(fields):
                add_load_field(row_index, pair_column, label_text, widget)
    else:
        selection_form = QFormLayout()
        for label_text, widget in [
            ("反映先", self.load_batch_scope),
            ("荷重ケース", self.load_case_selector),
            ("体積力材料", self.load_body_material),
            ("面荷重分布", self.load_surface_distribution),
            ("体積力 bx", self.load_body_bx),
            ("体積力 by", self.load_body_by),
            ("節点 fx", self.load_fx),
            ("節点 fy", self.load_fy),
            ("分布 tx", self.load_tx),
            ("分布 ty", self.load_ty),
            ("終端 tx", self.load_tx_end),
            ("終端 ty", self.load_ty_end),
            ("水平震度 kh", self.load_seismic_kh),
            ("鉛直震度 kv", self.load_seismic_kv),
            ("地震方向", self.load_seismic_direction),
        ]:
            selection_form.addRow(label_text, widget)
    selection_layout.addLayout(selection_form)
    load_selection_buttons: dict[str, Any] = {}
    load_selection_order: list[str] = []
    for key, label, callback in [
        ("selected_node", "選択節点→節点荷重", self.add_selected_nodal_load_condition),
        ("selected_body", "選択要素材料→体積力", self.add_selected_body_load_condition),
        ("selected_surface", "選択辺/要素→面荷重", self.add_selected_distributed_load_condition),
        ("selected_linear_surface", "選択辺/要素→偏分布面荷重", lambda _checked=False: self.add_selected_distributed_load_condition(distribution="linear")),
        ("earthquake", "疑似静的地震荷重", self.add_panel_earthquake_load),
        ("pore_csv", "浸透流CSV→水圧", self.import_pore_pressure_csv_to_selected_stage),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        load_selection_buttons[key] = button
        load_selection_order.append(key)
    for button, rule in zip(
        [load_selection_buttons[key] for key in load_selection_order],
        ["load_nodes", "load_body", "load_edges_or_elements", "load_edges_or_elements", "", ""],
        strict=False,
    ):
        if rule:
            _set_operation_rule(button, rule)
    self.load_action_button_rows = [
        [load_case_buttons["static"], load_case_buttons["earthquake"]],
        [load_case_buttons["delete"], load_buttons["delete"]],
        [load_case_buttons["apply"], load_buttons["apply"]],
        [load_buttons["node"], load_buttons["edge"]],
        [load_buttons["gravity"], load_selection_buttons["earthquake"]],
        [load_selection_buttons["selected_node"], load_selection_buttons["selected_body"]],
        [load_selection_buttons["selected_surface"], load_selection_buttons["selected_linear_surface"]],
        [load_selection_buttons["pore_csv"]],
        [load_undo_btn, load_redo_btn],
    ]
    layout.addWidget(selection_box)

    axisym_box = QGroupBox("Axisymmetric load presets")
    self.load_axisym_box = axisym_box
    axisym_layout = QVBoxLayout(axisym_box)
    axisym_form = QFormLayout()
    self.axisym_load_preset = QComboBox()
    for label, data in [
        ("outer radial pressure", "outer_radial_pressure"),
        ("top axial surcharge", "top_axial_surcharge"),
        ("axisymmetric self weight", "self_weight"),
        ("outer ring nodal force", "outer_ring_force"),
    ]:
        self.axisym_load_preset.addItem(label, data)
    self.axisym_load_value = QLineEdit("10.0")
    axisym_load_btn = QPushButton("Apply axisym load")
    axisym_load_btn.clicked.connect(self.apply_axisymmetric_load_preset)
    axisym_form.addRow("preset", self.axisym_load_preset)
    axisym_form.addRow("value", self.axisym_load_value)
    axisym_layout.addLayout(axisym_form)
    self._add_panel_button_rows(axisym_layout, [axisym_load_btn], columns=1)
    axisym_box.setVisible(False)
    self.loads_editor = QPlainTextEdit()
    self.loads_editor.setVisible(False)
    return page

def build_stages_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGridLayout = qt.get("QGridLayout")
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QMenu = qt["QMenu"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QPushButton = qt["QPushButton"]
    QScrollArea = qt["QScrollArea"]
    QTabWidget = qt["QTabWidget"]
    QTableWidget = qt["QTableWidget"]
    QTreeWidget = qt["QTreeWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    QSizePolicy = qt.get("QSizePolicy")
    page = QWidget()
    layout = QVBoxLayout(page)
    self.stage_form_workspace = QWidget()
    stage_form_workspace_layout = QVBoxLayout(self.stage_form_workspace)
    self.stage_workspace_tabs = QTabWidget()
    stage_form_workspace_layout.addWidget(self.stage_workspace_tabs)
    if hasattr(self, "form_workspace_stack"):
        self.form_workspace_stack.addWidget(self.stage_form_workspace)
        self.form_workspace_pages["stages"] = self.stage_form_workspace
    detail_scroll = QScrollArea()
    detail_scroll.setWidgetResizable(True)
    detail_tab = QWidget()
    detail_tab_layout = QVBoxLayout(detail_tab)
    detail_scroll.setWidget(detail_tab)
    diff_scroll = QScrollArea()
    diff_scroll.setWidgetResizable(True)
    diff_tab = QWidget()
    diff_tab_layout = QVBoxLayout(diff_tab)
    diff_scroll.setWidget(diff_tab)
    yaml_tab = QWidget()
    self.stage_internal_data_tab = yaml_tab
    yaml_layout = QVBoxLayout(yaml_tab)
    self.stage_workspace_tabs.addTab(detail_scroll, "詳細フォーム")
    self.stage_workspace_tabs.addTab(diff_scroll, "差分/承認")
    self.stage_workspace_tabs.addTab(yaml_tab, "内部データ")
    self.stage_table = QTableWidget(0, 10)
    self.stage_table.setHorizontalHeaderLabels(["No", "名称", "解析状態", "対象", "応力解放率", "境界", "荷重", "水理", "多点拘束", "個別設定"])
    self.stage_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.stage_table.itemSelectionChanged.connect(self.stage_selection_changed)
    self.stage_table.setMinimumHeight(240)
    self.stage_list_label = QLabel("ステージ一覧")
    layout.addWidget(self.stage_list_label)
    layout.addWidget(self.stage_table, 2)
    stage_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("add", "追加", self.add_stage),
        ("copy", "コピー", self.copy_selected_stage),
        ("up", "上へ", lambda _checked=False: self.move_selected_stage(-1)),
        ("down", "下へ", lambda _checked=False: self.move_selected_stage(1)),
        ("delete", "削除", self.delete_selected_stage),
        ("apply", "表を反映", self.apply_stage_table),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        button.setProperty("stageQuickButton", True)
        button.setProperty("stageSheetButton", True)
        button.setProperty("stageSheetButtonGroup", "stage_list")
        stage_buttons[key] = button
    self.stage_action_button_rows = [
        [stage_buttons["add"], stage_buttons["copy"]],
        [stage_buttons["up"], stage_buttons["down"]],
        [stage_buttons["delete"], stage_buttons["apply"]],
    ]
    self.stage_change_table = QTableWidget(0, 13)
    self.stage_change_table.setHorizontalHeaderLabels(["ステージ", "区分", "対象", "操作・条件", "材料", "X変位", "Y変位", "X荷重", "Y荷重", "X分布荷重", "Y分布荷重", "応力解放", "追加設定"])
    self.stage_change_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.stage_change_table.setMinimumHeight(260)
    selection_box = QGroupBox("画面選択からステージ条件を作成")
    selection_layout = QVBoxLayout(selection_box)
    self.stage_selection_summary = QLabel("モデルビューで節点・辺・要素を複数選択してから操作します。")
    self.stage_selection_summary.setWordWrap(True)
    selection_layout.addWidget(self.stage_selection_summary)
    selection_buttons = [
        ("施工setへ", self.apply_selected_elements_to_stage_set),
        ("選択節点→一括変位", self.add_selected_prescribed_displacement),
        ("選択辺→分布荷重", self.add_selected_edge_load),
        ("地震荷重", self.add_pseudo_static_earthquake_load),
        ("選択→水圧連携", self.add_selected_hydro_coupling),
        ("差分確認", self.show_stage_difference),
    ]
    stage_selection_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("set", "施工setへ", self.apply_selected_elements_to_stage_set),
        ("bc", "選択節点→一括変位", self.add_selected_prescribed_displacement),
        ("load", "選択辺→分布荷重", self.add_selected_edge_load),
        ("quake", "地震荷重", self.add_pseudo_static_earthquake_load),
        ("hydro", "選択→水圧連携", self.add_selected_hydro_coupling),
        ("diff", "差分確認", self.show_stage_difference),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        button.setProperty("stageQuickButton", True)
        button.setProperty("stageSheetButton", True)
        button.setProperty("stageSheetButtonGroup", "stage_selection")
        stage_selection_buttons[key] = button
    selection_box.setVisible(False)
    layout.addWidget(selection_box)
    stage_change_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("death", "無効化追加", lambda _checked=False: self.add_stage_change_row("death")),
        ("birth", "再有効化追加", lambda _checked=False: self.add_stage_change_row("birth")),
        ("material", "材料変更追加", lambda _checked=False: self.add_stage_change_row("material")),
        ("boundary", "境界変更追加", lambda _checked=False: self.add_stage_change_row("boundary")),
        ("load", "荷重変更追加", lambda _checked=False: self.add_stage_change_row("load")),
        ("delete", "変更行削除", lambda _checked=False: self.remove_selected_rows(self.stage_change_table)),
        ("apply", "変更表を反映", self.apply_stage_change_table),
    ]:
        button = QPushButton(label)
        button.clicked.connect(callback)
        button.setProperty("stageQuickButton", True)
        button.setProperty("stageSheetButton", True)
        button.setProperty("stageSheetButtonGroup", "stage_change")
        stage_change_buttons[key] = button

    self.stage_standard_action_bar = QWidget()
    standard_action_layout = QVBoxLayout(self.stage_standard_action_bar)
    standard_action_layout.setContentsMargins(0, 0, 0, 0)
    standard_action_layout.setSpacing(5)
    self.stage_standard_change_summary = QLabel("変更: 0件")
    self.stage_standard_change_summary.setWordWrap(True)
    standard_action_layout.addWidget(self.stage_standard_change_summary)
    self.stage_standard_add_stage_button = QPushButton("ステージ追加")
    self.stage_standard_add_stage_button.clicked.connect(self.add_stage)
    self.stage_standard_manage_button = QPushButton("ステージ整理")
    manage_stage_menu = QMenu(self.stage_standard_manage_button)
    for label_ja, label_en, callback in (
        ("選択ステージを複製", "Duplicate Selected Stage", self.copy_selected_stage),
        ("選択ステージを上へ", "Move Selected Stage Up", lambda _checked=False: self.move_selected_stage(-1)),
        ("選択ステージを下へ", "Move Selected Stage Down", lambda _checked=False: self.move_selected_stage(1)),
        ("選択ステージを削除", "Delete Selected Stage", self.delete_selected_stage),
        ("ステージ一覧を反映", "Apply Stage List", self.apply_stage_table),
    ):
        action = manage_stage_menu.addAction(label_ja)
        action.setProperty("menuTextJa", label_ja)
        action.setProperty("menuTextEn", label_en)
        action.triggered.connect(callback)
    self.stage_standard_manage_button.setMenu(manage_stage_menu)
    self.stage_standard_add_change_button = QPushButton("変更を追加")
    add_change_menu = QMenu(self.stage_standard_add_change_button)
    for label_ja, label_en, change_kind in (
        ("無効化を追加", "Add Deactivation", "death"),
        ("再有効化を追加", "Add Reactivation", "birth"),
        ("材料変更を追加", "Add Material Change", "material"),
        ("境界変更を追加", "Add Boundary Change", "boundary"),
        ("荷重変更を追加", "Add Load Change", "load"),
    ):
        action = add_change_menu.addAction(label_ja)
        action.setProperty("menuTextJa", label_ja)
        action.setProperty("menuTextEn", label_en)
        action.triggered.connect(lambda _checked=False, kind=change_kind: self.add_stage_change_row(kind))
    self.stage_standard_add_change_button.setMenu(add_change_menu)
    self.stage_standard_remove_change_button = QPushButton("選択削除")
    self.stage_standard_remove_change_button.clicked.connect(
        lambda _checked=False: (self.remove_selected_rows(self.stage_change_table), self._refresh_stage_standard_change_summary())
    )
    self.stage_standard_apply_changes_button = QPushButton("反映")
    self.stage_standard_apply_changes_button.clicked.connect(self.apply_stage_change_table)
    self.stage_standard_open_detail_button = QPushButton("詳細編集")
    self.stage_standard_open_detail_button.clicked.connect(self.open_stage_detail_editor)
    for button, icon_role in (
        (self.stage_standard_add_stage_button, "stage.add"),
        (self.stage_standard_manage_button, "stage.manage"),
        (self.stage_standard_add_change_button, "stage.change.add"),
        (self.stage_standard_remove_change_button, "stage.change.delete"),
        (self.stage_standard_apply_changes_button, "stage.change.apply"),
        (self.stage_standard_open_detail_button, "panel.detail"),
    ):
        self._apply_button_icon(button, icon_role)
        button.setProperty("stageStandardButton", True)
    self._add_uniform_panel_button_grid(
        standard_action_layout,
        [
            self.stage_standard_add_stage_button,
            self.stage_standard_manage_button,
            self.stage_standard_add_change_button,
            self.stage_standard_open_detail_button,
            self.stage_standard_remove_change_button,
            self.stage_standard_apply_changes_button,
        ],
        columns=2,
        min_width=120,
    )
    layout.addWidget(self.stage_standard_action_bar)

    self.stage_change_actions_widget = QWidget()
    stage_change_actions_layout = QVBoxLayout(self.stage_change_actions_widget)
    stage_change_actions_layout.setContentsMargins(0, 0, 0, 0)
    self.stage_change_section_label = QLabel("ステージごとの変更入力（材料・境界・荷重・無効化/再有効化）")
    stage_change_actions_layout.addWidget(self.stage_change_section_label)
    self._add_uniform_panel_button_pair_grid(
        stage_change_actions_layout,
        [
            [stage_change_buttons["death"], stage_change_buttons["birth"], stage_change_buttons["material"], stage_change_buttons["boundary"]],
            [stage_change_buttons["load"], stage_change_buttons["delete"], stage_change_buttons["apply"]],
        ],
        columns=4,
        min_width=120,
    )
    layout.addWidget(self.stage_change_actions_widget)
    layout.addWidget(self.stage_change_table, 3)

    detail_box = QGroupBox("選択ステージ詳細フォーム")
    self.stage_detail_box = detail_box
    detail_layout = QVBoxLayout(detail_box)
    recommendation_box = QGroupBox("ステージ推奨値")
    recommendation_layout = QHBoxLayout(recommendation_box)
    self.stage_recommendation_label = QLabel("解析状態を選ぶと推奨値を表示します。")
    self.stage_recommendation_label.setWordWrap(True)
    self.stage_recommendation_label.setProperty("informationRole", "detail")
    if QSizePolicy is not None:
        self.stage_recommendation_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    recommendation_layout.addWidget(self.stage_recommendation_label, 1)
    detail_layout.addWidget(recommendation_box)
    detail_form = QFormLayout()
    self.stage_detail_name = QLineEdit()
    self.stage_detail_type = QComboBox()
    populate_labeled_combo(
        self.stage_detail_type,
        "stage_type",
        ["static", "large_deformation", "geostatic", "k0", "excavation", "death", "deactivate", "srm", "safety_factor", "consolidation", "u-p", "riks", "arc_length"],
        locale=getattr(self, "gui_locale", "ja"),
    )
    self.stage_detail_type.currentTextChanged.connect(
        lambda _text: (self.refresh_stage_recommendation_label(), self._refresh_stage_inspector_visibility())
    )
    self.stage_detail_target = QLineEdit()
    self.stage_detail_stress_release = QLineEdit()
    self.stage_detail_apply_gravity = QComboBox()
    populate_labeled_combo(self.stage_detail_apply_gravity, "boolean", ["true", "false"], locale=getattr(self, "gui_locale", "ja"))
    self.stage_detail_k0 = QLineEdit()
    self.stage_detail_surface_y = QLineEdit()
    self.stage_detail_gx = QLineEdit()
    self.stage_detail_gy = QLineEdit()
    self.stage_detail_scale = QLineEdit()
    self.stage_detail_dt = QLineEdit()
    self.stage_detail_steps = QLineEdit()
    self.stage_detail_storage = QLineEdit()
    self.stage_detail_permeability = QLineEdit()
    self.stage_detail_biot_alpha = QLineEdit()
    self.stage_detail_srm_start = QLineEdit()
    self.stage_detail_srm_end = QLineEdit()
    self.stage_detail_srm_step = QLineEdit()
    self.stage_detail_srm_failure_ratio = QLineEdit()
    self.stage_detail_riks_arc = QLineEdit()
    self.stage_detail_riks_steps = QLineEdit()
    self.stage_detail_increments = QLineEdit()
    self.stage_detail_solver = QPlainTextEdit()
    self.stage_detail_solver.setMinimumHeight(56)
    self.stage_detail_solver.setMaximumHeight(88)
    for label, widget in [
        ("名称", self.stage_detail_name),
        ("解析状態", self.stage_detail_type),
        ("対象要素・セット", self.stage_detail_target),
        ("応力解放率", self.stage_detail_stress_release),
        ("自重/K0を使う", self.stage_detail_apply_gravity),
        ("K0", self.stage_detail_k0),
        ("地表面Y座標", self.stage_detail_surface_y),
        ("X方向重力", self.stage_detail_gx),
        ("Y方向重力", self.stage_detail_gy),
        ("荷重倍率", self.stage_detail_scale),
        ("圧密時間刻み", self.stage_detail_dt),
        ("圧密ステップ数", self.stage_detail_steps),
        ("貯留係数", self.stage_detail_storage),
        ("透水係数", self.stage_detail_permeability),
        ("Biot係数", self.stage_detail_biot_alpha),
        ("SRM開始係数", self.stage_detail_srm_start),
        ("SRM終了係数", self.stage_detail_srm_end),
        ("SRM探索刻み", self.stage_detail_srm_step),
        ("SRM破壊判定率", self.stage_detail_srm_failure_ratio),
        ("Riks弧長", self.stage_detail_riks_arc),
        ("Riks最大ステップ", self.stage_detail_riks_steps),
        ("増分数", self.stage_detail_increments),
        ("解析詳細設定", self.stage_detail_solver),
    ]:
        detail_form.addRow(label, widget)
    detail_layout.addLayout(detail_form)
    self.stage_construction_table = QTableWidget(0, 6)
    self.stage_construction_table.setHorizontalHeaderLabels(["操作", "対象", "応力解放率", "再有効化", "材料", "追加設定"])
    self.stage_construction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.stage_construction_table.setMinimumHeight(140)
    detail_layout.addWidget(QLabel("施工/death/材料変更イベント"))
    detail_layout.addWidget(self.stage_construction_table)
    construction_buttons: dict[str, Any] = {}
    for key, label, callback in [
        ("death", "掘削/death追加", lambda _checked=False: self.add_stage_construction_row(action="excavation", target=self.stage_detail_target.text().strip() or "all", stress_release=self.stage_detail_stress_release.text().strip() or "1.0")),
        ("birth", "再有効化記録", lambda _checked=False: self.add_stage_construction_row(action="reactivate", target=self.stage_detail_target.text().strip() or "all", reactivate=True)),
        ("material", "材料変更追加", lambda _checked=False: self.add_stage_construction_row(action="material", target=self.stage_detail_target.text().strip() or "all", material=self.mesh_material.text().strip() or "soil")),
        ("delete", "選択行削除", lambda _checked=False: self.remove_selected_rows(self.stage_construction_table)),
        ("recommend", "推奨値を反映", lambda _checked=False: self.apply_stage_recommended_defaults(force=True)),
        ("k0", "K0プリセット", lambda _checked=False: self.apply_stage_recommended_defaults(force=True, stage_type="geostatic")),
        ("consolidation", "圧密プリセット", lambda _checked=False: self.apply_stage_recommended_defaults(force=True, stage_type="consolidation")),
        ("riks", "Riksプリセット", lambda _checked=False: self.apply_stage_recommended_defaults(force=True, stage_type="riks")),
        ("template_death", "掘削テンプレート", lambda _checked=False: self.apply_stage_construction_template("excavation")),
        ("template_boundary", "境界切替テンプレート", lambda _checked=False: self.apply_stage_construction_template("boundary_change")),
        ("template_hydro", "水理切替テンプレート", lambda _checked=False: self.apply_stage_construction_template("hydro_change")),
        ("detail_apply", "詳細を反映", self.apply_stage_detail_tables),
        ("detail_tab", "詳細入力", lambda _checked=False: self.show_stage_workspace_tab(0)),
        ("diff_tab", "差分/承認", lambda _checked=False: self.show_stage_workspace_tab(1)),
    ]:
        button = QPushButton(label)
        button.setProperty("stageQuickButton", True)
        button.clicked.connect(callback)
        construction_buttons[key] = button
    detail_tab_layout.addWidget(detail_box)

    self.stage_detail_tabs = QTabWidget()
    self.stage_detail_action_button_groups: dict[str, list[Any]] = {}
    self.stage_material_table = QTableWidget(0, 3)
    self.stage_material_table.setHorizontalHeaderLabels(["target element/set", "material", "extra YAML"])
    self.stage_boundary_table = QTableWidget(0, 4)
    self.stage_boundary_table.setHorizontalHeaderLabels(["target node/set", "ux", "uy", "extra YAML"])
    self.stage_load_table = QTableWidget(0, 8)
    self.stage_load_table.setHorizontalHeaderLabels(["type", "target", "fx", "fy", "tx", "ty", "scale", "extra YAML"])
    self.stage_hydro_table = QTableWidget(0, 4)
    self.stage_hydro_table.setHorizontalHeaderLabels(["kind", "target", "value", "extra YAML"])
    self.stage_mpc_table = QTableWidget(0, 6)
    self.stage_mpc_table.setHorizontalHeaderLabels(["master", "slave", "dof", "coefficient", "value", "method"])
    for table in (self.stage_material_table, self.stage_boundary_table, self.stage_load_table, self.stage_hydro_table, self.stage_mpc_table):
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def detail_page(table: QTableWidget, buttons: list[tuple[str, Any]], action_group: str, *, expose_actions: bool = True) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(table)
        if expose_actions:
            row_buttons = []
            for label, callback in buttons:
                button = QPushButton(label)
                button.setProperty("stageDetailSheetButton", True)
                button.setProperty("stageDetailSheetActionGroup", action_group)
                button.setProperty("stageDetailSheetButtonCount", len(buttons))
                button.clicked.connect(callback)
                row_buttons.append(button)
            self.stage_detail_action_button_groups[action_group] = row_buttons
        return tab

    self.stage_detail_tabs.addTab(
        detail_page(
            self.stage_material_table,
            [
                ("材料変更追加", lambda _checked=False: self.add_stage_material_row(target="all", material=self.mesh_material.text().strip() or "soil")),
                ("選択行削除", lambda _checked=False: self.remove_selected_rows(self.stage_material_table)),
            ],
            "material",
            expose_actions=False,
        ),
        "材料変更",
    )
    self.stage_detail_tabs.addTab(
        detail_page(
            self.stage_boundary_table,
            [
                ("固定", lambda _checked=False: self.add_stage_boundary_row(target="left", ux="0.0", uy="0.0")),
                ("ローラ", lambda _checked=False: self.add_stage_boundary_row(target="bottom", ux="", uy="0.0")),
                ("強制変位", lambda _checked=False: self.add_stage_boundary_row(target="right", ux="0.0", uy="")),
                ("選択節点→支点/変位", lambda _checked=False: self.add_selected_boundary_condition(scope="stage")),
                ("選択行削除", lambda _checked=False: self.remove_selected_rows(self.stage_boundary_table)),
            ],
            "boundary",
        ),
        "境界",
    )
    self.stage_detail_tabs.addTab(
        detail_page(
            self.stage_load_table,
            [
                ("節点荷重", lambda _checked=False: self.add_stage_load_row(load_type="node", target="right", fx="0.0", fy="-10.0")),
                ("分布荷重", lambda _checked=False: self.add_stage_load_row(load_type="edge", target="top", tx="0.0", ty="-10.0")),
                ("自重", lambda _checked=False: self.add_stage_load_row(load_type="gravity", scale="1.0")),
                ("選択行削除", lambda _checked=False: self.remove_selected_rows(self.stage_load_table)),
            ],
            "load",
        ),
        "荷重",
    )
    self.stage_detail_tabs.addTab(
        detail_page(
            self.stage_hydro_table,
            [
                ("水圧", lambda _checked=False: self.add_stage_hydro_row(kind="pressure", target="top", value="0.0")),
                ("流量", lambda _checked=False: self.add_stage_hydro_row(kind="flux", target="top", value="0.0")),
                ("Robin", lambda _checked=False: self.add_stage_hydro_row(kind="robin", target="top", value="1.0")),
                ("選択辺→水理", self.add_selected_hydro_boundary_condition),
                ("選択行削除", lambda _checked=False: self.remove_selected_rows(self.stage_hydro_table)),
            ],
            "hydro",
        ),
        "水理",
    )
    self.stage_detail_tabs.addTab(
        detail_page(
            self.stage_mpc_table,
            [
                ("MPC追加", lambda _checked=False: self.add_stage_mpc_row(master="1", slave="2", dof="ux", coefficient="1.0", value="0.0", method="elimination")),
                ("選択節点→MPC", self.add_selected_mpc_constraints),
                ("選択行削除", lambda _checked=False: self.remove_selected_rows(self.stage_mpc_table)),
            ],
            "mpc",
        ),
        "MPC",
    )
    self.stage_detail_tabs.currentChanged.connect(lambda _index: self._refresh_stage_detail_sheet_action_buttons())
    self.stage_detail_tabs.setMinimumHeight(260)
    detail_tab_layout.addWidget(self.stage_detail_tabs)
    diff_box = QGroupBox("前ステージとの差分")
    diff_layout = QVBoxLayout(diff_box)
    self.show_stage_diff_overlay = QCheckBox("モデルビューで差分を色分け表示")
    self.show_stage_diff_overlay.setChecked(True)
    self.show_stage_diff_overlay.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="stage diff overlay"))
    diff_layout.addWidget(self.show_stage_diff_overlay)
    self.stage_diff_table = QTableWidget(0, 4)
    self.stage_diff_table.setColumnCount(5)
    self.stage_diff_table.setHorizontalHeaderLabels(["区分", "対象", "前ステージ", "選択ステージ", "承認"])
    self.stage_diff_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.stage_diff_table.setMinimumHeight(180)
    self.stage_diff_table.cellDoubleClicked.connect(self.repair_stage_difference_cell)
    diff_layout.addWidget(self.stage_diff_table)
    refresh_diff_btn = QPushButton("差分表を更新")
    refresh_diff_btn.clicked.connect(self.refresh_stage_difference_table)
    repair_diff_btn = QPushButton("選択差分を前ステージへ戻す")
    repair_diff_btn.clicked.connect(self.repair_selected_stage_difference)
    approve_diff_btn = QPushButton("選択差分を承認")
    approve_diff_btn.clicked.connect(self.approve_selected_stage_difference)
    reject_diff_btn = QPushButton("差戻し")
    reject_diff_btn.clicked.connect(self.reject_selected_stage_difference)
    reapprove_diff_btn = QPushButton("再承認")
    reapprove_diff_btn.clicked.connect(self.reapprove_selected_stage_difference)
    history_diff_btn = QPushButton("承認履歴")
    history_diff_btn.clicked.connect(lambda _checked=False: self.refresh_stage_approval_history_table())
    compare_stage_btn = QPushButton("全ステージ横断比較")
    compare_stage_btn.clicked.connect(self.refresh_stage_cross_compare_table)
    conflict_btn = QPushButton("累積矛盾診断")
    conflict_btn.clicked.connect(self.refresh_stage_conflict_table)
    auto_repair_btn = QPushButton("自動修復提案を適用")
    auto_repair_btn.clicked.connect(self.apply_stage_conflict_repair_suggestions)
    repair_conflict_btn = QPushButton("選択矛盾を修復")
    repair_conflict_btn.clicked.connect(self.repair_selected_stage_conflict)
    self._add_panel_button_rows(
        diff_layout,
        [refresh_diff_btn, repair_diff_btn, approve_diff_btn, reject_diff_btn, reapprove_diff_btn, history_diff_btn, compare_stage_btn, conflict_btn, auto_repair_btn, repair_conflict_btn],
        columns=2,
    )
    approval_form = QFormLayout()
    self.stage_approval_user = QLineEdit(getpass.getuser())
    self.stage_approval_note = QLineEdit()
    self.stage_approval_lock = QCheckBox("承認後ロック")
    self.stage_approval_lock.setChecked(True)
    approval_form.addRow("承認者", self.stage_approval_user)
    approval_form.addRow("承認メモ", self.stage_approval_note)
    approval_form.addRow("", self.stage_approval_lock)
    diff_layout.addLayout(approval_form)
    stage_template_host = QWidget()
    stage_template_host.setVisible(False)
    template_controls = QHBoxLayout(stage_template_host)
    self.stage_template_combo = QComboBox()
    self._refresh_stage_template_combo()
    apply_template_btn = QPushButton("業務テンプレート適用")
    apply_template_btn.clicked.connect(lambda _checked=False: self.apply_stage_template_from_library(self.stage_template_combo.currentData() or self.stage_template_combo.currentText()))
    save_template_btn = QPushButton("テンプレート保存")
    save_template_btn.clicked.connect(self.save_stage_template_library)
    load_template_btn = QPushButton("テンプレート読込")
    load_template_btn.clicked.connect(self.load_stage_template_library)
    self.stage_template_apply_button = apply_template_btn
    self.stage_template_save_button = save_template_btn
    self.stage_template_load_button = load_template_btn
    self.stage_template_combo.setMaximumWidth(150)
    for compact_button, compact_text in (
        (apply_template_btn, "適用"),
        (save_template_btn, "保存"),
        (load_template_btn, "読込"),
    ):
        compact_button.setToolTip(compact_button.text())
        compact_button.setText(compact_text)
    for widget in (QLabel("施工テンプレート"), self.stage_template_combo, apply_template_btn, save_template_btn, load_template_btn):
        template_controls.addWidget(widget)
    template_controls.addStretch(1)
    diff_layout.addWidget(stage_template_host)
    self.stage_compare_table = QTableWidget(0, 7)
    self.stage_compare_table.setHorizontalHeaderLabels(["ステージ", "種別", "要素", "材料", "境界", "荷重", "水理/MPC"])
    self.stage_compare_table.setToolTip("ステージ × 要素/材料/境界/荷重の一覧です。黄色のセルは前ステージから変わった項目です。")
    self.stage_compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    diff_layout.addWidget(self.stage_compare_table)
    self.stage_approval_history_table = QTableWidget(0, 7)
    self.stage_approval_history_table.setHorizontalHeaderLabels(["time", "action", "stage", "diff", "actor", "locked", "note"])
    self.stage_approval_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    diff_layout.addWidget(self.stage_approval_history_table)
    self.stage_conflict_table = QTableWidget(0, 6)
    self.stage_conflict_table.setHorizontalHeaderLabels(["stage", "区分", "対象", "内容", "修復", "提案"])
    self.stage_conflict_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    diff_layout.addWidget(self.stage_conflict_table)
    self.stage_guidance_label = QLabel("ステージを選択するとGeoFEAS風の入力ガイダンスを表示します。")
    self.stage_guidance_label.setWordWrap(True)
    diff_layout.addWidget(self.stage_guidance_label)
    self.stage_wizard_table = QTableWidget(0, 3)
    self.stage_wizard_table.setHorizontalHeaderLabels(["step", "作業", "確認"])
    self.stage_wizard_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    diff_layout.addWidget(self.stage_wizard_table)
    diff_tab_layout.addWidget(diff_box)
    apply_detail_btn = QPushButton("選択ステージ詳細を反映")
    apply_detail_btn.clicked.connect(self.apply_stage_detail_tables)
    detail_tab_layout.addWidget(apply_detail_btn)
    detail_tab_layout.addStretch(1)
    diff_tab_layout.addStretch(1)
    yaml_layout.addWidget(QLabel("詳細YAML"))
    self.stages_editor = QPlainTextEdit()
    yaml_layout.addWidget(self.stages_editor, 1)
    apply_btn = QPushButton("ステージYAMLを反映")
    apply_btn.clicked.connect(lambda _checked=False: self.apply_yaml_fragment("stages", expected=list))
    yaml_layout.addWidget(apply_btn)
    self.refresh_stage_recommendation_label()
    self._apply_panel_button_policy("stages", self.stage_form_workspace)
    self._apply_help_policy("stages", self.stage_form_workspace)
    self._apply_accessibility_policy(self.stage_form_workspace)
    return page

def build_results_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QFormLayout = qt["QFormLayout"]
    QGridLayout = qt["QGridLayout"]
    QGraphicsScene = qt["QGraphicsScene"]
    QGraphicsView = qt["QGraphicsView"]
    QGroupBox = qt["QGroupBox"]
    QHBoxLayout = qt["QHBoxLayout"]
    QHeaderView = qt["QHeaderView"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QPushButton = qt["QPushButton"]
    QTableWidget = qt["QTableWidget"]
    QTabWidget = qt["QTabWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    self.results_summary = QLabel("")
    self.results_summary.setWordWrap(True)

    self.result_empty_state = QWidget()
    self.result_empty_state.setProperty("informationRole", "primary")
    empty_layout = QVBoxLayout(self.result_empty_state)
    empty_layout.setContentsMargins(12, 14, 12, 14)
    empty_title = QLabel("解析結果はまだありません")
    empty_title.setProperty("informationRole", "primary")
    empty_detail = QLabel("解析を実行すると、ここに判定、主要値、探索精度、解析時間を表示します。")
    empty_detail.setWordWrap(True)
    empty_run_btn = QPushButton("解析実行へ進む")
    empty_run_btn.clicked.connect(lambda _checked=False: self._activate_panel("solver"))
    empty_layout.addWidget(empty_title)
    empty_layout.addWidget(empty_detail)
    empty_layout.addWidget(empty_run_btn)

    self.result_judgment_panel = QWidget()
    self.result_judgment_panel.setProperty("informationRole", "primary")
    judgment_layout = QVBoxLayout(self.result_judgment_panel)
    judgment_layout.setContentsMargins(10, 10, 10, 10)
    judgment_layout.setSpacing(6)
    judgment_title = QLabel("判定サマリ")
    judgment_title.setProperty("informationRole", "primary")
    self.result_judgment_headline = QLabel("-")
    self.result_judgment_headline.setWordWrap(True)
    self.result_judgment_headline.setMinimumHeight(42)
    self.result_judgment_status = QLabel("")
    self.result_judgment_status.setWordWrap(True)
    self.result_judgment_detail = QLabel("")
    self.result_judgment_detail.setWordWrap(True)
    judgment_layout.addWidget(judgment_title)
    judgment_layout.addWidget(self.result_judgment_headline)
    judgment_layout.addWidget(self.result_judgment_status)
    judgment_layout.addWidget(self.result_judgment_detail)

    metrics_widget = QWidget()
    metrics_grid = QGridLayout(metrics_widget)
    metrics_grid.setContentsMargins(0, 4, 0, 4)
    metrics_grid.setHorizontalSpacing(8)
    metrics_grid.setVerticalSpacing(6)
    self.result_judgment_metric_captions: list[QLabel] = []
    self.result_judgment_metric_values: list[QLabel] = []
    for index in range(6):
        metric = QWidget()
        metric.setProperty("informationRole", "auxiliary")
        metric_layout = QVBoxLayout(metric)
        metric_layout.setContentsMargins(6, 4, 6, 4)
        metric_layout.setSpacing(1)
        caption = QLabel("-")
        caption.setProperty("informationRole", "detail")
        value = QLabel("-")
        value.setWordWrap(True)
        value_font = value.font()
        value_font.setBold(True)
        value.setFont(value_font)
        metric_layout.addWidget(caption)
        metric_layout.addWidget(value)
        metrics_grid.addWidget(metric, index // 2, index % 2)
        self.result_judgment_metric_captions.append(caption)
        self.result_judgment_metric_values.append(value)
    metrics_grid.setColumnStretch(0, 1)
    metrics_grid.setColumnStretch(1, 1)
    judgment_layout.addWidget(metrics_widget)
    self.result_judgment_warning = QLabel("")
    self.result_judgment_warning.setWordWrap(True)
    self.result_judgment_warning.setProperty("severity", "warning")
    judgment_layout.addWidget(self.result_judgment_warning)
    self.result_judgment_panel.setVisible(False)

    self.result_stage_controls_widget = QWidget()
    stage_form = QFormLayout(self.result_stage_controls_widget)
    stage_form.setContentsMargins(0, 0, 0, 0)
    self.result_stage_selector = QComboBox()
    self.result_stage_selector.currentIndexChanged.connect(self.result_stage_changed)
    self.result_table_component = QComboBox()
    self.result_table_component.currentTextChanged.connect(lambda _text: self.refresh_distribution_plot())
    stage_form.addRow("結果ステージ", self.result_stage_selector)
    self.result_table_component_widget = QWidget()
    component_form = QFormLayout(self.result_table_component_widget)
    component_form.setContentsMargins(0, 0, 0, 0)
    component_form.addRow("表/分布成分", self.result_table_component)
    self.result_paging_widget = QWidget()
    paging_controls = QHBoxLayout(self.result_paging_widget)
    paging_controls.setContentsMargins(0, 0, 0, 0)
    prev_page_btn = QPushButton("前ページ")
    prev_page_btn.clicked.connect(self.result_table_previous_page)
    next_page_btn = QPushButton("次ページ")
    next_page_btn.clicked.connect(self.result_table_next_page)
    self.result_table_page_label = QLabel("数値表: 0件")
    self.result_table_page_label.setWordWrap(True)
    paging_controls.addWidget(prev_page_btn)
    paging_controls.addWidget(next_page_btn)
    paging_controls.addWidget(self.result_table_page_label)
    paging_controls.addStretch(1)
    open_btn = QPushButton("最新結果フォルダを表示")
    open_btn.clicked.connect(self.show_last_run_path)
    disp_btn = QPushButton("変形図/変位表")
    disp_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("displacements"))
    disp_contour_btn = QPushButton("変位コンタ")
    disp_contour_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("displacement_contour"))
    vector_btn = QPushButton("変位ベクトル")
    vector_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("displacement_vectors"))
    elem_btn = QPushButton("応力コンター")
    elem_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("element_stress"))
    plastic_btn = QPushButton("塑性/SRM表示")
    plastic_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("plastic"))
    safety_btn = QPushButton("FL/安全率図")
    safety_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("safety_factor"))
    srm_post_btn = QPushButton("SRM専用Post")
    srm_post_btn.clicked.connect(self.show_srm_post_async)
    reaction_btn = QPushButton("反力表")
    reaction_btn.clicked.connect(lambda _checked=False: self.load_result_table("reactions"))
    iface_btn = QPushButton("界面状態")
    iface_btn.clicked.connect(lambda _checked=False: self.load_result_table("interface_state"))
    structural_btn = QPushButton("Structural Post")
    structural_btn.clicked.connect(lambda _checked=False: self.load_result_table("structural_state"))
    structural_section_btn = QPushButton("Beam section")
    structural_section_btn.clicked.connect(lambda _checked=False: self.load_result_table("structural_section_forces"))
    pore_btn = QPushButton("水圧表")
    pore_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("pore_pressure"))
    riks_btn = QPushButton("Riks経路")
    riks_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("riks_path"))
    quality_btn = QPushButton("メッシュ品質")
    quality_btn.clicked.connect(lambda _checked=False: self.load_result_table("mesh_quality"))
    analysis_log_btn = QPushButton("解析ログ")
    analysis_log_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("analysis_log"))
    performance_btn = QPushButton("性能")
    performance_btn.clicked.connect(lambda _checked=False: self.load_result_table_async("performance"))
    large_ops_btn = QPushButton("大規模操作")
    large_ops_btn.clicked.connect(lambda _checked=False: self.load_result_table("large_model_operations"))
    node_index_btn = QPushButton("節点検索")
    node_index_btn.clicked.connect(lambda _checked=False: self.load_result_table("node_search_index"))
    element_index_btn = QPushButton("要素検索")
    element_index_btn.clicked.connect(lambda _checked=False: self.load_result_table("element_search_index"))
    standard_report_btn = QPushButton("標準帳票")
    standard_report_btn.clicked.connect(lambda _checked=False: self.load_result_table("standard_report"))
    srm_btn = QPushButton("SRM要約")
    srm_btn.clicked.connect(self.show_srm_summary)
    dist_btn = QPushButton("分布図")
    dist_btn.clicked.connect(self.refresh_distribution_plot)
    export_btn = QPushButton("表CSV保存")
    export_btn.clicked.connect(self.export_result_table_csv)
    drawing_btn = QPushButton("図面出力")
    drawing_btn.clicked.connect(self.export_scene_image_async)
    measurement_btn = QPushButton("任意測線分布")
    measurement_btn.clicked.connect(self.create_measurement_distribution)
    snapshot_btn = QPushButton("Postビュー複製")
    snapshot_btn.clicked.connect(self.duplicate_post_view_async)
    layout_add_btn = QPushButton("図面配置へ追加")
    layout_add_btn.clicked.connect(self.add_current_view_to_drawing_layout_async)
    verify_post_btn = QPushButton("Post図検証")
    verify_post_btn.clicked.connect(self.verify_post_view_render)
    save_template_btn = QPushButton("図面枠保存")
    save_template_btn.clicked.connect(lambda _checked=False: self.save_drawing_template())
    load_template_btn = QPushButton("図面枠読込")
    load_template_btn.clicked.connect(lambda _checked=False: self.load_drawing_template())
    install_template_btn = QPushButton("企業様式配置")
    install_template_btn.clicked.connect(lambda _checked=False: self.install_project_drawing_templates())
    save_baseline_btn = QPushButton("Post基準保存")
    save_baseline_btn.clicked.connect(lambda _checked=False: self.save_post_baseline_async())
    compare_baseline_btn = QPushButton("Post画像差分")
    compare_baseline_btn.clicked.connect(lambda _checked=False: self.compare_post_to_baseline_async())
    ci_job_btn = QPushButton("画像差分CI")
    ci_job_btn.clicked.connect(lambda _checked=False: self.write_post_image_diff_ci_job())
    pdf_btn = QPushButton("PDF図面")
    pdf_btn.clicked.connect(self.export_scene_pdf_async)
    result_visual_btn = QPushButton("結果図を開く")
    result_visual_btn.clicked.connect(lambda _checked=False: self.show_default_result_visual_if_available(force=True))
    result_report_btn = QPushButton("帳票へ進む")
    result_report_btn.clicked.connect(lambda _checked=False: self._activate_panel("report"))
    self.result_primary_visual_button = result_visual_btn
    self.result_primary_srm_button = srm_btn
    self.result_primary_report_button = result_report_btn
    self.result_primary_folder_button = open_btn
    for button, icon_role in (
        (result_visual_btn, "result.visual"),
        (result_report_btn, "result.report"),
        (open_btn, "result.folder"),
        (srm_btn, "result.srm"),
        (export_btn, "result.export"),
        (drawing_btn, "result.export"),
        (pdf_btn, "result.export"),
    ):
        self._apply_button_icon(button, icon_role)
    primary_result_buttons = [result_visual_btn, result_report_btn, open_btn, srm_btn]
    detailed_result_buttons = [
        disp_btn,
        disp_contour_btn,
        vector_btn,
        elem_btn,
        plastic_btn,
        safety_btn,
        srm_post_btn,
        pore_btn,
        quality_btn,
        analysis_log_btn,
        performance_btn,
        dist_btn,
        measurement_btn,
        export_btn,
        drawing_btn,
    ]
    advanced_result_buttons = [
        riks_btn,
        reaction_btn,
        iface_btn,
        structural_btn,
        structural_section_btn,
        large_ops_btn,
        node_index_btn,
        element_index_btn,
        standard_report_btn,
        snapshot_btn,
        layout_add_btn,
        verify_post_btn,
        save_template_btn,
        load_template_btn,
        install_template_btn,
        save_baseline_btn,
        compare_baseline_btn,
        ci_job_btn,
        pdf_btn,
    ]
    self.result_data_buttons = [
        *primary_result_buttons,
        *detailed_result_buttons,
        riks_btn,
        reaction_btn,
        iface_btn,
        structural_btn,
        structural_section_btn,
        large_ops_btn,
        node_index_btn,
        element_index_btn,
        standard_report_btn,
    ]
    scale_form = QFormLayout()
    self.deformation_scale = QLineEdit("1.0")
    self.deformation_scale.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="deformation scale"))
    self.result_component = QComboBox()
    populate_labeled_combo(
        self.result_component,
        "result_component",
        ["q", "p", "sigma_x", "sigma_y", "sigma_z", "tau_xy", "sigma_1", "sigma_2", "sigma_3", "tau_max", "eps_x", "eps_y", "gamma_xy", "plastic", "yield_value", "FL", "safety_factor"],
        locale=getattr(self, "gui_locale", "ja"),
    )
    self.result_component.currentIndexChanged.connect(lambda _index: self.reload_selected_result_component_async())
    self.show_deformed_overlay = QCheckBox("変形前後重ね表示")
    self.show_deformed_overlay.setChecked(True)
    self.show_deformed_overlay.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="deformed overlay"))
    self.show_element_boundaries = QCheckBox("要素境界ON/OFF")
    self.show_element_boundaries.setChecked(True)
    self.show_element_boundaries.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="element boundaries"))
    self.show_contour_labels = QCheckBox("等値線ラベル")
    self.show_contour_labels.setChecked(False)
    self.show_contour_labels.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="contour labels"))
    self.show_contour_lines = QCheckBox("等値線ポリライン")
    self.show_contour_lines.setChecked(False)
    self.show_contour_lines.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="contour lines"))
    self.contour_interpolation = QComboBox()
    self.contour_interpolation.addItems(["線形", "曲線補間"])
    self.contour_interpolation.currentTextChanged.connect(lambda _text: self.request_preview_update(reset_view=False, reason="contour interpolation"))
    self.contour_level_count = QLineEdit("7")
    self.contour_level_count.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="contour levels"))
    self.contour_curve_segments = QLineEdit("4")
    self.contour_curve_segments.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="contour segments"))
    self.display_vector_limit = QLineEdit("400")
    self.display_vector_limit.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="vector limit"))
    self.smooth_contours = QCheckBox("節点平均スムージング")
    self.smooth_contours.setChecked(False)
    self.smooth_contours.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="smooth contours"))
    self.clip_contours_to_legend = QCheckBox("凡例範囲でクリップ")
    self.clip_contours_to_legend.setChecked(False)
    self.clip_contours_to_legend.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="clip contours"))
    self.result_colormap = QComboBox()
    self.result_colormap.setEditable(True)
    self.result_colormap.addItems(["Geo blue-red", "Safety FL", "Viridis", "Terrain", "Gray", "custom:#313695,#ffffbf,#a50026"])
    self.result_colormap.currentTextChanged.connect(self._result_colormap_changed)
    self.legend_title_edit = QLineEdit()
    self.legend_title_edit.setPlaceholderText("auto")
    self.legend_title_edit.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="legend title"))
    self.legend_min_edit = QLineEdit()
    self.legend_min_edit.setPlaceholderText("auto")
    self.legend_min_edit.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="legend min"))
    self.legend_max_edit = QLineEdit()
    self.legend_max_edit.setPlaceholderText("auto")
    self.legend_max_edit.editingFinished.connect(lambda: self.request_preview_update(reset_view=False, reason="legend max"))
    self.measure_line_start = QLineEdit("0,0")
    self.measure_line_end = QLineEdit("1,0")
    self.drawing_title_edit = QLineEdit("GeoFEM 2D Post")
    self.drawing_scale_edit = QLineEdit("1:100")
    self.drawing_template = QComboBox()
    self._refresh_drawing_template_combo()
    scale_form.addRow("変形倍率", self.deformation_scale)
    scale_form.addRow("コンター成分", self.result_component)
    scale_form.addRow("変形図", self.show_deformed_overlay)
    scale_form.addRow("要素境界", self.show_element_boundaries)
    scale_form.addRow("等値線ラベル", self.show_contour_labels)
    scale_form.addRow("等値線", self.show_contour_lines)
    scale_form.addRow("等値線補間", self.contour_interpolation)
    scale_form.addRow("等値線本数", self.contour_level_count)
    scale_form.addRow("曲線分割数", self.contour_curve_segments)
    scale_form.addRow("ベクトル上限", self.display_vector_limit)
    scale_form.addRow("スムージング", self.smooth_contours)
    scale_form.addRow("クリッピング", self.clip_contours_to_legend)
    scale_form.addRow("カラーマップ", self.result_colormap)
    scale_form.addRow("凡例タイトル", self.legend_title_edit)
    scale_form.addRow("凡例最小", self.legend_min_edit)
    scale_form.addRow("凡例最大", self.legend_max_edit)
    scale_form.addRow("測線始点 x,y", self.measure_line_start)
    scale_form.addRow("測線終点 x,y", self.measure_line_end)
    scale_form.addRow("図面タイトル", self.drawing_title_edit)
    scale_form.addRow("縮尺", self.drawing_scale_edit)
    scale_form.addRow("図面枠テンプレート", self.drawing_template)
    compare_group = QGroupBox("出力比較")
    compare_layout = QVBoxLayout(compare_group)
    self.case_compare_table = QTableWidget(0, 9)
    self.case_compare_table.setHorizontalHeaderLabels(["区分", "成果物", "指標", "状態", "現行", "基準", "絶対差", "相対差", "内容"])
    self.case_compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.case_compare_table.setMaximumHeight(170)
    previous_compare_btn = QPushButton("前回差分")
    previous_compare_btn.clicked.connect(self.compare_current_result_to_previous_async)
    baseline_compare_btn = QPushButton("基準差分")
    baseline_compare_btn.clicked.connect(self.compare_current_result_to_baseline_async)
    design_compare_btn = QPushButton("設計差分")
    design_compare_btn.clicked.connect(self.compare_current_result_to_design_async)
    compare_btn = QPushButton("比較読込")
    compare_btn.clicked.connect(self.load_compare_result)
    compare_help = QLabel("前回、指定基準、設計ケースの数値表、Post図、帳票を比較します。")
    compare_help.setWordWrap(True)
    compare_layout.addWidget(compare_help)
    compare_layout.addWidget(self.case_compare_table)
    self._add_panel_button_rows(compare_layout, [previous_compare_btn, baseline_compare_btn, design_compare_btn, compare_btn], columns=2)
    srm_group = QGroupBox("SRM専用Post")
    srm_layout = QVBoxLayout(srm_group)
    srm_form = QFormLayout()
    self.srm_fl_limit = QLineEdit("1.05")
    self.srm_plastic_threshold = QLineEdit("0.5")
    self.srm_fl_limit.editingFinished.connect(lambda: self.show_srm_post_async(update_only=True))
    self.srm_plastic_threshold.editingFinished.connect(lambda: self.show_srm_post_async(update_only=True))
    self.srm_local_fl_aggregation = QComboBox()
    populate_labeled_combo(self.srm_local_fl_aggregation, "srm_aggregation", ["mean", "min", "max"], locale=getattr(self, "gui_locale", "ja"))
    self.srm_local_fl_aggregation.currentIndexChanged.connect(lambda _index: self.show_srm_post_async(update_only=True))
    self.srm_search_mode = QComboBox()
    populate_labeled_combo(self.srm_search_mode, "srm_search_mode", ["all", "circular", "non-circular", "optimized path"], locale=getattr(self, "gui_locale", "ja"))
    self.srm_search_mode.currentIndexChanged.connect(lambda _index: self.show_srm_post_async(update_only=True))
    self.srm_slope_direction = QComboBox()
    populate_labeled_combo(self.srm_slope_direction, "srm_direction", ["auto", "left-to-right", "right-to-left"], locale=getattr(self, "gui_locale", "ja"))
    self.srm_slope_direction.currentIndexChanged.connect(lambda _index: self.show_srm_post_async(update_only=True))
    self.srm_min_candidate_length = QLineEdit("0.0")
    self.srm_max_circle_radius = QLineEdit("0.0")
    self.srm_min_candidate_length.editingFinished.connect(lambda: self.show_srm_post_async(update_only=True))
    self.srm_max_circle_radius.editingFinished.connect(lambda: self.show_srm_post_async(update_only=True))
    self.srm_require_boundary_exit = QCheckBox("entry/exit boundary")
    self.srm_require_boundary_exit.stateChanged.connect(lambda _state: self.show_srm_post_async(update_only=True))
    self.srm_show_fl_contour = QCheckBox("FL等高線")
    self.srm_show_fl_contour.setChecked(True)
    self.srm_show_fl_contour.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="srm fl contour"))
    self.srm_show_slip_candidates = QCheckBox("すべり面候補")
    self.srm_show_slip_candidates.setChecked(True)
    self.srm_show_slip_candidates.stateChanged.connect(lambda _state: self.request_preview_update(reset_view=False, reason="srm slip candidates"))
    srm_form.addRow("危険FL上限", self.srm_fl_limit)
    srm_form.addRow("塑性判定しきい値", self.srm_plastic_threshold)
    srm_form.addRow("積分点FL集計", self.srm_local_fl_aggregation)
    srm_form.addRow("探索", self.srm_search_mode)
    srm_form.addRow("探索方向", self.srm_slope_direction)
    srm_form.addRow("最小候補長", self.srm_min_candidate_length)
    srm_form.addRow("最大円弧半径", self.srm_max_circle_radius)
    srm_form.addRow("境界入出", self.srm_require_boundary_exit)
    srm_form.addRow("表示", self.srm_show_fl_contour)
    srm_form.addRow("", self.srm_show_slip_candidates)
    self.srm_post_summary = QLabel("SRM専用Postは解析後に更新します。")
    self.srm_post_summary.setWordWrap(True)
    self.srm_trial_table = QTableWidget(0, 5)
    self.srm_trial_table.setHorizontalHeaderLabels(["factor", "converged", "plastic_ratio", "ok", "note"])
    self.srm_trial_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.srm_trial_table.setMaximumHeight(130)
    self.srm_slip_table = QTableWidget(0, 6)
    self.srm_slip_table.setHorizontalHeaderLabels(["rank", "elements", "length", "mean_y", "min_FL", "score"])
    self.srm_slip_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.srm_slip_table.setMaximumHeight(150)
    self.srm_candidate_compare_table = QTableWidget(0, 10)
    self.srm_candidate_compare_table.setHorizontalHeaderLabels(["rank", "type", "elements", "length", "min_FL", "mean_FL", "max_FL", "plastic_ratio", "score", "optimized"])
    self.srm_candidate_compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.srm_candidate_compare_table.setMaximumHeight(170)
    self.srm_local_fl_table = QTableWidget(0, 8)
    self.srm_local_fl_table.setHorizontalHeaderLabels(["element", "ip", "x", "y", "FL", "q", "p", "note"])
    self.srm_local_fl_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.srm_local_fl_table.setMaximumHeight(170)
    srm_update_btn = QPushButton("SRM Post更新")
    srm_update_btn.clicked.connect(self.show_srm_post_async)
    srm_export_btn = QPushButton("すべり面CSV保存")
    srm_export_btn.clicked.connect(self.export_srm_slip_csv)
    srm_report_btn = QPushButton("候補詳細帳票")
    srm_report_btn.clicked.connect(lambda _checked=False: self.build_srm_candidate_report())
    srm_layout.addLayout(srm_form)
    srm_layout.addWidget(self.srm_post_summary)
    srm_layout.addWidget(QLabel("SRM試行履歴"))
    srm_layout.addWidget(self.srm_trial_table)
    srm_layout.addWidget(QLabel("すべり面候補"))
    srm_layout.addWidget(self.srm_slip_table)
    srm_layout.addWidget(QLabel("候補比較"))
    srm_layout.addWidget(self.srm_candidate_compare_table)
    srm_layout.addWidget(QLabel("積分点局所FL"))
    srm_layout.addWidget(self.srm_local_fl_table)
    self._add_panel_button_rows(srm_layout, [srm_update_btn, srm_export_btn, srm_report_btn], columns=2)
    drawing_group = QGroupBox("図面レイアウト")
    drawing_layout = QVBoxLayout(drawing_group)
    self.drawing_layout_table = QTableWidget(0, 7)
    self.drawing_layout_table.setHorizontalHeaderLabels(["image", "title", "scale", "x", "y", "w", "h"])
    self.drawing_layout_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    drawing_layout.addWidget(self.drawing_layout_table)
    drawing_controls = QHBoxLayout()
    clear_layout_btn = QPushButton("配置クリア")
    clear_layout_btn.clicked.connect(lambda _checked=False: self.drawing_layout_table.setRowCount(0))
    drawing_controls.addWidget(clear_layout_btn)
    drawing_controls.addStretch(1)
    drawing_layout.addLayout(drawing_controls)
    report_group = QGroupBox("帳票ページWYSIWYG")
    report_layout = QVBoxLayout(report_group)
    self.report_page_table = QTableWidget(0, 7)
    report_help = documentation_payload("report.item")
    self.report_page_table.setProperty("helpId", report_help["help_id"])
    self.report_page_table.setProperty("helpUrl", report_help["help_url"])
    self.report_page_table.setHorizontalHeaderLabels(["type", "source/text", "title", "x", "y", "w", "h"])
    self.report_page_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    report_layout.addWidget(self.report_page_table)
    self.report_layout_scene = QGraphicsScene(self)
    self.report_layout_view = QGraphicsView(self.report_layout_scene)
    self.report_layout_view.setMinimumHeight(260)
    report_layout.addWidget(self.report_layout_view)
    add_text_btn = QPushButton("テキスト追加")
    add_text_btn.clicked.connect(lambda _checked=False: self.add_report_text_block("判定表", "ここに判定を入力"))
    add_post_btn = QPushButton("Post図追加")
    add_post_btn.clicked.connect(self.add_current_post_to_report_page)
    preview_report_btn = QPushButton("帳票プレビュー")
    preview_report_btn.clicked.connect(self.preview_report_page_layout)
    refresh_canvas_btn = QPushButton("配置更新")
    refresh_canvas_btn.clicked.connect(self.refresh_report_canvas)
    apply_canvas_btn = QPushButton("ドラッグ反映")
    apply_canvas_btn.clicked.connect(self.apply_report_canvas_positions)
    nudge_left_btn = QPushButton("←")
    nudge_left_btn.clicked.connect(lambda _checked=False: self.nudge_selected_report_block(dx=-0.02))
    nudge_right_btn = QPushButton("→")
    nudge_right_btn.clicked.connect(lambda _checked=False: self.nudge_selected_report_block(dx=0.02))
    nudge_up_btn = QPushButton("↑")
    nudge_up_btn.clicked.connect(lambda _checked=False: self.nudge_selected_report_block(dy=-0.02))
    nudge_down_btn = QPushButton("↓")
    nudge_down_btn.clicked.connect(lambda _checked=False: self.nudge_selected_report_block(dy=0.02))
    grow_btn = QPushButton("拡大")
    grow_btn.clicked.connect(lambda _checked=False: self.nudge_selected_report_block(dw=0.03, dh=0.03))
    shrink_btn = QPushButton("縮小")
    shrink_btn.clicked.connect(lambda _checked=False: self.nudge_selected_report_block(dw=-0.03, dh=-0.03))
    self._add_panel_button_rows(
        report_layout,
        [add_text_btn, add_post_btn, preview_report_btn, refresh_canvas_btn, apply_canvas_btn, nudge_left_btn, nudge_right_btn, nudge_up_btn, nudge_down_btn, grow_btn, shrink_btn],
        columns=2,
    )
    self.results_tabs = QTabWidget()
    self.results_tabs.setDocumentMode(True)

    overview_page = QWidget()
    overview_layout = QVBoxLayout(overview_page)
    overview_layout.addWidget(self.result_empty_state)
    overview_layout.addWidget(self.result_judgment_panel)
    overview_layout.addWidget(self.result_stage_controls_widget)
    self.result_primary_actions_widget = QWidget()
    primary_actions_layout = QVBoxLayout(self.result_primary_actions_widget)
    primary_actions_layout.setContentsMargins(0, 0, 0, 0)
    self._add_panel_button_rows(primary_actions_layout, primary_result_buttons, columns=2)
    overview_layout.addWidget(self.result_primary_actions_widget)
    self.result_more_button = QPushButton("詳細な結果を表示")
    self.result_more_button.setCheckable(True)
    overview_layout.addWidget(self.result_more_button)
    self.result_detail_actions_widget = QWidget()
    detail_actions_layout = QVBoxLayout(self.result_detail_actions_widget)
    detail_actions_layout.setContentsMargins(0, 0, 0, 0)
    detail_actions_layout.addWidget(self.result_table_component_widget)
    self._add_panel_button_rows(detail_actions_layout, detailed_result_buttons, columns=2)
    detail_actions_layout.addWidget(self.result_paging_widget)
    self.result_detail_actions_widget.setVisible(False)

    def toggle_result_details(checked: bool) -> None:
        self.result_detail_actions_widget.setVisible(bool(checked))
        if str(getattr(self, "gui_locale", "ja")).startswith("en"):
            self.result_more_button.setText("Hide Detailed Results" if checked else "Show Detailed Results")
        else:
            self.result_more_button.setText("詳細な結果を隠す" if checked else "詳細な結果を表示")

    self.result_more_button.toggled.connect(toggle_result_details)
    overview_layout.addWidget(self.result_detail_actions_widget)
    overview_layout.addWidget(self.results_summary)
    overview_layout.addStretch(1)

    display_page = QWidget()
    display_layout = QVBoxLayout(display_page)
    display_layout.addLayout(scale_form)
    display_layout.addStretch(1)

    srm_page = QWidget()
    srm_page_layout = QVBoxLayout(srm_page)
    srm_page_layout.addWidget(srm_group)
    srm_page_layout.addStretch(1)

    output_page = QWidget()
    output_layout = QVBoxLayout(output_page)
    self._add_panel_button_rows(output_layout, advanced_result_buttons, columns=2)
    output_layout.addWidget(drawing_group)
    output_layout.addWidget(report_group)
    output_layout.addWidget(compare_group)
    output_layout.addStretch(1)

    self.results_tabs.addTab(overview_page, "結果")
    self.results_tabs.addTab(display_page, "表示設定")
    self.results_tabs.addTab(srm_page, "SRM")
    self.results_tabs.addTab(output_page, "出力・比較")
    layout.addWidget(self.results_tabs, 1)
    return page

def build_report_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    self = owner
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    page = QWidget()
    layout = QVBoxLayout(page)
    self.report_summary = QLabel("HTMLレポートは各ステージの出力フォルダに生成されます。")
    self.report_summary.setWordWrap(True)
    self.report_include_summary = QCheckBox("解析サマリ")
    self.report_include_summary.setChecked(True)
    self.report_include_tables = QCheckBox("数値表リンク")
    self.report_include_tables.setChecked(True)
    self.report_template_combo = QComboBox()
    self.report_template_combo.addItems(["標準帳票", "GeoFEAS風 判定付き", "設計照査帳票"])
    preview_btn = QPushButton("既存レポートをプレビュー")
    preview_btn.clicked.connect(self.preview_stage_report)
    build_btn = QPushButton("選択内容で計算書作成")
    build_btn.clicked.connect(self.build_selected_report_async)
    pdf_btn = QPushButton("計算書PDF保存")
    pdf_btn.clicked.connect(self.export_calculation_report_pdf_async)
    audit_btn = QPushButton("Post/帳票監査")
    audit_help = documentation_payload("report.audit")
    audit_btn.setProperty("helpId", audit_help["help_id"])
    audit_btn.setProperty("helpUrl", audit_help["help_url"])
    audit_btn.clicked.connect(self.audit_post_report_async)
    wysiwyg_btn = QPushButton("WYSIWYG帳票プレビュー")
    wysiwyg_btn.clicked.connect(self.preview_report_page_layout)
    layout.addWidget(self.report_summary)
    layout.addWidget(self.report_include_summary)
    layout.addWidget(self.report_include_tables)
    layout.addWidget(self.report_template_combo)
    layout.addWidget(preview_btn)
    layout.addWidget(build_btn)
    layout.addWidget(pdf_btn)
    layout.addWidget(audit_btn)
    layout.addWidget(wysiwyg_btn)
    layout.addStretch(1)
    return page

__all__ = [
    "DOMAIN_PANEL_KEYS",
    "domain_panel_contract",
    "build_mesh_panel",
    "build_geometry_panel",
    "build_external_link_panel",
    "build_materials_panel",
    "build_boundary_conditions_panel",
    "build_loads_panel",
    "build_stages_panel",
    "build_results_panel",
    "build_report_panel",
]
