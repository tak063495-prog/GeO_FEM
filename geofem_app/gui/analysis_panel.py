"""Analysis settings panel construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from geofem_app.gui.presentation_labels import combo_value, populate_labeled_combo


ANALYSIS_TYPE_OPTIONS = ("static_plane_strain", "axisymmetric_static", "geostatic", "srm", "consolidation", "riks")
ANALYSIS_GEOMETRY_OPTIONS = ("plane_strain", "axisymmetric")
ANALYSIS_DEFORMATION_MODE_OPTIONS = ("small_deformation", "large_deformation")


@dataclass(frozen=True)
class AnalysisFieldSpec:
    label: str
    attribute: str


ANALYSIS_FORM_FIELDS: tuple[AnalysisFieldSpec, ...] = (
    AnalysisFieldSpec("解析種別", "analysis_type"),
    AnalysisFieldSpec("解析形状", "analysis_geometry"),
    AnalysisFieldSpec("変形モード", "analysis_deformation_mode"),
    AnalysisFieldSpec("連成", "analysis_up_fields"),
    AnalysisFieldSpec("単位系", "unit_system"),
)


def analysis_panel_contract() -> dict[str, Any]:
    """Return a UI-independent description of the analysis settings panel."""

    return {
        "schema": "geofem.gui.analysis_panel.v1",
        "analysis_types": list(ANALYSIS_TYPE_OPTIONS),
        "analysis_geometries": list(ANALYSIS_GEOMETRY_OPTIONS),
        "deformation_modes": list(ANALYSIS_DEFORMATION_MODE_OPTIONS),
        "fields": [{"label": field.label, "attribute": field.attribute} for field in ANALYSIS_FORM_FIELDS],
        "apply_callback": "apply_analysis_panel",
        "axisymmetric_callbacks": ["apply_axisymmetric_reference_sets", "apply_axisymmetric_standard_presets"],
        "template_callbacks": ["refresh_input_template_combo", "load_selected_input_template", "load_project_template_input"],
    }


def build_analysis_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    """Build the analysis settings QWidget while keeping MainWindow layout-light."""

    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QFormLayout = qt["QFormLayout"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QLineEdit = qt["QLineEdit"]
    QComboBox = qt["QComboBox"]
    QCheckBox = qt["QCheckBox"]
    QGroupBox = qt["QGroupBox"]
    QPushButton = qt["QPushButton"]

    page = QWidget()
    layout = QVBoxLayout(page)
    form = QFormLayout()

    owner.analysis_type = QComboBox()
    populate_labeled_combo(owner.analysis_type, "analysis_type", ANALYSIS_TYPE_OPTIONS, locale=getattr(owner, "gui_locale", "ja"))
    owner.analysis_geometry = QComboBox()
    populate_labeled_combo(owner.analysis_geometry, "analysis_geometry", ANALYSIS_GEOMETRY_OPTIONS, locale=getattr(owner, "gui_locale", "ja"))
    owner.analysis_deformation_mode = QComboBox()
    populate_labeled_combo(owner.analysis_deformation_mode, "deformation_mode", ANALYSIS_DEFORMATION_MODE_OPTIONS, locale=getattr(owner, "gui_locale", "ja"))
    owner.axisym_show_radial_guides = QCheckBox("軸対称 r-z ガイド")
    owner.axisym_show_radial_guides.setChecked(True)
    owner.axisym_show_radial_guides.stateChanged.connect(lambda _state: owner.request_preview_update(reset_view=False, reason="axisymmetric guide"))
    owner.analysis_up_fields = QCheckBox("u-p/圧密フィールド")
    owner.unit_system = QLineEdit()
    owner.analysis_summary = QLabel()
    owner.analysis_summary.setWordWrap(True)

    owner.analysis_field_labels = {}
    for field in ANALYSIS_FORM_FIELDS:
        field_widget = getattr(owner, field.attribute)
        field_label = QLabel(field.label)
        field_label.setBuddy(field_widget)
        owner.analysis_field_labels[field.attribute] = field_label
        form.addRow(field_label, field_widget)
    layout.addLayout(form)
    layout.addWidget(owner.axisym_show_radial_guides)

    template_box = QGroupBox("入力テンプレート")
    template_layout = QVBoxLayout(template_box)
    owner.input_template_combo = QComboBox()
    owner.input_template_combo.setToolTip("標準サンプル、examples、プロジェクトtemplates配下の入力テンプレートを選びます。")
    template_layout.addWidget(owner.input_template_combo)
    template_buttons = QHBoxLayout()
    load_selected_template_btn = QPushButton("選択テンプレートを読込")
    load_selected_template_btn.clicked.connect(owner.load_selected_input_template)
    load_template_file_btn = QPushButton("ファイルから読込")
    load_template_file_btn.clicked.connect(owner.load_project_template_input)
    template_buttons.addWidget(load_selected_template_btn)
    template_buttons.addWidget(load_template_file_btn)
    template_buttons.addStretch(1)
    template_layout.addLayout(template_buttons)
    if hasattr(owner, "refresh_input_template_combo"):
        owner.refresh_input_template_combo()
    layout.addWidget(template_box)

    axisym_box = QGroupBox("軸対称プリセット")
    owner.analysis_axisym_box = axisym_box
    axisym_layout = QHBoxLayout(axisym_box)
    axisym_sets_btn = QPushButton("r/z集合")
    axisym_sets_btn.clicked.connect(owner.apply_axisymmetric_reference_sets)
    axisym_standard_btn = QPushButton("標準設定")
    axisym_standard_btn.clicked.connect(owner.apply_axisymmetric_standard_presets)
    axisym_layout.addWidget(axisym_sets_btn)
    axisym_layout.addWidget(axisym_standard_btn)
    axisym_layout.addStretch(1)
    layout.addWidget(axisym_box)

    output_box = QGroupBox("出力先")
    output_layout = QFormLayout(output_box)
    owner.analysis_output_policy = QComboBox()
    owner.analysis_output_policy.addItem("入力YAMLと同じ場所", "same_as_input")
    owner.analysis_output_policy.addItem("プロジェクト runs", "project_runs")
    owner.analysis_output_policy.addItem("任意フォルダー", "custom")
    owner.analysis_output_policy.currentIndexChanged.connect(owner.update_analysis_output_path_preview)
    owner.analysis_output_custom_dir = QLineEdit()
    owner.analysis_output_custom_dir.setPlaceholderText("任意フォルダー")
    owner.analysis_output_custom_dir.editingFinished.connect(owner.update_analysis_output_path_preview)
    browse_output_btn = QPushButton("参照")
    browse_output_btn.clicked.connect(owner.browse_analysis_output_directory)
    owner.analysis_output_custom_row = QWidget()
    custom_row = QHBoxLayout(owner.analysis_output_custom_row)
    custom_row.setContentsMargins(0, 0, 0, 0)
    custom_row.addWidget(owner.analysis_output_custom_dir)
    custom_row.addWidget(browse_output_btn)
    owner.analysis_output_csv = QCheckBox("CSV")
    owner.analysis_output_csv.setChecked(True)
    owner.analysis_output_vtk = QCheckBox("VTK")
    owner.analysis_output_vtk.setChecked(True)
    owner.analysis_output_log = QCheckBox("log")
    owner.analysis_output_log.setChecked(True)
    format_row = QHBoxLayout()
    for widget in (owner.analysis_output_csv, owner.analysis_output_vtk, owner.analysis_output_log):
        widget.stateChanged.connect(lambda _state: owner.update_analysis_output_path_preview())
        format_row.addWidget(widget)
    format_row.addStretch(1)
    owner.analysis_output_preview = QLabel()
    owner.analysis_output_preview.setWordWrap(True)
    owner.analysis_output_preview.setMinimumHeight(38)
    owner.analysis_output_preview.setMaximumHeight(64)
    output_layout.addRow("場所", owner.analysis_output_policy)
    owner.analysis_output_custom_label = QLabel("任意")
    owner.analysis_output_custom_label.setBuddy(owner.analysis_output_custom_dir)
    output_layout.addRow(owner.analysis_output_custom_label, owner.analysis_output_custom_row)
    output_layout.addRow("形式", format_row)
    output_layout.addRow("解決後", owner.analysis_output_preview)
    layout.addWidget(output_box)

    apply_btn = QPushButton("解析条件を反映")
    apply_btn.clicked.connect(owner.apply_analysis_panel)
    layout.addWidget(apply_btn)
    layout.addWidget(owner.analysis_summary)
    layout.addStretch(1)

    def refresh_conditional_fields() -> None:
        axisymmetric = (
            combo_value(owner.analysis_geometry) == "axisymmetric"
            or combo_value(owner.analysis_type) == "axisymmetric_static"
        )
        owner.axisym_show_radial_guides.setVisible(axisymmetric)
        owner.analysis_axisym_box.setVisible(axisymmetric)
        custom_output = str(owner.analysis_output_policy.currentData() or "") == "custom"
        owner.analysis_output_custom_label.setVisible(custom_output)
        owner.analysis_output_custom_row.setVisible(custom_output)

    owner.analysis_geometry.currentTextChanged.connect(lambda _text: refresh_conditional_fields())
    owner.analysis_type.currentTextChanged.connect(lambda _text: refresh_conditional_fields())
    owner.analysis_output_policy.currentIndexChanged.connect(lambda _index: refresh_conditional_fields())
    owner.refresh_analysis_conditional_fields = refresh_conditional_fields
    refresh_conditional_fields()
    return page
