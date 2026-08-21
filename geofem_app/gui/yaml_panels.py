"""Reusable YAML editor panel helpers for GUI configuration surfaces."""

from __future__ import annotations

from typing import Any, Mapping


YAML_FRAGMENT_EXPECTED_TYPES = {
    "list": list,
    "mapping": dict,
}


def yaml_panel_contract() -> dict[str, Any]:
    """Return a UI-independent description of YAML editor panel behavior."""

    return {
        "schema": "geofem.gui.yaml_panels.v1",
        "fragment_expected_types": sorted(YAML_FRAGMENT_EXPECTED_TYPES),
        "editor_attribute_pattern": "{key}_editor",
        "apply_callback": "apply_yaml_fragment",
        "root_sync_callback": "sync_from_yaml",
        "surfaces": ["solver", "generic_list", "generic_mapping", "root_yaml"],
    }


def build_yaml_list_panel(owner: Any, title: str, key: str, qt: Mapping[str, Any]) -> Any:
    """Build a YAML list fragment editor panel."""

    return build_yaml_fragment_panel(owner, title, key, list, "YAMLリスト", qt)


def build_yaml_mapping_panel(owner: Any, title: str, key: str, qt: Mapping[str, Any]) -> Any:
    """Build a YAML mapping fragment editor panel."""

    return build_yaml_fragment_panel(owner, title, key, dict, "YAMLマッピング", qt)


def build_yaml_fragment_panel(owner: Any, title: str, key: str, expected: type, suffix: str, qt: Mapping[str, Any]) -> Any:
    """Build a typed YAML fragment editor and wire it to MainWindow state."""

    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QLabel = qt["QLabel"]
    QPlainTextEdit = qt["QPlainTextEdit"]
    QPushButton = qt["QPushButton"]

    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(QLabel(f"{title} {suffix}"))
    editor = QPlainTextEdit()
    setattr(owner, f"{key}_editor", editor)
    layout.addWidget(editor)
    apply_btn = QPushButton(f"{title}を反映")
    apply_btn.clicked.connect(lambda _checked=False, k=key, t=expected: owner.apply_yaml_fragment(k, expected=t))
    layout.addWidget(apply_btn)
    return page


def build_root_yaml_panel(owner: Any, qt: Mapping[str, Any]) -> Any:
    """Build the root YAML helper panel."""

    QWidget = qt["QWidget"]
    QVBoxLayout = qt["QVBoxLayout"]
    QLabel = qt["QLabel"]
    QPushButton = qt["QPushButton"]

    page = QWidget()
    layout = QVBoxLayout(page)
    sync_btn = QPushButton("YAMLからフォームへ反映")
    sync_btn.clicked.connect(owner.sync_from_yaml)
    layout.addWidget(sync_btn)
    layout.addWidget(QLabel("中央のYAMLタブで直接編集できます。"))
    layout.addStretch(1)
    return page
