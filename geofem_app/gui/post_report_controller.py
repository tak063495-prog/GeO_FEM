"""Post, result, and report operation controller functions split from MainWindow.

The MainWindow keeps Qt widget ownership and scene-level primitives; this module
owns the larger result-table, Post export, SRM Post, and report workflow actions.
"""
from __future__ import annotations

import csv
import copy
from datetime import datetime
import hashlib
import html
from pathlib import Path
import json
import math
import shutil
from typing import Any, Mapping

import yaml

from geofem_app.customization import apply_organization_profile, default_organization_profile, load_organization_profile, write_customization_artifacts
from geofem_app.gui.file_preview import preview_text_file
from geofem_app.gui.help_system import documentation_payload
from geofem_app.gui.post_export_worker import (
    audit_post_report_snapshot,
    build_selected_report_snapshot,
    compare_post_image_snapshot,
    copy_report_pdf_snapshot,
    export_scene_pdf_snapshot,
    save_post_image_snapshot,
)
from geofem_app.gui.post_worker import load_post_table_snapshot, load_srm_post_snapshot, materialize_post_component_snapshot
from geofem_app.gui.result_paging import DEFAULT_RESULT_TABLE_PAGE_SIZE, CsvTableSummary, ResultTablePage, read_csv_table_page, result_table_page, summarize_csv_table
from geofem_app.gui.result_summary import build_result_judgment_summary
from geofem_app.gui.result_table_routes import is_root_result_kind, result_table_path
from geofem_app.gui.surface_texts import gui_surface_message
from geofem_app.fem2d_io import write_deferred_run_artifacts_from_files
from geofem_app.output_comparison import compare_result_cases
from geofem_app.version_info import build_version_info
from geofem_app.workspace_management import write_workspace_dashboard


def _notify_information(owner: Any, message_box: Any, message: str) -> None:
    if hasattr(owner, "notify_user"):
        owner.notify_user(message)
    else:
        message_box.information(owner, "GeoFEM", message)


def _performance_result_summary(path: Path) -> str:
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return ""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, Mapping):
        return ""
    parts = [
        f"総時間 {data.get('elapsed_seconds', '')}s",
        f"支配カテゴリ {data.get('dominant_category', '')}",
        f"最遅ステージ {data.get('dominant_stage', '')}",
        f"キャッシュ再利用 {data.get('cache_reuse_count', 0)}",
    ]
    slowest = data.get("slowest_iteration_path", "")
    if slowest:
        parts.append(f"最遅反復 {slowest}")
    return " / ".join(str(part) for part in parts if str(part).strip())


def _csv_summary_from_payload(payload: Any) -> CsvTableSummary | None:
    if isinstance(payload, CsvTableSummary):
        return payload
    if not isinstance(payload, Mapping):
        return None
    try:
        row_count = int(payload.get("row_count", 0) or 0)
    except (TypeError, ValueError):
        row_count = 0
    minimums = payload.get("minimums", {})
    maximums = payload.get("maximums", {})
    return CsvTableSummary(
        path=str(payload.get("path", "") or ""),
        headers=[str(item) for item in list(payload.get("headers", []))],
        row_count=max(0, row_count),
        numeric_fields=[str(item) for item in list(payload.get("numeric_fields", []))],
        minimums={str(key): float(value) for key, value in dict(minimums if isinstance(minimums, Mapping) else {}).items()},
        maximums={str(key): float(value) for key, value in dict(maximums if isinstance(maximums, Mapping) else {}).items()},
    )


def _csv_page_from_payload(payload: Any, *, default_headers: list[str]) -> ResultTablePage | None:
    if isinstance(payload, ResultTablePage):
        return payload
    if not isinstance(payload, Mapping):
        return None
    rows = [dict(row) for row in list(payload.get("rows", [])) if isinstance(row, Mapping)]
    headers = [str(item) for item in list(payload.get("headers", []))] or list(default_headers)
    try:
        page_index = int(payload.get("page_index", 0) or 0)
        page_size = int(payload.get("page_size", DEFAULT_RESULT_TABLE_PAGE_SIZE) or DEFAULT_RESULT_TABLE_PAGE_SIZE)
        total_rows = int(payload.get("total_rows", len(rows)) or 0)
        page_count = int(payload.get("page_count", 1) or 1)
        start_row = int(payload.get("start_row", 0) or 0)
        end_row = int(payload.get("end_row", len(rows)) or 0)
    except (TypeError, ValueError):
        return None
    return ResultTablePage(
        rows=rows,
        headers=headers,
        page_index=max(0, page_index),
        page_size=max(1, page_size),
        total_rows=max(0, total_rows),
        page_count=max(1, page_count),
        start_row=max(0, start_row),
        end_row=max(0, end_row),
    )

POST_REPORT_CONTROLLER_METHODS = (
    "show_report_note",
    "refresh_workspace_dashboard",
    "apply_default_customization_profile",
    "show_version_info",
    "show_last_run_path",
    "_load_result_summary",
    "refresh_result_judgment_summary",
    "show_default_result_visual_if_available",
    "_default_result_visual_kind",
    "show_latest_report_if_available",
    "result_stage_changed",
    "clear_post_state",
    "load_result_table_async",
    "_post_table_finished",
    "_apply_post_table_result",
    "_post_job_failed",
    "load_result_table",
    "reload_selected_result_component_async",
    "reload_selected_result_component",
    "load_compare_result",
    "compare_current_result_to_previous_async",
    "compare_current_result_to_baseline_async",
    "compare_current_result_to_design_async",
    "_compare_result_pair_async",
    "_current_results_dir",
    "_result_dir_from_candidate",
    "_previous_results_dir",
    "export_result_table_csv",
    "_start_post_export_job",
    "_post_export_finished",
    "_post_export_job_failed",
    "verify_post_view_render",
    "save_post_baseline_async",
    "save_post_baseline",
    "compare_post_to_baseline_async",
    "compare_post_to_baseline",
    "write_post_image_diff_ci_job",
    "add_report_text_block",
    "_set_report_page_item",
    "refresh_report_canvas",
    "apply_report_canvas_positions",
    "nudge_selected_report_block",
    "add_current_post_to_report_page",
    "_report_page_specs",
    "build_report_wysiwyg_html",
    "preview_report_page_layout",
    "export_scene_pdf_async",
    "export_scene_pdf",
    "_add_post_snapshot_tab",
    "duplicate_post_view_async",
    "duplicate_post_view",
    "show_srm_post_async",
    "_srm_post_options",
    "_srm_post_finished",
    "show_srm_post",
    "export_srm_slip_csv",
    "build_srm_candidate_report",
    "show_srm_summary",
    "preview_stage_report",
    "build_selected_report_async",
    "audit_post_report_async",
    "build_selected_report",
    "export_calculation_report_pdf_async",
    "export_calculation_report_pdf",
    "_set_result_view_file",
    "_srm_post_rows",
    "_srm_post_summary_text",
    "_set_result_component_without_reload",
    "_set_result_table_component_without_reload",
    "_populate_result_table",
    "_populate_result_table_from_csv",
    "result_table_previous_page",
    "result_table_next_page",
    "_render_result_table_page",
    "_result_table_page_count",
    "_can_stream_result_table",
    "_refresh_result_component_options",
)


def post_report_controller_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.gui.post_report_controller.v1",
        "method_count": len(POST_REPORT_CONTROLLER_METHODS),
        "methods": list(POST_REPORT_CONTROLLER_METHODS),
        "owner_boundary": "controller mutates owner result/report widgets; MainWindow delegates Post and report workflow operations",
    }


def show_report_note(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    self.report_summary.setText(self.ui_message("status.report.note"))
    self.tabs.setCurrentWidget(self.result_view)
    self.panel_stack.setCurrentWidget(self.panel_pages["report"])
    self.append_log(self.ui_message("log.report.selected"))

def refresh_workspace_dashboard(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    from geofem_app.workspace_management import write_workspace_dashboard

    out = self.project_root / "runs" / "workspace_dashboard"
    paths = write_workspace_dashboard(self.project_root, out)
    self.result_view.setPlainText(yaml.safe_dump(paths, allow_unicode=True, sort_keys=False))
    self.tabs.setCurrentWidget(self.result_view)
    if "report" in self.panel_pages:
        self.panel_stack.setCurrentWidget(self.panel_pages["report"])
    self.statusBar().showMessage(self.ui_message("status.workspace.updated", path=paths.get("html", "")))
    self.append_log(self.ui_message("log.workspace.updated", path=paths.get("json", "")))

def apply_default_customization_profile(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    from geofem_app.customization import apply_organization_profile, default_organization_profile, load_organization_profile, write_customization_artifacts

    profile_dir = self.project_root / "templates" / "organization_profile"
    profile_path = self.project_root / "templates" / "organization_profile.yaml"
    if profile_path.exists():
        profile = load_organization_profile(profile_path)
    else:
        profile = default_organization_profile()
    template_id = str(profile.get("defaults", {}).get("project_template", "geofem_review")) if isinstance(profile.get("defaults", {}), Mapping) else "geofem_review"
    self.cfg = apply_organization_profile(self.cfg, profile, template_id=template_id)
    paths = write_customization_artifacts(profile_dir, profile=profile, template_id=template_id)
    if not profile_path.exists():
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths["profile_yaml"], profile_path)
    self._after_form_change("組織プロファイルを適用しました")
    self.result_view.setPlainText(yaml.safe_dump({"template_id": template_id, "artifacts": paths}, allow_unicode=True, sort_keys=False))
    self.tabs.setCurrentWidget(self.result_view)
    if "report" in self.panel_pages:
        self.panel_stack.setCurrentWidget(self.panel_pages["report"])
    self.statusBar().showMessage(self.ui_message("status.customization.applied", path=paths.get("profile_yaml", "")))
    self.append_log(self.ui_message("log.customization.applied", path=paths.get("profile_yaml", "")))

def show_version_info(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    info = build_version_info(project_root=self.project_root, include_gui=True)
    locale = str(getattr(self, "gui_locale", "ja") or "ja")
    QMessageBox.information(
        self,
        gui_surface_message("dialog.version.title", locale=locale),
        _version_info_popup_text(info, locale=locale),
    )
    status_bar = self.statusBar() if hasattr(self, "statusBar") else None
    if status_bar is not None:
        status_bar.showMessage(_version_info_status_text(info, locale=locale))
    self.append_log(self.ui_message("log.version.shown"))


def _version_info_popup_text(info: Mapping[str, Any], *, locale: str) -> str:
    product = info.get("product", {}) if isinstance(info.get("product", {}), Mapping) else {}
    summary = info.get("dependency_summary", {}) if isinstance(info.get("dependency_summary", {}), Mapping) else {}
    tools = info.get("external_tool_summary", {}) if isinstance(info.get("external_tool_summary", {}), Mapping) else {}
    fonts = info.get("font_summary", {}) if isinstance(info.get("font_summary", {}), Mapping) else {}
    if locale == "en":
        return "\n".join(
            [
                str(product.get("name", "GeoFEM 2D")),
                f"Version: {product.get('version', '')}",
                f"Input schema: {product.get('input_config_schema', '')}",
                f"Python: {product.get('python', '')}",
                f"Platform: {product.get('platform', '')}",
                f"Dependencies: {summary.get('installed_count', 0)} / {summary.get('dependency_count', 0)} installed",
                f"Missing required dependencies: {summary.get('missing_required_count', 0)}",
                f"External tools available: {tools.get('available_count', 0)} / {tools.get('tool_count', 0)}",
                f"Preferred GUI fonts: {fonts.get('available_preferred_count', 0)}",
            ]
        )
    return "\n".join(
        [
            str(product.get("name", "GeoFEM 2D")),
            f"バージョン: {product.get('version', '')}",
            f"入力スキーマ: {product.get('input_config_schema', '')}",
            f"Python: {product.get('python', '')}",
            f"プラットフォーム: {product.get('platform', '')}",
            f"依存パッケージ: {summary.get('installed_count', 0)} / {summary.get('dependency_count', 0)} 導入済み",
            f"未導入必須: {summary.get('missing_required_count', 0)}",
            f"外部ツール利用可能: {tools.get('available_count', 0)} / {tools.get('tool_count', 0)}",
            f"優先GUIフォント: {fonts.get('available_preferred_count', 0)}",
        ]
    )


def _version_info_status_text(info: Mapping[str, Any], *, locale: str) -> str:
    product = info.get("product", {}) if isinstance(info.get("product", {}), Mapping) else {}
    summary = info.get("dependency_summary", {}) if isinstance(info.get("dependency_summary", {}), Mapping) else {}
    if locale == "en":
        return (
            f"Version Info: GeoFEM {product.get('version', '')} / "
            f"dependencies={summary.get('dependency_count', 0)} / missing required={summary.get('missing_required_count', 0)}"
        )
    return (
        f"バージョン情報: GeoFEM {product.get('version', '')} / "
        f"依存={summary.get('dependency_count', 0)} / 未導入必須={summary.get('missing_required_count', 0)}"
    )

def show_last_run_path(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    run_dir = self._active_run_dir() if hasattr(self, "_active_run_dir") else self.last_run_dir
    summary = self._current_summary_path() if hasattr(self, "_current_summary_path") else None
    if run_dir is None or summary is None:
        self.results_summary.setText("まだ解析結果がありません。")
        return
    detail_mode = getattr(self, "_show_internal_representation", lambda: False)()
    self.results_summary.setText(
        str(run_dir)
        if detail_mode
        else ("Loaded analysis results." if str(getattr(self, "gui_locale", "ja")).startswith("en") else "解析結果を読み込みました。")
    )
    self.results_summary.setToolTip(str(run_dir) if detail_mode else "")
    self._load_result_summary(summary)

def _load_result_summary(owner: Any, qt: Mapping[str, Any], summary: Path) -> None:
    self = owner
    self._loaded_result_summary_path = Path(summary)
    if getattr(self, "results_summary", None) is not None and not self.results_summary.text().strip():
        self.results_summary.setText("判定サマリを読み込みました。" if str(getattr(self, "gui_locale", "ja")).startswith("ja") else "Judgment summary loaded.")
        self.results_summary.setToolTip(
            str(summary)
            if getattr(self, "_show_internal_representation", lambda: False)()
            else ("Result summary loaded." if str(getattr(self, "gui_locale", "ja")).startswith("en") else "解析結果サマリを読み込みました。")
        )
    self.result_stage_dirs = self._stage_dirs_from_summary(summary)
    self.result_stage_dir = self.result_stage_dirs[-1] if self.result_stage_dirs else None
    selector = getattr(self, "result_stage_selector", None)
    if selector is not None:
        selector.blockSignals(True)
        selector.clear()
        for index, stage_dir in enumerate(self.result_stage_dirs):
            label = self._result_stage_display_name(stage_dir, index) if hasattr(self, "_result_stage_display_name") else stage_dir.name
            selector.addItem(label, str(stage_dir))
        if self.result_stage_dirs:
            selector.setCurrentIndex(len(self.result_stage_dirs) - 1)
        selector.blockSignals(False)
    refresh_result_judgment_summary(owner, qt, summary_path=summary)
    if hasattr(self, "_refresh_result_action_states"):
        self._refresh_result_action_states()
    if hasattr(self, "refresh_workflow_guidance"):
        self.refresh_workflow_guidance()


def refresh_result_judgment_summary(
    owner: Any,
    qt: Mapping[str, Any],
    *,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Refresh the first-view judgment summary without materializing Post rows."""

    self = owner
    data: Mapping[str, Any] | None = None
    path = Path(summary_path) if summary_path is not None else self._current_summary_path()
    if path is None:
        loaded_path = getattr(self, "_loaded_result_summary_path", None)
        path = Path(loaded_path) if loaded_path is not None else None
    if path is not None and path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            data = loaded
    model = build_result_judgment_summary(
        data,
        stage_dir=getattr(self, "result_stage_dir", None),
        locale=str(getattr(self, "gui_locale", "ja")),
    )
    self.result_judgment_model = model
    available = str(model.get("kind", "empty")) != "empty"
    if getattr(self, "result_empty_state", None) is not None:
        self.result_empty_state.setVisible(not available)
    panel = getattr(self, "result_judgment_panel", None)
    if panel is not None:
        panel.setVisible(available)
        panel.setProperty("resultKind", str(model.get("kind", "empty")))
        panel.setProperty("resultTone", str(model.get("tone", "neutral")))
    headline = getattr(self, "result_judgment_headline", None)
    if headline is not None:
        headline.setText(str(model.get("headline", "")))
        severity = {"success": "ok", "warning": "warning", "danger": "error"}.get(str(model.get("tone", "")), "")
        headline.setProperty("severity", severity)
        headline.setAccessibleName(str(model.get("headline", "")))
        style = headline.style()
        style.unpolish(headline)
        style.polish(headline)
    status = getattr(self, "result_judgment_status", None)
    if status is not None:
        status.setText(str(model.get("status", "")))
    detail = getattr(self, "result_judgment_detail", None)
    if detail is not None:
        detail.setText(str(model.get("detail", "")))
        detail.setVisible(bool(detail.text()))
    metric_rows = list(model.get("metrics", [])) if isinstance(model.get("metrics", []), list) else []
    captions = list(getattr(self, "result_judgment_metric_captions", []))
    values = list(getattr(self, "result_judgment_metric_values", []))
    for index, (caption, value) in enumerate(zip(captions, values)):
        if index < len(metric_rows) and isinstance(metric_rows[index], (list, tuple)) and len(metric_rows[index]) >= 2:
            caption.setText(str(metric_rows[index][0]))
            value.setText(str(metric_rows[index][1]))
            caption.parentWidget().setVisible(True)
        else:
            caption.setText("")
            value.setText("")
            caption.parentWidget().setVisible(False)
    warning = getattr(self, "result_judgment_warning", None)
    if warning is not None:
        warning.setText(str(model.get("warning", "")))
        warning.setVisible(bool(warning.text()))
    srm_button = getattr(self, "result_primary_srm_button", None)
    if srm_button is not None:
        srm_button.setVisible(available and str(model.get("kind", "")) == "srm")
    tabs = getattr(self, "results_tabs", None)
    if tabs is not None and tabs.count() > 2:
        tabs.setTabVisible(2, available and str(model.get("kind", "")) == "srm")
    return model

def show_default_result_visual_if_available(owner: Any, qt: Mapping[str, Any], *, force: bool = False) -> str:
    """Open the first useful Post figure for the current completed run."""

    self = owner
    stage_dir = self._latest_stage_dir()
    if stage_dir is None:
        summary = self._current_summary_path()
        if summary is not None:
            self._load_result_summary(summary)
            stage_dir = self._latest_stage_dir()
    if stage_dir is None:
        if getattr(self, "results_summary", None) is not None:
            self.results_summary.setText(
                "No analysis results are available. Run the analysis before opening result views."
                if str(getattr(self, "gui_locale", "ja")).startswith("en")
                else "解析結果がまだありません。解析実行後に結果図へ移動できます。"
            )
        return ""
    kind = self._default_result_visual_kind(stage_dir)
    if not kind:
        if getattr(self, "results_summary", None) is not None:
            self.results_summary.setText(f"表示できるPost図用CSVがありません: {stage_dir}")
        return ""
    path = result_table_path(kind, stage_dir=stage_dir, summary_path=self._current_summary_path() if is_root_result_kind(kind) else None)
    signature = f"{stage_dir.resolve()}|{kind}|{path.stat().st_mtime_ns if path.exists() else 0}"
    if not force and getattr(self, "_last_auto_result_visual_signature", "") == signature and getattr(self, "post_mode", "none") not in {"none", "table"}:
        return kind
    self._last_auto_result_visual_signature = signature
    self.load_result_table_async(kind)
    return kind

def _default_result_visual_kind(owner: Any, qt: Mapping[str, Any], stage_dir: Path) -> str:
    """Choose the result figure that best answers a first Post view request."""

    stage = Path(stage_dir)
    candidates = (
        ("element_stress", "element_stress.csv"),
        ("displacement_vectors", "displacements.csv"),
        ("displacement_contour", "displacements.csv"),
        ("pore_pressure", "pore_pressure.csv"),
        ("safety_factor", "element_stress.csv"),
    )
    for kind, filename in candidates:
        path = stage / filename
        if path.exists() and path.stat().st_size > 0:
            return kind
    return ""

def _summary_has_deferred_report_artifacts(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, Mapping):
        return False
    for key in ("calculation_report", "standard_report", "result_view_index"):
        value = data.get(key)
        if isinstance(value, Mapping) and bool(value.get("deferred", False)):
            return True
    output = data.get("output_generation", {})
    return isinstance(output, Mapping) and bool(output.get("lazy_reports", False)) and not (summary_path.parent / "calculation_report.html").exists()

def materialize_deferred_reports_async(owner: Any, qt: Mapping[str, Any], *, include: tuple[str, ...] | None = None) -> bool:
    """Generate GUI-deferred report artifacts from persisted run outputs."""

    self = owner
    QMessageBox = qt["QMessageBox"]
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    run_dir = self._active_run_dir() if hasattr(self, "_active_run_dir") else self.last_run_dir
    if results_dir is None or run_dir is None:
        _notify_information(self, QMessageBox, "No analysis result directory is available.")
        return False
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        _notify_information(self, QMessageBox, f"summary.json does not exist: {summary_path}")
        return False
    if not _summary_has_deferred_report_artifacts(summary_path):
        return False
    input_path = run_dir / "input.yaml"
    if getattr(self, "report_summary", None) is not None:
        self.report_summary.setText(f"帳票を生成中: {results_dir}")
    if hasattr(self, "append_log"):
        self.append_log(f"[GUI] Deferred report generation started: {results_dir}")
    return self._start_post_export_job(
        "materialize_reports",
        results_dir,
        lambda results_dir=results_dir, input_path=input_path, include=include: {
            "operation": "materialize_reports",
            **write_deferred_run_artifacts_from_files(results_dir, input_path=input_path, include=include),
        },
        metadata={"include": ",".join(include or ())},
    )

def _materialize_and_copy_report_pdf_snapshot(
    *,
    results_dir: str | Path,
    input_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    results = Path(results_dir)
    materialized = write_deferred_run_artifacts_from_files(
        results,
        input_path=input_path,
        include=("calculation_report", "standard_report", "result_view_index"),
    )
    copied = copy_report_pdf_snapshot(results / "calculation_report.pdf", destination, manifest=results / "calculation_report_manifest.json")
    copied["operation"] = "materialize_and_copy_report_pdf"
    copied["materialized"] = materialized
    return copied

def show_latest_report_if_available(owner: Any, qt: Mapping[str, Any]) -> bool:
    """Display the latest generated calculation report when one exists."""

    self = owner
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    if results_dir is None:
        if getattr(self, "report_summary", None) is not None:
            self.report_summary.setText(
                "No analysis results are available. Run the analysis before opening reports."
                if str(getattr(self, "gui_locale", "ja")).startswith("en")
                else "解析結果がまだありません。解析実行後に帳票へ移動できます。"
            )
        return False
    for name in ("calculation_report.html", "standard_report.html", "result_view_index.html", "gui_report.html", "report_manifest.json"):
        path = results_dir / name
        if path.exists():
            self._set_result_view_file(path)
            if getattr(self, "report_summary", None) is not None:
                self.report_summary.setText(f"帳票/結果プレビュー: {path}")
            return True
    if _summary_has_deferred_report_artifacts(results_dir / "summary.json"):
        if getattr(self, "report_summary", None) is not None:
            self.report_summary.setText(f"Deferred report artifacts are available. Use the report preview/export button to generate them: {results_dir}")
        return False
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        self._set_result_view_file(summary_path)
        if getattr(self, "report_summary", None) is not None:
            self.report_summary.setText(f"帳票/結果プレビュー: {summary_path}")
        return True
    if getattr(self, "report_summary", None) is not None:
        self.report_summary.setText(f"帳票ファイルがまだありません: {results_dir}")
    return False

def result_stage_changed(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    selector = getattr(self, "result_stage_selector", None)
    if selector is None:
        return
    raw = selector.currentData()
    if raw:
        self.result_stage_dir = Path(str(raw))
        label = self._result_stage_display_name(self.result_stage_dir, selector.currentIndex()) if hasattr(self, "_result_stage_display_name") else self.result_stage_dir.name
        self.results_summary.setText(f"結果ステージ: {label}")
        refresh_result_judgment_summary(owner, qt)
        self.clear_post_state()

def clear_post_state(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    self.result_displacements = {}
    self.result_element_values = {}
    self.result_node_values = {}
    self.result_distribution = []
    self.compare_element_values = {}
    self.srm_slip_candidates = []
    self.srm_trial_rows = []
    if hasattr(self, "srm_trial_table"):
        self.srm_trial_table.setRowCount(0)
    if hasattr(self, "srm_slip_table"):
        self.srm_slip_table.setRowCount(0)
    if hasattr(self, "srm_candidate_compare_table"):
        self.srm_candidate_compare_table.setRowCount(0)
    if hasattr(self, "srm_local_fl_table"):
        self.srm_local_fl_table.setRowCount(0)
    if hasattr(self, "srm_post_summary"):
        self.srm_post_summary.setText("SRM専用Postは解析後に更新します。")
    self.post_mode = "none"
    self.update_preview()
    if hasattr(self, "_sync_aux_result_controls"):
        self._sync_aux_result_controls()

def _displacement_map_from_rows(rows: list[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for row in rows:
        node_id = str(row.get("node_id", "") or "")
        if not node_id:
            continue
        try:
            ux = float(row.get("ux", 0.0) or 0.0)
            uy = float(row.get("uy", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        out[node_id] = (ux, uy)
    return out

def _optional_stage_displacements(owner: Any, stage_dir: Path | None) -> dict[str, tuple[float, float]]:
    if stage_dir is None:
        return {}
    path = stage_dir / "displacements.csv"
    if not path.exists():
        return {}
    try:
        return _displacement_map_from_rows(owner._read_csv_rows(path))
    except Exception:
        return {}

def _numeric_result_headers(headers: list[str]) -> list[str]:
    ignored = {"node_id", "element_id", "id", "type", "material", "integration", "state", "active", "active_set", "dof", "x", "y", "z"}
    return [header for header in headers if header not in ignored]

def _resolve_result_field(rows: list[dict[str, str]], requested: str, preferred: tuple[str, ...]) -> str:
    headers = list(rows[0]) if rows else []
    numeric = _numeric_result_headers(headers)
    requested = str(requested or "").strip()
    if requested in numeric:
        return requested
    for field in preferred:
        if field in numeric:
            return field
    return numeric[0] if numeric else requested

def load_result_table_async(owner: Any, qt: Mapping[str, Any], kind: str) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    QtCallableRunner = qt["QtCallableRunner"]
    if self._post_job_id:
        self.statusBar().showMessage("Post処理ジョブを実行中です。")
        return
    stage_dir = self._latest_stage_dir()
    if stage_dir is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    if self._can_stream_result_table(kind):
        self.load_result_table(kind)
        return
    if is_root_result_kind(kind):
        summary_path = self._current_summary_path()
        if summary_path is None:
            _notify_information(self, QMessageBox, "解析結果サマリがありません。")
            return
    else:
        summary_path = None
    try:
        path = result_table_path(kind, stage_dir=stage_dir, summary_path=summary_path)
    except KeyError:
        _notify_information(self, QMessageBox, "選択した結果表示には対応していません。")
        return
    if not path.exists():
        _notify_information(self, QMessageBox, "選択した結果データがありません。")
        return
    cfg_snapshot = copy.deepcopy(self.cfg)
    result_component = self._combo_value(self.result_component, "q")
    table_component = self.result_table_component.currentText()
    active_run_dir = self._active_run_dir() if hasattr(self, "_active_run_dir") else self.last_run_dir
    last_run_dir = str(active_run_dir or "")
    stage_dir_text = str(stage_dir)
    job_id = self.gui_jobs.start_job("post", target=str(path), metadata={"kind": kind})
    self._post_job_id = job_id
    runner = QtCallableRunner(
        job_id,
        lambda: load_post_table_snapshot(
            type(self),
            cfg_snapshot,
            path=path,
            kind=kind,
            result_component=result_component,
            table_component=table_component,
            last_run_dir=last_run_dir,
            result_stage_dir=stage_dir_text,
            page_size=getattr(self, "result_table_page_size", DEFAULT_RESULT_TABLE_PAGE_SIZE),
        ),
    )
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._post_table_finished)
    runner.signals.failed.connect(self._post_job_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage(f"Post結果をバックグラウンド読込中: {path.name}")

def _post_table_finished(owner: Any, qt: Mapping[str, Any], job_id: str, result_obj: Any) -> None:
    self = owner
    if job_id != self._post_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    result = dict(result_obj or {})
    rows = [dict(row) for row in list(result.get("rows", [])) if isinstance(row, Mapping)]
    summary = _csv_summary_from_payload(result.get("table_summary"))
    total_rows = summary.row_count if summary is not None else len(rows)
    self._complete_gui_worker_job(job_id, status="finished", message=f"{len(rows)}/{total_rows} rows")
    self._post_job_id = ""
    self._apply_post_table_result(result, rows)
    self.append_log(f"[GUI] Post結果読込完了: {result.get('path', '')} rows={len(rows)}/{total_rows}")

def _apply_post_table_result(owner: Any, qt: Mapping[str, Any], result: Mapping[str, Any], rows: list[dict[str, str]]) -> None:
    self = owner
    self.current_result_kind = str(result.get("kind", "") or getattr(self, "current_result_kind", ""))
    component_store = result.get("component_store", {})
    self._post_component_store = dict(component_store) if isinstance(component_store, Mapping) else {}
    summary = _csv_summary_from_payload(result.get("table_summary"))
    page = _csv_page_from_payload(result.get("table_page"), default_headers=(summary.headers if summary is not None else list(rows[0]) if rows else []))
    path_text = str(result.get("path", "") or "")
    if summary is not None and path_text:
        self._populate_result_table_from_csv(Path(path_text), summary, page=page)
    else:
        self._populate_result_table(rows)
    self.result_displacements = {
        str(key): (float(value[0]), float(value[1]))
        for key, value in self._mapping(result.get("result_displacements", {})).items()
        if isinstance(value, (list, tuple)) and len(value) >= 2
    }
    self.result_element_values = {
        str(key): float(value)
        for key, value in self._mapping(result.get("result_element_values", {})).items()
    }
    self.result_node_values = {
        str(key): float(value)
        for key, value in self._mapping(result.get("result_node_values", {})).items()
    }
    self.result_distribution = [
        (float(item[0]), float(item[1]))
        for item in list(result.get("result_distribution", []))
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    self.post_component = str(result.get("post_component", "") or self.post_component)
    self.post_mode = str(result.get("post_mode", "table") or "table")
    display_component = str(result.get("display_component", "") or "")
    if display_component:
        self._set_result_component_without_reload(display_component)
    colormap = str(result.get("colormap", "") or "")
    if colormap:
        self._set_colormap_without_reload(colormap)
    if self.post_mode in {"contour", "node_contour", "vector", "deformed", "plastic", "srm", "distribution"}:
        self.tabs.setCurrentIndex(0)
    else:
        self.tabs.setCurrentWidget(self.result_table)
    self.update_preview()
    if hasattr(self, "_sync_aux_result_controls"):
        self._sync_aux_result_controls()
    perf_summary = _performance_result_summary(Path(path_text)) if self.current_result_kind == "performance" and path_text else ""
    suffix = f" / {perf_summary}" if perf_summary else ""
    self.results_summary.setText(f"表示中: {path_text}{suffix}")

def _post_job_failed(owner: Any, qt: Mapping[str, Any], job_id: str, message: str) -> None:
    self = owner
    if job_id != self._post_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    manifest = self._write_gui_worker_failure_manifest(job_id, message)
    self._complete_gui_worker_job(job_id, status="failed", message=message)
    self._post_job_id = ""
    suffix = f" manifest={manifest}" if manifest is not None else ""
    self.statusBar().showMessage(f"Post処理ジョブ失敗: {message}")
    self.append_log(f"[GUI] Post処理ジョブ失敗: {message}{suffix}")

def load_result_table(owner: Any, qt: Mapping[str, Any], kind: str) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    self.current_result_kind = str(kind)
    stage_dir = self._latest_stage_dir()
    if stage_dir is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    if is_root_result_kind(kind):
        summary_path = self._current_summary_path()
        if summary_path is None:
            _notify_information(self, QMessageBox, "解析結果サマリがありません。")
            return
    else:
        summary_path = None
    try:
        path = result_table_path(kind, stage_dir=stage_dir, summary_path=summary_path)
    except KeyError:
        _notify_information(self, QMessageBox, "選択した結果表示には対応していません。")
        return
    if not path.exists():
        _notify_information(self, QMessageBox, "選択した結果データがありません。")
        return
    if self._can_stream_result_table(kind):
        summary = summarize_csv_table(path)
        self._populate_result_table_from_csv(path, summary)
        self.result_displacements = {}
        self.result_element_values = {}
        self.result_node_values = {}
        self.result_distribution = []
        self.post_mode = "table"
        self.tabs.setCurrentWidget(self.result_table)
        self.update_preview()
        perf_summary = _performance_result_summary(path) if kind == "performance" else ""
        suffix = f" / {perf_summary}" if perf_summary else ""
        self.results_summary.setText(f"表示中: {path} ({summary.row_count}行, ページ読込){suffix}")
        return
    rows = self._read_csv_rows(path)
    if kind == "safety_factor":
        rows = self._with_safety_factor_rows(rows)
    self._populate_result_table(rows)
    if kind in {"displacements", "displacement_contour", "displacement_vectors"}:
        self.result_displacements = _displacement_map_from_rows(rows)
        component = self.result_table_component.currentText() if self.result_table_component.currentText() in {"ux", "uy", "u_norm", "settlement"} else "u_norm"
        self.result_node_values = self._node_rows_to_values(rows, component)
        self.result_element_values = {}
        self.post_component = component
        if kind == "displacement_contour":
            self.post_mode = "node_contour"
        else:
            self.post_mode = "vector" if kind == "displacement_vectors" else "deformed"
    elif kind in {"element_stress", "plastic"}:
        field = "plastic" if kind == "plastic" else _resolve_result_field(rows, self._combo_value(self.result_component, "q"), ("q", "p", "sigma_x", "sigma_y"))
        if kind != "plastic":
            self._set_result_component_without_reload(field)
        self.post_component = field
        self.result_displacements = _optional_stage_displacements(self, stage_dir)
        self.result_element_values = self._element_rows_to_values(rows, field)
        self.result_node_values = {}
        self.post_mode = "plastic" if kind == "plastic" else "contour"
    elif kind == "safety_factor":
        self.result_displacements = _optional_stage_displacements(self, stage_dir)
        self.result_node_values = {}
        self.result_element_values = self._element_rows_to_values(rows, "FL")
        self.post_component = "FL"
        self.post_mode = "contour"
        self._set_result_component_without_reload("FL")
        self._set_colormap_without_reload("Safety FL")
    elif kind == "pore_pressure":
        self.result_displacements = _optional_stage_displacements(self, stage_dir)
        self.result_element_values = {}
        self.result_node_values = self._node_rows_to_values(rows, "pore_pressure")
        self.post_component = "pore_pressure"
        self.post_mode = "node_contour"
    elif kind == "riks_path":
        self.result_displacements = {}
        self.result_element_values = {}
        self.result_node_values = {}
        self.result_distribution = self._distribution_from_rows(rows, preferred=("lambda", "load_factor", "arc_length", "step"))
        self.post_component = self.result_table_component.currentText() or "Riks path"
        self.post_mode = "distribution"
    elif kind == "analysis_log":
        self.result_displacements = {}
        self.result_element_values = {}
        self.result_node_values = {}
        self.result_distribution = self._distribution_from_rows(rows, preferred=("residual_norm", "pressure_residual_norm", "iteration", "step"))
        self.post_component = self.result_table_component.currentText() or "residual_norm"
        self.post_mode = "distribution" if self.result_distribution else "table"
    elif kind == "performance":
        self.result_displacements = {}
        self.result_element_values = {}
        self.result_node_values = {}
        self.result_distribution = self._distribution_from_rows(rows, preferred=("elapsed_seconds", "matrix_nnz", "solver_iterations"))
        self.post_component = self.result_table_component.currentText() or "elapsed_seconds"
        self.post_mode = "distribution" if self.result_distribution else "table"
    else:
        self.result_displacements = {}
        self.result_element_values = {}
        self.result_node_values = {}
        self.post_mode = "table"
    if self.post_mode in {"contour", "node_contour", "vector", "deformed", "plastic", "srm", "distribution"}:
        self.tabs.setCurrentIndex(0)
    else:
        self.tabs.setCurrentWidget(self.result_table)
    self.update_preview()
    if hasattr(self, "_sync_aux_result_controls"):
        self._sync_aux_result_controls()
    perf_summary = _performance_result_summary(path) if kind == "performance" else ""
    suffix = f" / {perf_summary}" if perf_summary else ""
    self.results_summary.setText(f"表示中: {path}{suffix}")

def reload_selected_result_component_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    if self.post_mode in {"node_contour", "contour", "plastic"} and _load_compact_post_component_async(self, qt):
        return
    if self.post_mode == "srm":
        self.show_srm_post_async(update_only=True)
        return
    if self.post_mode not in {"contour", "plastic"}:
        return
    field = self._combo_value(self.result_component, "q")
    if field in {"FL", "safety_factor", "factor_of_safety", "local_safety_factor"}:
        self.load_result_table_async("safety_factor")
    elif field == "plastic":
        self.load_result_table_async("plastic")
    else:
        self.load_result_table_async("element_stress")


def _load_compact_post_component_async(owner: Any, qt: Mapping[str, Any]) -> bool:
    self = owner
    store = getattr(self, "_post_component_store", {})
    if not isinstance(store, Mapping) or not store:
        return False
    columns = store.get("columns", {})
    if not isinstance(columns, Mapping) or not columns:
        return False
    if self.post_mode == "node_contour":
        requested = self.result_table_component.currentText() or self.post_component
        preferred = ("u_norm", "ux", "uy", "pore_pressure")
    else:
        requested = self._combo_value(self.result_component, "q")
        preferred = ("q", "p", "sigma_x", "sigma_y")
    normalized = {
        "safety_factor": "FL",
        "factor_of_safety": "FL",
        "local_safety_factor": "FL",
    }.get(str(requested), str(requested))
    if normalized not in columns:
        return False
    if self._post_job_id:
        self.statusBar().showMessage("Post component job is already running.")
        return True
    QtCallableRunner = qt["QtCallableRunner"]
    job_id = self.gui_jobs.start_job(
        "post",
        target=str(store.get("path", "") or ""),
        metadata={"kind": "component_materialize", "component": normalized},
    )
    self._post_job_id = job_id
    runner = QtCallableRunner(
        job_id,
        lambda: materialize_post_component_snapshot(store, normalized, preferred=preferred),
    )
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(lambda finished_id, payload: _post_component_finished(self, qt, finished_id, payload))
    runner.signals.failed.connect(self._post_job_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage(f"Post component loading: {normalized}")
    return True


def _post_component_finished(owner: Any, qt: Mapping[str, Any], job_id: str, result_obj: Any) -> None:
    self = owner
    if job_id != self._post_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    result = dict(result_obj or {})
    field = str(result.get("field", "") or "")
    self._complete_gui_worker_job(job_id, status="finished", message=f"component={field}")
    self._post_job_id = ""
    element_values = result.get("result_element_values", {})
    node_values = result.get("result_node_values", {})
    if isinstance(element_values, Mapping) and element_values:
        self.result_element_values = {str(key): float(value) for key, value in element_values.items()}
        self.result_node_values = {}
    if isinstance(node_values, Mapping) and node_values:
        self.result_node_values = {str(key): float(value) for key, value in node_values.items()}
        self.result_element_values = {}
    self.post_component = field or self.post_component
    if field:
        if str(result.get("id_field", "")) == "element_id":
            self._set_result_component_without_reload(field)
        if field == "FL":
            self._set_colormap_without_reload("Safety FL")
        if self.post_mode != "node_contour":
            self.post_mode = "plastic" if field == "plastic" else "contour"
    self.update_preview()
    self.statusBar().showMessage(f"Post component ready: {field}", 3000)
    self.append_log(
        f"[GUI] Post component cache hit: field={field} values={int(result.get('value_count', 0) or 0)}"
    )

def reload_selected_result_component(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    if self.post_mode == "node_contour" and self.result_rows:
        field = self.result_table_component.currentText() or self.post_component
        self.result_node_values = self._node_rows_to_values(self.result_rows, field)
        self.post_component = field
        self.update_preview()
        return
    if self.post_mode not in {"contour", "plastic", "srm"}:
        return
    stage_dir = self._latest_stage_dir()
    if stage_dir is None:
        return
    path = stage_dir / "element_stress.csv"
    if not path.exists():
        return
    rows = self._read_csv_rows(path)
    field = self._combo_value(self.result_component, "q")
    if field in {"FL", "safety_factor", "factor_of_safety", "local_safety_factor"}:
        rows = self._with_safety_factor_rows(rows)
        field = "FL"
        self._set_result_component_without_reload("FL")
        self._set_colormap_without_reload("Safety FL")
    else:
        field = _resolve_result_field(rows, field, ("q", "p", "sigma_x", "sigma_y"))
        self._set_result_component_without_reload(field)
    self.post_component = field
    keep_srm = self.post_mode == "srm"
    self.post_mode = "srm" if keep_srm else ("plastic" if field == "plastic" else "contour")
    self.result_displacements = _optional_stage_displacements(self, stage_dir)
    self.result_element_values = self._element_rows_to_values(rows, field)
    self._populate_result_table(rows)
    if keep_srm:
        self.srm_slip_candidates = self._estimate_srm_slip_candidates(rows)
        self._populate_srm_slip_table()
    self.update_preview()

def load_compare_result(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    path, _ = QFileDialog.getOpenFileName(self, "比較するelement_stress.csvを選択", str(self.project_root), "CSV (*.csv)")
    if not path:
        return
    try:
        rows = self._read_csv_rows(Path(path))
        field = self._combo_value(self.result_component, "q")
        if field in {"FL", "safety_factor", "factor_of_safety", "local_safety_factor"}:
            rows = self._with_safety_factor_rows(rows)
            field = "FL"
        self.compare_element_values = self._element_rows_to_values(rows, field)
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"比較結果を読込めません: {exc}")
        return
    self.update_preview()
    self.results_summary.setText(f"比較結果: {path}")

def compare_current_result_to_previous_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    current = self._current_results_dir()
    baseline = self._previous_results_dir(current)
    if current is None or baseline is None:
        _notify_information(self, QMessageBox, "比較できる前回結果が見つかりません。")
        return
    self._compare_result_pair_async(current, baseline, baseline_label="previous")

def compare_current_result_to_baseline_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    current = self._current_results_dir()
    if current is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    selected = QFileDialog.getExistingDirectory(self, "基準結果フォルダを選択", str(self.project_root / "runs"))
    if not selected:
        return
    self._compare_result_pair_async(current, self._result_dir_from_candidate(Path(selected)), baseline_label="baseline")

def compare_current_result_to_design_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    current = self._current_results_dir()
    if current is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    selected = QFileDialog.getExistingDirectory(self, "設計ケース結果フォルダを選択", str(self.project_root / "design_cases"))
    if not selected:
        return
    self._compare_result_pair_async(current, self._result_dir_from_candidate(Path(selected)), baseline_label="design")

def _compare_result_pair_async(owner: Any, qt: Mapping[str, Any], current: Path, baseline: Path, *, baseline_label: str) -> None:
    self = owner
    out_dir = current / "case_output_comparison" / baseline_label
    self._start_post_export_job(
        "case_compare",
        out_dir,
        lambda current=current, baseline=baseline, out_dir=out_dir, baseline_label=baseline_label: {
            **compare_result_cases(
                current,
                baseline,
                output_dir=out_dir,
                current_label="current",
                baseline_label=baseline_label,
            ),
            "operation": "case_compare",
        },
    )

def _current_results_dir(owner: Any, qt: Mapping[str, Any]) -> Path | None:
    self = owner
    if hasattr(self, "_active_result_dir"):
        active = self._active_result_dir()
        if active is not None and active.exists():
            return active
    if self.last_run_dir is not None:
        candidate = self.last_run_dir / "results"
        if candidate.exists():
            return candidate
        if self.last_run_dir.exists():
            return self._result_dir_from_candidate(self.last_run_dir)
    if self.result_stage_dir is not None:
        for candidate in (self.result_stage_dir.parent, self.result_stage_dir.parent.parent):
            if (candidate / "summary.json").exists() or (candidate / "case_manifest.json").exists():
                return candidate
    return None

def _result_dir_from_candidate(owner: Any, qt: Mapping[str, Any], candidate: Path) -> Path:
    self = owner
    if (candidate / "summary.json").exists() or (candidate / "case_manifest.json").exists():
        return candidate
    if (candidate / "results").exists():
        return candidate / "results"
    return candidate

def _previous_results_dir(owner: Any, qt: Mapping[str, Any], current: Path | None) -> Path | None:
    self = owner
    if current is None:
        return None
    for name in ("input_diff.json", "case_manifest.json"):
        data = self._read_json_mapping(current / name)
        raw = data.get("previous_output_dir", "")
        if not raw and isinstance(data.get("input_diff", ""), str):
            diff = self._read_json_mapping(Path(str(data["input_diff"])))
            raw = diff.get("previous_output_dir", "")
        if raw:
            path = self._result_dir_from_candidate(Path(str(raw)))
            if path.exists() and path != current:
                return path
    runs_root = self.project_root / "runs"
    candidates = []
    if runs_root.exists():
        for run_dir in runs_root.glob("run_*"):
            results = self._result_dir_from_candidate(run_dir)
            if results.exists() and results != current and (results / "summary.json").exists():
                candidates.append(results)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def export_result_table_csv(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    if not self.result_rows and self.result_table_csv_path is None:
        _notify_information(self, QMessageBox, "保存する数値表がありません。")
        return
    default = self.project_root / "result_table_export.csv"
    if self.result_stage_dir is not None:
        default = self.result_stage_dir / "gui_table_export.csv"
    path, _ = QFileDialog.getSaveFileName(self, "数値表CSV保存", str(default), "CSV (*.csv)")
    if not path:
        return
    if self.result_table_csv_path is not None:
        shutil.copyfile(self.result_table_csv_path, Path(path))
        self.results_summary.setText(f"数値表を保存しました: {path}")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=self.result_headers or list(self.result_rows[0]))
        writer.writeheader()
        writer.writerows(self.result_rows)
    self.results_summary.setText(f"数値表を保存しました: {path}")

def _start_post_export_job(owner: Any, qt: Mapping[str, Any], operation: str, target: str | Path, fn: Any, *, metadata: Mapping[str, Any] | None = None) -> bool:
    self = owner
    QtCallableRunner = qt["QtCallableRunner"]
    if self._post_export_job_id:
        self.statusBar().showMessage("Post export job is already running.")
        return False
    job_id = self.gui_jobs.start_job("post_export", target=str(target), metadata={"operation": operation, **dict(metadata or {})})
    self._post_export_job_id = job_id
    runner = QtCallableRunner(job_id, fn)
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._post_export_finished)
    runner.signals.failed.connect(self._post_export_job_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage(f"Post export running: {operation}")
    return True

def _post_export_finished(owner: Any, qt: Mapping[str, Any], job_id: str, result_obj: Any) -> None:
    self = owner
    if job_id != self._post_export_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    result = dict(result_obj or {})
    operation = str(result.get("operation", "post_export"))
    self._complete_gui_worker_job(job_id, status="finished", message=operation)
    self._post_export_job_id = ""
    if operation == "save_image":
        out = Path(str(result.get("path", "")))
        role = str(result.get("role", "post_image"))
        if role in {"layout_image", "snapshot"}:
            self.post_snapshot_paths.append(out)
        if role == "layout_image":
            self._add_drawing_layout_row(out)
            self.results_summary.setText(f"Post layout image saved: {out}")
        elif role == "snapshot":
            self._add_post_snapshot_tab(out)
            self.results_summary.setText(f"Post snapshot saved: {out}")
        elif role == "baseline":
            self.results_summary.setText(f"Post baseline saved: {out}")
        else:
            self.results_summary.setText(f"Post image saved: {out}")
        self.append_log(f"[GUI] Post image worker finished role={role} path={out}")
    elif operation == "compare_image":
        self.post_image_diff_result = result
        self.results_summary.setText(f"Post image diff: {result}")
        self.append_log(f"[GUI] Post image diff worker finished ok={result.get('ok')} ratio={result.get('diff_ratio')}")
    elif operation == "scene_pdf":
        out = Path(str(result.get("path", "")))
        self.results_summary.setText(f"PDF drawing saved: {out} pages={result.get('page_count')}")
        self.append_log(f"[GUI] PDF drawing worker finished path={out} pages={result.get('page_count')}")
    elif operation == "build_report":
        out = Path(str(result.get("path", "")))
        self._set_result_view_file(out)
        self.report_summary.setText(f"Report ready: {out}")
    elif operation == "materialize_reports":
        generated = result.get("generated", {})
        generated_map = generated if isinstance(generated, Mapping) else {}
        candidates: list[Path] = []
        results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
        if results_dir is not None:
            candidates.extend(
                [
                    results_dir / "calculation_report.html",
                    results_dir / "standard_report.html",
                    results_dir / "result_view_index.html",
                ]
            )
        summary_raw = str(result.get("summary", "") or "")
        summary = Path(summary_raw) if summary_raw else None
        if summary is not None:
            candidates.append(summary)
        for candidate in candidates:
            if candidate.exists():
                self._set_result_view_file(candidate)
                break
        generated_names = ", ".join(sorted(str(key) for key in generated_map)) or "none"
        self.report_summary.setText(f"Deferred reports ready: {generated_names}")
        self.results_summary.setText(f"Deferred reports generated: {summary_raw}")
        self.append_log(f"[GUI] Deferred reports materialized: {generated_map}")
    elif operation in {"copy_report_pdf", "materialize_and_copy_report_pdf"}:
        out = Path(str(result.get("path", "")))
        manifest = str(result.get("manifest", ""))
        self.result_view.setPlainText(f"PDF calculation report saved:\n{out}\n\nmanifest:\n{manifest}")
        self.tabs.setCurrentWidget(self.result_view)
        self.report_summary.setText(f"PDF calculation report saved: {out}")
        if operation == "materialize_and_copy_report_pdf":
            materialized = result.get("materialized", {})
            self.append_log(f"[GUI] Deferred reports materialized and PDF copied: {materialized}")
    elif operation == "report_audit":
        paths = self._mapping(result.get("paths", {}))
        summary = self._mapping(result.get("summary", {}))
        json_path = Path(str(paths.get("json", "")))
        if json_path.exists():
            self._set_result_view_file(json_path)
        self.report_summary.setText(
            f"Post/report audit: passed={summary.get('passed')} errors={summary.get('error_count')} warnings={summary.get('warning_count')}"
        )
        self.results_summary.setText(f"Post/report audit written: {paths.get('json', '')}")
    elif operation == "case_compare":
        self._populate_case_compare_table(result)
        paths = self._mapping(result.get("paths", {}))
        html_path = Path(str(paths.get("html", "")))
        json_path = Path(str(paths.get("json", "")))
        if html_path.exists():
            self._set_result_view_file(html_path)
        elif json_path.exists():
            self._set_result_view_file(json_path)
        text = (
            f"ケース比較: 差分={result.get('difference_count', 0)} "
            f"欠落={result.get('missing_count', 0)} 行={result.get('row_count', 0)}"
        )
        self.results_summary.setText(text)
        if getattr(self, "report_summary", None) is not None:
            self.report_summary.setText(f"ケース比較を作成しました: {paths.get('html', paths.get('json', ''))}")
        self.append_log(f"[GUI] ケース比較完了: {text}")
    self.statusBar().showMessage("Post export finished", 3500)

def _post_export_job_failed(owner: Any, qt: Mapping[str, Any], job_id: str, message: str) -> None:
    self = owner
    if job_id == self._post_export_job_id:
        self._post_export_job_id = ""
    manifest = self._write_gui_worker_failure_manifest(job_id, message)
    self._complete_gui_worker_job(job_id, status="failed", message=message)
    suffix = f" manifest={manifest}" if manifest is not None else ""
    self.statusBar().showMessage(f"Post export failed: {message}", 7000)
    self.append_log(f"[GUI] Post export worker failed: {message}{suffix}")

def verify_post_view_render(owner: Any, qt: Mapping[str, Any]) -> dict[str, Any]:
    self = owner
    image = self._scene_image_with_layout(max_side=1200.0)
    if image is None:
        self.post_render_verification = {"ok": False, "reason": "empty scene"}
        self.results_summary.setText("Post図検証: empty scene")
        return self.post_render_verification
    step_x = max(1, image.width() // 80)
    step_y = max(1, image.height() // 80)
    non_white = 0
    samples = 0
    for x in range(0, image.width(), step_x):
        for y in range(0, image.height(), step_y):
            color = image.pixelColor(x, y)
            samples += 1
            if color.alpha() > 0 and (color.red() < 245 or color.green() < 245 or color.blue() < 245):
                non_white += 1
    contour_lines = sum(1 for item in self.scene.items() if isinstance(item.data(0), Mapping) and item.data(0).get("kind") == "contour_line")
    labels = sum(1 for item in self.scene.items() if isinstance(item.data(0), Mapping) and str(item.data(0).get("kind", "")).endswith("label"))
    self.post_render_verification = {
        "ok": non_white > 0 and image.width() > 10 and image.height() > 10,
        "width": image.width(),
        "height": image.height(),
        "non_white_samples": non_white,
        "samples": samples,
        "contour_lines": contour_lines,
        "labels": labels,
    }
    self.results_summary.setText(f"Post図検証: {self.post_render_verification}")
    return self.post_render_verification

def save_post_baseline_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    image = self._scene_image_with_layout(max_side=1200.0)
    if image is None:
        _notify_information(self, QMessageBox, "No Post image is available for baseline save.")
        return
    out_dir = self.project_root / "post_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, _ = QFileDialog.getSaveFileName(self, "Post baseline save", str(out_dir / "post_baseline.png"), "PNG (*.png)")
    if not selected:
        return
    out = Path(selected)
    self._start_post_export_job(
        "save_image",
        out,
        lambda image=image.copy(), out=out: save_post_image_snapshot(image, out, role="baseline"),
    )

def save_post_baseline(owner: Any, qt: Mapping[str, Any], path: str | Path | None = None) -> Path | None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    image = self._scene_image_with_layout(max_side=1200.0)
    if image is None:
        _notify_information(self, QMessageBox, "保存するPost図がありません。")
        return None
    if path is None:
        out_dir = self.project_root / "post_baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        selected, _ = QFileDialog.getSaveFileName(self, "Post基準画像保存", str(out_dir / "post_baseline.png"), "PNG (*.png)")
        if not selected:
            return None
        path = selected
    out = Path(path)
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")
    if not image.save(str(out)):
        QMessageBox.warning(self, "GeoFEM", f"Post基準画像を保存できませんでした: {out}")
        return None
    self.results_summary.setText(f"Post基準画像を保存しました: {out}")
    return out

def compare_post_to_baseline_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    selected, _ = QFileDialog.getOpenFileName(self, "Post baseline select", str(self.project_root / "post_baselines"), "PNG (*.png)")
    if not selected:
        return
    current = self._scene_image_with_layout(max_side=1200.0)
    if current is None:
        _notify_information(self, QMessageBox, "No Post image is available for comparison.")
        return
    baseline = Path(selected)
    self._start_post_export_job(
        "compare_image",
        baseline,
        lambda image=current.copy(), baseline=baseline: compare_post_image_snapshot(image, baseline),
    )

def compare_post_to_baseline(owner: Any, qt: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QImage = qt["QImage"]
    if path is None:
        selected, _ = QFileDialog.getOpenFileName(self, "Post基準画像を選択", str(self.project_root / "post_baselines"), "PNG (*.png)")
        if not selected:
            return {}
        path = selected
    current = self._scene_image_with_layout(max_side=1200.0)
    baseline = QImage(str(path))
    if current is None or baseline.isNull():
        self.post_image_diff_result = {"ok": False, "reason": "missing image"}
        return self.post_image_diff_result
    width = min(current.width(), baseline.width())
    height = min(current.height(), baseline.height())
    step_x = max(1, width // 100)
    step_y = max(1, height // 100)
    total = 0
    diff = 0
    accum = 0.0
    for x in range(0, width, step_x):
        for y in range(0, height, step_y):
            ca = current.pixelColor(x, y)
            cb = baseline.pixelColor(x, y)
            delta = abs(ca.red() - cb.red()) + abs(ca.green() - cb.green()) + abs(ca.blue() - cb.blue()) + abs(ca.alpha() - cb.alpha())
            total += 1
            accum += delta
            if delta > 18:
                diff += 1
    ratio = diff / max(total, 1)
    self.post_image_diff_result = {
        "ok": ratio <= 0.02 and current.width() == baseline.width() and current.height() == baseline.height(),
        "diff_ratio": ratio,
        "mean_delta": accum / max(total, 1),
        "samples": total,
        "current_size": [current.width(), current.height()],
        "baseline_size": [baseline.width(), baseline.height()],
    }
    self.results_summary.setText(f"Post画像差分: {self.post_image_diff_result}")
    return self.post_image_diff_result

def write_post_image_diff_ci_job(owner: Any, qt: Mapping[str, Any], path: str | Path | None = None, *, baseline: str | Path | None = None) -> Path:
    self = owner
    if path is None:
        path = self.project_root / ".github" / "workflows" / "post-image-diff.yml"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    baseline_text = str(baseline or "post_baselines/post_baseline.png")
    result_text = "post_image_diff_result.json"
    matrix_path = out.parent / "post-image-diff-cases.yaml"
    matrix = {
        "post_image_diff_cases": [
            {"name": "contour", "baseline": baseline_text, "threshold": 0.02, "generate_sample": True},
            {"name": "deformed_overlay", "baseline": "post_baselines/post_deformed_overlay.png", "threshold": 0.02, "generate_sample": True},
            {"name": "srm_fl", "baseline": "post_baselines/post_srm_fl.png", "threshold": 0.02, "generate_sample": True},
        ]
    }
    matrix_path.write_text(yaml.safe_dump(matrix, allow_unicode=True, sort_keys=False), encoding="utf-8")
    workflow = {
        "name": "GeoFEM Post Image Diff",
        "on": {"push": None, "pull_request": None},
        "jobs": {
            "post-image-diff": {
                "runs-on": "windows-latest",
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"uses": "actions/setup-python@v5", "with": {"python-version": "3.12"}},
                    {"name": "Install", "run": "pip install -r requirements.txt"},
                    {
                        "name": "Generate and compare Post image matrix",
                        "run": f"python tools/post_image_diff_ci.py --matrix \"{matrix_path.as_posix()}\" --out \"{result_text}\"",
                    },
                    {"name": "Run GUI Post smoke regression", "run": "python -m unittest tests.test_import_and_mesh.GuiModelCheckTests.test_p0_selection_stage_diff_and_post_controls"},
                ],
            }
        },
    }
    out.write_text(yaml.safe_dump(workflow, allow_unicode=True, sort_keys=False), encoding="utf-8")
    self.write_audit_event("post_image_diff_ci_written", str(out), {"matrix": str(matrix_path), "cases": len(matrix["post_image_diff_cases"])})
    self.results_summary.setText(f"Post画像差分CIジョブを作成しました: {out}")
    return out

def add_report_text_block(owner: Any, qt: Mapping[str, Any], title: str, text: str) -> None:
    self = owner
    table = getattr(self, "report_page_table", None)
    if table is None:
        return
    row = table.rowCount()
    table.insertRow(row)
    values = ["text", text, title, "0.08", f"{0.08 + 0.18 * (row % 4):.2f}", "0.84", "0.14"]
    for col, value in enumerate(values):
        self._set_report_page_item(table, row, col, value)
    self.refresh_report_canvas()

def _set_report_page_item(owner: Any, qt: Mapping[str, Any], table: QTableWidget, row: int, col: int, value: Any) -> None:
    self = owner
    QTableWidget = qt["QTableWidget"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    Qt = qt["Qt"]
    item = QTableWidgetItem(str(value))
    help_payload = documentation_payload("report.item")
    item.setData(Qt.ItemDataRole.UserRole, {"row": row, "col": col, **help_payload})
    item.setToolTip(f"{help_payload['summary']}\nHelp: {help_payload['help_id']}\n{help_payload['help_url']}")
    table.setItem(row, col, item)

def refresh_report_canvas(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QBrush = qt["QBrush"]
    QColor = qt["QColor"]
    QGraphicsItem = qt["QGraphicsItem"]
    QPen = qt["QPen"]
    Qt = qt["Qt"]
    scene = getattr(self, "report_layout_scene", None)
    view = getattr(self, "report_layout_view", None)
    if scene is None:
        return
    scene.clear()
    page_w = 280.0
    page_h = 396.0
    scene.setSceneRect(0.0, 0.0, page_w, page_h)
    scene.addRect(0.0, 0.0, page_w, page_h, QPen(QColor("#adb5bd")), QBrush(QColor("#ffffff")))
    for row, spec in enumerate(self._report_page_specs()):
        x, y, w, h = spec["rect"]
        item_x = max(0.0, min(1.0, x)) * page_w
        item_y = max(0.0, min(1.0, y)) * page_h
        item_w = max(0.05, min(1.0, w)) * page_w
        item_h = max(0.05, min(1.0, h)) * page_h
        rect_item = scene.addRect(
            0.0,
            0.0,
            item_w,
            item_h,
            QPen(QColor("#2563eb"), 1),
            QBrush(QColor(37, 99, 235, 35)),
        )
        rect_item.setPos(item_x, item_y)
        rect_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        rect_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        rect_item.setData(0, {"kind": "report_block", "row": row})
        label = scene.addText(str(spec.get("title", ""))[:40])
        label.setDefaultTextColor(QColor("#111827"))
        label.setPos(4.0, 2.0)
        label.setParentItem(rect_item)
        handle = scene.addRect(item_w - 8.0, item_h - 8.0, 10.0, 10.0, QPen(QColor("#dc2626"), 1), QBrush(QColor("#fee2e2")))
        handle.setParentItem(rect_item)
        handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        handle.setData(0, {"kind": "report_resize_handle", "row": row})
    if view is not None:
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

def apply_report_canvas_positions(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QPointF = qt["QPointF"]
    scene = getattr(self, "report_layout_scene", None)
    table = getattr(self, "report_page_table", None)
    if scene is None or table is None:
        return
    page_rect = scene.sceneRect()
    page_w = max(page_rect.width(), 1.0)
    page_h = max(page_rect.height(), 1.0)
    handles: dict[int, QPointF] = {}
    for item in scene.items():
        data = item.data(0)
        if isinstance(data, Mapping) and data.get("kind") == "report_resize_handle":
            handles[int(data.get("row", -1))] = item.sceneBoundingRect().center()
    for item in scene.items():
        data = item.data(0)
        if not isinstance(data, Mapping) or data.get("kind") != "report_block":
            continue
        row = int(data.get("row", -1))
        if row < 0 or row >= table.rowCount():
            continue
        rect = item.sceneBoundingRect()
        x = max(0.0, min(1.0, rect.left() / page_w))
        y = max(0.0, min(1.0, rect.top() / page_h))
        handle = handles.get(row)
        if handle is not None:
            w = max(0.05, min(1.0 - x, (handle.x() - rect.left()) / page_w))
            h = max(0.05, min(1.0 - y, (handle.y() - rect.top()) / page_h))
        else:
            w = max(0.05, min(1.0 - x, rect.width() / page_w))
            h = max(0.05, min(1.0 - y, rect.height() / page_h))
        for col, value in zip((3, 4, 5, 6), (x, y, w, h), strict=False):
            self._set_report_page_item(table, row, col, f"{value:.3f}")
    self.preview_report_page_layout()
    self.refresh_report_canvas()

def nudge_selected_report_block(owner: Any, qt: Mapping[str, Any], dx: float = 0.0, dy: float = 0.0, dw: float = 0.0, dh: float = 0.0) -> None:
    self = owner
    table = getattr(self, "report_page_table", None)
    if table is None or table.currentRow() < 0:
        return
    row = table.currentRow()

    def read(col: int, fallback: float) -> float:
        try:
            return float(self._table_text(table, row, col) or fallback)
        except ValueError:
            return fallback

    x = read(3, 0.08) + dx
    y = read(4, 0.08) + dy
    w = read(5, 0.84) + dw
    h = read(6, 0.20) + dh
    w = max(0.05, min(1.0, w))
    h = max(0.05, min(1.0, h))
    x = max(0.0, min(1.0 - w, x))
    y = max(0.0, min(1.0 - h, y))
    for col, value in zip((3, 4, 5, 6), (x, y, w, h), strict=False):
        self._set_report_page_item(table, row, col, f"{value:.3f}")
    self.preview_report_page_layout()
    self.refresh_report_canvas()

def add_current_post_to_report_page(owner: Any, qt: Mapping[str, Any]) -> Path | None:
    self = owner
    path = self.add_current_view_to_drawing_layout()
    if path is None:
        return None
    table = getattr(self, "report_page_table", None)
    if table is None:
        return path
    row = table.rowCount()
    table.insertRow(row)
    values = ["image", str(path), self.drawing_title_edit.text().strip() or path.stem, "0.08", f"{0.10 + 0.42 * (row % 2):.2f}", "0.84", "0.36"]
    for col, value in enumerate(values):
        self._set_report_page_item(table, row, col, value)
    self.refresh_report_canvas()
    return path

def _report_page_specs(owner: Any, qt: Mapping[str, Any]) -> list[dict[str, Any]]:
    self = owner
    table = getattr(self, "report_page_table", None)
    if table is None:
        return []
    specs = []
    for row in range(table.rowCount()):
        try:
            rect = (
                float(self._table_text(table, row, 3) or 0.08),
                float(self._table_text(table, row, 4) or 0.08),
                float(self._table_text(table, row, 5) or 0.84),
                float(self._table_text(table, row, 6) or 0.20),
            )
        except ValueError:
            rect = (0.08, 0.08, 0.84, 0.20)
        specs.append(
            {
                "type": self._table_text(table, row, 0).strip() or "text",
                "source": self._table_text(table, row, 1),
                "title": self._table_text(table, row, 2),
                "rect": rect,
            }
        )
    return specs

def build_report_wysiwyg_html(owner: Any, qt: Mapping[str, Any]) -> str:
    self = owner
    specs = self._report_page_specs()
    title = self.drawing_title_edit.text().strip() or "GeoFEM 2D Report"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,'Meiryo',sans-serif;margin:0;background:#f1f3f5}.page{position:relative;width:1120px;height:1584px;margin:24px auto;background:white;border:1px solid #adb5bd}.box{position:absolute;border:1px solid #495057;padding:10px;box-sizing:border-box;overflow:hidden}.box h3{margin:0 0 6px 0;font-size:18px}.box img{max-width:100%;max-height:calc(100% - 28px);display:block;margin:auto}</style>",
        f"<title>{html.escape(title)}</title></head><body><div class='page'>",
        f"<div style='position:absolute;left:40px;top:20px;font-size:24px;font-weight:bold'>{html.escape(title)}</div>",
    ]
    for spec in specs:
        x, y, w, h = spec["rect"]
        style = f"left:{x*100:.3f}%;top:{y*100:.3f}%;width:{w*100:.3f}%;height:{h*100:.3f}%;"
        parts.append(f"<div class='box' style='{style}'><h3>{html.escape(str(spec['title']))}</h3>")
        if str(spec["type"]).lower() == "image":
            parts.append(f"<img src='{html.escape(str(spec['source']))}' alt='post image'>")
        else:
            parts.append(f"<div>{html.escape(str(spec['source'])).replace(chr(10), '<br>')}</div>")
        parts.append("</div>")
    parts.append("</div></body></html>")
    return "".join(parts)

def preview_report_page_layout(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    html_text = self.build_report_wysiwyg_html()
    self.result_view.setPlainText(html_text)
    self.tabs.setCurrentWidget(self.result_view)
    self.results_summary.setText("WYSIWYG帳票HTMLを生成しました。")

def export_scene_pdf_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    default = self.project_root / "post_view.pdf"
    if self.result_stage_dir is not None:
        default = self.result_stage_dir / "post_view.pdf"
    path, _ = QFileDialog.getSaveFileName(self, "PDF drawing export", str(default), "PDF (*.pdf)")
    if not path:
        return
    out = Path(path)
    specs = [
        {**dict(spec), "path": str(spec.get("path", ""))}
        for spec in self._drawing_layout_specs()
    ]
    image = None
    if not specs:
        image = self._scene_image_with_layout(max_side=3200.0)
        if image is None:
            _notify_information(self, QMessageBox, "No Post scene is available for PDF export.")
            return
    snapshots = [str(snapshot) for snapshot in self.post_snapshot_paths]
    page_title = self.drawing_title_edit.text().strip() if getattr(self, "drawing_title_edit", None) is not None else "GeoFEM 2D Post"
    self._start_post_export_job(
        "scene_pdf",
        out,
        lambda image=image.copy() if image is not None else None, specs=specs, snapshots=snapshots, out=out, page_title=page_title: export_scene_pdf_snapshot(
            out,
            current_image=image,
            layout_specs=specs,
            snapshot_paths=snapshots,
            page_title=page_title,
        ),
    )

def export_scene_pdf(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QImage = qt["QImage"]
    QMessageBox = qt["QMessageBox"]
    QPageSize = qt["QPageSize"]
    QPainter = qt["QPainter"]
    QPdfWriter = qt["QPdfWriter"]
    default = self.project_root / "post_view.pdf"
    if self.result_stage_dir is not None:
        default = self.result_stage_dir / "post_view.pdf"
    path, _ = QFileDialog.getSaveFileName(self, "PDF図面出力", str(default), "PDF (*.pdf)")
    if not path:
        return
    out = Path(path)
    if out.suffix.lower() != ".pdf":
        out = out.with_suffix(".pdf")
    image = self._scene_image_with_layout(max_side=3200.0)
    if image is None:
        _notify_information(self, QMessageBox, "出力する図面がありません。")
        return
    try:
        from PySide6.QtGui import QPageSize, QPdfWriter
    except Exception as exc:
        QMessageBox.warning(self, "GeoFEM", f"PDF出力を初期化できません: {exc}")
        return
    writer = QPdfWriter(str(out))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setResolution(160)
    painter = QPainter(writer)
    specs = self._drawing_layout_specs()
    if specs:
        self._draw_layout_pdf_page(painter, specs)
    else:
        painter.drawImage(painter.viewport(), image)
        for snapshot in self.post_snapshot_paths:
            if not snapshot.exists():
                continue
            snapshot_image = QImage(str(snapshot))
            if snapshot_image.isNull():
                continue
            writer.newPage()
            painter.drawImage(painter.viewport(), snapshot_image)
    painter.end()
    self.results_summary.setText(f"PDF図面を保存しました: {out}")

def _add_post_snapshot_tab(owner: Any, qt: Mapping[str, Any], out: Path) -> None:
    self = owner
    QLabel = qt["QLabel"]
    QPixmap = qt["QPixmap"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    try:
        from PySide6.QtGui import QPixmap
    except Exception:
        QPixmap = None  # type: ignore[assignment]
    page = QWidget()
    page_layout = QVBoxLayout(page)
    label = QLabel(str(out))
    label.setWordWrap(True)
    if QPixmap is not None:
        pixmap = QPixmap(str(out))
        if not pixmap.isNull():
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.setMinimumSize(360, 260)
    page_layout.addWidget(label)
    self.tabs.addTab(page, f"Post snapshot {self.post_snapshot_count}")
    self.tabs.setCurrentWidget(page)

def duplicate_post_view_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    image = self._scene_image_with_layout(max_side=1800.0)
    if image is None:
        _notify_information(self, QMessageBox, "No Post view is available for duplication.")
        return
    out_dir = (self.result_stage_dir or self.project_root) / "post_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    self.post_snapshot_count += 1
    out = out_dir / f"post_snapshot_{self.post_snapshot_count:03d}.png"
    self._start_post_export_job(
        "save_image",
        out,
        lambda image=image.copy(), out=out: save_post_image_snapshot(image, out, role="snapshot"),
    )

def duplicate_post_view(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QLabel = qt["QLabel"]
    QMessageBox = qt["QMessageBox"]
    QPixmap = qt["QPixmap"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]
    image = self._scene_image_with_layout(max_side=1800.0)
    if image is None:
        _notify_information(self, QMessageBox, "複製するPostビューがありません。")
        return
    out_dir = (self.result_stage_dir or self.project_root) / "post_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    self.post_snapshot_count += 1
    out = out_dir / f"post_snapshot_{self.post_snapshot_count:03d}.png"
    if not image.save(str(out)):
        QMessageBox.warning(self, "GeoFEM", f"Postビューを保存できませんでした: {out}")
        return
    self.post_snapshot_paths.append(out)
    try:
        from PySide6.QtGui import QPixmap
    except Exception:
        QPixmap = None  # type: ignore[assignment]
    page = QWidget()
    page_layout = QVBoxLayout(page)
    label = QLabel(str(out))
    label.setWordWrap(True)
    if QPixmap is not None:
        pixmap = QPixmap(str(out))
        if not pixmap.isNull():
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.setMinimumSize(360, 260)
    page_layout.addWidget(label)
    self.tabs.addTab(page, f"Post比較{self.post_snapshot_count}")
    self.tabs.setCurrentWidget(page)
    self.results_summary.setText(f"Postビューを複製しました: {out}")

def show_srm_post_async(owner: Any, qt: Mapping[str, Any], *_args: Any, update_only: bool = False) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    QtCallableRunner = qt["QtCallableRunner"]
    if self._post_job_id:
        self.statusBar().showMessage("Post処理ジョブを実行中です。")
        return
    stage_dir = self._latest_stage_dir()
    if stage_dir is None and not self.result_rows:
        if not update_only:
            _notify_information(self, QMessageBox, "SRM表示に使える要素応力データがありません。")
        return
    cfg_snapshot = copy.deepcopy(self.cfg)
    result_rows_snapshot = [dict(row) for row in self.result_rows]
    options_snapshot = self._srm_post_options()
    active_run_dir = self._active_run_dir() if hasattr(self, "_active_run_dir") else self.last_run_dir
    last_run_dir = str(active_run_dir or "")
    stage_dir_text = str(stage_dir or "")
    job_id = self.gui_jobs.start_job("post", target=str(stage_dir or "current-table"), metadata={"kind": "srm_post", "update_only": bool(update_only)})
    self._post_job_id = job_id
    runner = QtCallableRunner(
        job_id,
        lambda: {
            **load_srm_post_snapshot(
                type(self),
                cfg_snapshot,
                last_run_dir=last_run_dir,
                result_stage_dir=stage_dir_text,
                result_rows=result_rows_snapshot,
                options=options_snapshot,
            ),
            "update_only": bool(update_only),
        },
    )
    self._gui_task_runners[job_id] = runner
    runner.signals.finished.connect(self._srm_post_finished)
    runner.signals.failed.connect(self._post_job_failed)
    self.gui_thread_pool.start(runner)
    self.statusBar().showMessage("SRM Postをバックグラウンド生成中...")

def _srm_post_options(owner: Any, qt: Mapping[str, Any]) -> dict[str, Any]:
    self = owner
    return {
        "fl_limit": self.srm_fl_limit.text() if hasattr(self, "srm_fl_limit") else "1.05",
        "plastic_threshold": self.srm_plastic_threshold.text() if hasattr(self, "srm_plastic_threshold") else "0.5",
        "local_fl_aggregation": self._combo_value(self.srm_local_fl_aggregation, "mean") if hasattr(self, "srm_local_fl_aggregation") else "mean",
        "search_mode": self._combo_value(self.srm_search_mode, "all") if hasattr(self, "srm_search_mode") else "all",
        "slope_direction": self._combo_value(self.srm_slope_direction, "auto") if hasattr(self, "srm_slope_direction") else "auto",
        "min_candidate_length": self.srm_min_candidate_length.text() if hasattr(self, "srm_min_candidate_length") else "0.0",
        "max_circle_radius": self.srm_max_circle_radius.text() if hasattr(self, "srm_max_circle_radius") else "0.0",
        "require_boundary_exit": bool(self.srm_require_boundary_exit.isChecked()) if hasattr(self, "srm_require_boundary_exit") else False,
    }

def _srm_post_finished(owner: Any, qt: Mapping[str, Any], job_id: str, result_obj: Any) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    if job_id != self._post_job_id:
        self._gui_task_runners.pop(job_id, None)
        return
    result = dict(result_obj or {})
    rows = [dict(row) for row in list(result.get("rows", [])) if isinstance(row, Mapping)]
    self._complete_gui_worker_job(job_id, status="finished", message=f"{len(rows)} SRM rows")
    self._post_job_id = ""
    if not rows:
        if not bool(result.get("update_only", False)):
            _notify_information(self, QMessageBox, "SRM表示に使える要素応力データがありません。")
        return
    self._populate_result_table(rows)
    self._set_result_component_without_reload("FL")
    self._set_result_table_component_without_reload("FL")
    self._set_colormap_without_reload("Safety FL")
    self.result_displacements = {}
    self.result_node_values = {}
    self.result_element_values = {
        str(key): float(value)
        for key, value in self._mapping(result.get("result_element_values", {})).items()
    }
    self.post_component = "FL"
    self.post_mode = "srm"
    self.srm_slip_candidates = [dict(item) for item in list(result.get("srm_slip_candidates", [])) if isinstance(item, Mapping)]
    self.srm_trial_rows = [dict(item) for item in list(result.get("srm_trial_rows", [])) if isinstance(item, Mapping)]
    self._populate_srm_trial_table()
    self._populate_srm_slip_table()
    self._populate_srm_candidate_compare_table()
    self._populate_srm_local_fl_table(rows)
    summary_text = str(result.get("summary_text", ""))
    self.srm_post_summary.setText(summary_text)
    self.results_summary.setText(summary_text)
    if not bool(result.get("update_only", False)):
        self.tabs.setCurrentWidget(self.result_table)
    self.update_preview()
    self.append_log(f"[GUI] SRM Post生成完了: rows={len(rows)} candidates={len(self.srm_slip_candidates)}")

def show_srm_post(owner: Any, qt: Mapping[str, Any], *_args: Any, update_only: bool = False) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    rows = self._srm_post_rows()
    if not rows:
        if not update_only:
            _notify_information(self, QMessageBox, "SRM表示に使える要素応力データがありません。")
        return
    rows = self._with_safety_factor_rows(rows)
    self._populate_result_table(rows)
    self._set_result_component_without_reload("FL")
    self._set_result_table_component_without_reload("FL")
    self._set_colormap_without_reload("Safety FL")
    self.result_displacements = {}
    self.result_node_values = {}
    self.result_element_values = self._srm_element_fl_values(rows)
    self.post_component = "FL"
    self.post_mode = "srm"
    self.srm_slip_candidates = self._estimate_srm_slip_candidates(rows)
    self.srm_trial_rows = self._current_srm_trials()
    self._populate_srm_trial_table()
    self._populate_srm_slip_table()
    self._populate_srm_candidate_compare_table()
    self._populate_srm_local_fl_table(rows)
    summary_text = self._srm_post_summary_text(rows)
    self.srm_post_summary.setText(summary_text)
    self.results_summary.setText(summary_text)
    if not update_only:
        self.tabs.setCurrentWidget(self.result_table)
    self.update_preview()

def export_srm_slip_csv(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    if not self.srm_slip_candidates:
        _notify_information(self, QMessageBox, "保存するSRMすべり面候補がありません。")
        return
    default = self.project_root / "srm_slip_candidates.csv"
    if self.result_stage_dir is not None:
        default = self.result_stage_dir / "srm_slip_candidates_gui.csv"
    path, _ = QFileDialog.getSaveFileName(self, "SRMすべり面CSV保存", str(default), "CSV (*.csv)")
    if not path:
        return
    fieldnames = ["rank", "type", "elements", "length", "mean_y", "min_FL", "mean_FL", "max_FL", "plastic_ratio", "score", "optimized"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(self.srm_slip_candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "type": candidate.get("type", ""),
                    "elements": " ".join(str(eid) for eid in candidate.get("elements", [])),
                    "length": candidate.get("length", ""),
                    "mean_y": candidate.get("mean_y", ""),
                    "min_FL": candidate.get("min_fl", ""),
                    "mean_FL": candidate.get("mean_fl", ""),
                    "max_FL": candidate.get("max_fl", ""),
                    "plastic_ratio": candidate.get("plastic_ratio", ""),
                    "score": candidate.get("score", ""),
                    "optimized": candidate.get("optimized", ""),
                }
            )
    self.results_summary.setText(f"SRMすべり面候補を保存しました: {path}")

def build_srm_candidate_report(owner: Any, qt: Mapping[str, Any], path: str | Path | None = None) -> Path | None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    if not self.srm_slip_candidates:
        rows = self._srm_post_rows()
        if rows:
            rows = self._with_safety_factor_rows(rows)
            self.srm_slip_candidates = self._estimate_srm_slip_candidates(rows)
            self._populate_srm_candidate_compare_table()
            self._populate_srm_local_fl_table(rows)
    if not self.srm_slip_candidates:
        _notify_information(self, QMessageBox, "SRM候補詳細帳票に出力する候補がありません。")
        return None
    if path is None:
        out_dir = self.result_stage_dir or self.project_root
        path = out_dir / "srm_candidate_report.html"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = self._with_safety_factor_rows(self._srm_post_rows())
    local_stats = self._srm_local_fl_stats(rows)
    parts = [
        "<!doctype html><html lang='ja'><meta charset='utf-8'><title>SRM candidate report</title>",
        "<style>body{font-family:Meiryo,Arial,sans-serif;line-height:1.5}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #999;padding:5px}th{background:#eef2f7}.bad{color:#b00020;font-weight:bold}</style><body>",
        "<h1>SRMすべり面候補 詳細照査</h1>",
        f"<p>{html.escape(self._srm_post_summary_text(rows))}</p>",
        "<h2>候補比較</h2><table><tr><th>rank</th><th>type</th><th>elements</th><th>length</th><th>min FL</th><th>mean FL</th><th>max FL</th><th>plastic ratio</th><th>score</th><th>optimized</th></tr>",
    ]
    for rank, candidate in enumerate(self.srm_slip_candidates, start=1):
        parts.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(str(candidate.get('type', '')))}</td>"
            f"<td>{html.escape(' '.join(str(eid) for eid in candidate.get('elements', [])))}</td>"
            f"<td>{html.escape(self._format_float(candidate.get('length')))}</td>"
            f"<td>{html.escape(self._format_float(candidate.get('min_fl')))}</td>"
            f"<td>{html.escape(self._format_float(candidate.get('mean_fl')))}</td>"
            f"<td>{html.escape(self._format_float(candidate.get('max_fl')))}</td>"
            f"<td>{html.escape(self._format_float(candidate.get('plastic_ratio')))}</td>"
            f"<td>{html.escape(self._format_float(candidate.get('score')))}</td>"
            f"<td>{html.escape(str(candidate.get('optimized', '')))}</td>"
            "</tr>"
        )
    parts.append("</table><h2>要素別局所FL</h2><table><tr><th>element</th><th>ip count</th><th>min</th><th>mean</th><th>max</th></tr>")
    for eid, stats in sorted(local_stats.items(), key=lambda item: self._natural_sort_key(item[0])):
        cls = " class='bad'" if float(stats.get("min", 99.0)) <= 1.0 else ""
        parts.append(
            f"<tr{cls}><td>{html.escape(str(eid))}</td><td>{stats.get('count')}</td>"
            f"<td>{html.escape(self._format_float(stats.get('min')))}</td>"
            f"<td>{html.escape(self._format_float(stats.get('mean')))}</td>"
            f"<td>{html.escape(self._format_float(stats.get('max')))}</td></tr>"
        )
    parts.append("</table></body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    self._set_result_view_file(out)
    self.results_summary.setText(f"SRM候補詳細帳票を作成しました: {out}")
    return out

def show_srm_summary(owner: Any, qt: Mapping[str, Any], *, update_only: bool = False) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    summary = self._current_summary_data()
    if not summary:
        if not update_only:
            _notify_information(self, QMessageBox, "解析結果サマリがありません。")
        return
    refresh_result_judgment_summary(owner, qt)
    stage = self._current_summary_stage(summary)
    if not isinstance(stage, Mapping):
        return
    solver = stage.get("solver", {})
    if not isinstance(solver, Mapping):
        solver = {}
    srm = solver.get("srm", {})
    lines = [
        f"ステージ: {stage.get('name', '')}",
        f"最大変位: {stage.get('max_displacement')}",
        f"最大沈下: {stage.get('max_settlement')}",
        f"界面すべり点数: {stage.get('interface_slip_points')}",
        f"最大界面すべり: {stage.get('interface_slip_max')}",
    ]
    if isinstance(srm, Mapping):
        lines.insert(1, f"SRM安全率: {srm.get('factor_of_safety')}")
        trials = srm.get("trials", [])
        if isinstance(trials, list):
            lines.append("")
            lines.append("SRM trials:")
            for trial in trials[:30]:
                if isinstance(trial, Mapping):
                    lines.append(
                        f"  factor={trial.get('factor')} converged={trial.get('converged')} "
                        f"plastic_ratio={trial.get('plastic_ratio')} ok={trial.get('ok')}"
                    )
    else:
        plastic_ratio = None
        stage_dir = self._latest_stage_dir()
        if stage_dir is not None and (stage_dir / "element_stress.csv").exists():
            rows = self._read_csv_rows(stage_dir / "element_stress.csv")
            active_rows = [row for row in rows if str(row.get("active", "1")).strip() not in {"0", "0.0", "false", "False"}]
            if active_rows:
                plastic_ratio = sum(1 for row in active_rows if float(row.get("plastic", 0.0) or 0.0) > 0.0) / len(active_rows)
        if plastic_ratio is not None:
            lines.insert(1, f"塑性化要素比: {plastic_ratio:.6g}")
    text = "\n".join(lines)
    if update_only:
        return
    self.results_summary.setText(text.replace("\n", " / "))
    self.result_view.setPlainText(text)
    self.tabs.setCurrentWidget(self.result_view)

def preview_stage_report(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    stage_dir = self._latest_stage_dir()
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    if results_dir is not None:
        run_report = results_dir / "calculation_report.html"
        if run_report.exists():
            self._set_result_view_file(run_report)
            self.report_summary.setText(f"計算書プレビュー: {run_report}")
            return
        if _summary_has_deferred_report_artifacts(results_dir / "summary.json"):
            materialize_deferred_reports_async(owner, qt)
            return
    if stage_dir is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    path = stage_dir / "report.html"
    if not path.exists():
        _notify_information(self, QMessageBox, "選択ステージのレポートがありません。")
        return
    self._set_result_view_file(path)
    self.report_summary.setText(f"プレビュー: {path}")

def build_selected_report_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    if results_dir is None:
        _notify_information(self, QMessageBox, "No analysis result directory is available.")
        return
    if _summary_has_deferred_report_artifacts(results_dir / "summary.json"):
        materialize_deferred_reports_async(owner, qt)
        return
    stage_dir = self._latest_stage_dir()
    include_summary = bool(self.report_include_summary.isChecked())
    include_tables = bool(self.report_include_tables.isChecked())
    self._start_post_export_job(
        "build_report",
        results_dir / "gui_report.html",
        lambda results_dir=results_dir, stage_dir=str(stage_dir) if stage_dir is not None else "", include_summary=include_summary, include_tables=include_tables: build_selected_report_snapshot(
            results_dir=results_dir,
            stage_dir=stage_dir or None,
            include_summary=include_summary,
            include_tables=include_tables,
        ),
    )

def audit_post_report_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    result_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    if result_dir is None:
        _notify_information(self, QMessageBox, "No analysis result directory is available.")
        return
    out_dir = result_dir / "post_report_audit"
    baseline_dir = self.project_root / "post_baselines"
    baseline = str(baseline_dir) if baseline_dir.exists() else None
    self._start_post_export_job(
        "report_audit",
        out_dir,
        lambda result_dir=result_dir, out_dir=out_dir, baseline=baseline: audit_post_report_snapshot(
            result_dir=result_dir,
            output_dir=out_dir,
            baseline_dir=baseline,
        ),
    )

def build_selected_report(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QMessageBox = qt["QMessageBox"]
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    if results_dir is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    commercial_report = results_dir / "calculation_report.html"
    if commercial_report.exists():
        self._set_result_view_file(commercial_report)
        self.report_summary.setText(f"計算書を表示しました: {commercial_report}")
        return
    if _summary_has_deferred_report_artifacts(results_dir / "summary.json"):
        materialize_deferred_reports_async(owner, qt)
        return
    summary_path = results_dir / "summary.json"
    stage_dir = self._latest_stage_dir()
    parts = [
        "<!doctype html><html lang=\"ja\"><meta charset=\"utf-8\"><title>GeoFEM 2D 計算書</title>",
        "<style>body{font-family:Meiryo,sans-serif;line-height:1.55}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:5px}th{background:#f3f4f6}pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #ddd;padding:10px}</style><body>",
        "<h1>GeoFEM 2D 計算書</h1>",
    ]
    if self.report_include_summary.isChecked() and summary_path.exists():
        summary = html.escape(preview_text_file(summary_path).text)
        parts.append("<h2>解析サマリ</h2><pre>")
        parts.append(summary)
        parts.append("</pre>")
    if self.report_include_tables.isChecked() and stage_dir is not None:
        parts.append("<h2>数値表リンク</h2><ul>")
        for name in ["displacements.csv", "reactions.csv", "element_stress.csv", "interface_state.csv", "pore_pressure.csv", "riks_path.csv"]:
            path = stage_dir / name
            if path.exists():
                parts.append(f"<li>{html.escape(name)}: {html.escape(str(path))}</li>")
        parts.append("</ul>")
    run_log = results_dir / "run.log"
    if run_log.exists():
        parts.append("<h2>ログ</h2><pre>")
        parts.append(html.escape(preview_text_file(run_log).text))
        parts.append("</pre>")
    if stage_dir is not None and (stage_dir / "report.html").exists():
        parts.append(f"<p>ステージレポート: {html.escape(str(stage_dir / 'report.html'))}</p>")
    parts.append("</body></html>")
    out = results_dir / "gui_report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    self._set_result_view_file(out)
    self.report_summary.setText(f"計算書を作成しました: {out}")

def export_calculation_report_pdf_async(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    run_dir = self._active_run_dir() if hasattr(self, "_active_run_dir") else self.last_run_dir
    if results_dir is None or run_dir is None:
        _notify_information(self, QMessageBox, "No analysis result directory is available.")
        return
    source = results_dir / "calculation_report.pdf"
    if not source.exists():
        if _summary_has_deferred_report_artifacts(results_dir / "summary.json"):
            selected, _ = QFileDialog.getSaveFileName(self, "Calculation report PDF save", str(source), "PDF (*.pdf)")
            if not selected:
                return
            self._start_post_export_job(
                "materialize_and_copy_report_pdf",
                Path(selected),
                lambda results_dir=results_dir, input_path=run_dir / "input.yaml", selected=selected: _materialize_and_copy_report_pdf_snapshot(
                    results_dir=results_dir,
                    input_path=input_path,
                    destination=selected,
                ),
            )
            return
        _notify_information(self, QMessageBox, "PDF calculation report does not exist.")
        return
    selected, _ = QFileDialog.getSaveFileName(self, "Calculation report PDF save", str(source), "PDF (*.pdf)")
    if not selected:
        return
    out = Path(selected)
    manifest = results_dir / "calculation_report_manifest.json"
    self._start_post_export_job(
        "copy_report_pdf",
        out,
        lambda source=source, out=out, manifest=manifest: copy_report_pdf_snapshot(source, out, manifest=manifest),
    )

def export_calculation_report_pdf(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    QFileDialog = qt["QFileDialog"]
    QMessageBox = qt["QMessageBox"]
    results_dir = self._active_result_dir() if hasattr(self, "_active_result_dir") else None
    run_dir = self._active_run_dir() if hasattr(self, "_active_run_dir") else self.last_run_dir
    if results_dir is None or run_dir is None:
        _notify_information(self, QMessageBox, "解析結果がまだありません。")
        return
    source = results_dir / "calculation_report.pdf"
    if not source.exists():
        if _summary_has_deferred_report_artifacts(results_dir / "summary.json"):
            selected, _ = QFileDialog.getSaveFileName(self, "計算書PDF保存", str(source), "PDF (*.pdf)")
            if not selected:
                return
            self._start_post_export_job(
                "materialize_and_copy_report_pdf",
                Path(selected),
                lambda results_dir=results_dir, input_path=run_dir / "input.yaml", selected=selected: _materialize_and_copy_report_pdf_snapshot(
                    results_dir=results_dir,
                    input_path=input_path,
                    destination=selected,
                ),
            )
            return
        _notify_information(self, QMessageBox, "PDF計算書がありません。解析を再実行してください。")
        return
    selected, _ = QFileDialog.getSaveFileName(self, "計算書PDF保存", str(source), "PDF (*.pdf)")
    if not selected:
        return
    out = Path(selected)
    out.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != out.resolve():
        shutil.copyfile(source, out)
    self.result_view.setPlainText(f"PDF計算書を保存しました:\n{out}\n\n固定化manifest:\n{results_dir / 'calculation_report_manifest.json'}")
    self.tabs.setCurrentWidget(self.result_view)
    self.report_summary.setText(f"PDF計算書を保存しました: {out}")

def _set_result_view_file(owner: Any, qt: Mapping[str, Any], path: Path) -> None:
    self = owner
    preview = preview_text_file(path)
    self.result_view.setPlainText(preview.text)
    self.tabs.setCurrentWidget(self.result_view)
    if preview.truncated:
        self.append_log(f"[GUI] 表示を先頭{preview.preview_bytes} bytesに制限しました: {path}")

def _srm_post_rows(owner: Any, qt: Mapping[str, Any]) -> list[dict[str, str]]:
    self = owner
    stage_dir = self._latest_stage_dir()
    if stage_dir is not None:
        path = stage_dir / "element_stress.csv"
        if path.exists():
            return self._read_csv_rows(path)
    if self.result_rows and any("element_id" in row for row in self.result_rows):
        return [dict(row) for row in self.result_rows]
    return []

def _srm_post_summary_text(owner: Any, qt: Mapping[str, Any], rows: list[dict[str, str]]) -> str:
    self = owner
    srm = self._current_srm_info()
    fos = srm.get("factor_of_safety", self._current_srm_factor())
    fl_limit = self._line_edit_float("srm_fl_limit", 1.05)
    plastic_threshold = self._line_edit_float("srm_plastic_threshold", 0.5)
    active = [row for row in rows if str(row.get("active", "1")).strip().lower() not in {"0", "0.0", "false", "no"}]
    active_count = len(active)
    fl_values = [self._row_float(row, "FL", 99.0) for row in active]
    critical_fl = sum(1 for value in fl_values if (99.0 if value is None else value) <= 1.0)
    low_fl = sum(1 for value in fl_values if (99.0 if value is None else value) <= fl_limit)
    plastic = sum(1 for row in active if (self._row_float(row, "plastic", 0.0) or 0.0) >= plastic_threshold)
    top = self.srm_slip_candidates[0] if self.srm_slip_candidates else {}
    top_text = ""
    if top:
        top_text = (
            f" / 主すべり面: 要素 {' '.join(str(eid) for eid in top.get('elements', []))}, "
            f"minFL={self._format_float(top.get('min_fl'))}, length={self._format_float(top.get('length'))}"
        )
    fos_text = self._format_float(fos) if fos not in (None, "") else "未計算"
    return (
        f"SRM安全率: {fos_text} / FL<=1: {critical_fl}/{active_count} / "
        f"FL<={fl_limit:g}: {low_fl}/{active_count} / 塑性化: {plastic}/{active_count} / "
        f"候補数: {len(self.srm_slip_candidates)} / 試行数: {len(self.srm_trial_rows)}{top_text}"
    )

def _set_result_component_without_reload(owner: Any, qt: Mapping[str, Any], value: str) -> None:
    self = owner
    combo = getattr(self, "result_component", None)
    if combo is None:
        return
    combo.blockSignals(True)
    self._set_combo(combo, value)
    combo.blockSignals(False)

def _set_result_table_component_without_reload(owner: Any, qt: Mapping[str, Any], value: str) -> None:
    self = owner
    combo = getattr(self, "result_table_component", None)
    if combo is None:
        return
    combo.blockSignals(True)
    self._set_combo(combo, value)
    combo.blockSignals(False)

def _populate_result_table(owner: Any, qt: Mapping[str, Any], rows: list[dict[str, str]]) -> None:
    self = owner
    self.result_table_csv_path = None
    self.result_table_csv_summary = None
    self.result_rows = rows
    self.result_headers = list(rows[0]) if rows else []
    self.result_table_page_index = 0
    self._render_result_table_page()

def _populate_result_table_from_csv(owner: Any, qt: Mapping[str, Any], path: Path, summary: CsvTableSummary, *, page: ResultTablePage | None = None) -> None:
    self = owner
    self.result_table_csv_path = Path(path)
    self.result_table_csv_summary = summary
    self.result_rows = []
    self.result_headers = list(summary.headers)
    self.result_table_page_index = page.page_index if page is not None else 0
    self._render_result_table_page(page=page)

def result_table_previous_page(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    if self.result_table_page_index <= 0:
        return
    self.result_table_page_index -= 1
    self._render_result_table_page()

def result_table_next_page(owner: Any, qt: Mapping[str, Any]) -> None:
    self = owner
    page_count = self._result_table_page_count()
    if self.result_table_page_index >= page_count - 1:
        return
    self.result_table_page_index += 1
    self._render_result_table_page()

def _set_result_table_model_page(
    table: Any,
    rows: list[dict[str, str]],
    headers: list[str],
    QHeaderView: Any,
    QTableWidgetItem: Any,
) -> None:
    if hasattr(table, "set_result_page"):
        table.set_result_page(rows, headers)
        return
    table.setUpdatesEnabled(False)
    try:
        table.setRowCount(0)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row, row_data in enumerate(rows):
            for col, header in enumerate(headers):
                table.setItem(row, col, QTableWidgetItem(str(row_data.get(header, ""))))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    finally:
        table.setUpdatesEnabled(True)

def _render_result_table_page(owner: Any, qt: Mapping[str, Any], *, page: ResultTablePage | None = None) -> None:
    self = owner
    QHeaderView = qt["QHeaderView"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    if page is not None:
        pass
    elif self.result_table_csv_path is not None and self.result_table_csv_summary is not None:
        page = read_csv_table_page(
            self.result_table_csv_path,
            page_index=self.result_table_page_index,
            page_size=self.result_table_page_size,
            total_rows=self.result_table_csv_summary.row_count,
        )
    else:
        page = result_table_page(self.result_rows, page_index=self.result_table_page_index, page_size=self.result_table_page_size)
    self.result_table_page_index = page.page_index
    self.result_headers = page.headers
    if page.total_rows == 0:
        _set_result_table_model_page(self.result_table, [], page.headers, QHeaderView, QTableWidgetItem)
        self._refresh_result_component_options([])
        if hasattr(self, "result_table_page_label"):
            self.result_table_page_label.setText("数値表: 0件")
        return
    headers = page.headers
    self._refresh_result_component_options(headers)
    _set_result_table_model_page(self.result_table, page.rows, headers, QHeaderView, QTableWidgetItem)
    if hasattr(self, "result_table_page_label"):
        suffix = f" page {page.page_index + 1}/{page.page_count}" if page.page_count > 1 else ""
        self.result_table_page_label.setText(f"数値表: {page.label}{suffix}")

def _result_table_page_count(owner: Any, qt: Mapping[str, Any]) -> int:
    self = owner
    if self.result_table_csv_summary is not None:
        total = self.result_table_csv_summary.row_count
    else:
        total = len(self.result_rows)
    return max(1, (total + self.result_table_page_size - 1) // self.result_table_page_size)

def _can_stream_result_table(owner: Any, qt: Mapping[str, Any], kind: str) -> bool:
    full_rows_required = {
        "displacements",
        "displacement_contour",
        "displacement_vectors",
        "element_stress",
        "plastic",
        "safety_factor",
        "pore_pressure",
        "riks_path",
        "analysis_log",
        "performance",
    }
    return kind not in full_rows_required

def _refresh_result_component_options(owner: Any, qt: Mapping[str, Any], headers: list[str]) -> None:
    self = owner
    numeric = _numeric_result_headers(headers)
    combo = getattr(self, "result_table_component", None)
    if combo is not None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(numeric)
        if current and current in numeric:
            combo.setCurrentText(current)
        elif "u_norm" in numeric:
            combo.setCurrentText("u_norm")
        elif "q" in numeric:
            combo.setCurrentText("q")
        elif numeric:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
    if "element_id" not in headers:
        return
    stress_combo = getattr(self, "result_component", None)
    if stress_combo is None:
        return
    current = stress_combo.currentText()
    stress_combo.blockSignals(True)
    stress_combo.clear()
    stress_combo.addItems(numeric)
    if current and current in numeric:
        stress_combo.setCurrentText(current)
    elif self.post_component in numeric:
        stress_combo.setCurrentText(self.post_component)
    elif "q" in numeric:
        stress_combo.setCurrentText("q")
    elif numeric:
        stress_combo.setCurrentIndex(0)
    stress_combo.blockSignals(False)

__all__ = [
    "POST_REPORT_CONTROLLER_METHODS",
    "post_report_controller_contract",
    *POST_REPORT_CONTROLLER_METHODS,
]
