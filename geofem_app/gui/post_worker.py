"""Detached Post-processing worker helpers for GUI operations."""

from __future__ import annotations

from array import array
import copy
import csv
import math
from pathlib import Path
from typing import Any, Mapping

from geofem_app.gui.result_paging import DEFAULT_RESULT_TABLE_PAGE_SIZE


class _CurrentTextValue:
    def __init__(self, value: Any) -> None:
        self._value = "" if value is None else str(value)

    def currentText(self) -> str:
        return self._value


class _TextValue:
    def __init__(self, value: Any) -> None:
        self._value = "" if value is None else str(value)

    def text(self) -> str:
        return self._value


class _CheckValue:
    def __init__(self, value: Any) -> None:
        self._value = bool(value)

    def isChecked(self) -> bool:
        return self._value


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


def _displacement_map_from_csv(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def _optional_displacement_overlay(context: Any, result_stage_dir: str) -> dict[str, tuple[float, float]]:
    if not result_stage_dir:
        return {}
    path = Path(result_stage_dir) / "displacements.csv"
    try:
        return _displacement_map_from_csv(path)
    except Exception:
        return {}


def _numeric_result_headers(headers: list[str]) -> list[str]:
    ignored = {"node_id", "element_id", "id", "type", "material", "integration", "state", "active", "active_set", "dof", "x", "y", "z"}
    return [header for header in headers if header not in ignored]


def _resolve_result_field(rows: list[dict[str, str]], requested: str, preferred: tuple[str, ...]) -> str:
    numeric = _numeric_result_headers(list(rows[0]) if rows else [])
    return _resolve_result_field_from_headers(numeric, requested, preferred)


def _resolve_result_field_from_headers(headers: list[str], requested: str, preferred: tuple[str, ...]) -> str:
    numeric = _numeric_result_headers(headers)
    requested = str(requested or "").strip()
    if requested in numeric:
        return requested
    for field in preferred:
        if field in numeric:
            return field
    return numeric[0] if numeric else requested


def load_post_table_snapshot(
    window_cls: type[Any],
    cfg: Mapping[str, Any],
    *,
    path: str | Path,
    kind: str,
    result_component: str,
    table_component: str,
    last_run_dir: str = "",
    result_stage_dir: str = "",
    page_index: int = 0,
    page_size: int = DEFAULT_RESULT_TABLE_PAGE_SIZE,
) -> dict[str, Any]:
    """Read a result CSV and prepare lightweight Post display state."""

    source = Path(path)
    context = _post_context(
        window_cls,
        cfg,
        result_component=result_component,
        table_component=table_component,
        last_run_dir=last_run_dir,
        result_stage_dir=result_stage_dir,
    )
    display_component = ""
    colormap = ""
    result_displacements: dict[str, tuple[float, float]] = {}
    result_element_values: dict[str, float] = {}
    result_node_values: dict[str, float] = {}
    result_distribution: list[tuple[float, float]] = []
    post_component = ""
    post_mode = "table"

    safe_size = max(1, int(page_size))
    safe_index = max(0, int(page_index))
    page_start = safe_index * safe_size
    page_end = page_start + safe_size
    page_rows: list[dict[str, str]] = []
    headers: list[str] = []
    table_headers: list[str] = []
    row_count = 0
    numeric_candidates: set[str] = set()
    minimums: dict[str, float] = {}
    maximums: dict[str, float] = {}
    value_field = ""
    distribution_x_field = ""
    distribution_y_field = ""
    component_id_field = ""
    component_ids: list[str] = []
    component_columns: dict[str, array] = {}

    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        table_headers = list(headers)
        if kind == "safety_factor":
            for extra in ("FL", "safety_factor_note"):
                if extra not in table_headers:
                    table_headers.append(extra)
        numeric_candidates = set(table_headers)

        if kind in {"displacements", "displacement_contour", "displacement_vectors"}:
            value_field = table_component if table_component in {"ux", "uy", "u_norm", "settlement"} else "u_norm"
            post_component = value_field
            post_mode = "node_contour" if kind == "displacement_contour" else ("vector" if kind == "displacement_vectors" else "deformed")
        elif kind in {"element_stress", "plastic"}:
            value_field = "plastic" if kind == "plastic" else _resolve_result_field_from_headers(headers, result_component or "q", ("q", "p", "sigma_x", "sigma_y"))
            post_component = value_field
            display_component = "" if kind == "plastic" else value_field
            post_mode = "plastic" if kind == "plastic" else "contour"
        elif kind == "safety_factor":
            value_field = "FL"
            post_component = "FL"
            display_component = "FL"
            colormap = "Safety FL"
            post_mode = "contour"
        elif kind == "pore_pressure":
            value_field = "pore_pressure"
            post_component = "pore_pressure"
            post_mode = "node_contour"
        elif kind == "riks_path":
            distribution_y_field, distribution_x_field = _resolve_distribution_fields(headers, table_component, ("lambda", "load_factor", "arc_length", "step"))
            post_component = table_component or "Riks path"
            post_mode = "distribution"
        elif kind == "analysis_log":
            distribution_y_field, distribution_x_field = _resolve_distribution_fields(headers, table_component, ("residual_norm", "pressure_residual_norm", "iteration", "step"))
            post_component = table_component or "residual_norm"
            post_mode = "distribution" if distribution_y_field else "table"
        elif kind == "performance":
            distribution_y_field, distribution_x_field = _resolve_distribution_fields(headers, table_component, ("elapsed_seconds", "matrix_nnz", "solver_iterations"))
            post_component = table_component or "elapsed_seconds"
            post_mode = "distribution" if distribution_y_field else "table"

        if kind in {"element_stress", "plastic", "safety_factor"}:
            component_id_field = "element_id"
        elif kind in {"displacements", "displacement_contour", "displacement_vectors", "pore_pressure"}:
            component_id_field = "node_id"
        if component_id_field:
            component_columns = {field: array("d") for field in _numeric_result_headers(table_headers)}

        for index, raw_row in enumerate(reader):
            row = dict(raw_row)
            if kind == "safety_factor":
                row = _with_safety_factor_row(context, row)
            if component_id_field:
                component_ids.append(str(row.get(component_id_field, "") or ""))
                for field, values in component_columns.items():
                    try:
                        value = float(row.get(field, math.nan))
                    except (TypeError, ValueError):
                        value = math.nan
                    values.append(value if math.isfinite(value) else math.nan)
            if page_start <= index < page_end:
                page_rows.append(dict(row))
            _update_numeric_summary(row, table_headers, numeric_candidates, minimums, maximums)
            if kind in {"displacements", "displacement_contour", "displacement_vectors"}:
                _add_displacement_row(result_displacements, row)
                _add_node_value_row(result_node_values, row, value_field)
            elif kind in {"element_stress", "plastic", "safety_factor"}:
                _add_element_value_row(result_element_values, row, value_field)
            elif kind == "pore_pressure":
                _add_node_value_row(result_node_values, row, value_field)
            elif distribution_y_field:
                point = _distribution_point(row, index, distribution_y_field, distribution_x_field)
                if point is not None:
                    result_distribution.append(point)
            row_count = index + 1

    if result_distribution:
        result_distribution.sort(key=lambda item: item[0])
    if kind not in {"displacements", "displacement_contour", "displacement_vectors"}:
        result_displacements = _optional_displacement_overlay(context, result_stage_dir)
    numeric_fields = [header for header in table_headers if header in numeric_candidates]
    page_count = max(1, (row_count + safe_size - 1) // safe_size)
    end_row = min(row_count, page_end)
    table_summary = {
        "path": str(source),
        "headers": table_headers,
        "row_count": row_count,
        "numeric_fields": numeric_fields,
        "minimums": {key: minimums[key] for key in numeric_fields if key in minimums},
        "maximums": {key: maximums[key] for key in numeric_fields if key in maximums},
    }
    table_page = {
        "rows": page_rows,
        "headers": table_headers,
        "page_index": safe_index,
        "page_size": safe_size,
        "total_rows": row_count,
        "page_count": page_count,
        "start_row": min(row_count, page_start),
        "end_row": end_row,
    }
    component_store = {
        "schema": "geofem.gui.post_component_store.v1",
        "path": str(source),
        "mtime_ns": source.stat().st_mtime_ns,
        "size_bytes": source.stat().st_size,
        "kind": kind,
        "id_field": component_id_field,
        "ids": tuple(component_ids),
        "columns": component_columns,
        "row_count": row_count,
        "storage_bytes": sum(len(values) * values.itemsize for values in component_columns.values()),
    }
    return {
        "kind": kind,
        "path": str(source),
        "rows": page_rows,
        "table_summary": table_summary,
        "table_page": table_page,
        "result_displacements": result_displacements,
        "result_element_values": result_element_values,
        "result_node_values": result_node_values,
        "result_distribution": result_distribution,
        "post_component": post_component,
        "post_mode": post_mode,
        "display_component": display_component,
        "colormap": colormap,
        "component_store": component_store,
    }


def materialize_post_component_snapshot(
    store: Mapping[str, Any],
    requested: str,
    *,
    preferred: tuple[str, ...] = ("q", "p", "sigma_x", "sigma_y"),
) -> dict[str, Any]:
    """Expand one compact Post component off the GUI thread without rereading CSV."""

    ids = tuple(str(item) for item in store.get("ids", ()))
    raw_columns = store.get("columns", {})
    columns = raw_columns if isinstance(raw_columns, Mapping) else {}
    requested_field = str(requested or "").strip()
    aliases = {
        "safety_factor": "FL",
        "factor_of_safety": "FL",
        "local_safety_factor": "FL",
    }
    requested_field = aliases.get(requested_field, requested_field)
    field = requested_field if requested_field in columns else next((name for name in preferred if name in columns), "")
    if not field and columns:
        field = str(next(iter(columns)))
    values = columns.get(field, ())
    count = min(len(ids), len(values))
    materialized = {
        ids[index]: (float(values[index]) if math.isfinite(float(values[index])) else 0.0)
        for index in range(count)
        if ids[index]
    }
    id_field = str(store.get("id_field", "") or "")
    return {
        "schema": "geofem.gui.post_component_snapshot.v1",
        "path": str(store.get("path", "") or ""),
        "kind": str(store.get("kind", "") or ""),
        "field": field,
        "requested": requested_field,
        "id_field": id_field,
        "result_element_values": materialized if id_field == "element_id" else {},
        "result_node_values": materialized if id_field == "node_id" else {},
        "value_count": len(materialized),
        "source": "compact_component_store",
    }


def _with_safety_factor_row(context: Any, row: dict[str, str]) -> dict[str, str]:
    try:
        rows = context._with_safety_factor_rows([row])
    except Exception:
        return row
    return dict(rows[0]) if rows else row


def _update_numeric_summary(
    row: Mapping[str, Any],
    headers: list[str],
    numeric_candidates: set[str],
    minimums: dict[str, float],
    maximums: dict[str, float],
) -> None:
    for header in headers:
        raw = row.get(header, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            numeric_candidates.discard(header)
            continue
        if not math.isfinite(value):
            numeric_candidates.discard(header)
            continue
        if header in numeric_candidates:
            minimums[header] = min(minimums.get(header, value), value)
            maximums[header] = max(maximums.get(header, value), value)


def _float_value(row: Mapping[str, Any], field: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(field, default) or default)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _add_displacement_row(out: dict[str, tuple[float, float]], row: Mapping[str, Any]) -> None:
    node_id = str(row.get("node_id", "") or "")
    if not node_id:
        return
    out[node_id] = (_float_value(row, "ux"), _float_value(row, "uy"))


def _add_node_value_row(out: dict[str, float], row: Mapping[str, Any], field: str) -> None:
    node_id = str(row.get("node_id", "") or "")
    if not node_id:
        return
    out[node_id] = _float_value(row, field)


def _add_element_value_row(out: dict[str, float], row: Mapping[str, Any], field: str) -> None:
    element_id = str(row.get("element_id", "") or "")
    if not element_id:
        return
    fallback = "q" if field != "q" else "sigma_1"
    out[element_id] = _float_value(row, field, _float_value(row, fallback, 0.0))


def _resolve_distribution_fields(headers: list[str], requested: str, preferred: tuple[str, ...]) -> tuple[str, str]:
    y_field = next((field for field in preferred if field in headers), "")
    requested = str(requested or "").strip()
    if requested and requested in headers:
        y_field = requested
    if not y_field:
        return "", ""
    if "x" in headers:
        x_field = "x"
    elif "step" in headers:
        x_field = "step"
    elif "lambda" in headers and y_field != "lambda":
        x_field = "lambda"
    elif "arc_length" in headers and y_field != "arc_length":
        x_field = "arc_length"
    else:
        x_field = ""
    return y_field, x_field


def _distribution_point(row: Mapping[str, Any], index: int, y_field: str, x_field: str) -> tuple[float, float] | None:
    try:
        xval = float(row.get(x_field, index) if x_field else index)
        yval = float(row.get(y_field, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(xval) and math.isfinite(yval)):
        return None
    return (xval, yval)


def load_srm_post_snapshot(
    window_cls: type[Any],
    cfg: Mapping[str, Any],
    *,
    last_run_dir: str = "",
    result_stage_dir: str = "",
    result_rows: list[dict[str, str]] | None = None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare SRM Post rows, FL values, and slip candidates off the GUI thread."""

    context = _post_context(
        window_cls,
        cfg,
        result_component="FL",
        table_component="FL",
        last_run_dir=last_run_dir,
        result_stage_dir=result_stage_dir,
        options=options,
    )
    context.result_rows = [dict(row) for row in (result_rows or [])]
    context.srm_slip_candidates = []
    context.srm_trial_rows = []
    rows = context._srm_post_rows()
    if not rows:
        return {"rows": [], "result_element_values": {}, "srm_slip_candidates": [], "srm_trial_rows": [], "summary_text": ""}
    rows = context._with_safety_factor_rows(rows)
    result_element_values = context._srm_element_fl_values(rows)
    srm_slip_candidates = context._estimate_srm_slip_candidates(rows)
    srm_trial_rows = context._current_srm_trials()
    context.srm_slip_candidates = srm_slip_candidates
    context.srm_trial_rows = srm_trial_rows
    summary_text = context._srm_post_summary_text(rows)
    return {
        "rows": rows,
        "result_element_values": result_element_values,
        "srm_slip_candidates": srm_slip_candidates,
        "srm_trial_rows": srm_trial_rows,
        "summary_text": summary_text,
    }


def _post_context(
    window_cls: type[Any],
    cfg: Mapping[str, Any],
    *,
    result_component: str,
    table_component: str,
    last_run_dir: str = "",
    result_stage_dir: str = "",
    options: Mapping[str, Any] | None = None,
) -> Any:
    from geofem_app.gui.model_check_worker import DetachedGuiContext

    context = DetachedGuiContext(window_cls, copy.deepcopy(dict(cfg)))
    context.last_run_dir = Path(last_run_dir) if last_run_dir else None
    context.result_stage_dir = Path(result_stage_dir) if result_stage_dir else None
    context.result_component = _CurrentTextValue(result_component)
    context.result_table_component = _CurrentTextValue(table_component)
    opts = dict(options or {})
    context.srm_fl_limit = _TextValue(opts.get("fl_limit", "1.05"))
    context.srm_plastic_threshold = _TextValue(opts.get("plastic_threshold", "0.5"))
    context.srm_local_fl_aggregation = _CurrentTextValue(opts.get("local_fl_aggregation", "mean"))
    context.srm_search_mode = _CurrentTextValue(opts.get("search_mode", "all"))
    context.srm_slope_direction = _CurrentTextValue(opts.get("slope_direction", "auto"))
    context.srm_min_candidate_length = _TextValue(opts.get("min_candidate_length", "0.0"))
    context.srm_max_circle_radius = _TextValue(opts.get("max_circle_radius", "0.0"))
    context.srm_require_boundary_exit = _CheckValue(opts.get("require_boundary_exit", False))
    context.result_distribution = []
    return context


__all__ = ["load_post_table_snapshot", "load_srm_post_snapshot", "materialize_post_component_snapshot"]
