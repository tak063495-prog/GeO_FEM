"""Case-to-case result comparison artifacts for completed GeoFEM runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


JSON_NUMERIC_FILES = (
    "summary.json",
    "performance_summary.json",
    "performance_kpi_matrix.json",
    "reliability_summary.json",
    "case_manifest.json",
    "calculation_report_manifest.json",
)
REPORT_SUFFIXES = (".html", ".pdf")
POST_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
CSV_SKIP_NAMES = {"case_output_comparison.csv"}


def build_case_output_comparison(
    current_dir: str | Path,
    baseline_dir: str | Path,
    *,
    current_label: str = "current",
    baseline_label: str = "baseline",
    abs_tolerance: float = 1.0e-9,
    rel_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare two completed result directories by numeric tables, Post images, and reports."""

    current = _normalize_result_dir(current_dir)
    baseline = _normalize_result_dir(baseline_dir)
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    _add_dir_check(checks, current, current_label)
    _add_dir_check(checks, baseline, baseline_label)
    if current.exists() and baseline.exists():
        rows.extend(_compare_json_numeric_files(current, baseline, abs_tolerance, rel_tolerance))
        rows.extend(_compare_csv_files(current, baseline, abs_tolerance, rel_tolerance))
        rows.extend(_compare_post_images(current, baseline))
        rows.extend(_compare_report_files(current, baseline))
    status_counts = _status_counts(rows)
    missing_count = status_counts.get("missing", 0)
    different_count = sum(status_counts.get(key, 0) for key in ("different", "shape_mismatch"))
    error_count = sum(1 for row in checks if row["status"] == "ERROR")
    return {
        "schema": "geofem.case_output_comparison.v1",
        "current": {"label": current_label, "path": str(current)},
        "baseline": {"label": baseline_label, "path": str(baseline)},
        "features": [
            "arbitrary_result_directory_pair",
            "previous_baseline_design_case_labels",
            "numeric_json_scalar_diff",
            "csv_numeric_table_diff",
            "post_image_diff_or_hash_fallback",
            "report_artifact_hash_diff",
            "gui_table_ready_rows",
        ],
        "passed": error_count == 0,
        "completed": error_count == 0,
        "error_count": error_count,
        "difference_count": different_count,
        "missing_count": missing_count,
        "row_count": len(rows),
        "status_counts": status_counts,
        "checks": checks,
        "rows": rows,
    }


def write_case_output_comparison(
    comparison: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON/CSV/HTML comparison artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out / "case_output_comparison.json"),
        "csv": str(out / "case_output_comparison.csv"),
        "html": str(out / "case_output_comparison.html"),
    }
    write_json_artifact(paths["json"], comparison)
    write_dict_rows_csv(
        paths["csv"],
        [row for row in comparison.get("rows", []) if isinstance(row, Mapping)],
        ["category", "artifact", "metric", "status", "current", "baseline", "abs_difference", "rel_difference", "detail"],
    )
    write_html_artifact(paths["html"], _comparison_html(comparison))
    return paths


def compare_result_cases(
    current_dir: str | Path,
    baseline_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    current_label: str = "current",
    baseline_label: str = "baseline",
    abs_tolerance: float = 1.0e-9,
    rel_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Build and optionally write a case-output comparison report."""

    comparison = build_case_output_comparison(
        current_dir,
        baseline_dir,
        current_label=current_label,
        baseline_label=baseline_label,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    if output_dir is not None:
        paths = write_case_output_comparison(comparison, output_dir)
        comparison = {**comparison, "paths": paths}
    return comparison


def _normalize_result_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if (candidate / "summary.json").exists() or (candidate / "case_manifest.json").exists():
        return candidate
    if (candidate / "results" / "summary.json").exists() or (candidate / "results" / "case_manifest.json").exists():
        return candidate / "results"
    return candidate


def _add_dir_check(checks: list[dict[str, Any]], path: Path, label: str) -> None:
    checks.append(
        {
            "name": f"result_dir:{label}",
            "status": "OK" if path.exists() and path.is_dir() else "ERROR",
            "path": str(path),
            "detail": "result directory exists" if path.exists() and path.is_dir() else "result directory is missing",
        }
    )


def _compare_json_numeric_files(current: Path, baseline: Path, abs_tol: float, rel_tol: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in JSON_NUMERIC_FILES:
        left_path = current / name
        right_path = baseline / name
        if not left_path.exists() and not right_path.exists():
            continue
        if not left_path.exists() or not right_path.exists():
            rows.append(_row("numeric", name, "file", "missing", _exists_text(left_path), _exists_text(right_path), "", "", "file is present on only one side"))
            continue
        left = _read_json(left_path)
        right = _read_json(right_path)
        left_values = _flatten_numeric(left)
        right_values = _flatten_numeric(right)
        keys = sorted(set(left_values) | set(right_values))
        if not keys:
            continue
        changed = 0
        max_abs = 0.0
        max_rel = 0.0
        max_key = ""
        for key in keys:
            if key not in left_values or key not in right_values:
                changed += 1
                max_key = max_key or key
                continue
            diff = abs(left_values[key] - right_values[key])
            rel = diff / max(abs(right_values[key]), abs(left_values[key]), 1.0)
            if diff > max_abs:
                max_abs = diff
                max_rel = rel
                max_key = key
            if diff > abs_tol and rel > rel_tol:
                changed += 1
        status = "same" if changed == 0 else "different"
        rows.append(_row("numeric", name, max_key or "numeric_scalars", status, len(left_values), len(right_values), max_abs, max_rel, f"changed scalars={changed}"))
    return rows


def _compare_csv_files(current: Path, baseline: Path, abs_tol: float, rel_tol: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    left_files = _discover_relative_files(current, (".csv",))
    right_files = _discover_relative_files(baseline, (".csv",))
    for rel in sorted(set(left_files) | set(right_files)):
        if Path(rel).name in CSV_SKIP_NAMES:
            continue
        left_path = current / rel
        right_path = baseline / rel
        if not left_path.exists() or not right_path.exists():
            rows.append(_row("csv", rel, "file", "missing", _exists_text(left_path), _exists_text(right_path), "", "", "CSV is present on only one side"))
            continue
        left_rows = _read_csv_rows(left_path)
        right_rows = _read_csv_rows(right_path)
        status = "same" if len(left_rows) == len(right_rows) else "shape_mismatch"
        headers = sorted(set(_headers(left_rows)) & set(_headers(right_rows)))
        comparable = 0
        changed = 0
        max_abs = 0.0
        max_rel = 0.0
        max_metric = "row_count"
        for header in headers:
            values = []
            for left_row, right_row in zip(left_rows, right_rows):
                left_value = _to_float(left_row.get(header, ""))
                right_value = _to_float(right_row.get(header, ""))
                if left_value is None or right_value is None:
                    continue
                values.append((left_value, right_value))
            if not values:
                continue
            comparable += len(values)
            for left_value, right_value in values:
                diff = abs(left_value - right_value)
                rel_diff = diff / max(abs(right_value), abs(left_value), 1.0)
                if diff > max_abs:
                    max_abs = diff
                    max_rel = rel_diff
                    max_metric = header
                if diff > abs_tol and rel_diff > rel_tol:
                    changed += 1
        if changed:
            status = "different"
        detail = f"rows={len(left_rows)}/{len(right_rows)}, comparable numeric cells={comparable}, changed cells={changed}"
        rows.append(_row("csv", rel, max_metric, status, len(left_rows), len(right_rows), max_abs, max_rel, detail))
    return rows


def _compare_post_images(current: Path, baseline: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    left_files = _discover_relative_files(current, POST_IMAGE_SUFFIXES)
    right_files = _discover_relative_files(baseline, POST_IMAGE_SUFFIXES)
    for rel in sorted(set(left_files) | set(right_files)):
        left_path = current / rel
        right_path = baseline / rel
        if not left_path.exists() or not right_path.exists():
            rows.append(_row("post", rel, "image", "missing", _exists_text(left_path), _exists_text(right_path), "", "", "Post image is present on only one side"))
            continue
        image_diff = _try_image_diff(left_path, right_path)
        if image_diff is not None:
            ratio = float(image_diff.get("diff_ratio", 1.0) or 0.0)
            status = "same" if bool(image_diff.get("ok", False)) else "different"
            rows.append(_row("post", rel, "pixel_diff", status, image_diff.get("current_size", ""), image_diff.get("baseline_size", ""), ratio, ratio, f"changed pixels={image_diff.get('changed_pixels', '')}"))
        else:
            same = _sha256(left_path) == _sha256(right_path)
            rows.append(_row("post", rel, "sha256", "same" if same else "different", left_path.stat().st_size, right_path.stat().st_size, 0 if same else 1, 0 if same else 1, "image pixel diff unavailable; compared hashes"))
    return rows


def _compare_report_files(current: Path, baseline: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    left_files = _discover_relative_files(current, REPORT_SUFFIXES)
    right_files = _discover_relative_files(baseline, REPORT_SUFFIXES)
    for rel in sorted(set(left_files) | set(right_files)):
        left_path = current / rel
        right_path = baseline / rel
        if not left_path.exists() or not right_path.exists():
            rows.append(_row("report", rel, "file", "missing", _exists_text(left_path), _exists_text(right_path), "", "", "report artifact is present on only one side"))
            continue
        same = _sha256(left_path) == _sha256(right_path)
        size_diff = abs(left_path.stat().st_size - right_path.stat().st_size)
        rows.append(_row("report", rel, "sha256", "same" if same else "different", left_path.stat().st_size, right_path.stat().st_size, size_diff, "", "report bytes are identical" if same else "report bytes differ"))
    return rows


def _discover_relative_files(root: Path, suffixes: Sequence[str]) -> set[str]:
    if not root.exists():
        return set()
    suffix_set = {suffix.lower() for suffix in suffixes}
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffix_set:
            files.add(path.relative_to(root).as_posix())
    return files


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric(item, next_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(_flatten_numeric(item, f"{prefix}[{index}]"))
    else:
        number = _to_float(value)
        if number is not None and prefix:
            out[prefix] = number
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def _headers(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    headers: set[str] = set()
    for row in rows:
        headers.update(str(key) for key in row.keys())
    return headers


def _try_image_diff(current: Path, baseline: Path) -> dict[str, Any] | None:
    try:
        from .post_image_diff import compare_images

        result = compare_images(current, baseline)
        if str(result.get("reason", "")) == "missing image":
            return None
        return result
    except Exception:
        return None


def _row(
    category: str,
    artifact: str,
    metric: str,
    status: str,
    current: Any,
    baseline: Any,
    abs_difference: Any,
    rel_difference: Any,
    detail: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "artifact": artifact,
        "metric": metric,
        "status": status,
        "current": current,
        "baseline": baseline,
        "abs_difference": abs_difference,
        "rel_difference": rel_difference,
        "detail": detail,
    }


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        number = float(text)
        if number != number:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _exists_text(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def _comparison_html(comparison: Mapping[str, Any]) -> str:
    rows = [
        [
            row.get("category", ""),
            row.get("artifact", ""),
            row.get("metric", ""),
            row.get("status", ""),
            row.get("current", ""),
            row.get("baseline", ""),
            row.get("abs_difference", ""),
            row.get("rel_difference", ""),
            row.get("detail", ""),
        ]
        for row in comparison.get("rows", [])
        if isinstance(row, Mapping)
    ]
    lead = (
        f"current={comparison.get('current', {}).get('path', '')}; "
        f"baseline={comparison.get('baseline', {}).get('path', '')}; "
        f"differences={comparison.get('difference_count', 0)}, missing={comparison.get('missing_count', 0)}"
    )
    return html_table_document(
        title="GeoFEM ケース出力比較",
        lead=lead,
        headers=["category", "artifact", "metric", "status", "current", "baseline", "abs_difference", "rel_difference", "detail"],
        rows=rows,
    )


__all__ = [
    "build_case_output_comparison",
    "compare_result_cases",
    "write_case_output_comparison",
]
