"""Structured failure classification for solver and GUI diagnostics."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


def classify_failure(error: BaseException | str, *, diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return user-actionable failure diagnostics for reports and GUI views."""

    error_message = str(error)
    error_type = type(error).__name__ if not isinstance(error, str) else "Error"
    findings = _findings_from_input_diagnostics(diagnostics or {})
    if not findings:
        findings.append(_finding_from_error_message(error_message))
    primary = _primary_finding(findings)
    return {
        "schema": "geofem.failure_diagnostics.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "error_type": error_type,
        "error_message": error_message,
        "primary_category": primary["category"],
        "primary_cause": primary["cause"],
        "primary_stage": primary.get("stage", ""),
        "primary_target_id": primary.get("target_id", ""),
        "primary_input_path": primary.get("input_path", ""),
        "primary_gui_panel": primary.get("gui_panel", ""),
        "primary_recommended_fix": primary["recommended_fix"],
        "primary_rerun_condition": primary["rerun_condition"],
        "finding_count": len(findings),
        "findings": findings,
    }


def write_failure_diagnostics(analysis: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write machine-readable and human-readable failure diagnostics."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "failure_diagnostics.json"
    csv_path = out / "failure_diagnostics.csv"
    html_path = out / "failure_diagnostics.html"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(csv_path, analysis.get("findings", []))
    html_path.write_text(_html(analysis), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _findings_from_input_diagnostics(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues = diagnostics.get("issues", [])
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        severity = str(issue.get("severity", "ERROR")).upper()
        if severity not in {"ERROR", "WARN"}:
            continue
        path = str(issue.get("path", ""))
        message = str(issue.get("message", ""))
        suggestion = str(issue.get("suggestion", ""))
        category = _category_for(path=path, message=message, source="input_diagnostics")
        rows.append(
            _finding(
                severity=severity,
                category=category,
                source="input_diagnostics",
                message=message,
                input_path=path,
                stage=_stage_from_path(path),
                target_id=_target_from_path_or_message(path, message),
                recommended_fix=suggestion or _recommendation(category),
            )
        )
    errors = [row for row in rows if row["severity"] == "ERROR"]
    return errors or rows


def _finding_from_error_message(message: str) -> dict[str, Any]:
    category = _category_for(path="", message=message, source="exception")
    return _finding(
        severity="ERROR",
        category=category,
        source="exception",
        message=message,
        input_path=_input_path_hint(message),
        stage=_stage_from_message(message),
        target_id=_target_from_path_or_message("", message),
        recommended_fix=_recommendation(category),
    )


def _finding(
    *,
    severity: str,
    category: str,
    source: str,
    message: str,
    input_path: str = "",
    stage: str = "",
    target_id: str = "",
    recommended_fix: str = "",
) -> dict[str, Any]:
    spec = _CATEGORY_SPECS.get(category, _CATEGORY_SPECS["unknown"])
    return {
        "severity": severity,
        "category": category,
        "cause": spec["cause"],
        "source": source,
        "stage": stage,
        "target_id": target_id,
        "input_path": input_path,
        "gui_panel": _gui_panel_for_path(input_path, category),
        "message": message,
        "recommended_fix": recommended_fix or spec["recommended_fix"],
        "rerun_condition": spec["rerun_condition"],
    }


def _primary_finding(findings: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in findings:
        if str(row.get("severity", "")).upper() == "ERROR":
            return row
    return findings[0] if findings else _finding_from_error_message("unknown failure")


def _category_for(*, path: str, message: str, source: str) -> str:
    text = f"{path} {message}".lower()
    if source == "input_diagnostics" and path:
        if path.startswith(("materials", "material")):
            return "material_parameter"
        if path.startswith(("boundary_conditions", "bc")) or "constraint" in text:
            return "boundary_condition"
        if path.startswith(("stages", "steps")):
            return "stage_definition"
        if path.startswith("loads"):
            return "load_definition"
        if path.startswith("mesh"):
            return "mesh_definition"
        if path.startswith("analysis"):
            return "analysis_setting"
    if any(word in text for word in ("did not converge", "nonlinear step", "cutback", "convergence")):
        return "convergence_failure"
    if any(word in text for word in ("singular", "direct solver failed", "sparse solve", "non-finite displacement", "non-finite values")):
        return "numerical_failure"
    if any(word in text for word in ("detj", "non-positive element", "edge length", "unknown nodes", "mesh coordinates")):
        return "mesh_quality"
    if any(word in text for word in ("material", "poisson", "cohesion", "yield_stress", "tensile_strength", "unsupported 2d core material")):
        return "material_parameter"
    if any(word in text for word in ("boundary", "constraint", "mpc", "fixed displacement")):
        return "boundary_condition"
    if "stage" in text:
        return "stage_definition"
    if "input diagnostics failed" in text:
        return "input_error"
    return "unknown"


def _recommendation(category: str) -> str:
    return _CATEGORY_SPECS.get(category, _CATEGORY_SPECS["unknown"])["recommended_fix"]


def _stage_from_path(path: str) -> str:
    match = re.search(r"(?:stages|steps)\[(\d+)\]", path)
    return f"stages[{match.group(1)}]" if match else ""


def _stage_from_message(message: str) -> str:
    match = re.match(r"\s*([^:]{1,80}):\s+", message)
    if not match:
        return ""
    candidate = match.group(1).strip()
    if candidate.lower() in {"error", "warning", "geofem"}:
        return ""
    return candidate


def _input_path_hint(message: str) -> str:
    match = re.search(r"input diagnostics failed:\s*([^:]+):", message)
    return match.group(1).strip() if match else ""


def _target_from_path_or_message(path: str, message: str) -> str:
    if path:
        for pattern in (
            r"materials\.([^.[]+)",
            r"mesh\.elements\[(\d+)\]",
            r"boundary_conditions\[(\d+)\]",
            r"loads\[(\d+)\]",
            r"(?:stages|steps)\[(\d+)\]",
        ):
            match = re.search(pattern, path)
            if match:
                return match.group(1)
    for pattern in (
        r"\b(element|node|material|interface|structural element)\s+([A-Za-z0-9_.:-]+)",
        r"\bunknown elements:\s*([A-Za-z0-9_.:-]+)",
        r"\bunknown nodes\s+([A-Za-z0-9_.:-]+)",
    ):
        match = re.search(pattern, message)
        if match:
            return match.group(match.lastindex or 1).strip(" .,:;")
    return ""


def _gui_panel_for_path(path: str, category: str) -> str:
    if path.startswith("analysis"):
        return "analysis"
    if path.startswith("mesh"):
        return "mesh"
    if path.startswith(("materials", "material")):
        return "materials"
    if path.startswith(("boundary_conditions", "bc")):
        return "boundary_conditions"
    if path.startswith("loads"):
        return "loads"
    if path.startswith(("stages", "steps")):
        return "stages"
    if category in {"convergence_failure", "numerical_failure"}:
        return "solver"
    if category == "mesh_quality":
        return "mesh"
    if category == "material_parameter":
        return "materials"
    if category == "boundary_condition":
        return "boundary_conditions"
    return "model_check"


def _write_csv(path: Path, rows: Any) -> None:
    fields = ["severity", "category", "cause", "stage", "target_id", "input_path", "gui_panel", "message", "recommended_fix", "rerun_condition", "source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []:
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fields})


def _html(analysis: Mapping[str, Any]) -> str:
    rows = []
    for row in analysis.get("findings", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('severity', '')))}</td>"
            f"<td>{html.escape(str(row.get('category', '')))}</td>"
            f"<td>{html.escape(str(row.get('cause', '')))}</td>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{html.escape(str(row.get('target_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('input_path', '')))}</td>"
            f"<td>{html.escape(str(row.get('gui_panel', '')))}</td>"
            f"<td>{html.escape(str(row.get('recommended_fix', '')))}</td>"
            f"<td>{html.escape(str(row.get('rerun_condition', '')))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\"><title>GeoFEM failure diagnostics</title>"
        "<style>body{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top;}th{background:#f2f2f2;}</style></head><body>"
        "<h1>解析エラー診断</h1>"
        f"<p>主原因: {html.escape(str(analysis.get('primary_cause', '')))} / "
        f"GUI画面: {html.escape(str(analysis.get('primary_gui_panel', '')))}</p>"
        "<table><thead><tr><th>severity</th><th>category</th><th>cause</th><th>stage</th><th>target</th>"
        "<th>input path</th><th>GUI panel</th><th>recommended fix</th><th>rerun condition</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )


_CATEGORY_SPECS: dict[str, dict[str, str]] = {
    "analysis_setting": {
        "cause": "解析条件の指定に不整合があります。",
        "recommended_fix": "解析条件画面で次元、単位系、解析種別を確認してください。",
        "rerun_condition": "解析条件を保存し、モデルチェックがERRORなしになったら再実行してください。",
    },
    "input_error": {
        "cause": "入力診断で必須条件を満たしていません。",
        "recommended_fix": "入力診断のpathとsuggestionに沿ってYAMLまたは該当フォームを修正してください。",
        "rerun_condition": "入力診断のERRORが0件になったら再実行してください。",
    },
    "mesh_definition": {
        "cause": "メッシュ定義または要素参照に不備があります。",
        "recommended_fix": "メッシュ画面で節点、要素、要素タイプ、分割数、材料参照を確認してください。",
        "rerun_condition": "メッシュ品質と入力診断のERRORが0件になったら再実行してください。",
    },
    "mesh_quality": {
        "cause": "メッシュ形状が解析に不適切です。",
        "recommended_fix": "負のJacobian、潰れ要素、未知節点、過大アスペクト比を修正してください。",
        "rerun_condition": "メッシュ品質レポートの重大違反を解消してから再実行してください。",
    },
    "material_parameter": {
        "cause": "材料パラメータに不足または範囲外の値があります。",
        "recommended_fix": "材料画面でE、nu、強度、密度、液状化パラメータなどの必須値と範囲を確認してください。",
        "rerun_condition": "材料診断と入力診断のERRORが0件になったら再実行してください。",
    },
    "boundary_condition": {
        "cause": "境界条件、拘束、MPCに不備または競合があります。",
        "recommended_fix": "剛体運動を止める拘束、重複拘束、MPCのslave/master、固定変位との競合を確認してください。",
        "rerun_condition": "拘束不足や拘束競合を解消し、モデルチェックがERRORなしになったら再実行してください。",
    },
    "load_definition": {
        "cause": "荷重定義または荷重対象に不備があります。",
        "recommended_fix": "荷重画面で対象set、節点、辺、荷重成分、荷重ケースを確認してください。",
        "rerun_condition": "荷重の対象と成分が有効になったら再実行してください。",
    },
    "stage_definition": {
        "cause": "ステージ定義またはステージ対象に不備があります。",
        "recommended_fix": "ステージ画面でtype、対象set、施工イベント、荷重、境界条件を確認してください。",
        "rerun_condition": "ステージ入力診断のERRORが0件になったら再実行してください。",
    },
    "convergence_failure": {
        "cause": "非線形反復が収束していません。",
        "recommended_fix": "荷重増分を小さくし、line search、cutback、許容残差、材料強度、メッシュ集中を確認してください。",
        "rerun_condition": "収束履歴の残差低下と支配自由度を確認し、増分/材料/拘束を調整して再実行してください。",
    },
    "numerical_failure": {
        "cause": "数値解法が破綻しました。",
        "recommended_fix": "特異行列、拘束不足、過拘束、非有限値、線形ソルバー設定を確認してください。",
        "rerun_condition": "拘束条件とメッシュ品質を修正し、モデルチェックがERRORなしになったら再実行してください。",
    },
    "unknown": {
        "cause": "未分類の解析エラーです。",
        "recommended_fix": "failure_report.html、traceback、入力診断、直前の解析ログを確認してください。",
        "rerun_condition": "原因を修正し、入力診断とモデルチェックが通ったら再実行してください。",
    },
}


__all__ = ["classify_failure", "write_failure_diagnostics"]
