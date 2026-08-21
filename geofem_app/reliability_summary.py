"""Human-readable reliability summary for solved 2D FEM cases."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fem2d_types import SolveResult2D, StageResult2D
from .mesh_quality import evaluate_mesh_quality


def build_reliability_summary(result: SolveResult2D) -> dict[str, Any]:
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    mesh_quality = evaluate_mesh_quality(result.mesh, cfg)
    stage_rows = [_stage_reliability_row(stage) for stage in result.stages]
    checks = _case_checks(result, stage_rows, mesh_quality)
    failed_required = [row for row in checks if row["severity"] == "ERROR" and not row["passed"]]
    warnings = [row for row in checks if row["severity"] == "WARN" and not row["passed"]]
    return {
        "schema": "geofem.reliability_summary.v1",
        "passed": not failed_required,
        "error_count": len(failed_required),
        "warning_count": len(warnings),
        "input": {
            "hash_source": "input_config_stable_json",
            "input_sha256": _stable_json_hash(_json_safe(cfg)),
            "node_count": len(result.mesh.node_ids),
            "element_count": len(result.mesh.elements),
            "stage_count": len(result.stages),
        },
        "mesh_quality": {
            "passed": bool(mesh_quality.get("passed", False)),
            "violation_count": int(_mapping(mesh_quality.get("summary", {})).get("violation_count", 0) or 0),
            "error_count": int(_mapping(mesh_quality.get("summary", {})).get("error_count", 0) or 0),
            "warning_count": int(_mapping(mesh_quality.get("summary", {})).get("warning_count", 0) or 0),
        },
        "features": [
            "convergence_status",
            "equilibrium_residual_and_boundary_reaction_summary",
            "energy_terms_when_available",
            "mass_balance_terms_when_available",
            "mesh_quality_summary",
            "input_hash",
        ],
        "checks": checks,
        "stages": stage_rows,
    }


def write_reliability_summary_reports(result: SolveResult2D, output_dir: str | Path | None = None) -> dict[str, str]:
    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = build_reliability_summary(result)
    json_path = out / "reliability_summary.json"
    csv_path = out / "reliability_summary.csv"
    html_path = out / "reliability_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_summary_csv(summary, csv_path)
    html_path.write_text(_summary_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _case_checks(result: SolveResult2D, stage_rows: list[dict[str, Any]], mesh_quality: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _add_check(checks, "input.hash", len(_stable_json_hash(_json_safe(result.input_config or {}))) == 64, "ERROR", "入力条件のハッシュを記録している")
    _add_check(checks, "stage.count", len(stage_rows) == len(result.stages) and len(stage_rows) > 0, "ERROR", "解析ステージの信頼性行が揃っている")
    _add_check(checks, "mesh_quality.passed", bool(mesh_quality.get("passed", False)), "WARN", "メッシュ品質のERRORがない")
    for row in stage_rows:
        name = str(row.get("stage", "stage"))
        _add_check(checks, f"stage.{name}.converged", bool(row.get("converged", True)), "ERROR", f"{name} の収束状態")
        residual = _number_or_none(row.get("residual_norm"))
        _add_check(checks, f"stage.{name}.residual_recorded", residual is not None, "WARN", f"{name} の残差ノルムが記録されている")
        _add_check(checks, f"stage.{name}.boundary_reaction", _number_or_none(row.get("max_abs_reaction")) is not None, "WARN", f"{name} の境界反力サマリが記録されている")
        if bool(row.get("has_mass_balance", False)):
            mass = abs(float(row.get("mass_balance", 0.0) or 0.0))
            _add_check(checks, f"stage.{name}.mass_balance", math.isfinite(mass), "WARN", f"{name} の質量収支指標が有限である")
        if bool(row.get("has_energy", False)):
            energy = float(row.get("total_energy", 0.0) or 0.0)
            _add_check(checks, f"stage.{name}.energy", math.isfinite(energy), "WARN", f"{name} のエネルギー指標が有限である")
    return checks


def _stage_reliability_row(stage: StageResult2D) -> dict[str, Any]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    reactions = np.asarray(stage.reactions, dtype=float).reshape(-1)
    rx = float(np.sum(reactions[0::2])) if reactions.size else 0.0
    ry = float(np.sum(reactions[1::2])) if reactions.size > 1 else 0.0
    max_abs = float(np.max(np.abs(reactions))) if reactions.size else 0.0
    mass = _mass_balance_value(solver)
    energy = _energy_terms(solver)
    return {
        "stage": stage.name,
        "converged": bool(solver.get("converged", True)),
        "iterations": int(solver.get("iterations", 0) or 0),
        "residual_norm": _number_or_blank(solver.get("residual_norm")),
        "pressure_residual_norm": _number_or_blank(solver.get("pressure_residual_norm")),
        "constraint_norm": _number_or_blank(solver.get("constraint_norm")),
        "reaction_sum_x": rx,
        "reaction_sum_y": ry,
        "max_abs_reaction": max_abs,
        "has_mass_balance": mass is not None,
        "mass_balance": "" if mass is None else mass,
        "mass_balance_residual_sum": _number_or_blank(_nested(solver, "consolidation", "mass_balance_residual_sum")),
        "has_energy": bool(energy),
        "strain_energy": energy.get("strain_energy", ""),
        "kinetic_energy": energy.get("kinetic_energy", ""),
        "damping_energy": energy.get("damping_energy", ""),
        "total_energy": _total_energy(energy),
    }


def _mass_balance_value(solver: Mapping[str, Any]) -> float | None:
    for source in (solver, _mapping(solver.get("consolidation")), _mapping(solver.get("dynamic"))):
        value = _number_or_none(source.get("mass_balance"))
        if value is not None:
            return value
    return None


def _energy_terms(solver: Mapping[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for source in (solver, _mapping(solver.get("dynamic")), _mapping(solver.get("energy"))):
        for key in ("strain_energy", "kinetic_energy", "damping_energy"):
            value = _number_or_none(source.get(key))
            if value is not None:
                terms[key] = value
    return terms


def _total_energy(energy: Mapping[str, float]) -> float | str:
    if not energy:
        return ""
    return float(sum(float(value) for value in energy.values()))


def _write_summary_csv(summary: Mapping[str, Any], path: Path) -> None:
    fields = ["section", "id", "stage", "severity", "passed", "metric", "value", "detail"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary.get("checks", []):
            if isinstance(row, Mapping):
                writer.writerow({"section": "check", **{field: row.get(field, "") for field in fields}})
        for row in summary.get("stages", []):
            if isinstance(row, Mapping):
                stage = row.get("stage", "")
                for key, value in row.items():
                    if key == "stage":
                        continue
                    writer.writerow({"section": "stage", "stage": stage, "metric": key, "value": value})
        for key, value in _mapping(summary.get("mesh_quality")).items():
            writer.writerow({"section": "mesh_quality", "metric": key, "value": value})


def _summary_html(summary: Mapping[str, Any]) -> str:
    checks = []
    for row in summary.get("checks", []):
        if not isinstance(row, Mapping):
            continue
        checks.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('id', '')))}</td>"
            f"<td>{html.escape(str(row.get('severity', '')))}</td>"
            f"<td>{html.escape(str(row.get('passed', '')))}</td>"
            f"<td>{html.escape(str(row.get('detail', '')))}</td>"
            "</tr>"
        )
    stages = []
    for row in summary.get("stages", []):
        if not isinstance(row, Mapping):
            continue
        stages.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{html.escape(str(row.get('converged', '')))}</td>"
            f"<td>{html.escape(str(row.get('residual_norm', '')))}</td>"
            f"<td>{html.escape(str(row.get('max_abs_reaction', '')))}</td>"
            f"<td>{html.escape(str(row.get('mass_balance', '')))}</td>"
            f"<td>{html.escape(str(row.get('total_energy', '')))}</td>"
            "</tr>"
        )
    input_hash = _mapping(summary.get("input")).get("input_sha256", "")
    mesh_quality = ", ".join(f"{key}={value}" for key, value in _mapping(summary.get("mesh_quality")).items())
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM Reliability Summary</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#f3f3f3}}</style></head>
<body><h1>信頼性サマリ</h1>
<p>passed={html.escape(str(summary.get('passed', '')))}, errors={html.escape(str(summary.get('error_count', '')))}, warnings={html.escape(str(summary.get('warning_count', '')))}</p>
<p>input_sha256={html.escape(str(input_hash))}</p>
<p>mesh_quality: {html.escape(mesh_quality)}</p>
<h2>ステージ指標</h2><table><thead><tr><th>stage</th><th>converged</th><th>residual</th><th>max reaction</th><th>mass balance</th><th>total energy</th></tr></thead><tbody>{''.join(stages)}</tbody></table>
<h2>確認項目</h2><table><thead><tr><th>id</th><th>severity</th><th>passed</th><th>detail</th></tr></thead><tbody>{''.join(checks)}</tbody></table>
</body></html>
"""


def _add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, severity: str, detail: str) -> None:
    checks.append({"id": check_id, "severity": severity, "passed": bool(passed), "detail": detail})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number_or_blank(value: Any) -> float | str:
    number = _number_or_none(value)
    return "" if number is None else number


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_json_hash(value: Any) -> str:
    import hashlib

    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["build_reliability_summary", "write_reliability_summary_reports"]
