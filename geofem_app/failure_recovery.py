"""Recovery-plan generation for failed GeoFEM analyses."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_failure_recovery_plan(
    failure_analysis: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ranked recovery actions from failure classification and available artifacts."""

    root = Path(output_dir) if output_dir is not None else None
    actions: list[dict[str, Any]] = []
    category = str(failure_analysis.get("primary_category", "unknown"))
    stage = str(failure_analysis.get("primary_stage", ""))
    panel = str(failure_analysis.get("primary_gui_panel", "model_check"))
    input_path = str(failure_analysis.get("primary_input_path", ""))

    actions.extend(_base_actions(category, stage=stage, panel=panel, input_path=input_path))

    diag_actions = _diagnostic_actions(diagnostics or {}, category)
    actions.extend(diag_actions)

    if root is not None:
        analysis_log = _read_json(root / "analysis_log.json")
        mesh_quality = _read_json(root / "mesh_quality.json")
        if analysis_log:
            actions.extend(_analysis_log_actions(analysis_log))
        if mesh_quality:
            actions.extend(_mesh_quality_actions(mesh_quality))
        actions.extend(_plastic_concentration_actions(root))

    actions = _rank_actions(actions)
    return {
        "schema": "geofem.failure_recovery_plan.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(root or ""),
        "primary_category": category,
        "primary_stage": stage,
        "primary_gui_panel": panel,
        "action_count": len(actions),
        "features": [
            "ranked_recovery_actions",
            "convergence_history_scan",
            "dominant_residual_dof_hint",
            "constraint_deficiency_candidates",
            "plastic_concentration_scan",
            "mesh_quality_repair_candidates",
        ],
        "actions": actions,
    }


def write_failure_recovery_plan(plan: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write JSON/CSV/HTML recovery-plan artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "failure_recovery_plan.json"
    csv_path = out / "failure_recovery_plan.csv"
    html_path = out / "failure_recovery_plan.html"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(csv_path, plan.get("actions", []))
    html_path.write_text(_html(plan), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _base_actions(category: str, *, stage: str, panel: str, input_path: str) -> list[dict[str, Any]]:
    common = {
        "stage": stage,
        "target_path": input_path,
        "source": "failure_category",
    }
    if category == "convergence_failure":
        return [
            _action("reduce_increment", "HIGH", "solver", "荷重増分/時間刻みを小さくする", "increments.stepsを増やす、dynamic.dtを下げる、cutbackを有効にする。", "残差が急増する前に小刻みに平衡経路を追跡できる。", target_panel="solver", **common),
            _action("enable_line_search", "HIGH", "solver", "line searchとcutbackを有効にする", "solver.newton.line_search=true、cutback_factorを0.5程度から試す。", "非線形反復の発散と過大ステップを抑える。", target_panel="solver", **common),
            _action("review_material_yielding", "MEDIUM", "materials", "材料降伏集中を確認する", "塑性化要素比、yield_value、FL履歴を確認し、強度/硬化/ダイレイタンシーを見直す。", "局所降伏が原因の収束悪化を切り分ける。", target_panel="materials", **common),
        ]
    if category == "numerical_failure":
        return [
            _action("check_constraints", "HIGH", "boundary_conditions", "拘束不足と拘束競合を確認する", "水平/鉛直剛体運動、MPC slave/master、固定変位との競合をモデルチェックで確認する。", "特異行列や非有限変位の主因を除去する。", target_panel="boundary_conditions", **common),
            _action("inspect_solver_matrix", "HIGH", "solver", "線形ソルバー設定を確認する", "direct/cg、前処理、拘束自由度数、ゼロ剛性要素を確認する。", "行列破綻が設定由来かモデル由来か切り分ける。", target_panel="solver", **common),
        ]
    if category in {"mesh_quality", "mesh_definition"}:
        return [_action("repair_mesh", "HIGH", "mesh", "メッシュ不良候補を修復する", "mesh_quality_repairs.csvの候補を適用し、負のJacobian/潰れ要素/過大アスペクト比を解消する。", "要素積分と剛性行列の安定性を回復する。", target_panel="mesh", **common)]
    if category == "material_parameter":
        return [_action("repair_material", "HIGH", "materials", "材料パラメータを修正する", "E、nu、強度、密度、液状化/非線形パラメータの不足と範囲外を直す。", "非物理的な剛性や強度条件を除去する。", target_panel="materials", **common)]
    if category == "boundary_condition":
        return [_action("repair_boundary_conditions", "HIGH", "boundary_conditions", "境界条件を修正する", "拘束不足、過拘束、未知set、MPC競合を入力画面で修正する。", "剛体運動と拘束競合による解法破綻を避ける。", target_panel="boundary_conditions", **common)]
    return [_action("rerun_model_check", "MEDIUM", "model_check", "モデルチェックを再実行する", "入力診断、モデルチェック、メッシュ品質を確認してから再実行する。", "未分類エラーの原因候補を狭める。", target_panel=panel or "model_check", **common)]


def _diagnostic_actions(diagnostics: Mapping[str, Any], category: str) -> list[dict[str, Any]]:
    issues = diagnostics.get("issues", [])
    if not isinstance(issues, Sequence) or isinstance(issues, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        severity = str(issue.get("severity", "")).upper()
        if severity not in {"ERROR", "WARN"}:
            continue
        path = str(issue.get("path", ""))
        message = str(issue.get("message", ""))
        suggestion = str(issue.get("suggestion", ""))
        if path.startswith("boundary_conditions") or "拘束" in message or "rigid-body" in message.lower():
            rows.append(_action("constraint_candidate", "HIGH", "boundary_conditions", "拘束不足候補を確認する", suggestion or message, "拘束不足や拘束競合を早期に解消する。", source="input_diagnostics", stage="", target_panel="boundary_conditions", target_path=path))
        elif path.startswith("mesh"):
            rows.append(_action("mesh_input_candidate", "HIGH", "mesh", "メッシュ入力候補を修正する", suggestion or message, "メッシュ起因の解析停止を解消する。", source="input_diagnostics", stage="", target_panel="mesh", target_path=path))
        elif path.startswith(("materials", "material")):
            rows.append(_action("material_input_candidate", "HIGH", "materials", "材料入力候補を修正する", suggestion or message, "材料定義起因の解析停止を解消する。", source="input_diagnostics", stage="", target_panel="materials", target_path=path))
    if not rows and category == "numerical_failure":
        rows.append(_action("constraint_candidate", "HIGH", "boundary_conditions", "拘束不足候補を追加確認する", "入力診断に明示候補がないため、支持条件とMPCをモデルチェックで再確認する。", "特異行列の候補を手早く切り分ける。", source="inferred", stage="", target_panel="boundary_conditions", target_path="boundary_conditions"))
    return rows


def _analysis_log_actions(log: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = log.get("events", [])
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return []
    residual_rows = [row for row in events if isinstance(row, Mapping) and any(key in row for key in ("residual_norm", "pressure_residual_norm", "constraint_norm"))]
    if not residual_rows:
        return []
    latest = residual_rows[-1]
    dominant_field = _dominant_residual_field(latest)
    stage = str(latest.get("stage", ""))
    actions = [
        _action(
            "inspect_convergence_history",
            "HIGH",
            "solver",
            "収束履歴を確認する",
            f"最終残差フィールド={dominant_field}、stage={stage}、iteration={latest.get('iteration', latest.get('step', ''))}",
            "支配残差が変位、間隙水圧、拘束のどれかを判定する。",
            source="analysis_log",
            stage=stage,
            target_panel="solver",
            target_path="analysis_log.json",
        )
    ]
    dominant_dof = _dominant_dof(latest)
    if dominant_dof:
        actions.append(
            _action(
                "dominant_residual_dof",
                "HIGH",
                "model_check",
                "残差支配自由度を確認する",
                f"支配自由度候補: {dominant_dof}",
                "該当節点/自由度の拘束、荷重、接続要素、材料状態を確認する。",
                source="analysis_log",
                stage=stage,
                target_panel="model_check",
                target_path=str(dominant_dof),
            )
        )
    else:
        actions.append(
            _action(
                "enable_dominant_residual_output",
                "MEDIUM",
                "solver",
                "残差支配自由度出力を有効化する",
                "解析ログにdominant_dof/dominant_nodeがないため、詳細残差ログを有効にして再実行する。",
                "次回失敗時に節点/自由度単位で原因を絞れる。",
                source="analysis_log",
                stage=stage,
                target_panel="solver",
                target_path="solver.diagnostics",
            )
        )
    return actions


def _mesh_quality_actions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), Mapping) else {}
    violation_count = int(summary.get("violation_count", 0) or 0)
    repair_count = len(report.get("repair_candidates", [])) if isinstance(report.get("repair_candidates", []), Sequence) else 0
    if violation_count <= 0:
        return []
    return [
        _action(
            "mesh_quality_repairs",
            "HIGH",
            "mesh",
            "メッシュ不良候補を比較/適用する",
            f"violation_count={violation_count}, repair_candidate_count={repair_count}",
            "潰れ要素や悪条件要素による収束悪化を減らす。",
            source="mesh_quality",
            stage="",
            target_panel="mesh",
            target_path="mesh_quality_repairs.csv",
        )
    ]


def _plastic_concentration_actions(root: Path) -> list[dict[str, Any]]:
    scanned = 0
    plastic = 0
    max_yield = 0.0
    stage_hint = ""
    for path in root.rglob("*.csv"):
        if "element" not in path.name.lower():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scanned += 1
                    p = _float(row.get("plastic", row.get("plastic_flag", 0.0)))
                    yv = _float(row.get("yield_value", row.get("yield_ratio", 0.0)))
                    max_yield = max(max_yield, yv)
                    if p > 0.0 or yv >= 0.99:
                        plastic += 1
                        stage_hint = path.parent.name
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
    if scanned <= 0:
        return []
    ratio = plastic / max(scanned, 1)
    if ratio < 0.25 and max_yield < 0.99:
        return []
    return [
        _action(
            "plastic_concentration",
            "MEDIUM",
            "materials",
            "材料降伏集中を確認する",
            f"plastic_ratio={ratio:.6g}, max_yield_value={max_yield:.6g}, scanned_rows={scanned}",
            "局所降伏が収束失敗の主因かを判断し、強度/硬化/施工増分を調整する。",
            source="element_results",
            stage=stage_hint,
            target_panel="materials",
            target_path="element_stresses.csv",
        )
    ]


def _rank_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    source_rank = {"failure_category": 0, "analysis_log": 1, "mesh_quality": 2, "element_results": 3, "input_diagnostics": 4, "inferred": 5}
    for raw in actions:
        key = (str(raw.get("id", "")), str(raw.get("target_panel", "")), str(raw.get("target_path", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(raw))
    unique.sort(key=lambda row: (priority_rank.get(str(row.get("priority", "LOW")), 9), source_rank.get(str(row.get("source", "")), 9), str(row.get("id", ""))))
    for index, row in enumerate(unique, start=1):
        row["rank"] = index
    return unique


def _action(
    action_id: str,
    priority: str,
    category: str,
    title: str,
    action: str,
    expected_effect: str,
    *,
    source: str,
    stage: str,
    target_panel: str,
    target_path: str,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "priority": priority,
        "category": category,
        "title": title,
        "action": action,
        "expected_effect": expected_effect,
        "source": source,
        "stage": stage,
        "target_panel": target_panel,
        "target_path": target_path,
    }


def _dominant_residual_field(row: Mapping[str, Any]) -> str:
    fields = {
        "residual_norm": abs(_float(row.get("residual_norm", 0.0))),
        "pressure_residual_norm": abs(_float(row.get("pressure_residual_norm", 0.0))),
        "constraint_norm": abs(_float(row.get("constraint_norm", 0.0))),
    }
    return max(fields, key=fields.get)


def _dominant_dof(row: Mapping[str, Any]) -> str:
    for key in ("dominant_dof", "dominant_node", "dominant_component", "residual_dof", "max_residual_dof"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _write_csv(path: Path, rows: Any) -> None:
    fields = ["rank", "id", "priority", "category", "title", "action", "expected_effect", "source", "stage", "target_panel", "target_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []:
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fields})


def _html(plan: Mapping[str, Any]) -> str:
    rows = []
    for action in plan.get("actions", []):
        if not isinstance(action, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(action.get('rank', '')))}</td>"
            f"<td>{html.escape(str(action.get('priority', '')))}</td>"
            f"<td>{html.escape(str(action.get('title', '')))}</td>"
            f"<td>{html.escape(str(action.get('target_panel', '')))}</td>"
            f"<td>{html.escape(str(action.get('target_path', '')))}</td>"
            f"<td>{html.escape(str(action.get('action', '')))}</td>"
            f"<td>{html.escape(str(action.get('expected_effect', '')))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\"><title>GeoFEM failure recovery plan</title>"
        "<style>body{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top;}th{background:#f2f2f2;}</style></head><body>"
        "<h1>解析エラー復旧計画</h1>"
        f"<p>category={html.escape(str(plan.get('primary_category', '')))}, actions={html.escape(str(plan.get('action_count', '')))}</p>"
        "<table><thead><tr><th>rank</th><th>priority</th><th>title</th><th>panel</th><th>target</th><th>action</th><th>expected effect</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )


__all__ = ["build_failure_recovery_plan", "write_failure_recovery_plan"]
