"""Build compact, user-facing judgment summaries from solver output."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from geofem_app.gui.presentation_labels import choice_label, friendly_stage_name
from geofem_app.srm_reporting import (
    BRACKET_TOLERANCE_NOT_MET,
    CONFIRMED_BRACKET,
    LOWER_BOUND_INDETERMINATE,
    MATERIAL_FALLBACK_EVIDENCE,
    NONMONOTONIC_EVIDENCE,
    UNBOUNDED_SEARCH_LIMIT,
    UNRESOLVED_INDETERMINATE_INTERVAL,
    srm_fos_display,
    srm_fos_is_confirmed,
    srm_result_confidence,
    srm_result_status,
)


def build_result_judgment_summary(
    summary: Mapping[str, Any] | None,
    *,
    stage_dir: Path | None = None,
    locale: str = "ja",
) -> dict[str, Any]:
    """Return presentation-ready judgment data for the selected result stage."""

    japanese = str(locale).lower().startswith("ja")
    if not isinstance(summary, Mapping):
        return _empty_summary(japanese)
    stage = _selected_stage(summary, stage_dir)
    if stage is None:
        return _empty_summary(japanese)
    solver = stage.get("solver", {})
    solver = solver if isinstance(solver, Mapping) else {}
    srm = solver.get("srm")
    if isinstance(srm, Mapping):
        return _srm_summary(summary, stage, solver, srm, japanese=japanese)
    return _analysis_summary(summary, stage, solver, japanese=japanese)


def _empty_summary(japanese: bool) -> dict[str, Any]:
    return {
        "kind": "empty",
        "tone": "neutral",
        "headline": "解析結果はまだありません" if japanese else "No analysis results yet",
        "status": "解析実行後に判定サマリを表示します。" if japanese else "A judgment summary will appear after analysis.",
        "detail": "",
        "warning": "",
        "metrics": [],
    }


def _srm_summary(
    summary: Mapping[str, Any],
    stage: Mapping[str, Any],
    solver: Mapping[str, Any],
    srm: Mapping[str, Any],
    *,
    japanese: bool,
) -> dict[str, Any]:
    fos = _finite_number(srm.get("factor_of_safety"))
    stable = _finite_number(srm.get("stable_factor", srm.get("bracket_stable_factor")))
    failed = _finite_number(srm.get("failed_factor", srm.get("bracket_failed_factor")))
    tolerance = _finite_number(srm.get("factor_tol"))
    bracket_width = failed - stable if stable is not None and failed is not None and failed >= stable else None
    status_key = srm_result_status(srm)
    confidence_key = srm_result_confidence(srm)
    confirmed = srm_fos_is_confirmed(srm)

    if confirmed and fos is not None and fos < 1.0:
        tone = "danger"
    elif confirmed:
        tone = "success"
    else:
        tone = "warning"

    status_label = _srm_status_label(status_key, japanese=japanese)
    confidence_label = _confidence_label(confidence_key, japanese=japanese)
    display = srm_fos_display(srm, locale="ja" if japanese else "en")
    if japanese and status_key == CONFIRMED_BRACKET and fos is not None:
        display = f"FOS={_format_number(fos)}（判定区間確定）"
    headline = display or ((f"FOS={_format_number(fos)}") if fos is not None else ("SRM判定" if japanese else "SRM judgment"))
    status_text = (
        f"{status_label} / 信頼度 {confidence_label}"
        if japanese
        else f"{status_label} / confidence {confidence_label}"
    )

    interval = "-"
    if stable is not None and failed is not None:
        interval = f"{_format_number(stable)} - {_format_number(failed)}"
    elif stable is not None:
        interval = f">= {_format_number(stable)}"

    precision = "-"
    if bracket_width is not None:
        width_text = _format_number(bracket_width)
        if tolerance is not None:
            met = bracket_width <= tolerance + max(1.0e-12, abs(tolerance) * 1.0e-9)
            if japanese:
                precision = f"幅 {width_text}\n許容 {_format_number(tolerance)} / {'達成' if met else '未達'}"
            else:
                precision = f"width {width_text} / tolerance {_format_number(tolerance)} ({'met' if met else 'not met'})"
        else:
            precision = (f"幅 {width_text}" if japanese else f"width {width_text}")

    trials = srm.get("trials", [])
    trial_count = len(trials) if isinstance(trials, list) else 0
    elapsed = _srm_elapsed_seconds(srm, solver)
    converged = solver.get("converged")
    convergence_text = _boolean_result(converged, japanese=japanese)
    max_displacement = _format_optional_number(stage.get("max_displacement"))
    raw_search_mode = str(srm.get("search_mode", "") or "-")
    search_mode = choice_label("srm_trial_search_mode", raw_search_mode, locale="ja" if japanese else "en")
    stage_name = friendly_stage_name(stage.get("name", "") or "-", locale="ja" if japanese else "en")

    warning = ""
    if confirmed and fos is not None and fos < 1.0:
        warning = (
            "FOSが1.0未満です。安定側の設計条件と破壊形態を確認してください。"
            if japanese
            else "FOS is below 1.0. Review the stable-side design conditions and failure mode."
        )
    elif not confirmed:
        warning = (
            "このFOSは確定値ではありません。未確定trial、探索上限、材料fallbackを確認してください。"
            if japanese
            else "This FOS is not final. Review indeterminate trials, the search limit, and material fallback evidence."
        )
    elif tolerance is not None and bracket_width is not None and bracket_width > tolerance:
        warning = (
            "判定区間は得られていますが、指定した探索精度に達していません。"
            if japanese
            else "A bracket was found, but it does not meet the requested search tolerance."
        )

    return {
        "kind": "srm",
        "tone": tone,
        "headline": headline,
        "status": status_text,
        "detail": (
            f"ステージ: {stage_name} / 探索: {search_mode}"
            if japanese
            else f"Stage: {stage_name} / search: {search_mode}"
        ),
        "warning": warning,
        "metrics": [
            ("判定区間" if japanese else "Bracket", interval),
            ("探索精度" if japanese else "Precision", precision),
            ("試行数" if japanese else "Trials", str(trial_count)),
            ("解析時間" if japanese else "Elapsed", _format_duration(elapsed, japanese=japanese)),
            ("最大変位" if japanese else "Max displacement", max_displacement),
            ("最終状態" if japanese else "Final state", convergence_text),
        ],
        "raw_status": status_key,
        "confidence": confidence_key,
        "confirmed": confirmed,
        "fos": fos,
        "stable_factor": stable,
        "failed_factor": failed,
        "factor_tol": tolerance,
        "bracket_width": bracket_width,
    }


def _analysis_summary(
    summary: Mapping[str, Any],
    stage: Mapping[str, Any],
    solver: Mapping[str, Any],
    *,
    japanese: bool,
) -> dict[str, Any]:
    converged = solver.get("converged")
    if converged is True:
        tone = "success"
        headline = "解析は収束しました" if japanese else "Analysis converged"
    elif converged is False:
        tone = "danger"
        headline = "解析は収束していません" if japanese else "Analysis did not converge"
    else:
        tone = "neutral"
        headline = "解析結果" if japanese else "Analysis result"
    stage_name = friendly_stage_name(stage.get("name", "") or "-", locale="ja" if japanese else "en")
    residual = _format_optional_number(solver.get("residual_norm"))
    iterations = solver.get("iterations")
    if isinstance(iterations, list):
        iterations_text = str(sum(int(value) for value in iterations if isinstance(value, (int, float))))
    else:
        iterations_text = _format_optional_number(iterations, integer=True)
    performance = solver.get("performance", {})
    elapsed = _finite_number(performance.get("elapsed_seconds")) if isinstance(performance, Mapping) else None
    warnings = summary.get("warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0
    warning = ""
    if converged is False:
        warning = (
            "非収束結果です。残差、増分、境界条件を確認してください。"
            if japanese
            else "The result is unconverged. Review residuals, increments, and boundary conditions."
        )
    elif warning_count:
        warning = (
            f"解析警告が{warning_count}件あります。ログとモデルチェックを確認してください。"
            if japanese
            else f"There are {warning_count} analysis warnings. Review the log and model check."
        )
    return {
        "kind": "analysis",
        "tone": tone,
        "headline": headline,
        "status": (
            f"ステージ {stage_name} / {_boolean_result(converged, japanese=japanese)}"
            if japanese
            else f"Stage {stage_name} / {_boolean_result(converged, japanese=japanese)}"
        ),
        "detail": str(summary.get("analysis", "") or ""),
        "warning": warning,
        "metrics": [
            ("最大変位" if japanese else "Max displacement", _format_optional_number(stage.get("max_displacement"))),
            ("最大沈下" if japanese else "Max settlement", _format_optional_number(stage.get("max_settlement"))),
            ("残差" if japanese else "Residual", residual),
            ("反復回数" if japanese else "Iterations", iterations_text),
            ("解析時間" if japanese else "Elapsed", _format_duration(elapsed, japanese=japanese)),
            ("警告" if japanese else "Warnings", str(warning_count)),
        ],
        "converged": converged,
    }


def _selected_stage(summary: Mapping[str, Any], stage_dir: Path | None) -> Mapping[str, Any] | None:
    stages = summary.get("stages", [])
    if not isinstance(stages, list):
        return None
    rows = [stage for stage in stages if isinstance(stage, Mapping)]
    if not rows:
        return None
    if stage_dir is not None:
        target = Path(stage_dir)
        for stage in rows:
            raw = stage.get("output_dir")
            if raw and Path(str(raw)).name == target.name:
                return stage
    return rows[-1]


def _srm_status_label(status: str, *, japanese: bool) -> str:
    labels_ja = {
        CONFIRMED_BRACKET: "判定区間確定",
        BRACKET_TOLERANCE_NOT_MET: "探索精度未達",
        LOWER_BOUND_INDETERMINATE: "下限のみ・未確定trialあり",
        UNRESOLVED_INDETERMINATE_INTERVAL: "未確定区間あり",
        UNBOUNDED_SEARCH_LIMIT: "探索上限未確定",
        NONMONOTONIC_EVIDENCE: "非単調な判定証拠あり",
        MATERIAL_FALLBACK_EVIDENCE: "材料fallback要検証",
    }
    labels_en = {
        CONFIRMED_BRACKET: "Confirmed bracket",
        BRACKET_TOLERANCE_NOT_MET: "Tolerance not met",
        LOWER_BOUND_INDETERMINATE: "Lower bound with indeterminate trial",
        UNRESOLVED_INDETERMINATE_INTERVAL: "Unresolved interval",
        UNBOUNDED_SEARCH_LIMIT: "Upper bound unknown",
        NONMONOTONIC_EVIDENCE: "Nonmonotonic evidence",
        MATERIAL_FALLBACK_EVIDENCE: "Material fallback requires review",
    }
    return (labels_ja if japanese else labels_en).get(status, status or ("未確定" if japanese else "Unconfirmed"))


def _confidence_label(confidence: str, *, japanese: bool) -> str:
    if japanese:
        return {"high": "高", "medium": "中", "limited": "限定", "low": "低"}.get(confidence, confidence or "-")
    return confidence or "-"


def _srm_elapsed_seconds(srm: Mapping[str, Any], solver: Mapping[str, Any]) -> float | None:
    timing = srm.get("trial_timing", {})
    if isinstance(timing, Mapping):
        elapsed = _finite_number(timing.get("total_elapsed_seconds"))
        if elapsed is not None:
            return elapsed
    performance = solver.get("performance", {})
    if isinstance(performance, Mapping):
        return _finite_number(performance.get("elapsed_seconds"))
    return None


def _boolean_result(value: Any, *, japanese: bool) -> str:
    if value is True:
        return "収束" if japanese else "Converged"
    if value is False:
        return "未収束" if japanese else "Not converged"
    return "記録なし" if japanese else "Not recorded"


def _format_duration(value: float | None, *, japanese: bool) -> str:
    if value is None:
        return "-"
    seconds_total = max(0, int(round(value)))
    hours, remainder = divmod(seconds_total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if japanese:
        if hours:
            return f"{hours}時間{minutes:02d}分{seconds:02d}秒"
        return f"{minutes}分{seconds:02d}秒"
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _format_optional_number(value: Any, *, integer: bool = False) -> str:
    number = _finite_number(value)
    if number is None:
        return "-"
    return str(int(round(number))) if integer else _format_number(number)


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.8g}"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = ["build_result_judgment_summary"]
