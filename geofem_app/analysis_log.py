"""Structured analysis-log artifacts for solver diagnostics."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .fem2d_types import SolveResult2D, StageResult2D
from .srm_reporting import srm_fos_display, srm_result_confidence, srm_result_status


SRM_TRIAL_DIAGNOSTIC_FIELDS = [
    "srm_trial_state",
    "auto_decision",
    "auto_failure_class",
    "auto_failure_score",
    "auto_decision_reason",
    "auto_trial_action",
    "auto_retry",
    "auto_retry_of",
    "auto_retry_index",
    "auto_retry_planned",
    "auto_retry_reason",
    "auto_retry_result",
    "auto_superseded_by_retry",
    "auto_cluster_fraction",
    "auto_last_accepted_strength_factor_estimate",
    "estimated_fos_from_last_load",
    "boundary_verification",
    "boundary_verification_reason",
    "boundary_verification_of",
    "boundary_verification_result",
    "boundary_verification_superseded",
    "boundary_verification_pending",
    "boundary_verification_deferred",
    "boundary_verification_deferred_reason",
    "boundary_verification_trigger",
    "boundary_verification_cold_start",
    "boundary_verification_early_failure_disabled",
    "boundary_checkpoint_continuation_requested",
    "boundary_checkpoint_continuation_used",
    "boundary_checkpoint_fallback_reason",
    "factor_tol_numerical_failure_boundary",
    "factor_tol_numerical_failure_rejected",
    "factor_tol_physical_failure_evidence",
    "factor_tol_enforcement_original_state",
    "factor_tol_enforcement_reason",
    "checkpoint_residual_prediction_enabled",
    "checkpoint_residual_prediction_reason",
    "checkpoint_residual_prediction_sample_count",
    "checkpoint_residual_prediction_ratio",
    "checkpoint_residual_prediction_extra_cutbacks",
    "warm_start_used",
    "warm_start_source",
    "warm_start_source_factor",
    "warm_start_target_factor",
    "warm_start_factor_distance",
    "warm_start_displacement_only",
    "warm_start_displacement_size",
    "warm_start_max_displacement_norm",
    "early_failure_stop",
    "early_failure_policy",
    "early_failure_score",
    "early_failure_score_threshold",
    "early_failure_reason",
    "early_failure_cutback_ratio",
    "early_failure_effective_cutbacks",
    "adaptive_increment_control",
    "adaptive_increment_source",
    "adaptive_increment_source_factor",
    "adaptive_increment_target_factor",
    "adaptive_increment_reason",
    "adaptive_increment_last_accepted_load_factor",
    "adaptive_increment_final_step_size",
    "adaptive_increment_cutback_count",
    "adaptive_increment_max_cutbacks",
    "adaptive_increment_cutback_ratio",
    "adaptive_increment_target_initial_step_factor",
    "adaptive_increment_max_steps_multiplier",
    "adaptive_increment_extra_cutbacks",
    "adaptive_increment_min_step_factor",
    "elapsed_seconds",
    "solver_elapsed_seconds",
    "overhead_elapsed_seconds",
    "solver_cancel_requested",
    "solver_cancel_checkpoint",
    "solver_cancel_scope",
    "increment_checkpoint_available",
    "increment_checkpoint_schema",
    "increment_checkpoint_fingerprint",
    "increment_checkpoint_load_factor",
    "increment_checkpoint_accepted_steps",
    "increment_checkpoint_cutbacks",
    "increment_checkpoint_continuation_requested",
    "increment_checkpoint_continuation_used",
    "increment_checkpoint_fallback_reason",
    "increment_checkpoint_source_load_factor",
    "increment_checkpoint_resumed_accepted_steps",
    "increment_checkpoint_resumed_cutbacks",
    "increment_checkpoint_reused_history_rows",
    "trial_status",
    "attempted_load_factor",
    "last_accepted_load_factor",
    "accepted_increment_count",
    "cutback_count",
    "max_cutbacks",
    "failed_step_size",
    "next_step_size",
    "min_step",
    "cutback_factor",
    "last_accepted_plastic_ratio",
    "last_accepted_plastic_point_count",
    "last_accepted_max_displacement",
    "last_accepted_residual_norm",
    "last_accepted_iterations",
    "last_accepted_line_search_reductions",
    "residual_reduction_ratio",
    "failure_diagnostic_source",
    "diagnostic_summary",
    "plastic_ratio_delta",
    "plastic_ratio_delta_reference_factor",
    "max_equivalent_plastic_strain",
    "mean_equivalent_plastic_strain",
    "top_percentile_equivalent_plastic_strain",
    "top_percentile_equivalent_plastic_strain_percentile",
    "yielded_element_count",
    "connected_plastic_cluster_size",
    "plastic_cluster_spans_boundary",
    "plastic_cluster_boundary_side_count",
    "final_step_size",
    "newton_iterations_total",
    "newton_iterations_max",
    "line_search_reductions_total",
    "line_search_batch_calls_total",
    "line_search_batch_candidates_total",
    "line_search_batch_fallback_count",
    "residual_norm_final",
    "min_det_j",
    "max_displacement_norm",
    "displacement_increment_norm",
    "internal_external_work_ratio",
    "plastic_point_count",
    "active_element_count",
    "plastic_diagnostics_mode",
    "mc_numba_to_python_fallback_count",
    "mc_numba_regularized_projection_count",
    "mc_regularized_projection_count",
    "mc_apex_regularization_count",
    "mc_associated_apex_projection_count",
    "mc_legacy_bounded_projection_count",
    "mc_regularization_method",
    "mc_configured_apex_policy_verified",
    "mc_base_nonassociated_flow_rule_verified",
    "mc_constitutive_model_fidelity",
    "mc_regularized_projection_above_relaxed_tolerance_count",
    "mc_active_set_update_attempt_count",
    "mc_active_set_update_hit_count",
    "mc_active_set_regularized_update_hit_count",
    "mc_active_set_full_scan_avoided_count",
    "mc_active_set_policy",
    "mc_active_set_tangent_reuse_enabled",
    "mc_active_set_tangent_reuse_disabled_reason",
    "mc_active_set_direct_consistent_tangent_enabled",
    "mc_active_set_numerical_tangent_switch_count",
    "mc_active_set_numerical_tangent_switch_reason",
    "mc_active_set_tangent_invalidation_count",
    "mc_active_set_tangent_invalidated_point_count",
    "mc_active_set_consistent_tangent",
    "mc_active_set_cutback_reset_policy",
    "mc_regularized_projection_max_yield_violation",
    "mc_regularized_projection_max_relative_yield_violation",
    "mc_regularized_projection_samples",
]

SRM_BOUNDARY_VERIFICATION_FIELDS = [
    "srm_boundary_verification_strategy",
    "srm_boundary_verification_defer_min_failure_score",
    "srm_boundary_verification_deferred_count",
    "srm_boundary_verification_executed_count",
    "srm_boundary_verification_recovery_count",
    "srm_boundary_verification_stable_reversal_count",
    "srm_boundary_verification_cold_retry_on_indeterminate",
    "srm_boundary_verification_cold_retry_count",
    "srm_boundary_verification_cold_retry_factors",
    "srm_boundary_checkpoint_continuation_enabled",
    "srm_boundary_checkpoint_continuation_extra_cutbacks",
    "srm_boundary_checkpoint_residual_prediction_enabled",
    "srm_boundary_checkpoint_residual_prediction_max_extra_cutbacks",
    "srm_boundary_checkpoint_residual_prediction_used_count",
    "srm_boundary_verification_strict_tangent",
    "srm_retry_strict_tangent",
    "srm_boundary_checkpoint_continuation_requested_count",
    "srm_boundary_checkpoint_continuation_used_count",
    "srm_boundary_checkpoint_fallback_count",
    "srm_factor_tol_enforcement_enabled",
    "srm_factor_tol_enforcement_extra_bisections_used",
    "srm_factor_tol_numerical_failure_boundary_count",
    "srm_factor_tol_numerical_failure_boundary_factors",
    "srm_factor_tol_require_physical_failure_evidence",
]

SRM_PARALLEL_FIELDS = [
    "srm_parallel_enabled",
    "srm_parallel_strategy",
    "srm_parallel_context",
    "srm_parallel_policy",
    "srm_parallel_executor",
    "srm_parallel_effective_executor",
    "srm_parallel_process_executor_fallback_count",
    "srm_parallel_process_executor_errors",
    "srm_parallel_max_workers",
    "srm_parallel_requested_workers",
    "srm_parallel_selected_threads_per_worker",
    "srm_parallel_lookahead_depth",
    "srm_parallel_speculative_trial_count",
    "srm_parallel_used_speculative_trial_count",
    "srm_parallel_unused_speculative_trial_count",
    "srm_parallel_window_evaluated_trials",
    "srm_parallel_canceled_speculative_trial_count",
    "srm_parallel_speculative_cancellation_requested",
    "srm_parallel_speculative_cancellation_note",
    "srm_parallel_decision_linked_cancellation_enabled",
    "srm_parallel_decision_linked_requested_count",
    "srm_parallel_decision_linked_pending_cancel_count",
    "srm_parallel_decision_linked_safe_stop_count",
    "srm_parallel_decision_linked_completed_after_request_count",
    "srm_parallel_decision_linked_requested_factors",
    "srm_parallel_speculative_prefetch_call_count",
    "srm_parallel_speculative_prefetch_wall_elapsed_seconds",
    "srm_parallel_speculative_trial_elapsed_seconds",
    "srm_parallel_speculative_queue_wait_elapsed_seconds",
    "srm_parallel_speculative_worker_elapsed_seconds",
    "srm_parallel_speculative_estimated_wall_clock_saving_seconds",
    "srm_parallel_bisection_speculation_enabled",
    "srm_parallel_bisection_speculative_trial_count",
    "srm_parallel_bisection_used_speculative_trial_count",
    "srm_parallel_bisection_unused_speculative_trial_count",
    "srm_parallel_cost_aware_lookahead_enabled",
    "srm_parallel_cost_aware_depth_limited_count",
    "srm_parallel_cost_aware_asymmetric_bisection_count",
    "srm_parallel_cost_aware_deferred_candidate_count",
    "srm_parallel_event_driven_cost_cancellation_enabled",
    "srm_parallel_event_driven_cost_shrink_count",
    "srm_parallel_event_driven_cost_cancel_candidate_count",
    "srm_parallel_cost_aware_sample_count",
    "srm_parallel_cost_aware_stable_median_elapsed_seconds",
    "srm_parallel_cost_aware_failed_median_elapsed_seconds",
    "srm_parallel_cost_aware_failure_to_stable_cost_ratio",
    "srm_parallel_cost_aware_reason",
    "srm_parallel_logical_cpus",
    "srm_parallel_physical_cpus",
    "srm_parallel_available_memory_mb",
    "srm_parallel_memory_limit_mb",
    "srm_parallel_memory_per_worker_mb",
    "srm_parallel_memory_limited",
    "srm_parallel_node_count",
    "srm_parallel_element_count",
    "srm_parallel_active_element_count",
    "srm_parallel_dof_count",
    "srm_parallel_thread_control_applied",
    "srm_parallel_thread_control_method",
    "srm_parallel_thread_control_error",
    "srm_parallel_thread_environment_restored",
    "srm_parallel_threadpoolctl_available",
]


def build_structured_analysis_log(result: SolveResult2D) -> dict[str, Any]:
    stages = [_stage_record(index, stage) for index, stage in enumerate(result.stages, start=1)]
    events: list[dict[str, Any]] = []
    for warning in result.warnings:
        events.append({"event_type": "warning", "stage": "", "message": str(warning)})
    for stage_index, stage in enumerate(result.stages, start=1):
        solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
        srm = _stage_srm_info(stage)
        srm_trials = srm.get("trials", []) if srm else []
        srm_trial_count = len(srm_trials) if isinstance(srm_trials, list) else 0
        stage_message = f"stage output: {stage.output_dir}"
        if srm:
            stage_message += f", {srm_fos_display(srm)}, SRM trials={srm_trial_count}"
        events.append(
            {
                "event_type": "stage_completed",
                "stage": stage.name,
                "index": stage_index,
                "method": solver.get("method", ""),
                "iterations": solver.get("iterations", ""),
                "residual_norm": solver.get("residual_norm", ""),
                "converged": solver.get("converged", True),
                "factor_of_safety": srm.get("factor_of_safety", "") if srm else "",
                "factor_of_safety_status": srm_result_status(srm) if srm else "",
                "factor_of_safety_confidence": srm_result_confidence(srm) if srm else "",
                "srm_trial_count": srm_trial_count if srm else "",
                **_srm_parallel_event_fields(srm),
                "message": stage_message,
            }
        )
        events.extend(_srm_events(stage))
        events.extend(_convergence_events(stage))
        events.extend(_cutback_events(stage))
    return {
        "schema": "geofem.analysis_log.v1",
        "stage_count": len(stages),
        "warning_count": len(result.warnings),
        "stages": stages,
        "events": events,
    }


def write_structured_analysis_log(result: SolveResult2D, output_dir: str | Path | None = None) -> dict[str, str]:
    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = build_structured_analysis_log(result)
    json_path = out / "analysis_log.json"
    csv_path = out / "analysis_log.csv"
    html_path = out / "analysis_log.html"
    json_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fields = [
        "event_type",
        "stage",
        "index",
        "step",
        "iteration",
        "time",
        "dt",
        "method",
        "linear_method",
        "iterations",
        "residual_norm",
        "pressure_residual_norm",
        "constraint_norm",
        "cutbacks",
        "converged",
        "factor_of_safety",
        "factor_of_safety_status",
        "factor_of_safety_confidence",
        "stable_factor",
        "failed_factor",
        "factor_of_safety_interval",
        "factor_of_safety_boundary_certified",
        "factor_of_safety_tolerance_met",
        "factor_of_safety_certified",
        "factor_of_safety_value_kind",
        "boundary_quality",
        "boundary_verified",
        "material_fallback_verification_required",
        "indeterminate_trial_count",
        "indeterminate_factors",
        "srm_factor",
        "plastic_ratio",
        "srm_ok",
        "srm_trial_count",
        *SRM_BOUNDARY_VERIFICATION_FIELDS,
        *SRM_PARALLEL_FIELDS,
        "failure_reason",
        "error",
        *SRM_TRIAL_DIAGNOSTIC_FIELDS,
        "message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in log["events"]:
            writer.writerow({field: row.get(field, "") for field in fields})
    html_path.write_text(_analysis_log_html(log), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _stage_record(index: int, stage: StageResult2D) -> dict[str, Any]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    performance = solver.get("performance", {}) if isinstance(solver.get("performance", {}), Mapping) else {}
    matrix = solver.get("matrix", {}) if isinstance(solver.get("matrix", {}), Mapping) else {}
    srm = _stage_srm_info(stage)
    srm_trials = srm.get("trials", []) if srm else []
    return {
        "index": index,
        "name": stage.name,
        "time": stage.time,
        "output_dir": str(stage.output_dir) if stage.output_dir else "",
        "active_element_count": len(stage.active_elements),
        "method": solver.get("method", ""),
        "linear_method": solver.get("linear_method", solver.get("method", "")),
        "iterations": solver.get("iterations", 0),
        "residual_norm": solver.get("residual_norm", ""),
        "converged": solver.get("converged", True),
        "elapsed_seconds": performance.get("elapsed_seconds", ""),
        "matrix_size": matrix.get("size", ""),
        "matrix_nnz": matrix.get("nnz", ""),
        "factor_of_safety": srm.get("factor_of_safety", "") if srm else "",
        "srm_trial_count": len(srm_trials) if isinstance(srm_trials, list) else 0,
    }


def _stage_srm_info(stage: StageResult2D) -> Mapping[str, Any]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    srm = solver.get("srm")
    return srm if isinstance(srm, Mapping) else {}


def _srm_events(stage: StageResult2D) -> list[dict[str, Any]]:
    srm = _stage_srm_info(stage)
    if not srm:
        return []
    trials = srm.get("trials", [])
    trial_rows = trials if isinstance(trials, list) else []
    fos = srm.get("factor_of_safety", "")
    auto = srm.get("auto", {}) if isinstance(srm.get("auto", {}), Mapping) else {}
    rows: list[dict[str, Any]] = [
        {
            "event_type": "srm_summary",
            "stage": stage.name,
            "method": "srm",
            "factor_of_safety": fos,
            "factor_of_safety_status": srm_result_status(srm),
            "factor_of_safety_confidence": srm_result_confidence(srm),
            "stable_factor": srm.get("stable_factor", ""),
            "failed_factor": srm.get("failed_factor", ""),
            "factor_of_safety_interval": srm.get("factor_of_safety_interval", ""),
            "factor_of_safety_boundary_certified": srm.get(
                "factor_of_safety_boundary_certified", ""
            ),
            "factor_of_safety_tolerance_met": srm.get("factor_of_safety_tolerance_met", ""),
            "factor_of_safety_certified": srm.get("factor_of_safety_certified", ""),
            "factor_of_safety_value_kind": srm.get("factor_of_safety_value_kind", ""),
            "boundary_quality": srm.get("boundary_quality", ""),
            "boundary_verified": srm.get("boundary_verified", ""),
            "bracket_resolved": srm.get("bracket_resolved", ""),
            "indeterminate_factors_inside_bracket": srm.get(
                "indeterminate_factors_inside_bracket", []
            ),
            "material_fallback_verification_required": srm.get(
                "material_fallback_verification_required", ""
            ),
            "indeterminate_trial_count": srm.get("indeterminate_trial_count", 0),
            "indeterminate_factors": srm.get("indeterminate_factors", []),
            "srm_trial_count": len(trial_rows),
            "srm_boundary_verification_strategy": auto.get(
                "boundary_verification_strategy", ""
            ),
            "srm_boundary_verification_defer_min_failure_score": auto.get(
                "boundary_verification_defer_min_failure_score", ""
            ),
            "srm_boundary_verification_deferred_count": auto.get(
                "boundary_verification_deferred_count", 0
            ),
            "srm_boundary_verification_executed_count": auto.get(
                "boundary_verification_executed_count", 0
            ),
            "srm_boundary_verification_recovery_count": auto.get(
                "boundary_verification_recovery_count", 0
            ),
            "srm_boundary_verification_stable_reversal_count": auto.get(
                "boundary_verification_stable_reversal_count", 0
            ),
            "srm_boundary_verification_cold_retry_on_indeterminate": auto.get(
                "boundary_verification_cold_retry_on_indeterminate", ""
            ),
            "srm_boundary_verification_cold_retry_count": auto.get(
                "boundary_verification_cold_retry_count", 0
            ),
            "srm_boundary_verification_cold_retry_factors": auto.get(
                "boundary_verification_cold_retry_factors", []
            ),
            "srm_boundary_checkpoint_continuation_enabled": auto.get(
                "boundary_checkpoint_continuation_enabled", ""
            ),
            "srm_boundary_checkpoint_continuation_extra_cutbacks": auto.get(
                "boundary_checkpoint_continuation_extra_cutbacks", ""
            ),
            "srm_boundary_checkpoint_residual_prediction_enabled": auto.get(
                "boundary_checkpoint_residual_prediction_enabled", ""
            ),
            "srm_boundary_checkpoint_residual_prediction_max_extra_cutbacks": auto.get(
                "boundary_checkpoint_residual_prediction_max_extra_cutbacks", ""
            ),
            "srm_boundary_checkpoint_residual_prediction_used_count": auto.get(
                "boundary_checkpoint_residual_prediction_used_count", 0
            ),
            "srm_boundary_verification_strict_tangent": auto.get(
                "boundary_verification_strict_tangent", ""
            ),
            "srm_retry_strict_tangent": auto.get(
                "retry_strict_tangent", ""
            ),
            "srm_boundary_checkpoint_continuation_requested_count": auto.get(
                "boundary_checkpoint_continuation_requested_count", 0
            ),
            "srm_boundary_checkpoint_continuation_used_count": auto.get(
                "boundary_checkpoint_continuation_used_count", 0
            ),
            "srm_boundary_checkpoint_fallback_count": auto.get(
                "boundary_checkpoint_fallback_count", 0
            ),
            "srm_factor_tol_enforcement_enabled": auto.get(
                "factor_tol_enforcement_enabled", ""
            ),
            "srm_factor_tol_enforcement_extra_bisections_used": auto.get(
                "factor_tol_enforcement_extra_bisections_used", 0
            ),
            "srm_factor_tol_numerical_failure_boundary_count": auto.get(
                "factor_tol_numerical_failure_boundary_count", 0
            ),
            "srm_factor_tol_numerical_failure_boundary_factors": auto.get(
                "factor_tol_numerical_failure_boundary_factors", []
            ),
            "srm_factor_tol_require_physical_failure_evidence": auto.get(
                "factor_tol_require_physical_failure_evidence", ""
            ),
            **_srm_parallel_event_fields(srm),
            "message": (
                f"{srm_fos_display(srm)}, trials={len(trial_rows)}, search_mode={srm.get('search_mode', '')}, "
                f"stable={srm.get('stable_factor', '')}, failed={srm.get('failed_factor', '')}"
            ),
        }
    ]
    for index, raw in enumerate(trial_rows, start=1):
        if not isinstance(raw, Mapping):
            continue
        message = str(raw.get("failure_reason") or raw.get("error") or "")
        diagnostics = {key: raw.get(key, "") for key in SRM_TRIAL_DIAGNOSTIC_FIELDS if raw.get(key, "") != ""}
        if diagnostics.get("diagnostic_summary"):
            message = f"{message}; {diagnostics['diagnostic_summary']}" if message else str(diagnostics["diagnostic_summary"])
        rows.append(
            {
                "event_type": "srm_trial",
                "stage": stage.name,
                "index": index,
                "method": "srm_trial",
                "factor_of_safety": fos,
                "srm_factor": raw.get("factor", ""),
                "plastic_ratio": raw.get("plastic_ratio", ""),
                "srm_ok": raw.get("ok", ""),
                "converged": raw.get("converged", ""),
                "failure_reason": raw.get("failure_reason", ""),
                "error": raw.get("error", ""),
                **diagnostics,
                "message": message,
            }
        )
    return rows


def _srm_parallel_event_fields(srm: Mapping[str, Any]) -> dict[str, Any]:
    parallel = srm.get("parallel") if isinstance(srm, Mapping) else None
    if not isinstance(parallel, Mapping):
        return {}
    env = parallel.get("environment", {})
    if not isinstance(env, Mapping):
        env = {}
    mesh = parallel.get("mesh", {})
    if not isinstance(mesh, Mapping):
        mesh = {}
    thread_control = parallel.get("thread_control", {})
    if not isinstance(thread_control, Mapping):
        thread_control = {}
    cost_observation = parallel.get("cost_aware_observation", {})
    if not isinstance(cost_observation, Mapping):
        cost_observation = {}
    return {
        "srm_parallel_enabled": parallel.get("enabled", ""),
        "srm_parallel_strategy": parallel.get("strategy", ""),
        "srm_parallel_context": parallel.get("context", env.get("context", "")),
        "srm_parallel_policy": parallel.get("policy", env.get("policy", "")),
        "srm_parallel_executor": parallel.get("executor", ""),
        "srm_parallel_effective_executor": parallel.get(
            "effective_executor", parallel.get("executor", "")
        ),
        "srm_parallel_process_executor_fallback_count": parallel.get(
            "process_executor_fallback_count", 0
        ),
        "srm_parallel_process_executor_errors": parallel.get(
            "process_executor_errors", []
        ),
        "srm_parallel_max_workers": parallel.get("max_workers", ""),
        "srm_parallel_requested_workers": parallel.get("requested_workers", ""),
        "srm_parallel_selected_threads_per_worker": parallel.get("selected_threads_per_worker", ""),
        "srm_parallel_lookahead_depth": parallel.get("lookahead_depth", ""),
        "srm_parallel_speculative_trial_count": parallel.get("speculative_trial_count", ""),
        "srm_parallel_used_speculative_trial_count": parallel.get("used_speculative_trial_count", ""),
        "srm_parallel_unused_speculative_trial_count": parallel.get("unused_speculative_trial_count", ""),
        "srm_parallel_window_evaluated_trials": parallel.get("window_evaluated_trials", ""),
        "srm_parallel_canceled_speculative_trial_count": parallel.get("canceled_speculative_trial_count", ""),
        "srm_parallel_speculative_cancellation_requested": parallel.get("speculative_cancellation_requested", ""),
        "srm_parallel_speculative_cancellation_note": parallel.get("speculative_cancellation_note", ""),
        "srm_parallel_decision_linked_cancellation_enabled": parallel.get(
            "decision_linked_cancellation_enabled", ""
        ),
        "srm_parallel_decision_linked_requested_count": parallel.get(
            "decision_linked_requested_count", 0
        ),
        "srm_parallel_decision_linked_pending_cancel_count": parallel.get(
            "decision_linked_pending_cancel_count", 0
        ),
        "srm_parallel_decision_linked_safe_stop_count": parallel.get(
            "decision_linked_safe_stop_count", 0
        ),
        "srm_parallel_decision_linked_completed_after_request_count": parallel.get(
            "decision_linked_completed_after_request_count", 0
        ),
        "srm_parallel_decision_linked_requested_factors": parallel.get(
            "decision_linked_requested_factors", []
        ),
        "srm_parallel_speculative_prefetch_call_count": parallel.get("speculative_prefetch_call_count", ""),
        "srm_parallel_speculative_prefetch_wall_elapsed_seconds": parallel.get("speculative_prefetch_wall_elapsed_seconds", ""),
        "srm_parallel_speculative_trial_elapsed_seconds": parallel.get("speculative_trial_elapsed_seconds", ""),
        "srm_parallel_speculative_queue_wait_elapsed_seconds": parallel.get("speculative_queue_wait_elapsed_seconds", ""),
        "srm_parallel_speculative_worker_elapsed_seconds": parallel.get("speculative_worker_elapsed_seconds", ""),
        "srm_parallel_speculative_estimated_wall_clock_saving_seconds": parallel.get("speculative_estimated_wall_clock_saving_seconds", ""),
        "srm_parallel_bisection_speculation_enabled": parallel.get("bisection_speculation_enabled", ""),
        "srm_parallel_bisection_speculative_trial_count": parallel.get("bisection_speculative_trial_count", ""),
        "srm_parallel_bisection_used_speculative_trial_count": parallel.get("bisection_used_speculative_trial_count", ""),
        "srm_parallel_bisection_unused_speculative_trial_count": parallel.get("bisection_unused_speculative_trial_count", ""),
        "srm_parallel_cost_aware_lookahead_enabled": parallel.get("cost_aware_lookahead_enabled", ""),
        "srm_parallel_cost_aware_depth_limited_count": parallel.get("cost_aware_depth_limited_count", 0),
        "srm_parallel_cost_aware_asymmetric_bisection_count": parallel.get("cost_aware_asymmetric_bisection_count", 0),
        "srm_parallel_cost_aware_deferred_candidate_count": parallel.get("cost_aware_deferred_candidate_count", 0),
        "srm_parallel_event_driven_cost_cancellation_enabled": parallel.get("event_driven_cost_cancellation_enabled", ""),
        "srm_parallel_event_driven_cost_shrink_count": parallel.get("event_driven_cost_shrink_count", 0),
        "srm_parallel_event_driven_cost_cancel_candidate_count": parallel.get("event_driven_cost_cancel_candidate_count", 0),
        "srm_parallel_cost_aware_sample_count": cost_observation.get("sample_count", 0),
        "srm_parallel_cost_aware_stable_median_elapsed_seconds": cost_observation.get("stable_median_elapsed_seconds", ""),
        "srm_parallel_cost_aware_failed_median_elapsed_seconds": cost_observation.get("failed_median_elapsed_seconds", ""),
        "srm_parallel_cost_aware_failure_to_stable_cost_ratio": cost_observation.get("failure_to_stable_cost_ratio", ""),
        "srm_parallel_cost_aware_reason": cost_observation.get("reason", ""),
        "srm_parallel_logical_cpus": parallel.get("logical_cpu_count", parallel.get("cpu_count", env.get("logical_cpu_count", ""))),
        "srm_parallel_physical_cpus": parallel.get("physical_cpu_count", env.get("physical_cpu_count", "")),
        "srm_parallel_available_memory_mb": parallel.get("available_memory_mb", env.get("available_memory_mb", "")),
        "srm_parallel_memory_limit_mb": parallel.get("memory_limit_mb", ""),
        "srm_parallel_memory_per_worker_mb": parallel.get("memory_per_worker_mb", ""),
        "srm_parallel_memory_limited": parallel.get("memory_limited", ""),
        "srm_parallel_node_count": mesh.get("node_count", ""),
        "srm_parallel_element_count": mesh.get("element_count", ""),
        "srm_parallel_active_element_count": mesh.get("active_element_count", ""),
        "srm_parallel_dof_count": mesh.get("dof_count", ""),
        "srm_parallel_thread_control_applied": thread_control.get("applied", ""),
        "srm_parallel_thread_control_method": thread_control.get("apply_method", ""),
        "srm_parallel_thread_control_error": thread_control.get("apply_error", thread_control.get("threadpoolctl_error", "")),
        "srm_parallel_thread_environment_restored": thread_control.get("environment_restored", ""),
        "srm_parallel_threadpoolctl_available": thread_control.get("threadpoolctl_available", ""),
    }


def _convergence_events(stage: StageResult2D) -> list[dict[str, Any]]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    history = solver.get("convergence_history")
    rows: list[dict[str, Any]] = []
    if isinstance(history, list) and history:
        for index, raw in enumerate(history, start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "event_type": "convergence_iteration",
                    "stage": stage.name,
                    "index": index,
                    "iteration": raw.get("iteration", index),
                    "method": solver.get("method", ""),
                    "linear_method": raw.get("linear_method", solver.get("linear_method", "")),
                    "residual_norm": raw.get("residual_norm", ""),
                    "pressure_residual_norm": raw.get("pressure_residual_norm", ""),
                    "constraint_norm": raw.get("constraint_norm", ""),
                    "converged": raw.get("converged", ""),
                    "message": raw.get("message", ""),
                }
            )
        return rows
    dynamic = solver.get("dynamic", {})
    if isinstance(dynamic, Mapping) and isinstance(dynamic.get("history"), list):
        for index, raw in enumerate(dynamic["history"], start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "event_type": "time_step",
                    "stage": stage.name,
                    "index": index,
                    "step": raw.get("step", index),
                    "time": raw.get("time", ""),
                    "dt": raw.get("dt", ""),
                    "method": solver.get("method", ""),
                    "iterations": raw.get("nonlinear_iterations", ""),
                    "residual_norm": raw.get("residual_norm", ""),
                    "pressure_residual_norm": raw.get("pressure_residual_norm", ""),
                    "cutbacks": raw.get("cutbacks", ""),
                    "converged": raw.get("converged", ""),
                }
            )
        return rows
    riks = solver.get("riks", {})
    if isinstance(riks, Mapping) and isinstance(riks.get("path"), list):
        for index, raw in enumerate(riks["path"], start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "event_type": "arc_length_step",
                    "stage": stage.name,
                    "index": index,
                    "step": raw.get("step", index),
                    "method": solver.get("method", ""),
                    "iterations": raw.get("iterations", ""),
                    "residual_norm": raw.get("residual_norm", ""),
                    "pressure_residual_norm": raw.get("pressure_residual_norm", ""),
                    "cutbacks": raw.get("cutbacks", ""),
                    "converged": True,
                }
            )
        return rows
    large = solver.get("large_deformation", {})
    if isinstance(large, Mapping) and isinstance(large.get("history"), list):
        for index, raw in enumerate(large["history"], start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "event_type": "large_deformation_increment",
                    "stage": stage.name,
                    "index": index,
                    "step": raw.get("step", index),
                    "method": solver.get("method", ""),
                    "iterations": raw.get("iterations", ""),
                    "residual_norm": raw.get("residual_norm", ""),
                    "pressure_residual_norm": raw.get("pressure_residual_norm", ""),
                    "cutbacks": large.get("cutbacks", ""),
                    "converged": raw.get("converged", ""),
                    "message": f"load={raw.get('load_end', '')}, action={raw.get('adaptive_action', '')}, postprocessed={raw.get('postprocessed', '')}",
                }
            )
        return rows
    consolidation = solver.get("consolidation", {})
    if isinstance(consolidation, Mapping) and isinstance(consolidation.get("step_history"), list):
        for index, raw in enumerate(consolidation["step_history"], start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "event_type": "pressure_time_step",
                    "stage": stage.name,
                    "index": index,
                    "step": raw.get("step", index),
                    "time": raw.get("time", ""),
                    "method": solver.get("method", ""),
                    "iterations": raw.get("outer_iterations", ""),
                    "pressure_residual_norm": raw.get("pressure_residual_norm", ""),
                    "converged": consolidation.get("pressure_converged", ""),
                    "message": f"flow_balance={raw.get('flow_balance', '')}, max_p={raw.get('max_pore_pressure', '')}",
                }
            )
        return rows
    increments = solver.get("increments", {})
    if isinstance(increments, Mapping) and isinstance(increments.get("log"), list):
        for index, raw in enumerate(increments["log"], start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "event_type": "load_increment",
                    "stage": stage.name,
                    "index": index,
                    "step": raw.get("target", ""),
                    "method": solver.get("method", ""),
                    "iterations": raw.get("iterations", ""),
                    "residual_norm": raw.get("residual_norm", ""),
                    "converged": raw.get("accepted", ""),
                    "message": raw.get("error", ""),
                }
            )
        return rows
    return [
        {
            "event_type": "solver_summary",
            "stage": stage.name,
            "iteration": solver.get("iterations", 0),
            "method": solver.get("method", ""),
            "linear_method": solver.get("linear_method", ""),
            "residual_norm": solver.get("residual_norm", ""),
            "converged": solver.get("converged", True),
            "factor_of_safety": _stage_srm_info(stage).get("factor_of_safety", ""),
        }
    ]


def _cutback_events(stage: StageResult2D) -> list[dict[str, Any]]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    out: list[dict[str, Any]] = []
    for key in ("riks", "dynamic", "increments", "large_deformation"):
        block = solver.get(key, {})
        if not isinstance(block, Mapping):
            continue
        rows = block.get("cutback_log", block.get("log", []))
        if not isinstance(rows, list):
            continue
        for index, raw in enumerate(rows, start=1):
            if not isinstance(raw, Mapping) or not raw.get("error"):
                continue
            out.append(
                {
                    "event_type": "cutback",
                    "stage": stage.name,
                    "index": index,
                    "step": raw.get("step", raw.get("target", "")),
                    "time": raw.get("time", ""),
                    "dt": raw.get("dt", ""),
                    "cutbacks": block.get("cutbacks", ""),
                    "message": raw.get("error", ""),
                }
            )
    return out


def _analysis_log_html(log: Mapping[str, Any]) -> str:
    rows = []
    for event in log.get("events", []):
        if not isinstance(event, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(event.get('event_type', '')))}</td>"
            f"<td>{html.escape(str(event.get('stage', '')))}</td>"
            f"<td>{html.escape(str(event.get('iteration', event.get('step', event.get('index', '')))))}</td>"
            f"<td>{html.escape(str(event.get('residual_norm', '')))}</td>"
            f"<td>{html.escape(str(event.get('converged', '')))}</td>"
            f"<td>{html.escape(str(event.get('factor_of_safety', '')))}</td>"
            f"<td>{html.escape(str(event.get('srm_factor', '')))}</td>"
            f"<td>{html.escape(str(event.get('plastic_ratio', '')))}</td>"
            f"<td>{html.escape(str(event.get('srm_ok', '')))}</td>"
            f"<td>{html.escape(str(event.get('auto_decision', '')))}</td>"
            f"<td>{html.escape(str(event.get('auto_trial_action', '')))}</td>"
            f"<td>{html.escape(str(event.get('auto_retry_index', '')))}</td>"
            f"<td>{html.escape(str(event.get('elapsed_seconds', '')))}</td>"
            f"<td>{html.escape(str(event.get('estimated_fos_from_last_load', '')))}</td>"
            f"<td>{html.escape(str(event.get('trial_status', '')))}</td>"
            f"<td>{html.escape(str(event.get('last_accepted_load_factor', '')))}</td>"
            f"<td>{html.escape(str(event.get('last_accepted_plastic_ratio', '')))}</td>"
            f"<td>{html.escape(str(event.get('plastic_ratio_delta', '')))}</td>"
            f"<td>{html.escape(str(event.get('max_equivalent_plastic_strain', '')))}</td>"
            f"<td>{html.escape(str(event.get('connected_plastic_cluster_size', '')))}</td>"
            f"<td>{html.escape(str(event.get('message', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM analysis log</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>Analysis Log</h1>
<p>stages={int(log.get('stage_count', 0) or 0)}, warnings={int(log.get('warning_count', 0) or 0)}</p>
<table><thead><tr><th>event</th><th>stage</th><th>iteration/step</th><th>residual</th><th>converged</th><th>FOS</th><th>SRM factor</th><th>plastic ratio</th><th>SRM OK</th><th>auto decision</th><th>auto action</th><th>retry</th><th>elapsed</th><th>est. FOS</th><th>trial status</th><th>last load</th><th>last accepted plastic ratio</th><th>plastic ratio delta</th><th>max eqp</th><th>cluster</th><th>message</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


__all__ = ["build_structured_analysis_log", "write_structured_analysis_log"]
