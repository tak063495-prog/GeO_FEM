"""Hydraulic active-set iteration control for 2D coupled solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

HYDRO_ITERATION_FUNCTIONS = (
    "SeepageActiveSetState",
    "hydro_iteration_contract",
    "seepage_outer_iteration_limit",
    "seepage_active_signature",
    "advance_seepage_active_set",
    "observe_seepage_active_set",
)


@dataclass(frozen=True)
class SeepageActiveSetState:
    signature: tuple[int, int] | None = None
    toggle_count: int = 0


def hydro_iteration_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.hydro_iteration.v1",
        "module": "geofem_app.fem2d_hydro_iteration",
        "function_count": len(HYDRO_ITERATION_FUNCTIONS),
        "functions": list(HYDRO_ITERATION_FUNCTIONS),
        "covered_surfaces": [
            "seepage_active_set_signature",
            "seepage_outer_iteration_limit",
            "seepage_fixed_point_stop",
            "seepage_toggle_count",
        ],
    }


def seepage_outer_iteration_limit(hydro: Mapping[str, Any], *, default: int = 8) -> int:
    raw = hydro.get("seepage_max_outer", hydro.get("max_outer", default))
    return max(1, int(raw))


def seepage_active_signature(boundary_info: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int(boundary_info.get("seepage_active_edges", 0)),
        int(boundary_info.get("seepage_inactive_edges", 0)),
    )


def observe_seepage_active_set(
    state: SeepageActiveSetState,
    boundary_info: Mapping[str, Any],
) -> SeepageActiveSetState:
    signature = seepage_active_signature(boundary_info)
    toggles = state.toggle_count
    if state.signature is not None and signature != state.signature:
        toggles += 1
    return SeepageActiveSetState(signature=signature, toggle_count=toggles)


def advance_seepage_active_set(
    state: SeepageActiveSetState,
    boundary_info: Mapping[str, Any],
) -> tuple[SeepageActiveSetState, bool]:
    signature = seepage_active_signature(boundary_info)
    if int(boundary_info.get("seepage_count", 0)) == 0 or signature == state.signature:
        return state, True
    return observe_seepage_active_set(state, boundary_info), False


__all__ = list(HYDRO_ITERATION_FUNCTIONS)
