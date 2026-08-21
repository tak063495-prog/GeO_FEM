"""Shared GeoFEAS/VGFlow2D open hydrologic exchange helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .fem2d_types import Mesh2D


HYDRO_EXCHANGE_PROFILE = "geofem.shared_hydro_exchange.public_substitute.v1"


def shared_hydro_exchange_engine() -> dict[str, Any]:
    return {
        "schema": HYDRO_EXCHANGE_PROFILE,
        "module": "geofem_app.hydro_exchange",
        "functions": [
            "pressure_head_from_total",
            "total_head_from_pressure",
            "waterline_points_from_total_head",
            "element_potential_points",
        ],
        "used_by": ["GeoFEAS external seepage handoff", "VGFlow2D PRS/PTN open exchange"],
        "native_commercial_binary_equivalence": False,
    }


def pressure_head_from_total(mesh: Mesh2D, node_id: str, total_head: float, problem_type: str) -> float:
    if _is_horizontal(problem_type):
        return float(total_head)
    return float(total_head) - float(mesh.coords[mesh.node_index[node_id], 1])


def total_head_from_pressure(mesh: Mesh2D, node_id: str, pressure_head: float, problem_type: str) -> float:
    if _is_horizontal(problem_type):
        return float(pressure_head)
    return float(pressure_head) + float(mesh.coords[mesh.node_index[node_id], 1])


def waterline_points_from_total_head(
    mesh: Mesh2D,
    total_head: Sequence[float] | np.ndarray,
    problem_type: str,
    *,
    sort_points: bool = False,
) -> list[tuple[float, float]]:
    head = np.asarray(total_head, dtype=float)
    points: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for element in mesh.elements:
        corners = list(element.nodes[:4] if element.type.upper().startswith("QUAD") else element.nodes[:3])
        for a, b in zip(corners, [*corners[1:], corners[0]]):
            ia, ib = mesh.node_index[a], mesh.node_index[b]
            pa = pressure_head_from_total(mesh, a, float(head[ia]), problem_type)
            pb = pressure_head_from_total(mesh, b, float(head[ib]), problem_type)
            if pa == 0.0 and pb == 0.0:
                continue
            if pa * pb > 0.0:
                continue
            denom = pa - pb
            t = 0.0 if abs(denom) <= np.finfo(float).eps else pa / denom
            t = max(0.0, min(1.0, t))
            xy = mesh.coords[ia] + t * (mesh.coords[ib] - mesh.coords[ia])
            key = (round(float(xy[0]) * 1.0e9), round(float(xy[1]) * 1.0e9))
            if key not in seen:
                seen.add(key)
                points.append((float(xy[0]), float(xy[1])))
    return sorted(points) if sort_points else points


def element_potential_points(mesh: Mesh2D, total_head: Sequence[float] | np.ndarray) -> list[dict[str, Any]]:
    head = np.asarray(total_head, dtype=float)
    rows: list[dict[str, Any]] = []
    for element in mesh.elements:
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        center = np.mean(coords, axis=0)
        rows.append(
            {
                "element_id": element.id,
                "x": float(center[0]),
                "y": float(center[1]),
                "total_head_m": float(np.mean(head[conn])),
            }
        )
    return rows


def _is_horizontal(problem_type: str) -> bool:
    return str(problem_type).lower().replace("-", "_") == "horizontal"


__all__ = [
    "HYDRO_EXCHANGE_PROFILE",
    "element_potential_points",
    "pressure_head_from_total",
    "shared_hydro_exchange_engine",
    "total_head_from_pressure",
    "waterline_points_from_total_head",
]
