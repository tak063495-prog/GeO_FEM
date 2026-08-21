"""Small shared helpers for the 2D FEM modules."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .fem2d_types import FEM2DError


def _element_dofs(nodes: Iterable[str], node_index: Mapping[str, int]) -> np.ndarray:
    return _dofs_from_node_indices(_element_node_indices(nodes, node_index))


def _element_node_indices(nodes: Iterable[str], node_index: Mapping[str, int]) -> np.ndarray:
    return np.asarray([node_index[nid] for nid in nodes], dtype=int)


def _dofs_from_node_indices(node_indices: Iterable[int] | np.ndarray) -> np.ndarray:
    indices = np.asarray(list(node_indices) if not isinstance(node_indices, np.ndarray) else node_indices, dtype=int)
    dofs = np.empty(indices.size * 2, dtype=int)
    dofs[0::2] = 2 * indices
    dofs[1::2] = 2 * indices + 1
    return dofs


def _append_sparse_block(
    rows: list[int],
    cols: list[int],
    data: list[float],
    row_ids: Iterable[int] | np.ndarray,
    col_ids: Iterable[int] | np.ndarray,
    block: np.ndarray,
) -> None:
    row_arr = np.asarray(list(row_ids) if not isinstance(row_ids, np.ndarray) else row_ids, dtype=int)
    col_arr = np.asarray(list(col_ids) if not isinstance(col_ids, np.ndarray) else col_ids, dtype=int)
    block_arr = np.asarray(block, dtype=float)
    rows.extend(np.repeat(row_arr, col_arr.size).tolist())
    cols.extend(np.tile(col_arr, row_arr.size).tolist())
    data.extend(block_arr.ravel().tolist())


def _as_xy(value: Any, where: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return float(value.get("x", value.get("X"))), float(value.get("y", value.get("Y")))
    seq = _require_sequence(value, where)
    if len(seq) < 2:
        raise FEM2DError(f"{where} must contain x and y")
    return float(seq[0]), float(seq[1])


def _range2(value: Any, where: str) -> tuple[float, float]:
    seq = _require_sequence(value, where)
    if len(seq) != 2:
        raise FEM2DError(f"{where} must have exactly two values")
    return float(seq[0]), float(seq[1])


def _require_sequence(value: Any, where: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise FEM2DError(f"{where} must be a list")
    return list(value)


def _sets_from_mapping(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): [str(v) for v in _require_sequence(vals, f"set {k}")] for k, vals in raw.items()}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _merge_lists(a: Any, b: Any) -> list[Any]:
    return _ensure_list(a) + _ensure_list(b)


def _merge_solver_config(base: Any, override: Any) -> dict[str, Any]:
    if not isinstance(base, Mapping):
        merged: dict[str, Any] = {}
    else:
        merged = {str(k): v for k, v in base.items()}
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        key_str = str(key)
        if isinstance(value, Mapping) and isinstance(merged.get(key_str), Mapping):
            merged[key_str] = _merge_solver_config(merged[key_str], value)
        else:
            merged[key_str] = value
    return merged


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return safe or "stage"

__all__ = [
    "_element_dofs",
    "_element_node_indices",
    "_dofs_from_node_indices",
    "_append_sparse_block",
    "_as_xy",
    "_range2",
    "_require_sequence",
    "_sets_from_mapping",
    "_ensure_list",
    "_merge_lists",
    "_merge_solver_config",
    "_safe_name",
]

