"""Array views for plastic state histories keyed by element/integration point."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .fem2d_types import ElasticPlaneStrainMaterial, Mesh2D, PlasticState2D, PlasticStateView2D, normalize_integration


@dataclass(frozen=True)
class PlasticStateArrayCache:
    element_ids: tuple[str, ...]
    element_row: dict[str, int]
    state_point_counts: np.ndarray
    plastic_strains: np.ndarray
    kappas: np.ndarray
    present: np.ndarray
    state_var_flags: np.ndarray
    state_objects: np.ndarray
    source_state_count: int = 0

    def state_arrays(self, element_id: str, point_count: int, *, require_no_state_vars: bool = False) -> tuple[np.ndarray, np.ndarray]:
        count = int(point_count)
        if count < 0:
            raise ValueError("point_count must be non-negative")
        row = self.element_row.get(str(element_id))
        if row is None:
            return np.zeros((count, 4), dtype=float), np.zeros(count, dtype=float)
        if count > self.plastic_strains.shape[1]:
            raise ValueError(f"element {element_id}: requested {count} plastic state points, cache has {self.plastic_strains.shape[1]}")
        if require_no_state_vars and bool(np.any(self.state_var_flags[row, :count])):
            raise ValueError(f"element {element_id}: state_vars require the safe non-batch plastic path")
        return self.plastic_strains[row, :count, :], self.kappas[row, :count]

    def has_state_vars(self, element_id: str, point_count: int) -> bool:
        row = self.element_row.get(str(element_id))
        if row is None or int(point_count) <= 0:
            return False
        count = min(int(point_count), self.state_var_flags.shape[1])
        return bool(np.any(self.state_var_flags[row, :count]))

    def state_view_for_gp(self, element_id: str, gp_index: int) -> PlasticState2D | PlasticStateView2D | None:
        row = self.element_row.get(str(element_id))
        gp = int(gp_index)
        if row is None or gp < 0 or gp >= self.plastic_strains.shape[1]:
            return None
        stored = self.state_objects[row, gp]
        if isinstance(stored, PlasticState2D) and stored.state_vars:
            return stored
        if not bool(self.present[row, gp]):
            return None
        return PlasticStateView2D(self.plastic_strains[row, gp, :], float(self.kappas[row, gp]))

    def state_for_gp(self, element_id: str, gp_index: int) -> PlasticState2D | None:
        state = self.state_view_for_gp(element_id, gp_index)
        if isinstance(state, PlasticStateView2D):
            return PlasticState2D(state.plastic_strain.copy(), float(state.kappa))
        return state

    def solver_info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "layout": "active_element_major_integration_point_minor",
            "elements": int(len(self.element_ids)),
            "max_state_points_per_element": int(self.plastic_strains.shape[1]) if self.plastic_strains.ndim >= 2 else 0,
            "state_points": int(np.sum(self.state_point_counts)) if self.state_point_counts.size else 0,
            "present_points": int(np.count_nonzero(self.present)),
            "state_var_points": int(np.count_nonzero(self.state_var_flags)),
            "numeric_state_view_points": int(np.count_nonzero(np.logical_and(self.present, np.logical_not(self.state_var_flags)))),
            "numeric_state_view": "array_backed",
            "source_state_count": int(self.source_state_count),
        }

    def plastic_ratio(self, active_elements: list[str] | tuple[str, ...] | set[str] | None = None) -> float:
        active = {str(element_id) for element_id in active_elements} if active_elements is not None else set(self.element_ids)
        if not active:
            return 0.0
        plastic_elements = 0
        for element_id in active:
            row = self.element_row.get(str(element_id))
            if row is None:
                continue
            count = min(int(self.state_point_counts[row]), self.plastic_strains.shape[1])
            if count <= 0:
                continue
            strains = self.plastic_strains[row, :count, :]
            kappas = self.kappas[row, :count]
            has_plastic = bool(np.any(np.linalg.norm(strains, axis=1) > 0.0) or np.any(kappas > 0.0))
            if not has_plastic and self.state_objects.size:
                for stored in self.state_objects[row, :count]:
                    if isinstance(stored, PlasticState2D) and _state_object_is_plastic(stored):
                        has_plastic = True
                        break
            if has_plastic:
                plastic_elements += 1
        return plastic_elements / len(active)

    def plastic_point_count(self) -> int:
        if self.plastic_strains.size == 0:
            return 0
        strain_norm = np.linalg.norm(self.plastic_strains, axis=2) if self.plastic_strains.ndim == 3 else np.zeros(0, dtype=float)
        numeric = np.logical_or(strain_norm > 0.0, self.kappas > 0.0)
        if self.state_objects.size:
            object_plastic = np.zeros_like(numeric, dtype=bool)
            for index, stored in enumerate(self.state_objects.ravel()):
                if isinstance(stored, PlasticState2D) and _state_object_is_plastic(stored):
                    object_plastic.ravel()[index] = True
            numeric = np.logical_or(numeric, object_plastic)
        return int(np.count_nonzero(numeric))

    def to_state_dict(self, *, include_zero_states: bool = True) -> dict[str, PlasticState2D]:
        state: dict[str, PlasticState2D] = {}
        for row, element_id in enumerate(self.element_ids):
            count = min(int(self.state_point_counts[row]), self.plastic_strains.shape[1])
            for gp_index in range(count):
                stored = self.state_objects[row, gp_index]
                if isinstance(stored, PlasticState2D) and stored.state_vars:
                    state[f"{element_id}:{gp_index}"] = stored
                    continue
                strain = self.plastic_strains[row, gp_index, :].copy()
                kappa = float(self.kappas[row, gp_index])
                if include_zero_states or bool(np.linalg.norm(strain) > 0.0 or kappa > 0.0 or self.present[row, gp_index]):
                    state[f"{element_id}:{gp_index}"] = PlasticState2D(strain, kappa)
        return state


class ArrayBackedPlasticStateMapping(MappingABC):
    """Lazy Mapping view over PlasticStateArrayCache.

    Numeric states are exposed as PlasticStateView2D until callers explicitly
    iterate values/items or request a concrete dict.
    """

    __slots__ = ("_cache", "_overlay")

    def __init__(
        self,
        cache: PlasticStateArrayCache | None,
        overlay: Mapping[str, PlasticState2D] | None = None,
    ) -> None:
        self._cache = cache
        self._overlay = {str(key): value for key, value in (overlay or {}).items()}

    @property
    def cache(self) -> PlasticStateArrayCache | None:
        return self._cache

    def _cache_key_count(self) -> int:
        cache = self._cache
        if cache is None or cache.state_point_counts.size == 0:
            return 0
        return int(np.sum(cache.state_point_counts))

    def __len__(self) -> int:
        if self._cache is None:
            return len(self._overlay)
        overlay_extra = 0
        for key in self._overlay:
            if not self._has_cache_key(key):
                overlay_extra += 1
        return self._cache_key_count() + overlay_extra

    def __iter__(self):
        yielded: set[str] = set()
        for key in self._overlay:
            yielded.add(key)
            yield key
        cache = self._cache
        if cache is None:
            return
        for row, element_id in enumerate(cache.element_ids):
            count = min(int(cache.state_point_counts[row]), cache.plastic_strains.shape[1])
            for gp_index in range(count):
                key = f"{element_id}:{gp_index}"
                if key not in yielded:
                    yield key

    def __getitem__(self, key: str) -> PlasticState2D | PlasticStateView2D:
        key = str(key)
        if key in self._overlay:
            return self._overlay[key]
        cache = self._cache
        if cache is None:
            raise KeyError(key)
        element_id, gp_index = self._split_key(key)
        row = cache.element_row.get(element_id)
        if row is None:
            raise KeyError(key)
        gp = int(gp_index)
        count = min(int(cache.state_point_counts[row]), cache.plastic_strains.shape[1])
        if gp < 0 or gp >= count:
            raise KeyError(key)
        stored = cache.state_objects[row, gp]
        if isinstance(stored, PlasticState2D) and stored.state_vars:
            return stored
        return PlasticStateView2D(cache.plastic_strains[row, gp, :], float(cache.kappas[row, gp]))

    def _has_cache_key(self, key: str) -> bool:
        cache = self._cache
        if cache is None:
            return False
        try:
            element_id, gp_index = self._split_key(key)
        except KeyError:
            return False
        row = cache.element_row.get(element_id)
        if row is None:
            return False
        count = min(int(cache.state_point_counts[row]), cache.plastic_strains.shape[1])
        return 0 <= int(gp_index) < count

    @staticmethod
    def _split_key(key: str) -> tuple[str, int]:
        try:
            element_id, raw_gp = str(key).rsplit(":", 1)
            return element_id, int(raw_gp)
        except (ValueError, TypeError) as exc:
            raise KeyError(key) from exc

    def to_state_dict(self, *, include_zero_states: bool = True) -> dict[str, PlasticState2D]:
        if self._cache is None:
            return dict(self._overlay)
        state = self._cache.to_state_dict(include_zero_states=include_zero_states)
        state.update(self._overlay)
        return state

    def to_array_cache(
        self,
        mesh: Mesh2D,
        materials: Mapping[str, ElasticPlaneStrainMaterial],
    ) -> PlasticStateArrayCache:
        if self._cache is None:
            return build_plastic_state_array_cache(mesh, materials, self._overlay)
        if not self._overlay:
            return self._cache
        strains = self._cache.plastic_strains.copy()
        kappas = self._cache.kappas.copy()
        present = self._cache.present.copy()
        state_var_flags = self._cache.state_var_flags.copy()
        state_objects = self._cache.state_objects.copy()
        for key, state in self._overlay.items():
            try:
                element_id, gp_index = self._split_key(key)
            except KeyError:
                continue
            row = self._cache.element_row.get(element_id)
            if row is None or gp_index < 0 or gp_index >= strains.shape[1]:
                continue
            strain = np.asarray(state.plastic_strain, dtype=float)
            if strain.shape == (4,):
                strains[row, gp_index, :] = strain
            kappas[row, gp_index] = float(state.kappa)
            present[row, gp_index] = True
            if state.state_vars:
                state_var_flags[row, gp_index] = True
                state_objects[row, gp_index] = state
            else:
                state_var_flags[row, gp_index] = False
                state_objects[row, gp_index] = None
        return PlasticStateArrayCache(
            element_ids=self._cache.element_ids,
            element_row=dict(self._cache.element_row),
            state_point_counts=self._cache.state_point_counts.copy(),
            plastic_strains=np.ascontiguousarray(strains, dtype=np.float64),
            kappas=np.ascontiguousarray(kappas, dtype=np.float64),
            present=present,
            state_var_flags=state_var_flags,
            state_objects=state_objects,
            source_state_count=int(self._cache.source_state_count + len(self._overlay)),
        )


def _state_object_is_plastic(state: PlasticState2D) -> bool:
    try:
        if bool(state.state_vars.get("plastic", False)):
            return True
        if float(state.kappa) > 0.0:
            return True
        return bool(np.linalg.norm(np.asarray(state.plastic_strain, dtype=float)) > 0.0)
    except (AttributeError, TypeError, ValueError):
        return False


def plastic_state_point_count(
    element_type: str,
    integration: str,
    material: ElasticPlaneStrainMaterial | None = None,
) -> int:
    if material is not None and not material.is_plastic:
        return 0
    etype = str(element_type).upper().strip()
    mode = normalize_integration(integration)
    if etype == "QUAD8":
        return 13 if mode == "SRI" else 9
    if etype == "QUAD4":
        return 5 if mode == "SRI" else 4
    if etype == "TRI6":
        return 3
    if etype == "TRI3":
        return 1
    return 0


def build_plastic_state_array_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> PlasticStateArrayCache:
    active = [(index, element) for index, element in enumerate(mesh.elements) if element.active]
    element_ids = tuple(str(element.id) for _index, element in active)
    counts = np.asarray(
        [plastic_state_point_count(element.type, element.integration, materials.get(element.material)) for _index, element in active],
        dtype=np.int64,
    )
    max_points = int(np.max(counts)) if counts.size else 0
    strains = np.zeros((len(active), max_points, 4), dtype=float)
    kappas = np.zeros((len(active), max_points), dtype=float)
    present = np.zeros((len(active), max_points), dtype=bool)
    state_var_flags = np.zeros((len(active), max_points), dtype=bool)
    state_objects = np.empty((len(active), max_points), dtype=object)
    state_objects[:, :] = None
    source = plastic_state or {}
    for row, (_index, element) in enumerate(active):
        for gp_index in range(int(counts[row])):
            state = source.get(f"{element.id}:{gp_index}")
            if state is None:
                continue
            present[row, gp_index] = True
            if state.state_vars:
                state_objects[row, gp_index] = state
                state_var_flags[row, gp_index] = True
            strain = np.asarray(state.plastic_strain, dtype=float)
            if strain.shape == (4,):
                strains[row, gp_index, :] = strain
            kappas[row, gp_index] = float(state.kappa)
    return PlasticStateArrayCache(
        element_ids=element_ids,
        element_row={element_id: row for row, element_id in enumerate(element_ids)},
        state_point_counts=counts,
        plastic_strains=np.ascontiguousarray(strains, dtype=np.float64),
        kappas=np.ascontiguousarray(kappas, dtype=np.float64),
        present=present,
        state_var_flags=state_var_flags,
        state_objects=state_objects,
        source_state_count=len(source),
    )


def plastic_state_array_cache_info(cache: PlasticStateArrayCache | None) -> dict[str, Any]:
    if cache is None:
        return {"enabled": False}
    return cache.solver_info()


__all__ = [
    "ArrayBackedPlasticStateMapping",
    "PlasticStateArrayCache",
    "build_plastic_state_array_cache",
    "plastic_state_array_cache_info",
    "plastic_state_point_count",
]
