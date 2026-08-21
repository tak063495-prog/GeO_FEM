"""Vectorized helpers for sparse finite-element matrix assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


_DIAGNOSTICS_LOCK = threading.Lock()
_DIAGNOSTICS_ENABLED = False
_DIAGNOSTICS: dict[str, int | float] = {
    "pattern_build_count": 0,
    "pattern_assemble_count": 0,
    "pattern_direct_scatter_count": 0,
    "pattern_duplicate_scatter_count": 0,
    "pattern_flat_value_count": 0,
    "builder_to_csr_count": 0,
    "builder_block_count": 0,
    "builder_value_count": 0,
    "builder_elapsed_seconds": 0.0,
}


def _record_diagnostics(**values: int | float) -> None:
    if not _DIAGNOSTICS_ENABLED:
        return
    with _DIAGNOSTICS_LOCK:
        for key, value in values.items():
            _DIAGNOSTICS[key] = _DIAGNOSTICS.get(key, 0) + value


def sparse_assembly_diagnostics(*, reset: bool = False) -> dict[str, Any]:
    with _DIAGNOSTICS_LOCK:
        snapshot = dict(_DIAGNOSTICS)
        if reset:
            for key, value in _DIAGNOSTICS.items():
                _DIAGNOSTICS[key] = 0.0 if isinstance(value, float) else 0
    snapshot["fallback_builder_used"] = bool(int(snapshot.get("builder_to_csr_count", 0) or 0) > 0)
    snapshot["duplicate_scatter_used"] = bool(int(snapshot.get("pattern_duplicate_scatter_count", 0) or 0) > 0)
    return snapshot


def reset_sparse_assembly_diagnostics() -> None:
    sparse_assembly_diagnostics(reset=True)


def set_sparse_assembly_diagnostics_enabled(enabled: bool) -> None:
    global _DIAGNOSTICS_ENABLED
    _DIAGNOSTICS_ENABLED = bool(enabled)


@dataclass(frozen=True)
class SparseAssemblyPattern:
    """Reusable CSR sparsity pattern with scatter positions for dense blocks."""

    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    scatter_positions: np.ndarray
    block_sizes: tuple[int, ...]
    block_offsets: np.ndarray
    scatter_has_duplicates: bool = False

    @classmethod
    def from_square_blocks(cls, dof_blocks: Iterable[Iterable[int] | np.ndarray], shape: tuple[int, int]) -> "SparseAssemblyPattern":
        block_rows: list[np.ndarray] = []
        block_cols: list[np.ndarray] = []
        block_sizes: list[int] = []
        for dofs in dof_blocks:
            arr = np.asarray(dofs if isinstance(dofs, np.ndarray) else list(dofs), dtype=np.int64)
            if arr.size == 0:
                block_sizes.append(0)
                continue
            block_rows.append(np.repeat(arr, arr.size))
            block_cols.append(np.tile(arr, arr.size))
            block_sizes.append(int(arr.size * arr.size))
        offsets = cls._offsets_from_sizes(block_sizes)
        if not block_rows:
            _record_diagnostics(pattern_build_count=1)
            template = csr_matrix(shape, dtype=np.float64)
            return cls(shape, template.indices.copy(), template.indptr.copy(), np.zeros(0, dtype=np.int64), tuple(block_sizes), offsets)

        rows = np.concatenate(block_rows)
        cols = np.concatenate(block_cols)
        ncols = int(shape[1])
        keys = rows * ncols + cols
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        scatter_has_duplicates = bool(unique_keys.size != keys.size)
        unique_rows = unique_keys // ncols
        unique_cols = unique_keys % ncols
        template = coo_matrix((np.ones(unique_keys.size, dtype=np.float64), (unique_rows, unique_cols)), shape=shape).tocsr()
        positions = np.empty(unique_keys.size, dtype=np.int64)
        for index, (row, col) in enumerate(zip(unique_rows, unique_cols, strict=False)):
            start = int(template.indptr[int(row)])
            end = int(template.indptr[int(row) + 1])
            local = int(np.searchsorted(template.indices[start:end], int(col)))
            positions[index] = start + local
        _record_diagnostics(pattern_build_count=1)
        return cls(shape, template.indices.copy(), template.indptr.copy(), positions[inverse].astype(np.int64, copy=False), tuple(block_sizes), offsets, scatter_has_duplicates)

    @classmethod
    def from_blocks(
        cls,
        row_blocks: Iterable[Iterable[int] | np.ndarray],
        col_blocks: Iterable[Iterable[int] | np.ndarray],
        shape: tuple[int, int],
    ) -> "SparseAssemblyPattern":
        block_rows: list[np.ndarray] = []
        block_cols: list[np.ndarray] = []
        block_sizes: list[int] = []
        for row_ids, col_ids in zip(row_blocks, col_blocks, strict=True):
            row_arr = np.asarray(row_ids if isinstance(row_ids, np.ndarray) else list(row_ids), dtype=np.int64)
            col_arr = np.asarray(col_ids if isinstance(col_ids, np.ndarray) else list(col_ids), dtype=np.int64)
            if row_arr.size == 0 or col_arr.size == 0:
                block_sizes.append(0)
                continue
            block_rows.append(np.repeat(row_arr, col_arr.size))
            block_cols.append(np.tile(col_arr, row_arr.size))
            block_sizes.append(int(row_arr.size * col_arr.size))
        offsets = cls._offsets_from_sizes(block_sizes)
        if not block_rows:
            _record_diagnostics(pattern_build_count=1)
            template = csr_matrix(shape, dtype=np.float64)
            return cls(shape, template.indices.copy(), template.indptr.copy(), np.zeros(0, dtype=np.int64), tuple(block_sizes), offsets)

        rows = np.concatenate(block_rows)
        cols = np.concatenate(block_cols)
        ncols = int(shape[1])
        keys = rows * ncols + cols
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        scatter_has_duplicates = bool(unique_keys.size != keys.size)
        unique_rows = unique_keys // ncols
        unique_cols = unique_keys % ncols
        template = coo_matrix((np.ones(unique_keys.size, dtype=np.float64), (unique_rows, unique_cols)), shape=shape).tocsr()
        positions = np.empty(unique_keys.size, dtype=np.int64)
        for index, (row, col) in enumerate(zip(unique_rows, unique_cols, strict=False)):
            start = int(template.indptr[int(row)])
            end = int(template.indptr[int(row) + 1])
            local = int(np.searchsorted(template.indices[start:end], int(col)))
            positions[index] = start + local
        _record_diagnostics(pattern_build_count=1)
        return cls(shape, template.indices.copy(), template.indptr.copy(), positions[inverse].astype(np.int64, copy=False), tuple(block_sizes), offsets, scatter_has_duplicates)

    @staticmethod
    def _offsets_from_sizes(block_sizes: Iterable[int]) -> np.ndarray:
        sizes = [int(size) for size in block_sizes]
        offsets = np.empty(len(sizes) + 1, dtype=np.int64)
        offsets[0] = 0
        for index, size in enumerate(sizes):
            offsets[index + 1] = offsets[index] + size
        return offsets

    @property
    def block_count(self) -> int:
        return len(self.block_sizes)

    @property
    def flat_value_size(self) -> int:
        return int(self.block_offsets[-1]) if self.block_offsets.size else 0

    def empty_flat_values(self) -> np.ndarray:
        return np.zeros(self.flat_value_size, dtype=np.float64)

    def fill_block(self, flat_values: np.ndarray, block_index: int, block: np.ndarray) -> None:
        if block_index < 0 or block_index >= self.block_count:
            raise ValueError(f"sparse block index out of range: {block_index}")
        arr = np.asarray(block, dtype=np.float64).ravel()
        expected_size = int(self.block_sizes[block_index])
        if arr.size != expected_size:
            raise ValueError(f"sparse block size mismatch at {block_index}: expected {expected_size}, got {arr.size}")
        start = int(self.block_offsets[block_index])
        end = int(self.block_offsets[block_index + 1])
        if end > np.asarray(flat_values).size:
            raise ValueError("sparse flat value buffer is too small")
        if arr.size:
            flat_values[start:end] = arr

    def fill_blocks_flat(self, flat_values: np.ndarray, block_index: int, values: np.ndarray, block_count: int) -> int:
        count = int(block_count)
        if count < 0 or block_index < 0 or block_index + count > self.block_count:
            raise ValueError(f"sparse block range out of range: start={block_index}, count={count}")
        arr = np.asarray(values, dtype=np.float64).ravel()
        start = int(self.block_offsets[block_index])
        end = int(self.block_offsets[block_index + count])
        expected_size = end - start
        if arr.size != expected_size:
            raise ValueError(f"sparse flat block range size mismatch: expected {expected_size}, got {arr.size}")
        if arr.size:
            flat_values[start:end] = arr
        return block_index + count

    def validate_filled_block_count(self, block_count: int) -> None:
        if int(block_count) != self.block_count:
            raise ValueError(f"sparse block count mismatch: expected {self.block_count}, got {int(block_count)}")

    def direct_fill_info(self) -> dict[str, int | bool | str]:
        return {
            "enabled": True,
            "mode": "flat_offset_direct_fill",
            "block_count": self.block_count,
            "flat_value_size": self.flat_value_size,
            "nnz": int(self.indices.size),
            "scatter_mode": "accumulate" if self.scatter_has_duplicates else "direct",
        }

    def assemble(self, blocks: Iterable[np.ndarray]) -> csr_matrix:
        flat_values = self.empty_flat_values()
        block_index = 0
        for block, _expected_size in zip(blocks, self.block_sizes, strict=True):
            self.fill_block(flat_values, block_index, block)
            block_index += 1
        self.validate_filled_block_count(block_index)
        return self.assemble_flat_values(flat_values)

    def assemble_flat_values(self, flat_values: np.ndarray) -> csr_matrix:
        values = np.asarray(flat_values, dtype=np.float64).ravel()
        expected = int(sum(self.block_sizes))
        if values.size != expected:
            raise ValueError(f"sparse flat value size mismatch: expected {expected}, got {values.size}")
        data = np.zeros(self.indices.size, dtype=np.float64)
        if values.size:
            if not self.scatter_has_duplicates and self.scatter_positions.size == values.size:
                data[self.scatter_positions] = values
                _record_diagnostics(
                    pattern_assemble_count=1,
                    pattern_direct_scatter_count=1,
                    pattern_flat_value_count=int(values.size),
                )
            else:
                np.add.at(data, self.scatter_positions, values)
                _record_diagnostics(
                    pattern_assemble_count=1,
                    pattern_duplicate_scatter_count=1,
                    pattern_flat_value_count=int(values.size),
                )
        else:
            _record_diagnostics(pattern_assemble_count=1)
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)


@dataclass
class SparseAssemblyBuilder:
    """Collect dense element blocks as NumPy chunks before one COO conversion."""

    rows: list[np.ndarray] = field(default_factory=list)
    cols: list[np.ndarray] = field(default_factory=list)
    data: list[np.ndarray] = field(default_factory=list)

    def add_block(
        self,
        row_ids: Iterable[int] | np.ndarray,
        col_ids: Iterable[int] | np.ndarray,
        block: np.ndarray,
    ) -> None:
        row_arr = np.asarray(row_ids if isinstance(row_ids, np.ndarray) else list(row_ids), dtype=np.int64)
        col_arr = np.asarray(col_ids if isinstance(col_ids, np.ndarray) else list(col_ids), dtype=np.int64)
        block_arr = np.asarray(block, dtype=np.float64)
        if row_arr.size == 0 or col_arr.size == 0 or block_arr.size == 0:
            return
        self.rows.append(np.repeat(row_arr, col_arr.size))
        self.cols.append(np.tile(col_arr, row_arr.size))
        self.data.append(np.ascontiguousarray(block_arr).ravel())

    def to_csr(self, shape: tuple[int, int]) -> csr_matrix:
        started = perf_counter()
        if not self.data:
            _record_diagnostics(builder_to_csr_count=1, builder_elapsed_seconds=max(perf_counter() - started, 0.0))
            return csr_matrix(shape, dtype=np.float64)
        rows = np.concatenate(self.rows)
        cols = np.concatenate(self.cols)
        data = np.concatenate(self.data)
        result = coo_matrix((data, (rows, cols)), shape=shape).tocsr()
        _record_diagnostics(
            builder_to_csr_count=1,
            builder_block_count=len(self.data),
            builder_value_count=int(data.size),
            builder_elapsed_seconds=max(perf_counter() - started, 0.0),
        )
        return result


__all__ = [
    "SparseAssemblyBuilder",
    "SparseAssemblyPattern",
    "reset_sparse_assembly_diagnostics",
    "set_sparse_assembly_diagnostics_enabled",
    "sparse_assembly_diagnostics",
]
