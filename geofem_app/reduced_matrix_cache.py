"""Reusable CSR submatrix extraction for constrained linear solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class CSRSubmatrixCache:
    """Map a CSR source pattern to one fixed row/column submatrix."""

    source_shape: tuple[int, int]
    row_indices: np.ndarray
    col_indices: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    source_positions: np.ndarray
    row_entry_indices: np.ndarray
    source_nnz: int

    @classmethod
    def from_pattern(
        cls,
        source_shape: tuple[int, int],
        source_indptr: np.ndarray,
        source_indices: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> "CSRSubmatrixCache":
        rows_arr = np.asarray(rows, dtype=np.int64).ravel()
        cols_arr = np.asarray(cols, dtype=np.int64).ravel()
        indptr = np.asarray(source_indptr, dtype=np.int64).ravel()
        indices = np.asarray(source_indices, dtype=np.int64).ravel()
        nrows, ncols = int(source_shape[0]), int(source_shape[1])
        if rows_arr.size and (int(np.min(rows_arr)) < 0 or int(np.max(rows_arr)) >= nrows):
            raise ValueError("submatrix row index is outside the source matrix")
        if cols_arr.size and (int(np.min(cols_arr)) < 0 or int(np.max(cols_arr)) >= ncols):
            raise ValueError("submatrix column index is outside the source matrix")

        col_to_local = np.full(ncols, -1, dtype=np.int64)
        if cols_arr.size:
            col_to_local[cols_arr] = np.arange(cols_arr.size, dtype=np.int64)

        out_indptr = np.empty(rows_arr.size + 1, dtype=np.int64)
        out_indptr[0] = 0
        out_indices: list[np.ndarray] = []
        out_positions: list[np.ndarray] = []
        out_rows: list[np.ndarray] = []
        nnz = 0
        for local_row, source_row in enumerate(rows_arr):
            start = int(indptr[int(source_row)])
            end = int(indptr[int(source_row) + 1])
            if start == end or cols_arr.size == 0:
                out_indptr[local_row + 1] = nnz
                continue
            local_cols = col_to_local[indices[start:end]]
            keep = local_cols >= 0
            if np.any(keep):
                kept_cols = local_cols[keep]
                kept_positions = np.arange(start, end, dtype=np.int64)[keep]
                order = np.argsort(kept_cols, kind="stable")
                kept_cols = kept_cols[order]
                kept_positions = kept_positions[order]
                out_indices.append(kept_cols.astype(np.int32, copy=False))
                out_positions.append(kept_positions)
                out_rows.append(np.full(kept_cols.size, local_row, dtype=np.int64))
                nnz += int(kept_cols.size)
            out_indptr[local_row + 1] = nnz

        if out_indices:
            result_indices = np.concatenate(out_indices).astype(np.int32, copy=False)
            source_positions = np.concatenate(out_positions).astype(np.int64, copy=False)
            row_entry_indices = np.concatenate(out_rows).astype(np.int64, copy=False)
        else:
            result_indices = np.zeros(0, dtype=np.int32)
            source_positions = np.zeros(0, dtype=np.int64)
            row_entry_indices = np.zeros(0, dtype=np.int64)
        return cls(
            source_shape=(nrows, ncols),
            row_indices=rows_arr.copy(),
            col_indices=cols_arr.copy(),
            indptr=out_indptr,
            indices=result_indices,
            source_positions=source_positions,
            row_entry_indices=row_entry_indices,
            source_nnz=int(indices.size),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.row_indices.size), int(self.col_indices.size))

    @property
    def nnz(self) -> int:
        return int(self.source_positions.size)

    def extract(self, matrix: csr_matrix) -> csr_matrix:
        data = np.asarray(matrix.data, dtype=float)[self.source_positions] if self.source_positions.size else np.zeros(0, dtype=float)
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def matvec(self, matrix: csr_matrix, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float).ravel()
        if values.size != self.col_indices.size:
            raise ValueError(f"submatrix vector size mismatch: expected {self.col_indices.size}, got {values.size}")
        out = np.zeros(self.row_indices.size, dtype=float)
        if self.source_positions.size:
            products = np.asarray(matrix.data, dtype=float)[self.source_positions] * values[self.indices]
            np.add.at(out, self.row_entry_indices, products)
        return out

    def info(self) -> dict[str, int | list[int]]:
        return {
            "shape": [int(self.row_indices.size), int(self.col_indices.size)],
            "nnz": self.nnz,
            "source_nnz": int(self.source_nnz),
        }


@dataclass(frozen=True)
class ReducedMatrixCache:
    """Cached extraction plans for the free-free and free-fixed blocks."""

    source_shape: tuple[int, int]
    source_indptr: np.ndarray
    source_indices: np.ndarray
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    free_free: CSRSubmatrixCache
    free_fixed: CSRSubmatrixCache
    source: str = "csr_pattern"

    @classmethod
    def from_csr(
        cls,
        matrix: csr_matrix,
        free_dofs: np.ndarray,
        fixed_dofs: np.ndarray,
        *,
        source: str = "csr_matrix",
    ) -> "ReducedMatrixCache":
        matrix_csr = matrix.tocsr()
        return cls.from_pattern(
            matrix_csr.shape,
            matrix_csr.indptr,
            matrix_csr.indices,
            free_dofs,
            fixed_dofs,
            source=source,
        )

    @classmethod
    def from_pattern(
        cls,
        shape: tuple[int, int],
        indptr: np.ndarray,
        indices: np.ndarray,
        free_dofs: np.ndarray,
        fixed_dofs: np.ndarray,
        *,
        source: str = "sparse_pattern",
    ) -> "ReducedMatrixCache":
        free = np.asarray(free_dofs, dtype=np.int64).ravel()
        fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
        shape_tuple = (int(shape[0]), int(shape[1]))
        source_indptr = np.asarray(indptr, dtype=np.int64).ravel().copy()
        source_indices = np.asarray(indices, dtype=np.int64).ravel().copy()
        return cls(
            source_shape=shape_tuple,
            source_indptr=source_indptr,
            source_indices=source_indices,
            free_dofs=free.copy(),
            fixed_dofs=fixed.copy(),
            free_free=CSRSubmatrixCache.from_pattern(shape_tuple, source_indptr, source_indices, free, free),
            free_fixed=CSRSubmatrixCache.from_pattern(shape_tuple, source_indptr, source_indices, free, fixed),
            source=source,
        )

    def matches(
        self,
        matrix: csr_matrix,
        free_dofs: np.ndarray,
        fixed_dofs: np.ndarray,
        *,
        validate_structure: bool = True,
    ) -> bool:
        if tuple(matrix.shape) != self.source_shape:
            return False
        if matrix.indices.size != self.source_indices.size or matrix.indptr.size != self.source_indptr.size:
            return False
        if not np.array_equal(np.asarray(free_dofs, dtype=np.int64).ravel(), self.free_dofs):
            return False
        if not np.array_equal(np.asarray(fixed_dofs, dtype=np.int64).ravel(), self.fixed_dofs):
            return False
        if validate_structure:
            return bool(np.array_equal(matrix.indptr, self.source_indptr) and np.array_equal(matrix.indices, self.source_indices))
        return True

    def extract_free_free(self, matrix: csr_matrix) -> csr_matrix:
        return self.free_free.extract(matrix)

    def fixed_correction(self, matrix: csr_matrix, fixed_values: np.ndarray) -> np.ndarray:
        if self.fixed_dofs.size == 0:
            return np.zeros(self.free_dofs.size, dtype=float)
        return self.free_fixed.matvec(matrix, fixed_values)

    def reduced_rhs(self, matrix: csr_matrix, rhs: np.ndarray, fixed_values: np.ndarray | None = None) -> np.ndarray:
        reduced = np.asarray(rhs, dtype=float).ravel()[self.free_dofs].copy()
        if fixed_values is not None and self.fixed_dofs.size:
            values = np.asarray(fixed_values, dtype=float).ravel()
            if values.size == self.source_shape[1]:
                fixed_vector = values[self.fixed_dofs]
            elif values.size == self.fixed_dofs.size:
                fixed_vector = values
            else:
                raise ValueError(f"fixed value vector size mismatch: expected {self.fixed_dofs.size} or {self.source_shape[1]}, got {values.size}")
            reduced -= self.fixed_correction(matrix, fixed_vector)
        return reduced

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "source": self.source,
            "matrix_shape": [int(self.source_shape[0]), int(self.source_shape[1])],
            "source_nnz": int(self.source_indices.size),
            "free_dofs": int(self.free_dofs.size),
            "fixed_dofs": int(self.fixed_dofs.size),
            "free_free": self.free_free.info(),
            "free_fixed": self.free_fixed.info(),
        }


def build_reduced_matrix_cache_from_pattern(
    shape: tuple[int, int],
    indptr: np.ndarray,
    indices: np.ndarray,
    free_dofs: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    source: str = "sparse_pattern",
) -> ReducedMatrixCache:
    return ReducedMatrixCache.from_pattern(shape, indptr, indices, free_dofs, fixed_dofs, source=source)


def build_reduced_matrix_cache_from_csr(
    matrix: csr_matrix,
    free_dofs: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    source: str = "csr_matrix",
) -> ReducedMatrixCache:
    return ReducedMatrixCache.from_csr(matrix, free_dofs, fixed_dofs, source=source)


__all__ = [
    "CSRSubmatrixCache",
    "ReducedMatrixCache",
    "build_reduced_matrix_cache_from_csr",
    "build_reduced_matrix_cache_from_pattern",
]
