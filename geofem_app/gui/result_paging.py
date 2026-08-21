"""Paging helpers for large GUI result tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_RESULT_TABLE_PAGE_SIZE = 2000


@dataclass(frozen=True)
class ResultTablePage:
    rows: list[dict[str, str]]
    headers: list[str]
    page_index: int
    page_size: int
    total_rows: int
    page_count: int
    start_row: int
    end_row: int

    @property
    def label(self) -> str:
        if self.total_rows == 0:
            return "0件"
        return f"{self.start_row + 1}-{self.end_row} / {self.total_rows}件"


@dataclass(frozen=True)
class CsvTableSummary:
    path: str
    headers: list[str]
    row_count: int
    numeric_fields: list[str]
    minimums: dict[str, float]
    maximums: dict[str, float]


def result_table_page(rows: Sequence[Mapping[str, str]], *, page_index: int = 0, page_size: int = DEFAULT_RESULT_TABLE_PAGE_SIZE) -> ResultTablePage:
    total = len(rows)
    headers = list(rows[0]) if rows else []
    safe_size = max(1, int(page_size))
    page_count = max(1, (total + safe_size - 1) // safe_size)
    safe_index = min(max(0, int(page_index)), page_count - 1)
    start = min(total, safe_index * safe_size)
    end = min(total, start + safe_size)
    page_rows = [dict(row) for row in rows[start:end]]
    return ResultTablePage(
        rows=page_rows,
        headers=headers,
        page_index=safe_index,
        page_size=safe_size,
        total_rows=total,
        page_count=page_count,
        start_row=start,
        end_row=end,
    )


def summarize_csv_table(path: str | Path) -> CsvTableSummary:
    """Scan a CSV once for headers, row count, and numeric min/max."""

    source = Path(path)
    row_count = 0
    minimums: dict[str, float] = {}
    maximums: dict[str, float] = {}
    numeric_candidates: set[str] | None = None
    headers: list[str] = []
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        numeric_candidates = set(headers)
        for row in reader:
            row_count += 1
            for header in headers:
                raw = row.get(header, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    if numeric_candidates is not None:
                        numeric_candidates.discard(header)
                    continue
                if numeric_candidates is not None and header in numeric_candidates:
                    minimums[header] = min(minimums.get(header, value), value)
                    maximums[header] = max(maximums.get(header, value), value)
    numeric_fields = [header for header in headers if numeric_candidates is not None and header in numeric_candidates]
    return CsvTableSummary(
        path=str(source),
        headers=headers,
        row_count=row_count,
        numeric_fields=numeric_fields,
        minimums={key: minimums[key] for key in numeric_fields if key in minimums},
        maximums={key: maximums[key] for key in numeric_fields if key in maximums},
    )


def read_csv_table_page(path: str | Path, *, page_index: int = 0, page_size: int = DEFAULT_RESULT_TABLE_PAGE_SIZE, total_rows: int | None = None) -> ResultTablePage:
    """Read only one display page from a CSV file."""

    source = Path(path)
    safe_size = max(1, int(page_size))
    known_total = _csv_row_count(source) if total_rows is None else max(0, int(total_rows))
    page_count = max(1, (known_total + safe_size - 1) // safe_size)
    safe_index = min(max(0, int(page_index)), page_count - 1)
    start = min(known_total, safe_index * safe_size)
    end = min(known_total, start + safe_size)
    rows: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        for index, row in enumerate(reader):
            if index < start:
                continue
            if index >= end:
                break
            rows.append(dict(row))
    return ResultTablePage(
        rows=rows,
        headers=headers,
        page_index=safe_index,
        page_size=safe_size,
        total_rows=known_total,
        page_count=page_count,
        start_row=start,
        end_row=end,
    )


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _row in reader)


__all__ = [
    "DEFAULT_RESULT_TABLE_PAGE_SIZE",
    "CsvTableSummary",
    "ResultTablePage",
    "read_csv_table_page",
    "result_table_page",
    "summarize_csv_table",
]
