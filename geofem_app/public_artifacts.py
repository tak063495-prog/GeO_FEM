"""Shared writers for public substitute artifacts.

GeoFEAS and VGFlow2D substitute modules both publish small JSON/CSV/HTML
packages. Keeping the serialization details here avoids each product adapter
drifting in encoding, newline, or escaping behavior.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .html_report_utils import html_escape, report_css, table


def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_dict_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    fields = [str(field) for field in fieldnames]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def html_table_document(
    *,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    lead: str = "",
    css: str | None = None,
) -> str:
    lead_html = f"<p>{html_escape(lead)}</p>" if lead else ""
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        f"<title>{html_escape(title)}</title>"
        f"<style>{css if css is not None else report_css()}</style></head><body>"
        f"<h1>{html_escape(title)}</h1>"
        f"{lead_html}"
        f"{table([str(header) for header in headers], [list(row) for row in rows])}"
        "</body></html>"
    )


def write_html_artifact(path: str | Path, document: str) -> None:
    Path(path).write_text(document, encoding="utf-8")


__all__ = [
    "html_table_document",
    "write_dict_rows_csv",
    "write_html_artifact",
    "write_json_artifact",
]
