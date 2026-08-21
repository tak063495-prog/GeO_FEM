"""Shared HTML report helpers for dependency-light report writers."""

from __future__ import annotations

import html as html_lib
from pathlib import Path
from typing import Any

import numpy as np


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, np.floating):
        return f"{float(value):.8g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def html_escape(value: Any) -> str:
    return html_lib.escape(str(value), quote=True)


def kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(f"<tr><th>{html_escape(key)}</th><td>{html_escape(format_value(value))}</td></tr>" for key, value in rows)
    return f"<table class=\"kv\"><tbody>{body}</tbody></table>"


def table(headers: list[str], rows: list[list[Any]], *, raw_columns: set[int] | None = None) -> str:
    raw_columns = raw_columns or set()
    head = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for col, cell in enumerate(row):
            text = format_value(cell)
            cells.append(f"<td>{text if col in raw_columns else html_escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def rel_link(path: Path | str | None, root: Path) -> str:
    if path is None:
        return "-"
    p = Path(path)
    try:
        label = p.relative_to(root)
    except ValueError:
        label = p
    href = html_lib.escape(str(label).replace("\\", "/"), quote=True)
    return f"<a href=\"{href}\">{html_escape(str(label))}</a>"


def report_css() -> str:
    return """
@page { size: A4; margin: 14mm; }
body { font-family: "Yu Gothic", "Meiryo", Arial, sans-serif; color: #111827; line-height: 1.5; margin: 0; }
h1, h2, h3, h4 { color: #0f172a; line-height: 1.25; }
h1 { font-size: 28px; margin: 0 0 10px; }
h2 { font-size: 20px; margin: 28px 0 10px; padding-bottom: 5px; border-bottom: 2px solid #1f2937; }
h3 { font-size: 15px; margin: 18px 0 8px; }
.cover { display: flex; justify-content: space-between; gap: 24px; padding: 22px 0 18px; border-bottom: 3px solid #111827; }
.eyebrow { letter-spacing: .08em; text-transform: uppercase; color: #475569; font-size: 11px; margin: 0 0 6px; }
.cover-summary { width: 220px; }
.title-block caption, .report-figure figcaption { text-align: left; font-weight: 700; margin: 6px 0; color: #334155; }
.title-block-section { break-inside: avoid; margin: 18px 0; }
.template-id { color: #475569; font-size: 12px; }
.toc { break-after: page; }
section { break-inside: avoid; }
.stage-block { break-inside: avoid; margin: 14px 0 20px; }
.report-figure { margin: 8px 0 16px; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 12px; }
th, td { border: 1px solid #cbd5e1; padding: 5px 7px; vertical-align: top; }
th { background: #f1f5f9; text-align: left; font-weight: 700; }
.kv th { width: 220px; }
pre { background: #f8fafc; border: 1px solid #cbd5e1; padding: 10px; overflow-wrap: anywhere; white-space: pre-wrap; font-size: 11px; }
.figure { width: 100%; max-height: 420px; border: 1px solid #cbd5e1; background: #ffffff; margin: 8px 0 12px; }
.active-element { fill: #f8fafc; stroke: #64748b; stroke-width: 1; }
.inactive-element { fill: #e5e7eb; stroke: #94a3b8; stroke-width: 1; stroke-dasharray: 4 3; }
.result-cell { stroke: #334155; stroke-width: .7; }
.deformed-cell { fill: none; stroke: #0f172a; stroke-width: 1.4; }
.bc-symbols rect { fill: #1d4ed8; stroke: #1e3a8a; stroke-width: 1; }
.bc-symbols text, .load-symbols text, .svg-title, .legend text { font-size: 12px; fill: #111827; }
.load-symbols line { stroke: #b91c1c; stroke-width: 2; }
.file-list { columns: 2; font-size: 12px; }
a { color: #1d4ed8; text-decoration: none; }
"""


__all__ = ["format_value", "html_escape", "kv_table", "rel_link", "report_css", "table"]
