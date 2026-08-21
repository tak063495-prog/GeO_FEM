"""Maintainability audit for large files and mixed responsibility boundaries."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


RESPONSIBILITY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gui", ("QWidget", "QMainWindow", "QMessageBox", "QTableWidget", "QGraphics")),
    ("solver", ("solve_", "assemble_", "stiffness", "newton", "riks", "newmark")),
    ("mesh", ("mesh", "polygon", "quad", "triangle", "boundary", "paving")),
    ("io", ("csv", "json", "html", "vtk", "write_", "read_")),
    ("report", ("report", "html", "pdf", "manifest")),
    ("verification", ("audit", "compare", "reference", "tolerance", "compatibility")),
    ("materials", ("material", "constitutive", "plastic", "liquefaction")),
)


def audit_maintainability(
    root: str | Path,
    *,
    include_dirs: Iterable[str] = ("geofem_app", "tools"),
    large_line_threshold: int = 1000,
) -> dict[str, Any]:
    """Scan Python source files and rank likely split candidates."""

    base = Path(root)
    files = []
    for rel_dir in include_dirs:
        directory = base / rel_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            hints = [] if path.name == "maintainability_audit.py" else _responsibility_hints(text)
            public_defs = sum(1 for line in lines if line.startswith("def ") or line.startswith("class "))
            private_defs = sum(1 for line in lines if line.startswith("def _") or line.startswith("class _"))
            score = _candidate_score(len(lines), len(hints), public_defs, private_defs, large_line_threshold)
            files.append(
                {
                    "path": str(path.relative_to(base)),
                    "lines": len(lines),
                    "public_defs": public_defs,
                    "private_defs": private_defs,
                    "responsibility_hints": hints,
                    "split_candidate_score": score,
                    "recommendation": _recommendation(path.name, len(lines), hints, score, large_line_threshold),
                }
            )
    files.sort(key=lambda row: (row["split_candidate_score"], row["lines"]), reverse=True)
    candidates = [row for row in files if row["lines"] >= large_line_threshold or row["split_candidate_score"] >= 4]
    return {
        "schema": "geofem.maintainability_audit.v1",
        "root": str(base),
        "large_line_threshold": large_line_threshold,
        "file_count": len(files),
        "candidate_count": len(candidates),
        "largest_files": sorted(files, key=lambda row: row["lines"], reverse=True)[:20],
        "split_candidates": candidates[:30],
    }


def write_maintainability_audit(summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "maintainability_audit.json"
    csv_path = out / "maintainability_audit.csv"
    html_path = out / "maintainability_audit.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows = list(summary.get("split_candidates", []))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["path", "lines", "public_defs", "private_defs", "responsibility_hints", "split_candidate_score", "recommendation"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if isinstance(row, Mapping):
                writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})
    html_path.write_text(_audit_html(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def run_maintainability_audit(root: str | Path, output_dir: str | Path, *, large_line_threshold: int = 1000) -> dict[str, Any]:
    summary = audit_maintainability(root, large_line_threshold=large_line_threshold)
    paths = write_maintainability_audit(summary, output_dir)
    summary["paths"] = paths
    return summary


def _responsibility_hints(text: str) -> list[str]:
    lower = text.lower()
    hints = []
    for name, tokens in RESPONSIBILITY_HINTS:
        hits = sum(1 for token in tokens if token.lower() in lower)
        if hits >= 2:
            hints.append(name)
    return hints


def _candidate_score(lines: int, hint_count: int, public_defs: int, private_defs: int, threshold: int) -> int:
    score = 0
    if lines >= threshold:
        score += 3
    if lines >= threshold * 3:
        score += 2
    if hint_count >= 2:
        score += hint_count
    if public_defs + private_defs >= 40:
        score += 2
    if public_defs >= 20:
        score += 1
    return score


def _recommendation(filename: str, lines: int, hints: list[str], score: int, threshold: int) -> str:
    if lines < threshold and score < 4:
        return "監視対象外"
    if "gui" in hints:
        return "GUI構築、イベント処理、結果表示、モデル変換をタブ/機能別モジュールへ分割してください。"
    if "solver" in hints and "mesh" in hints:
        return "解析制御、行列組立、境界/MPC、水理/動的補助を個別モジュールへ分割してください。"
    if "verification" in hints or filename.startswith("geofeas_"):
        return "監査、比較、外部形式正規化、HTML出力を別モジュールへ分割してください。"
    if "mesh" in hints:
        return "メッシュ生成アルゴリズム、幾何ユーティリティ、品質改善を別モジュールへ分割してください。"
    if "io" in hints or "report" in hints:
        return "CSV/VTK出力、Post図、帳票生成を別モジュールへ分割してください。"
    return "公開APIを維持したまま、まとまった責務単位でファサード化してください。"


def _audit_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for row in summary.get("split_candidates", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('path', '')))}</td>"
            f"<td>{html.escape(str(row.get('lines', '')))}</td>"
            f"<td>{html.escape(', '.join(row.get('responsibility_hints', [])))}</td>"
            f"<td>{html.escape(str(row.get('split_candidate_score', '')))}</td>"
            f"<td>{html.escape(str(row.get('recommendation', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM maintainability audit</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#f3f3f3}}</style></head>
<body><h1>保守性監査</h1>
<p>files={int(summary.get('file_count', 0) or 0)}, candidates={int(summary.get('candidate_count', 0) or 0)}, threshold={int(summary.get('large_line_threshold', 0) or 0)}</p>
<table><thead><tr><th>path</th><th>lines</th><th>hints</th><th>score</th><th>recommendation</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return value


__all__ = ["audit_maintainability", "run_maintainability_audit", "write_maintainability_audit"]
