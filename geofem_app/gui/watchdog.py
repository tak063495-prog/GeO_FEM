"""GUI event-loop watchdog persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
import uuid


@dataclass(frozen=True)
class GuiFreezeWatchdogRecord:
    timestamp: str
    operation: str
    job: str
    elapsed_ms: float
    delay_ms: float
    target_file: str
    line_count: int
    mesh_nodes: int | None
    mesh_elements: int | None
    last_ui_event: str
    active_jobs: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "job": self.job,
            "elapsed_ms": self.elapsed_ms,
            "delay_ms": self.delay_ms,
            "target_file": self.target_file,
            "line_count": self.line_count,
            "mesh_nodes": self.mesh_nodes,
            "mesh_elements": self.mesh_elements,
            "last_ui_event": self.last_ui_event,
            "active_jobs": self.active_jobs,
        }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_watchdog_record(path: Path, record: Mapping[str, Any], *, max_records: int = 100) -> list[dict[str, Any]]:
    """Append a freeze watchdog record to a compact JSON ring buffer."""

    target = Path(path)
    rows: list[dict[str, Any]] = []
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = [dict(row) for row in loaded if isinstance(row, Mapping)]
        except Exception:
            rows = []
    rows.append(dict(record))
    if max_records > 0:
        rows = rows[-max_records:]
    _atomic_write_text(target, json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


__all__ = ["GuiFreezeWatchdogRecord", "append_watchdog_record", "utc_timestamp"]
