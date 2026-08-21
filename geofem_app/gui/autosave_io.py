"""Autosave file IO helpers for GUI worker execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
import uuid


@dataclass(frozen=True)
class AutosaveResult:
    latest: Path
    stamped: Path
    byte_count: int
    sha256: str


def write_autosave_files(project_root: Path, base: str, text: str, *, timestamp: datetime | None = None) -> AutosaveResult:
    """Write latest and stamped autosave YAML files using atomic replacement."""

    out_dir = Path(project_root) / "autosave"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_base = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(base).strip()) or "model"
    payload = str(text)
    data = payload.encode("utf-8")
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    latest = out_dir / f"{safe_base}_autosave.yaml"
    stamped = out_dir / f"{safe_base}_autosave_{stamp}.yaml"
    _atomic_write_bytes(latest, data)
    _atomic_write_bytes(stamped, data)
    return AutosaveResult(latest=latest, stamped=stamped, byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


__all__ = ["AutosaveResult", "write_autosave_files"]
