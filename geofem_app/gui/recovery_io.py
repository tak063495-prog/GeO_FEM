"""Lightweight audit-log and recovery-candidate readers for the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .file_preview import DEFAULT_INPUT_PARSE_BYTES, preview_text_file, read_mapping_file_guarded


DEFAULT_TAIL_BYTES = 512 * 1024
DEFAULT_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RecoveryCandidateInfo:
    path: Path
    time: str
    bytes: int
    sha256: str
    changed: bool
    line_count: int
    preview: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "time": self.time,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "changed": self.changed,
            "line_count": self.line_count,
            "preview": self.preview,
        }


def read_jsonl_tail(path: str | Path, *, limit: int = 200, max_bytes: int = DEFAULT_TAIL_BYTES) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _tail_lines(source, max_lines=limit, max_bytes=max_bytes):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, Mapping):
            rows.append(dict(raw))
    return rows[-limit:]


def recovery_candidate_infos(paths: list[Path], *, current_text: str, preview_bytes: int = 2048) -> list[dict[str, Any]]:
    current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
    rows: list[RecoveryCandidateInfo] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            digest = sha256_recovery_candidate(path)
            line_count = count_file_lines(path)
            preview = preview_text_file(path, max_bytes=preview_bytes).text
        except OSError:
            continue
        rows.append(
            RecoveryCandidateInfo(
                path=path,
                time=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                bytes=int(stat.st_size),
                sha256=digest,
                changed=digest != current_hash,
                line_count=line_count,
                preview=preview,
            )
        )
    return [row.as_dict() for row in sorted(rows, key=lambda row: row.time, reverse=True)]


def compare_recovery_files(left_path: str | Path, right_path: str | Path, *, max_parse_bytes: int = DEFAULT_INPUT_PARSE_BYTES) -> dict[str, Any]:
    left = Path(left_path)
    right = Path(right_path)
    try:
        left_data, _left_preview = read_mapping_file_guarded(left, max_bytes=max_parse_bytes)
        right_data, _right_preview = read_mapping_file_guarded(right, max_bytes=max_parse_bytes)
    except Exception:
        left_preview = preview_text_file(left)
        right_preview = preview_text_file(right)
        left_text = left_preview.text
        right_text = right_preview.text
        result = {
            "ok": True,
            "left": str(left),
            "right": str(right),
            "mode": "preview",
            "same_preview": left_text == right_text,
            "left_bytes": left_preview.size_bytes,
            "right_bytes": right_preview.size_bytes,
            "left_preview_lines": len(left_text.splitlines()),
            "right_preview_lines": len(right_text.splitlines()),
            "truncated": left_preview.truncated or right_preview.truncated,
        }
        return result
    left_keys = set(map(str, left_data.keys()))
    right_keys = set(map(str, right_data.keys()))
    changed = sorted(key for key in left_keys & right_keys if left_data.get(key) != right_data.get(key))
    return {
        "ok": True,
        "left": str(left),
        "right": str(right),
        "mode": "mapping",
        "added": sorted(left_keys - right_keys),
        "removed": sorted(right_keys - left_keys),
        "changed": changed,
    }


def read_recovery_text(path: str | Path, *, max_parse_bytes: int = DEFAULT_INPUT_PARSE_BYTES) -> tuple[str, dict[str, Any] | None]:
    source = Path(path)
    size = source.stat().st_size
    if size > max_parse_bytes:
        preview = preview_text_file(source)
        return preview.text, None
    text = source.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return text, dict(data) if isinstance(data, Mapping) else None


def sha256_file(path: str | Path, *, chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_recovery_candidate(path: str | Path) -> str:
    source = Path(path)
    try:
        if source.stat().st_size <= DEFAULT_INPUT_PARSE_BYTES:
            return hashlib.sha256(source.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    except UnicodeDecodeError:
        pass
    return sha256_file(source)


def count_file_lines(path: str | Path, *, chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES) -> int:
    count = 0
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def _tail_lines(path: Path, *, max_lines: int, max_bytes: int) -> list[str]:
    size = path.stat().st_size
    if size <= 0:
        return []
    remaining = min(size, max(1, int(max_bytes)))
    chunks: list[bytes] = []
    with path.open("rb") as f:
        pos = size
        while remaining > 0 and len(b"".join(chunks).splitlines()) <= max_lines:
            read_size = min(8192, remaining, pos)
            pos -= read_size
            remaining -= read_size
            f.seek(pos)
            chunks.append(f.read(read_size))
            if pos <= 0:
                break
    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


__all__ = [
    "RecoveryCandidateInfo",
    "compare_recovery_files",
    "count_file_lines",
    "read_jsonl_tail",
    "read_recovery_text",
    "recovery_candidate_infos",
    "sha256_file",
    "sha256_recovery_candidate",
]
