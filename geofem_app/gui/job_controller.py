"""Small GUI job-state controller used to keep long operations cancellable."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
import uuid


@dataclass
class GuiJob:
    id: str
    kind: str
    target: str = ""
    status: str = "running"
    progress: float | None = None
    cancel_requested: bool = False
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "status": self.status,
            "progress": self.progress,
            "cancel_requested": self.cancel_requested,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


class GuiJobController:
    """Track running GUI-side and process-backed jobs without Qt dependencies."""

    def __init__(self) -> None:
        self._jobs: dict[str, GuiJob] = {}

    def start_job(self, kind: str, *, target: str = "", metadata: Mapping[str, Any] | None = None) -> str:
        job_id = f"{kind}:{uuid.uuid4().hex[:12]}"
        self._jobs[job_id] = GuiJob(id=job_id, kind=str(kind), target=str(target), metadata=dict(metadata or {}))
        return job_id

    def request_cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status not in {"running", "queued"}:
            return False
        job.cancel_requested = True
        job.message = "cancel requested"
        return True

    def update_job(self, job_id: str, *, progress: float | None = None, message: str | None = None) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if progress is not None:
            job.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            job.message = str(message)
        return True

    def finish_job(self, job_id: str, *, message: str = "") -> bool:
        return self.complete_job(job_id, status="finished", message=message)

    def fail_job(self, job_id: str, *, message: str = "") -> bool:
        return self.complete_job(job_id, status="failed", message=message)

    def cancel_job(self, job_id: str, *, message: str = "cancelled") -> bool:
        return self.complete_job(job_id, status="cancelled", message=message)

    def complete_job(self, job_id: str, *, status: str, message: str = "") -> bool:
        if status not in {"finished", "failed", "cancelled"}:
            raise ValueError(f"unsupported GUI job status: {status}")
        return self._close_job(job_id, status, message)

    def active_jobs(self) -> list[dict[str, Any]]:
        return [job.as_dict() for job in self._jobs.values() if job.status in {"running", "queued"}]

    def snapshot(self) -> list[dict[str, Any]]:
        return [job.as_dict() for job in self._jobs.values()]

    def job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.as_dict() if job is not None else None

    def failure_manifest(self, job_id: str, *, message: str = "", context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "ok": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": str(message),
            "job": self.job_snapshot(job_id) or {"id": str(job_id), "status": "unknown"},
            "context": dict(context or {}),
        }

    def is_cancel_requested(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return bool(job and job.cancel_requested)

    def _close_job(self, job_id: str, status: str, message: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.status = status
        job.message = str(message)
        job.finished_at = datetime.now(timezone.utc).isoformat()
        return True


__all__ = ["GuiJob", "GuiJobController"]
