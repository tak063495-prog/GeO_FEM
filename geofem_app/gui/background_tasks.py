"""Optional Qt background task helpers for GUI freeze prevention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class BackgroundTaskResult:
    job_id: str
    ok: bool
    value: Any = None
    error: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise RuntimeError("background task cancelled")


def run_callable_with_token(job_id: str, fn: Callable[[CancellationToken], Any], token: CancellationToken | None = None) -> BackgroundTaskResult:
    """Run a worker callable and normalize success/failure payloads."""

    active_token = token or CancellationToken()
    try:
        active_token.raise_if_cancelled()
        value = fn(active_token)
        active_token.raise_if_cancelled()
        return BackgroundTaskResult(job_id=job_id, ok=True, value=value)
    except Exception as exc:
        return BackgroundTaskResult(job_id=job_id, ok=False, error=str(exc))


__all__ = ["BackgroundTaskResult", "CancellationToken", "run_callable_with_token"]
