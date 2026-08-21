"""Qt worker adapters used by the GUI to keep long tasks off the UI thread."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal


class QtCallableSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)


class QtCallableRunner(QRunnable):
    """Run a no-argument callable in a QThreadPool and emit normalized signals."""

    def __init__(self, job_id: str, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.job_id = str(job_id)
        self.fn = fn
        self.signals = QtCallableSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self.job_id, self.fn())
        except Exception as exc:
            self.signals.failed.emit(self.job_id, str(exc))


__all__ = ["QtCallableRunner", "QtCallableSignals"]
