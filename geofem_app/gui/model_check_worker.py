"""Detached model-check execution helpers for GUI worker threads."""

from __future__ import annotations

import copy
from typing import Any


class DetachedGuiContext:
    """Expose selected MainWindow methods against a copied config only.

    Model checking has accumulated useful pure helpers on MainWindow.  This
    adapter lets a worker reuse them without touching the live QWidget object.
    """

    def __init__(self, window_cls: type[Any], cfg: dict[str, Any]) -> None:
        self._window_cls = window_cls
        self.cfg = copy.deepcopy(cfg)

    def __getattr__(self, name: str) -> Any:
        for cls in self._window_cls.__mro__:
            raw = cls.__dict__.get(name)
            if raw is not None:
                break
        else:
            raise AttributeError(name)
        if isinstance(raw, (staticmethod, classmethod)):
            return raw.__get__(self, type(self))
        if callable(raw):
            return raw.__get__(self, type(self))
        return raw


def collect_model_issues_snapshot(window_cls: type[Any], cfg: dict[str, Any]) -> list[Any]:
    """Run MainWindow.collect_model_issues on a detached config snapshot."""

    context = DetachedGuiContext(window_cls, cfg)
    return list(window_cls.collect_model_issues(context))


__all__ = ["DetachedGuiContext", "collect_model_issues_snapshot"]
