from __future__ import annotations

from typing import Any

__all__ = ["DagQaClient"]


def __getattr__(name: str) -> Any:
    if name == "DagQaClient":
        from dagqa.client import DagQaClient  # noqa: PLC0415

        return DagQaClient
    raise AttributeError(name)
