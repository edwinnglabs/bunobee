"""The bunobee model classes. ``HierarchicalCount`` is imported from here on first use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bunobee.models.count import HierarchicalCount

__all__ = ["HierarchicalCount"]


def __getattr__(name: str) -> Any:
    # Lazy, so that importing one model module does not import the others.
    if name == "HierarchicalCount":
        from bunobee.models.count import HierarchicalCount

        return HierarchicalCount
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
