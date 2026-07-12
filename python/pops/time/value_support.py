"""Small authoring helpers shared by temporal symbolic values."""
from __future__ import annotations

from typing import Any


def resolve_temporal_handle(value: Any) -> Any:
    """Resolve a readable temporal handle through its Program-owned table."""
    from pops.time.handles import HistoryHandle, StageHandle

    return (
        value._as_value()
        if isinstance(value, (StageHandle, HistoryHandle))
        else value
    )


def authoring_source_location() -> Any:
    """Return the first call site outside ``pops.time`` for optional debug provenance."""
    import os
    import traceback

    time_dir = os.path.dirname(__file__)
    for frame in reversed(traceback.extract_stack()[:-1]):
        if os.path.dirname(frame.filename) != time_dir:
            return "%s:%d" % (frame.filename, frame.lineno)
    return None


__all__ = ["authoring_source_location", "resolve_temporal_handle"]
