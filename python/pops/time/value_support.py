"""Small authoring helpers shared by temporal symbolic values."""
from __future__ import annotations

from typing import Any


class _ProgramValueBase:
    """Private nominal marker shared by value validators without reverse imports."""

    __slots__ = ()


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
    from pathlib import Path
    import traceback

    time_root = Path(__file__).resolve().parent
    for frame in reversed(traceback.extract_stack()[:-1]):
        # Internal builders live in nested packages such as ``time._program``.  Comparing only the
        # immediate dirname leaks the first nested builder frame instead of the user authoring site.
        if not Path(frame.filename).resolve().is_relative_to(time_root):
            return "%s:%d" % (frame.filename, frame.lineno)
    return None


__all__ = ["_ProgramValueBase", "authoring_source_location", "resolve_temporal_handle"]
