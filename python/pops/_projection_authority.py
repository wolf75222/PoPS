"""Private declarations from the owners of closed Program projections."""
from __future__ import annotations

from typing import TypeVar


_ProjectionType = TypeVar("_ProjectionType", bound=type)
_DECLARED_VALUES: dict[int, type] = {}
_DECLARED_REFERENCES: dict[int, type] = {}


def _closed_projection_value(kind: _ProjectionType) -> _ProjectionType:
    """Declare that a library value projects only its witnessed data."""
    _DECLARED_VALUES[id(kind)] = kind
    return kind


def _closed_projection_reference(kind: _ProjectionType) -> _ProjectionType:
    """Declare a library Handle using the canonical Case/definition witness."""
    _DECLARED_REFERENCES[id(kind)] = kind
    return kind


def _declared_projection_types() -> tuple[dict[int, type], dict[int, type]]:
    return _DECLARED_VALUES.copy(), _DECLARED_REFERENCES.copy()
