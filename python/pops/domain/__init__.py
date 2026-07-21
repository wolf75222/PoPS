"""Pure geometric domain descriptors."""

from .rectangle import (
    BoundaryPair,
    BoundarySide,
    DomainBoundary,
    DomainTag,
    Rectangle,
    RectangleBoundaries,
    RectangleBoundaryNames,
    RectangleFrame,
)
from .preview import DomainPreview

__all__ = [
    "BoundaryPair", "BoundarySide", "DomainBoundary", "DomainTag", "Rectangle",
    "RectangleBoundaries", "RectangleBoundaryNames", "RectangleFrame", "DomainPreview",
]
