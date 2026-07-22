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
from .preview import DomainPreview, PreviewDomainProvider, preview_domain, preview_geometry

__all__ = [
    "BoundaryPair", "BoundarySide", "DomainBoundary", "DomainTag", "Rectangle",
    "RectangleBoundaries", "RectangleBoundaryNames", "RectangleFrame", "DomainPreview",
    "PreviewDomainProvider", "preview_domain", "preview_geometry",
]
