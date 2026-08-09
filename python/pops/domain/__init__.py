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
from .cartesian import (
    CartesianBoundaries,
    CartesianBoundaryNames,
    CartesianDomain,
    CartesianDomainFrame,
)
from .preview import DomainPreview, PreviewDomainProvider, preview_domain, preview_geometry

__all__ = [
    "BoundaryPair", "BoundarySide", "DomainBoundary", "DomainTag", "Rectangle",
    "RectangleBoundaries", "RectangleBoundaryNames", "RectangleFrame", "DomainPreview",
    "PreviewDomainProvider", "preview_domain", "preview_geometry", "CartesianBoundaries",
    "CartesianBoundaryNames", "CartesianDomain", "CartesianDomainFrame",
]
