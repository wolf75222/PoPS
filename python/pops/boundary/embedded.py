"""Typed flux policies for embedded boundaries.

Geometry and transport metrics do not determine the physical boundary flux.  This module keeps
that third authority explicit and extensible; the current native provider implements exactly zero
normal numerical flux and never pretends to be a reflective Euler wall.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.mesh._descriptor import MeshDescriptor


_NATIVE_EMBEDDED_BOUNDARY_FLUXES = frozenset({"zero_flux"})


class EmbeddedBoundaryFlux(MeshDescriptor):
    """Extension interface for one prepared embedded-boundary numerical flux."""

    category = "embedded_boundary_flux"
    provider_token = ""

    def options(self) -> dict[str, Any]:
        return {"provider": self.provider_token}

    def lower(self, context: Any = None) -> str:
        del context
        return self.provider_token


class ZeroFlux(EmbeddedBoundaryFlux):
    """Close every cut or masked face with an exactly zero numerical flux."""

    provider_token = "zero_flux"

    def requirements(self) -> RequirementSet:
        return RequirementSet({"embedded_boundary_support": True})

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({"normal_numerical_flux": "zero"})


def lower_embedded_boundary_flux(value: Any) -> str:
    """Authenticate an extension policy and return its private native provider token."""

    if not isinstance(value, EmbeddedBoundaryFlux):
        raise TypeError(
            "embedded boundary flux must be a pops.boundary.EmbeddedBoundaryFlux descriptor, "
            "got %s" % type(value).__name__
        )
    token = value.lower()
    if not isinstance(token, str) or token not in _NATIVE_EMBEDDED_BOUNDARY_FLUXES:
        raise ValueError(
            "%s.lower() returned unsupported embedded-boundary flux provider %r"
            % (type(value).__name__, token)
        )
    return token


__all__ = ["EmbeddedBoundaryFlux", "ZeroFlux"]
