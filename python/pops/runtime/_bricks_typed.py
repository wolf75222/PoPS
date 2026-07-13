"""Typed native-brick constructors (Spec 5 sec.14.2.5).

The native bricks are named by typed constructors instead of magic ``bc=`` strings. This module
homes the elliptic boundary selectors that do not belong to a state/source/time module:

* the typed native ELLIPTIC boundary bricks ``Dirichlet`` / ``Neumann`` / ``Periodic``.

``System.set_poisson`` and ``AmrSystem.set_poisson`` accept these objects exclusively. Their
``.bc`` value is consumed only by the private Python/native lowering seam.

The boundary bricks are the NATIVE elliptic-solver boundary (the homogeneous BC of the system
Poisson solve), and are DISTINCT from two other typed boundary surfaces already in the package :

* ``pops.fields.bcs`` (Spec 5 sec.5.5) -- inert field-VALUE conditions attached per face of a
  ``pops.fields`` elliptic problem (Dirichlet value, Neumann flux, first-order extrapolation) ;
* ``pops.mesh.boundaries`` (Spec 5 sec.5.9) -- domain-TOPOLOGY descriptors (periodic vs physical
  faces of the mesh).

All objects here are inert (no ``_pops`` / numpy / runtime / codegen compute) and record/lower their
boundary token. ``pops.runtime.bricks`` re-exports them.
"""
from __future__ import annotations

from typing import Any



# The native elliptic boundary tokens accepted by System::set_poisson (C++ binding). "auto" lets the
# solver pick periodic vs wall from the mesh ; the three typed ctors below pin the explicit choices.
_BC_PERIODIC = "periodic"
_BC_DIRICHLET = "dirichlet"
_BC_NEUMANN = "neumann"


class _Boundary:
    """Base of the native elliptic boundary bricks and their private lowering token."""

    bc = ""

    def lower(self) -> Any:
        """The native ``bc=`` token this boundary lowers to (e.g. ``"dirichlet"``)."""
        return self.bc

    def __eq__(self, other: Any) -> Any:
        return isinstance(other, _Boundary) and other.bc == self.bc

    def __hash__(self) -> Any:
        return hash((type(self).__name__, self.bc))

    def __repr__(self) -> Any:
        return "%s(bc=%r)" % (type(self).__name__, self.bc)


class Periodic(_Boundary):
    """Native periodic elliptic boundary (the Poisson solve wraps).

    The mesh-topology counterpart is
    ``pops.mesh.boundaries.Periodic`` ; this brick is the native ELLIPTIC-solver boundary token.
    """

    bc = _BC_PERIODIC


class Dirichlet(_Boundary):
    """Native Dirichlet elliptic boundary (homogeneous ``phi=0`` wall).

    The native Poisson solve imposes a
    homogeneous Dirichlet value on the physical faces (a conducting wall is selected with a typed
    ``pops.mesh.geometry.Disc``). The field-VALUE per-face Dirichlet of a
    ``pops.fields`` problem (with a non-zero value) lives in ``pops.fields.bcs.Dirichlet``.
    """

    bc = _BC_DIRICHLET


class Neumann(_Boundary):
    """Native Neumann elliptic boundary (homogeneous zero-flux wall).

    The native Poisson solve imposes a
    homogeneous Neumann (zero normal derivative) condition on the physical faces.
    """

    bc = _BC_NEUMANN


__all__ = ["Periodic", "Dirichlet", "Neumann"]
