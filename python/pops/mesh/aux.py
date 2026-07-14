"""pops.mesh.aux -- the per-field aux halo / ghost boundary policy (Spec 5 sec.5.9).

``AuxHalo`` declares a per-field aux ghost boundary policy, passed to
``set_aux_field(..., halo=pops.mesh.AuxHalo(...))``. It has one canonical owner in
``pops.mesh`` and no root or field-side alias.
"""
from __future__ import annotations

from typing import Any

from ._descriptor import MeshDescriptor


class AuxHalo(MeshDescriptor):
    """Per-field aux halo/ghost boundary policy applied AFTER the shared fill.

    A model-NAMED aux field normally inherits the SHARED aux ghost behaviour derived from
    the potential phi (periodic preserved, otherwise zero-gradient). ``AuxHalo`` lets that
    ONE field declare its own boundary policy for its component:

    - ``kind='foextrap'``: zero-gradient (ghost = mirror interior cell);
    - ``kind='dirichlet'``: fixed value (ghost = 2*value - interior), ``value`` imposed.

    Applied UNIFORMLY to the NON-PERIODIC faces; periodic faces keep their wrap. Works on
    System (Cartesian + polar) and the AMR coarse level. No halo (default) -> the shared
    aux BC, bit-identical.
    """

    category = "aux_halo"

    # Mirrors pops::BCType on the C++ side: Periodic=0, Foextrap=1, Dirichlet=2.
    _KINDS = {"foextrap": 1, "dirichlet": 2}

    def __init__(self, kind: Any, value: Any = 0.0) -> None:
        if kind not in self._KINDS:
            raise ValueError("AuxHalo: kind must be 'foextrap' or 'dirichlet' (got %r)" % (kind,))
        self.kind = kind
        self.bc_type = self._KINDS[kind]
        self.value = float(value)

    def options(self) -> dict:
        return {"kind": self.kind, "value": self.value}

    def __repr__(self) -> str:
        return "AuxHalo(%r, value=%g)" % (self.kind, self.value)

    # Keep the original concise form for str() too (do not inherit the verbose base).
    __str__ = __repr__
