"""pops.numerics.terms -- typed right-hand-side composition terms (Spec 5 sec.9).

A residual right-hand side is a SUM of typed terms: a conservative finite-volume
:class:`Flux` divergence, a cell-local :class:`SourceTerm`, and a purely algebraic
:class:`LocalTerm`. These objects only DESCRIBE the composition (which kind of term,
with an optional name); they carry no numerics and compute nothing. The Program
``P.rhs(terms=[...])`` lowering that consumes them is a separate change.

Every term is an inert :class:`pops.descriptors.Descriptor`; codegen / runtime turn the
declared composition into the discrete residual after validation.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet


class Flux(Descriptor):
    """A conservative finite-volume flux-divergence term ``-div F`` of the residual.

    Names the flux contribution to the right-hand side; the numerical flux brick itself
    (Rusanov/HLL/...) is selected separately under :mod:`pops.numerics.riemann`.
    """

    category = "rhs_term"

    def options(self) -> dict:
        return {"term": "flux"}

    def capabilities(self) -> Any:
        return CapabilitySet({"conservative": True, "divergence": True})


class SourceTerm(Descriptor):
    """A cell-local source term added to the residual (``+ S(u)``).

    Optionally :paramref:`name`-d so a model can refer to a specific source contribution.
    """

    category = "rhs_term"

    def __init__(self, name: Any = None) -> None:
        self._name = None if name is None else str(name)

    @property
    def name(self) -> str:
        return self._name if self._name is not None else type(self).__name__

    def options(self) -> dict:
        return {"term": "source", "name": self._name}

    def capabilities(self) -> Any:
        return CapabilitySet({"local": True})


class LocalTerm(Descriptor):
    """A purely algebraic, cell-local term (no derivatives, no flux).

    Optionally :paramref:`name`-d. Used for reaction / relaxation contributions that
    depend only on the local state.
    """

    category = "rhs_term"

    def __init__(self, name: Any = None) -> None:
        self._name = None if name is None else str(name)

    @property
    def name(self) -> str:
        return self._name if self._name is not None else type(self).__name__

    def options(self) -> dict:
        return {"term": "local", "name": self._name}

    def capabilities(self) -> Any:
        return CapabilitySet({"local": True, "algebraic": True})


__all__ = ["Flux", "SourceTerm", "LocalTerm"]
