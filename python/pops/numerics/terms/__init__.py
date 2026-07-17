"""pops.numerics.terms -- typed right-hand-side composition terms (Spec 5 sec.9).

A residual right-hand side is a SUM of typed terms: a conservative finite-volume
:class:`Flux` divergence, the model's explicit :class:`DefaultSource`, a cell-local
:class:`SourceTerm`, and a purely algebraic :class:`LocalTerm`. Named terms retain the exact
``OperatorHandle`` that declared them; they never carry a free operator-name string. These objects
only DESCRIBE the composition; they carry no numerics and compute nothing. The Program
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

    ``Flux()`` selects the model's default physical flux. ``Flux(operator)`` selects one exact
    named grid operator returned by the model declarer; several named flux terms may be summed in
    one RHS. The numerical interface flux (Rusanov/HLL/...) remains a separate discretization
    choice under :mod:`pops.numerics.riemann`.
    """

    category = "rhs_term"

    def __init__(self, operator: Any = None) -> None:
        if operator is not None:
            from pops.model import OperatorHandle

            if not isinstance(operator, OperatorHandle):
                raise TypeError(
                    "Flux(operator) requires the exact OperatorHandle returned by a grid/flux "
                    "declarer; got %r" % (operator,))
            if operator.kind != "grid_operator":
                raise TypeError(
                    "Flux(operator) requires a grid_operator handle, got kind %r"
                    % operator.kind)
        self._operator = operator

    @property
    def operator(self) -> Any:
        return self._operator

    def options(self) -> dict:
        return {
            "term": "flux",
            "operator": None if self._operator is None else self._operator.inspect(),
        }

    def capabilities(self) -> Any:
        return CapabilitySet({"conservative": True, "divergence": True})


class DefaultSource(Descriptor):
    """The default/composite source of the model instantiated by the RHS state.

    This is a typed generic selector, not an ownerless operator name: the block-qualified state
    selects the exact model owner, and lowering asks that model for its declared default source. It
    also represents an explicitly empty default source, so ready-made schemes do not need a string
    alias merely to keep their historical ``flux + default source`` semantics.
    """

    category = "rhs_term"

    def options(self) -> dict:
        return {"term": "default_source"}

    def capabilities(self) -> Any:
        return CapabilitySet({"local": True, "model_default": True})


def _require_source_operator(operator: Any, where: str) -> Any:
    from pops.model import OperatorHandle

    if not isinstance(operator, OperatorHandle):
        raise TypeError(
            "%s requires the typed OperatorHandle returned by m.source_term(...), got %r"
            % (where, operator))
    return operator


class SourceTerm(Descriptor):
    """A cell-local source term added to the residual (``+ S(u)``).

    ``operator`` is the exact handle returned by ``m.source_term(...)``. The wrapper records the
    scientific term category while preserving the selector's owner/kind/signature identity.
    """

    category = "rhs_term"

    def __init__(self, operator: Any) -> None:
        self._operator = _require_source_operator(operator, "SourceTerm")

    @property
    def name(self) -> str:
        return self._operator.name

    @property
    def operator(self) -> Any:
        return self._operator

    def options(self) -> dict:
        return {"term": "source", "operator": self._operator.inspect()}

    def capabilities(self) -> Any:
        return CapabilitySet({"local": True})


class LocalTerm(Descriptor):
    """A purely algebraic, cell-local term (no derivatives, no flux).

    Used for reaction / relaxation contributions that depend only on the local state. ``operator``
    is a typed source ``OperatorHandle``; the local/source distinction is descriptive and cannot
    manufacture ownership for an operator name.
    """

    category = "rhs_term"

    def __init__(self, operator: Any) -> None:
        self._operator = _require_source_operator(operator, "LocalTerm")

    @property
    def name(self) -> str:
        return self._operator.name

    @property
    def operator(self) -> Any:
        return self._operator

    def options(self) -> dict:
        return {"term": "local", "operator": self._operator.inspect()}

    def capabilities(self) -> Any:
        return CapabilitySet({"local": True, "algebraic": True})


__all__ = ["DefaultSource", "Flux", "SourceTerm", "LocalTerm"]
