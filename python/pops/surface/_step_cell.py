"""``dt=`` of ``pops.Advance`` written as a function of ``cell`` (spec 0.3, 2.6.2, 2.6.4; N8
« dt= as a function of cell », design note ``n8/cell_step_rule/design_note.md``, D1, D3).

``pops.Advance(method, dt=f)`` takes a plain function of the cell (SU5). PoPS runs it once, when
the case is traced, with the cell of a step rule, and the expression it returns is the bound of
each step: after each accepted step the next step is its smallest value over the cells, the first
step its smallest value at the initial state (2.6.2, PF8). The cell of a step rule reads

* ``cell.value(u)``, the value of an unknown of the model at the cell;
* ``cell.width(axis)``, the width of the cell along an axis of the support;
* ``cell.speed(F, axis)``, the spectral radius of the transport flux F at the cell value along the
  axis (R9), |dF_a/du| for a scalar flux. Spec 2.6.4 reads the speeds that the faces of the scheme
  of F record, and their spectral radius at the cell value where they record none; before a step
  no face has been evaluated, so this lot reads the radius at the cell value, and the faces that
  record a larger speed during a step refuse their attempt with the frequency that one retry
  takes (ROM-108, route B; pilot decision D3 (a), 12/11).

Every other read of 2.6.4 is refused with ``NotYetElaborated``, which names its lot.
"""
from __future__ import annotations

from typing import Any

from ._expr import Expr, is_number, text, walk
from ._geometry import Coordinate
from ._model import DiffusiveFlux, Flux, Quantity
from ._refusal import NotYetElaborated
from ._scheme import SchemeLeaf, substitute

_LOT = "a later lot of N8 « dt= as a function of cell »"
_LATER = {
    "shift": _LOT, "displacement": _LOT, "gradient": _LOT, "center": _LOT, "lower": _LOT,
    "upper": _LOT, "measure": _LOT, "load": "a later lot of N8 with the fields of a solve",
    "lower_face": _LOT, "upper_face": _LOT,
    "parent": "a later lot of N8 with the hierarchies of N12",
    "level": "a later lot of N8 with the hierarchies of N12",
}


def value_key(quantity: Quantity) -> tuple:
    """The key of ``cell.value(u)`` among the inputs of the elaborator."""
    return ("cell_value", quantity)


def width_key(index: int) -> tuple:
    """The key of ``cell.width(axis)`` among the inputs of the elaborator."""
    return ("cell_width", index)


class StepCell:
    """``cell`` in a step rule: the cell whose bound is evaluated, read by ``value``, ``width`` and
    ``speed`` (2.6.4)."""

    def __init__(self, support: Any, unknowns: tuple) -> None:
        self._support = support
        self._unknowns = unknowns

    def _axis(self, axis: Any, word: str) -> int:
        if not isinstance(axis, Coordinate) or axis.support is not self._support:
            raise TypeError("cell.%s takes an axis of the support, such as Ω.x, got %r"
                            % (word, axis))
        return axis.index

    def value(self, quantity: Any) -> SchemeLeaf:
        if not isinstance(quantity, Quantity):
            raise TypeError("cell.value(u) takes an unknown of the model, got %r" % (quantity,))
        if not any(quantity is unknown for unknown in self._unknowns):
            raise TypeError("cell.value(%s): %s is not an unknown that the advance steps"
                            % (quantity, quantity))
        return SchemeLeaf(value_key(quantity), quantity, "cell.value(%s)" % quantity)

    def width(self, axis: Any) -> SchemeLeaf:
        index = self._axis(axis, "width")
        return SchemeLeaf(width_key(index), None, "cell.width(%s)" % axis.name)

    def speed(self, flux: Any, axis: Any) -> Any:
        """The spectral radius of ``flux`` at the cell value along ``axis`` (R9; D3 (a))."""
        if isinstance(flux, DiffusiveFlux) or not isinstance(flux, Flux):
            raise TypeError("cell.speed(F, axis) takes a transport flux of the model, "
                            "model.flux(of=u, value=...), got %r" % (flux,))
        if not any(flux.quantity is unknown for unknown in self._unknowns):
            raise TypeError("cell.speed(%s, %s): a flux of %s, which is not an unknown that the "
                            "advance steps (a flux of another model?)"
                            % (text(flux), getattr(axis, "name", axis), flux.quantity))
        index = self._axis(axis, "speed")
        slopes = flux.slopes()
        if slopes is None:
            from ._refusal import AmbiguousElaboration

            raise AmbiguousElaboration(
                "cell.speed(%s, %s) under %s, whose wave speed dF/d%s is not derived in this "
                "lot (R9)" % (text(flux), axis.name, text(flux.value), flux.quantity),
                write="dt= a function of cell that does not read the speed of this flux")
        return abs(substitute(slopes[index], {flux.quantity: self.value(flux.quantity)}))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _LATER:
            raise NotYetElaborated("cell.%s in dt= of pops.Advance" % name, _LATER[name])
        raise AttributeError("cell in dt= of pops.Advance reads value(u), width(axis) and "
                             "speed(F, axis), not %s" % name)

    def __repr__(self) -> str:
        return "cell"


def trace(rule: Any, support: Any, unknowns: tuple) -> Any:
    """Run the function of ``cell`` once and return the bound it writes (SU5, E4)."""
    result = rule(StepCell(support, unknowns))
    if not (is_number(result) or isinstance(result, Expr)):
        raise TypeError("dt=%s: a function of cell returns the bound of the step at the cell, "
                        "an expression, got %r" % (getattr(rule, "__name__", rule), result))
    if not any(isinstance(node, SchemeLeaf) for node in walk(result)):
        raise TypeError("dt=%s returns %s, which reads no cell: write dt=%s as a number"
                        % (getattr(rule, "__name__", rule), text(result), text(result)))
    return result
