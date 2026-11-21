"""The step bound of a cells rule: ``dt=`` of an Advance written as a function of ``cell`` (spec 0.3,
2.6.2, 2.6.4, PF8; N8 « dt= as a function of cell », design note ``n8/cell_step_rule``, D1, D2).

``StepBound(expression, fraction=φ)`` is the private carrier that the elaborator of the surface
builds from ``dt=f``: ``expression`` is the bound E that ``f`` returns at the cell of a step rule,
lowered to the cell reads of the evolved block state (``CellRead``), the widths of the cell
(``CellWidth``), runtime parameters and constants; ``fraction`` is the Courant budget φ that the
rule declares, in (0, 1], under which the face guards of its attempts run (C2). A library rule
declares it with the attribute ``step_fraction`` of the function of cell it returns (as
``pops.lib.time.cfl_limit`` does), a hand-written function declares none and takes 1, the budget
of a fixed step; ``origin`` says which (``"stated"`` or ``"default"``), and ``pops.inspect`` shows
both (review rev 3, P3).

``elaborate(..., proposal=StepBound(...))`` resolves it against the one state of the Advance
(``resolve_step_bound``): each step of the compiled run is the smallest value of E over the valid
cells at the accepted state, reduced over the ranks (PF8), generated into the Program
(``pops.codegen.program_emit_step_bound``). The identity of the resolved bound names E, the state
and the parameters it reads; the compiled control records it with φ (``pops.advance-control@6``).
The core does not know which formula a library names (CF2): it says ``fraction``, not ``cfl``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from pops._ir.expr import Const, Expr
from pops._ir.values import RuntimeParamRef
from pops._ir.visitors import _children, _key

_SCHEMA_VERSION = 1
_ORIGINS = ("stated", "default")


def step_fraction(rule: Any) -> tuple[float, str]:
    """The Courant budget that the function of cell ``rule`` declares, and its origin.

    A library declares it with the attribute ``step_fraction`` (a protocol of the step rules of a
    library, not a word of ``pops``); a function without it takes 1. The value is finite and in
    (0, 1]: a fraction can only tighten the face guards of a fixed step (C2, K2).
    """
    stated = getattr(rule, "step_fraction", None)
    if stated is None:
        return 1.0, "default"
    if isinstance(stated, bool) or not isinstance(stated, (int, float)):
        raise TypeError("the step_fraction of a step rule is a number in (0, 1], got %r"
                        % (stated,))
    value = float(stated)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("the step_fraction of a step rule is in (0, 1], got %r" % (stated,))
    return value, "stated"


def _leaves(expression: Expr) -> list[Expr]:
    found, seen, stack = [], set(), [expression]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        children = tuple(_children(node))
        if not children:
            found.append(node)
        stack.extend(reversed(children))
    return found


def _without_handles(key: Any) -> Any:
    """The structural key of E with each cell read keyed by its component and offset only."""
    if isinstance(key, tuple):
        if len(key) == 4 and key[0] == "cell_read":
            return ("cell_read", key[2], key[3])
        return tuple(_without_handles(item) for item in key)
    return key


class StepBound:
    """``StepBound(expression, dimension, fraction=1.0, origin="default")``: the step bound of a
    cells rule in place of the first ``Proposal`` of an Advance, on a layout of ``dimension`` axes.
    Private: the surface writes ``dt=f``."""

    __slots__ = ("expression", "dimension", "fraction", "origin")
    request = None  # the trace reads ``p.request`` (5.5): a bound carries no solve request

    def __init__(self, expression: Any, dimension: int, fraction: Any = 1.0,
                 origin: str = "default") -> None:
        from pops.mesh._amr.refinement_criterion import CellRead, CellWidth

        if not isinstance(expression, Expr):
            raise TypeError("StepBound(expression): an expression over the cell, got %r"
                            % (expression,))
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise TypeError("StepBound(dimension=): the number of axes of the layout")
        reads = False
        for leaf in _leaves(expression):
            if isinstance(leaf, (CellRead, CellWidth)):
                reads = True
                if isinstance(leaf, CellWidth) and not 0 <= leaf.axis < dimension:
                    raise ValueError("StepBound reads the width of axis %d of a layout of "
                                     "dimension %d" % (leaf.axis, dimension))
                if isinstance(leaf, CellRead) and leaf.offset:
                    raise TypeError("StepBound reads the cell itself, not a neighbour")
            elif not isinstance(leaf, (Const, RuntimeParamRef)):
                raise TypeError("StepBound reads cell values, cell widths, runtime parameters and "
                                "constants, not %r" % (leaf,))
        if not reads:
            raise TypeError("StepBound reads no cell: a fixed step is a Proposal")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) \
                or not math.isfinite(float(fraction)) or not 0.0 < float(fraction) <= 1.0:
            raise ValueError("StepBound(fraction=): a Courant budget in (0, 1], got %r"
                             % (fraction,))
        if origin not in _ORIGINS:
            raise ValueError("StepBound(origin=) is stated or default")
        self.expression = expression
        self.dimension = dimension
        self.fraction = float(fraction)
        self.origin = origin

    def __repr__(self) -> str:
        return "StepBound(%s, fraction=%r)" % (self.expression, self.fraction)


@dataclass(frozen=True, slots=True)
class ResolvedStepBound:
    """A step bound resolved for the one state of an Advance: E, the state, the components that
    E reads, the runtime parameters it reads, its dimension, and the declared fraction."""

    expression: Expr
    state: Any
    components: tuple[str, ...]
    parameters: tuple[Any, ...]
    dimension: int
    fraction: float
    origin: str
    __pops_ir_immutable__ = True

    def canonical_identity(self) -> dict[str, Any]:
        """The identity of the bound: E, the components and the parameters it reads, by name. A
        cell read names the component of the one state of the Advance, not the handle of that state,
        which the Program identifies; φ is the rule's, recorded beside it in the compiled control."""
        from pops.mesh._amr.refinement_criterion import _key_data

        return {
            "schema_version": _SCHEMA_VERSION,
            "resolved_type": "cells_step_bound",
            "bound": _key_data(_without_handles(_key(self.expression))),
            "components": list(self.components),
            "parameters": [handle.local_id for handle in self.parameters],
            "dimension": self.dimension,
        }

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def qualified_id(self) -> str:
        return "pops.cells-step-bound.v1::%s" % self.canonical_id


def resolve_step_bound(bound: StepBound, *, state: Any) -> ResolvedStepBound:
    """Resolve ``bound`` for the block-qualified ``state`` of the Advance: every cell read names
    a component of that state, every width an axis of the layout."""
    from pops.mesh._amr.refinement_criterion import CellRead

    if type(bound) is not StepBound:
        raise TypeError("resolve_step_bound takes a StepBound")
    space_components = tuple(getattr(getattr(state, "space", None), "components", ()))
    components: list[str] = []
    parameters: dict[str, Any] = {}
    for leaf in _leaves(bound.expression):
        if isinstance(leaf, CellRead):
            if leaf.handle.qualified_id != state.qualified_id:
                raise ValueError("dt= as a function of cell reads %s, a state that the Advance "
                                 "does not step" % leaf.handle.qualified_id)
            if leaf.component not in space_components:
                raise ValueError("a cell read names a component outside its typed Space")
            if leaf.component not in components:
                components.append(leaf.component)
        elif isinstance(leaf, RuntimeParamRef):
            if leaf.handle is None:
                raise TypeError("dt= as a function of cell reads a runtime parameter without "
                                "its declaration")
            parameters.setdefault(leaf.name, leaf.handle)
    return ResolvedStepBound(
        expression=bound.expression, state=state, components=tuple(components),
        parameters=tuple(parameters[name] for name in sorted(parameters)),
        dimension=bound.dimension,
        fraction=bound.fraction, origin=bound.origin)


__all__ = ["ResolvedStepBound", "StepBound", "resolve_step_bound", "step_fraction"]
