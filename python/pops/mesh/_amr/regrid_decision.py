"""The decision to regrid written in the Program: ``Regrid(criterion, when)`` (spec 0.3, DM13b).

``Program(..., regrid=Regrid(criterion, when))`` writes in the Program both the refinement
criterion and the decision to regrid. ``criterion`` is an ``AMRTagging`` whose ``Tag`` and
``Coarsen`` predicates are functions of ``cell`` (the regions of N12 lot 1, generated into the
compiled Program). ``when`` is a closed body over the accepted continuation ``B`` that returns a
Bool (T33): PoPS calls it once with a :class:`RegridContinuation` and keeps the Boolean expression
that it returns. The body reads

- ``B.steps``, the number of accepted steps of the run, counted from its start (spec 5.2; a run
  starts at ``pops.run``, at ``AmrSystem.run``, at a restored clock or at the bind, and
  ``pops.step`` continues it), as ``B.steps % k == 0``: ``pops.lib.time.every(k)`` is this body,
  true after every k-th accepted step of the run;
- ``B.time``, the accepted time;
- an indicator of the accepted state, ``largest_cells(f)`` or ``smallest_cells(f)``: the largest or
  the smallest value of ``f``, a function of ``cell`` read like a refinement region, over the valid
  cells of every level of the hierarchy;
- runtime parameters, written ``case.value(param)``, and constants.

The body is two-valued, as an IR4 ``cond``: it may compare with ``<``, ``<=``, ``>``, ``>=`` or
``==`` and join comparisons with ``&``, ``|`` and ``~``. Python's ``if``, ``and`` and ``or`` cannot
evaluate it. The decision is generated into the compiled Program with the criterion; after each
accepted step, the core reduces each indicator over the patches of every level and over the ranks,
calls the generated decision and regrids when it is true. No Python runs during the step (CC4,
CF9). An indicator that is not finite fails the step.

This lot reads an indicator at the cell itself: ``cell.shift`` inside ``largest_cells`` is refused,
since the accepted state has no filled halo when the decision runs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from pops._ir.expr import BooleanAnd, BooleanNot, BooleanOr, Compare, Const, Expr, Var
from pops._ir.handle_expr import ValueExpr
from pops._ir.values import RuntimeParamRef
from pops._ir.visitors import _children, _key
from pops.model import Handle, ParamHandle

from .refinement_criterion import (
    CellRead,
    CellWidth,
    CriterionCell,
    RefinementCriterionField,
    _key_data,
)

_SCHEMA_VERSION = 1
_INDICATOR_KINDS = ("largest", "smallest")


class RegridBodyError(TypeError):
    """A ``when`` body of ``Regrid`` reads outside what T33 allows, or does not return a Bool."""


def _not_a_predicate(self: Any, context: Any, *, action: str) -> Any:
    raise TypeError("%s belongs to the when body of a Regrid, not to a Tag or Coarsen rule"
                    % type(self).__name__)


class EveryAcceptedSteps(Expr):
    """``B.steps % k == 0``: true after every k-th accepted step of the run (5.11, ``every``)."""

    def __init__(self, k: int) -> None:
        self.k = k

    resolve_for_amr_predicate = _not_a_predicate

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("accepted_steps_every", self.k)

    def to_cpp(self) -> str:
        raise TypeError("a when body is lowered by the regrid decision of its Program")

    def _str(self) -> str:
        return "B.steps %% %d == 0" % self.k


class AcceptedTime(Expr):
    """``B.time``: the accepted time after the step."""

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("accepted_time",)

    def to_cpp(self) -> str:
        raise TypeError("a when body is lowered by the regrid decision of its Program")

    def _str(self) -> str:
        return "B.time"


class CellIndicator(Expr):
    """``largest_cells(f)`` or ``smallest_cells(f)``: an indicator of the accepted state.

    ``expression`` is the scalar that ``f`` returns for a :class:`CriterionCell`. The indicator is
    a leaf of the body: the core computes it over the valid cells of every level.
    """

    def __init__(self, kind: str, expression: Expr) -> None:
        self.kind = kind
        self.expression = expression

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("cell_indicator", self.kind, recurse(self.expression))

    def to_cpp(self) -> str:
        raise TypeError("an indicator is lowered by the regrid decision of its Program")

    def _str(self) -> str:
        return "%s_cells(%s)" % (self.kind, self.expression)


class _StepsRemainder:
    __slots__ = ("k",)

    def __init__(self, k: int) -> None:
        self.k = k

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        if isinstance(other, bool) or other != 0 or not isinstance(other, int):
            raise RegridBodyError("a when body tests B.steps % k == 0 only (every(k))")
        return EveryAcceptedSteps(self.k)

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        raise RegridBodyError("a when body tests B.steps % k == 0 only; write "
                              "~(B.steps % k == 0) for the steps where it does not hold")

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        raise RegridBodyError("B.steps % k has no number while the when body is traced")


class AcceptedSteps:
    """``B.steps``: the number of accepted steps, read as ``B.steps % k == 0``."""

    __slots__ = ()

    def __mod__(self, k: Any) -> _StepsRemainder:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise RegridBodyError("B.steps % k needs an integer k >= 1")
        return _StepsRemainder(k)

    def __bool__(self) -> bool:
        raise RegridBodyError("B.steps has no number while the when body is traced; write "
                              "B.steps % k == 0, as every(k) does")

    def _compare(self, other: Any) -> Any:
        raise RegridBodyError("a when body reads B.steps as B.steps % k == 0 only")

    __lt__ = __le__ = __gt__ = __ge__ = __eq__ = __ne__ = _compare  # type: ignore[assignment]
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _compare
    __hash__ = None  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:
        raise RegridBodyError("a when body reads B.steps as B.steps % k == 0 only")


class RegridContinuation:
    """``B`` inside the ``when`` body of a ``Regrid``: the accepted continuation (T33)."""

    __slots__ = ()

    @property
    def steps(self) -> AcceptedSteps:
        return AcceptedSteps()

    @property
    def time(self) -> AcceptedTime:
        return AcceptedTime()

    @property
    def state(self) -> Any:
        raise RegridBodyError(
            "a when body reads the accepted state through an indicator: largest_cells(f) or "
            "smallest_cells(f), with f a function of cell")

    @property
    def memory(self) -> Any:
        raise NotImplementedError(
            "the memory of the continuation (section 9, T27) is not read by a when body yet")


def _indicator(kind: str, function: Any, over: Any) -> CellIndicator:
    if over is not None:
        raise NotImplementedError(
            "%s_cells(f, over=...) reduces over a region; a when body reduces over every cell "
            "of the hierarchy, written without over=" % kind)
    if not callable(function):
        raise TypeError("%s_cells(f) takes a function of cell" % kind)
    value = function(CriterionCell())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = Const(value)
    if not isinstance(value, Expr) or isinstance(value, (Compare, BooleanAnd, BooleanOr,
                                                          BooleanNot)):
        raise TypeError("%s_cells(f): f returns a scalar of the cell, such as cell.value(q)"
                        % kind)
    return CellIndicator(kind, value)


def largest_cells(f: Callable[[CriterionCell], Any], over: Any = None) -> CellIndicator:
    """The largest value of ``f`` over the valid cells of every level (spec 0.3, 2.6.3).

    The valid cells of a coarse level include the cells that a finer level covers. For a linear
    ``f`` of a restricted state this is the largest value over the composite (uncovered) cells; for a
    nonlinear ``f``, or one that reads ``cell.width``, a covered coarse cell can give a value
    that no composite cell gives. A composite mask comes with ``over=``, refused in this lot.
    """
    return _indicator("largest", f, over)


def smallest_cells(f: Callable[[CriterionCell], Any], over: Any = None) -> CellIndicator:
    """The smallest value of ``f`` over the valid cells of every level (spec 0.3, 2.6.3).

    The valid cells of a coarse level include the cells that a finer level covers. For a linear
    ``f`` of a restricted state this is the smallest value over the composite (uncovered) cells; for a
    nonlinear ``f``, or one that reads ``cell.width``, a covered coarse cell can give a value
    that no composite cell gives. A composite mask comes with ``over=``, refused in this lot.
    """
    return _indicator("smallest", f, over)


def _bool_body(expression: Any) -> bool:
    if isinstance(expression, (EveryAcceptedSteps,)):
        return True
    if isinstance(expression, Compare):
        return not (_bool_body(expression.a) or _bool_body(expression.b))
    if isinstance(expression, (BooleanAnd, BooleanOr)):
        return _bool_body(expression.a) and _bool_body(expression.b)
    if isinstance(expression, BooleanNot):
        return _bool_body(expression.a)
    return False


def _leaves(roots: tuple[Expr, ...]) -> list[Expr]:
    found, seen = [], set()

    def visit(node: Expr) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        children = _children(node)
        if not children:
            found.append(node)
        for child in children:
            visit(child)

    for root in roots:
        visit(root)
    return found


def _refuse_model_parameter(leaf: RuntimeParamRef, where: str) -> None:
    raise RegridBodyError(
        "%s reads the runtime parameter %r of a model; a when body and its indicators read the "
        "parameters of the case in this lot: declare the threshold with case.param and read it "
        "with case.value (Q4). Reading a model parameter there is a named follow-up of N12 "
        "(model parameter in a Regrid when body)" % (where, leaf.name))


def _check_body_leaves(expression: Expr) -> None:
    for leaf in _leaves((expression,)):
        if isinstance(leaf, (Const, EveryAcceptedSteps, AcceptedTime)):
            continue
        if isinstance(leaf, CellIndicator):
            _check_indicator_leaves(leaf)
            continue
        if isinstance(leaf, ValueExpr) and isinstance(leaf.handle, ParamHandle):
            continue
        if type(leaf) is RuntimeParamRef:
            _refuse_model_parameter(leaf, "a when body")
        if isinstance(leaf, (CellRead, CellWidth)):
            raise RegridBodyError(
                "a when body reads cells through largest_cells(f) or smallest_cells(f) only")
        if isinstance(leaf, Var):
            raise RegridBodyError("a when body reads %r, which is none of B.steps, B.time, an "
                                  "indicator or a runtime parameter (T33)" % leaf.name)
        raise RegridBodyError(
            "a when body reads B.steps, B.time, indicators of the accepted state and runtime "
            "parameters (T33); %s is none of them" % type(leaf).__name__)


def _check_indicator_leaves(indicator: CellIndicator) -> None:
    for leaf in _leaves((indicator.expression,)):
        if isinstance(leaf, (Const, CellWidth)):
            continue
        if isinstance(leaf, CellRead):
            if leaf.offset:
                raise NotImplementedError(
                    "%s_cells(f) reads the cell itself in this lot: the accepted state has no "
                    "filled halo when the regrid decision runs, so cell.shift is refused"
                    % indicator.kind)
            continue
        if isinstance(leaf, ValueExpr) and isinstance(leaf.handle, ParamHandle):
            continue
        if type(leaf) is RuntimeParamRef:
            _refuse_model_parameter(leaf, "the indicator %s_cells(f)" % indicator.kind)
        if isinstance(leaf, (EveryAcceptedSteps, AcceptedTime, CellIndicator)):
            raise RegridBodyError("an indicator is a function of cell; it reads neither B nor "
                                  "another indicator")
        raise RegridBodyError(
            "an indicator reads the cell through cell.value and cell.width, and runtime "
            "parameters; %s is none of them" % type(leaf).__name__)


@dataclass(frozen=True, slots=True)
class RegridWhen:
    """A ``when`` body traced once: the Boolean expression it returns."""

    expression: Expr
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.expression, Expr) or not _bool_body(self.expression):
            raise RegridBodyError(
                "a when body returns a Bool: a comparison, B.steps % k == 0 or a join of them "
                "with &, | and ~")
        _check_body_leaves(self.expression)

    @classmethod
    def trace(cls, body: Any) -> RegridWhen:
        if type(body) is cls:
            return body
        if not callable(body):
            raise TypeError("Regrid(when=...) takes a closed body of B, such as "
                            "pops.lib.time.every(4)")
        value: Any = body(RegridContinuation())
        if isinstance(value, bool):
            raise RegridBodyError(
                "a when body returned the Python value %r: it must return a traced Bool of B, "
                "such as every(1) for every step" % value)
        return cls(value)

    def resolve_references(self, resolver: Any) -> RegridWhen:
        return type(self)(self.expression.resolve_references(resolver))

    def key_data(self) -> Any:
        return _key_data(_key(self.expression))


@dataclass(frozen=True, slots=True)
class Regrid:
    """``Regrid(criterion, when)``: the refinement criterion and the decision to regrid (DM13b).

    ``criterion`` is an ``AMRTagging`` whose ``Tag`` and ``Coarsen`` predicates are functions of
    ``cell``; ``when`` is a closed body of the accepted continuation that returns a Bool, such as
    ``pops.lib.time.every(4)`` or ``lambda B: largest_cells(lambda cell: cell.value(q)) > c``.
    A Program receives it with ``program.regrid(Regrid(...))``; the adaptive layout then writes
    neither ``tagging=`` nor ``regrid=``.
    """

    criterion: Any
    when: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops.amr.authoring import AMRTagging, Buffer
        from .refinement_criterion import RefinementRegion

        if type(self.criterion) is not AMRTagging:
            raise TypeError("Regrid(criterion, when): the criterion is an AMRTagging")
        if any(type(rule.predicate) is not RefinementRegion
               for rule in self.criterion.rules if type(rule) is not Buffer):
            raise TypeError(
                "Regrid(criterion, when): the criterion writes every Tag and Coarsen predicate "
                "as a function of cell, generated into the Program (DM13b)")
        object.__setattr__(self, "when", RegridWhen.trace(self.when))

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "authority_type": "program_regrid",
                "criterion": self.criterion.inspect(), "when": self.when.key_data()}

    inspect = to_data


@dataclass(frozen=True, slots=True)
class ResolvedRegridDecision:
    """A ``when`` body resolved for one adaptive layout: its indicators and their field table."""

    expression: Expr
    indicators: tuple[CellIndicator, ...]
    dimension: int
    fields: tuple[RefinementCriterionField, ...]
    parameters: tuple[ParamHandle, ...]
    __pops_ir_immutable__ = True

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "resolved_type": "amr_regrid_decision",
            "when": _key_data(_key(self.expression)),
            "indicators": [_key_data(_key(indicator)) for indicator in self.indicators],
            "dimension": self.dimension,
            "fields": [row.canonical_identity() for row in self.fields],
            "parameters": [handle.canonical_identity() for handle in self.parameters],
        }

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def qualified_id(self) -> str:
        return "pops.resolved-amr-regrid-decision.v1::%s" % self.canonical_id

    def runtime_decision_data(self, params: Any) -> dict[str, Any]:
        """The tables that ``bind`` installs, with the values bound to the runtime parameters."""
        values = []
        for handle in self.parameters:
            if params is None or handle not in params:
                raise ValueError("regrid decision parameter %s is not bound"
                                 % handle.qualified_id)
            value = params[handle]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise TypeError("regrid decision parameters must be finite scalars")
            values.append(float(value))
        return {
            "schema_version": _SCHEMA_VERSION,
            "decision_type": "amr_regrid_decision_runtime",
            "identity": self.qualified_id,
            "fields": [{**row.canonical_identity(),
                        "state": row.state.canonical_identity()["qualified_id"]}
                       for row in self.fields],
            "parameters": values,
            "indicator_count": len(self.indicators),
        }


def resolve_regrid_decision(
    when: RegridWhen, *, dimension: int, admit_state: Callable[[Handle], None],
) -> ResolvedRegridDecision:
    """Resolve a ``when`` body: its indicators in first-visit order, their fields and parameters."""
    if type(when) is not RegridWhen:
        raise TypeError("a regrid decision resolves a traced when body")
    indicators: dict[Any, CellIndicator] = {}
    parameters: dict[str, ParamHandle] = {}
    reads: dict[str, dict[str, Any]] = {}

    def parameter(leaf: ValueExpr) -> None:
        handle = leaf.handle
        if not isinstance(handle, ParamHandle) or not handle.is_resolved \
                or handle.param_kind != "runtime":
            raise TypeError("a when body reads canonical runtime parameters")
        parameters.setdefault(handle.qualified_id, handle)

    for leaf in _leaves((when.expression,)):
        if isinstance(leaf, ValueExpr):
            parameter(leaf)
        if not isinstance(leaf, CellIndicator):
            continue
        indicators.setdefault(_key(leaf), leaf)
    for indicator in indicators.values():
        for leaf in _leaves((indicator.expression,)):
            if isinstance(leaf, ValueExpr):
                parameter(leaf)
            elif isinstance(leaf, CellWidth):
                if leaf.axis >= dimension:
                    raise ValueError("an indicator reads the width of axis %d of a level of "
                                     "dimension %d" % (leaf.axis, dimension))
            elif isinstance(leaf, CellRead):
                state = leaf.handle
                if not state.is_resolved:
                    raise TypeError("an indicator reads canonical block states")
                components = tuple(getattr(getattr(state, "space", None), "components", ()))
                if leaf.component not in components:
                    raise ValueError("a cell read names a component outside its typed Space")
                row = reads.get(state.qualified_id)
                if row is None:
                    admit_state(state)
                    row = reads[state.qualified_id] = {"state": state, "variables": []}
                if leaf.component not in row["variables"]:
                    row["variables"].append(leaf.component)
    fields = []
    for row in reads.values():
        state = row["state"]
        block = state.canonical_identity().get("block_ref", {}).get("local_id")
        if not isinstance(block, str) or not block:
            raise ValueError("an indicator reads a state that no block owns")
        components = tuple(state.space.components)
        fields.append(RefinementCriterionField(
            state, block, tuple(row["variables"]),
            tuple(components.index(name) for name in row["variables"]), (0,) * dimension))
    return ResolvedRegridDecision(
        expression=when.expression, indicators=tuple(indicators.values()), dimension=dimension,
        fields=tuple(fields), parameters=tuple(parameters.values()))


__all__ = [
    "AcceptedSteps",
    "AcceptedTime",
    "CellIndicator",
    "EveryAcceptedSteps",
    "Regrid",
    "RegridBodyError",
    "RegridContinuation",
    "RegridWhen",
    "ResolvedRegridDecision",
    "largest_cells",
    "resolve_regrid_decision",
    "smallest_cells",
]
