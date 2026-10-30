"""Refinement criteria written as regions of ``cell`` (spec 0.3, DM13b and 2.6.6).

The predicate of a ``Tag`` or ``Coarsen`` rule may be a function of ``cell``. PoPS calls it once
with a :class:`CriterionCell` and keeps the Boolean expression that it returns: ``cell.value(q)``
reads a block state at the cell, ``cell.shift(axis, k)`` is the cell ``k`` steps along an axis,
read like ``cell``, and ``cell.width(axis)`` is the width of the cells of the level along an axis.
A runtime parameter enters as ``case.value(param)`` and is bound at ``bind``.

A region decides as a rule of the tag graph does. It compares with a strict ``<`` or ``>``, and a
comparison at exact equality decides nothing (``EqualityPolicy.HOLD``); ``~``, ``&`` and ``|``
follow Kleene's three-valued logic, and a rule acts on the cells where its region is certainly
true (:func:`certainly`). Every comparison of every rule is evaluated, so a non-finite value that
any of them reads rejects the level, as it does for the bytecode.

The resolved criterion decides ``where(refine, refine, where(coarsen, coarsen, keep))`` on each
cell of a level, ``refine`` being the union of the refine regions and ``coarsen`` the union of the
coarsen regions. The compiled Program carries it as generated C++; the native tagging plan hands it
the patches of a level and applies hysteresis and the buffer downstream, as for the bytecode of the
tag graph. The core never learns which quantities, offsets or arithmetic the criterion uses.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from pops._ir.expr import BooleanAnd, BooleanNot, BooleanOr, Compare, Const, Expr, Var
from pops._ir.handle_expr import ValueExpr
from pops._ir.quantity import QuantityRef
from pops._ir.visitors import _children, _key
from pops.model import Handle, OwnerPath, ParamHandle

from .tagging_graph import ConflictPolicy, EqualityPolicy, Hysteresis

_SCHEMA_VERSION = 1


def _program_lowering() -> dict[str, Any]:
    """The lowering of every region: generated into the compiled Program, which the native tagging
    plan calls on each patch (the indicator provider that the bootstrap receipts name)."""
    lowering = Handle("program_region", kind="tag_lowering",
                      owner=OwnerPath.shared("pops.amr.refinement-criterion.v1"))
    return {"schema_version": _SCHEMA_VERSION, "node_type": "refinement_region",
            "lowering": lowering.canonical_identity()}


def _axis_index(axis: Any, *, where: str) -> int:
    from pops.frames.cartesian import CartesianAxis

    if type(axis) is not CartesianAxis:
        raise TypeError("%s requires an axis of a Cartesian frame, such as frame.axes[0]" % where)
    return axis.index


def _offset_text(offset: tuple[tuple[int, int], ...]) -> str:
    return "".join(".shift(%d, %d)" % pair for pair in offset)


class CellRead(Expr):
    """The value of one component of a block state at the cell shifted by ``offset``.

    ``offset`` holds the nonzero steps as sorted ``(axis, k)`` pairs, so two reads of the same cell
    have the same structural key whatever order their shifts were written in.
    """

    def __init__(self, handle: Any, component: str, offset: tuple[tuple[int, int], ...]) -> None:
        self.handle = handle
        self.component = component
        self.offset = offset

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("cell_read", self.handle.qualified_id, self.component,
                tuple(tuple(pair) for pair in self.offset))

    def deps(self) -> set[str]:
        return {"%s::component::%s%s" % (self.handle.qualified_id, self.component,
                                         _offset_text(self.offset))}

    def to_cpp(self) -> str:
        raise TypeError("a cell read is lowered by the refinement criterion of its Program")

    def _str(self) -> str:
        return "cell%s.value(%s[%s])" % (_offset_text(self.offset), self.handle.local_id,
                                         self.component)


class CellWidth(Expr):
    """The width of the cells of the level along one axis."""

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("cell_width", self.axis)

    def deps(self) -> set[str]:
        return {"cell::width::%d" % self.axis}

    def to_cpp(self) -> str:
        raise TypeError("a cell width is lowered by the refinement criterion of its Program")

    def _str(self) -> str:
        return "cell.width(%d)" % self.axis


class CriterionCell:
    """The cell that a refinement region reads (2.6.2): ``value``, ``shift`` and ``width``."""

    __slots__ = ("_offset",)

    def __init__(self, offset: tuple[tuple[int, int], ...] = ()) -> None:
        self._offset = offset

    def shift(self, axis: Any, k: int) -> CriterionCell:
        """The cell ``k`` steps along ``axis``, read like ``cell``."""
        index = _axis_index(axis, where="cell.shift")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("cell.shift requires an integer number of steps")
        steps = dict(self._offset)
        steps[index] = steps.get(index, 0) + k
        return CriterionCell(tuple(sorted((a, s) for a, s in steps.items() if s != 0)))

    def value(self, quantity: Any) -> Expr:
        """The value of a block state, or of one component of it, at this cell."""
        if isinstance(quantity, QuantityRef):
            handle, component = quantity.handle, quantity.component
        else:
            handle = quantity.handle if isinstance(quantity, ValueExpr) else quantity
            components = tuple(getattr(getattr(handle, "space", None), "components", ()))
            if not isinstance(handle, Handle) or len(components) != 1:
                raise TypeError(
                    "cell.value reads a scalar block state or one typed component of a state, "
                    "such as cell.value(block[U]) or cell.value(ValueExpr(block[U])['u'])")
            component = components[0]
        if not isinstance(handle, Handle) or handle.kind != "state":
            raise TypeError("cell.value reads a block state")
        return CellRead(handle, component, self._offset)

    def width(self, axis: Any) -> Expr:
        """The width of the cells of the level along ``axis``."""
        return CellWidth(_axis_index(axis, where="cell.width"))


def _boolean(value: Any) -> bool:
    return isinstance(value, Expr) and callable(getattr(value, "resolve_for_amr_predicate", None))


_STRICT = {"lt": "gt", "gt": "lt"}
_DECISION = ("a refinement region compares with a strict < or > and joins comparisons with &, | "
             "and ~: as for a leaf of the tag graph, a comparison at exact equality decides "
             "nothing (EqualityPolicy.HOLD)")


def certainly(expression: Expr, truth: bool = True) -> Expr:
    """The two-valued expression of the cells where a region is certainly ``truth``.

    As a leaf of the tag graph, a strict comparison at exact equality decides nothing: ``a < b`` is
    certainly true where a < b and certainly false where a > b. ``~``, ``&`` and ``|`` follow
    Kleene's three-valued logic, so the result joins strict comparisons of the same operands with
    ``&`` and ``|`` and holds no negation: ``~(a < b)`` is certainly true where a > b only.
    """
    if isinstance(expression, BooleanNot):
        return certainly(expression.a, not truth)
    if isinstance(expression, (BooleanAnd, BooleanOr)):
        join = BooleanAnd if isinstance(expression, BooleanAnd) == truth else BooleanOr
        return join(certainly(expression.a, truth), certainly(expression.b, truth))
    if isinstance(expression, Compare) and expression.comparison in _STRICT:
        if truth:
            return expression
        return Compare(_STRICT[expression.comparison], expression.a, expression.b)
    raise TypeError(_DECISION)


@dataclass(frozen=True, slots=True)
class RefinementRegion:
    """A refine or coarsen predicate written as a function of ``cell``, traced once."""

    expression: Expr
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not _boolean(self.expression):
            raise TypeError(
                "a refinement region must return a Boolean expression, such as a comparison of "
                "cell values with a threshold")
        # Refused at the author's line: a comparison that no leaf of the tag graph decides.
        certainly(self.expression)

    @classmethod
    def trace(cls, function: Callable[[CriterionCell], Any]) -> RefinementRegion:
        return cls(function(CriterionCell()))

    def resolve_references(self, resolver: Any) -> RefinementRegion:
        return type(self)(self.expression.resolve_references(resolver))

    def key_data(self) -> Any:
        return _key_data(_key(self.expression))


def _key_data(value: Any) -> Any:
    """Canonical JSON data of a structural key (finite floats as their binary64 text)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("a refinement criterion identity contains a non-finite float")
        return {"binary64": value.hex()}
    if isinstance(value, (tuple, list)):
        return [_key_data(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _key_data(item) for key, item in value.items()}
    to_data = getattr(value, "to_data", None)
    if callable(to_data):
        return _key_data(to_data())
    raise TypeError("a refinement criterion identity contains %s" % type(value).__name__)


@dataclass(frozen=True, slots=True)
class RefinementCriterionField:
    """One block state that a criterion reads: the variables it reads, their component indices in
    the state and, per axis, the largest step of its reads, which the halo of the state covers."""

    state: Handle
    block: str
    variables: tuple[str, ...]
    components: tuple[int, ...]
    reach: tuple[int, ...]
    __pops_ir_immutable__ = True

    def canonical_identity(self) -> dict[str, Any]:
        return {"state": self.state.qualified_id, "block": self.block,
                "variables": list(self.variables), "components": list(self.components),
                "reach": list(self.reach)}


@dataclass(frozen=True, slots=True)
class ResolvedRefinementCriterion:
    """A refinement criterion resolved for one adaptive layout (DM13b)."""

    refine: tuple[Expr, ...]
    coarsen: tuple[Expr, ...]
    hysteresis: Hysteresis
    conflict_policy: ConflictPolicy
    dimension: int
    fields: tuple[RefinementCriterionField, ...]
    parameters: tuple[ParamHandle, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not self.refine or any(not _boolean(root) for root in self.refine + self.coarsen):
            raise TypeError("a refinement criterion needs Boolean refine and coarsen regions")
        if type(self.hysteresis) is not Hysteresis:
            raise TypeError("a refinement criterion needs an exact Hysteresis")
        if self.conflict_policy is not ConflictPolicy.REFINE_WINS \
                or self.hysteresis.equality is not EqualityPolicy.HOLD:
            raise ValueError(
                "a criterion written as regions of cell decides refine before coarsen and never "
                "meets an equality: it requires ConflictPolicy.REFINE_WINS and "
                "EqualityPolicy.HOLD")
        if not self.fields or any(type(row) is not RefinementCriterionField
                                  for row in self.fields):
            raise TypeError("a refinement criterion reads at least one block state")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "resolved_type": "amr_refinement_criterion",
            "decision": "refine_then_coarsen",
            "refine": [_key_data(_key(root)) for root in self.refine],
            "coarsen": [_key_data(_key(root)) for root in self.coarsen],
            "hysteresis": self.hysteresis.canonical_identity(),
            "conflict_policy": self.conflict_policy.value,
            "dimension": self.dimension,
            "fields": [row.canonical_identity() for row in self.fields],
            "parameters": [handle.canonical_identity() for handle in self.parameters],
            "lowerings": [_program_lowering()],
        }

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def qualified_id(self) -> str:
        return "pops.resolved-amr-refinement-criterion.v1::%s" % self.canonical_id

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "resolved_amr_refinement_criterion",
                "canonical_id": self.canonical_id, "qualified_id": self.qualified_id,
                **self.canonical_identity()}

    def runtime_criterion_data(self, params: Any) -> dict[str, Any]:
        """The tables that ``bind`` installs, with the values bound to the runtime parameters."""
        values = []
        for handle in self.parameters:
            if params is None or handle not in params:
                raise ValueError("refinement criterion parameter %s is not bound"
                                 % handle.qualified_id)
            value = params[handle]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise TypeError("refinement criterion parameters must be finite scalars")
            values.append(float(value))
        return {
            "schema_version": _SCHEMA_VERSION,
            "criterion_type": "amr_refinement_criterion_runtime",
            "identity": self.qualified_id,
            "fields": [{**row.canonical_identity(),
                        "state": row.state.canonical_identity()["qualified_id"]}
                       for row in self.fields],
            "parameters": values,
            "hysteresis": self.hysteresis.canonical_identity(),
        }


def _leaves(roots: tuple[Expr, ...]) -> list[Expr]:
    """The leaves of a DAG in first-visit order, each object once."""
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


def criterion_inputs(roots: tuple[Expr, ...]) -> list[Expr]:
    """The cell reads, widths and runtime parameters of a criterion, in first-visit order.

    A constant is compiled in. Any other leaf, a free variable or a handle read without
    ``cell.value``, is refused: the region has no other way to read the level.
    """
    inputs = []
    for leaf in _leaves(roots):
        if isinstance(leaf, Const):
            continue
        if isinstance(leaf, (CellRead, CellWidth)):
            inputs.append(leaf)
            continue
        if isinstance(leaf, ValueExpr) and isinstance(leaf.handle, ParamHandle):
            inputs.append(leaf)
            continue
        if isinstance(leaf, Var):
            raise TypeError(
                "a refinement region reads %r without cell.value(...); read a block state at "
                "the cell" % leaf.name)
        raise TypeError(
            "a refinement region reads the level through cell.value, cell.width and runtime "
            "parameters; %s is none of them" % type(leaf).__name__)
    return inputs


def resolve_refinement_criterion(
    rules: tuple[tuple[str, RefinementRegion], ...],
    *,
    hysteresis: Hysteresis,
    conflict_policy: ConflictPolicy,
    dimension: int,
    admit_state: Callable[[Handle], None],
) -> ResolvedRefinementCriterion:
    """Resolve the regions of an AMRTagging into one criterion and its field and parameter tables.

    ``admit_state`` refuses a state that the adaptive layout does not own. The native tagging plan
    checks the reach of each field against the halo that it binds.
    """
    roots: dict[str, list[Expr]] = {"refine": [], "coarsen": []}
    for action, region in rules:
        if type(region) is not RefinementRegion or action not in roots:
            raise TypeError("a refinement criterion is built from refine and coarsen regions")
        roots[action].append(region.expression)
    refine, coarsen = tuple(roots["refine"]), tuple(roots["coarsen"])
    reads: dict[str, dict[str, Any]] = {}
    parameters: dict[str, ParamHandle] = {}
    for leaf in criterion_inputs(refine + coarsen):
        if isinstance(leaf, CellWidth):
            if leaf.axis >= dimension:
                raise ValueError("a refinement region reads the width of axis %d of a level of "
                                 "dimension %d" % (leaf.axis, dimension))
            continue
        if isinstance(leaf, ValueExpr):
            handle = leaf.handle
            if not isinstance(handle, ParamHandle) or not handle.is_resolved \
                    or handle.param_kind != "runtime":
                raise TypeError("a refinement region reads canonical runtime parameters")
            parameters.setdefault(handle.qualified_id, handle)
            continue
        if not isinstance(leaf, CellRead):
            raise TypeError("a refinement region reads the level through cell.value")
        state = leaf.handle
        if not state.is_resolved:
            raise TypeError("a refinement region reads canonical block states")
        components = tuple(getattr(getattr(state, "space", None), "components", ()))
        if leaf.component not in components:
            raise ValueError("a cell read names a component outside its typed Space")
        if any(axis >= dimension for axis, _ in leaf.offset):
            raise ValueError("a refinement region shifts along an axis outside the level")
        row = reads.get(state.qualified_id)
        if row is None:
            admit_state(state)
            row = reads[state.qualified_id] = {
                "state": state, "variables": [], "reach": [0] * dimension}
        if leaf.component not in row["variables"]:
            row["variables"].append(leaf.component)
        for axis, steps in leaf.offset:
            row["reach"][axis] = max(row["reach"][axis], abs(steps))
    fields = []
    for row in reads.values():
        state = row["state"]
        block = state.canonical_identity().get("block_ref", {}).get("local_id")
        if not isinstance(block, str) or not block:
            raise ValueError("a refinement region reads a state that no block owns")
        components = tuple(state.space.components)
        fields.append(RefinementCriterionField(
            state, block, tuple(row["variables"]),
            tuple(components.index(name) for name in row["variables"]), tuple(row["reach"])))
    return ResolvedRefinementCriterion(
        refine=refine, coarsen=coarsen, hysteresis=hysteresis, conflict_policy=conflict_policy,
        dimension=dimension, fields=tuple(fields), parameters=tuple(parameters.values()))


__all__ = [
    "CellRead",
    "CellWidth",
    "CriterionCell",
    "RefinementCriterionField",
    "RefinementRegion",
    "ResolvedRefinementCriterion",
    "certainly",
    "criterion_inputs",
    "resolve_refinement_criterion",
]
