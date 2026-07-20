"""Immutable, serializable expression trees for analytic scalar fields.

This module is intentionally independent from the PDE IR.  Analytic fields describe
coordinate-local data (initial conditions, masks and indicators); they are not state
handles and cannot be evaluated by Python control flow.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Real
from typing import Any, ClassVar

from pops.frames import CartesianAxis


SCHEMA_VERSION = 1
DEFAULT_MAX_DEPTH = 64
DEFAULT_MAX_NODES = 4096
DEFAULT_MAX_STACK = 64

_SCALAR_UNARY_OPS = frozenset({"neg", "sqrt", "abs", "sin", "cos", "exp", "log"})
_SCALAR_BINARY_OPS = frozenset({
    "add", "sub", "mul", "div", "pow", "atan2", "hypot", "minimum", "maximum",
})
_SCALAR_OPS = frozenset({"constant", "coordinate", "parameter", "input", "where"}) | _SCALAR_UNARY_OPS \
    | _SCALAR_BINARY_OPS
_COMPARISON_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_LOGICAL_BINARY_OPS = frozenset({"and", "or"})
_PREDICATE_OPS = _COMPARISON_OPS | _LOGICAL_BINARY_OPS | frozenset({"not", "between"})


class AnalyticTruthValueError(TypeError):
    """Raised when Python tries to decide an analytic expression's truth value."""

    def __init__(self, value: Any) -> None:
        super().__init__(
            "%s has no Python truth value; use where(...) for scalar selection and "
            "&, | or ~ for analytic predicates" % type(value).__name__
        )


@dataclass(frozen=True, slots=True)
class ExpressionStats:
    """Finite size measured while validating one expression tree."""

    node_count: int
    depth: int
    frame_id: str | None
    required_stack: int


@dataclass(frozen=True, slots=True)
class _CoordinateRef:
    frame_id: str
    axis: CartesianAxis

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise TypeError("analytic coordinate frame_id must be non-empty text")
        if not isinstance(self.axis, CartesianAxis):
            raise TypeError("analytic coordinate axis must be a typed CartesianAxis")


@dataclass(frozen=True, slots=True)
class _InputRef:
    """Typed reference to one value produced inside a data-only setup program.

    The integer is local to the owning program.  It is deliberately not a runtime storage index:
    the consuming lowerer authenticates the value/component pair and assigns a native input slot.
    """

    value_id: int
    component: str

    def __post_init__(self) -> None:
        if type(self.value_id) is not int or self.value_id < 0:
            raise TypeError("analytic input value_id must be a non-negative exact integer")
        if not isinstance(self.component, str) or not self.component \
                or self.component.strip() != self.component:
            raise TypeError("analytic input component must be canonical non-empty text")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ScalarExpr:
    """One immutable scalar analytic expression.

    Instances are created through :mod:`pops.analytic` factories and arithmetic.  ``==``
    and the ordering operators create :class:`PredicateExpr`; use :meth:`same_as` when
    structural equality is required.
    """

    _op: str
    _arguments: tuple[ScalarExpr | PredicateExpr, ...]
    _literal: float | None
    _coordinate: _CoordinateRef | None
    _parameter: Any
    _input: _InputRef | None
    _frame_id: str | None
    __hash__: ClassVar[None] = None
    __pops_ir_immutable__: ClassVar[bool] = True

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "ScalarExpr nodes are canonical; use constant(), coordinate() or analytic operators"
        )

    @classmethod
    def _new(
        cls,
        op: str,
        arguments: tuple[ScalarExpr | PredicateExpr, ...] = (),
        *,
        literal: float | None = None,
        coordinate: _CoordinateRef | None = None,
        parameter: Any = None,
        input_ref: _InputRef | None = None,
    ) -> ScalarExpr:
        result = object.__new__(cls)
        object.__setattr__(result, "_op", op)
        object.__setattr__(result, "_arguments", arguments)
        object.__setattr__(result, "_literal", literal)
        object.__setattr__(result, "_coordinate", coordinate)
        object.__setattr__(result, "_parameter", parameter)
        object.__setattr__(result, "_input", input_ref)
        object.__setattr__(result, "_frame_id", _merged_frame_id(arguments, coordinate))
        _validate_scalar_local(result)
        return result

    @property
    def op(self) -> str:
        return self._op

    @property
    def frame_id(self) -> str | None:
        return self._frame_id

    def __bool__(self) -> bool:
        raise AnalyticTruthValueError(self)

    def __repr__(self) -> str:
        return "ScalarExpr(op=%r, frame_id=%r)" % (self._op, self._frame_id)

    def __add__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("add", self, other)

    def __radd__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("add", other, self)

    def __sub__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("sub", self, other)

    def __rsub__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("sub", other, self)

    def __mul__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("mul", self, other)

    def __rmul__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("mul", other, self)

    def __truediv__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("div", self, other)

    def __rtruediv__(self, other: Any) -> ScalarExpr:
        return _scalar_binary("div", other, self)

    def __pow__(self, exponent: Any, modulo: Any = None) -> ScalarExpr:
        if modulo is not None:
            raise TypeError("analytic scalar power does not support a modulo argument")
        if isinstance(exponent, ScalarExpr):
            raise TypeError("analytic power exponent must be a finite scalar literal")
        return _scalar_binary("pow", self, constant(exponent))

    def __neg__(self) -> ScalarExpr:
        return _scalar_unary("neg", self)

    def __abs__(self) -> ScalarExpr:
        return _scalar_unary("abs", self)

    def __eq__(self, other: Any) -> PredicateExpr:  # type: ignore[override]
        return _comparison("eq", self, other)

    def __ne__(self, other: Any) -> PredicateExpr:  # type: ignore[override]
        return _comparison("ne", self, other)

    def __lt__(self, other: Any) -> PredicateExpr:
        return _comparison("lt", self, other)

    def __le__(self, other: Any) -> PredicateExpr:
        return _comparison("le", self, other)

    def __gt__(self, other: Any) -> PredicateExpr:
        return _comparison("gt", self, other)

    def __ge__(self, other: Any) -> PredicateExpr:
        return _comparison("ge", self, other)

    def validate(
        self,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_stack: int = DEFAULT_MAX_STACK,
    ) -> bool:
        _measure(self, max_depth=max_depth, max_nodes=max_nodes, max_stack=max_stack)
        return True

    def measure(
        self,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_stack: int = DEFAULT_MAX_STACK,
    ) -> ExpressionStats:
        return _measure(self, max_depth=max_depth, max_nodes=max_nodes, max_stack=max_stack)

    def to_data(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "expression_type": "scalar",
            "root": _node_to_data(self),
        }

    canonical_identity = to_data
    inspect = to_data

    @classmethod
    def from_data(
        cls,
        data: Any,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_stack: int = DEFAULT_MAX_STACK,
    ) -> ScalarExpr:
        result = _expression_from_data(
            data, expected="scalar", max_depth=max_depth, max_nodes=max_nodes,
            max_stack=max_stack)
        if not isinstance(result, ScalarExpr):
            raise TypeError("decoded analytic expression is not scalar")
        return result

    def same_as(self, other: Any) -> bool:
        return isinstance(other, ScalarExpr) and self.to_data() == other.to_data()

    def resolve_references(self, resolver: Any) -> ScalarExpr:
        """Authenticate every parameter leaf through one owning registry resolver."""

        if not callable(resolver):
            raise TypeError("analytic expression resolver must be callable")
        resolved = _resolve_references(self, resolver)
        if not isinstance(resolved, ScalarExpr):
            raise RuntimeError("analytic scalar reference resolution changed expression kind")
        return resolved

    @property
    def has_parameters(self) -> bool:
        return _has_parameter_leaf(self)

    def parameter_handles(self) -> tuple[Any, ...]:
        """Return parameter authorities in deterministic first-occurrence order.

        Handles remain separate from the expression graph: this projection is used only by
        owning registries that need to authenticate captured parameter leaves later.  The returned
        Handles are immutable values, never expression nodes or runtime values.
        """

        return _parameter_handles(self)

    def input_references(self) -> tuple[tuple[int, str], ...]:
        """Return program-local input leaves in deterministic first-occurrence order."""

        return _input_references(self)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class PredicateExpr:
    """One immutable analytic predicate composed with explicit bitwise operators."""

    _op: str
    _arguments: tuple[ScalarExpr | PredicateExpr, ...]
    _frame_id: str | None
    __hash__: ClassVar[None] = None
    __pops_ir_immutable__: ClassVar[bool] = True

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("PredicateExpr nodes are canonical; use comparisons, between(), &, | or ~")

    @classmethod
    def _new(
        cls,
        op: str,
        arguments: tuple[ScalarExpr | PredicateExpr, ...],
    ) -> PredicateExpr:
        result = object.__new__(cls)
        object.__setattr__(result, "_op", op)
        object.__setattr__(result, "_arguments", arguments)
        object.__setattr__(result, "_frame_id", _merged_frame_id(arguments, None))
        _validate_predicate_local(result)
        return result

    @property
    def op(self) -> str:
        return self._op

    @property
    def frame_id(self) -> str | None:
        return self._frame_id

    def __bool__(self) -> bool:
        raise AnalyticTruthValueError(self)

    def __repr__(self) -> str:
        return "PredicateExpr(op=%r, frame_id=%r)" % (self._op, self._frame_id)

    def __eq__(self, _other: Any) -> bool:  # type: ignore[override]
        raise TypeError("analytic predicates do not define ==; use same_as() for structure")

    def __ne__(self, _other: Any) -> bool:  # type: ignore[override]
        raise TypeError("analytic predicates do not define !=; use same_as() for structure")

    def __and__(self, other: Any) -> PredicateExpr:
        return _logical_binary("and", self, other)

    def __rand__(self, other: Any) -> PredicateExpr:
        return _logical_binary("and", other, self)

    def __or__(self, other: Any) -> PredicateExpr:
        return _logical_binary("or", self, other)

    def __ror__(self, other: Any) -> PredicateExpr:
        return _logical_binary("or", other, self)

    def __invert__(self) -> PredicateExpr:
        return PredicateExpr._new("not", (self,))

    def validate(
        self,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_stack: int = DEFAULT_MAX_STACK,
    ) -> bool:
        _measure(self, max_depth=max_depth, max_nodes=max_nodes, max_stack=max_stack)
        return True

    def measure(
        self,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_stack: int = DEFAULT_MAX_STACK,
    ) -> ExpressionStats:
        return _measure(self, max_depth=max_depth, max_nodes=max_nodes, max_stack=max_stack)

    def to_data(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "expression_type": "predicate",
            "root": _node_to_data(self),
        }

    canonical_identity = to_data
    inspect = to_data

    @classmethod
    def from_data(
        cls,
        data: Any,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        max_stack: int = DEFAULT_MAX_STACK,
    ) -> PredicateExpr:
        result = _expression_from_data(
            data, expected="predicate", max_depth=max_depth, max_nodes=max_nodes,
            max_stack=max_stack)
        if not isinstance(result, PredicateExpr):
            raise TypeError("decoded analytic expression is not a predicate")
        return result

    def same_as(self, other: Any) -> bool:
        return isinstance(other, PredicateExpr) and self.to_data() == other.to_data()

    def resolve_references(self, resolver: Any) -> PredicateExpr:
        """Authenticate every parameter leaf through one owning registry resolver."""

        if not callable(resolver):
            raise TypeError("analytic expression resolver must be callable")
        resolved = _resolve_references(self, resolver)
        if not isinstance(resolved, PredicateExpr):
            raise RuntimeError("analytic predicate reference resolution changed expression kind")
        return resolved

    @property
    def has_parameters(self) -> bool:
        return _has_parameter_leaf(self)

    def parameter_handles(self) -> tuple[Any, ...]:
        """Return parameter authorities in deterministic first-occurrence order."""

        return _parameter_handles(self)


Expression = ScalarExpr | PredicateExpr


def _finite_literal(value: Any, *, where: str = "analytic constant") -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal, Fraction)):
        raise TypeError("%s must be a finite real scalar, never bool or a callback" % where)
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("%s must be finite" % where) from exc
    if not math.isfinite(converted):
        raise ValueError("%s must be finite" % where)
    return 0.0 if converted == 0.0 else converted


def constant(value: Any) -> ScalarExpr:
    """Promote one finite real literal into an analytic scalar expression."""

    return ScalarExpr._new("constant", literal=_finite_literal(value))


def parameter(value: Any) -> ScalarExpr:
    """Return a symbolic scalar read of one exact :class:`ParamHandle`.

    This is deliberately a separate expression leaf: the handle itself remains immutable,
    hashable and Boolean-comparable.  Its registry resolves ownership before the expression enters
    a compiled plan, and bind substitutes the effective ``ResolvedBindings`` value as a native
    constant instruction.
    """

    from pops.model import ParamHandle

    if type(value) is not ParamHandle:
        raise TypeError(
            "analytic param(...) requires an exact ParamHandle, not %s"
            % type(value).__name__
        )
    return ScalarExpr._new("parameter", parameter=value)


def _program_input(value_id: Any, component: Any) -> ScalarExpr:
    """Internal constructor used by typed setup programs.

    This is intentionally absent from :mod:`pops.analytic`: a free input id has no authority.
    Only an owning program can issue one and later authenticate it during lowering.
    """

    return ScalarExpr._new("input", input_ref=_InputRef(value_id, component))


def _as_scalar(value: Any, *, where: str = "analytic scalar") -> ScalarExpr:
    if isinstance(value, ScalarExpr):
        return value
    if isinstance(value, PredicateExpr):
        raise TypeError("%s requires a ScalarExpr, not PredicateExpr" % where)
    return constant(value)


def _as_predicate(value: Any, *, where: str = "analytic logical operator") -> PredicateExpr:
    if not isinstance(value, PredicateExpr):
        raise TypeError("%s requires PredicateExpr operands; Python bool is not symbolic" % where)
    return value


def _coordinate(frame: Any, axis: Any) -> ScalarExpr:
    if not isinstance(axis, CartesianAxis):
        raise TypeError("coordinate axis must be a typed CartesianAxis, never a string")
    axes = getattr(frame, "axes", None)
    if not isinstance(axes, tuple) or not axes or any(
        not isinstance(item, CartesianAxis) for item in axes
    ):
        raise TypeError("coordinate frame must expose immutable typed Cartesian axes")
    if axis not in axes:
        raise ValueError("coordinate axis does not belong to the supplied frame")
    frame_id = getattr(frame, "canonical_id", None)
    if not isinstance(frame_id, str) or not frame_id:
        raise TypeError("coordinate frame must expose a stable canonical_id")
    projection = getattr(frame, "to_dict", None)
    if not callable(projection) or not isinstance(projection(), Mapping):
        raise TypeError("coordinate frame must expose canonical detached data")
    return ScalarExpr._new("coordinate", coordinate=_CoordinateRef(frame_id, axis))


def _scalar_unary(op: str, value: Any) -> ScalarExpr:
    return ScalarExpr._new(op, (_as_scalar(value),))


def _scalar_binary(op: str, left: Any, right: Any) -> ScalarExpr:
    return ScalarExpr._new(op, (_as_scalar(left), _as_scalar(right)))


def _comparison(op: str, left: Any, right: Any) -> PredicateExpr:
    return PredicateExpr._new(op, (_as_scalar(left), _as_scalar(right)))


def _logical_binary(op: str, left: Any, right: Any) -> PredicateExpr:
    return PredicateExpr._new(op, (_as_predicate(left), _as_predicate(right)))


def _where(predicate: Any, when_true: Any, when_false: Any) -> ScalarExpr:
    checked = _as_predicate(predicate, where="where predicate")
    return ScalarExpr._new(
        "where", (checked, _as_scalar(when_true), _as_scalar(when_false)))


def _between(value: Any, lower: Any, upper: Any) -> PredicateExpr:
    checked_value = _as_scalar(value)
    checked_lower = _as_scalar(lower)
    checked_upper = _as_scalar(upper)
    if checked_lower._op == "constant" and checked_upper._op == "constant" \
            and checked_lower._literal > checked_upper._literal:  # type: ignore[operator]
        raise ValueError("between lower bound must be <= upper bound")
    return PredicateExpr._new("between", (checked_value, checked_lower, checked_upper))


def _merged_frame_id(
    arguments: tuple[ScalarExpr | PredicateExpr, ...],
    coordinate: _CoordinateRef | None,
) -> str | None:
    frame_ids = {item._frame_id for item in arguments if item._frame_id is not None}
    if coordinate is not None:
        frame_ids.add(coordinate.frame_id)
    if len(frame_ids) > 1:
        raise ValueError("analytic expression cannot mix coordinates from different frames")
    return next(iter(frame_ids), None)


def _validate_scalar_local(value: ScalarExpr) -> None:
    if value._op not in _SCALAR_OPS:
        raise ValueError("unsupported analytic scalar operation %r" % value._op)
    if not isinstance(value._arguments, tuple):
        raise TypeError("analytic node arguments must be an immutable tuple")
    if value._op == "constant":
        if value._arguments or value._coordinate is not None or value._parameter is not None \
                or value._input is not None \
                or type(value._literal) is not float:
            raise TypeError("analytic constant node has an invalid shape")
        _finite_literal(value._literal)
    elif value._op == "coordinate":
        if value._arguments or value._literal is not None \
                or value._parameter is not None \
                or value._input is not None \
                or not isinstance(value._coordinate, _CoordinateRef):
            raise TypeError("analytic coordinate node has an invalid shape")
    elif value._op == "parameter":
        from pops.model import ParamHandle

        if value._arguments or value._literal is not None or value._coordinate is not None \
                or value._input is not None \
                or type(value._parameter) is not ParamHandle:
            raise TypeError("analytic parameter node has an invalid shape")
    elif value._op == "input":
        if value._arguments or value._literal is not None or value._coordinate is not None \
                or value._parameter is not None or not isinstance(value._input, _InputRef):
            raise TypeError("analytic input node has an invalid shape")
    else:
        if value._literal is not None or value._coordinate is not None \
                or value._parameter is not None or value._input is not None:
            raise TypeError(
                "analytic operator node cannot carry literal, coordinate or parameter metadata")
        expected = 1 if value._op in _SCALAR_UNARY_OPS else 2
        if value._op == "where":
            expected = 3
        if len(value._arguments) != expected:
            raise ValueError("analytic %s requires %d arguments" % (value._op, expected))
        if value._op == "where":
            if not isinstance(value._arguments[0], PredicateExpr) or any(
                not isinstance(item, ScalarExpr) for item in value._arguments[1:]
            ):
                raise TypeError("analytic where requires predicate, scalar, scalar")
        elif any(not isinstance(item, ScalarExpr) for item in value._arguments):
            raise TypeError("analytic scalar operators require scalar arguments")
        if value._op == "pow" and value._arguments[1]._op != "constant":  # type: ignore[union-attr]
            raise TypeError("analytic power exponent must be a finite scalar literal")
    if value._frame_id != _merged_frame_id(value._arguments, value._coordinate):
        raise ValueError("analytic scalar frame identity is inconsistent")


def _validate_predicate_local(value: PredicateExpr) -> None:
    if value._op not in _PREDICATE_OPS:
        raise ValueError("unsupported analytic predicate operation %r" % value._op)
    if not isinstance(value._arguments, tuple):
        raise TypeError("analytic node arguments must be an immutable tuple")
    if value._op in _COMPARISON_OPS:
        valid = len(value._arguments) == 2 and all(
            isinstance(item, ScalarExpr) for item in value._arguments)
    elif value._op in _LOGICAL_BINARY_OPS:
        valid = len(value._arguments) == 2 and all(
            isinstance(item, PredicateExpr) for item in value._arguments)
    elif value._op == "not":
        valid = len(value._arguments) == 1 and isinstance(value._arguments[0], PredicateExpr)
    else:
        valid = len(value._arguments) == 3 and all(
            isinstance(item, ScalarExpr) for item in value._arguments)
    if not valid:
        raise TypeError("analytic predicate %s has invalid argument kinds" % value._op)
    if value._op == "between":
        lower, upper = value._arguments[1:]
        if lower._op == "constant" and upper._op == "constant" \
                and lower._literal > upper._literal:  # type: ignore[operator]
            raise ValueError("between lower bound must be <= upper bound")
    if value._frame_id != _merged_frame_id(value._arguments, None):
        raise ValueError("analytic predicate frame identity is inconsistent")


def _positive_limit(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be a positive integer" % where)
    if value < 1:
        raise ValueError("%s must be >= 1" % where)
    return value


def _bounded_limit(value: Any, *, where: str, maximum: int) -> int:
    checked = _positive_limit(value, where=where)
    if checked > maximum:
        raise ValueError("%s must be <= %d" % (where, maximum))
    return checked


def _measure(
    root: Expression,
    *,
    max_depth: int,
    max_nodes: int,
    max_stack: int,
) -> ExpressionStats:
    checked_depth = _bounded_limit(
        max_depth, where="max_depth", maximum=DEFAULT_MAX_DEPTH)
    checked_nodes = _bounded_limit(
        max_nodes, where="max_nodes", maximum=DEFAULT_MAX_NODES)
    checked_stack = _bounded_limit(
        max_stack, where="max_stack", maximum=DEFAULT_MAX_STACK)
    count = 0
    observed_depth = 0
    active: set[int] = set()
    required_stacks: dict[int, int] = {}
    stack: list[tuple[bool, Expression, int]] = [(True, root, 1)]
    while stack:
        entering, node, depth = stack.pop()
        marker = id(node)
        if not entering:
            required = max(
                (index + required_stacks[id(child)]
                 for index, child in enumerate(node._arguments)),
                default=1,
            )
            required_stacks[marker] = required
            if required > checked_stack:
                raise ValueError("analytic expression exceeds max_stack=%d" % checked_stack)
            active.remove(marker)
            continue
        if marker in active:
            raise ValueError("analytic expression graph contains a cycle")
        if depth > checked_depth:
            raise ValueError("analytic expression exceeds max_depth=%d" % checked_depth)
        count += 1
        if count > checked_nodes:
            raise ValueError("analytic expression exceeds max_nodes=%d" % checked_nodes)
        observed_depth = max(observed_depth, depth)
        if isinstance(node, ScalarExpr):
            _validate_scalar_local(node)
        elif isinstance(node, PredicateExpr):
            _validate_predicate_local(node)
        else:
            raise TypeError("analytic graph contains an unsupported node")
        active.add(marker)
        stack.append((False, node, depth))
        stack.extend((True, child, depth + 1) for child in reversed(node._arguments))
    return ExpressionStats(
        node_count=count,
        depth=observed_depth,
        frame_id=root._frame_id,
        required_stack=required_stacks[id(root)],
    )


def _node_to_data(value: Expression) -> dict[str, Any]:
    if isinstance(value, ScalarExpr):
        if value._op == "constant":
            if value._literal is None:
                raise TypeError("analytic constant node is missing its binary64 literal")
            return {
                "kind": "scalar",
                "op": "constant",
                "value": {"binary64": value._literal.hex()},
            }
        if value._op == "coordinate":
            if value._coordinate is None:
                raise TypeError("analytic coordinate node is missing typed metadata")
            return {
                "kind": "scalar",
                "op": "coordinate",
                "frame_id": value._coordinate.frame_id,
                "axis": value._coordinate.axis.to_dict(),
            }
        if value._op == "parameter":
            if value._parameter is None:
                raise TypeError("analytic parameter node is missing its typed Handle")
            reference = (
                value._parameter.canonical_identity()
                if value._parameter.is_resolved
                else value._parameter.inspect()
            )
            return {
                "kind": "scalar",
                "op": "parameter",
                "reference": reference,
            }
        if value._op == "input":
            if value._input is None:
                raise TypeError("analytic input node is missing its typed reference")
            return {
                "kind": "scalar",
                "op": "input",
                "value_id": value._input.value_id,
                "component": value._input.component,
            }
        return {
            "kind": "scalar",
            "op": value._op,
            "arguments": [_node_to_data(argument) for argument in value._arguments],
        }
    return {
        "kind": "predicate",
        "op": value._op,
        "arguments": [_node_to_data(argument) for argument in value._arguments],
    }


@dataclass(slots=True)
class _DecodeBudget:
    max_depth: int
    max_nodes: int
    node_count: int = 0
    active: set[int] | None = None

    def __post_init__(self) -> None:
        self.max_depth = _bounded_limit(
            self.max_depth, where="max_depth", maximum=DEFAULT_MAX_DEPTH)
        self.max_nodes = _bounded_limit(
            self.max_nodes, where="max_nodes", maximum=DEFAULT_MAX_NODES)
        self.active = set()


def _expression_from_data(
    data: Any,
    *,
    expected: str,
    max_depth: int,
    max_nodes: int,
    max_stack: int,
) -> Expression:
    required = {"schema_version", "expression_type", "root"}
    if not isinstance(data, Mapping) or set(data) != required:
        raise TypeError("analytic expression data has an unsupported shape")
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ValueError("analytic expression data uses an unsupported schema version")
    if data["expression_type"] != expected:
        raise ValueError("analytic expression_type must be %r" % expected)
    budget = _DecodeBudget(max_depth=max_depth, max_nodes=max_nodes)
    result = _node_from_data(data["root"], expected=expected, budget=budget, depth=1)
    result.validate(max_depth=max_depth, max_nodes=max_nodes, max_stack=max_stack)
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "expression_type": expected,
        "root": _node_to_data(result),
    }
    if canonical != dict(data):
        raise ValueError("analytic expression data is not canonical")
    return result


def _node_from_data(
    data: Any,
    *,
    expected: str,
    budget: _DecodeBudget,
    depth: int,
) -> Expression:
    if not isinstance(data, Mapping):
        raise TypeError("analytic node data must be a mapping")
    if budget.active is None:
        raise RuntimeError("analytic decoder budget was not initialized")
    marker = id(data)
    if marker in budget.active:
        raise ValueError("analytic expression data contains a cycle")
    if depth > budget.max_depth:
        raise ValueError("analytic expression exceeds max_depth=%d" % budget.max_depth)
    budget.node_count += 1
    if budget.node_count > budget.max_nodes:
        raise ValueError("analytic expression exceeds max_nodes=%d" % budget.max_nodes)
    budget.active.add(marker)
    try:
        kind = data.get("kind")
        op = data.get("op")
        if kind != expected or not isinstance(op, str):
            raise ValueError("analytic node kind or operation is inconsistent")
        if expected == "scalar" and op == "constant":
            encoded = data.get("value")
            if set(data) != {"kind", "op", "value"} \
                    or not isinstance(encoded, Mapping) \
                    or set(encoded) != {"binary64"} \
                    or not isinstance(encoded["binary64"], str):
                raise TypeError(
                    "analytic constant data must contain one canonical binary64 payload")
            try:
                literal = float.fromhex(encoded["binary64"])
            except ValueError:
                raise ValueError("analytic constant binary64 payload is invalid") from None
            return ScalarExpr._new("constant", literal=_finite_literal(literal))
        if expected == "scalar" and op == "coordinate":
            if set(data) != {"kind", "op", "frame_id", "axis"}:
                raise TypeError("analytic coordinate data has an unsupported shape")
            axis = CartesianAxis.from_dict(data["axis"])
            return ScalarExpr._new(
                "coordinate", coordinate=_CoordinateRef(data["frame_id"], axis))
        if expected == "scalar" and op == "parameter":
            if set(data) != {"kind", "op", "reference"}:
                raise TypeError("analytic parameter data has an unsupported shape")
            from pops.model import Handle, ParamHandle

            reference = Handle.from_canonical_identity(data["reference"])
            if type(reference) is not ParamHandle or not reference.is_resolved:
                raise TypeError(
                    "analytic parameter data must contain a canonical ParamHandle")
            return ScalarExpr._new("parameter", parameter=reference)
        if expected == "scalar" and op == "input":
            if set(data) != {"kind", "op", "value_id", "component"}:
                raise TypeError("analytic input data has an unsupported shape")
            return ScalarExpr._new(
                "input", input_ref=_InputRef(data["value_id"], data["component"]))
        if set(data) != {"kind", "op", "arguments"} \
                or not isinstance(data["arguments"], list):
            raise TypeError("analytic operator data has an unsupported shape")
        raw_arguments = data["arguments"]
        if expected == "scalar":
            if op not in _SCALAR_OPS - {"constant", "coordinate", "parameter", "input"}:
                raise ValueError("unsupported analytic scalar operation %r" % op)
            child_kinds = (["predicate", "scalar", "scalar"] if op == "where"
                           else ["scalar"] * len(raw_arguments))
            if len(child_kinds) != len(raw_arguments):
                raise ValueError("analytic where requires three arguments")
            arguments = tuple(
                _node_from_data(item, expected=child_kind, budget=budget, depth=depth + 1)
                for item, child_kind in zip(raw_arguments, child_kinds, strict=True)
            )
            return ScalarExpr._new(op, arguments)
        if op not in _PREDICATE_OPS:
            raise ValueError("unsupported analytic predicate operation %r" % op)
        child_kinds = (["scalar", "scalar"] if op in _COMPARISON_OPS
                       else ["predicate", "predicate"] if op in _LOGICAL_BINARY_OPS
                       else ["predicate"] if op == "not"
                       else ["scalar", "scalar", "scalar"])
        if len(child_kinds) != len(raw_arguments):
            raise ValueError("analytic predicate %s has invalid arity" % op)
        arguments = tuple(
            _node_from_data(item, expected=child_kind, budget=budget, depth=depth + 1)
            for item, child_kind in zip(raw_arguments, child_kinds, strict=True)
        )
        return PredicateExpr._new(op, arguments)
    finally:
        budget.active.remove(marker)


def _resolve_references(value: Expression, resolver: Any) -> Expression:
    if isinstance(value, ScalarExpr):
        if value._op == "parameter":
            from pops.model import ParamHandle

            resolved = resolver(value._parameter)
            if type(resolved) is not ParamHandle or not resolved.is_resolved:
                raise TypeError(
                    "analytic parameter resolver must return an exact canonical ParamHandle")
            if resolved.param_kind != value._parameter.param_kind:
                raise ValueError("analytic parameter resolver changed the declared parameter kind")
            return ScalarExpr._new("parameter", parameter=resolved)
        if value._op in {"constant", "coordinate", "input"}:
            return value
        return ScalarExpr._new(
            value._op,
            tuple(_resolve_references(argument, resolver) for argument in value._arguments),
        )
    return PredicateExpr._new(
        value._op,
        tuple(_resolve_references(argument, resolver) for argument in value._arguments),
    )


def _has_parameter_leaf(value: Expression) -> bool:
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, ScalarExpr) and node._op == "parameter":
            return True
        stack.extend(node._arguments)
    return False


def _parameter_handles(value: Expression) -> tuple[Any, ...]:
    """Collect exact immutable parameter Handles without evaluating the expression."""

    from pops.model import ParamHandle

    ordered: list[ParamHandle] = []
    seen: set[ParamHandle] = set()
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, ScalarExpr) and node._op == "parameter":
            handle = node._parameter
            if type(handle) is not ParamHandle:
                raise TypeError("analytic parameter leaf does not carry an exact ParamHandle")
            if handle not in seen:
                seen.add(handle)
                ordered.append(handle)
        stack.extend(reversed(node._arguments))
    return tuple(ordered)


def _input_references(value: Expression) -> tuple[tuple[int, str], ...]:
    """Collect exact program-local input references without assigning storage slots."""

    ordered: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, ScalarExpr) and node._op == "input":
            reference = node._input
            if not isinstance(reference, _InputRef):
                raise TypeError("analytic input leaf does not carry an exact _InputRef")
            item = (reference.value_id, reference.component)
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        stack.extend(reversed(node._arguments))
    return tuple(ordered)


__all__ = [
    "AnalyticTruthValueError",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "DEFAULT_MAX_STACK",
    "ExpressionStats",
    "PredicateExpr",
    "SCHEMA_VERSION",
    "ScalarExpr",
    "_program_input",
    "parameter",
]
