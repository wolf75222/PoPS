"""Strict lowering of canonical analytic-expression data to native postfix programs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Real
from typing import Any


_SCALAR_UNARY = frozenset({"neg", "sqrt", "abs", "sin", "cos", "exp", "log"})
_SCALAR_BINARY = frozenset({
    "add", "sub", "mul", "div", "pow", "atan2", "hypot", "minimum", "maximum",
})
_COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_LOGICAL_BINARY = frozenset({"and", "or"})
_MAX_DEPTH = 64
_MAX_NODES = 4096


def _number(value: Any, *, where: str) -> float:
    if not isinstance(value, Mapping) or set(value) != {"binary64"} \
            or not isinstance(value["binary64"], str):
        raise TypeError("%s must be one canonical binary64 value" % where)
    try:
        result = float.fromhex(value["binary64"])
    except (OverflowError, ValueError):
        raise ValueError("%s contains an invalid binary64 payload" % where) from None
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % where)
    return result


def lower_analytic_components(
    components: Any,
    *,
    frame_id: str,
    bindings: Any = None,
) -> tuple[tuple[tuple[str, ...], tuple[float, ...]], ...]:
    """Return one validated postfix opcode/literal pair per scalar component."""

    if not isinstance(frame_id, str) or not frame_id:
        raise TypeError("analytic lowering requires a non-empty frame_id")
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence) \
            or not components:
        raise TypeError("analytic components must be a non-empty canonical sequence")
    return tuple(
        _lower_expression(
            expression,
            frame_id=frame_id,
            where="components[%d]" % index,
            bindings=bindings,
        )
        for index, expression in enumerate(components)
    )


def _lower_expression(
    data: Any,
    *,
    frame_id: str,
    where: str,
    bindings: Any,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    from pops.analytic import ScalarExpr

    try:
        data = ScalarExpr.from_data(data).to_data()
    except (TypeError, ValueError) as exc:
        raise type(exc)("%s: %s" % (where, exc)) from None
    opcodes: list[str] = []
    literals: list[float] = []
    budget = [0]
    _lower_node(
        data["root"], expected="scalar", frame_id=frame_id, where=where + ".root",
        depth=1, budget=budget, opcodes=opcodes, literals=literals,
        bindings=bindings,
    )
    if len(opcodes) != len(literals) or not opcodes:
        raise RuntimeError("analytic lowering produced an invalid postfix program")
    return tuple(opcodes), tuple(literals)


def _lower_node(
    data: Any,
    *,
    expected: str,
    frame_id: str,
    where: str,
    depth: int,
    budget: list[int],
    opcodes: list[str],
    literals: list[float],
    bindings: Any,
) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("%s exceeds analytic max_depth=%d" % (where, _MAX_DEPTH))
    budget[0] += 1
    if budget[0] > _MAX_NODES:
        raise ValueError("%s exceeds analytic max_nodes=%d" % (where, _MAX_NODES))
    if not isinstance(data, Mapping):
        raise TypeError("%s must be an analytic node mapping" % where)
    kind = data.get("kind")
    op = data.get("op")
    if kind != expected or not isinstance(op, str):
        raise ValueError("%s has an inconsistent analytic node kind" % where)

    if kind == "scalar" and op == "constant":
        if set(data) != {"kind", "op", "value"}:
            raise TypeError("%s constant node has an unsupported shape" % where)
        value = _number(data["value"], where=where + ".value")
        opcodes.append("constant")
        literals.append(value)
        return

    if kind == "scalar" and op == "coordinate":
        if set(data) != {"kind", "op", "frame_id", "axis"}:
            raise TypeError("%s coordinate node has an unsupported shape" % where)
        if data["frame_id"] != frame_id:
            raise ValueError("%s coordinate belongs to another frame" % where)
        from pops.frames import CartesianAxis

        axis = CartesianAxis.from_dict(data["axis"])
        opcodes.append(axis.name)
        literals.append(0.0)
        return

    if kind == "scalar" and op == "parameter":
        if set(data) != {"kind", "op", "reference"}:
            raise TypeError("%s parameter node has an unsupported shape" % where)
        opcodes.append("constant")
        literals.append(_bound_parameter(
            data["reference"], bindings=bindings, where=where + ".reference"))
        return

    if kind == "scalar" and op == "input":
        if set(data) != {"kind", "op", "value_id", "component"}:
            raise TypeError("%s input node has an unsupported shape" % where)
        value_id = data["value_id"]
        component = data["component"]
        if not isinstance(value_id, int) or isinstance(value_id, bool) or value_id < 0:
            raise TypeError("%s input value_id must be a non-negative integer" % where)
        if not isinstance(component, str) or not component:
            raise TypeError("%s input component must be non-empty text" % where)
        opcodes.append("input")
        literals.append(float(value_id))
        return

    if set(data) != {"kind", "op", "arguments"} \
            or not isinstance(data["arguments"], (tuple, list)):
        raise TypeError("%s operator node has an unsupported shape" % where)
    arguments = data["arguments"]
    if kind == "scalar":
        if op in _SCALAR_UNARY:
            expected_children = ("scalar",)
        elif op in _SCALAR_BINARY:
            expected_children = ("scalar", "scalar")
        elif op == "where":
            expected_children = ("predicate", "scalar", "scalar")
        else:
            raise ValueError("%s uses unsupported scalar op %r" % (where, op))
    elif op in _COMPARISONS:
        expected_children = ("scalar", "scalar")
    elif op in _LOGICAL_BINARY:
        expected_children = ("predicate", "predicate")
    elif op == "not":
        expected_children = ("predicate",)
    elif op == "between":
        expected_children = ("scalar", "scalar", "scalar")
    else:
        raise ValueError("%s uses unsupported predicate op %r" % (where, op))
    if len(arguments) != len(expected_children):
        raise ValueError(
            "%s op %r requires %d arguments" % (where, op, len(expected_children)))
    for index, (argument, child_kind) in enumerate(
        zip(arguments, expected_children, strict=True)
    ):
        _lower_node(
            argument, expected=child_kind, frame_id=frame_id,
            where="%s.arguments[%d]" % (where, index), depth=depth + 1, budget=budget,
            opcodes=opcodes, literals=literals,
            bindings=bindings,
        )
    # The canonical schema vocabulary is also the native ABI vocabulary.  Keeping one spelling
    # prevents the Python and C++ validators from accepting disjoint instruction sets.
    opcodes.append(op)
    literals.append(0.0)


def _bound_parameter(reference: Any, *, bindings: Any, where: str) -> float:
    """Authenticate one canonical parameter leaf and return its finite effective value."""

    from pops.model import Handle, ParamHandle
    from pops.model.bind_schema import BindSchema
    from pops.model.resolved_bindings import ResolvedBindings

    if type(bindings) is not ResolvedBindings or type(bindings.schema) is not BindSchema:
        raise TypeError(
            "%s requires exact ResolvedBindings from the compiled artifact" % where)
    handle = Handle.from_canonical_identity(reference)
    if type(handle) is not ParamHandle or not handle.is_resolved:
        raise TypeError("%s must identify one canonical ParamHandle" % where)
    try:
        slot = bindings.schema.slot(handle)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "%s is not authenticated by this artifact BindSchema: %s" % (where, exc)
        ) from None
    if slot.handle.canonical_identity() != handle.canonical_identity():
        raise ValueError("%s changed identity during BindSchema authentication" % where)
    try:
        value = bindings[slot.handle]
    except KeyError:
        raise ValueError(
            "%s has no effective value in ResolvedBindings" % where) from None
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal, Fraction)):
        raise TypeError("%s effective value must be a finite real scalar" % where)
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("%s effective value must be finite" % where) from None
    if not math.isfinite(result):
        raise ValueError("%s effective value must be finite" % where)
    return 0.0 if result == 0.0 else result


__all__ = ["lower_analytic_components"]
