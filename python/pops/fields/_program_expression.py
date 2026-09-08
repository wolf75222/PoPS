"""Bounded, owner-authenticated scalar expressions for native field load evaluation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.identity import canonical_bytes
from pops.identity.scalar import ScalarLiteral


def field_input_contract(states: Any) -> tuple[tuple[Any, tuple[str, ...]], ...]:
    from pops.time.references import canonical_handle

    rows = []
    for state in states:
        if getattr(state, "vtype", None) != "state":
            raise TypeError("field expression inputs must be exact Program State values")
        handle = canonical_handle(state.state_ref)
        components = tuple(getattr(state.space, "components", ()))
        if handle.kind != "state" or not handle.is_instance or not components:
            raise ValueError("field expression inputs require qualified typed state storage")
        rows.append((handle, components))
    if len({canonical_bytes(handle.canonical_identity()) for handle, _ in rows}) != len(rows):
        raise ValueError("field expression has ambiguous repeated state bindings")
    return tuple(rows)


def decode_field_literal(data: Any) -> ScalarLiteral:
    if not isinstance(data, Mapping) or data.get("kind") not in (
            "integer", "rational", "decimal", "binary64"):
        raise TypeError("field expression requires an exact numeric literal without raw C++")
    kind = data["kind"]
    if kind == "integer":
        payload = int(data["value"])
    elif kind == "rational":
        payload = (int(data["numerator"]), int(data["denominator"]))
    else:
        payload = data["value"]
    literal = ScalarLiteral(kind, payload, data.get("unit"), data.get("target"))
    if canonical_bytes(literal.to_data()) != canonical_bytes(dict(data)):
        raise ValueError("field expression literal is not canonical")
    return literal


def encode_field_expression(expression: Any, states: Any) -> tuple:
    """Encode only explicit qualified component reads and bounded scalar arithmetic."""
    from pops._ir.expr import Abs, Add, Const, Div, Mul, Neg, Pow, Sqrt, Sub, _wrap
    from pops._ir.handle_expr import ValueExpr
    from pops._ir.quantity import QuantityRef

    rows = field_input_contract(states)
    binary = {Add: "add", Sub: "sub", Mul: "mul", Div: "div", Pow: "pow"}
    unary = {Neg: "neg", Sqrt: "sqrt", Abs: "abs"}

    def encode(value: Any) -> tuple:
        value = _wrap(value)
        if type(value) is Const:
            return ("literal", decode_field_literal(value.literal.to_data()).to_data())
        if type(value) in (QuantityRef, ValueExpr):
            if not value.handle.is_resolved:
                raise ValueError("field expression must resolve its scientific input handles")
            matches = [(index, handle, components) for index, (handle, components) in enumerate(rows)
                       if handle == value.handle]
            if len(matches) != 1:
                raise ValueError("field expression read has no exact qualified State binding")
            index, handle, components = matches[0]
            if type(value) is ValueExpr:
                if len(components) != 1:
                    raise ValueError("a field load must select a component of a multi-component State")
                component = 0
            else:
                if tuple(value.space.components) != components:
                    raise ValueError("field expression component space differs from its State binding")
                component = components.index(value.component)
            return ("input", index, component, handle.canonical_identity())
        if type(value) in binary:
            return (binary[type(value)], encode(value.a), encode(value.b))
        if type(value) in unary:
            return (unary[type(value)], encode(value.a))
        raise NotImplementedError("field expression has unsupported node %s" % type(value).__name__)

    return encode(expression)


def field_expression_cpp(expression: Any, states: Any, *, views: tuple[str, ...]) -> tuple[str, tuple[Any, ...]]:
    """Re-authenticate the closed AST and return native code plus its actual declaration reads."""
    rows = field_input_contract(states)
    if len(views) != len(rows):
        raise ValueError("field expression input view arity changed")
    reads = {}
    binary = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

    def emit(node: Any) -> str:
        if not isinstance(node, (tuple, list)) or not node or not isinstance(node[0], str):
            raise TypeError("field expression requires a closed scalar AST")
        op = node[0]
        if op == "literal" and len(node) == 2:
            return "static_cast<pops::Real>(%s)" % decode_field_literal(node[1]).to_cpp()
        if op == "input" and len(node) == 4:
            index, component, data = node[1:]
            if type(index) is not int or index < 0 or index >= len(rows):
                raise ValueError("field expression input index is outside its declared State tuple")
            handle, components = rows[index]
            if type(component) is not int or not 0 <= component < len(components):
                raise ValueError("field expression component is outside its declared State space")
            if canonical_bytes(data) != canonical_bytes(handle.canonical_identity()):
                raise ValueError("field expression input identity changed after encoding")
            reads[handle.qualified_id] = handle
            return "%s(index, %d)" % (views[index], component)
        if op in binary and len(node) == 3:
            return "(%s %s %s)" % (emit(node[1]), binary[op], emit(node[2]))
        if op == "pow" and len(node) == 3:
            return "std::pow(%s, %s)" % (emit(node[1]), emit(node[2]))
        if op in ("neg", "sqrt", "abs") and len(node) == 2:
            value = emit(node[1])
            return "(-%s)" % value if op == "neg" else "std::%s(%s)" % (
                "fabs" if op == "abs" else "sqrt", value)
        raise ValueError("field expression has an unsupported operation or arity")

    result = emit(expression)
    return result, tuple(reads[key] for key in sorted(reads))


def field_expression_dependencies(expressions: Any, states: Any) -> tuple[Any, ...]:
    """Return actual encoded reads in their exact State input order."""
    rows = field_input_contract(states)
    views = tuple("input%d" % index for index in range(len(rows)))
    read_ids = set()
    for expression in expressions:
        _, reads = field_expression_cpp(expression, states, views=views)
        read_ids.update(handle.qualified_id for handle in reads)
    return tuple(handle.canonical_identity() for handle, _ in rows if handle.qualified_id in read_ids)


__all__ = ["decode_field_literal", "encode_field_expression", "field_expression_cpp",
           "field_expression_dependencies", "field_input_contract"]
