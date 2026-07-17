"""Closed Bind-phase evaluator for canonical ``pops.expr.key.v1`` payloads."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from ._bind_schema_data import literal_value


def qualified_expression_key(value: Any, *, where: str) -> Any:
    """Serialize one Bind-time expression with exact parameter identities.

    The general symbolic CSE key deliberately keeps ``RuntimeParamRef`` model-local.  Detached
    runtime consumers cannot use that spelling: two blocks may instantiate the same parameter
    name with different values.  This narrower protocol therefore accepts only the operations the
    Bind evaluator executes and requires every parameter leaf to carry a resolved ``ParamHandle``.
    """
    from pops._ir.expr import Abs, Const, Neg, Sign, Sqrt, Var, _Bin
    from pops._ir.handle_expr import ValueExpr
    from pops._ir.values import RuntimeParamRef
    from pops.model.handles import ParamHandle

    def parameter_qid(handle: Any, *, leaf: str) -> str:
        if not isinstance(handle, ParamHandle) or not handle.is_resolved:
            raise TypeError(
                "%s %s must carry a resolved ParamHandle; use model.value(parameter) and "
                "resolve the expression through its owning Case" % (where, leaf)
            )
        return handle.qualified_id

    def walk(node: Any) -> Any:
        if isinstance(node, Const):
            literal = json.dumps(
                node.literal.to_data(), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            )
            handle = getattr(node, "handle", None)
            if handle is None:
                return ("const", literal)
            return ("param_const_qid",
                    parameter_qid(handle, leaf="constant parameter"), literal)
        if isinstance(node, RuntimeParamRef):
            return ("handle_value", parameter_qid(node.handle, leaf="runtime parameter"))
        if isinstance(node, ValueExpr):
            return ("handle_value", parameter_qid(node.handle, leaf="value"))
        if isinstance(node, Var):
            raise TypeError(
                "%s contains free-name Var(%r, %r); Bind-time consumers require an exact "
                "model.value(parameter) Handle leaf" % (where, node.name, node.kind)
            )
        if isinstance(node, (Neg, Sqrt, Abs, Sign)):
            return ({Neg: "neg", Sqrt: "sqrt", Abs: "abs", Sign: "sign"}[type(node)],
                    walk(node.a))
        if isinstance(node, _Bin):
            if node.op not in {"+", "-", "*", "/", "**", "==", "!=", "<", "<=", ">", ">="}:
                raise NotImplementedError(
                    "%s uses expression operation %r, which has no Bind-phase evaluator"
                    % (where, node.op)
                )
            return (node.op, (walk(node.a), walk(node.b)))
        raise NotImplementedError(
            "%s uses expression node %s, which has no Bind-phase evaluator"
            % (where, type(node).__name__)
        )

    return walk(value)


def eval_expression_key(value: Any, env: Mapping[str, Any], *, where: str) -> Any:
    if not isinstance(value, (list, tuple)) or not value or not isinstance(value[0], str):
        raise TypeError("%s has an invalid pops.expr.key.v1 node" % where)
    op = value[0]
    if op == "const" and len(value) == 2 and isinstance(value[1], str):
        try:
            literal = json.loads(value[1])
        except json.JSONDecodeError as exc:
            raise TypeError("%s contains an invalid const literal" % where) from exc
        return literal_value(literal, where="%s const" % where)
    if (op == "param_const" and len(value) == 3
            and isinstance(value[1], str) and isinstance(value[2], str)):
        try:
            literal = literal_value(json.loads(value[2]), where="%s const" % where)
        except json.JSONDecodeError as exc:
            raise TypeError("%s contains an invalid param const literal" % where) from exc
        declared = _dependency(env, value[1], where=where)
        if literal != declared:
            raise ValueError(
                "%s parameter constant %r disagrees with dependency value %r"
                % (where, literal, declared)
            )
        return declared
    if (op == "param_const_qid" and len(value) == 3
            and isinstance(value[1], str) and isinstance(value[2], str)):
        try:
            literal = literal_value(json.loads(value[2]), where="%s const" % where)
        except json.JSONDecodeError as exc:
            raise TypeError("%s contains an invalid param const literal" % where) from exc
        declared = _dependency(env, value[1], where=where)
        if literal != declared:
            raise ValueError(
                "%s parameter constant %r disagrees with dependency value %r"
                % (where, literal, declared)
            )
        return declared
    if op == "var" and len(value) == 3:
        kind, name = value[1], value[2]
        if kind != "param" or not isinstance(name, str):
            raise TypeError(
                "%s may read only declared parameter dependencies at Bind phase, got Var(%r, %r)"
                % (where, name, kind)
            )
        return _dependency(env, name, where=where)
    if op in ("rparam", "handle_value") and len(value) == 2 and isinstance(value[1], str):
        # RuntimeParamRef is emitted by Module.value(handle); ValueExpr retains a handle qid.
        # In both cases depends_on authenticated the exact slot before this evaluator runs.
        return _dependency(env, value[1], where=where)
    if op in ("neg", "sqrt", "abs", "sign") and len(value) == 2:
        item = eval_expression_key(value[1], env, where=where)
        if op == "neg":
            return -item
        if op == "sqrt":
            return math.sqrt(item)
        if op == "abs":
            return abs(item)
        return (item > 0) - (item < 0)
    if op in ("+", "-", "*", "/", "**", "==", "!=", "<", "<=", ">", ">="):
        if len(value) != 2 or not isinstance(value[1], (list, tuple)) or len(value[1]) != 2:
            raise TypeError("%s binary node %r has an invalid shape" % (where, op))
        left = eval_expression_key(value[1][0], env, where=where)
        right = eval_expression_key(value[1][1], env, where=where)
        return {
            "+": lambda: left + right, "-": lambda: left - right,
            "*": lambda: left * right, "/": lambda: left / right,
            "**": lambda: left ** right, "==": lambda: left == right,
            "!=": lambda: left != right, "<": lambda: left < right,
            "<=": lambda: left <= right, ">": lambda: left > right,
            ">=": lambda: left >= right,
        }[op]()
    raise NotImplementedError(
        "%s uses expression operation %r, which has no Bind-phase evaluator" % (where, op)
    )


def expression_reference_keys(value: Any, *, where: str) -> frozenset[tuple[str, str]]:
    """Collect parameter leaves from one canonical structural expression key."""
    found: set[tuple[str, str]] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, (list, tuple)) or not node or not isinstance(node[0], str):
            raise TypeError("%s has an invalid pops.expr.key.v1 node" % where)
        op = node[0]
        if op == "var" and len(node) == 3:
            if node[1] == "param" and isinstance(node[2], str):
                found.add(("local", node[2]))
            return
        if op == "rparam" and len(node) == 2 and isinstance(node[1], str):
            found.add(("local", node[1]))
            return
        if op == "handle_value" and len(node) == 2 and isinstance(node[1], str):
            found.add(("qid", node[1]))
            return
        if op == "param_const" and len(node) == 3 and isinstance(node[1], str):
            found.add(("local", node[1]))
            return
        if op == "param_const_qid" and len(node) == 3 and isinstance(node[1], str):
            found.add(("qid", node[1]))
            return
        if op == "const" and len(node) == 2:
            return
        if op in ("neg", "sqrt", "abs", "sign") and len(node) == 2:
            walk(node[1])
            return
        if op in ("+", "-", "*", "/", "**", "==", "!=", "<", "<=", ">", ">="):
            if len(node) != 2 or not isinstance(node[1], (list, tuple)) or len(node[1]) != 2:
                raise TypeError("%s binary node %r has an invalid shape" % (where, op))
            walk(node[1][0])
            walk(node[1][1])
            return
        raise NotImplementedError(
            "%s uses expression operation %r, whose parameter references cannot be authenticated"
            % (where, op)
        )

    walk(value)
    return frozenset(found)


def _dependency(env: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in env:
        raise KeyError("%s dependency %r is unavailable" % (where, key))
    return env[key]


__all__ = ["eval_expression_key", "expression_reference_keys", "qualified_expression_key"]
