"""pops._ir.visitors -- pure-symbolic tree traversal helpers.

Originally in pops.dsl.

  _children(e)              -- children of an Expr node (for traversal / CSE discovery)
  _expr_uses_cons_or_prim(e) -- True if the tree references a cons or prim Var
  _key(e)                   -- structural CSE key of a node
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, cast

from .expr import Const, Expr, Var, _Bin, Neg, Sqrt, Exp, Abs, Sign
from .values import EigWitness, StateRef, RuntimeParamRef


def _children(e: Any) -> Any:
    protocol = getattr(e, "__pops_ir_children__", None)
    if callable(protocol):
        children = protocol()
        if children is not NotImplemented:
            if not isinstance(children, Iterable):
                raise TypeError(
                    "%s.__pops_ir_children__() must return an iterable of Expr nodes"
                    % type(e).__name__)
            children = tuple(cast(Iterable[Any], children))
            if any(not isinstance(child, Expr) for child in children):
                raise TypeError(
                    "%s.__pops_ir_children__() must return only Expr nodes"
                    % type(e).__name__)
            return children
    if isinstance(e, _Bin):
        return (e.a, e.b)
    if isinstance(e, (Neg, Sqrt, Exp, Abs, Sign)):
        return (e.a,)
    if isinstance(e, EigWitness):
        return tuple(e.entries())  # entrees de la matrice : enfants pour CSE / decouverte deps
    if isinstance(e, StateRef):
        return (e.expr,)  # left/right marker: a single child (discovery of runtime params, etc.)
    return ()


def _expr_uses_cons_or_prim(e: Any) -> bool:
    """True if the expression tree references a conservative or primitive Var. Tests the Var KIND, so
    the answer does not depend on declaration order. Used to enforce that linear_source coefficients
    are linear in U: a coefficient depending on U or a primitive is not a constant matrix entry."""
    stack = [e]
    while stack:
        node = stack.pop()
        if isinstance(node, Var) and node.kind in ("cons", "prim"):
            return True
        stack.extend(_children(node))
    return False


def _dependencies(exprs: Any) -> set[str]:
    """Collect symbolic environment names from one or more Expr DAG roots in linear time."""

    roots = (exprs,) if isinstance(exprs, Expr) else tuple(exprs)
    result: set[str] = set()
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Var):
            result.add(node.name)
            continue
        children = _children(node)
        if children:
            stack.extend(children)
        else:
            deps = node.deps()
            if not isinstance(deps, set) or any(not isinstance(name, str) for name in deps):
                raise TypeError("Expr dependencies must be a set of strings")
            result.update(deps)
    return result


def _key(e: Any, _memo: dict[int, Any] | None = None) -> Any:
    """Return the structural key of *e*, memoizing shared DAG nodes when requested.

    The optional memo is compiler-internal; omitting it preserves the public helper's historical
    result.  Sharing one memo across a traversal avoids rebuilding every descendant key at each
    parent of a large expression DAG.
    """
    memo = {} if _memo is None else _memo
    cached = memo.get(id(e))
    if cached is not None:
        return cached

    def recurse(child: Any) -> Any:
        return _key(child, memo)

    protocol = getattr(e, "__pops_ir_key__", None)
    if callable(protocol):
        key = protocol(recurse)
        if key is not NotImplemented:
            memo[id(e)] = key
            return key
    if isinstance(e, Const):
        literal = json.dumps(e.literal.to_data(), sort_keys=True, separators=(",", ":"))
        if getattr(e, "handle", None) is not None:
            key = ("param_const", e.handle.local_id, literal)
        else:
            key = ("const", literal)
    elif isinstance(e, RuntimeParamRef):
        key = ("rparam", e.name)  # key = name: two refs to the same runtime param share the CSE local
    elif isinstance(e, Var):
        # Conservative/primitive/aux namespaces can legally reuse a display name;
        # the declared kind is part of symbolic identity, not presentation metadata.
        key = ("var", e.kind, e.name)
    elif isinstance(e, Neg):
        key = ("neg", recurse(e.a))
    elif isinstance(e, Sqrt):
        key = ("sqrt", recurse(e.a))
    elif isinstance(e, Exp):
        key = ("exp", recurse(e.a))
    elif isinstance(e, Abs):
        key = ("abs", recurse(e.a))
    elif isinstance(e, Sign):
        key = ("sign", recurse(e.a))
    elif isinstance(e, EigWitness):
        # cle = (field, taille, cles des entrees) : deux temoins de la MEME matrice partagent une locale.
        # Un PREDICAT ajoute im_tol a la cle (verdict different a seuil different) ; le chemin scalaire
        # garde sa cle a 4 elements -> CSE et brique bit-identiques a l'historique.
        if e.is_predicate():
            key = ("eig", e.field, e.k, e.im_tol, tuple(recurse(c) for c in e.entries()))
        else:
            key = ("eig", e.field, e.k, tuple(recurse(c) for c in e.entries()))
    elif isinstance(e, StateRef):
        key = ("state", e.side, recurse(e.expr))  # defensive: Roe lines do not go through CSE
    elif isinstance(e, _Bin):
        key = (e.op, tuple(recurse(c) for c in _children(e)))
    else:
        raise TypeError(
            "Expr extension %s has no structural CSE key; implement "
            "__pops_ir_key__(recurse)" % type(e).__name__)
    memo[id(e)] = key
    return key


def _dag_key_ids(exprs: Any) -> tuple[dict[int, int], tuple[Any, ...], tuple[int, ...]]:
    """Intern an Expr DAG into compact structural ids in deterministic post-order.

    Equal subexpressions receive the same integer even when authored as distinct Python objects.
    Descriptors contain child ids rather than nested child tuples, so hashing and serialisation stay
    linear in the number of distinct symbolic nodes.
    """

    roots = (exprs,) if isinstance(exprs, Expr) else tuple(exprs)
    object_ids: dict[int, int] = {}
    interned: dict[Any, int] = {}
    descriptors: list[Any] = []

    def visit(e: Any) -> int:
        known = object_ids.get(id(e))
        if known is not None:
            return known

        protocol = getattr(e, "__pops_ir_key__", None)
        descriptor = NotImplemented
        if callable(protocol):
            descriptor = protocol(visit)
        if descriptor is NotImplemented:
            if isinstance(e, Const):
                literal = json.dumps(
                    e.literal.to_data(), sort_keys=True, separators=(",", ":"))
                descriptor = (
                    ("param_const", e.handle.local_id, literal)
                    if getattr(e, "handle", None) is not None else ("const", literal))
            elif isinstance(e, RuntimeParamRef):
                descriptor = ("rparam", e.name)
            elif isinstance(e, Var):
                descriptor = ("var", e.kind, e.name)
            elif isinstance(e, Neg):
                descriptor = ("neg", visit(e.a))
            elif isinstance(e, Sqrt):
                descriptor = ("sqrt", visit(e.a))
            elif isinstance(e, Exp):
                descriptor = ("exp", visit(e.a))
            elif isinstance(e, Abs):
                descriptor = ("abs", visit(e.a))
            elif isinstance(e, Sign):
                descriptor = ("sign", visit(e.a))
            elif isinstance(e, EigWitness):
                entries = tuple(visit(child) for child in e.entries())
                descriptor = (
                    ("eig", e.field, e.k, e.im_tol, entries)
                    if e.is_predicate() else ("eig", e.field, e.k, entries))
            elif isinstance(e, StateRef):
                descriptor = ("state", e.side, visit(e.expr))
            elif isinstance(e, _Bin):
                descriptor = (e.op, tuple(visit(child) for child in _children(e)))
            else:
                raise TypeError(
                    "Expr extension %s has no structural CSE key; implement "
                    "__pops_ir_key__(recurse)" % type(e).__name__)
        try:
            node_id = interned.get(descriptor)
        except TypeError:
            raise TypeError(
                "Expr extension %s returned an unhashable structural key" % type(e).__name__
            ) from None
        if node_id is None:
            node_id = len(descriptors)
            interned[descriptor] = node_id
            descriptors.append(descriptor)
        object_ids[id(e)] = node_id
        return node_id

    root_ids = tuple(visit(root) for root in roots)
    return object_ids, tuple(descriptors), root_ids


def _dag_key_data(exprs: Any) -> dict[str, Any]:
    """Return a compact, stable JSON-oriented structural identity for Expr roots."""

    _, nodes, roots = _dag_key_ids(exprs)
    return {"protocol": "pops.expr.dag.v1", "nodes": nodes, "roots": roots}


def validate_dag_key_data(graph, *, root_count=None):
    """Validate inert structural Expr DAG data, without reconstructing/evaluating IR.

    The wire descriptors are emitted above in deterministic post-order. Child references
    therefore point strictly backwards; metadata is never mistaken for a child index.
    """
    from collections.abc import Mapping
    import math

    def fail():
        raise ValueError("invalid canonical expression DAG descriptor")

    def seq(value):
        return type(value) in (tuple, list)

    def text(value):
        return type(value) is str and bool(value)

    def inert(value):
        if value is None or type(value) in (str, int, bool):
            return
        if type(value) is float and math.isfinite(value):
            return
        if seq(value):
            for item in value:
                inert(item)
            return
        if type(value) is dict and all(type(key) is str for key in value):
            for item in value.values():
                inert(item)
            return
        fail()

    if not isinstance(graph, Mapping) or set(graph) != {"protocol", "nodes", "roots"} or graph["protocol"] != "pops.expr.dag.v1":
        fail()
    nodes, roots = graph["nodes"], graph["roots"]
    if not seq(nodes) or not nodes or not seq(roots) or not roots:
        fail()
    if root_count is not None and len(roots) != root_count:
        fail()
    edges = []
    descriptors = set()

    def handle_key(value):
        if not seq(value) or not value: fail()
        if value[0] == "qualified":
            if len(value) != 2 or not text(value[1]): fail()
        elif value[0] == "local":
            if len(value) != 4 or not text(value[1]) or not text(value[2]) or (value[3] is not None and not text(value[3])): fail()
        else: fail()

    binary = {"+", "-", "*", "/", "**", "minimum", "maximum", "==", "!=", "<", "<=", ">", ">=", "&", "|"}
    unary = {"neg", "sqrt", "exp", "abs", "sign", "boolean_not"}
    for index, node in enumerate(nodes):
        if not seq(node) or not node or not text(node[0]):
            fail()
        inert(node)
        descriptor = json.dumps(node, sort_keys=True, separators=(",", ":"))
        if descriptor in descriptors: fail()
        descriptors.add(descriptor)
        op = node[0]
        children = []

        def ref(value):
            if type(value) is not int or not 0 <= value < index:
                fail()
            children.append(value)

        def refs(values):
            if not seq(values):
                fail()
            for value in values:
                ref(value)

        def arity(size):
            if len(node) != size:
                fail()

        if op in binary:
            arity(2)
            if not seq(node[1]) or len(node[1]) != 2:
                fail()
            refs(node[1])
        elif op in unary:
            arity(2); ref(node[1])
        elif op == "where":
            arity(4); refs(node[1:])
        elif op == "rounded":
            arity(3)
            if node[1] != "binary64": fail()
            ref(node[2])
        elif op in {"const", "param_const"}:
            arity(2 if op == "const" else 3)
            if op == "param_const" and not text(node[1]): fail()
            if not text(node[-1]): fail()
            try:
                from pops.identity.scalar import ScalarLiteral
                literal = json.loads(node[-1])
                if type(literal) is not dict or type(literal.get("kind")) is not str: fail()
                kind = literal["kind"]
                if kind not in {"integer", "rational", "decimal", "binary64", "algebraic"}: fail()
                required = {"kind", "numerator", "denominator"} if kind == "rational" else {"kind", "value"}
                if not required <= set(literal) or set(literal) - required - {"unit", "target", "cpp"}: fail()
                payload = ((int(literal["numerator"]), int(literal["denominator"])) if kind == "rational"
                           else int(literal["value"]) if kind == "integer" else literal["value"])
                restored = ScalarLiteral(kind, payload, **{key: literal[key] for key in ("unit", "target", "cpp") if key in literal})
                if restored.to_data() != literal: fail()
            except (TypeError, ValueError, KeyError):
                fail()
        elif op in {"rparam", "handle_value", "unknown", "field_logical_time", "temporal_tau@1", "method_coefficient"}:
            arity(2)
            if not text(node[1]): fail()
        elif op == "var":
            arity(3)
            if not all(text(value) for value in node[1:]): fail()
        elif op == "quantity":
            arity(4)
            if not seq(node[1]) or not seq(node[3]) or type(node[2]) is not int or node[2] < 0: fail()
            handle_key(node[1]); inert(node[3])
            if len(node[3]) < 3 or not seq(node[3][2]) or node[2] >= len(node[3][2]): fail()
        elif op == "state":
            arity(3)
            if not text(node[1]): fail()
            ref(node[2])
        elif op == "eig":
            if len(node) not in (4, 5) or not text(node[1]) or type(node[2]) is not int or node[2] < 0: fail()
            if len(node) == 5 and (type(node[3]) not in (int, float) or not math.isfinite(node[3])): fail()
            refs(node[-1])
        elif op in {"partial", "gradient"}:
            arity(4 if op == "partial" else 3)
            if type(node[1]) is int: ref(node[1])
            elif seq(node[1]): handle_key(node[1])
            else: fail()
            if op == "partial" and type(node[2]) is not int: fail()
            if not text(node[-1]): fail()
        elif op == "gradient_magnitude":
            arity(3); inert(node[1]); ref(node[2])
        elif op in {"boundary_value", "interior_trace"}:
            arity(3)
            if not text(node[1]) or type(node[2]) is not int or node[2] < 0: fail()
        elif op in {"program_component", "program_scalar", "program_global"}:
            arity(4 if op == "program_component" else 3)
            if type(node[1]) is not str or type(node[2]) is not int or node[2] < 0: fail()
            if len(node) == 4 and (type(node[3]) is not int or node[3] < 0): fail()
        elif op == "physical_global.v1":
            arity(3); handle_key(node[1]); inert(node[2])
        elif op in {"application_projection", "native_projection", "finite_projection_v1"}:
            arity(3 if op == "finite_projection_v1" else 4)
            ref(node[1])
            if op != "finite_projection_v1" and not text(node[2]): fail()
            if type(node[-1]) is not int or node[-1] < 0: fail()
        elif op == "rate_application_projection":
            arity(5); ref(node[1])
            if not text(node[2]): fail()
            inert(node[3]); inert(node[4])
        elif op in {"operator_application", "native_call"}:
            arity(7 if op == "operator_application" else 5)
            inert(node[1])
            inputs = node[2] if op == "operator_application" else node[3]
            if not seq(inputs): fail()
            for row in inputs: refs(row)
            if op == "operator_application":
                if not seq(node[3]): fail()
                for row in node[3]:
                    if not seq(row) or len(row) != 2 or not text(row[0]): fail()
                    refs(row[1])
                for value in node[4:]: inert(value)
            else:
                inert(node[2]); inert(node[4])
        elif op == "finite_linear_v1":
            arity(6)
            from .finite_linear import _support
            if node[1] not in ("apply", "solve"): fail()
            try:
                source, target = _support(node[2]), _support(node[3])
            except (TypeError, ValueError): fail()
            if not seq(node[4]) or len(node[4]) != len(target[1]) or any(not seq(row) or len(row) != len(source[1]) for row in node[4]): fail()
            if any(type(value) not in (int, float) or not math.isfinite(value) for row in node[4] for value in row): fail()
            if not seq(node[5]) or len(node[5]) != len(source[1]): fail()
            refs(node[5])
        else:
            fail()
        edges.append(children)
    for root in roots:
        if type(root) is not int or not 0 <= root < len(nodes): fail()
    reachable, pending = set(), list(roots)
    while pending:
        index = pending.pop()
        if index not in reachable:
            reachable.add(index); pending.extend(edges[index])
    if len(reachable) != len(nodes):
        fail()
