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

from .expr import Const, Expr, Var, _Bin, Neg, Sqrt, Abs, Sign
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
    if isinstance(e, (Neg, Sqrt, Abs, Sign)):
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
