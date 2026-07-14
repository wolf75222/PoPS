"""pops._ir.visitors -- pure-symbolic tree traversal helpers.

Originally in pops.dsl.

  _children(e)              -- children of an Expr node (for traversal / CSE discovery)
  _expr_uses_cons_or_prim(e) -- True if the tree references a cons or prim Var
  _key(e)                   -- structural CSE key of a node
"""
from __future__ import annotations

import json
from typing import Any

from .expr import Const, Expr, Var, _Bin, Neg, Sqrt, Abs, Sign
from .values import EigWitness, StateRef, RuntimeParamRef


def _children(e: Any) -> Any:
    protocol = getattr(e, "__pops_ir_children__", None)
    if callable(protocol):
        children = protocol()
        if children is not NotImplemented:
            children = tuple(children)
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


def _key(e: Any) -> Any:
    protocol = getattr(e, "__pops_ir_key__", None)
    if callable(protocol):
        key = protocol(_key)
        if key is not NotImplemented:
            return key
    if isinstance(e, Const):
        literal = json.dumps(e.literal.to_data(), sort_keys=True, separators=(",", ":"))
        if getattr(e, "handle", None) is not None:
            return ("param_const", e.handle.local_id, literal)
        return ("const", literal)
    if isinstance(e, RuntimeParamRef):
        return ("rparam", e.name)  # key = name: two refs to the same runtime param share the CSE local
    if isinstance(e, Var):
        # Conservative/primitive/aux namespaces can legally reuse a display name;
        # the declared kind is part of symbolic identity, not presentation metadata.
        return ("var", e.kind, e.name)
    if isinstance(e, Neg):
        return ("neg", _key(e.a))
    if isinstance(e, Sqrt):
        return ("sqrt", _key(e.a))
    if isinstance(e, Abs):
        return ("abs", _key(e.a))
    if isinstance(e, Sign):
        return ("sign", _key(e.a))
    if isinstance(e, EigWitness):
        # cle = (field, taille, cles des entrees) : deux temoins de la MEME matrice partagent une locale.
        # Un PREDICAT ajoute im_tol a la cle (verdict different a seuil different) ; le chemin scalaire
        # garde sa cle a 4 elements -> CSE et brique bit-identiques a l'historique.
        if e.is_predicate():
            return ("eig", e.field, e.k, e.im_tol, tuple(_key(c) for c in e.entries()))
        return ("eig", e.field, e.k, tuple(_key(c) for c in e.entries()))
    if isinstance(e, StateRef):
        return ("state", e.side, _key(e.expr))  # defensive: the Roe lines do not go through CSE
    if isinstance(e, _Bin):
        return (e.op, tuple(_key(c) for c in _children(e)))
    raise TypeError(
        "Expr extension %s has no structural CSE key; implement "
        "__pops_ir_key__(recurse)" % type(e).__name__)
