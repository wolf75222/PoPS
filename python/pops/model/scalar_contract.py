"""Model-owned symbolic contracts consumed by source-authored numerical policies.

Policies declare a mathematical body against these immutable model expressions;
they do not select compiler IR implementations or depend on the physics facade.
"""
from pops._ir.expr import Const, Expr, Var, _wrap as scalar_expression, is_scalar_expression
from pops._ir.native_call import NativeCall
from pops._ir.values import RuntimeParamRef
from pops._ir.vector_expr import VectorExpr
from pops._ir.visitors import _children as scalar_children
from .handles import StateShapeHandle


def state_component_count(state):
    if not isinstance(state, StateShapeHandle):
        raise TypeError("requires an exact model StateHandle")
    return len(state.state_components)


__all__ = ["Const", "Expr", "Var", "NativeCall", "RuntimeParamRef", "VectorExpr",
           "scalar_children", "scalar_expression", "is_scalar_expression", "state_component_count"]
