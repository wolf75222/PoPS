"""Typed mathematical expression surface for model and numerical library authors.

These aliases preserve the existing language primitives and transformations exactly;
this declaration layer supplies no model, numerical recipe or execution authority.
"""
from pops._ir.application import substitute_quantities
from pops._ir.balance import _handle_data, source_balance_supported
from pops._ir.expr import Const, Expr, Var, _wrap
from pops._ir.expr_references import collect_reference_value, resolve_reference_value
from pops._ir.lowering import diff
from pops._ir.quantity import QuantityRef, local_expression_identity
from pops._ir.values import RuntimeParamRef
from pops._ir.visitors import _children

EXPRESSION_LANGUAGE_VERSION = 1
__all__ = (
    "Const", "Expr", "Var", "QuantityRef", "RuntimeParamRef", "_wrap", "_children",
    "substitute_quantities", "diff", "collect_reference_value", "resolve_reference_value",
    "local_expression_identity", "_handle_data", "source_balance_supported",
)
