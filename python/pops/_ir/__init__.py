"""Internal symbolic intermediate representation.

This package is compiler plumbing, not an authoring surface.  Public scientific expressions are
constructed through :mod:`pops.math`; internal modules import concrete nodes from here to implement
validation, lowering and code generation.

Node base classes
  Expr

Flux-DSL node classes
  Const, Var, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign

Board AST node classes
  Equation, Partial, Gradient, Laplacian,
  RateTerm, RateExpr, Divergence, TimeDerivative, Unknown, OpApply, Integral

Reference / witness value classes
  EigWitness, StateRef, RuntimeParamRef

Internal operator nodes and helpers (board + DSL)
  grad, dx, dy, div, laplacian, ddt, rate, unknown, integral,
  sqrt (flux-DSL canonical), board_sqrt (board delegate to pops.dsl.sqrt),
  abs_, sign,
  eig_max_im, eig_lmin, eig_lmax, eig_all_real, eig_real_status,
  left, right

Pure-symbolic helpers
  _children, _expr_uses_cons_or_prim, _key   (visitors)
  _is_const, _s_add, _s_neg, _s_sub, _s_mul, _s_div, _s_pow, diff  (lowering)
"""

# -- node classes ---------------------------------------------------------
from .expr import (
    Expr, _wrap,
    Const, Var, _Bin, Add, Sub, Mul, Div, Pow, Minimum, Maximum, Compare,
    BooleanAnd, BooleanOr, BooleanNot, Neg, Sqrt, Abs, Sign,
    # board nodes
    Equation, _BoardNode,
    Partial, Gradient, GradientMagnitude, Laplacian,
    RateTerm, _as_rate, RateExpr, Divergence,
    TimeDerivative, Unknown, OpApply, Integral,
)
from .handle_expr import ValueExpr
from .param_values import parameter_value
from pops.identity.scalar import ScalarLiteral, scalar_cpp, scalar_data, scalar_literal, scalar_to_native
from .symbolic import SourceLocation, SymbolicTruthValueError

# -- reference / witness values -------------------------------------------
from .values import (
    _EIG_FIELDS, _EIG_PREDICATES,
    EigWitness, StateRef, RuntimeParamRef,
)

# -- free-function ops ----------------------------------------------------
from .ops import (
    # flux-DSL
    sqrt, abs_, sign, minimum, maximum,
    eig_max_im, eig_lmin, eig_lmax, eig_all_real, eig_real_status,
    left, right,
    # board
    grad, norm, dx, dy, laplacian, div, ddt, rate, unknown, integral,
    board_sqrt,
)

# -- pure-symbolic helpers ------------------------------------------------
from .visitors import (
    _children, _dag_key_data, _dag_key_ids, _dependencies, _expr_uses_cons_or_prim, _key,
)
from .lowering import (
    _is_const, _s_add, _s_neg, _s_sub, _s_mul, _s_div, _s_pow,
    diff,
)

__all__ = [
    # node classes
    "Expr", "Const", "Var", "ValueExpr", "parameter_value", "Add", "Sub", "Mul", "Div", "Pow", "Minimum", "Maximum", "Compare",
    "BooleanAnd", "BooleanOr", "BooleanNot",
    "Neg", "Sqrt", "Abs", "Sign",
    # board nodes
    "Equation",
    "Partial", "Gradient", "GradientMagnitude", "Laplacian",
    "RateTerm", "RateExpr", "Divergence",
    "TimeDerivative", "Unknown", "OpApply", "Integral",
    # values
    "EigWitness", "StateRef", "RuntimeParamRef",
    # ops
    "sqrt", "abs_", "sign", "minimum", "maximum",
    "eig_max_im", "eig_lmin", "eig_lmax", "eig_all_real", "eig_real_status",
    "left", "right",
    "grad", "norm", "dx", "dy", "laplacian", "div", "ddt", "rate", "unknown", "integral",
    "board_sqrt",
    # helpers
    "diff",
    # exact constants and structured symbolic diagnostics
    "ScalarLiteral", "scalar_literal", "scalar_data", "scalar_cpp", "scalar_to_native",
    "SourceLocation", "SymbolicTruthValueError",
]
