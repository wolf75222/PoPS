"""pops._ir.lowering -- pure-symbolic algebraic simplification helpers and diff.

Originally in pops.dsl.

  _is_const(e, val)   -- True if e is a numeric Const (optionally equal to val)
  _s_add / _s_neg / _s_sub / _s_mul / _s_div / _s_pow  -- minimal simplifying constructors
  diff(expr, var, defs) -- symbolic differentiation of the Expr tree
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any

from .expr import (
    Expr, _wrap,
    Const, Var, Add, Sub, Mul, Div, Pow, Neg, Sqrt, Abs, Sign,
)
from pops.identity.scalar import (
    exact_decimal_add,
    exact_decimal_divide,
    exact_decimal_multiply,
    exact_decimal_negate,
    numeric_domains_compatible,
)
from .values import RuntimeParamRef


_NOT_NUMERIC = object()


def _const_value(expr: Any) -> Any:
    try:
        return expr.value
    except TypeError:
        return _NOT_NUMERIC


def _constant_annotations(expr: Any) -> tuple[str | None, str | None]:
    return expr.literal.unit, expr.literal.target


def _combined_annotations(a: Any, b: Any, *, operation: str) -> tuple[str | None, str | None]:
    """Validate annotations without pretending that PoPS has a unit algebra."""
    left_unit, left_target = _constant_annotations(a)
    right_unit, right_target = _constant_annotations(b)
    if operation in ("add", "sub"):
        if left_unit != right_unit:
            raise TypeError(
                "symbolic %s requires identical units, got %r and %r"
                % (operation, left_unit, right_unit))
        unit = left_unit
    else:
        if left_unit is not None or right_unit is not None:
            raise TypeError(
                "symbolic %s of unit-bearing constants requires an explicit unit-system "
                "operation; PoPS will not invent or erase a compound unit"
                % operation)
        unit = None
    if left_target is not None and right_target is not None and left_target != right_target:
        raise TypeError(
            "symbolic %s requires compatible scalar targets, got %r and %r"
            % (operation, left_target, right_target))
    return unit, left_target or right_target


def _fold_constants(a: Any, b: Any, operation: Any, node: Any, *, kind: str) -> Any:
    """Fold constants without crossing domains or discarding annotations."""
    left, right = _const_value(a), _const_value(b)
    if left is _NOT_NUMERIC or right is _NOT_NUMERIC:
        return node(a, b)
    if not numeric_domains_compatible(left, right):
        raise TypeError(
            "symbolic constant folding cannot mix %s and %s without an explicit target conversion"
            % (type(left).__name__, type(right).__name__))
    unit, target = _combined_annotations(a, b, operation=kind)
    result = operation(left, right)
    if result is NotImplemented:
        return node(a, b)
    return Const(a.literal.from_value(result, unit=unit, target=target))


def _exact_add_numbers(left: Any, right: Any) -> Any:
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        return exact_decimal_add(left, right)
    return left + right


def _exact_subtract_numbers(left: Any, right: Any) -> Any:
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        decimal_right = right if isinstance(right, Decimal) else Decimal(right)
        return exact_decimal_add(left, exact_decimal_negate(decimal_right))
    return left - right


def _exact_multiply_numbers(left: Any, right: Any) -> Any:
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        return exact_decimal_multiply(left, right)
    return left * right


def _exact_divide(left: Any, right: Any) -> Any:
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        result = exact_decimal_divide(left, right)
        return NotImplemented if result is None else result
    if isinstance(left, int) and isinstance(right, int):
        return Fraction(left, right)
    return left / right


def _exact_power(left: Any, right: Any) -> Any:
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        return NotImplemented
    if isinstance(left, int) and isinstance(right, int) and right < 0:
        return Fraction(1, left ** (-right))
    return left ** right


# --- Symbolic differentiation (autodiff of the Expr tree) -------------------
# dsl.diff(expr, var) differentiates the tree node by node: linearity (+, -), product (a*b)' = a'b + ab',
# quotient (a/b)' = (a'b - ab')/b^2, power (a^n)' = n a^(b-1) a' (constant exponent), root
# sqrt(a)' = a'/(2 sqrt(a)), negation. Used to build the flux Jacobian A = dF/dU (flux_jacobian)
# that the user employs to write its Roe dissipation (m.roe_dissipation). A DEFINED primitive
# is differentiated BY ITS DEFINITION (chain rule); a NON differentiated occurrence stays a
# symbol (readable emission), only the DERIVATIVE descends to the conservatives. Unknown node ->
# NotImplementedError (never a silent zero). Minimal simplifications (0*x, 1*x, x+0).

def _is_const(e: Any, val: Any = None) -> bool:
    """True if e is a numeric constant (Const); if @p val is given, equal to val."""
    if not isinstance(e, Const):
        return False
    value = _const_value(e)
    return value is not _NOT_NUMERIC and (val is None or value == val)


def _is_unannotated_const(expr: Any, value: Any) -> bool:
    return (_is_const(expr, value)
            and expr.literal.unit is None and expr.literal.target is None)


def _s_add(a: Any, b: Any) -> Any:
    if isinstance(a, Const) and isinstance(b, Const):
        return _fold_constants(a, b, _exact_add_numbers, Add, kind="add")
    if _is_unannotated_const(a, 0):
        return b
    if _is_unannotated_const(b, 0):
        return a
    return Add(a, b)


def _s_neg(a: Any) -> Any:
    if _is_unannotated_const(a, 0):
        return Const(0)
    if isinstance(a, Const):
        value = _const_value(a)
        if value is _NOT_NUMERIC:
            return Neg(a)
        negated = exact_decimal_negate(value) if isinstance(value, Decimal) else -value
        return Const(a.literal.from_value(
            negated, unit=a.literal.unit, target=a.literal.target))
    if isinstance(a, Neg):
        return a.a
    return Neg(a)


def _s_sub(a: Any, b: Any) -> Any:
    if isinstance(a, Const) and isinstance(b, Const):
        return _fold_constants(a, b, _exact_subtract_numbers, Sub, kind="sub")
    if _is_unannotated_const(b, 0):
        return a
    if _is_unannotated_const(a, 0):
        return _s_neg(b)
    return Sub(a, b)


def _s_mul(a: Any, b: Any) -> Any:
    if isinstance(a, Const) and isinstance(b, Const):
        return _fold_constants(a, b, _exact_multiply_numbers, Mul, kind="mul")
    if _is_unannotated_const(a, 0) or _is_unannotated_const(b, 0):
        return Const(0)
    if _is_unannotated_const(a, 1):
        return b
    if _is_unannotated_const(b, 1):
        return a
    return Mul(a, b)


def _s_div(a: Any, b: Any) -> Any:
    if isinstance(a, Const) and isinstance(b, Const):
        return _fold_constants(a, b, _exact_divide, Div, kind="div")
    if _is_unannotated_const(a, 0):
        return Const(0)
    if _is_unannotated_const(b, 1):
        return a
    return Div(a, b)


def _s_pow(a: Any, b: Any) -> Any:
    # b: exponent (Expr), here assumed INDEPENDENT of the differentiation variable.
    if isinstance(a, Const) and isinstance(b, Const):
        exponent = _const_value(b)
        if isinstance(exponent, int):
            return _fold_constants(a, b, _exact_power, Pow, kind="pow")
        _combined_annotations(a, b, operation="pow")
    if _is_unannotated_const(b, 0):
        return Const(1)
    if _is_unannotated_const(b, 1):
        return a
    return Pow(a, b)


def diff(expr: Any, var: Any, defs: Any = None) -> Any:
    """Symbolic derivative of @p expr with respect to @p var (variable name or Var).

    @p defs (optional): dictionary {primitive name: definition Expr}. When the differentiation
    meets a DEFINED primitive, it differentiates its DEFINITION (chain rule) -- the primitives
    are expanded down to the conservatives without manual substitution. A primitive with the same name
    as @p var is treated as the independent variable (derivative 1). Without defs, any variable
    other than @p var is independent (derivative 0).

    @return an Expr minimally simplified (0*x, 1*x, x+0, ... removed for a readable emission).
    Raises NotImplementedError on a non differentiable node (naming its type) or a power whose
    exponent depends on @p var (would need a logarithm, a node absent from the DSL)."""
    target = var.name if isinstance(var, Var) else var
    if not isinstance(target, str) or not target:
        if not callable(getattr(var, "canonical_identity", None)):
            raise TypeError("diff variable must be a Var, declaration Handle, or non-empty string")
        target = None  # extension nodes receive the original Handle through __pops_ir_diff__
    d = defs or {}

    def go(e: Any) -> Any:
        protocol = getattr(e, "__pops_ir_diff__", None)
        if callable(protocol):
            derivative = protocol(recurse=go, target=var, definitions=d)
            if derivative is not NotImplemented:
                if not isinstance(derivative, Expr):
                    raise TypeError(
                        "%s.__pops_ir_diff__() must return an Expr or NotImplemented"
                        % type(e).__name__)
                return derivative
        if isinstance(e, Const):
            return Const(0)
        if isinstance(e, RuntimeParamRef):
            return Const(0)  # runtime parameter: constant with respect to the conservative state
        if isinstance(e, Var):
            if e.name == target:
                return Const(1)
            if e.name in d:
                return go(d[e.name])  # defined primitive -> derivative of its definition (chain)
            return Const(0)           # another variable, independent of var
        if isinstance(e, Add):
            return _s_add(go(e.a), go(e.b))
        if isinstance(e, Sub):
            return _s_sub(go(e.a), go(e.b))
        if isinstance(e, Mul):
            return _s_add(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
        if isinstance(e, Div):
            num = _s_sub(_s_mul(go(e.a), e.b), _s_mul(e.a, go(e.b)))
            return _s_div(num, _s_mul(e.b, e.b))
        if isinstance(e, Neg):
            return _s_neg(go(e.a))
        if isinstance(e, Sqrt):
            return _s_div(go(e.a), _s_mul(Const(2), Sqrt(e.a)))
        if isinstance(e, Abs):
            # d|u| = (u / |u|) u' -- exact derivative away from the fold u = 0 (the smooth floors
            # max(x, eps) = ((x+eps) + |x-eps|)/2 of the 'robust' models give there exactly
            # the expected indicator); AT the fold, u/|u| is NaN: a zero-measure singularity,
            # documented (like the division of quotients).
            return _s_mul(_s_div(e.a, Abs(e.a)), go(e.a))
        if isinstance(e, Sign):
            # d sign(u) = 0 presque partout (saut en u = 0, mesure nulle -- meme convention que le
            # pli de Abs : singularite documentee, jamais rencontree par les clamps sur un ouvert).
            return Const(0)
        if isinstance(e, Pow):
            if not _is_const(go(e.b), 0):
                raise NotImplementedError(
                    "dsl.diff: derivative of a**b with exponent depending on '%s' (needs a "
                    "logarithm, a node absent from the DSL)" % target)
            # constant exponent with respect to var: (a^b)' = b a^(b-1) a'
            return _s_mul(_s_mul(e.b, _s_pow(e.a, _s_sub(e.b, Const(1)))), go(e.a))
        raise NotImplementedError("dsl.diff: non differentiable node %s (%r)" % (type(e).__name__, e))

    return go(_wrap(expr))
