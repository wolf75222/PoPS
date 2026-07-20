"""Public symbolic algebra for PoPS scientific authoring.

Users build immutable expressions here; the concrete intermediate-representation modules remain an
internal compiler detail.  Handles are imported from their owning scientific objects and wrapped
explicitly with :class:`ValueExpr` when an expression is required.
"""
from __future__ import annotations

__all__ = [
    "sqrt", "minimum", "maximum", "grad", "norm", "div", "laplacian", "dx", "dy", "ddt", "rate", "unknown",
    "integral",
    # Public symbolic values and node types.
    "Expr", "Const", "Var", "ValueExpr", "SymbolicTruthValueError",
    "Equation", "Gradient", "GradientMagnitude", "Partial", "Laplacian", "Divergence",
    "TimeDerivative", "Unknown", "OpApply", "Integral", "RateTerm", "RateExpr",
    # elliptic field-operator algebra (Spec 5 sec.9.2)
    "Reaction", "CoeffGradient", "DivCoeffGrad", "EllipticSum", "elliptic_terms",
    "principal_kinds",
]

from pops._ir.expr import (  # noqa: F401
    Expr,
    Const,
    Var,
    Equation,
    _BoardNode,
    Partial,
    Gradient,
    GradientMagnitude,
    Laplacian,
    RateTerm,
    RateExpr,
    Divergence,
    TimeDerivative,
    Unknown,
    OpApply,
    Integral,
)
from pops._ir.handle_expr import ValueExpr  # noqa: F401
from pops._ir.symbolic import SymbolicTruthValueError  # noqa: F401
from pops._ir.elliptic import (  # noqa: F401  (Spec 5 sec.9.2 elliptic field-operator algebra)
    Reaction,
    CoeffGradient,
    DivCoeffGrad,
    EllipticSum,
    elliptic_terms,
    principal_kinds,
)
from pops._ir.expr import _as_rate  # noqa: F401  (used internally by RateTerm)
from pops._ir.ops import (  # noqa: F401
    grad,
    norm,
    dx,
    dy,
    laplacian,
    div,
    ddt,
    rate,
    unknown,
    integral,
    minimum,
    maximum,
)
from pops._ir.ops import board_sqrt as sqrt  # noqa: F401


# --- scalar dtypes (Spec 5 sec.5.12: a typed param declares its dtype) -------------------
class _DType:
    """An inert scalar dtype marker (``Real`` / ``Integer`` / ``Bool``).

    Used by :mod:`pops.params` so a parameter declares a typed dtype instead of a string;
    the codegen / runtime consume it. It computes nothing.
    """

    def __init__(self, name: str) -> None:
        self._name = str(name)

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return self._name

    __str__ = __repr__


Real = _DType("Real")
Integer = _DType("Integer")
Bool = _DType("Bool")

__all__ += ["Real", "Integer", "Bool"]
