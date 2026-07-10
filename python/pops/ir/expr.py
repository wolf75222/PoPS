"""Symbolic node classes shared by the flux DSL and board AST."""
from __future__ import annotations

from typing import Any

from .literals import exact_numeric_scalar, exact_scale_prefix, multiply_exact_scalars, scalar_literal
from .symbolic import ImmutableSymbolic, freeze_symbolic_metadata

class Expr(ImmutableSymbolic):
    """Symbolic tree node with fail-loud extension protocols."""

    def __pops_ir_children__(self) -> Any:
        """Return child Expr nodes, or ``NotImplemented`` for built-in dispatch."""
        return NotImplemented

    def __pops_ir_key__(self, recurse: Any) -> Any:
        """Return a structural CSE key, or ``NotImplemented`` for built-in dispatch."""
        return NotImplemented

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Any:
        """Return a symbolic derivative, or ``NotImplemented`` when unsupported."""
        return NotImplemented

    def resolve_references(self, resolver: Any) -> Expr:
        """Return a detached graph whose declaration Handle leaves are canonical."""
        from .expr_references import resolve_expr_references
        return resolve_expr_references(self, resolver, {})

    def declaration_references(self) -> tuple[Any, ...]:
        """Return the typed Handle leaves carried by this immutable graph."""
        from .expr_references import collect_expr_references

        references = []
        collect_expr_references(self, references, set())
        return tuple(references)

    def __add__(self, o: Any) -> Any: return Add(self, _wrap(o))
    def __radd__(self, o: Any) -> Any: return Add(_wrap(o), self)
    def __sub__(self, o: Any) -> Any: return Sub(self, _wrap(o))
    def __rsub__(self, o: Any) -> Any: return Sub(_wrap(o), self)
    def __mul__(self, o: Any) -> Any: return Mul(self, _wrap(o))
    def __rmul__(self, o: Any) -> Any: return Mul(_wrap(o), self)
    def __truediv__(self, o: Any) -> Any: return Div(self, _wrap(o))
    def __rtruediv__(self, o: Any) -> Any: return Div(_wrap(o), self)
    def __neg__(self) -> Any: return Neg(self)
    def __pos__(self) -> Any: return self  # +expr = identity (the CoupledSource API writes +k*ne*ng)
    def __abs__(self) -> Any: return Abs(self)  # abs(expr) -> |expr| (absolute value, e.g. |lambda| of Roe)
    def __pow__(self, o: Any) -> Any: return Pow(self, _wrap(o))
    def __eq__(self, o: Any) -> Any: return Compare("eq", self, _wrap(o))
    def __ne__(self, o: Any) -> Any: return Compare("ne", self, _wrap(o))
    def __lt__(self, o: Any) -> Any: return Compare("lt", self, _wrap(o))
    def __le__(self, o: Any) -> Any: return Compare("le", self, _wrap(o))
    def __gt__(self, o: Any) -> Any: return Compare("gt", self, _wrap(o))
    def __ge__(self, o: Any) -> Any: return Compare("ge", self, _wrap(o))

    def eval(self, env: Any) -> Any: raise NotImplementedError
    def deps(self) -> Any: return set()
    def __repr__(self) -> str: return self._str()
    def _str(self) -> str: return "?"


def _wrap(o: Any) -> Any:
    if isinstance(o, Expr):
        return o
    # Promote a Param through its symbolic node; float(param) would erase runtime identity.
    node = getattr(o, "_node", None)
    if isinstance(node, Expr):
        return node
    return Const(o)


class Const(Expr):
    def __init__(self, value: Any) -> None:
        self.literal = scalar_literal(value)

    @property
    def value(self) -> Any:
        """The exact Python value when numerically representable (never coerced to float)."""
        return self.literal.to_python()

    def eval(self, env: Any) -> Any: return self.value
    def to_cpp(self) -> str: return self.literal.to_cpp()
    def _str(self) -> str: return repr(self.literal)


class Var(Expr):
    """Named variable: conservative, primitive, auxiliary (field) or constant."""

    def __init__(self, name: Any, kind: Any) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("Var: name must be a non-empty string")
        if not isinstance(kind, str) or not kind:
            raise TypeError("Var: kind must be a non-empty string")
        self.name = name
        self.kind = kind
    def eval(self, env: Any) -> Any:
        if self.name not in env:
            raise KeyError("variable '%s' (%s) missing from the environment" % (self.name, self.kind))
        return env[self.name]
    def deps(self) -> Any: return {self.name}
    def to_cpp(self) -> Any: return self.name
    def _str(self) -> Any: return self.name


class _Bin(Expr):
    op = "?"
    semantic_operation = None
    def __init__(self, a: Any, b: Any) -> None:
        if self.semantic_operation is not None and isinstance(a, Const) and isinstance(b, Const):
            # Local import avoids the expr <-> lowering cycle; both paths preserve annotations.
            from .lowering import _combined_annotations
            _combined_annotations(a, b, operation=self.semantic_operation)
        self.a = a
        self.b = b
    def deps(self) -> Any: return self.a.deps() | self.b.deps()
    def to_cpp(self) -> str: return "(%s %s %s)" % (self.a.to_cpp(), self.op, self.b.to_cpp())
    def _str(self) -> str: return "(%s %s %s)" % (self.a, self.op, self.b)


class Add(_Bin):
    op = "+"
    semantic_operation = "add"
    def eval(self, env: Any) -> Any: return self.a.eval(env) + self.b.eval(env)


class Sub(_Bin):
    op = "-"
    semantic_operation = "sub"
    def eval(self, env: Any) -> Any: return self.a.eval(env) - self.b.eval(env)


class Mul(_Bin):
    op = "*"
    semantic_operation = "mul"
    def eval(self, env: Any) -> Any: return self.a.eval(env) * self.b.eval(env)


class Div(_Bin):
    op = "/"
    semantic_operation = "div"
    def eval(self, env: Any) -> Any: return self.a.eval(env) / self.b.eval(env)


class Pow(_Bin):
    op = "**"
    semantic_operation = "pow"
    def eval(self, env: Any) -> Any: return self.a.eval(env) ** self.b.eval(env)
    def to_cpp(self) -> str: return "std::pow(%s, %s)" % (self.a.to_cpp(), self.b.to_cpp())


class Compare(_Bin):
    """A scalar symbolic comparison; Python never evaluates it as a bool."""

    _OPS = {
        "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
    }

    def __init__(self, comparison: Any, a: Any, b: Any) -> None:
        if comparison not in self._OPS:
            raise ValueError("unknown symbolic comparison %r" % (comparison,))
        self.comparison = comparison
        super().__init__(a, b)

    @property
    def op(self) -> str:
        return self._OPS[self.comparison]

    def eval(self, env: Any) -> Any:
        a, b = self.a.eval(env), self.b.eval(env)
        return {
            "eq": lambda: a == b,
            "ne": lambda: a != b,
            "lt": lambda: a < b,
            "le": lambda: a <= b,
            "gt": lambda: a > b,
            "ge": lambda: a >= b,
        }[self.comparison]()


class Neg(Expr):
    def __init__(self, a: Any) -> None: self.a = a
    def eval(self, env: Any) -> Any: return -self.a.eval(env)
    def deps(self) -> Any: return self.a.deps()
    def to_cpp(self) -> str: return "(-%s)" % self.a.to_cpp()
    def _str(self) -> str: return "(-%s)" % self.a


class Sqrt(Expr):
    def __init__(self, a: Any) -> None: self.a = a
    def eval(self, env: Any) -> Any:
        import numpy as np
        return np.sqrt(self.a.eval(env))
    def deps(self) -> Any: return self.a.deps()
    def to_cpp(self) -> str: return "std::sqrt(%s)" % self.a.to_cpp()
    def _str(self) -> str: return "sqrt(%s)" % self.a


class Abs(Expr):
    """Absolute value ``|a|`` (e.g. ``|lambda_k|`` of a Roe dissipation). Emitted as std::fabs at codegen
    (equal to the ternary a<0?-a:a outside -0.0). Not differentiable by dsl.diff (no sign node)."""
    def __init__(self, a: Any) -> None: self.a = a
    def eval(self, env: Any) -> Any:
        import numpy as np
        return np.abs(self.a.eval(env))
    def deps(self) -> Any: return self.a.deps()
    def to_cpp(self) -> str: return "std::fabs(%s)" % self.a.to_cpp()
    def _str(self) -> str: return "abs(%s)" % self.a


class Sign(Expr):
    """Sign of ``a`` (-1, 0 or 1), emitted branch-free as ``(a > 0) - (a < 0)``.
    Its derivative is zero almost everywhere; see ``dsl.diff``."""
    def __init__(self, a: Any) -> None: self.a = a
    def eval(self, env: Any) -> Any:
        import numpy as np
        return np.sign(self.a.eval(env))
    def deps(self) -> Any: return self.a.deps()
    def to_cpp(self) -> str:
        s = self.a.to_cpp()
        return "(pops::Real(%s > 0) - pops::Real(%s < 0))" % (s, s)
    def _str(self) -> str: return "sign(%s)" % self.a


# SECTION 2 -- BOARD AST NODES  (from pops.math)

class Equation(Expr):
    """A board equation ``lhs == rhs``.

    Produced by ``ddt(U) == ...``, ``-laplacian(phi) == rhs`` or
    ``(I - dt*C) @ unknown("x") == rhs``. The consuming API owns lowering."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs: Any, rhs: Any) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self) -> str:
        return "Equation(%r == %r)" % (self.lhs, self.rhs)

    def eval(self, env: Any) -> Any:
        raise TypeError("an Equation is a declarative graph node and cannot be numerically evaluated")

    def _str(self) -> str:
        return repr(self)


class _BoardNode(Expr):
    """Base of every board node: owns ``==`` (build an :class:`Equation`)."""

    def __eq__(self, other: Any) -> Any:  # noqa: D105 -- equation builder, not a comparison
        return Equation(self, other)

    def __ne__(self, other: Any) -> Any:
        # Board equality has equation semantics.  ``!=`` is not a different equation spelling;
        # require an explicit symbolic comparison instead of asking Python to negate Equation.
        return Compare("ne", self, _wrap(other))


# Elliptic concrete terms and helpers live in pops.ir.elliptic; this is inert IR.
class _EllipticTerm(_BoardNode):
    """A summand of an elliptic field-operator left-hand side.

    Elliptic terms compose with ``+`` / ``-`` / unary ``-`` into a
    ``pops.ir.elliptic.EllipticSum``; ``==`` builds the field :class:`Equation`.
    """

    def _elliptic_terms(self) -> Any:
        return [self]

    def _principal_kinds(self) -> Any:
        return {self._kind()}

    def _kind(self) -> Any:
        raise NotImplementedError

    def __neg__(self) -> Any:  # concrete terms (Laplacian / Reaction / ...) override this
        raise NotImplementedError

    def __add__(self, other: Any) -> Any:
        from .elliptic import EllipticSum, _as_elliptic
        return EllipticSum(self._elliptic_terms() + _as_elliptic(other)._elliptic_terms())

    def __radd__(self, other: Any) -> Any:
        from .elliptic import EllipticSum, _as_elliptic
        return EllipticSum(_as_elliptic(other)._elliptic_terms() + self._elliptic_terms())

    def __sub__(self, other: Any) -> Any:
        from .elliptic import _as_elliptic
        return self.__add__(-_as_elliptic(other))


class Partial(_BoardNode):
    """A first partial derivative ``scale * d(field)/dx_axis`` (axis 0=x, 1=y).

    ``grad(phi).x`` and ``dx(phi)`` build this. A model resolves it to the field's
    canonical gradient aux (``grad_x`` / ``grad_y``). Negation/scaling track the
    leading coefficient so ``-grad(phi).x`` resolves to ``-grad_x``.
    """

    def __init__(self, field: Any, axis: Any, scale: Any = 1.0) -> None:
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1):
            raise ValueError("Partial: axis must be the integer 0 or 1")
        self.field = field
        self.axis = axis
        self.scale = exact_numeric_scalar(scale, where="Partial scale")

    def __neg__(self) -> Any:
        return Partial(self.field, self.axis, -self.scale)

    def __mul__(self, k: Any) -> Any:
        return Partial(
            self.field, self.axis,
            multiply_exact_scalars(self.scale, k, where="Partial scale"))

    __rmul__ = __mul__

    def __repr__(self) -> str:
        d = "x" if self.axis == 0 else "y"
        return "Partial(%s%r.d%s)" % (exact_scale_prefix(self.scale), self.field, d)


class Gradient(_BoardNode):
    """The gradient of a scalar field; ``grad(phi).x`` / ``.y`` are :class:`Partial`."""

    def __init__(self, field: Any, scale: Any = 1.0) -> None:
        self.field = field
        self.scale = exact_numeric_scalar(scale, where="Gradient scale")

    @property
    def x(self) -> Any:
        return Partial(self.field, 0, self.scale)

    @property
    def y(self) -> Any:
        return Partial(self.field, 1, self.scale)

    def __neg__(self) -> Any:
        return Gradient(self.field, -self.scale)

    def __mul__(self, coeff: Any) -> Any:
        # coeff * grad(phi) -- a coefficient-scaled gradient (the flux of div(coeff*grad)).
        from .elliptic import CoeffGradient
        return CoeffGradient(self.field, coeff, self.scale)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return "Gradient(%r)" % (self.field,)


class Laplacian(_EllipticTerm):
    """``scale * Laplacian(field)`` -- the elliptic operator of a field solve."""

    def __init__(self, field: Any, scale: Any = 1.0) -> None:
        self.field = field
        self.scale = exact_numeric_scalar(scale, where="Laplacian scale")

    def _kind(self) -> Any:
        return "laplacian"

    def __neg__(self) -> Any:
        return Laplacian(self.field, -self.scale)

    def __repr__(self) -> str:
        return "Laplacian(%s%r)" % (exact_scale_prefix(self.scale), self.field)


class RateTerm(_BoardNode):
    """A summand of a rate equation right-hand side.

    Divergences and source handles compose through ``+`` / ``-`` into a
    :class:`RateExpr`, which the model splits into flux and source terms."""

    def _rate_terms(self) -> Any:
        """Return ``[(kind, payload, sign)]`` -- one entry per primitive summand."""
        raise NotImplementedError

    def __neg__(self) -> Any:
        return RateExpr([(k, p, -s) for (k, p, s) in self._rate_terms()])

    def __add__(self, other: Any) -> Any:
        return RateExpr(self._rate_terms() + _as_rate(other)._rate_terms())

    def __radd__(self, other: Any) -> Any:
        return RateExpr(_as_rate(other)._rate_terms() + self._rate_terms())

    def __sub__(self, other: Any) -> Any:
        return self + (-_as_rate(other))


def _as_rate(x: Any) -> Any:
    """Coerce ``x`` to a :class:`RateTerm` or raise a clear error."""
    if isinstance(x, RateTerm):
        return x
    coercion = getattr(x, "__pops_rate_term__", None)
    if callable(coercion):
        term = coercion()
        if isinstance(term, RateTerm):
            return term
        raise TypeError("__pops_rate_term__() must return a RateTerm expression")
    raise TypeError(
        "a rate equation right-hand side must be a sum of -div(flux) and source "
        "terms; got %r" % (x,))


class RateExpr(RateTerm):
    """An accumulated sum of rate terms (flux contributions and source handles)."""

    def __init__(self, terms: Any) -> None:
        normalized = []
        for term in terms:
            if not isinstance(term, (tuple, list)) or len(term) != 3:
                raise TypeError("a rate term must be a (kind, payload, sign) triple")
            kind, payload, sign = term
            if kind not in ("flux", "source"):
                raise ValueError("unknown rate term kind %r" % (kind,))
            if getattr(payload, "kind", None) != kind:
                raise TypeError("rate term %s payload must be a matching declaration Handle" % kind)
            sign = exact_numeric_scalar(sign, where="rate term sign")
            normalized.append((kind, payload, sign))
        self.terms = tuple(normalized)

    def _rate_terms(self) -> Any:
        return list(self.terms)

    def __repr__(self) -> str:
        return "RateExpr(%r)" % (self.terms,)


class Divergence(RateTerm):
    """``scale * div(flux)``; usually written ``-div(F)`` for a hyperbolic rate."""

    def __init__(self, flux: Any, scale: Any = 1.0) -> None:
        self.flux = flux
        self.scale = exact_numeric_scalar(scale, where="Divergence scale")

    def _rate_terms(self) -> Any:
        return [("flux", self.flux, self.scale)]

    def __repr__(self) -> str:
        return "Divergence(%s%r)" % (exact_scale_prefix(self.scale), self.flux)


class TimeDerivative(_BoardNode):
    """``ddt(U)`` / ``rate(U)`` -- the left-hand side of a rate equation."""

    def __init__(self, state: Any) -> None:
        self.state = state

    def __repr__(self) -> str:
        return "ddt(%r)" % (self.state,)


class Unknown(_BoardNode):
    """A solve unknown: ``unknown("U*")`` in ``(I - dt*C) @ unknown("U*") == rhs``."""

    def __init__(self, name: Any) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("Unknown: name must be a non-empty string")
        self.name = name

    def __rmatmul__(self, operator: Any) -> Any:
        """``operator @ unknown("U*")`` -- the left-hand side of an implicit solve."""
        return OpApply(operator, self)

    def __mul__(self, coeff: Any) -> Any:
        # coeff * phi -- a zeroth-order reaction term (the k*phi of a screened Poisson).
        from .elliptic import Reaction
        return Reaction(self, coeff)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return "unknown(%r)" % (self.name,)


class OpApply(_BoardNode):
    """``operator @ unknown`` -- a board solve left-hand side, completed by ``== rhs``.

    Carries the operator (a Program ``_Operator`` / linear-source value) and the
    :class:`Unknown`; :meth:`pops.time.Program.solve` destructures it.
    """

    def __init__(self, operator: Any, unknown: Any) -> None:
        self.operator = operator
        self.unknown = unknown

    def __repr__(self) -> str:
        return "OpApply(%r @ %r)" % (self.operator, self.unknown)

class Integral(_BoardNode):
    """A spatial integral of an expression -- the value of a generic invariant."""

    def __init__(self, expr: Any, over: Any = None) -> None:
        self.expr = expr
        self.over = freeze_symbolic_metadata(over)

    def __repr__(self) -> str:
        return "integral(%r)" % (self.expr,)
