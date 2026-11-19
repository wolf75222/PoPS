"""Symbolic node classes shared by the flux DSL and board AST."""
from __future__ import annotations

from typing import Any

from pops.identity.scalar import exact_numeric_scalar, exact_scale_prefix, multiply_exact_scalars, scalar_literal
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

    # A vector, or a term of a balance (RateTerm), owns the operation with an expression: a vector
    # applies it per component, and a term reads the expression as an operand of its balance (R7).
    def __add__(self, o: Any) -> Any:
        owner = _operation_owner(o)
        return owner.__radd__(self) if owner is not None else Add(self, _wrap(o))
    def __radd__(self, o: Any) -> Any: return Add(_wrap(o), self)
    def __sub__(self, o: Any) -> Any:
        owner = _operation_owner(o)
        return owner.__rsub__(self) if owner is not None else Sub(self, _wrap(o))
    def __rsub__(self, o: Any) -> Any: return Sub(_wrap(o), self)
    def __mul__(self, o: Any) -> Any:
        owner = _operation_owner(o)
        if owner is not None:
            return owner.__rmul__(self)
        # ``kappa_expr * unknown`` is an elliptic reaction term, not an ordinary pointwise
        # multiplication.  Keeping this conversion here makes the natural screened-Poisson
        # spelling work for an explicit ``model.value(RuntimeParam(...))`` while Handles retain
        # ordinary Boolean equality/hash semantics.
        if isinstance(o, Unknown):
            from .elliptic import Reaction
            return Reaction(o, self)
        return Mul(self, _wrap(o))
    def __rmul__(self, o: Any) -> Any: return Mul(_wrap(o), self)
    def __truediv__(self, o: Any) -> Any:
        owner = _operation_owner(o)
        return owner.__rtruediv__(self) if owner is not None else Div(self, _wrap(o))
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
    def __and__(self, o: Any) -> Any: return BooleanAnd(self, o)
    def __rand__(self, o: Any) -> Any: return BooleanAnd(o, self)
    def __or__(self, o: Any) -> Any: return BooleanOr(self, o)
    def __ror__(self, o: Any) -> Any: return BooleanOr(o, self)
    def __invert__(self) -> Any: return BooleanNot(self)

    def eval(self, env: Any) -> Any: raise NotImplementedError
    def deps(self) -> Any: return set()
    def __repr__(self) -> str: return self._str()
    def _str(self) -> str: return "?"


def is_scalar_expression(value: Any) -> bool:
    return isinstance(value, Expr) or callable(getattr(value, "__pops_scalar_plan__", None))


def _wrap(o: Any) -> Any:
    if isinstance(o, Expr):
        if isinstance(o, RateTerm):
            # A term of a balance is a summand of its right-hand side, never the operand of
            # another node (R7): refused where the node is written.
            from .rate_algebra import OPERAND, refuse
            raise refuse(o, OPERAND)
        return o
    # Promote a Param through its symbolic node; float(param) would erase runtime identity.
    node = getattr(o, "_node", None)
    if isinstance(node, Expr):
        return node
    scalar_expression = getattr(o, "__pops_scalar_expression__", None)
    if callable(scalar_expression):
        node = scalar_expression()
        if not isinstance(node, Expr):
            raise TypeError("a symbolic scalar expression protocol must return an Expr")
        return node
    if callable(getattr(o, "__pops_scalar_plan__", None)):
        from .finite_linear import lower_finite_scalars
        return lower_finite_scalars((o,))[0]
    if callable(getattr(o, "__pops_rate_term__", None)):
        from .rate_algebra import refuse_operand
        refusal = refuse_operand(o)
        if refusal is not None:
            raise refusal
    return Const(o)

class Const(Expr):
    def __init__(self, value: Any, *, handle: Any = None) -> None:
        self.literal = scalar_literal(value)
        self.handle = handle

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
    a: Any
    b: Any
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


class Minimum(_Bin):
    """Pointwise IEEE ``fmin``: return the numeric operand when exactly one input is NaN."""

    op = "minimum"

    def eval(self, env: Any) -> Any:
        import numpy as np
        return np.fmin(self.a.eval(env), self.b.eval(env))

    def to_cpp(self) -> str:
        return "Kokkos::fmin(%s, %s)" % (self.a.to_cpp(), self.b.to_cpp())

    def _str(self) -> str:
        return "minimum(%s, %s)" % (self.a, self.b)


class Maximum(_Bin):
    """Pointwise IEEE ``fmax``: return the numeric operand when exactly one input is NaN."""

    op = "maximum"

    def eval(self, env: Any) -> Any:
        import numpy as np
        return np.fmax(self.a.eval(env), self.b.eval(env))

    def to_cpp(self) -> str:
        return "Kokkos::fmax(%s, %s)" % (self.a.to_cpp(), self.b.to_cpp())

    def _str(self) -> str:
        return "maximum(%s, %s)" % (self.a, self.b)


class Compare(_Bin):
    """A scalar symbolic comparison; Python never evaluates it as a bool."""

    __pops_ir_boolean__ = True

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



def is_boolean_expression(value: Any) -> bool:
    """Whether ``value`` is a symbolic Boolean expression of the IR.

    A Boolean-valued node class declares ``__pops_ir_boolean__ = True``: the comparisons, the
    Boolean connectives and the control nodes that a ``when`` body reads as a Boolean.
    """
    return isinstance(value, Expr) and getattr(type(value), "__pops_ir_boolean__", False) is True


def _boolean_operand(value: Any, *, where: str) -> Any:
    if not is_boolean_expression(value):
        raise TypeError("%s requires a typed symbolic Boolean expression" % where)
    return value


class BooleanAnd(_Bin):
    __pops_ir_boolean__ = True
    op = "&"

    def __init__(self, a: Any, b: Any) -> None:
        super().__init__(
            _boolean_operand(a, where="BooleanAnd"),
            _boolean_operand(b, where="BooleanAnd"),
        )

    def eval(self, env: Any) -> Any:
        return self.a.eval(env) & self.b.eval(env)



class BooleanOr(_Bin):
    __pops_ir_boolean__ = True
    op = "|"

    def __init__(self, a: Any, b: Any) -> None:
        super().__init__(
            _boolean_operand(a, where="BooleanOr"),
            _boolean_operand(b, where="BooleanOr"),
        )

    def eval(self, env: Any) -> Any:
        return self.a.eval(env) | self.b.eval(env)



class BooleanNot(Expr):
    __pops_ir_boolean__ = True

    def __init__(self, value: Any) -> None:
        self.a = _boolean_operand(value, where="BooleanNot")

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return (self.a,)

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("boolean_not", recurse(self.a))

    def eval(self, env: Any) -> Any:
        return ~self.a.eval(env)

    def _str(self) -> str:
        return "~(%s)" % self.a


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


class Exp(Expr):
    """Pointwise exponential, distinct from powers and analytic initial-data expressions."""
    def __init__(self, a: Any) -> None: self.a = a
    def eval(self, env: Any) -> Any:
        import numpy as np
        return np.exp(self.a.eval(env))
    def deps(self) -> Any: return self.a.deps()
    def to_cpp(self) -> str: return "std::exp(%s)" % self.a.to_cpp()
    def _str(self) -> str: return "exp(%s)" % self.a


class UnaryMath(Expr):
    """A closed elementary scalar operation, shared by all native lowering paths.

    The operation belongs to the compiler's arithmetic vocabulary, never to a physical
    model. Subclasses name a supported operation; authors compose the public functions.
    """
    operation: str

    def __init__(self, a: Any) -> None:
        if type(self) not in (Sin, Cos, Log, Expm1):
            raise TypeError("unsupported elementary scalar operation")
        self.a = _wrap(a)

    def __pops_ir_children__(self):
        return (self.a,)

    def __pops_ir_key__(self, recurse):
        return (self.operation, recurse(self.a))

    def __pops_ir_diff__(self, recurse, target, definitions):
        from .lowering import _s_mul, _s_div, _s_neg
        derivative = recurse(self.a)
        if type(self) is Sin:
            return _s_mul(Cos(self.a), derivative)
        if type(self) is Cos:
            return _s_mul(_s_neg(Sin(self.a)), derivative)
        if type(self) is Log:
            return _s_div(derivative, self.a)
        return _s_mul(Exp(self.a), derivative)

    def eval(self, env: Any) -> Any:
        import numpy as np
        return getattr(np, self.operation)(self.a.eval(env))

    def deps(self) -> Any:
        return self.a.deps()

    def to_cpp(self) -> str:
        return "Kokkos::%s(%s)" % (self.operation, self.a.to_cpp())

    def _str(self) -> str:
        return "%s(%s)" % (self.operation, self.a)


class Sin(UnaryMath):
    operation = "sin"


class Cos(UnaryMath):
    operation = "cos"


class Log(UnaryMath):
    operation = "log"


class Expm1(UnaryMath):
    operation = "expm1"


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


# Elliptic concrete terms and helpers live in pops._ir.elliptic; this is inert IR.
class _EllipticTerm(_BoardNode):
    """A summand of an elliptic field-operator left-hand side.

    Elliptic terms compose with ``+`` / ``-`` / unary ``-`` into a
    ``pops._ir.elliptic.EllipticSum``; ``==`` builds the field :class:`Equation`.
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
    """A first partial derivative ``scale * d(field)/dx_axis``.

    Axis ordinals are canonical Cartesian ordinals (0=x, 1=y, 2=z).  The owning model authenticates
    the ordinal against its inferred frame rank before lowering it to a native field component.
    """

    def __init__(self, field: Any, axis: Any, scale: Any = 1.0) -> None:
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in range(3):
            raise ValueError("Partial: axis must be a canonical Cartesian ordinal (0, 1, or 2)")
        self.field = field
        self.axis = axis
        self.scale = exact_numeric_scalar(scale, where="Partial scale")

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return (self.field,) if isinstance(self.field, Expr) else ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        from .quantity import expression_handle_key
        field = (recurse(self.field) if isinstance(self.field, Expr)
                 else expression_handle_key(self.field))
        return ("partial", field, self.axis, repr(self.scale))

    def __neg__(self) -> Any:
        return Partial(self.field, self.axis, -self.scale)

    def __mul__(self, k: Any) -> Any:
        return Partial(
            self.field, self.axis,
            multiply_exact_scalars(self.scale, k, where="Partial scale"))

    def __rmul__(self, o: Any) -> Any:
        return self.__mul__(o)

    def __repr__(self) -> str:
        d = ("x", "y", "z")[self.axis]
        return "Partial(%s%r.d%s)" % (exact_scale_prefix(self.scale), self.field, d)


class Gradient(_BoardNode):
    """The gradient of a scalar field; each Cartesian component is a :class:`Partial`."""

    def __init__(self, field: Any, scale: Any = 1.0) -> None:
        self.field = field
        self.scale = exact_numeric_scalar(scale, where="Gradient scale")

    @property
    def x(self) -> Any:
        return Partial(self.field, 0, self.scale)

    @property
    def y(self) -> Any:
        return Partial(self.field, 1, self.scale)

    @property
    def z(self) -> Any:
        return Partial(self.field, 2, self.scale)

    def __getitem__(self, axis: Any) -> Any:
        """Select a component with a typed Cartesian axis or its exact ordinal."""
        ordinal = getattr(axis, "index", axis)
        return Partial(self.field, ordinal, self.scale)

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return (self.field,) if isinstance(self.field, Expr) else ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        from .quantity import expression_handle_key
        field = (recurse(self.field) if isinstance(self.field, Expr)
                 else expression_handle_key(self.field))
        return ("gradient", field, repr(self.scale))

    def __neg__(self) -> Any:
        return Gradient(self.field, -self.scale)

    def __mul__(self, coeff: Any) -> Any:
        # coeff * grad(phi) -- a coefficient-scaled gradient (the flux of div(coeff*grad)).
        from .elliptic import CoeffGradient
        return CoeffGradient(self.field, coeff, self.scale)

    def __rmul__(self, o: Any) -> Any:
        return self.__mul__(o)

    def __repr__(self) -> str:
        return "Gradient(%r)" % (self.field,)


class GradientMagnitude(Expr):
    """Semantic ``|grad(field)|`` resolved by a discrete consumer context.

    The node deliberately has no context-free evaluator or C++ spelling: a consumer must resolve
    it against a layout, a discretization and a stencil. No consumer does since the AMR tag graph
    went (N12 lot 3c-2b); a refinement region reads the gradient of the method through
    ``pops.lib.amr.criteria.gradient_norm``.
    """

    def __init__(self, gradient: Any) -> None:
        if not isinstance(gradient, Gradient) or not isinstance(gradient.field, Expr):
            raise TypeError("norm(grad(...)) requires a gradient of a PoPS Expr")
        self.field = gradient.field
        self.scale = gradient.scale

    def __pops_ir_children__(self) -> tuple[Expr, ...]:
        return (self.field,)

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("gradient_magnitude", scalar_literal(self.scale).to_data(), recurse(self.field))

    def eval(self, env: Any) -> Any:
        del env
        raise TypeError("GradientMagnitude requires a resolved discrete context")

    def to_cpp(self) -> str:
        raise TypeError("GradientMagnitude lowers only through a discrete consumer context")

    def _str(self) -> str:
        return "norm(grad(%s))" % self.field


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

    Declared terms (``div`` of a declared flux, a source handle, ...) compose through ``+`` and
    ``-``, and scale by exact numbers, into a :class:`RateExpr`, whose signed occurrences the model
    retains as one balance. Algebra without terms (``k*u**2``) joins the sum as an inline operand,
    which the model declares as an anonymous source of the balance (R7). A term under any other
    node is refused with ``TermNotAdditive`` (``pops._ir.rate_algebra``)."""

    def _rate_terms(self) -> Any:
        """Return ``[(kind, payload, coefficient)]`` -- one entry per summand."""
        raise NotImplementedError

    def __neg__(self) -> Any:
        return RateExpr([(k, p, -s) for (k, p, s) in self._rate_terms()])

    def __add__(self, other: Any) -> Any:
        return RateExpr(self._rate_terms() + _as_rate(other)._rate_terms())

    def __radd__(self, other: Any) -> Any:
        return RateExpr(_as_rate(other)._rate_terms() + self._rate_terms())

    def __sub__(self, other: Any) -> Any:
        return self + (-_as_rate(other))

    def __rsub__(self, other: Any) -> Any:
        return _as_rate(other) + (-self)

    def __mul__(self, coefficient: Any) -> Any:
        from .rate_algebra import exact_number, refuse
        number = exact_number(coefficient)
        if number is None:
            raise refuse(self, "a product by %r" % (coefficient,))
        return RateExpr([(kind, payload, multiply_exact_scalars(
            scale, number, where="rate term coefficient"))
            for kind, payload, scale in self._rate_terms()])

    def __rmul__(self, coefficient: Any) -> Any:
        return self * coefficient

    def __truediv__(self, divisor: Any) -> Any:
        from .rate_algebra import exact_reciprocal, refuse
        reciprocal = exact_reciprocal(divisor)
        if reciprocal is None:
            raise refuse(self, "a quotient by %r" % (divisor,))
        return self * reciprocal

    def __rtruediv__(self, other: Any) -> Any:
        from .rate_algebra import refuse
        raise refuse(self, "the denominator of a quotient of %r" % (other,))

    # Every other operation that Expr defines would put the term under a node: refused at the
    # line that writes it (R7). ``==`` stays the equation builder of the board nodes.
    def _not_additive(self, construct: str) -> Any:
        from .rate_algebra import refuse
        return refuse(self, construct)

    def __pow__(self, other: Any) -> Any:
        raise self._not_additive("a power with exponent %r" % (other,))

    def __rpow__(self, other: Any) -> Any:
        raise self._not_additive("the exponent of a power of %r" % (other,))

    def __abs__(self) -> Any:
        raise self._not_additive("an absolute value")

    def __lt__(self, other: Any) -> Any:
        raise self._not_additive("a comparison with %r" % (other,))

    def __le__(self, other: Any) -> Any:
        raise self._not_additive("a comparison with %r" % (other,))

    def __gt__(self, other: Any) -> Any:
        raise self._not_additive("a comparison with %r" % (other,))

    def __ge__(self, other: Any) -> Any:
        raise self._not_additive("a comparison with %r" % (other,))


def rate_term_of(value: Any) -> Any:
    """The term of a balance that ``value`` denotes, or ``None`` (R7).

    A :class:`RateTerm` is itself; an object of the ``__pops_rate_term__`` protocol (a declared
    source handle, a nonconservative product handle) is the term that the protocol returns. Every
    other value, every other ``Expr`` included, denotes no term.
    """
    if isinstance(value, RateTerm):
        return value
    if isinstance(value, Expr):
        return None
    coercion = getattr(value, "__pops_rate_term__", None)
    if callable(coercion):
        term = coercion()
        return term if isinstance(term, RateTerm) else None
    return None


def _operation_owner(o: Any) -> Any:
    """The operand that owns a binary operation with an expression, or ``None``.

    A vector applies the operation per component; a term of a balance reads the expression as an
    operand of its balance (R7).
    """
    from .vector_expr import VectorExpr
    if isinstance(o, VectorExpr):
        return o
    return rate_term_of(o)


def _as_rate(x: Any) -> Any:
    """Coerce ``x`` to a :class:`RateTerm`: a term, or what joins one in a balance (R7).

    A term, or an object of the ``__pops_rate_term__`` protocol, is returned as is. Anything else is
    read by ``pops._ir.rate_algebra.rate_summands``: the number 0 adds nothing, algebra without
    terms is one inline operand, and an expression that holds terms is read through its additive
    structure, or refused.
    """
    if isinstance(x, RateTerm):
        return x
    coercion = getattr(x, "__pops_rate_term__", None)
    if callable(coercion):
        term = coercion()
        if isinstance(term, RateTerm):
            return term
        raise TypeError("__pops_rate_term__() must return a RateTerm expression")
    from .rate_algebra import rate_summands
    return RateExpr(rate_summands(x))


class RateExpr(RateTerm):
    """An accumulated sum of rate terms (flux contributions and source handles).

    The kind ``"inline"`` holds an operand without terms (``pops._ir.rate_algebra.InlineAlgebra``)
    while the sum is written; the model declares it as an anonymous source of the balance, so it
    never reaches a ``BalanceOccurrence``."""

    def __init__(self, terms: Any) -> None:
        normalized = []
        for term in terms:
            if not isinstance(term, (tuple, list)) or len(term) != 3:
                raise TypeError("a rate term must be a (kind, payload, sign) triple")
            kind, payload, sign = term
            if kind not in ("flux", "diffusion", "coupled_gradient", "drift", "source", "projection", "nonconservative", "inline"):
                raise ValueError("unknown rate term kind %r" % (kind,))
            if kind == "inline":
                from .rate_algebra import InlineAlgebra
                if not isinstance(payload, InlineAlgebra):
                    raise TypeError("an inline rate term holds an InlineAlgebra operand")
            elif kind == "projection":
                from .application import RateApplicationProjection
                if not isinstance(payload, RateApplicationProjection):
                    raise TypeError("a projection rate term requires a whole typed RateSpace projection")
            elif getattr(payload, "kind", None) != ({"diffusion": "diffusive_flux",
                    "coupled_gradient": "coupled_gradient_flux",
                    "drift": "drift_flux", "nonconservative": "nonconservative_product"}.get(kind, kind)):
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

    def __init__(self, flux: Any, scale: Any = 1) -> None:
        self.flux = flux
        self.scale = exact_numeric_scalar(scale, where="Divergence scale")

    def _rate_terms(self) -> Any:
        kind = {"diffusive_flux":"diffusion","coupled_gradient_flux":"coupled_gradient",
                "drift_flux":"drift"}.get(getattr(self.flux,"kind",""),"flux")
        return [(kind, self.flux, self.scale)]

    def __repr__(self) -> str:
        return "Divergence(%s%r)" % (exact_scale_prefix(self.scale), self.flux)


class TimeDerivative(_BoardNode):
    """``ddt(U)`` / ``rate(U)`` -- the left-hand side of a rate equation."""

    def __init__(self, state: Any) -> None:
        from .balance import Accumulation
        self.accumulation = state if isinstance(state, Accumulation) else Accumulation(state)
        self.state = self.accumulation.state

    def __repr__(self) -> str:
        return "ddt(%r)" % (self.state,)


class Unknown(_BoardNode):
    """A solve unknown: ``unknown("U*")`` in ``(I - dt*C) @ unknown("U*") == rhs``."""

    def __init__(self, name: Any) -> None:
        from pops.model import Handle

        if isinstance(name, Handle):
            if name.kind != "field":
                raise TypeError("Unknown Handle must be a declared field Handle")
            self.reference = name
            self.name = name.local_id
            return
        if not isinstance(name, str) or not name:
            raise TypeError("Unknown requires a non-empty name or a declared field Handle")
        self.reference = None
        self.name = name

    def __pops_ir_key__(self, recurse: Any) -> Any:
        # The surrounding FieldOperator owns the canonical, owner-qualified field Handle and
        # validates that this symbolic leaf denotes that exact declaration.  Keeping the leaf key
        # local makes it usable while the model is still authoring-owned without weakening the
        # resolved operator identity.
        return ("unknown", self.name)

    def __rmatmul__(self, operator: Any) -> Any:
        """``operator @ unknown("U*")`` -- the left-hand side of an implicit solve."""
        return OpApply(operator, self)

    def __mul__(self, coeff: Any) -> Any:
        # coeff * phi -- a zeroth-order reaction term (the k*phi of a screened Poisson).
        from .elliptic import Reaction
        return Reaction(self, coeff)

    def __rmul__(self, o: Any) -> Any:
        return self.__mul__(o)

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
