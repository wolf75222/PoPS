"""Expressions of the surface: an algebra over symbols, as on paper (spec 0.3, section 2.6).

An expression is a tree of operations over symbols (quantities, constants, parameters, data,
coordinates and the terms of a model) and Python numbers. It keeps what the user wrote, prints it
as mathematics (SU12) and is lowered into the expressions of the elaborated form only when a case
is elaborated.

SU4. The operators keep their meaning on paper: ``+ - * / **``, unary minus, ``abs``, the
comparisons ``< <= >= >``, which build predicates, and ``==``, which builds a relation. ``@``,
``&``, ``|``, ``~``, ``^`` and ``!=`` are refused (``NotAPopsOperation``) with the word to write,
and so is a NumPy function or ufunc given an expression, caught through ``__array_ufunc__`` and
``__array_function__``. SU11. An expression, a predicate or a relation has no Python truth value
(``SymbolicTruthValue``): ``if``, ``and``, ``or``, ``not``, ``assert``, a chained comparison and
Python's ``max`` and ``min`` would need one; so would ``float``, ``int`` and ``numpy.asarray``.
A handle hashes by identity, so a dictionary or a set keyed by handles compares by identity, while
``in`` on a list compares with ``==`` and is refused, as SU4 states.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.time._program.traced_method import TracedField, TracedScalar, TracedTime

from ._refusal import NotAPopsOperation, NotYetElaborated, SymbolicTruthValue

Number = (int, float, Fraction, Decimal)

# The values of a method body that N8 traces (SU10), which an expression of pops.math does not
# combine yet: a weight such as 1 - 1 / sqrt(2), or a duration such as exp(-step.dt).
_TRACED = ((TracedField, "the state"), (TracedScalar, "a duration"), (TracedTime, "a time"))
_SYMBOLIC_STEP = "a later lot of N8 with N11 (open point 16, SD95)"


def _symbolic_step(here: str) -> str:
    return ("a weight or a duration of SU10 that is an expression of pops.math, beyond the "
            "rationals and the polynomials of step.dt (here %s)" % here)

_TRUTH_WORDS = ("where(p, a, b) between values, all_of([...]), any_of([...]) or none_of([...]) "
                "between predicates, largest([...]) or smallest([...]) in place of max and min")

# The word of pops.math that a NumPy ufunc or function computes (SU4, 2.6.3).
_NUMPY_WORDS = {
    "sin": "sin(e)", "cos": "cos(e)", "tan": "tan(e)", "exp": "exp(e)", "expm1": "expm1(e)",
    "log": "log(e)", "log1p": "log1p(e)", "sqrt": "sqrt(e)", "cbrt": "cbrt(e)",
    "sign": "sign(e)", "arctan": "atan(e)", "arctan2": "atan2(y, x)", "sinh": "sinh(e)",
    "cosh": "cosh(e)", "tanh": "tanh(e)", "absolute": "abs(e)", "fabs": "abs(e)",
    "maximum": "largest([a, b])", "minimum": "smallest([a, b])", "fmax": "largest([a, b])",
    "fmin": "smallest([a, b])", "max": "largest([...])", "min": "smallest([...])",
    "amax": "largest([...])", "amin": "smallest([...])", "where": "where(p, a, b)",
    "clip": "clamp(e, lower, upper)", "dot": "dot(a, b)", "outer": "outer(a, b)",
    "matmul": "dot(a, b) or outer(a, b)", "power": "a ** b", "add": "a + b",
    "subtract": "a - b", "multiply": "a * b", "true_divide": "a / b", "divide": "a / b",
    "negative": "-a", "greater": "a > b", "greater_equal": "a >= b", "less": "a < b",
    "less_equal": "a <= b", "equal": "equal(a, b)", "not_equal": "none_of([equal(a, b)])",
    "logical_and": "all_of([p, q])", "logical_or": "any_of([p, q])",
    "logical_not": "none_of([p])", "sum": "sum([...]) over a static list of terms",
}


def _numpy_word(name: str) -> str:
    return _NUMPY_WORDS.get(name, "the word of pops.math that computes it (2.6.3)")


def is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def operand(value: Any, where: str) -> Any:
    """An operand of an expression: an expression or a Python number, never anything else."""
    if isinstance(value, Expr) or is_number(value):
        return value
    if isinstance(value, (Relation, Predicate)):
        raise NotAPopsOperation(
            "%s takes values; %r is a %s, which arithmetic does not combine"
            % (where, value, "relation" if isinstance(value, Relation) else "predicate"),
            write="the value itself, or where(p, a, b) to choose between two values")
    for kind, what in _TRACED:
        if isinstance(value, kind):
            raise NotYetElaborated(_symbolic_step("%s with %s of the method" % (where, what)),
                                   _SYMBOLIC_STEP)
    raise TypeError("%s takes PoPS values and Python numbers, got %r" % (where, value))


class _Symbolic:
    """What every symbolic value of the surface refuses (SU4, SU11)."""

    __slots__ = ()

    def __bool__(self) -> bool:
        raise SymbolicTruthValue(
            "%s is symbolic and has no Python truth value; if, and, or, not, assert, a chained "
            "comparison, max and min would need one" % (self,), write=_TRUTH_WORDS)

    def __float__(self) -> float:
        raise SymbolicTruthValue("%s is symbolic and has no Python number value" % (self,),
                                 write="the expression itself; PoPS evaluates it")

    def __int__(self) -> int:
        raise SymbolicTruthValue("%s is symbolic and has no Python number value" % (self,),
                                 write="the expression itself; PoPS evaluates it")

    __index__ = __int__

    def __complex__(self) -> complex:
        raise SymbolicTruthValue("%s is symbolic and has no Python number value" % (self,),
                                 write="the expression itself; PoPS evaluates it")

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        raise SymbolicTruthValue("%s is symbolic and has no NumPy array value" % (self,),
                                 write="the expression itself, with the words of pops.math")

    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any) -> Any:
        name = getattr(ufunc, "__name__", str(ufunc))
        raise NotAPopsOperation(
            "the NumPy function %s does not read PoPS values" % name, write=_numpy_word(name))

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        name = getattr(func, "__name__", str(func))
        raise NotAPopsOperation(
            "the NumPy function %s does not read PoPS values" % name, write=_numpy_word(name))

    def __ne__(self, other: Any) -> Any:
        raise NotAPopsOperation("!= is not an operation of PoPS; a relation is written with ==",
                                write="none_of([equal(a, b)])")

    def __and__(self, other: Any) -> Any:
        raise NotAPopsOperation("& is not an operation of PoPS", write="all_of([p, q])")

    __rand__ = __and__

    def __or__(self, other: Any) -> Any:
        raise NotAPopsOperation("| is not an operation of PoPS", write="any_of([p, q])")

    __ror__ = __or__

    def __xor__(self, other: Any) -> Any:
        raise NotAPopsOperation(
            "^ is not an operation of PoPS",
            write="any_of([all_of([p, none_of([q])]), all_of([q, none_of([p])])])")

    __rxor__ = __xor__

    def __invert__(self) -> Any:
        raise NotAPopsOperation("~ is not an operation of PoPS", write="none_of([p])")

    def __matmul__(self, other: Any) -> Any:
        raise NotAPopsOperation("@ is not an operation of PoPS", write="dot(a, b) or outer(a, b)")

    __rmatmul__ = __matmul__

    def __iter__(self) -> Any:
        raise TypeError("%s is one symbolic value, not a Python sequence" % (self,))

    def __repr__(self) -> str:
        return self._text(0)

    __str__ = __repr__

    def _text(self, precedence: int) -> str:  # pragma: no cover - every subclass defines it
        raise NotImplementedError

    __hash__ = object.__hash__


def text(value: Any, precedence: int = 0) -> str:
    """The mathematics of a value as the user wrote it (SU12)."""
    if isinstance(value, _Symbolic):
        return value._text(precedence)
    if isinstance(value, Fraction):
        body = str(value) if value.denominator != 1 else str(value.numerator)
        return "(%s)" % body if precedence > 2 and value.denominator != 1 else body
    if is_number(value) and value < 0 and precedence > 1:
        return "(%r)" % value
    return repr(value) if isinstance(value, float) else str(value)


class Expr(_Symbolic):
    """A scalar or vector field, a constant or a term, as an expression."""

    __slots__ = ()

    def __pops_scalar_literal__(self) -> dict:
        """The trace of N8 asks a weight or a duration of a method for its number (SU10): an
        expression of pops.math has none at this level, whichever side of the product it is."""
        raise NotYetElaborated(_symbolic_step("%s in a step of the method" % (self,)),
                               _SYMBOLIC_STEP)

    def __add__(self, other: Any) -> Any:
        return Arith("+", (self, operand(other, "+")))

    def __radd__(self, other: Any) -> Any:
        return Arith("+", (operand(other, "+"), self))

    def __sub__(self, other: Any) -> Any:
        return Arith("-", (self, operand(other, "-")))

    def __rsub__(self, other: Any) -> Any:
        return Arith("-", (operand(other, "-"), self))

    def __mul__(self, other: Any) -> Any:
        return Arith("*", (self, operand(other, "*")))

    def __rmul__(self, other: Any) -> Any:
        return Arith("*", (operand(other, "*"), self))

    def __truediv__(self, other: Any) -> Any:
        return Arith("/", (self, operand(other, "/")))

    def __rtruediv__(self, other: Any) -> Any:
        return Arith("/", (operand(other, "/"), self))

    def __pow__(self, other: Any) -> Any:
        return Arith("**", (self, operand(other, "**")))

    def __rpow__(self, other: Any) -> Any:
        return Arith("**", (operand(other, "**"), self))

    def __neg__(self) -> Any:
        return Arith("neg", (self,))

    def __pos__(self) -> Any:
        return self

    def __abs__(self) -> Any:
        return Call("abs", (self,))

    def __lt__(self, other: Any) -> Any:
        return Predicate("<", self, operand(other, "<"))

    def __le__(self, other: Any) -> Any:
        return Predicate("<=", self, operand(other, "<="))

    def __gt__(self, other: Any) -> Any:
        return Predicate(">", self, operand(other, ">"))

    def __ge__(self, other: Any) -> Any:
        return Predicate(">=", self, operand(other, ">="))

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return Relation(self, operand(other, "=="))

    __hash__ = object.__hash__

    def children(self) -> tuple:
        return ()


_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "**": 4}


class Arith(Expr):
    """``a + b``, ``a - b``, ``a * b``, ``a / b``, ``a ** b`` and ``-a``."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, args: tuple) -> None:
        self.op = op
        self.args = tuple(args)

    def children(self) -> tuple:
        return self.args

    def _text(self, precedence: int) -> str:
        mine = _PRECEDENCE[self.op]
        if self.op == "neg":
            body = "-" + text(self.args[0], mine)
        elif self.op == "**":
            body = "%s**%s" % (text(self.args[0], mine + 1), text(self.args[1], mine))
        else:
            left, right = self.args
            # a - (b + c) and a / (b * c) keep their parentheses on the right
            right_precedence = mine + 1 if self.op in ("-", "/") else mine
            body = "%s %s %s" % (text(left, mine), self.op, text(right, right_precedence))
        return "(%s)" % body if mine < precedence else body


class Call(Expr):
    """A word of pops.math applied to its operands: ``exp(e)``, ``abs(e)``, ``grad(e)``, ..."""

    __slots__ = ("word", "args")

    def __init__(self, word: str, args: tuple) -> None:
        self.word = word
        self.args = tuple(args)

    def children(self) -> tuple:
        return self.args

    def _text(self, precedence: int) -> str:
        return "%s(%s)" % (self.word, ", ".join(text(arg) for arg in self.args))


class Relation(_Symbolic):
    """``a == b``: a relation, which has no orientation (SU4)."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs: Any, rhs: Any) -> None:
        self.lhs = lhs
        self.rhs = rhs

    def sides(self) -> tuple:
        return (self.lhs, self.rhs)

    def _text(self, precedence: int) -> str:
        return "%s == %s" % (text(self.lhs), text(self.rhs))

    __hash__ = object.__hash__


class Predicate(_Symbolic):
    """``a < b`` and the other comparisons, and the combinations of predicates."""

    __slots__ = ("op", "args")

    def __init__(self, op: str, *args: Any) -> None:
        self.op = op
        self.args = tuple(args)

    def _text(self, precedence: int) -> str:
        if self.op in ("all_of", "any_of", "none_of"):
            return "%s([%s])" % (self.op, ", ".join(text(arg) for arg in self.args))
        if self.op == "equal":
            return "equal(%s, %s)" % (text(self.args[0]), text(self.args[1]))
        return "%s %s %s" % (text(self.args[0], 1), self.op, text(self.args[1], 1))

    __hash__ = object.__hash__


def walk(value: Any) -> Any:
    """Every node of an expression, the root first, each node once: a node that several others
    share (a Python variable read twice, as the body of ``weno5`` reads its subexpressions) is
    visited at its first occurrence only, so that the walk is linear in the nodes and not in the
    unfolded tree (N3 lot 3)."""
    stack = [value]
    seen: set = set()
    while stack:
        node = stack.pop()
        if isinstance(node, (Expr, Predicate, Relation)):
            if id(node) in seen:
                continue
            seen.add(id(node))
        yield node
        if isinstance(node, Expr):
            stack.extend(reversed(node.children()))
        elif isinstance(node, Predicate):
            stack.extend(reversed(node.args))
        elif isinstance(node, Relation):
            stack.extend((node.rhs, node.lhs))
