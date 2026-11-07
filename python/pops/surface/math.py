"""The words of ``pops.math`` (spec 0.3, 2.6.3), the one module meant to be star-imported (SU13).

This lot elaborates the field algebra (``+ - * / **``, unary minus, ``abs``, ``pi`` and Python's
``Fraction``, which this module re-exports so that a case that star-imports it keeps its numbers
exact), the elementary functions that the expressions of b6 lower (``exp``, ``expm1``, ``log``,
``sqrt``, ``sin``, ``cos``), the differential words of a balance (``ddt``, ``grad``, ``div``) and
the predicate words, whose refusals SU4 and SU11 state (``where``, ``equal``, ``all_of``,
``any_of``, ``none_of``). Every other word of 2.6.3 is importable and refuses its call with
``NotYetElaborated``, which names the lot that elaborates it. Two words of 2.6.3 are not here:
the core of PoPS may not name them (CF2 refuses them as named operators and boundary families in
the core paths), so they wait for ``pops.lib.math``, which this module will re-export.

An unknown name, or a name of an earlier text, is refused on import with ``SymbolNotInSurface``,
a subclass of ``ImportError`` whose message names the word to write (SU13, §21.7).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._expr import Call, Expr, Predicate, Relation, is_number, operand
from ._geometry import infinity, pi
from ._refusal import NotYetElaborated, SpellingRefusal, SymbolNotInSurface, surface_entry

_ELEMENTARY = ("exp", "expm1", "log", "sqrt", "sin", "cos")

_N11 = "N11 (expression closure, C1)"
_NONLOCAL = "a later lot of N15 with the nonlocal terms of G05"
_TRACES = "a later lot of N15 with the boundary relations of G04"
_LATER = {
    "log1p": _N11, "cbrt": _N11, "sign": _N11, "clamp": _N11, "tan": _N11, "atan": _N11,
    "atan2": _N11, "sinh": _N11, "cosh": _N11, "tanh": _N11, "erf": _N11, "erfc": _N11,
    "largest": _N11, "smallest": _N11, "vector": _N11, "component": _N11, "unit": _N11,
    "identity": _N11, "dot": _N11, "outer": _N11, "contract": _N11, "transpose": _N11,
    "matrix_trace": _N11, "determinant": _N11, "inverse": _N11, "solve_small": _N11,
    "symmetric_eigen": _N11, "norm": _N11, "reflect": _N11, "sum_over": _N11,
    "derivative": _N11, "jvp": _N11, "vjp": _N11,
    "integral": _NONLOCAL, "mean": _NONLOCAL, "restrict": _NONLOCAL,
    "trace": _TRACES, "normal": _TRACES, "jump": _TRACES, "minus_side": _TRACES,
    "plus_side": _TRACES, "incoming_projector": _TRACES,
}

# Earlier spellings that this module refuses on import, with the word to write (2.6.3, §21.7).
_EARLIER = {
    "Q": SpellingRefusal("Fraction(1, 2), Python's exact fraction"),
    "const": SpellingRefusal("the number itself, or model.constant(name, value=v) for a value of "
                             "the physics"),
    "value": SpellingRefusal("the constant or the parameter itself: a constant or a parameter of "
                             "the model is a value"),
    "maximum": SpellingRefusal("largest([a, b, ...])"),
    "minimum": SpellingRefusal("smallest([a, b, ...])"),
    "lt": SpellingRefusal("a < b"), "le": SpellingRefusal("a <= b"),
    "ge": SpellingRefusal("a >= b"), "gt": SpellingRefusal("a > b"),
    "not_": SpellingRefusal("none_of([p])"),
    "cond": SpellingRefusal("where(p, a, b) between values, or pops.choose(p, then=f, "
                            "otherwise=g) between computations with effects"),
    "trace_t": SpellingRefusal("matrix_trace(e)"), "det": SpellingRefusal("determinant(e)"),
    "inv": SpellingRefusal("inverse(e)"), "eig_sym": SpellingRefusal("symmetric_eigen(e)"),
    "D": SpellingRefusal("derivative(e, wrt=q)"),
    "moment": SpellingRefusal("integral(w * f, over=v), v the support integrated over"),
    "pullback": SpellingRefusal("the field itself: a field read on a product support is pulled "
                                "back along its unique projection"),
    "side": SpellingRefusal("minus_side(e) or plus_side(e)"),
    "normal_flux": SpellingRefusal("dot(F, normal())"),
    "normal_component": SpellingRefusal("dot(e, normal())"),
}


def _value(value: Any, word: str) -> Any:
    return operand(value, word)


@surface_entry
def ddt(q: Any) -> Expr:
    """``ddt(q)``: the time derivative of an evolved quantity, the side of its balance (SU4)."""
    from ._model import Quantity

    if not isinstance(q, Quantity):
        raise TypeError("ddt(q) takes a quantity of the model, got %r" % (q,))
    return Call("ddt", (q,))


@surface_entry
def grad(e: Any) -> Expr:
    """``grad(e)``: the gradient of a field."""
    return Call("grad", (_value(e, "grad"),))


@surface_entry
def div(e: Any) -> Expr:
    """``div(e)``: the divergence of a flux, or of a sum of fluxes."""
    return Call("div", (_value(e, "div"),))


def _elementary(word: str) -> Any:
    @surface_entry
    def function(e: Any) -> Expr:
        return Call(word, (_value(e, word),))

    function.__name__ = function.__qualname__ = word
    function.__doc__ = "``%s(e)``, pointwise (2.6.3)." % word
    return function


exp, expm1, log, sqrt, sin, cos = (_elementary(word) for word in _ELEMENTARY)


def _predicate(value: Any, word: str) -> Predicate:
    if isinstance(value, Relation):
        raise TypeError("%s takes predicates; %s is a relation" % (word, value))
    if not isinstance(value, Predicate):
        raise TypeError("%s takes predicates, built by < <= >= > and equal(a, b); got %r"
                        % (word, value))
    return value


def _relation_refusal(relation: Relation) -> Any:
    from ._refusal import NotAPopsOperation

    lhs, rhs = relation.sides()
    return NotAPopsOperation(
        "`%s` is a relation, and where(p, a, b) chooses by a predicate" % relation,
        write="equal(%s, %s) or abs(%s - %s) <= tolerance" % (lhs, rhs, lhs, rhs))


@surface_entry
def where(p: Any, a: Any, b: Any) -> Expr:
    """``where(p, a, b)``: a where the predicate p holds, b elsewhere."""
    if isinstance(p, Relation):
        raise _relation_refusal(p)
    return Call("where", (_predicate(p, "where(p, a, b)"), _value(a, "where"), _value(b, "where")))


@surface_entry
def equal(a: Any, b: Any) -> Predicate:
    """``equal(a, b)``: the predicate a = b."""
    return Predicate("equal", _value(a, "equal"), _value(b, "equal"))


def _combination(word: str) -> Any:
    @surface_entry
    def combine(predicates: Any) -> Predicate:
        if isinstance(predicates, tuple):
            raise TypeError("%s takes a list of predicates, %s([...]); a tuple is a point, a size "
                            "or an interval (SU14)" % (word, word))
        if not isinstance(predicates, list) or not predicates:
            raise TypeError("%s takes a nonempty list of predicates, %s([p, q])" % (word, word))
        for item in predicates:
            if isinstance(item, Relation):
                raise _relation_refusal(item)
        return Predicate(word, *(_predicate(item, word) for item in predicates))

    combine.__name__ = combine.__qualname__ = word
    return combine


all_of, any_of, none_of = (_combination(word) for word in ("all_of", "any_of", "none_of"))


def _later(word: str, owner: str) -> Any:
    def later(*args: Any, **kwargs: Any) -> Any:
        raise NotYetElaborated("the word %s of pops.math" % word, owner).with_traceback(None)

    later.__name__ = later.__qualname__ = word
    later.__doc__ = "``%s``, a word of 2.6.3 that a later lot elaborates (%s)." % (word, owner)
    return later


for _word, _owner in _LATER.items():
    globals()[_word] = _later(_word, _owner)

__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    "Fraction", "pi", "infinity", "ddt", "grad", "div", *_ELEMENTARY, "where", "equal", "all_of",
    "any_of", "none_of", *_LATER]


def __getattr__(name: str) -> Any:
    """Refuse a name that pops.math does not list (SU3, SU13, §21.7); export nothing."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _EARLIER:
        raise _EARLIER[name].refuse(name, "pops.math")
    raise SymbolNotInSurface("pops.math has no word %s; its words are %s"
                             % (name, ", ".join(__all__)),
                             write="one of these words, or Python's own numbers and operators")


def is_symbolic(value: Any) -> bool:
    """Whether a value is an expression of the surface rather than a Python number."""
    return isinstance(value, Expr) and not is_number(value)
