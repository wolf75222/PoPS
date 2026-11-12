"""The words of ``pops.math`` (spec 0.3, 2.6.3), the one module meant to be star-imported (SU13).

This lot elaborates the field algebra (``+ - * / **``, unary minus, ``abs``, ``pi`` and Python's
``Fraction``, which this module re-exports so that a case that star-imports it keeps its numbers
exact), the elementary functions that the expressions of b6 lower (``exp``, ``expm1``, ``log``,
``sqrt``, ``sin``, ``cos``; ``erf`` and ``erfc`` in the initial values of ``pops.project``
only, lot D-E), the differential words of a balance (``ddt``, ``grad``, ``div``) and
the predicate words, whose refusals SU4 and SU11 state (``where``, ``equal``, ``all_of``,
``any_of``, ``none_of``), in the body of a scheme (lot 2), ``largest``, ``smallest`` and
``dot(term.law(...), face.normal)``, and ``outflow(q)``, the relation of ``model.on`` whose piece
carries no data (lot B). Every other word of 2.6.3 is importable and refuses its call with
``NotYetElaborated``, which names the lot that elaborates it, ``curl`` included:
CF2 admits the definition of a word of 2.6.3 in this module, by a top-level ``def`` or assignment,
and refuses the word everywhere else (FC2, R20).

An unknown name, or a name of an earlier text, is refused on import with ``SymbolNotInSurface``,
a subclass of ``ImportError`` whose message names the word to write (SU13, §21.7).
"""
from fractions import Fraction
from typing import Any as _Any

from ._expr import Call as _Call
from ._expr import Expr as _Expr
from ._expr import Predicate as _Predicate
from ._expr import Relation as _Relation
from ._expr import operand as _operand
from ._geometry import infinity, pi
from ._refusal import NotYetElaborated as _NotYetElaborated
from ._refusal import SpellingRefusal as _SpellingRefusal
from ._refusal import SymbolNotInSurface as _SymbolNotInSurface
from ._refusal import surface_entry as _surface_entry

_ELEMENTARY = ("exp", "expm1", "log", "sqrt", "sin", "cos")
_INITIAL_ONLY = ("erf", "erfc")

_N11 = "N11 (expression closure, C1)"
_NONLOCAL = "a later lot of N15 with the nonlocal terms of G05"
_TRACES = "a later lot of N15 with the boundary relations of G04"
_LATER = {
    "log1p": _N11, "cbrt": _N11, "sign": _N11, "clamp": _N11, "tan": _N11, "atan": _N11,
    "atan2": _N11, "sinh": _N11, "cosh": _N11, "tanh": _N11,
    "component": _N11, "unit": _N11,
    "identity": _N11, "outer": _N11, "contract": _N11, "transpose": _N11,
    "matrix_trace": _N11, "determinant": _N11, "inverse": _N11, "solve_small": _N11,
    "symmetric_eigen": _N11, "norm": _N11, "reflect": _N11, "sum_over": _N11,
    "derivative": _N11, "jvp": _N11, "vjp": _N11,
    "integral": _NONLOCAL, "mean": _NONLOCAL, "restrict": _NONLOCAL,
    "trace": _TRACES, "normal": _TRACES, "jump": _TRACES, "minus_side": _TRACES,
    "plus_side": _TRACES, "incoming_projector": _TRACES,
}

# Earlier spellings that this module refuses on import, with the word to write (2.6.3, §21.7).
_EARLIER = {
    "Q": _SpellingRefusal("Fraction(1, 2), Python's exact fraction"),
    "const": _SpellingRefusal("the number itself, or model.constant(name, value=v) for a value of "
                             "the physics"),
    "value": _SpellingRefusal("the constant or the parameter itself: a constant or a parameter of "
                             "the model is a value"),
    "laplacian": _SpellingRefusal("div(grad(e))"),
    "maximum": _SpellingRefusal("largest([a, b, ...])"),
    "minimum": _SpellingRefusal("smallest([a, b, ...])"),
    "lt": _SpellingRefusal("a < b"), "le": _SpellingRefusal("a <= b"),
    "ge": _SpellingRefusal("a >= b"), "gt": _SpellingRefusal("a > b"),
    "not_": _SpellingRefusal("none_of([p])"),
    "cond": _SpellingRefusal("where(p, a, b) between values, or pops.choose(p, then=f, "
                            "otherwise=g) between computations with effects"),
    "trace_t": _SpellingRefusal("matrix_trace(e)"), "det": _SpellingRefusal("determinant(e)"),
    "inv": _SpellingRefusal("inverse(e)"), "eig_sym": _SpellingRefusal("symmetric_eigen(e)"),
    "D": _SpellingRefusal("derivative(e, wrt=q)"),
    "moment": _SpellingRefusal("integral(w * f, over=v), v the support integrated over"),
    "pullback": _SpellingRefusal("the field itself: a field read on a product support is pulled "
                                "back along its unique projection"),
    "side": _SpellingRefusal("minus_side(e) or plus_side(e)"),
    "normal_flux": _SpellingRefusal("dot(F, normal())"),
    "normal_component": _SpellingRefusal("dot(e, normal())"),
}


def _value(value: _Any, word: str) -> _Any:
    return _operand(value, word)


@_surface_entry
def ddt(q: _Any) -> _Expr:
    """``ddt(q)``: the time derivative of an evolved quantity, the side of its balance (SU4)."""
    from ._model import Quantity

    if not isinstance(q, Quantity):
        raise TypeError("ddt(q) takes a quantity of the model, got %r" % (q,))
    return _Call("ddt", (q,))


@_surface_entry
def grad(e: _Any) -> _Expr:
    """``grad(e)``: the gradient of a field."""
    return _Call("grad", (_value(e, "grad"),))


@_surface_entry
def div(e: _Any) -> _Expr:
    """``div(e)``: the divergence of a flux, or of a sum of fluxes."""
    return _Call("div", (_value(e, "div"),))


def _elementary(word: str) -> _Any:
    @_surface_entry
    def function(e: _Any) -> _Any:
        hook = getattr(type(e), "__pops_elementary__", None)
        if hook is not None:
            # a scalar of a solver computes its own elementary function (N9 lot 4)
            return hook(e, word)
        return _Call(word, (_value(e, word),))

    function.__name__ = function.__qualname__ = word
    function.__doc__ = "``%s(e)``, pointwise (2.6.3)." % word
    return function


exp, expm1, log, sqrt, sin, cos = (_elementary(word) for word in _ELEMENTARY)
# erf and erfc build their call; the analytic initial values of pops.project lower them (lot D-E,
# the exact cell integral of a Gaussian, CN12), and every other value refuses them with N11.
erf, erfc = (_elementary(word) for word in _INITIAL_ONLY)


def _predicate(value: _Any, word: str) -> _Predicate:
    if isinstance(value, _Relation):
        raise TypeError("%s takes predicates; %s is a relation" % (word, value))
    if not isinstance(value, _Predicate):
        raise TypeError("%s takes predicates, built by < <= >= > and equal(a, b); got %r"
                        % (word, value))
    return value


def _relation_refusal(relation: _Relation) -> _Any:
    from ._refusal import NotAPopsOperation

    lhs, rhs = relation.sides()
    return NotAPopsOperation(
        "`%s` is a relation, and where(p, a, b) chooses by a predicate" % relation,
        write="equal(%s, %s) or abs(%s - %s) <= tolerance" % (lhs, rhs, lhs, rhs))


@_surface_entry
def where(p: _Any, a: _Any, b: _Any) -> _Expr:
    """``where(p, a, b)``: a where the predicate p holds, b elsewhere."""
    if isinstance(p, _Relation):
        raise _relation_refusal(p)
    return _Call("where", (_predicate(p, "where(p, a, b)"), _value(a, "where"), _value(b, "where")))


@_surface_entry
def equal(a: _Any, b: _Any) -> _Predicate:
    """``equal(a, b)``: the predicate a = b."""
    return _Predicate("equal", _value(a, "equal"), _value(b, "equal"))


def _combination(word: str) -> _Any:
    @_surface_entry
    def combine(predicates: _Any) -> _Predicate:
        if isinstance(predicates, tuple):
            raise TypeError("%s takes a list of predicates, %s([...]); a tuple is a point, a size "
                            "or an interval (SU14)" % (word, word))
        if not isinstance(predicates, list) or not predicates:
            raise TypeError("%s takes a nonempty list of predicates, %s([p, q])" % (word, word))
        for item in predicates:
            if isinstance(item, _Relation):
                raise _relation_refusal(item)
        return _Predicate(word, *(_predicate(item, word) for item in predicates))

    combine.__name__ = combine.__qualname__ = word
    return combine


all_of, any_of, none_of = (_combination(word) for word in ("all_of", "any_of", "none_of"))


def _extremum(word: str) -> _Any:
    @_surface_entry
    def extremum(values: _Any) -> _Expr:
        if isinstance(values, tuple):
            raise TypeError("%s takes a list of values, %s([a, b]); a tuple is a point, a size or "
                            "an interval (SU14)" % (word, word))
        if not isinstance(values, list) or not values:
            raise TypeError("%s takes a nonempty list of values, %s([a, b])" % (word, word))
        return _Call(word, tuple(_value(item, word) for item in values))

    extremum.__name__ = extremum.__qualname__ = word
    extremum.__doc__ = ("``%s([a, b, ...])``, pointwise; lot 2 of N15 lowers it in the body of a "
                        "scheme (b6 ``%s``), its derivative being the lot 2 of N11 (2.6.3)."
                        % (word, "maximum" if word == "largest" else "minimum"))
    return extremum


largest, smallest = (_extremum(word) for word in ("largest", "smallest"))


@_surface_entry
def vector(*components: _Any) -> _Expr:
    """``vector(c_1, ..., c_d)``, a vector of one component per axis. Lot 2b of N15 elaborates it
    as the value of a flux, ``model.flux(of=u, value=vector(a, b) * u)``, and in
    ``dot(vector(...), face.normal)`` in a scheme; its other uses belong to N11 (2.6.3)."""
    if len(components) == 1 and isinstance(components[0], (list, tuple)):
        raise TypeError("vector(c_1, ..., c_d) takes its components as arguments, "
                        "vector(a, b), not a sequence")
    if not components:
        raise TypeError("vector(c_1, ..., c_d) takes one component per axis")
    return _Call("vector", tuple(_value(item, "vector") for item in components))


@_surface_entry
def dot(a: _Any, b: _Any) -> _Any:
    """``dot(a, b)``; lot 2 of N15 elaborates ``dot(term.law(...), face.normal)``, the normal
    component of the flux of the key in a face scheme, and leaves the others to N11 (2.6.3)."""
    from ._scheme import dot as _dot

    return _dot(a, b)


def _later(word: str, owner: str) -> _Any:
    def later(*args: _Any, **kwargs: _Any) -> _Any:
        raise _NotYetElaborated("the word %s of pops.math" % word, owner).with_traceback(None)

    later.__name__ = later.__qualname__ = word
    later.__doc__ = "``%s``, a word of 2.6.3 that a later lot elaborates (%s)." % (word, owner)
    return later


for _word, _owner in _LATER.items():
    globals()[_word] = _later(_word, _owner)

# CF2 admits these two words only as a top-level definition of this module (FC2, R20).
curl = _later("curl", _N11)


def outflow(q: _Any) -> _Any:
    """``outflow(q)``: the empty relation on the trace of q, which states that a piece of
    ``model.on`` carries no data (R20, SD43); its form is ``NoData`` (lot B of N15)."""
    from ._model import NoDataRelation

    return NoDataRelation(q, "outflow")


__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    "Fraction", "pi", "infinity", "ddt", "grad", "div", "curl", *_ELEMENTARY, *_INITIAL_ONLY,
    "where", "equal",
    "all_of", "any_of", "none_of", "largest", "smallest", "vector", "dot", "outflow", *_LATER]


def __getattr__(name: str) -> _Any:
    """Refuse a name that pops.math does not list (SU3, SU13, §21.7); export nothing."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _EARLIER:
        raise _EARLIER[name].refuse(name, "pops.math")
    raise _SymbolNotInSurface("pops.math has no word %s; its words are %s"
                             % (name, ", ".join(__all__)),
                             write="one of these words, or Python's own numbers and operators")
