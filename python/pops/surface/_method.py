"""Time methods of the surface, Π (spec 0.3, 2.6.2, 2.6.6; SU5, SU10, T4, T33, T34).

``@pops.method def name(step, rate): ... return pops.Candidate(state)`` records a method. PoPS
supplies ``step``; every other parameter is an option, given by keyword at the call:
``ssprk2(rate=model)`` (SU5, SU8). A rate is a model, a term or a list of terms; in the body it is
called with a state and then its time, ``rate(u, t)`` (T4). The body is written with the step
algebra of SU10: a rate times a duration is an increment, a state plus an increment is a state.

The elaborated form is the traced method of N8, ``@method def f(B, p, flow)``, with ``step.state``
= ``B.state``, ``step.time`` = ``B.time`` and ``step.dt`` = ``p.dt``; a rate option becomes a
component of the method, whose evaluation is the rate of the terms that the option names (2.6.6).
The body is traced when its case is traced (SU5), once, by the trace of N8, which checks the step
algebra and the closed body. A refusal of that trace is reported at the line of the body that
caused it, or at the line of ``pops.Candidate`` when it concerns the candidate (SU11).
"""
from __future__ import annotations

import inspect
import re
from typing import Any

from pops._ir.symbolic import SymbolicTruthValueError
from pops.time._program import traced_method as _n8

from ._model import Balance, Model, Term
from ._refusal import (
    ClosedBodyViolation, IncrementNotDerivable, IncrementTypeMismatch, NotYetElaborated, Site,
    SurfaceRefusal, SymbolicTruthValue, site_of_traceback, user_site,
)

# SU5: the parameters that PoPS supplies, named after an entity.
ENTITY_PARAMETERS = ("step", "candidate", "cell", "face", "edge", "axis", "depth", "interval",
                     "incident", "term", "region", "problem", "payload")
_N8_LATER = "a later lot of N8"


class SurfaceMethod:
    """A function decorated with ``@pops.method``: its body, its name and its options."""

    __slots__ = ("body", "name", "options", "defaults", "site")

    def __init__(self, body: Any) -> None:
        if not callable(body) or not hasattr(body, "__code__"):
            raise TypeError("@pops.method decorates a function def name(step, ...)")
        parameters = list(inspect.signature(body).parameters.values())
        if not parameters or parameters[0].name != "step":
            raise TypeError("@pops.method: the method %s takes step first, the step that PoPS "
                            "supplies: def %s(step, rate)" % (body.__name__, body.__name__))
        for parameter in parameters:
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise TypeError("@pops.method: %s takes named options, not *%s (SU4)"
                                % (body.__name__, parameter.name))
        options = parameters[1:]
        for parameter in options:
            if parameter.name == "region":
                raise NotYetElaborated("a method with a region parameter, level 3 (SU15)",
                                       "a later lot of N15")
            if parameter.name in ENTITY_PARAMETERS:
                raise TypeError("@pops.method: PoPS supplies step to a method, not %s; an option "
                                "of %s takes another name" % (parameter.name, body.__name__))
        self.body = body
        self.name = body.__name__
        self.options = tuple(parameter.name for parameter in options)
        self.defaults = {parameter.name: parameter.default for parameter in options
                         if parameter.default is not parameter.empty}
        self.site = user_site(2)

    def __call__(self, *args: Any, **options: Any) -> MethodValue:
        if args:
            raise TypeError("%s takes its options by keyword, %s(%s=...) (SU8)"
                            % (self.name, self.name, self.options[0] if self.options else "rate"))
        unknown = sorted(set(options) - set(self.options))
        if unknown:
            raise TypeError("%s has the options %s, not %s"
                            % (self.name, ", ".join(self.options) or "none", ", ".join(unknown)))
        missing = [name for name in self.options
                   if name not in options and name not in self.defaults]
        if missing:
            raise TypeError("%s needs the options %s, given by keyword: %s(%s)"
                            % (self.name, ", ".join(missing), self.name,
                               ", ".join("%s=..." % name for name in missing)))
        for name, value in options.items():
            if isinstance(value, tuple):
                raise TypeError("%s(%s=(...)): a tuple is a point, a size or an interval; several "
                                "terms are a list, %s=[...] (SU14)" % (self.name, name, name))
        return MethodValue(self, {**self.defaults, **options}, user_site(2))

    def __repr__(self) -> str:
        return "@pops.method %s" % self.name


class MethodValue:
    """A method called with its options, ``ssprk2(rate=model)``: what ``pops.Advance`` takes."""

    __slots__ = ("method", "options", "site")

    def __init__(self, method: SurfaceMethod, options: dict, site: Site | None) -> None:
        self.method = method
        self.options = options
        self.site = site

    @property
    def name(self) -> str:
        return self.method.name

    def rate_options(self) -> tuple:
        """The options that are rates: a model, a term or a list of terms (SU5, 2.6.6)."""
        return tuple(name for name in self.method.options if is_rate(self.options[name]))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotYetElaborated("method.%s, the name of a solve of a method (SU8)" % name,
                               "a later lot of N15 with the solves of N9 and G10")

    def __repr__(self) -> str:
        return "%s(%s)" % (self.method.name, ", ".join("%s=%r" % item
                                                        for item in self.options.items()))


def is_rate(value: Any) -> bool:
    if isinstance(value, (Model, Term)):
        return True
    return isinstance(value, list) and bool(value) and all(isinstance(item, Term) for item in value)


def method(function: Any = None, **options: Any) -> Any:
    """``@pops.method``, written bare (SU16)."""
    if options or function is None:
        raise TypeError("@pops.method takes no option and is written bare, @pops.method (SU16)")
    if isinstance(function, SurfaceMethod):
        return function
    return SurfaceMethod(function)


# --- the step and the rates inside a body ---------------------------------------------------------

class Step:
    """``step`` in a method: the accepted state and its time, and the proposed duration."""

    __slots__ = ("_B", "_p")

    def __init__(self, B: Any, p: Any) -> None:
        self._B = B
        self._p = p

    @property
    def state(self) -> Any:
        return self._B.state

    @property
    def time(self) -> Any:
        return self._B.time

    @property
    def dt(self) -> Any:
        return self._p.dt

    @property
    def tolerance(self) -> Any:
        raise NotYetElaborated("step.tolerance, the request of the step (SV3)",
                               "a later lot of N8 with the requests of N9")

    @property
    def memory(self) -> Any:
        raise NotYetElaborated("step.memory (T27)", _N8_LATER)

    def previous(self, *args: Any) -> Any:
        raise NotYetElaborated("step.previous(k), a method that remembers past steps (T27)",
                               _N8_LATER)

    @property
    def inputs(self) -> Any:
        raise NotYetElaborated("step.inputs, the inputs of a child step (T35)", _N8_LATER)

    @property
    def choices(self) -> Any:
        raise NotYetElaborated("step.choices (T28)", _N8_LATER)

    def __repr__(self) -> str:
        return "step"


class Rate:
    """A rate option inside a body: ``rate(u, t)``, the rate at a state and its time (T4, SU5)."""

    __slots__ = ("_name", "_component")

    def __init__(self, name: str, component: Any) -> None:
        self._name = name
        self._component = component

    def __call__(self, *args: Any, **options: Any) -> Any:
        if options:
            raise NotYetElaborated("%s(X, t, %s=...)" % (self._name, ", ".join(sorted(options))),
                                   "a later lot of N8 with the roles and the fields of T5, T6 "
                                   "and T18")
        if len(args) != 2:
            raise TypeError("%s(X, t): each call states the state and then its time, such as "
                            "%s(step.state, step.time) (T4)" % (self._name, self._name))
        return self._component(*args)

    def __repr__(self) -> str:
        return self._name


class Candidate:
    """``pops.Candidate(state)``: the candidate state of a method, at the time where it ends."""

    __slots__ = ("state", "site")

    def __init__(self, state: Any, *, increment: Any = None, memory: Any = None,
                 observations: Any = None, evidence: Any = None) -> None:
        for name, value in (("increment", increment), ("memory", memory),
                            ("observations", observations), ("evidence", evidence)):
            if value is not None:
                raise NotYetElaborated("pops.Candidate(state, %s=...)" % name, _N8_LATER)
        if not isinstance(state, _n8.TracedField):
            raise TypeError("pops.Candidate(state) takes a state of the method, built from "
                            "step.state with the step algebra of SU10, got %r" % (state,))
        self.state = state
        self.site = user_site(2)


# --- elaboration onto the traced methods of N8 ----------------------------------------------

_N8_REFUSALS = (
    (_n8.IncrementTypeMismatch, IncrementTypeMismatch,
     "a rate times a duration, step.dt * rate(u, t), added to a state, and states combined with "
     "weights that sum to one"),
    (_n8.IncrementNotDerivable, IncrementNotDerivable,
     "a candidate built from rates, or pops.exact_flow(method, of=term) for a closed form"),
    (_n8.ClosedBodyViolation, ClosedBodyViolation,
     "a body that reads only step, its options and immutable values"),
)


class _Trace:
    """What the elaboration of one method needs to report a refusal at the user's line."""

    __slots__ = ("candidate_site", "site")

    def __init__(self, site: Site | None) -> None:
        self.candidate_site: Site | None = None
        self.site = site


def translate(error: BaseException, trace: _Trace) -> BaseException:
    """The refusal of the surface for an error of the trace of N8, at the user's line (SU11)."""
    if isinstance(error, SurfaceRefusal):
        return error
    site = site_of_traceback(error.__traceback__) or trace.candidate_site or trace.site
    if isinstance(error, SymbolicTruthValueError):
        return SymbolicTruthValue(
            "a state, a rate or a time of a method is symbolic and has no Python truth value",
            write="where(p, a, b) between values; the tableau of a method is Python data",
            site=site)
    if isinstance(error, _n8.NotYetElaborated):
        return NotYetElaborated(_in_surface_words(str(error)), None, site=site)
    for source, target, write in _N8_REFUSALS:
        if isinstance(error, source):
            return target(_surface_words(str(error)), write=write, site=site)
    return error


_RULES = re.compile(r"\s*\((?:[A-Z]+\d+[a-z]?(?:, )?)+\)\s*\.?\s*$")
_N8_WORDS = (("B, p and its components", "step and its options"), ("B.state", "step.state"),
             ("B.time", "step.time"), ("p.dt", "step.dt"))


def _in_surface_words(message: str) -> str:
    """An N8 message with ``step`` for B and p; its rule ids stay where the sentence has them."""
    for n8, surface in _N8_WORDS:
        message = message.replace(n8, surface)
    return message


def _surface_words(message: str) -> str:
    """An N8 message in the words of the surface: ``step`` for B and p, without its rule ids,
    which the refusal of the surface states."""
    return _RULES.sub("", _in_surface_words(message).rstrip(".")).rstrip(".")


def specialize(value: MethodValue, components: dict, trace: _Trace) -> _n8.SpecializedMethod:
    """The traced method of N8 for ``value``, its rate options bound to ``components``."""
    body = value.method.body
    names = value.rate_options()
    options = value.options

    def elaborated(B: Any, p: Any, *rates: Any) -> Any:
        arguments = dict(options)
        for name, rate in zip(names, rates, strict=True):
            arguments[name] = Rate(name, rate)
        try:
            result = body(Step(B, p), **arguments)
        except Exception as error:
            translated = translate(error, trace)
            if translated is error:
                raise
            raise translated.with_traceback(None) from None
        if not isinstance(result, Candidate):
            raise TypeError("the method %s returns pops.Candidate(state), got %r"
                            % (value.name, result))
        trace.candidate_site = result.site
        return _n8.Candidate(result.state, B.time + p.dt)

    elaborated.__name__ = elaborated.__qualname__ = value.name
    # N8 reads f(B, p, *components) and calls it by position: the components keep the names of
    # the rate options, and a leading parameter gives way to a rate option of its name.
    leading: list = []
    for parameter in list(inspect.signature(elaborated).parameters.values())[:2]:
        name = parameter.name
        while name in names or name in leading:
            name = "_" + name
        leading.append(name)
    setattr(elaborated, "__signature__", inspect.Signature(  # noqa: B010 (pyright)
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for name in (*leading, *names)]))
    return _n8.Method(elaborated)(**{name: components[name] for name in names})


__all__ = ["method", "Candidate", "SurfaceMethod", "MethodValue", "Step", "Rate", "Balance"]
