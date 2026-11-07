"""The numerics level of the surface, 𝔇 (spec 0.3, 2.6.2, 2.6.6; D1, D5).

``pops.Numerics(represent={u: pops.CellAverage()}, schemes={F: s})`` states the representation of
each quantity and the scheme of each term, keyed by handles (SU8, SU14). The scheme of a key is a
word called with its options (SU5): ``upwind()``, never ``upwind``. Each term of each balance is
covered exactly once (``CoverageNotExact``), and the representation of an evolved quantity is
stated (SU9). ``numerics.of(transport)`` is the realization of a balance or of a term, which a
claim reads.

The scheme words of ``pops.lib`` are written with ``@pops.scheme`` in spec 0.3 (A4). Until that
construct is elaborated, a word of the library returns a ``SchemeValue`` whose ``realize`` gives
the realization of b6 that its formula has at b6, for the keys whose form it checks (``TermView``).
That bridge is the one place where ``pops.lib`` reaches an elaborated form, and it goes away when
the words are written as schemes (CF4 then compares the two).
"""
from __future__ import annotations

from typing import Any

from ._expr import Call, walk
from ._model import Balance, DiffusiveFlux, Flux, Quantity, Term
from ._refusal import (
    CoverageNotExact, LevelMismatch, NotYetElaborated, Site, SpellingRefusal, user_site,
)


class CellAverage:
    """``pops.CellAverage()``: the cell averages of a field, the representation of b6's states."""

    __slots__ = ()

    def __init__(self, *, quadrature: Any = None) -> None:
        if quadrature is not None:
            raise NotYetElaborated("pops.CellAverage(quadrature=...) (D3)",
                                   "a later lot of N15 with the quadratures of G03")

    def __repr__(self) -> str:
        return "pops.CellAverage()"


class TermView:
    """The key that a scheme word realizes, as the bridge of this lot sees it.

    ``kind`` is ``"Flux"`` or ``"DiffusiveFlux"`` (2.6.4); ``value`` the value of the key as the
    model states it; ``elaborated`` its handle in the elaborated b6 model and ``state`` the b6
    state handle of its quantity; ``coefficient`` the k of a diffusive flux F = -k grad(u).
    ``slope`` is dF/du, the wave speed that R9 derives and the elaborated flux carries, or
    ``None`` when this lot derives none; ``unit_flux`` is F(1) when F = F(1) u, else ``None``.
    """

    __slots__ = ("term", "kind", "quantity", "value", "elaborated", "state", "coefficient",
                 "slope", "unit_flux", "site")

    def __init__(self, term: Term, elaborated: Any, state: Any, slope: Any = None,
                 unit_flux: Any = None) -> None:
        self.term = term
        self.kind = term.kind
        self.quantity = term.quantity
        self.value = term.value
        self.elaborated = elaborated
        self.state = state
        self.coefficient = getattr(term, "coefficient", None)
        self.slope = slope
        self.unit_flux = unit_flux
        self.site = term.site

    def reads_quantity(self, value: Any) -> bool:
        """Whether a value reads a quantity of the model, so varies with the state."""
        return any(isinstance(node, Quantity) for node in walk(value))

    def reads_derivative(self, value: Any) -> bool:
        """Whether a value reads the derivative of a field, ``grad(...)``."""
        return any(isinstance(node, Call) and node.word == "grad" for node in walk(value))


class SchemeValue:
    """A scheme word called with its options: the value of a key of ``schemes=``.

    ``word`` names the word for messages, ``options`` holds what the call stated, and ``kinds``
    the kinds of key that the word covers. ``realize(term)`` returns the realization of b6 of the
    key; the library writes it (see the module docstring).
    """

    kinds: tuple = ()

    def __init__(self, word: str, **options: Any) -> None:
        self.word = word
        self.options = dict(options)
        self.site = user_site(2)

    def realize(self, term: TermView) -> Any:  # pragma: no cover - the library defines it
        raise NotImplementedError

    def describe(self) -> str:
        """The elaborated builder, for ``pops.inspect``."""
        return self.word

    def __repr__(self) -> str:
        options = ", ".join("%s=%r" % item for item in self.options.items()
                            if item[1] is not None)
        return "%s(%s)" % (self.word, options)


class Realization:
    """``numerics.of(x)``: the realization of a balance or of a term under a numerics."""

    __slots__ = ("numerics", "subject")

    def __init__(self, numerics: Numerics, subject: Any) -> None:
        self.numerics = numerics
        self.subject = subject

    def __repr__(self) -> str:
        return "numerics.of(%r)" % (self.subject,)


_EARLIER_NUMERICS = {
    "realizations": SpellingRefusal("pops.Numerics(schemes={F: s}), the key being the term that "
                                    "the scheme covers"),
}


class Numerics:
    """``pops.Numerics(represent={u: pops.CellAverage()}, schemes={F: upwind()})``."""

    def __init__(self, *, represent: Any = None, schemes: Any = None, boundary: Any = None,
                 **earlier: Any) -> None:
        for name in earlier:
            if name in _EARLIER_NUMERICS:
                raise _EARLIER_NUMERICS[name].refuse("pops.Numerics(%s=...)" % name,
                                                     "pops.Numerics")
            raise TypeError("pops.Numerics takes represent=, schemes= and boundary=, got %s="
                            % name)
        if boundary is not None:
            raise NotYetElaborated("pops.Numerics(boundary=...), the boundary policy",
                                   "a later lot of N15 with the boundary relations of G04")
        self.site = user_site(2)
        self.represent = self._table(represent, "represent", Quantity)
        for _, representation in self.represent:
            if not isinstance(representation, CellAverage):
                raise NotYetElaborated("the representation %r" % (representation,),
                                       "a later lot of N15 with the representations of G03")
        self.schemes = self._table(schemes, "schemes", (Flux, DiffusiveFlux))
        for key, scheme in self.schemes:
            self._check_scheme(key, scheme)

    def _table(self, table: Any, name: str, keys: Any) -> tuple:
        if table is None:
            return ()
        if not isinstance(table, dict):
            raise TypeError("pops.Numerics(%s=...) takes a dictionary keyed by handles, "
                            "%s={key: value}" % (name, name))
        rows = []
        for key, value in table.items():
            if isinstance(key, str):
                raise TypeError("pops.Numerics(%s=...) is keyed by handles, not by the string %r "
                                "(SU8)" % (name, key))
            if isinstance(key, tuple):
                raise NotYetElaborated("a key that several terms share, pops.jointly(F, G)",
                                       "a later lot of N15 with the joint realizations of G05")
            if not isinstance(key, keys):
                raise NotYetElaborated("%s={%r: ...}, a key of another kind" % (name, key),
                                       "a later lot of N15")
            rows.append((key, value))
        return tuple(rows)

    def _check_scheme(self, key: Any, scheme: Any) -> None:
        from ._method import MethodValue, SurfaceMethod

        if isinstance(scheme, (MethodValue, SurfaceMethod)):
            raise LevelMismatch("schemes={%s: ...} holds a time method, a choice of the program"
                                % key, write="the method in pops.Advance(...), and a scheme "
                                             "of the numerics under the key")
        if not isinstance(scheme, SchemeValue):
            if callable(scheme):
                name = getattr(scheme, "__name__", "the word")
                raise TypeError("schemes={%s: %s}: a scheme word is always called, with or "
                                "without options (SU5); write %s()" % (key, name, name))
            raise TypeError("schemes={%s: ...} takes a scheme word called with its options, got %r"
                            % (key, scheme))
        if key.kind not in scheme.kinds:
            from ._refusal import RealizationMismatch

            raise RealizationMismatch(
                "%s covers the keys of kind %s, and %s is a %s" % (
                    scheme, " or ".join(scheme.kinds), key, key.kind),
                write="a scheme of the kind of the key", site=scheme.site)

    def scheme_of(self, term: Term) -> Any:
        for key, scheme in self.schemes:
            if key is term:
                return scheme
        return None

    def representation_of(self, quantity: Quantity) -> Any:
        for key, representation in self.represent:
            if key is quantity:
                return representation
        return None

    def of(self, subject: Any) -> Realization:
        """``numerics.of(x)``: the realization of a balance or of a term."""
        if not isinstance(subject, (Balance, Term)):
            raise TypeError("numerics.of(x) takes a balance or a term of the model, got %r"
                            % (subject,))
        return Realization(self, subject)

    def check_coverage(self, balances: list, terms: list) -> None:
        """D5: each term of each balance is covered by one scheme, and each key is such a term."""
        integrated = [term for balance in balances for term in balance.terms]
        for term in integrated:
            if self.scheme_of(term) is None:
                raise CoverageNotExact(
                    "the term %s of the balance of %s, declared at %s, has no scheme"
                    % (term, term.quantity, term.site),
                    write="pops.Numerics(schemes={%s: <a scheme word>()})" % term,
                    site=self.site)
        for key, _ in self.schemes:
            if not any(key is term for term in integrated):
                raise CoverageNotExact(
                    "the key %s is a term that no balance integrates" % key,
                    write="the term in the balance of its quantity, or no scheme for it",
                    site=self.site)
        for balance in balances:
            if self.representation_of(balance.quantity) is None:
                from ._refusal import AmbiguousElaboration

                raise AmbiguousElaboration(
                    "the quantity %s has no representation" % balance.quantity,
                    write="pops.Numerics(represent={%s: pops.CellAverage()})" % balance.quantity,
                    site=self.site)

    def __repr__(self) -> str:
        return "pops.Numerics(...)"


__all__ = ["CellAverage", "Numerics", "Realization", "SchemeValue", "TermView", "Site"]
