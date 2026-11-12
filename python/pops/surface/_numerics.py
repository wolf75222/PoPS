"""The numerics level of the surface, 𝔇 (spec 0.3, 2.6.2, 2.6.6; D1, D5).

``pops.Numerics(represent={u: pops.CellAverage()}, schemes={F: s}, boundary=ghosts())`` states the
representation of each quantity, the scheme of each term, keyed by handles (SU8, SU14), and the
policy that realizes the relations of ``model.on`` (lot B). The scheme of a key is a
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
                                   "G03 lot R-b, under CellAverage(quadrature=...) of "
                                   "pops.lib.representations (N16 lot 1)")

    def definition(self, rank: int) -> Any:
        """The definition of contract @2 that this spelling elaborates, the one of
        ``CellAverage()`` of ``pops.lib.representations`` (N16 5.5, SU-T3)."""
        from pops.representations.definition import cell_average

        return cell_average(rank)

    def __repr__(self) -> str:
        return "pops.CellAverage()"


class TermView:
    """The key that a scheme word realizes, as the bridge of this lot sees it.

    ``kind`` is ``"Flux"`` or ``"DiffusiveFlux"`` (2.6.4); ``value`` the value of the key as the
    model states it; ``elaborated`` its handle in the elaborated b6 model and ``state`` the b6
    state handle of its quantity; ``coefficient`` the k of a diffusive flux F = -k grad(u).
    ``slope`` is dF/du, the wave speed that R9 derives and the elaborated flux carries, or
    ``None`` when this lot derives none; ``unit_flux`` is F(1) when F = F(1) u, else ``None``.
    ``lower(value, inputs)`` is the b6 expression of a value of the model whose reads of the
    entities of a scheme are given by ``inputs`` (``@pops.scheme``, lot 2).
    """

    __slots__ = ("term", "kind", "quantity", "value", "elaborated", "state", "coefficient",
                 "slope", "unit_flux", "site", "lower")

    def __init__(self, term: Term, elaborated: Any, state: Any, slope: Any = None,
                 unit_flux: Any = None, lower: Any = None) -> None:
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
        self.lower = lower

    def reads_quantity(self, value: Any) -> bool:
        """Whether a value reads a quantity of the model, so varies with the state."""
        return any(isinstance(node, Quantity) for node in walk(value))

    def face_function(self, function: Any, where: str, reconstructed: bool = False) -> Any:
        """A plain function of ``face`` given as an option, ``upwind(speed=f)``, run on the face
        of the key: the function of the b6 face of ``pops.lib.numerics.fluxes`` that lowers it."""
        from ._scheme import face_function

        return face_function(function, self, where, reconstructed)

    def parameters(self, value: Any) -> list:
        """The names of the parameters that a value reads, bound at bind (Q4)."""
        from ._model import Parameter

        return [node.name for node in walk(value) if isinstance(node, Parameter)]

    def number(self, value: Any) -> Any:
        """The number of a value written with numbers, mathematical constants and model
        constants (folded at compile time, Q4), or ``None`` when it reads anything else."""
        from fractions import Fraction

        from ._expr import Arith
        from ._geometry import MathConstant
        from ._model import ModelConstant

        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float, Fraction)):
            return value
        if isinstance(value, (MathConstant, ModelConstant)):
            return value.value
        if isinstance(value, Arith):
            args = [self.number(arg) for arg in value.args]
            if any(arg is None for arg in args):
                return None
            if value.op == "neg":
                return -args[0]
            left, right = args
            return {"+": lambda: left + right, "-": lambda: left - right,
                    "*": lambda: left * right, "/": lambda: left / right,
                    "**": lambda: left ** right}[value.op]()
        return None

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


class _Realization:
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
        self.site = user_site(2)
        self.boundary = self._boundary(boundary)
        self.represent = self._table(represent, "represent", Quantity)
        for quantity, representation in self.represent:
            self._check_representation(quantity, representation)
        self.schemes = self._table(schemes, "schemes", (Flux, DiffusiveFlux))
        for key, scheme in self.schemes:
            self._check_scheme(key, scheme)

    def _check_representation(self, quantity: Quantity, representation: Any) -> None:
        """A word of ``pops.lib.representations`` or a ``@pops.representation`` of the case runs
        here, at the rank of the key (SU5, N16 3.1). This lot elaborates the definition of
        ``CellAverage()`` on the route of ``pops.CellAverage()`` and refuses every other with the lot
        that elaborates it (N16 5.5, the transitional table)."""
        if isinstance(representation, CellAverage):
            return
        from ._representation import (
            Coefficients, Dual, RepresentationValue, SurfaceRepresentation, check,
        )

        if isinstance(representation, (RepresentationValue, SurfaceRepresentation)):
            check(representation, quantity, self.site)
            return
        if isinstance(representation, (Coefficients, Dual)):
            raise TypeError("represent={%s: %r}: pops.Coefficients and pops.Dual are what a "
                            "@pops.representation returns; write the representation called with "
                            "its options" % (quantity, representation))
        raise TypeError("represent={%s: ...} takes a representation called with its options, "
                        "such as CellAverage() of pops.lib.representations or a "
                        "@pops.representation of the case, got %r" % (quantity, representation))

    def _boundary(self, policy: Any) -> Any:
        """``boundary=ghosts()``: a boundary policy, a scheme over ``face`` that returns
        ``Ghost`` for each relation of ``model.on`` (2.6.2, D9). It runs when the case is built,
        where the relations of the model are known (lot B)."""
        if policy is None:
            return None
        from ._scheme import SurfaceScheme, UserScheme

        if isinstance(policy, SurfaceScheme):
            from ._refusal import NotAPopsOperation

            raise NotAPopsOperation(
                "boundary=%s holds the policy uncalled; a call states its options and runs "
                "nothing (SU5)" % policy.name, write="boundary=%s()" % policy.name,
                site=self.site)
        if isinstance(policy, dict):
            raise NotYetElaborated("pops.Numerics(boundary={piece: ...}), a policy per piece; "
                                   "a scheme under the handle of model.on replaces the policy "
                                   "on its pieces", "a later lot of N15 with the boundary "
                                                    "relations of G04")
        if not isinstance(policy, UserScheme) or policy.scheme.is_cell:
            raise TypeError("pops.Numerics(boundary=...) takes a boundary policy called with its "
                            "options, such as ghosts(), got %r" % (policy,))
        if policy.scheme.is_boundary or "face" in policy.scheme.entities:
            return policy
        raise TypeError("pops.Numerics(boundary=...) takes a boundary policy, a scheme over face "
                        "that returns Ghost, got %r" % (policy,))

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
        from ._scheme import SurfaceScheme, UserScheme

        if isinstance(scheme, SurfaceScheme):
            from ._refusal import NotAPopsOperation

            raise NotAPopsOperation(
                "schemes={%s: %s} holds the scheme uncalled; a call states its options and runs "
                "nothing (SU5)" % (key, scheme.name), write="%s()" % scheme.name,
                site=self.site)
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
        if isinstance(scheme, UserScheme):
            # SU5: PoPS runs the scheme when pops.Numerics receives it; a refusal names the line of
            # the scheme and the line of this call.
            from ._refusal import SurfaceRefusal

            try:
                scheme.check_key(key)
            except SurfaceRefusal as refusal:
                raise refusal.traced_at(self.site).with_traceback(None) from None

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

    def of(self, subject: Any) -> _Realization:
        """``numerics.of(x)``: the realization of a balance or of a term."""
        if not isinstance(subject, (Balance, Term)):
            raise TypeError("numerics.of(x) takes a balance or a term of the model, got %r"
                            % (subject,))
        return _Realization(self, subject)

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


__all__ = ["CellAverage", "Numerics", "SchemeValue", "TermView", "Site"]
