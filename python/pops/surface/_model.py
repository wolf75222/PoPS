"""The model level of the surface, 𝔓 (spec 0.3, 2.6.2 and 2.6.6; Q1-Q4, R1, R2, R7, §3.7).

``model = pops.Model("name")`` takes its frame from the supports of its quantities. A model
declares its quantities, constants, parameters and data by name (SU8), and its terms and balances
by value: ``model.flux(of=u, value=a * u)``, ``model.diffusive_flux(of=u, value=-nu * grad(u))``
and ``model.balance(ddt(u) == -div(advection + diffusion))``. Each declaration returns its handle,
an expression of the surface; the model is elaborated into a b6 ``pops.Model`` when a case that
holds it is traced (``_elaborate``).

A diffusive flux is the physical flux, with its sign, as Fick and Fourier write it; it enters its
balance under -div as every flux does. Its gradient variables are the arguments of ``grad`` that
the model evolves (SU9), or those that ``gradients=`` states, which may not omit one that the value
reads (``ContradictsDerivation``). A value that the derivation finds anti-diffusive, or a diffusive
flux under +div, is refused (``AntiDiffusive``); a value with no gradient variable is refused
(``NotDiffusive``). The elaborated diffusive flux of b6 is the opposite of the physical flux, the
F^d of the canonical form of R7 (2.6.6).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from ._expr import Arith, Call, Expr, Relation, is_number, operand, text, walk
from ._geometry import Box, MathConstant
from ._refusal import (
    AmbiguousElaboration, AntiDiffusive, ContradictsDerivation, LevelMismatch, NotDiffusive,
    NotYetElaborated, ReservedName, Site, SpellingRefusal, SupportMismatch, SymbolNotInSurface,
    TermCountedTwice, user_site,
)

# SU8: the readings of a quantity, which no component or field may take.
QUANTITY_READINGS = ("space", "support", "at", "role")

_LATER_MODEL = "a later lot of N15"
_EARLIER_MODEL = {
    "param": SpellingRefusal('model.constant("a", value=v), folded at compile time, or '
                             'model.parameter("nu", default=d), bound at bind'),
    "value": SpellingRefusal("the constant or the parameter itself, which is a value"),
    "state": SpellingRefusal('model.quantity("u", pops.Scalar(Ω))'),
    "rate": SpellingRefusal("model.balance(ddt(u) == -div(F) + S)"),
    "field": SpellingRefusal('model.data("d", pops.Scalar(Ω)) for a tabulated field'),
}


def _name(value: Any, what: str) -> str:
    if (not isinstance(value, str) or not value or value.strip() != value
            or not value.isidentifier()):
        raise TypeError("%s takes its name first, as an identifier string, got %r (SU8)"
                        % (what, value))
    return value


class Positive:
    """``pops.Positive()``: the admissible set of the positive reals (S5)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "pops.Positive()"

    def __eq__(self, other: Any) -> bool:
        return type(other) is Positive

    __hash__ = object.__hash__


class Scalar:
    """``pops.Scalar(Ω)``: the space of a scalar field on the support Ω (Q2, S5)."""

    __slots__ = ("support", "admissible", "site")

    def __init__(self, support: Any = None, *, admissible: Any = None) -> None:
        if not isinstance(support, Box):
            raise TypeError("pops.Scalar(Ω) takes the support of the field, a pops.Box, got %r"
                            % (support,))
        if admissible is not None:
            raise NotYetElaborated("pops.Scalar(Ω, admissible=...), the admissible set of a "
                                   "quantity (S5)", "a later lot of N15 with the supports of G03")
        self.support = support
        self.admissible = admissible
        self.site = user_site(2)

    def __repr__(self) -> str:
        return "pops.Scalar(%r)" % (self.support,)


def _later_space(name: str) -> Any:
    def space(*args: Any, **kwargs: Any) -> Any:
        raise NotYetElaborated("pops.%s, a space of several components" % name,
                               "a later lot of N15 with the vector and tensor fields of N11"
                               ).with_traceback(None)

    space.__name__ = space.__qualname__ = name
    return space


Vector, Tensor, Components = (_later_space(name) for name in ("Vector", "Tensor", "Components"))


class Symbol(Expr):
    """A declared entity of a model, read in its expressions by its handle."""

    __slots__ = ("model", "name", "site")

    def __init__(self, model: Model, name: str, site: Site | None) -> None:
        self.model = model
        self.name = name
        self.site = site

    def _text(self, precedence: int) -> str:
        return self.name


class Quantity(Symbol):
    """A quantity of the model, ``model.quantity("u", pops.Scalar(Ω))`` (Q1, Q2)."""

    __slots__ = ("space",)

    def __init__(self, model: Model, name: str, space: Scalar, site: Site | None) -> None:
        super().__init__(model, name, site)
        self.space = space

    @property
    def support(self) -> Box:
        return self.space.support

    def at(self, *args: Any, **kwargs: Any) -> Any:
        raise NotYetElaborated("q.at(y), a read at a bound point", _LATER_MODEL)

    def role(self, *args: Any, **kwargs: Any) -> Any:
        raise NotYetElaborated("q.role(r), a read bound to another state (T18)", _LATER_MODEL)


class ModelConstant(Symbol):
    """``model.constant("a", value=v)``: a value of the physics, folded at compile time (Q4)."""

    __slots__ = ("value",)

    def __init__(self, model: Model, name: str, value: Any, site: Site | None) -> None:
        super().__init__(model, name, site)
        self.value = value


class Parameter(Symbol):
    """``model.parameter("nu", default=d, admissible=A)``: a value bound at bind (Q4)."""

    __slots__ = ("default", "admissible")

    def __init__(self, model: Model, name: str, default: Any, admissible: Any,
                 site: Site | None) -> None:
        super().__init__(model, name, site)
        self.default = default
        self.admissible = admissible


class Datum(Symbol):
    """``model.data("u0", pops.Scalar(Ω))``: a field tabulated by the user, given at bind (Q3)."""

    __slots__ = ("space",)

    def __init__(self, model: Model, name: str, space: Scalar, site: Site | None) -> None:
        super().__init__(model, name, site)
        self.space = space


class Term(Expr):
    """A term of the model, read in a balance by its handle and shown by its value (SU8)."""

    __slots__ = ("model", "quantity", "value", "name", "site", "kind")

    def __init__(self, model: Model, kind: str, quantity: Quantity, value: Any,
                 name: str | None, site: Site | None) -> None:
        self.model = model
        self.kind = kind
        self.quantity = quantity
        self.value = value
        self.name = name
        self.site = site

    def _text(self, precedence: int) -> str:
        return self.name if self.name is not None else text(self.value, precedence)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from ._refusal import NotAPopsOperation

        raise NotAPopsOperation(
            "a term of the model is not callable here; it reaches a method only as one of its "
            "options, which converts it to a rate", write="ssprk2(rate=[F]) and rate(u, t) in "
            "the method")


class Flux(Term):
    """``model.flux(of=u, value=e)``: a transport flux (R1, §3.6)."""

    __slots__ = ()


class DiffusiveFlux(Term):
    """``model.diffusive_flux(of=u, value=-k * grad(w))``: the physical flux with its sign (R2).

    ``variables`` holds the gradient variables, ``coefficient`` the k of ``F = -k grad(w)``, and
    ``origin`` says whether the variables were derived or stated.
    """

    __slots__ = ("variables", "coefficient", "origin")


class Balance:
    """``model.balance(ddt(u) == r)``: the balance of one quantity (§3.7).

    ``occurrences`` is the signed sum of r: pairs (term, coefficient) in the order written.
    """

    __slots__ = ("model", "quantity", "relation", "occurrences", "name", "site")

    def __init__(self, model: Model, quantity: Quantity, relation: Relation,
                 occurrences: tuple, name: str | None, site: Site | None) -> None:
        self.model = model
        self.quantity = quantity
        self.relation = relation
        self.occurrences = occurrences
        self.name = name
        self.site = site

    @property
    def terms(self) -> tuple:
        return tuple(term for term, _ in self.occurrences)

    def __repr__(self) -> str:
        return text(self.relation)

    def __bool__(self) -> bool:
        return True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from ._refusal import NotAPopsOperation

        raise NotAPopsOperation(
            "a balance of the model is not callable here; a model reaches a method only as one "
            "of its options, which converts it to a rate", write="ssprk2(rate=model) and "
            "rate(u, t) in the method")


def _scheme_or_method(value: Any) -> str | None:
    from ._method import MethodValue, SurfaceMethod
    from ._numerics import SchemeValue

    if isinstance(value, SchemeValue):
        return "a scheme, a choice of the numerics"
    if isinstance(value, (MethodValue, SurfaceMethod)):
        return "a time method, a choice of the program"
    return None


def _level_check(value: Any, where: str) -> None:
    owner = _scheme_or_method(value)
    if owner is not None:
        raise LevelMismatch("%s is %s, and the model level states relations only" % (where, owner),
                            write="the value at the model level, and the choice in pops.Numerics "
                                  "or pops.Program")


class Model:
    """``pops.Model("name")``: a model, its frame that of the supports of its quantities."""

    def __init__(self, name: Any = None, **earlier: Any) -> None:
        if "frame" in earlier:
            raise SpellingRefusal('pops.Model("name"): the frame of a model is that of the '
                                  'supports of its quantities').refuse(
                "pops.Model(name, frame=...)", "pops.Model")
        if earlier:
            raise TypeError("pops.Model takes its name only, got %s" % ", ".join(sorted(earlier)))
        self.name = _name(name, "pops.Model")
        self.site = user_site(2)
        self.quantities: list = []
        self.constants: list = []
        self.parameters: list = []
        self.data_fields: list = []
        self.terms: list = []
        self.balances: list = []
        self._declared: dict = {}

    def __repr__(self) -> str:
        return "pops.Model(%r)" % self.name

    def __bool__(self) -> bool:
        return True

    __hash__ = object.__hash__

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _EARLIER_MODEL:
            raise _EARLIER_MODEL[name].refuse("model.%s" % name, "a model")
        raise AttributeError("a model has no reading %s" % name)

    def _declare(self, name: str, what: str) -> str:
        name = _name(name, what)
        if name in self._declared:
            raise ValueError("%s: the name %r is already declared in the model %r"
                             % (what, name, self.name))
        self._declared[name] = what
        return name

    def _own(self, value: Any, what: str) -> None:
        for node in walk(value):
            if isinstance(node, (Symbol, Term)) and node.model is not self:
                raise ValueError("%s reads %s, which another model declares" % (what, node))

    # --- declarations -------------------------------------------------------------------------

    def quantity(self, name: Any, space: Any = None) -> Quantity:
        """``model.quantity("u", pops.Scalar(Ω))``: a quantity and the space it lives in (Q1)."""
        name = self._declare(name, "model.quantity")
        if not isinstance(space, Scalar):
            raise TypeError("model.quantity(%r, space) takes the space of the quantity, "
                            "pops.Scalar(Ω), got %r" % (name, space))
        handle = Quantity(self, name, space, user_site(2))
        self.quantities.append(handle)
        return handle

    def constant(self, name: Any, *, value: Any) -> ModelConstant:
        """``model.constant("g", value=v)``: a value of the physics, folded at compile time."""
        name = self._declare(name, "model.constant")
        if not is_number(value):
            raise TypeError("model.constant(%r, value=v) takes a Python number, got %r"
                            % (name, value))
        handle = ModelConstant(self, name, value, user_site(2))
        self.constants.append(handle)
        return handle

    def parameter(self, name: Any, *, default: Any, admissible: Any = None) -> Parameter:
        """``model.parameter("nu", default=d, admissible=pops.Positive())``, bound at bind."""
        name = self._declare(name, "model.parameter")
        if not is_number(default):
            raise TypeError("model.parameter(%r, default=d) takes a Python number, got %r"
                            % (name, default))
        if admissible is not None and not isinstance(admissible, Positive):
            raise NotYetElaborated("model.parameter(admissible=%r)" % (admissible,),
                                   "a later lot of N15 with the admissible sets of G03")
        if isinstance(admissible, Positive) and not float(default) > 0:
            raise ValueError("model.parameter(%r): the default %r is outside its admissible set "
                             "pops.Positive()" % (name, default))
        handle = Parameter(self, name, default, admissible, user_site(2))
        self.parameters.append(handle)
        return handle

    def data(self, name: Any, space: Any = None) -> Datum:
        """``model.data("d", pops.Scalar(Ω))``: a field tabulated by the user (Q3)."""
        name = self._declare(name, "model.data")
        if not isinstance(space, Scalar):
            raise TypeError("model.data(%r, space) takes the space of the datum, pops.Scalar(Ω), "
                            "got %r" % (name, space))
        handle = Datum(self, name, space, user_site(2))
        self.data_fields.append(handle)
        return handle

    def _term_quantity(self, of: Any, what: str) -> Quantity:
        if not isinstance(of, Quantity):
            raise TypeError("%s(of=u, ...) takes the quantity that the term changes, got %r"
                            % (what, of))
        if of.model is not self:
            raise ValueError("%s(of=%s): the quantity belongs to another model" % (what, of))
        return of

    def _term_name(self, name: Any, what: str) -> str | None:
        return None if name is None else self._declare(name, what)

    def flux(self, *, of: Any, value: Any, name: Any = None) -> Flux:
        """``model.flux(of=u, value=e)``: on a support of rank one, a scalar e is its component."""
        quantity = self._term_quantity(of, "model.flux")
        _level_check(value, "the value of model.flux")
        value = operand(value, "model.flux(value=...)")
        self._own(value, "model.flux")
        site = user_site(2)
        rank = quantity.support.rank
        if rank != 1:
            raise SupportMismatch(
                "the flux of %s on a support of rank %d is a vector of %d components, and %s is "
                "a scalar" % (quantity, rank, rank, text(value)),
                write="model.flux(of=%s, value=vector(%s))"
                      % (quantity, ", ".join(["..."] * rank)), site=site)
        handle = Flux(self, "Flux", quantity, value, self._term_name(name, "model.flux"), site)
        self.terms.append(handle)
        return handle

    def diffusive_flux(self, *, of: Any, value: Any, gradients: Any = None,
                       name: Any = None) -> DiffusiveFlux:
        """``model.diffusive_flux(of=u, value=-nu * grad(u))``: the physical flux with its sign."""
        quantity = self._term_quantity(of, "model.diffusive_flux")
        _level_check(value, "the value of model.diffusive_flux")
        value = operand(value, "model.diffusive_flux(value=...)")
        self._own(value, "model.diffusive_flux")
        site = user_site(2)
        coefficient, variables = _diffusive_form(value, site)
        if gradients is not None:
            if isinstance(gradients, tuple):
                raise TypeError("model.diffusive_flux(gradients=[...]) takes a list; a tuple is "
                                "a point, a size or an interval (SU14)")
            if not isinstance(gradients, list) or not all(isinstance(item, Quantity)
                                                          for item in gradients):
                raise TypeError("model.diffusive_flux(gradients=[...]) takes a list of quantities")
            omitted = [item for item in variables if not any(item is g for g in gradients)]
            if omitted:
                raise ContradictsDerivation(
                    "gradients=%s omits %s, whose gradient the value %s reads"
                    % ("[%s]" % ", ".join(map(str, gradients)), ", ".join(map(str, omitted)),
                       text(value)), write="gradients=[%s], or leave it out to derive it"
                    % ", ".join(map(str, variables)), site=site)
            extra = [item for item in gradients if not any(item is v for v in variables)]
            if extra:
                raise ContradictsDerivation(
                    "gradients=[...] states %s, whose gradient the value %s does not read"
                    % (", ".join(map(str, extra)), text(value)),
                    write="gradients=[%s], or leave it out to derive it"
                    % ", ".join(map(str, variables)), site=site)
        if len(variables) != 1 or variables[0] is not quantity:
            raise NotYetElaborated(
                "a diffusive flux whose gradient variable is not its own quantity (%s)"
                % text(value), "a later lot of N15 with the coupled and drift realizations of G05",
                site=site)
        sign = _sign(coefficient)
        if sign is not None and sign < 0:
            raise AntiDiffusive(
                "the physical flux %s is anti-diffusive: written F = -k grad(%s), its k = %s is "
                "negative" % (text(value), variables[0], text(coefficient)),
                write="the physical flux with its sign, as Fick writes it: "
                      "model.diffusive_flux(of=%s, value=-k * grad(%s)) with k positive"
                      % (quantity, variables[0]), site=site)
        handle = DiffusiveFlux(self, "DiffusiveFlux", quantity, value,
                               self._term_name(name, "model.diffusive_flux"), site)
        handle.variables = tuple(variables)
        handle.coefficient = coefficient
        handle.origin = "derived" if gradients is None else "stated"
        self.terms.append(handle)
        return handle

    def balance(self, relation: Any, *, name: Any = None) -> Balance:
        """``model.balance(ddt(u) == r)``: the balance of the quantity under ``ddt``."""
        site = user_site(2)
        if not isinstance(relation, Relation):
            raise TypeError("model.balance takes a relation ddt(u) == r, got %r" % (relation,))
        sides = relation.sides()
        derivative = [side for side in sides
                      if isinstance(side, Call) and side.word == "ddt"]
        if len(derivative) != 1:
            raise ValueError("model.balance(%s): one side of the relation is ddt(q), the time "
                             "derivative of the quantity that the balance evolves (SU4)"
                             % relation)
        quantity = derivative[0].args[0]
        if quantity.model is not self:
            raise ValueError("model.balance(%s): the quantity belongs to another model" % relation)
        rhs = sides[1] if sides[0] is derivative[0] else sides[0]
        _level_check(rhs, "a term of model.balance")
        occurrences = _occurrences(rhs, site)
        for term, coefficient in occurrences:
            if term.quantity is not quantity:
                raise ValueError("model.balance(%s): %s is a term of %s, not of %s"
                                 % (relation, term, term.quantity, quantity))
            if isinstance(term, DiffusiveFlux) and coefficient > 0:
                raise AntiDiffusive(
                    "the diffusive flux %s enters its balance under +div; a physical flux, "
                    "diffusive or not, enters as -div(F)" % term,
                    write="ddt(%s) == -div(... + %s)" % (quantity, term), site=site)
        seen: list = []
        for term, _ in occurrences:
            if any(term is other for other in seen):
                raise TermCountedTwice("the term %s appears twice in the balance %s"
                                       % (term, relation), write="each term once", site=site)
            seen.append(term)
        if any(other.quantity is quantity for other in self.balances):
            raise AmbiguousElaboration(
                "%s already has a balance; a second balance of one quantity is an alternative "
                "equation, which the program would have to choose" % quantity,
                write="one balance per quantity", site=site)
        handle = Balance(self, quantity, relation, occurrences,
                         None if name is None else self._declare(name, "model.balance"), site)
        self.balances.append(handle)
        return handle

    # --- constructs of later lots -------------------------------------------------------------

    def _later(self, construct: str) -> Any:
        def later(*args: Any, **kwargs: Any) -> Any:
            raise NotYetElaborated(construct, _LATER_MODEL).with_traceback(None)

        return later

    @property
    def define(self) -> Any:
        return self._later("model.define, a definition (R16)")

    @property
    def source(self) -> Any:
        return self._later("model.source, a source (R4)")

    @property
    def interaction(self) -> Any:
        return self._later("model.interaction (R21)")

    @property
    def inventory(self) -> Any:
        return self._later("model.inventory (R22)")

    @property
    def constraint(self) -> Any:
        return self._later("model.constraint (R10-R14)")

    @property
    def relation(self) -> Any:
        return self._later("model.relation, an implicit relation (R16)")

    @property
    def on(self) -> Any:
        return self._later("model.on, a boundary relation (R10, R18, R20)")

    @property
    def wave_speeds(self) -> Any:
        return self._later("model.wave_speeds, declared wave speeds (R9)")

    @property
    def time(self) -> Any:
        raise NotYetElaborated("model.time, the time in a model value", _LATER_MODEL)


# --- the forms that a model recognizes ------------------------------------------------------------

def _negate(value: Any) -> Any:
    if is_number(value):
        return -value
    if isinstance(value, Arith) and value.op == "neg":
        return value.args[0]
    return Arith("neg", (value,))


def _reads(value: Any, word: str) -> bool:
    return any(isinstance(node, Call) and node.word == word for node in walk(value))


def _product(items: list) -> Any:
    if not items:
        return 1
    result = items[0]
    for item in items[1:]:
        result = Arith("*", (result, item))
    return result


def _diffusive_form(value: Any, site: Site | None) -> tuple:
    """(k, [w]) for a physical flux F = -k * grad(w): the k of the gradient and its variable.

    This lot reads the form of one gradient times factors free of gradients, with unary minus
    and divisions: ``-nu * grad(u)``, ``-(nu * grad(u))``, ``grad(u) * c / d``. The general linear
    forms come with N11.
    """
    factors: list = []
    divisors: list = []
    negative = False
    node = value
    while True:
        if isinstance(node, Arith) and node.op == "neg":
            negative = not negative
            node = node.args[0]
        elif isinstance(node, Arith) and node.op == "*" and not _reads(node.args[0], "grad"):
            factors.append(node.args[0])
            node = node.args[1]
        elif isinstance(node, Arith) and node.op == "*" and not _reads(node.args[1], "grad"):
            factors.append(node.args[1])
            node = node.args[0]
        elif isinstance(node, Arith) and node.op == "/" and not _reads(node.args[1], "grad"):
            divisors.append(node.args[1])
            node = node.args[0]
        else:
            break
    if isinstance(node, Call) and node.word == "grad":
        argument = node.args[0]
        if not isinstance(argument, Quantity):
            raise NotYetElaborated("grad of %s in a diffusive flux" % text(argument),
                                   "a later lot of N15 with the definitions of R16", site=site)
        magnitude = _product(factors)
        if divisors:
            magnitude = Arith("/", (magnitude, _product(divisors)))
        # F = c grad(w) with c = +-magnitude, so k = -c.
        return (magnitude if negative else _negate(magnitude)), [argument]
    if not _reads(value, "grad"):
        raise NotDiffusive("the diffusive flux %s reads no gradient, so it has no gradient "
                           "variable" % text(value),
                           write="model.diffusive_flux(of=u, value=-k * grad(u)), or "
                                 "model.flux(of=u, value=e) for a transport flux", site=site)
    raise NotYetElaborated("the diffusive flux %s, which is not one coefficient times one "
                           "gradient" % text(value), "N11 (expression closure, C1)", site=site)


def _sign(value: Any) -> int | None:
    """+1, -1 or None: the sign that the declarations of the model derive for a coefficient."""
    if is_number(value):
        return (value > 0) - (value < 0)
    if isinstance(value, MathConstant):
        return 1
    if isinstance(value, ModelConstant):
        return (value.value > 0) - (value.value < 0)
    if isinstance(value, Parameter):
        return 1 if isinstance(value.admissible, Positive) else None
    if isinstance(value, Arith):
        if value.op == "neg":
            inner = _sign(value.args[0])
            return None if inner is None else -inner
        if value.op in ("*", "/"):
            left, right = (_sign(arg) for arg in value.args)
            return None if left is None or right is None else left * right
    return None


def _occurrences(rhs: Any, site: Site | None) -> tuple:
    """The signed sum of div(term) of a balance: [(term, coefficient)], in the order written."""
    rows: list = []

    def add(node: Any, coefficient: Fraction) -> None:
        if isinstance(node, Arith) and node.op == "+":
            add(node.args[0], coefficient)
            add(node.args[1], coefficient)
        elif isinstance(node, Arith) and node.op == "-":
            add(node.args[0], coefficient)
            add(node.args[1], -coefficient)
        elif isinstance(node, Arith) and node.op == "neg":
            add(node.args[0], -coefficient)
        elif isinstance(node, Arith) and node.op == "*" and is_number(node.args[0]):
            add(node.args[1], coefficient * Fraction(node.args[0]))
        elif isinstance(node, Arith) and node.op == "*" and is_number(node.args[1]):
            add(node.args[0], coefficient * Fraction(node.args[1]))
        elif isinstance(node, Call) and node.word == "div":
            flux(node.args[0], coefficient)
        else:
            raise NotYetElaborated(
                "the part %s of a balance, which is not the divergence of a flux" % text(node),
                "a later lot of N15 with the sources of R4", site=site)

    def flux(node: Any, coefficient: Fraction) -> None:
        if isinstance(node, Arith) and node.op in ("+", "-"):
            flux(node.args[0], coefficient)
            flux(node.args[1], coefficient if node.op == "+" else -coefficient)
        elif isinstance(node, Arith) and node.op == "neg":
            flux(node.args[0], -coefficient)
        elif isinstance(node, (Flux, DiffusiveFlux)):
            rows.append((node, coefficient))
        else:
            raise TypeError("div(...) in a balance takes the fluxes of the model and their sums, "
                            "got %s" % text(node))

    add(rhs, Fraction(1))
    if not rows:
        raise ValueError("a balance holds the divergence of at least one flux in this lot")
    return tuple(rows)


def model_names(model: Model) -> dict:
    """The elaborated name of each term: its name, or ``kind_quantity`` with an ordinal among the
    unnamed terms of one kind and one quantity (presentation-free: no source line, no label)."""
    names: dict = {}
    counts: dict = {}
    for term in model.terms:
        if term.name is not None:
            names[term] = term.name
            continue
        stem = "%s_%s" % ("flux" if isinstance(term, Flux) else "diffusive_flux",
                          term.quantity.name)
        counts[stem] = counts.get(stem, 0) + 1
        names[term] = stem if counts[stem] == 1 else "%s_%d" % (stem, counts[stem])
    taken = set(model._declared)
    for term, name in names.items():
        if term.name is None and name in taken:
            raise AmbiguousElaboration(
                "the unnamed term %s would take the name %r, which the model declares for "
                "another entity" % (term, name), write="name=... for this term", site=term.site)
    return names


def reserved_component(name: str) -> None:
    """SU8: a component or a field may not take the name of a reading of its quantity."""
    if name in QUANTITY_READINGS or name.startswith("_"):
        raise ReservedName("the field %r takes the name of a reading of a quantity; the reserved "
                           "names are %s" % (name, ", ".join(QUANTITY_READINGS)),
                           write="another field name")


__all__ = ["Model", "Scalar", "Vector", "Tensor", "Components", "Positive", "Quantity",
           "ModelConstant", "Parameter", "Datum", "Flux", "DiffusiveFlux", "Balance",
           "model_names", "SymbolNotInSurface"]
