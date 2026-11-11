"""N11 lot 1 (spec 0.3 R7, gap C1): the right-hand side of a balance is a free algebra.

A summand without terms joins the balance as an anonymous source; a typed term keeps its exact
coefficient; a term under any other node is refused with ``TermNotAdditive`` at the line that wrote
it. Coverage names the uncovered terms with the line of each.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction

import pytest

import pops
import pops.math as pm
from pops._ir.balance import BalanceOccurrence
from pops._ir.expr import Const, Mul, Neg
from pops._ir.refusals import CoverageNotExact, TermNotAdditive
from pops._ir.vector_expr import VectorExpr
from pops.math import RateExpr, ddt, div
from pops.numerics import DiscretizationPlan
from pops.model import Rate, RateBundle, Signature
from pops.physics import Density, Model
from pops.provenance import SourceSpan
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
from tests.python.unit.physics.test_board_authoring_atomicity import _snapshot


def _scalar():
    model = Model("scalar_reaction", frame=FRAME)
    state = model.state("U", components=["u"], roles={"u": Density()})
    (u,) = state
    flux = model.flux("F", frame=FRAME, state=state, components={X_AXIS: [u], Y_AXIS: [u]},
                      waves={X_AXIS: [1], Y_AXIS: [1]})
    return model, state, u, flux


def _pair():
    model = Model("pair_reaction", frame=FRAME)
    state = model.state("U", components=["a", "b"])
    a, b = state
    flux = model.flux("F", frame=FRAME, state=state, components={X_AXIS: [a, b], Y_AXIS: [a, b]},
                      waves={X_AXIS: [1, 1], Y_AXIS: [1, 1]})
    return model, state, (a, b), flux


def _summary(rate):
    return [(item.kind, getattr(item.payload, "name", None), item.coefficient)
            for item in rate.occurrences]


def _value(model, occurrence):
    return model._dsl._m._source_terms[occurrence.payload.reg_name]


def _registry(model):
    module = model._multi_module
    registry = (module.operator_registry() if module is not None
                else model._dsl._m.operator_registry())
    return tuple(registry.names()), tuple(sorted(registry.aliases()))


def _observed(model):
    """Everything a refused balance must leave as it was (R4)."""
    return (_snapshot(model), tuple(sorted(model._sources)), _registry(model),
            repr(model.inspect()), model.owner_path.definition_fingerprint,
            tuple(item.local_id for item in getattr(model, "_retained_rates", {})))


def _line_of(text):
    """The line of this file that holds ``text`` (the spelling a refusal must name)."""
    with open(__file__, encoding="utf-8") as handle:
        lines = [number for number, line in enumerate(handle, 1) if text in line]
    assert len(lines) == 1, (text, lines)
    return lines[0]


def test_a_term_free_summand_is_an_anonymous_source_of_the_balance():
    model, state, u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == -div(flux) + u * (1 - u))
    assert _summary(rate) == [("flux", "F", -1), ("source", "r_inline_1", 1)]
    assert rate.occurrences[1].payload.reg_name in model._sources
    assert rate.occurrences[0].payload is flux


def test_the_operand_order_and_the_side_of_the_term_do_not_matter():
    model, state, u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == 2 * u ** 2 - div(flux))
    assert _summary(rate) == [("source", "r_inline_1", 1), ("flux", "F", -1)]


def test_an_operand_at_coefficient_one_is_kept_as_written():
    model, state, u, flux = _scalar()
    operand = u * (1 - u)
    rate = model.rate("r", equation=ddt(state) == operand - div(flux))
    [stored] = _value(model, rate.occurrences[0])
    assert stored is operand


def test_a_negated_operand_folds_its_sign_into_the_source():
    model, state, u, flux = _scalar()
    operand = 2 * u ** 2
    rate = model.rate("r", equation=ddt(state) == -div(flux) - operand)
    assert _summary(rate) == [("flux", "F", -1), ("source", "r_inline_1", 1)]
    [stored] = _value(model, rate.occurrences[1])
    assert isinstance(stored, Neg)
    assert stored.a is operand


def test_an_exact_rational_scale_reaches_terms_and_operands():
    model, state, u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == (div(flux) + u) / 2)
    assert _summary(rate) == [("flux", "F", Fraction(1, 2)), ("source", "r_inline_1", 1)]
    [stored] = _value(model, rate.occurrences[1])
    assert isinstance(stored, Mul)
    assert stored.b is u


@pytest.mark.parametrize("spelling", ["product", "quotient"])
def test_exact_numbers_scale_a_term_from_either_side(spelling):
    model, state, _u, flux = _scalar()
    rhs = (Fraction(1, 3) * div(flux)) if spelling == "product" else (div(flux) / 3)
    rate = model.rate("r", equation=ddt(state) == rhs)
    assert _summary(rate) == [("flux", "F", Fraction(1, 3))]


def test_a_declared_source_and_inline_algebra_share_a_balance():
    model, state, u, flux = _scalar()
    forcing = model.source("forcing", on=state, value=[u])
    rate = model.rate("r", equation=ddt(state) == u ** 2 - forcing - div(flux))
    assert _summary(rate) == [("source", "r_inline_1", 1), ("source", "forcing", -1),
                              ("flux", "F", -1)]


def test_the_number_zero_adds_nothing_so_sum_starts_cleanly():
    model, state, _u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == sum([-div(flux), -div(flux)]))
    assert _summary(rate) == [("flux", "F", -1), ("flux", "F", -1)]


def test_one_anonymous_source_per_operand_in_written_order():
    model, state, u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == u - div(flux) + 3 * u ** 2)
    assert _summary(rate) == [("source", "r_inline_1", 1), ("flux", "F", -1),
                              ("source", "r_inline_2", 1)]


def test_a_tuple_or_a_vector_is_one_operand_with_one_value_per_component():
    model, state, (a, b), flux = _pair()
    rate = model.rate("r", equation=ddt(state) == -div(flux) + (a * b, -a * b))
    assert _summary(rate) == [("flux", "F", -1), ("source", "r_inline_1", 1)]
    assert len(_value(model, rate.occurrences[1])) == 2
    other = model.rate("s", equation=ddt(state) == VectorExpr([a, b]) - div(flux))
    assert _summary(other) == [("source", "s_inline_1", 1), ("flux", "F", -1)]


def test_an_operand_of_the_wrong_width_is_refused_with_its_line():
    model, state, (a, _b), flux = _pair()
    before = _observed(model)
    with pytest.raises(ValueError, match="written at .*test_rate_expression.*1 component"):
        model.rate("r", equation=ddt(state) == -div(flux) + a)
    assert _observed(model) == before
    model.rate("r", equation=ddt(state) == -div(flux) + (a, a))


def test_a_term_used_as_one_component_of_a_vector_is_refused():
    model, state, (a, _b), flux = _pair()
    with pytest.raises(TermNotAdditive, match="not one component"):
        model.rate("r", equation=ddt(state) == -div(flux) + (a, div(flux)))


@pytest.mark.parametrize("spelling", [
    "product by a quantity", "product by a quantity, left", "quotient by a quantity",
    "denominator", "function", "where", "source product",
])
def test_a_term_under_a_non_additive_node_is_refused_at_its_line(spelling):
    model, state, u, flux = _scalar()
    forcing = model.source("forcing", on=state, value=[u])
    build = {
        "product by a quantity": lambda: div(flux) * u,
        "product by a quantity, left": lambda: u * div(flux),
        "quotient by a quantity": lambda: div(flux) / u,
        "denominator": lambda: u / div(flux),
        "function": lambda: pm.exp(-div(flux) + u),
        "where": lambda: pm.where(u > 0, lambda: div(flux), lambda: 0),
        "source product": lambda: u * forcing,
    }[spelling]
    with pytest.raises(TermNotAdditive) as caught:
        model.rate("r", equation=ddt(state) == build())
    assert caught.value.rule == "R7"
    assert caught.value.site is not None and caught.value.site.file == __file__
    assert "r" not in model.module.list_operators()


def test_an_algebraic_operand_with_a_term_below_a_function_is_refused_not_split():
    model, state, u, flux = _scalar()
    with pytest.raises(TermNotAdditive):
        model.rate("r", equation=ddt(state) == u * (1 - div(flux)))


def test_a_refused_balance_declares_no_source():
    model, state, u, flux = _scalar()
    with pytest.raises(TermNotAdditive):
        model.rate("r", equation=ddt(state) == u + u * div(flux))
    assert model._sources == {}


def test_a_duplicate_balance_name_is_refused_before_any_anonymous_source():
    model, state, u, flux = _scalar()
    model.rate("r", equation=ddt(state) == -div(flux))
    before = dict(model._sources)
    with pytest.raises(ValueError, match="already declared"):
        model.rate("r", equation=ddt(state) == -div(flux) + u)
    assert model._sources == before


def test_an_anonymous_source_cannot_shadow_a_declared_source():
    model, state, u, flux = _scalar()
    model.source("r_inline_1", on=state, value=[u])
    with pytest.raises(ValueError, match="collides with a declared source or operator"):
        model.rate("r", equation=ddt(state) == -div(flux) + u)


def test_a_failure_while_declaring_the_second_source_restores_every_registry(monkeypatch):
    model, state, u, flux = _scalar()
    hyp = model._dsl._m
    snapshot = (dict(model._sources), dict(hyp._source_terms), list(model.module.list_operators()))
    real = type(model).source
    calls = []

    def failing(self, name, *args, **kwargs):
        calls.append(name)
        if len(calls) == 2:
            raise RuntimeError("injected failure")
        return real(self, name, *args, **kwargs)

    monkeypatch.setattr(type(model), "source", failing)
    with pytest.raises(RuntimeError, match="injected"):
        model.rate("r", equation=ddt(state) == u - div(flux) + u ** 2)
    monkeypatch.undo()
    assert calls == ["r_inline_1", "r_inline_2"]
    assert (dict(model._sources), dict(hyp._source_terms),
            list(model.module.list_operators())) == snapshot
    rate = model.rate("r", equation=ddt(state) == u - div(flux) + u ** 2)
    assert _summary(rate)[0] == ("source", "r_inline_1", 1)


def test_each_occurrence_records_the_line_that_wrote_it_outside_identity():
    model, state, u, flux = _scalar()
    rate = model.rate(
        "r",
        equation=ddt(state) == -div(flux) + u * (1 - u))
    declared, inline = rate.occurrences
    assert declared.origin.file == inline.origin.file == __file__
    assert declared.origin.line != inline.origin.line
    data = rate.view.balance.to_data()
    assert "origin" not in repr(data)
    assert replace(inline, origin=None).to_data() == inline.to_data()
    assert BalanceOccurrence.__dataclass_fields__["origin"].default is None


def test_rate_expr_accepts_an_inline_summand_only_with_its_operand():
    with pytest.raises((TypeError, ValueError)):
        RateExpr([("inline", object(), 1)])


@dataclass(frozen=True)
class _InspectionMethod:
    def validate(self, context=None):
        return True

    def validate_rate_contract(self, contract):
        return True

    def resolve_references(self, resolver):
        return self

    def to_data(self):
        return {"method": "inspection-only"}

    def freeze(self):
        return self


def _plan(*rates):
    plan = DiscretizationPlan()
    for rate in rates:
        plan.rates.add(rate, _InspectionMethod())
    return plan


def _logistic_transport():
    """L1 on the board: oblique advection and anisotropic diffusion, as M04 writes them, plus an
    inline logistic reaction ``r*u*(1-u)``."""
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    frame = Rectangle("unit-square-n11", lower=(0.0, 0.0), upper=(1.0, 1.0)
                      ).frame(Cartesian2D())
    model = pops.Model("logistic_transport", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    velocity = (0.4, -0.2)
    flux = model.flux("F", frame=frame, state=state,
                      components={axis: (a * u,) for axis, a in zip(frame.axes, velocity)},
                      waves={axis: (a,) for axis, a in zip(frame.axes, velocity)})
    diffusion = model.diffusive_flux("conduction", state=state, value=pm.CoeffGradient(
        u, ((0.02, 0.006), (0.006, 0.015))))
    growth = 3
    rate = model.rate("A", equation=ddt(state) == (
        -div(flux) + div(diffusion) + growth * u * (1 - u)))  # n11-l1-reaction
    model.select_balance(rate)
    return model, (flux, diffusion), rate


def test_partitions_cover_the_anonymous_source_exactly_once():
    model, (flux, diffusion), rate = _logistic_transport()
    assert [item.kind for item in rate.occurrences] == ["flux", "diffusion", "source"]
    assert _plan(rate).validate_for(model)
    assert _plan(rate.select(flux), rate.select(diffusion),
                 rate.select(*rate.inline)).validate_for(model)


def test_an_uncovered_anonymous_source_is_named_with_its_line():
    model, (flux, diffusion), rate = _logistic_transport()
    with pytest.raises(CoverageNotExact) as caught:
        _plan(rate.select(flux)).validate_for(model)
    message = str(caught.value)
    assert "missing=" in message
    assert "conduction" in message and "A_inline_1" in message
    assert "written at %s:%d" % (__file__, _line_of("# n11-l1-" + "reaction")) in message
    first, second = caught.value.terms
    assert "conduction" in first and "A_inline_1" in second and "written x1" in second
    assert caught.value.term is caught.value.terms[0] and caught.value.count == 0
    with pytest.raises(CoverageNotExact, match="duplicate physical balance occurrence"):
        _plan(rate, rate.select(*rate.inline)).validate_for(model)
    del diffusion


# --- Revision 2 of the lot 1 note: folding, grouping, zeros, names ---------------------------------


def test_folding_keeps_the_written_coefficient_as_documentary_data():
    model, state, u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == (div(flux) + u) / 3 - div(flux) - 2 * u ** 2)
    occurrences = rate.occurrences
    assert [item.written for item in occurrences] == [None, Fraction(1, 3), None, -1]
    [third] = _value(model, occurrences[1])
    assert isinstance(third, Mul) and isinstance(third.a, Const)
    assert third.a.value == Fraction(1, 3) and type(third.a.value) is Fraction
    assert third.b is u
    # 2*u**2 is one term-free operand; its sign folds in, so its occurrence is +1 times -(2*u**2).
    assert _summary(rate)[3] == ("source", "r_inline_2", 1)
    [negated] = _value(model, occurrences[3])
    assert isinstance(negated, Neg)


def test_an_exact_literal_folds_exactly_and_a_literal_zero_stays_as_written():
    model, state, (a, _b), flux = _pair()
    rate = model.rate("r", equation=ddt(state) == -div(flux) - (a, 0))
    first, zero = _value(model, rate.occurrences[1])
    assert isinstance(first, Neg) and first.a is a
    assert isinstance(zero, Const) and type(zero.value) is int and zero.value == 0
    signed = model.rate("s", equation=ddt(state) == -div(flux) - (a, 0.0))
    _first, float_zero = _value(model, signed.occurrences[1])
    assert float_zero.value == 0.0 and math.copysign(1.0, float_zero.value) == 1.0
    scalar, scalar_state, _u, scalar_flux = _scalar()
    halves = scalar.rate("h", equation=ddt(scalar_state) == (div(scalar_flux) + 3) / 2)
    [value] = _value(scalar, halves.occurrences[1])
    assert isinstance(value, Const) and value.value == Fraction(3, 2)


def test_python_grouping_decides_how_many_anonymous_sources_a_balance_holds():
    model, state, u, flux = _scalar()
    two = model.rate("two", equation=ddt(state) == -div(flux) + u + u ** 2)
    one = model.rate("one", equation=ddt(state) == -div(flux) + (u + u ** 2))
    tail = model.rate("tail", equation=ddt(state) == u + u ** 2 - div(flux))
    assert [name for _, name, _ in _summary(two)] == ["F", "two_inline_1", "two_inline_2"]
    assert [name for _, name, _ in _summary(one)] == ["F", "one_inline_1"]
    assert [name for _, name, _ in _summary(tail)] == ["tail_inline_1", "F"]


def test_a_numeric_zero_is_the_identity_but_a_scaled_term_and_an_empty_balance_are_not():
    model, state, _u, flux = _scalar()
    assert _summary(model.rate("plus_zero", equation=ddt(state) == -div(flux) + 0.0)) == [
        ("flux", "F", -1)]
    assert _summary(model.rate("zero_flux", equation=ddt(state) == 0 * div(flux))) == [
        ("flux", "F", 0)]
    before = _observed(model)
    with pytest.raises(ValueError, match="holds no term"):
        model.rate("nothing", equation=ddt(state) == 0)
    assert _observed(model) == before


def test_the_empty_sum_is_the_zero_rate_of_b6():
    # R7 refuses no empty balance; the b6 zero identity (test_implicit_stage_request) declares it
    # as the empty sum, which registers a balance with no occurrence and no anonymous source.
    from pops.math import RateExpr

    model, state, _u, _flux = _scalar()
    rate = model.rate("still", equation=ddt(state) == RateExpr(()))
    assert rate.occurrences == ()
    assert rate.inline == ()


def test_the_anonymous_sources_are_named_after_the_balance_and_selected_by_handle():
    model, (flux, diffusion), rate = _logistic_transport()
    assert [source.name for source in rate.inline] == ["A_inline_1"]
    assert rate.inline[0].reg_name == "A_inline_1" == rate.inline[0].name
    assert rate.inline[0] is rate.occurrences[2].payload
    assert rate.select(*rate.inline).occurrences == (rate.occurrences[2],)
    assert _plan(rate.select(flux, diffusion), rate.select(*rate.inline)).validate_for(model)
    with pytest.raises(ValueError, match="collides with already declared source"):
        model.source("A_inline_1", on=model._states["U"], value=[0])


def test_origins_and_written_coefficients_stay_outside_every_identity():
    model, state, u, flux = _scalar()
    rate = model.rate("r", equation=ddt(state) == -div(flux) - 2 * u ** 2 / 3)
    for item in rate.occurrences:
        assert replace(item, origin=None, written=None).to_data() == item.to_data()
        assert isinstance(item.origin, SourceSpan)
    assert BalanceOccurrence.__dataclass_fields__["written"].default is None
    assert "written" not in repr(rate.view.balance.to_data())


# --- Refusals at the written line, nonconservative products, joint applications ---------------


def _nonconservative():
    model = Model("nonconservative_n11", frame=FRAME)
    state = model.state("U", components=("mass", "stress"))
    mass, stress = state
    flux = model.flux("F", frame=FRAME, state=state,
                      components={X_AXIS: (mass, stress), Y_AXIS: (mass, stress)},
                      waves={X_AXIS: (1, 1), Y_AXIS: (1, 1)})
    product = model.nonconservative_product(
        "B grad U", state=state,
        matrices={X_AXIS: ((0, 0), (stress / mass, 2)), Y_AXIS: ((0, 0), (3, mass))},
        conservative_components=("mass",))
    return model, state, (mass, stress), flux, product


_NONCONSERVATIVE = {
    "n11-ncp-product": lambda m, s, p: m * p,  # n11-ncp-product
    "n11-ncp-exp": lambda m, s, p: pm.exp(p),  # n11-ncp-exp
    "n11-ncp-vector": lambda m, s, p: VectorExpr([m, s]) * p,  # n11-ncp-vector
}


def test_a_nonconservative_product_is_a_term_of_the_inline_algebra():
    model, state, (mass, stress), flux, product = _nonconservative()
    rate = model.rate("r", equation=ddt(state) == -div(flux) - product + (0 * mass, stress))
    assert [(item.kind, item.coefficient) for item in rate.occurrences] == [
        ("flux", -1), ("nonconservative", -1), ("source", 1)]
    before = _observed(model)
    for marker, build in _NONCONSERVATIVE.items():
        with pytest.raises(TermNotAdditive) as caught:
            build(mass, stress, product)
        assert caught.value.site.file == __file__
        assert caught.value.site.line == _line_of("# " + marker)
    assert _observed(model) == before


_SPELLINGS = {
    "n11-source-product": lambda u, f, s: u * s,  # n11-source-product
    "n11-source-quotient": lambda u, f, s: u / s,  # n11-source-quotient
    "n11-source-exp": lambda u, f, s: pm.exp(s),  # n11-source-exp
    "n11-flux-power": lambda u, f, s: div(f) ** 2,  # n11-flux-power
    "n11-flux-base": lambda u, f, s: 2 ** div(f),  # n11-flux-base
    "n11-flux-abs": lambda u, f, s: abs(div(f)),  # n11-flux-abs
    "n11-flux-less": lambda u, f, s: u < div(f),  # n11-flux-less
    "n11-flux-greater": lambda u, f, s: div(f) > 0,  # n11-flux-greater
    "n11-flux-where": lambda u, f, s: pm.where(u > 0, lambda: div(f), lambda: 0),  # n11-flux-where
    "n11-vector-source": lambda u, f, s: VectorExpr([u]) * s,  # n11-vector-source
    "n11-vector-flux": lambda u, f, s: VectorExpr([div(f)]),  # n11-vector-flux
}


@pytest.mark.parametrize("marker", sorted(_SPELLINGS))
def test_each_refused_spelling_names_the_line_that_wrote_it(marker):
    model, state, u, flux = _scalar()
    source = model.source("S", on=state, value=[u])
    before = _observed(model)
    with pytest.raises(TermNotAdditive) as caught:
        _SPELLINGS[marker](u, flux, source)
    assert caught.value.rule == "R7" and caught.value.phase == "tracing"
    assert caught.value.site.file == __file__
    assert caught.value.site.line == _line_of("# " + marker)
    assert _observed(model) == before


class _SurfaceRefusal(ValueError):
    """The shape of the user surface's refusal family: a site, then ``super().__init__(message)``."""

    def __init__(self, message, *, site):
        self.site = site
        super().__init__(message)


class _SurfaceTermNotAdditive(_SurfaceRefusal, TermNotAdditive):
    pass


class _SurfaceCoverage(_SurfaceRefusal, CoverageNotExact):
    pass


def test_a_cooperative_subclass_keeps_its_own_site_and_is_caught_as_the_board_refusal():
    site = SourceSpan(file="user.py", line=12, column=0)
    with pytest.raises(TermNotAdditive) as caught:
        raise _SurfaceTermNotAdditive("u * div(F) is not a summand", site=site)
    assert caught.value.site is site
    assert caught.value.term is None and caught.value.construct is None
    assert str(caught.value) == "u * div(F) is not a summand"
    assert caught.value.rule == "R7" and isinstance(caught.value, ValueError)
    with pytest.raises(CoverageNotExact) as covered:
        raise _SurfaceCoverage("div(F) is not covered", site=site)
    assert covered.value.site is site and covered.value.terms == ()
    assert str(covered.value) == "div(F) is not covered"
    positional = TermNotAdditive("written by hand")
    assert str(positional) == "written by hand" and positional.site is None


def _joint():
    model = pops.Model("joint_n11")
    left = model.species("left", state=["rho"])
    right = model.species("right", state=["rho", "energy"])
    module = model.module
    signature = Signature((left.space, right.space), RateBundle({
        "particle_exchange": Rate(left.space), "gas_exchange": Rate(right.space)}))

    @module.operator("collision", signature=signature, kind="coupled_rate")
    def collision(a, b):
        return {"particle_exchange": [b[0] - a[0]], "gas_exchange": [a[0] - b[0], b[1]]}

    @module.operator("temperature", signature=Signature((right.space,), right.space),
                     kind="expression")
    def temperature(b):
        return [b[1] / b[0], b[1]]

    return (model, left, right, module.apply(collision, left, right),
            module.apply(temperature, right))


def test_a_component_of_a_data_output_is_read_inline_and_a_rate_component_is_a_term():
    model, left, _right, collision, temperature = _joint()
    (rho,) = left
    data = temperature["value"][0]
    declared = model.source("declared", on=left, value=[rho * data])
    rate = model.rate("r", equation=ddt(left) == rho * data)
    [anonymous] = rate.inline
    registry = model._multi_module.operator_registry()
    assert registry.get(anonymous.reg_name).signature == registry.get(declared.reg_name).signature
    assert repr(registry.get(anonymous.reg_name).body) == repr(
        registry.get(declared.reg_name).body)
    before = _observed(model)
    for build in (lambda: rho * collision[left][0], lambda: collision[left][0] + rho):
        with pytest.raises(TermNotAdditive):
            model.rate("q", equation=ddt(left) == build())
        assert _observed(model) == before
    joint = model.rate("joint", equation=ddt(left) == collision[left] - rho * data)
    assert [item.kind for item in joint.occurrences] == ["projection", "source"]


# --- R4: a refused balance leaves the model as it was --------------------------------------------


def test_every_hoisted_refusal_leaves_the_model_unchanged_and_the_fix_then_declares():
    model, state, u, flux = _scalar()
    foreign, _foreign_state, _foreign_u, foreign_flux = _scalar()
    model.source("r_inline_1", on=state, value=[u])
    model.source("operator", on=state, value=[u])
    before = _observed(model)
    for name, rhs, error in (
            ("r", -div(foreign_flux) + u, ValueError),
            ("operator", -div(flux) + u, ValueError),
            ("r", -div(flux) + u, ValueError),
            ("w", -div(flux) + (u, u), ValueError),
    ):
        with pytest.raises(error):
            model.rate(name, equation=ddt(state) == rhs)
        assert _observed(model) == before, name
    del foreign
    assert _summary(model.rate("w", equation=ddt(state) == -div(flux) + u)) == [
        ("flux", "F", -1), ("source", "w_inline_1", 1)]


@pytest.mark.parametrize("seam", ["source", "_balance_signature", "_register_balance_view"])
def test_a_failure_inside_the_transaction_restores_the_model(monkeypatch, seam):
    model, state, u, flux = _scalar()
    model.rate("kept", equation=ddt(state) == -div(flux) + u)
    model.module  # materialize the caches the transaction must restore
    before = _observed(model)
    real = getattr(type(model), seam)
    calls = []

    def failing(self, *args, **kwargs):
        calls.append(1)
        if seam != "source" or len(calls) == 2:
            raise RuntimeError("injected failure")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(model), seam, failing)
    with pytest.raises(RuntimeError, match="injected"):
        model.rate("r", equation=ddt(state) == u - div(flux) + u ** 2)
    monkeypatch.undo()
    assert _observed(model) == before
    assert _summary(model.rate("r", equation=ddt(state) == u - div(flux) + u ** 2)) == [
        ("source", "r_inline_1", 1), ("flux", "F", -1), ("source", "r_inline_2", 1)]


def _multi():
    model = pops.Model("multi_n11")
    first = model.species("first", state=("u",))
    second = model.species("second", state=("v",))
    phi = model.field("phi", components=("p",))
    psi = model.field("psi", components=("q",))
    spaces = model.module.field_spaces()
    (p,) = model.module.field_symbols(spaces["phi"])
    (q,) = model.module.field_symbols(spaces["psi"])
    return model, first, second, (phi, p), (psi, q)


def test_multi_state_inline_operands_read_what_a_declared_source_reads():
    model, first, second, (phi, p), _psi = _multi()
    (u,), (v,) = first, second
    declared = model.source("declared", on=first, value=[u * v])
    with_field = model.source("with_field", on=first, value=[u * p], fields=phi)
    other = model.rate("other", equation=ddt(first) == u * v)
    field = model.rate("field", equation=ddt(first) == u * p)
    registry = model._multi_module.operator_registry()
    for rate, reference in ((other, declared), (field, with_field)):
        [anonymous] = rate.inline
        assert registry.get(anonymous.reg_name).signature == registry.get(
            reference.reg_name).signature
        assert repr(registry.get(anonymous.reg_name).body) == repr(
            registry.get(reference.reg_name).body)


def _validated_multi(spelling):
    from pops.numerics import StateStorage
    model, first, second, (phi, p), _psi = _multi()
    (u,), (v,) = first, second
    if spelling == "declared":
        rhs = model.source("declared", on=first, value=[u * v + p], fields=phi)
    else:
        rhs = u * v + p
    rate = model.rate("balance", equation=ddt(first) == rhs)
    case = pops.Case("multi_parity_" + spelling)
    block = case.block("first", model, states=(first,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case.numerics(plan, block=block)
    return pops.validate(case) is case


def test_a_multi_state_inline_balance_validates_as_its_declared_spelling():
    assert _validated_multi("declared") is _validated_multi("inline") is True


def test_multi_state_refusals_of_two_fields_and_of_the_legacy_route_are_atomic():
    model, first, _second, (_phi, p), (psi, q) = _multi()
    (u,) = first
    reader = model.source("reader", on=first, value=[q], fields=psi)
    before = _observed(model)
    with pytest.raises(ValueError, match="written at .*test_rate_expression.*reads 2 fields"):
        model.rate("two", equation=ddt(first) == p * q)
    assert _observed(model) == before
    with pytest.raises(ValueError, match="incompatible FieldSpaces"):
        model.rate("mixed", equation=ddt(first) == u * p + reader)
    assert _observed(model) == before
    assert [item.kind for item in model.rate(
        "mixed", equation=ddt(first) == u * q + reader).occurrences] == ["source", "source"]


# --- The source-name fix: a source-only view finds a source by its registry name -----------------


def test_a_declared_source_whose_name_has_a_space_builds_its_source_only_program():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.lib.time import ForwardEuler
    from pops.numerics import StateStorage
    frame = Rectangle("n11-names", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("spaced_source", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    growth = model.source("my growth", on=state, value=(3.0 * q,))
    rate = model.rate("balance", equation=ddt(state) == growth)
    case = pops.Case("spaced")
    block = case.block("material", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case.numerics(plan, block=block)
    case.program(ForwardEuler(block[state], rate=rate))
    assert pops.validate(case) is case
