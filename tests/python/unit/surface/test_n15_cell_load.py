"""N15 FaceFlux/CellRate lot FC-2 (ROM-98, note §3.6): ``cell.load(e)`` and the coordinates read
at the centre of the cell.

Under one constant mode the rate of ``cell.load(e)`` is the mean of e under the rule of the
representation, m_K(e) = Σ_q ŵ_q e(x_q): the load over μ(K) for an average. Its integrand reads
numbers, constants of the model and the coordinates of the support, so m_K(e) is a field fixed on
a static layout, an analytic auxiliary input of b6 (``AnalyticAux`` over the bounds of the cell).
A coordinate read in a ``CellRate`` value, or in the law of a source under ``pointwise()``, is the
coordinate of the centre of the cell, an auxiliary input as well (2.6.4).

SU-T2: each refusal names the user's line (``# <- name``), the rule or the lot: a load under a
representation that states no rule (``AmbiguousElaboration`` naming ``quadrature=``, N16 4.4), a
load added to a value read at the centre or scaled by a factor that varies on the cell (D2), an
integrand that reads the state (N16 lot 2) or a parameter (a later lot of N15).
"""
import os

import pytest

import pops
import pops.surface as surface
from pops.lib.numerics import pointwise, upwind
from pops.lib.quadrature import GaussLegendre
from pops.lib.representations import CellAverage, CellTotal, PointValue
from pops.lib.time.methods import ssprk2
from pops.surface.discrete import CellRate
from pops.surface.math import ddt, div, pi, sin

HERE = os.path.abspath(__file__)
LENGTH = 3.0
DOMAIN = surface.Box(x=surface.Periodic(0, LENGTH))
X = DOMAIN.x
RULE = CellAverage(quadrature=GaussLegendre(2))


def _line(marker):
    lines = [number for number, text in enumerate(open(HERE, encoding="utf-8"), start=1)
             if text.rstrip().endswith("# <- " + marker)]
    assert len(lines) == 1, "marker %r is not on exactly one line" % marker
    return lines[0]


def _at(error, marker):
    return (error.site.file, error.site.line) == (HERE, _line(marker))


def _case(scheme, representation=RULE, value=None, flux=True):
    model = surface.Model("forced")
    u = model.quantity("u", surface.Scalar(DOMAIN))
    k = model.parameter("k", default=2.0)
    u0 = model.data("u0", surface.Scalar(DOMAIN))
    source = model.source(of=u, value=(-k * u) if value is None else value(u, k))
    schemes = {source: scheme}
    if flux:
        transport = model.flux(of=u, value=u)
        schemes[transport] = upwind()
        model.balance(ddt(u) == -div(transport) + source)
    else:
        model.balance(ddt(u) == source)
    numerics = surface.Numerics(represent={u: representation}, schemes=schemes)
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    return dict(model=model, numerics=numerics, program=program,
                layout={DOMAIN: surface.Uniform(cells=16)})


def _elaborated(scheme, **options):
    return surface.Case(**_case(scheme, **options))


def _row(case):
    (row,) = [row for row in surface.inspect(case).rows if row.construct.startswith(
        "model.source(")]
    return row, {choice.name: choice.value for choice in row.choices}


def _resolved(case):
    elaborated = case.elaborated
    return pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)


# --- the schemes -----------------------------------------------------------------------------------

@surface.scheme
def forcing(cell, term):
    return CellRate(cell.load(sin(2 * pi * X / LENGTH)))


@surface.scheme
def scaled_forcing(cell, term):
    k = term.quantity.model.parameters[0]
    return CellRate(k * cell.load(sin(2 * pi * X / LENGTH)))


@surface.scheme
def two_loads(cell, term):
    return CellRate(cell.load(X) - cell.load(X * X) / 2)


@surface.scheme
def at_centre(cell, term):
    q = term.quantity
    return CellRate(-X * cell.value(q))


@surface.scheme
def load_and_value(cell, term):
    q = term.quantity
    return CellRate(cell.load(X) + cell.value(q))


@surface.scheme
def varying_factor(cell, term):
    return CellRate(X * cell.load(X))


@surface.scheme
def load_of_state(cell, term):
    q = term.quantity
    return CellRate(cell.load(cell.value(q) * X))


@surface.scheme
def load_of_parameter(cell, term):
    k = term.quantity.model.parameters[0]
    return CellRate(cell.load(k * X))


@surface.scheme
def load_under_sin(cell, term):
    return CellRate(sin(cell.load(X)))


# --- what is served --------------------------------------------------------------------------------

def test_a_load_under_a_rule_is_the_mean_of_its_integrand_an_auxiliary_input():
    case = _elaborated(forcing())
    row, choices = _row(case)
    assert row.elaborated == ("source_u = model.source('source_u', on=u, "
                              "value=(Var('source_u_load_1', 'aux'),))")
    assert choices["kind of value"].startswith("a load")
    assert choices["source_u_load_1"].startswith("the mean of sin(2 * pi * x / 3.0) on the cell")
    assert "the mean of the load on the cell" in choices["delivered"]
    module = case.elaborated.model.module
    assert "source_u_load_1" in module.aux()
    _resolved(case)


def test_a_load_times_a_parameter_and_a_sum_of_loads_are_loads():
    row, choices = _row(_elaborated(scaled_forcing()))
    assert "(k * Var('source_u_load_1', 'aux'))" in row.elaborated
    row, choices = _row(_elaborated(two_loads()))
    assert "source_u_load_1" in choices and "source_u_load_2" in choices
    assert "(Var('source_u_load_1', 'aux') - (Var('source_u_load_2', 'aux') / 2))" in \
        row.elaborated


def test_the_mean_follows_the_rule_of_the_representation():
    """Two rules give two means: the auxiliary input is built from the rule that the
    representation states, never from a name."""
    two = _resolved(_elaborated(forcing()))
    three = _resolved(_elaborated(forcing(), representation=CellAverage(
        quadrature=GaussLegendre(3))))
    assert two.snapshot.semantic_identity != three.snapshot.semantic_identity


def test_a_coordinate_read_by_a_cell_rate_is_the_centre_of_the_cell():
    case = _elaborated(at_centre(), representation=CellAverage())
    row, choices = _row(case)
    assert "Var('coordinate_x', 'aux')" in row.elaborated
    assert choices["x"] == "the coordinate x at the centre of the cell, an analytic auxiliary " \
                           "input (2.6.4)"
    _resolved(case)


def test_a_coordinate_in_the_law_of_a_source_under_pointwise_is_the_centre_of_the_cell():
    case = _elaborated(pointwise(), representation=CellAverage(),
                       value=lambda u, k: -k * X * u)
    row, _ = _row(case)
    assert "Var('coordinate_x', 'aux')" in row.elaborated
    _resolved(case)


@pytest.mark.parametrize("representation", [CellTotal(), PointValue()])
def test_a_coordinate_at_the_centre_is_served_under_every_constant_mode(representation):
    _resolved(_elaborated(at_centre(), representation=representation))


# --- SU-T2 -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("representation", [CellAverage(), CellTotal(), PointValue()])
def test_a_load_under_a_representation_without_a_rule_is_ambiguous(representation):
    chosen = forcing()  # <- no-rule
    options = _case(chosen, representation=representation)
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        surface.Case(**options)
    assert _at(caught.value, "no-rule")
    assert "quadrature=GaussLegendre(n)" in str(caught.value)
    assert "states no rule" in str(caught.value)


@pytest.mark.parametrize("scheme, hint", [
    (load_and_value, "a sum of loads"),
    (varying_factor, "cell.load(x * e)"),
    (load_under_sin, "cell.load(e) at the top of the value"),
])
def test_a_load_that_is_not_a_sum_of_scaled_loads_is_ambiguous(scheme, hint):
    chosen = scheme()  # <- mixed
    with pytest.raises(surface.AmbiguousElaboration) as caught:
        _elaborated(chosen)
    assert _at(caught.value, "mixed")
    assert hint in str(caught.value)
    assert "(D2)" in str(caught.value)


def test_a_load_whose_integrand_reads_the_state_waits_for_n16_lot_2():
    chosen = load_of_state()  # <- state
    with pytest.raises(surface.NotYetElaborated) as caught:
        _elaborated(chosen)
    assert _at(caught.value, "state")
    assert "N16 lot 2 (the reads at the points of a rule, 4.1)" in str(caught.value)


def test_a_load_whose_integrand_reads_a_parameter_waits_for_its_lot():
    chosen = load_of_parameter()  # <- parameter
    with pytest.raises(surface.NotYetElaborated) as caught:
        _elaborated(chosen)
    assert _at(caught.value, "parameter")
    assert "auxiliary inputs that read parameters" in str(caught.value)


def test_a_coordinate_in_a_flux_keeps_its_owner():
    model = surface.Model("drift")
    u = model.quantity("u", surface.Scalar(DOMAIN))
    u0 = model.data("u0", surface.Scalar(DOMAIN))
    transport = model.flux(of=u, value=X * u)
    model.balance(ddt(u) == -div(transport))
    numerics = surface.Numerics(represent={u: CellAverage()}, schemes={transport: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.NotYetElaborated, match="coordinate fields of G03"):
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={DOMAIN: surface.Uniform(cells=16)})


def test_two_sources_that_read_a_coordinate_share_its_auxiliary_input():
    model = surface.Model("two")
    u = model.quantity("u", surface.Scalar(DOMAIN))
    u0 = model.data("u0", surface.Scalar(DOMAIN))
    first = model.source(of=u, value=-X * u)
    second = model.source(of=u, value=X * X * u)
    model.balance(ddt(u) == first + second)
    numerics = surface.Numerics(represent={u: CellAverage()},
                                schemes={first: pointwise(), second: pointwise()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={DOMAIN: surface.Uniform(cells=16)})
    assert [name for name in case.elaborated.model.module.aux()] == ["coordinate_x"]
    _resolved(case)
