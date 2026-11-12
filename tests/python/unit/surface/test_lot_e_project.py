"""Lot D-E of N15 (ROM-98, OD31), part E: the projection of an analytic initial value.

``initial={u: pops.project(e, quadrature=Q)}`` averages e over each cell by the rule Q mapped
onto the cell, and ``pops.project(e, integral=I)`` divides the exact integral I(cell) by the
measure of the cell (2.6.2, CN12); a value that reads no coordinate is exact and written alone.
The elaborator lowers each on the analytic route of b6: ``Analytic(frame, components=(e,),
cell_integrals=(I,))`` under ``ConservativeCellAverage``, I being the measure times the sum of the
weights times e at the mapped nodes for a rule. Each case gives the case written by hand for b6
(SU-T3: program IR, artifact without provenance, semantic identity, C++). The values are checked
natively by ``tests/python/integration/runtime/test_surface_lot_e_project_native.py``.
"""
import json
import types

import pytest

from examples.api_v02 import b6_numerics

import pops
import pops.analytic as an
import pops.surface as surface
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Analytic
from pops.lib.numerics import rusanov
from pops.lib.quadrature import GaussHermite, GaussLegendre
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import ConstParam, RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.surface.math import ddt, div, erf, exp, sqrt, vector
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.traced_method import Candidate, method


def _n8_method():
    @method
    def ssprk2(B, p, rate):
        u, t, h = B.state, B.time, p.dt
        first = u + h * rate(u, t)
        return Candidate((u + first + h * rate(first, t + h)) / 2, t + h)

    return ssprk2


_n8_ssprk2 = _n8_method()


# --- the cases ------------------------------------------------------------------------------------

def _parts():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    b = model.constant("b", value=0.25)
    advection = model.flux(of=u, value=vector(a, b) * u)
    model.balance(ddt(u) == -div(advection))
    return types.SimpleNamespace(domain=domain, model=model, u=u, x=domain.x, y=domain.y,
                                 advection=advection)


def _case(parts, initial):
    numerics = surface.Numerics(represent={parts.u: surface.CellAverage()},
                                schemes={parts.advection: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=parts.model), dt=1e-3),
                              initial={parts.u: initial})
    return surface.Case(model=parts.model, numerics=numerics, program=program,
                        layout={parts.domain: surface.Uniform(cells=(16, 8))})


def _bump(x, y, exp=exp):
    return 0.05 + 0.95 * exp(-120.0 * ((x - 0.3) ** 2 + (y - 0.35) ** 2))


def _hand_case(value):
    """The case written for b6 and N8; ``value(frame, model)`` gives its initial provider."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.25)))
    provider = value(frame, model)
    u = model.state("u", components=("u",))
    flux_u = model.flux("flux_u", state=u, frame=frame,
                        components={x: (a * u[0],), y: (b * u[0],)}, waves={x: (a,), y: (b,)})
    rate = model.rate("u", equation=pm.ddt(u) == -pm.div(flux_u))
    plan = DiscretizationPlan()
    selection = rate.select(flux_u)
    plan.rates.add(selection, b6_numerics.rusanov(flux_u, u))
    case = pops.Case("advection")
    block = case.block("advection", model)
    case.numerics(plan, block=block)
    case.initials.add(InitialCondition(state=block[u], value=provider,
                                       projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=Proposal(0.001))
    return case, Uniform(CartesianGrid(frame=frame, cells=(16, 8),
                                       periodic=PeriodicAxes((x, y))))


def _resolved(case, layout):
    return pops.resolve(pops.validate(case), layout=layout)


def _without_provenance(value):
    if isinstance(value, dict):
        return {key: _without_provenance(item) for key, item in value.items()
                if key != "provenance"}
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value


def _assert_same(case, hand):
    elaborated = case.elaborated
    assert elaborated.case._time._serialize(include_provenance=False) == hand[0]._time._serialize(
        include_provenance=False)
    resolved = _resolved(elaborated.case, elaborated.layout)
    by_hand = _resolved(*hand)
    assert _without_provenance(json.loads(resolved.snapshot._artifact_canonical_json)) \
        == _without_provenance(json.loads(by_hand.snapshot._artifact_canonical_json))
    assert resolved.snapshot.semantic_identity == by_hand.snapshot.semantic_identity
    cpp = [emit_cpp_program(r.time, model_graph=ProgramModelGraph.from_resolved_blocks(r.blocks))
           for r in (resolved, by_hand)]
    assert cpp[0] == cpp[1]


def _mapped(bounds, axis, fraction):
    lower, upper = bounds.lower(axis), bounds.upper(axis)
    return lower + (upper - lower) * fraction


# --- SU-T3: the elaborated case is the case written by hand --------------------------------------

def test_a_one_point_rule_samples_the_centre_of_each_cell():
    parts = _parts()
    case = _case(parts, surface.project(_bump(parts.x, parts.y), quadrature=GaussLegendre(1)))

    def value(frame, model):
        x, y = frame.axes
        bounds = an.CellBounds(frame)
        integral = bounds.measure * _bump(_mapped(bounds, x, 0.5), _mapped(bounds, y, 0.5),
                                          exp=an.exp)
        return Analytic(frame=frame, components=(_bump(an.x(frame), an.y(frame), exp=an.exp),),
                        cell_integrals=(integral,))

    _assert_same(case, _hand_case(value))


def test_a_rule_of_two_points_per_axis_is_the_tensor_sum_of_its_weights():
    parts = _parts()
    rule = GaussLegendre(2)
    case = _case(parts, surface.project(_bump(parts.x, parts.y), quadrature=rule))

    def value(frame, model):
        x, y = frame.axes
        bounds = an.CellBounds(frame)
        nodes = [((node + 1) / 2, weight / 2) for node, weight in zip(rule.nodes, rule.weights,
                                                                         strict=True)]
        terms = [wx * wy * _bump(_mapped(bounds, x, tx), _mapped(bounds, y, ty), exp=an.exp)
                 for tx, wx in nodes for ty, wy in nodes]
        average = terms[0]
        for term in terms[1:]:
            average = average + term
        return Analytic(frame=frame, components=(_bump(an.x(frame), an.y(frame), exp=an.exp),),
                        cell_integrals=(bounds.measure * average,))

    _assert_same(case, _hand_case(value))


def _gaussian_integral(lower_x, upper_x, lower_y, upper_y, erf, sqrt):
    """The exact integral of exp(-beta ((x - 0.5)^2 + (y - 0.5)^2)) over a cell, beta = 30."""
    scale = sqrt(30.0)
    factor = 3.141592653589793 / 30.0 / 4

    return factor * (erf(scale * (upper_x - 0.5)) - erf(scale * (lower_x - 0.5))) \
        * (erf(scale * (upper_y - 0.5)) - erf(scale * (lower_y - 0.5)))


def test_an_exact_integral_written_with_the_bounds_of_the_cell():
    parts = _parts()
    x, y = parts.x, parts.y
    gaussian = exp(-30.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    def integral(cell):
        return _gaussian_integral(cell.lower(x), cell.upper(x), cell.lower(y), cell.upper(y),
                                  erf, sqrt)

    case = _case(parts, surface.project(gaussian, integral=integral))

    def value(frame, model):
        fx, fy = frame.axes
        bounds = an.CellBounds(frame)
        exact = _gaussian_integral(bounds.lower(fx), bounds.upper(fx), bounds.lower(fy),
                                   bounds.upper(fy), an.erf, an.sqrt)
        point = an.exp(-30.0 * ((an.x(frame) - 0.5) ** 2 + (an.y(frame) - 0.5) ** 2))
        return Analytic(frame=frame, components=(point,), cell_integrals=(exact,))

    _assert_same(case, _hand_case(value))


def test_a_value_that_reads_no_coordinate_is_exact_and_written_alone():
    parts = _parts()
    g = parts.model.constant("g", value=0.125)
    case = _case(parts, 2 * g + 0.5)

    def value(frame, model):
        model.param(ConstParam("g", 0.125))
        return Analytic(frame=frame, components=(2.0 * an.constant(0.125) + 0.5,))

    _assert_same(case, _hand_case(value))
    report = str(surface.inspect(case))
    assert "cell average = exact, the value reads no coordinate (derived)" in report


def test_a_runtime_parameter_of_the_expression_is_bound_at_bind():
    parts = _parts()
    amplitude = parts.model.parameter("amplitude", default=0.95)
    x, y = parts.x, parts.y
    e = 0.05 + amplitude * exp(-120.0 * ((x - 0.3) ** 2 + (y - 0.35) ** 2))
    case = _case(parts, surface.project(e, quadrature=GaussLegendre(1)))

    def value(frame, model):
        fx, fy = frame.axes
        bounds = an.CellBounds(frame)
        handle = model.param(RuntimeParam("amplitude", default=0.95))

        def bump(px, py):
            return 0.05 + an.param(handle) * an.exp(-120.0 * ((px - 0.3) ** 2
                                                             + (py - 0.35) ** 2))

        return Analytic(frame=frame, components=(bump(an.x(frame), an.y(frame)),),
                        cell_integrals=(bounds.measure * bump(_mapped(bounds, fx, 0.5),
                                                              _mapped(bounds, fy, 0.5)),))

    _assert_same(case, _hand_case(value))


def test_inspect_reports_the_rule_and_the_route():
    parts = _parts()
    report = str(surface.inspect(_case(
        parts, surface.project(_bump(parts.x, parts.y), quadrature=GaussLegendre(1)))))
    assert "value=Analytic(frame=frame, components=(" in report
    assert "quadrature = pops.Quadrature(nodes=(0.0,), weights=(2.0,), interval=(-1.0, 1.0)) " \
        "(stated)" in report
    assert "projection = ConservativeCellAverage() (derived)" in report


# --- the refusals ----------------------------------------------------------------------------------

def test_project_takes_a_rule_or_an_integral():
    parts = _parts()
    e = _bump(parts.x, parts.y)
    with pytest.raises(surface.AmbiguousElaboration, match="neither the rule nor the integral"):
        surface.project(e)
    with pytest.raises(TypeError, match="one of them"):
        surface.project(e, quadrature=GaussLegendre(1), integral=lambda cell: 0)
    with pytest.raises(TypeError, match="takes a pops.Quadrature"):
        surface.project(e, quadrature=2)
    with pytest.raises(TypeError, match="a function of cell"):
        surface.project(e, integral=1.0)
    with pytest.raises(ValueError, match="finite interval"):
        surface.project(e, quadrature=GaussHermite(2))
    with pytest.raises(TypeError, match="quadrature= or integral="):
        surface.project(e, projection="average")


def test_a_coordinate_written_alone_is_ambiguous():
    parts = _parts()
    with pytest.raises(surface.AmbiguousElaboration, match="reads a coordinate") as caught:
        _case(parts, _bump(parts.x, parts.y))
    assert "pops.project(e, quadrature=GaussLegendre(n))" in caught.value.write


def test_a_coordinate_of_another_support_is_a_support_mismatch():
    parts = _parts()
    other = surface.Box(x=(0.0, 2.0), y=(0.0, 1.0))
    with pytest.raises(surface.SupportMismatch, match="coordinate x of another support"):
        _case(parts, surface.project(other.x, quadrature=GaussLegendre(1)))


def test_a_value_that_reads_a_datum_or_a_quantity_names_its_lot():
    parts = _parts()
    u0 = parts.model.data("u0", surface.Scalar(parts.domain))
    with pytest.raises(surface.NotYetElaborated, match="reads the datum u0"):
        _case(parts, 2 * u0)
    with pytest.raises(surface.NotYetElaborated, match="reads u"):
        _case(parts, surface.project(parts.u * parts.x, quadrature=GaussLegendre(1)))


def test_one_rule_per_axis_counts_the_axes():
    parts = _parts()
    with pytest.raises(ValueError, match="gives 1 rules for a support of 2 axes"):
        _case(parts, surface.project(_bump(parts.x, parts.y), quadrature=(GaussLegendre(1),)))


def test_the_cell_of_an_integral_reads_its_bounds():
    parts = _parts()

    def width(cell):
        return cell.width(parts.x)

    with pytest.raises(surface.NotYetElaborated, match="cell.width"):
        _case(parts, surface.project(parts.x, integral=width))

    def coordinate(cell):
        return parts.x * cell.measure

    with pytest.raises(TypeError, match="not a coordinate"):
        _case(parts, surface.project(parts.x, integral=coordinate))


def test_erf_is_a_word_of_the_initial_values_only():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    advection = model.flux(of=u, value=vector(a, erf(a)) * u)
    model.balance(ddt(u) == -div(advection))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=advection)
    with pytest.raises(surface.NotYetElaborated, match=r"erf\(a\) in a value of the model.*N11"):
        _case(parts, 0.5)


def test_a_word_that_the_analytic_route_lacks_names_g07():
    from pops.surface.math import expm1

    parts = _parts()
    with pytest.raises(surface.NotYetElaborated, match=r"expm1\(.*pops.analytic.*G07"):
        _case(parts, surface.project(expm1(parts.x), quadrature=GaussLegendre(1)))
