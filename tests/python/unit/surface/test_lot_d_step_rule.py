"""Lot D-E of N15 (ROM-98, OD31), part D: ``dt=cfl_limit(flux, cfl=, max_dt=)``.

``pops.Advance(method, dt=cfl_limit(advection, cfl=0.45, max_dt=1e-2))`` elaborates onto the
compiled step rule of N8, ``elaborate(..., proposal=StepRule(cfl, max_dt, strategy=AdaptiveCFL(cfl, max_dt)))``: each step is
b6's CFL selection on the wave speeds of the model, capped by ``max_dt``, and the run lands on
its frontiers. The rule names every transport flux of the model, each with derived waves (R9),
and the model has no diffusive flux: these are what b6's selection reads. Each case gives the
case written by hand for b6 (SU-T3: program IR, artifact without provenance, semantic identity,
C++). The run is checked natively by
``tests/python/integration/runtime/test_surface_lot_d_step_rule_native.py``.
"""
import inspect
import json
import types

import pytest

import pops
import pops.analytic as an
from examples.api_v02 import b6_numerics
import pops.surface as surface
from pops import math as pm
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Analytic
from pops.lib.numerics import rusanov, upwind
from pops.lib.time import cfl_limit
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import ConstParam
from pops.projection import ConservativeCellAverage
from pops.surface.math import ddt, div, dot, grad, vector
from pops.time._program.elaboration import Advance, StepRule, elaborate
from pops.time._step.strategy import AdaptiveCFL
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
    return types.SimpleNamespace(domain=domain, model=model, u=u, advection=advection)


def _case(parts, dt, schemes=None, **advance):
    numerics = surface.Numerics(represent={parts.u: surface.CellAverage()},
                                schemes=schemes or {parts.advection: rusanov()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=parts.model), dt=dt, **advance),
                              initial={parts.u: 0.05})
    return surface.Case(model=parts.model, numerics=numerics, program=program,
                        layout={parts.domain: surface.Uniform(cells=(16, 8))})


def _rule(fraction, max_dt=None):
    """N8's rule as cfl_limit realizes it: the rule, and master's AdaptiveCFL for pops.run."""
    return StepRule(fraction, max_dt, strategy=AdaptiveCFL(fraction, max_dt))


def _hand_case(rule):
    """The case written for b6 and N8, its Advance elaborated from ``rule``."""
    frame = CartesianDomain("box", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x, y = frame.axes
    model = pops.Model("advection", frame=frame)
    a = model.value(model.param(ConstParam("a", 1.0)))
    b = model.value(model.param(ConstParam("b", 0.25)))
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
    case.initials.add(InitialCondition(
        state=block[u], value=Analytic(frame=frame, components=(an.constant(0.05),)),
        projection=ConservativeCellAverage()))
    elaborate(Advance(_n8_ssprk2(rate=lambda state: selection(state))), case=case,
              state=block[u], proposal=rule)
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


# --- SU-T3: the elaborated case is the case written by hand --------------------------------------

def test_cfl_limit_is_the_compiled_step_rule_of_n8():
    parts = _parts()
    case = _case(parts, cfl_limit(parts.advection, cfl=0.45, max_dt=1e-2))
    _assert_same(case, _hand_case(_rule(0.45, 1e-2)))
    control = case.elaborated.case._time._advance_control
    assert control.step_rule == (0.45, 1e-2) and control.first_dt is None


def test_cfl_limit_without_a_cap_and_with_a_list_of_fluxes():
    parts = _parts()
    _assert_same(_case(parts, cfl_limit(parts.advection, cfl=0.8)),
                 _hand_case(_rule(0.8)))
    parts = _parts()
    _assert_same(_case(parts, cfl_limit([parts.advection], cfl=0.8)),
                 _hand_case(_rule(0.8)))


def test_the_rule_and_its_strategy_are_one_pair():
    """Review R1 of #782: the core does not tie the strategy of a StepRule to its rule, so the
    word does: pops.run takes master's AdaptiveCFL with the fraction and the cap of the rule."""
    for cfl, cap in ((0.45, 1e-2), (0.8, None)):
        parts = _parts()
        limit = cfl_limit(parts.advection, cfl=cfl, max_dt=cap)
        rule = limit.realize()
        assert isinstance(rule, StepRule) and type(rule.strategy) is AdaptiveCFL
        assert rule.fraction == rule.strategy.cfl == cfl
        assert rule.max_dt == rule.strategy.max_dt == cap
        program = _case(parts, limit).elaborated.case._time
        assert program._step_strategy == AdaptiveCFL(cfl, cap)


def test_a_fixed_step_keeps_its_proposal():
    parts = _parts()
    control = _case(parts, 1e-3).elaborated.case._time._advance_control
    assert control.step_rule is None and control.first_dt == 1e-3


def test_inspect_reports_the_rule_and_the_speeds_it_reads():
    parts = _parts()
    report = str(surface.inspect(_case(parts, cfl_limit(parts.advection, cfl=0.45,
                                                        max_dt=1e-2))))
    assert "dt = cfl_limit(vector(a, b) * u, cfl=0.45, max_dt=0.01) (stated)" in report
    assert "proposal=StepRule(fraction=0.45, max_dt=0.01)" in report
    assert "speeds = the wave speeds dF/du of vector(a, b) * u" in report


# --- the refusals ----------------------------------------------------------------------------------

def test_cfl_limit_takes_a_flux_and_a_courant_number():
    parts = _parts()
    with pytest.raises(TypeError, match="needs cfl="):
        cfl_limit(parts.advection)
    with pytest.raises(TypeError, match="not number="):
        cfl_limit(parts.advection, number=0.45)
    with pytest.raises(ValueError, match="positive finite"):
        cfl_limit(parts.advection, cfl=-0.45)
    with pytest.raises(ValueError, match="positive finite"):
        cfl_limit(parts.advection, cfl=0.45, max_dt=float("inf"))
    with pytest.raises(TypeError, match="transport fluxes of the model"):
        cfl_limit(parts.u, cfl=0.45)
    with pytest.raises(ValueError, match="each flux once"):
        cfl_limit([parts.advection, parts.advection], cfl=0.45)
    with pytest.raises(TypeError, match="takes the flux"):
        cfl_limit(cfl=0.45)


def test_a_rule_that_leaves_out_a_flux_of_the_model_names_its_lot():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    first = model.flux(of=u, value=vector(1.0, 0.0) * u)
    second = model.flux(of=u, value=vector(0.0, 0.25) * u)
    model.balance(ddt(u) == -div(first) - div(second))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=first)
    rule = cfl_limit(first, cfl=0.45); line = inspect.currentframe().f_lineno  # noqa: E702
    with pytest.raises(surface.NotYetElaborated, match="leaves out the flux") as caught:
        _case(parts, rule, schemes={first: rusanov(), second: rusanov()})
    # the refusal points at the rule, not at the Case (review O1 of #788)
    assert (caught.value.site.file, caught.value.site.line) == (__file__, line)


def test_a_flux_that_no_balance_reads_is_refused_by_the_coverage():
    """Review O2 of #788: a flux in no balance is the coverage's refusal (T5), not the rule's."""
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    used = model.flux(of=u, value=vector(1.0, 0.25) * u)
    model.flux(of=u, value=vector(5.0, 0.0) * u)
    model.balance(ddt(u) == -div(used))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=used)
    with pytest.raises(surface.CoverageNotExact, match="is in no balance"):
        _case(parts, cfl_limit(used, cfl=0.45))


def test_a_flux_without_derived_waves_names_its_lot():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(abs(u), 0.25 * u))
    model.balance(ddt(u) == -div(advection))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=advection)
    rule = cfl_limit(advection, cfl=0.45); line = inspect.currentframe().f_lineno  # noqa: E702
    # rusanov() refuses such a flux itself (its spectral radius); upwind states its speed
    with pytest.raises(surface.NotYetElaborated, match="are not derived") as caught:
        _case(parts, rule,
              schemes={advection: upwind(speed=lambda face: dot(vector(1.0, 0.25), face.normal))})
    assert "recorded face speeds of ROM-108" in str(caught.value)
    assert (caught.value.site.file, caught.value.site.line) == (__file__, line)


def test_a_rule_with_a_solve_tolerance_names_its_lot():
    """D6: N8's StepRule carries no solve request, so tolerance= beside a rule is refused."""
    parts = _parts()
    with pytest.raises(surface.NotYetElaborated, match="the solve requests of a step rule"):
        _case(parts, cfl_limit(parts.advection, cfl=0.45),
              tolerance=surface.Tolerance(rtol=1e-8))


def test_a_rule_over_a_flux_of_another_model_is_refused():
    parts, other = _parts(), _parts()
    with pytest.raises(ValueError, match="a flux of another model"):
        _case(parts, cfl_limit(other.advection, cfl=0.45))


def test_a_rule_in_a_model_with_diffusion_names_its_lot():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(1.0, 0.25) * u)
    diffusion = model.diffusive_flux(of=u, value=-0.01 * grad(u))
    model.balance(ddt(u) == -div(advection) - div(diffusion))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=advection)
    with pytest.raises(surface.NotYetElaborated, match="diffusive flux"):
        _case(parts, cfl_limit(advection, cfl=0.45))


def test_a_function_of_cell_is_still_a_later_step_rule():
    """The refusal that the cfl_limit bridge carries (MG9): it names the lot of N8 and the
    bridge to write until then."""
    parts = _parts()
    with pytest.raises(surface.NotYetElaborated, match="a function of cell") as caught:
        surface.Advance(ssprk2(rate=parts.model), dt=lambda cell: 1e-3)
    message = str(caught.value)
    assert "a later lot of N8 (the step rules of 5.11, PF8)" in message
    assert "until then, write dt=pops.lib.time.cfl_limit(flux, cfl=..., max_dt=...)" in message
