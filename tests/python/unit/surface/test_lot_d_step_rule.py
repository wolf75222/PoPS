"""Lot D-E of N15 (ROM-98, OD31), part D, as N8 « dt= as a function of cell » writes it:
``dt=cfl_limit(flux, cfl=, max_dt=)``.

``pops.Advance(method, dt=cfl_limit(advection, cfl=0.45, max_dt=1e-2))`` is a function of ``cell``
(design note ``n8/cell_step_rule``, D1): ``where(s > 0, (cfl * w) / s, M)``, capped by ``max_dt``,
with the Courant budget ``cfl`` that it declares (``step_fraction``). It elaborates onto the cells
rule of N8, ``elaborate(..., proposal=StepBound(E, 2, 0.45, "stated"))``: each step is the
smallest value of E over the cells at the accepted state, and the run lands on its frontiers. Each
case gives the case written by hand for b6 and N8 (SU-T3: program IR, artifact without provenance,
semantic identity, C++). The bridge of #788 (``StepRuleValue`` realizing b6's ``AdaptiveCFL``)
left MG9 with this lot. The run is checked natively by
``tests/python/integration/runtime/test_cell_step_rule_n8_native.py``.
"""
import json
import math
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
from pops.lib.numerics import rusanov, two_point, upwind
from pops.lib.time import cfl_limit
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan
from pops.params import ConstParam
from pops.projection import ConservativeCellAverage
from pops.surface.math import ddt, div, dot, grad, vector
from pops._ir.control_expr import Where
from pops._ir.expr import Abs, Compare, Const, Div, Maximum, Minimum, Mul
from pops.mesh._amr.refinement_criterion import CellWidth
from pops.time._program.elaboration import Advance, elaborate
from pops.time._step.bound import StepBound
from pops.time._step.strategy import CellStepBound
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


def _bound(fraction, max_dt=None):
    """cfl_limit's bound written by hand in b6's expressions, as a function of the values of the
    constants a and b of the model: (cfl * w) / s, or where(s > 0, (cfl * w) / s, max_dt) capped
    by max_dt, with s = max(|a|, |b|) and w = min(h_x, h_y)."""
    def rule(a, b):
        speed = Maximum(Abs(a), Abs(b))
        width = Minimum(CellWidth(0), CellWidth(1))
        quotient = Div(Mul(Const(fraction), width), speed)
        if max_dt is None:
            return StepBound(quotient, 2, fraction, "stated")
        bound = Where(Compare("gt", speed, Const(0)), quotient, Const(max_dt))
        return StepBound(Minimum(bound, Const(max_dt)), 2, fraction, "stated")
    return rule


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
              state=block[u], proposal=rule(a, b))
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

def test_cfl_limit_is_the_cells_rule_of_n8():
    parts = _parts()
    case = _case(parts, cfl_limit(parts.advection, cfl=0.45, max_dt=1e-2))
    _assert_same(case, _hand_case(_bound(0.45, 1e-2)))
    program = case.elaborated.case._time
    control = program._advance_control
    assert control.step_rule is None and control.first_dt is None
    assert control.step_bound == (0.45, program._advance_step_bound.qualified_id)
    assert control.to_data()["schema"] == "pops.advance-control@6"
    assert program._serialize(include_provenance=False)["version"] == 36


def test_cfl_limit_without_a_cap_and_with_a_list_of_fluxes():
    parts = _parts()
    _assert_same(_case(parts, cfl_limit(parts.advection, cfl=0.8)), _hand_case(_bound(0.8)))
    parts = _parts()
    _assert_same(_case(parts, cfl_limit([parts.advection], cfl=0.8)), _hand_case(_bound(0.8)))


def test_pops_run_takes_the_bound_at_the_declared_budget():
    """D4: pops.run on the artifact takes CellStepBound with the rule's fraction and the bound's
    identity; the core names neither cfl_limit nor its options (CF2)."""
    for cfl, cap in ((0.45, 1e-2), (0.8, None)):
        parts = _parts()
        limit = cfl_limit(parts.advection, cfl=cfl, max_dt=cap)
        assert limit.step_fraction == cfl
        program = _case(parts, limit).elaborated.case._time
        assert program._step_strategy == CellStepBound(
            cfl, program._advance_step_bound.qualified_id)
        assert program._step_strategy.face_frequency_rule == "largest_axis"


def test_a_fixed_step_keeps_its_proposal():
    parts = _parts()
    control = _case(parts, 1e-3).elaborated.case._time._advance_control
    assert control.step_rule is None and control.step_bound is None and control.first_dt == 1e-3


def test_inspect_reports_the_bound_and_its_budget():
    parts = _parts()
    report = str(surface.inspect(_case(parts, cfl_limit(parts.advection, cfl=0.45,
                                                        max_dt=1e-2))))
    assert "dt = cfl_limit, the smallest over the cells of " in report
    assert "fraction = 0.45, the budget of the face guards of each attempt (C2), " \
           "declared by the rule (step_fraction) (stated)" in report
    assert "proposal=StepBound(" in report


def test_the_bound_is_emitted_in_the_order_it_is_written():
    """P3: (cfl * w) / s in double, without a reciprocal: b6's cfl * h_min / s."""
    parts = _parts()
    case = _case(parts, cfl_limit(parts.advection, cfl=0.45, max_dt=1e-2))
    resolved = _resolved(case.elaborated.case, case.elaborated.layout)
    cpp = emit_cpp_program(resolved.time,
                           model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "pops_program_advance_step_bound(" in cpp
    assert "guard1_cse3_ = (0.45 * guard1_cse2_); guard1_cse4_ = (guard1_cse3_ / cse2_);" in cpp
    assert "1.0 /" not in cpp.split("pops_program_advance_step_bound_kernel", 1)[1]


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


def test_a_courant_number_above_one_is_refused_by_name():
    """P1 of the rev 3 review: master's AdaptiveCFL and #788's bridge accepted cfl > 1."""
    parts = _parts()
    with pytest.raises(ValueError, match=r"cfl_limit\(vector\(a, b\) \* u, cfl=1.5\): the "
                                         r"Courant number of a step rule is in \(0, 1\]"):
        cfl_limit(parts.advection, cfl=1.5)
    assert cfl_limit(parts.advection, cfl=1).step_fraction == 1.0


def test_a_rule_that_leaves_out_a_flux_of_the_model_names_its_lot():
    """The refusal is cfl_limit's own (P2): a model of two transport fluxes of one unknown is not
    elaborated by the schemes of this lot, so the rule is run on the cell of its support."""
    from pops.surface._step_cell import StepCell

    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    first = model.flux(of=u, value=vector(1.0, 0.0) * u)
    second = model.flux(of=u, value=vector(0.0, 0.25) * u)
    model.balance(ddt(u) == -div(first) - div(second))
    cell = StepCell(domain, (u,))
    with pytest.raises(ValueError, match="leaves out the transport flux") as caught:
        cfl_limit(first, cfl=0.45)(cell)
    message = str(caught.value)
    assert message.startswith("cfl_limit(vector(1.0, 0.0) * u, cfl=0.45) leaves out")
    assert "step rules of 5.11 per flux are a later lot of N8" in message
    # the list form names every flux, and bounds by the largest of their speeds
    assert "largest(" in str(cfl_limit([first, second], cfl=0.45)(cell))
    # the schemes of this lot refuse the model itself, before any step rule
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=first)
    with pytest.raises(ValueError, match="an additional physical flux"):
        _case(parts, 0.01, schemes={first: rusanov(), second: rusanov()})


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


def test_a_flux_without_derived_waves_is_refused_by_its_speed():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(abs(u), 0.25 * u))
    model.balance(ddt(u) == -div(advection))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=advection)
    # rusanov() refuses such a flux itself (its spectral radius); upwind states its speed
    with pytest.raises(surface.AmbiguousElaboration, match="is not derived in this lot"):
        _case(parts, cfl_limit(advection, cfl=0.45),
              schemes={advection: upwind(speed=lambda face: dot(vector(1.0, 0.25), face.normal))})


def test_a_rule_with_a_solve_tolerance_names_its_lot():
    """D5: a step bound carries no solve request, so tolerance= beside it is refused."""
    parts = _parts()
    with pytest.raises(surface.NotYetElaborated, match="the solve requests of a step rule"):
        _case(parts, cfl_limit(parts.advection, cfl=0.45),
              tolerance=surface.Tolerance(rtol=1e-8))


def test_a_rule_over_a_flux_of_another_model_is_refused():
    parts, other = _parts(), _parts()
    with pytest.raises(TypeError, match=r"cell.speed\(vector\(a, b\) \* u, x\): a flux of u, "
                                        "which is not an unknown that the advance steps"):
        _case(parts, cfl_limit(other.advection, cfl=0.45))


def test_a_rule_in_a_model_with_diffusion_names_its_lot():
    domain = surface.Box(x=surface.Periodic(0, 1), y=surface.Periodic(0, 1))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=vector(1.0, 0.25) * u)
    diffusion = model.diffusive_flux(of=u, value=-0.01 * grad(u))
    model.balance(ddt(u) == -div(advection) - div(diffusion))
    parts = types.SimpleNamespace(domain=domain, model=model, u=u, advection=advection)
    schemes = {advection: rusanov(), diffusion: two_point()}
    _case(parts, 0.01, schemes=schemes)  # the case itself elaborates under a fixed step
    with pytest.raises(ValueError, match="diffusive flux .* whose limit cfl_limit does not write"):
        _case(parts, cfl_limit(advection, cfl=0.45), schemes=schemes)
