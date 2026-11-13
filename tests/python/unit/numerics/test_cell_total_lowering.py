"""G03 lot R-b (ROM-21): the representation of a state on the plan, and what a total is served by.

``plan.representations.add(state, definition)`` states the representation of an evolved state
(contract @2). A total (c = 1/cell.measure) is lowered by decoding ū = U/μ(K) around today's rate
and encoding its rate; an average emits the code of today (MG1). The consumers that this lowering
does not serve are refused at resolution, by name, with the lot that serves them (note §4.3, §14;
driver, 12/11 23:00): a step rule (B1, lot T), a hierarchy (C1, G11), an initial projection, a
Program operation outside the rates that read the decoded state.
"""
from __future__ import annotations

import pops
import pytest
from pops.codegen._cell_total_consumers import CellTotalNotYetElaborated
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program
from pops.numerics import DiscretizationPlan
from pops.representations.definition import (
    RepresentationDefinition, cell_average, cell_total, coefficients, point, point_value)
from pops.lib.quadrature import GaussLegendre
from pops.time import AdaptiveCFL
from pops.time._program.traced_method import NotYetElaborated

from examples.api_v02.models_G03_totals import (
    build_burgers, build_graded_burgers, build_graded_diffusion_total)


def _resolved(build, **keywords):
    case, layout, _, _ = build(**keywords)
    return pops.resolve(pops.validate(case), layout=layout)


def _source(build, **keywords):
    resolved = _resolved(build, **keywords)
    return emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved),
                            target="system")


def _normal(source):
    """The generated C++ without the lines that carry the Program's hash, which names its Case."""
    return [line for line in source.splitlines()
            if "pops_program_hash" not in line and "pops-source-map" not in line]


def _state(model_name="G03_Rb_unit"):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    frame = CartesianDomain("g03_rb_unit", (0.0,), (1.0,)).frame(Cartesian1D())
    model = pops.Model(model_name, frame=frame)
    return model.state("u", components=("u",))


def test_a_representation_row_takes_a_state_and_a_definition_of_contract_2():
    plan = DiscretizationPlan()
    U = _state()
    with pytest.raises(TypeError, match="exact state declaration"):
        plan.representations.add("u", cell_total(1))
    with pytest.raises(TypeError, match="RepresentationDefinition of contract @2"):
        plan.representations.add(U, "total")
    plan.representations.add(U, cell_total(1))
    with pytest.raises(ValueError, match="already has a numerical method"):
        plan.representations.add(U, cell_average(1))


def test_a_decode_outside_one_constant_mode_names_n16_lot_2():
    """A modal decode of two coefficients is the Gram path of N16 lot 2."""
    modal = RepresentationDefinition(
        decode=coefficients[0] + coefficients[1]*point.reference[0], rank=1, size=2,
        kind="coefficient")
    with pytest.raises(NotYetElaborated, match="N16 lot 2"):
        DiscretizationPlan().representations.add(_state(), modal)


@pytest.mark.parametrize("definition", [point_value(1), cell_average(1, GaussLegendre(3))],
                         ids=["sample", "stated-rule"])
def test_a_sample_or_a_stated_rule_waits_for_the_projection_by_encode(definition):
    """R1 of the review of #803: a sample and an average under a stated rule have the average's
    constant mode, but b6 would project their initial value as a cell average (N16 5.5, A6). Only
    cell_average(rank) and cell_total(rank) are accepted; the others name the N16 lot of line 4."""
    with pytest.raises(NotYetElaborated, match="transitional table of CellAverage"):
        DiscretizationPlan().representations.add(_state(), definition)
    plan = DiscretizationPlan()
    plan.representations.add(_state("G03_Rb_unit_average"), cell_average(1))
    plan.representations.add(_state("G03_Rb_unit_total"), cell_total(1))


def _with_consumers(*, integral, field):
    """Periodic Burgers as a total, with an Integral under a ConservationCheck or a field output
    (the reviewer's probe probe_rb_consumers.py)."""
    import math

    import numpy as np
    from pops.diagnostics import ConservationCheck, Integral
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.output import ConsumerGraph, ParaView, ScientificOutput
    from pops.output._consumer_contracts import ParallelMode
    from pops.time import every

    import examples.api_v02.models_G03_totals as m

    frame = CartesianDomain("g03_rb_consumers", (0.0,), (1.0,)).frame(Cartesian1D())
    model, U, flux, rate = m._burgers_model("G03_Rb_consumers", frame)
    case = pops.Case("G03_Rb_consumers_%s_%s" % (integral, field))
    block = case.block("line", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, m._finite_volume(flux, U))
    m._represent(plan, U, "total")
    case.numerics(plan, block=block)
    program = m._ssprk2(case, block, U, rate, m.BURGERS_DT, "G03_Rb_consumers_SSPRK2")
    layout = Uniform(CartesianGrid(frame=frame, cells=(m.BURGERS_CELLS,),
                                   periodic=PeriodicAxes((frame.axes[0],))))
    averages = (1.0 + 0.5*np.sin(2*math.pi*m.burgers_centres()))[None]
    m._bound(case, block[U], averages, np.full(m.BURGERS_CELLS, 1/m.BURGERS_CELLS), "total")
    schedule = every(1, clock=program.clock)
    diagnostics = ((ConservationCheck(Integral(block=block, cadence=schedule), tolerance=1e-10),)
                   if integral else ())
    case.consumers(ConsumerGraph.from_consumers((
        ScientificOutput(format=ParaView(mode=ParallelMode.SERIAL), schedule=schedule,
                         fields=(block[U],) if field else (), diagnostics=diagnostics,
                         target="g03/rb"),)))
    return pops.resolve(pops.validate(case), layout=layout)


def test_a_diagnostic_of_a_total_is_refused():
    """R2 of the review of #803: an Integral weighs the stored totals by the cell measure, h·ΣU on
    periodic Burgers instead of ΣU; on graded cells it would flag a conserving run."""
    with pytest.raises(CellTotalNotYetElaborated, match="inventory") as refusal:
        _with_consumers(integral=True, field=False)
    assert refusal.value.reason == "cell_total_diagnostic"


def test_a_field_output_of_a_total_is_refused():
    with pytest.raises(CellTotalNotYetElaborated, match="under the name of the state") as refusal:
        _with_consumers(integral=False, field=True)
    assert refusal.value.reason == "cell_total_output"


def test_the_resolved_plan_carries_the_row_and_keeps_its_identity_without_one():
    """MG1: a plan without a representation row has the payload of before R-b."""
    total = _resolved(build_burgers, represent="total").blocks[0].numerics
    average = _resolved(build_burgers, represent="average").blocks[0].numerics
    bare = _resolved(build_burgers, represent=None).blocks[0].numerics
    assert "representations" not in bare.to_data()
    assert len(total.cell_total_states()) == 1 and total.representation_of(
        total.cell_total_states()[0]) == cell_total(1)
    assert average.cell_total_states() == () and len(average.representations) == 1
    assert total.identity != average.identity != bare.identity


def test_a_rank_that_differs_from_the_frame_is_refused():
    case, layout, _, _ = build_burgers(represent="total")
    plan = case._numerics_assignments["line"]
    model = case._block_registry.spec("line")["model"]
    wrong = DiscretizationPlan()
    for rate, method in plan.rates.items():
        wrong.rates.add(rate, method)
    wrong.representations.add(next(iter(model.states.values())), cell_total(2))
    with pytest.raises(ValueError, match="rank 2"):
        wrong.validate_for(model)


def test_a_total_emits_the_decode_and_the_encode_and_an_average_emits_todays_code():
    total = _source(build_burgers, represent="total")
    average = _source(build_burgers, represent="average")
    bare = _source(build_burgers, represent=None)
    assert total.count("ctx.decode_cell_total(") == 2      # two rates of SSPRK2
    assert total.count("ctx.encode_cell_total_rate(") == 2
    assert "cell_total" not in average and _normal(average) == _normal(bare)
    diffusion = _source(build_graded_diffusion_total)
    assert diffusion.count("ctx.decode_cell_total(") == 2
    assert diffusion.count("ctx.encode_cell_total_rate(") == 2


def test_a_step_rule_on_a_total_is_refused_naming_lot_t():
    """B1: the native step selection reads the stored totals U, not ū = U/μ(K); Burgers' speed
    would be off by μ(K). A total takes a fixed step until lot T (driver, 23:00). The native test
    runs the same case at a fixed step against the average."""
    with pytest.raises(CellTotalNotYetElaborated, match="G03 lot T") as caught:
        _resolved(build_burgers, represent="total", step=AdaptiveCFL(0.4, max_dt=0.01))
    assert caught.value.reason == "cell_total_step_rule"
    assert isinstance(caught.value, NotYetElaborated)


def test_the_same_step_rule_on_an_average_resolves():
    _resolved(build_burgers, represent="average", step=AdaptiveCFL(0.4, max_dt=0.01))


def test_an_analytic_initial_projection_of_a_total_is_refused():
    with pytest.raises(CellTotalNotYetElaborated) as caught:
        _resolved(build_burgers, represent="total", initial="analytic")
    assert caught.value.reason == "cell_total_initial_projection"
    _resolved(build_burgers, represent="average", initial="analytic")


def test_the_graded_cases_resolve_under_a_total():
    for build in (build_graded_burgers, build_graded_diffusion_total):
        assert _resolved(build).blocks[0].numerics.cell_total_states()
