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
def test_a_sample_or_a_stated_rule_is_served_bound_as_data(definition):
    """R1 of the review of #803, as N16 lifts the table for three words: a sample and an average
    under a stated rule have the average's constant mode and rate, so the row serves them; their
    initial value is bound as data, the analytic route refused with N16 lot 2
    (test_n16_three_words_rows.py)."""
    plan = DiscretizationPlan()
    plan.representations.add(_state(), definition)
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


# --- N15 FaceFlux/CellRate lot FC-1 (ROM-98, note §3.4): the named source of a total ----------

def _decaying_burgers(shape="split"):
    """Burgers with the named source -2u on 16 periodic cells, under a total, on the split route:
    the flux by a finite volume, the source selected and realized by ``StateStorage()``.
    ``shape`` names how the Program evaluates the source: ``split`` beside the rate of the flux,
    ``absorbed`` inside a pointwise rate, ``solved`` by the local solve of the source."""
    from pops import math as pm
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.initial import InitialCondition
    from pops.layouts import Uniform
    from pops.lib.initial import BindArray
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import StateStorage, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt

    frame = CartesianDomain("n15_fc1_source", (0.0,), (3.0,)).frame(Cartesian1D())
    (x,) = frame.axes
    model = pops.Model("N15_FC1_decaying_burgers", frame=frame)
    U = model.state("u", components=("u",))
    (u,) = U
    flux = model.flux("burgers_flux", state=U, frame=frame, components={x: (0.5 * u * u,)},
                      waves={x: (u,)})
    decay = model.source("decay", on=U, value=(-2.0 * u,))
    rate = model.rate("burgers_decay", equation=pm.ddt(U) == -pm.div(flux) + decay)
    transport, source = rate.select(flux), rate.select(decay)
    case = pops.Case("N15_FC1_%s" % shape)
    block = case.block("line", model)
    plan = DiscretizationPlan()
    plan.rates.add(transport, FiniteVolume(flux=flux, variables=variables.Conservative(U),
                                           reconstruction=reconstruction.FirstOrder(),
                                           riemann=riemann.Rusanov()))
    plan.rates.add(source, StateStorage())
    plan.representations.add(U, cell_total(1))
    case.numerics(plan, block=block)
    program = pops.Program("N15_FC1_%s_program" % shape)
    q = program.state(block[U])
    if shape == "solved":
        relaxed = program.solve_implicit_source(q.n)
        update = program.value("next", relaxed + program.dt * transport(relaxed), at=q.next.point)
    else:
        local = source(q.n)
        if shape == "absorbed":
            local = program.rate("absorbed", (2 * local[0],), of=q.n)
        update = program.value("next", q.n + program.dt * (transport(q.n) + local),
                               at=q.next.point)
    program.commit(q.next, update)
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    case.initials.add(InitialCondition(state=block[U], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(16,), periodic=PeriodicAxes((x,))))
    return pops.resolve(pops.validate(case), layout=layout)


def test_the_named_source_of_a_total_reads_the_density_and_encodes_its_rate():
    """The split ``source`` operation reads ū = U/μ(K) from the decode and its rate is multiplied
    by μ(K) after, as the rate of the flux (route line 6, extended by FC-1)."""
    resolved = _decaying_burgers()
    source = emit_cpp_program(resolved.time, model_graph=build_program_model_graph(resolved),
                              target="system")
    lines = [line.strip() for line in source.splitlines()]
    assert source.count("ctx.decode_cell_total(") == 2  # the flux and the source
    assert source.count("ctx.encode_cell_total_rate(") == 2
    scope = next(number for number, line in enumerate(lines)
                 if 'ProfileScope "node:decay' in line)
    (decode,) = [number for number in range(scope - 2, scope)
                 if lines[number].startswith("ctx.decode_cell_total(")]
    density = lines[decode].split(",")[1].split(")")[0].strip()
    kernel = next(number for number in range(scope, len(lines)) if "rhs_scratch(" in lines[number])
    assert lines[kernel].endswith(", %s);" % density)
    output = lines[kernel].split("&")[1].split("=")[0].strip()
    encode = lines.index("ctx.encode_cell_total_rate(%s);  // G03 R-b: the rate of the total"
                         % output)
    assert encode > kernel


@pytest.mark.parametrize("shape, operation", [("absorbed", "pointwise_expression"),
                                              ("solved", "solve_implicit_source")])
def test_a_source_evaluated_inside_another_operation_stays_refused_on_a_total(shape, operation):
    """O1: the gate is an allowlist over every node of the Program, so a source that a pointwise
    rate absorbs, or that a local solve holds in its residual, is refused by its outer operation,
    and only the standalone ``source`` is served."""
    with pytest.raises(CellTotalNotYetElaborated) as caught:
        _decaying_burgers(shape)
    assert caught.value.reason == "cell_total_operation"
    assert operation in str(caught.value)
