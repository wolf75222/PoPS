"""Public source-balance occurrences: authoring identity and one native Euler step."""

import numpy as np
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import ForwardEuler
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.time import ExternalTimeGrid
from tests.python.support.native_execution_context import artifact_execution_context


CELLS = 8
DT = 0.01


def _case(expression):
    frame = Rectangle("balance_occurrence_box", lower=(0., 0.),
                      upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("balance_occurrence_model", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    source = model.source("drive", on=state, value=(2. * q,))
    rhs = source + source if expression == "double" else 3 * source - 2 * source
    rate = model.rate("balance", equation=ddt(state) == rhs)
    case = pops.Case("balance_occurrence_" + expression)
    block = case.block("material", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(ExternalTimeGrid("time_grid"))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(CELLS, CELLS),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, model.balance_contract(rate), program


@pytest.mark.parametrize("expression,coefficients", [
    ("double", (1, 1)), ("signed", (3, -2)),
])
def test_repeated_source_retains_distinct_signed_occurrences(expression, coefficients):
    case, layout, view, program = _case(expression)
    occurrences = view.occurrences
    assert tuple(row.ordinal for row in occurrences) == (0, 1)
    assert occurrences[0].payload is occurrences[1].payload
    assert tuple(row.coefficient for row in occurrences) == coefficients
    source_calls = [value for value in program._values if value.op == "source"]
    balances = [value for value in program._values
                if value.op == "linear_combine" and "physical_balance" in value.attrs]
    assert len(source_calls) == 2
    assert len(balances) == 1 and len(balances[0].inputs) == 2
    assert tuple(row.ordinal for row in balances[0].attrs["physical_balance"].occurrences) == (0, 1)
    pops.resolve(pops.validate(case), layout=layout)


def test_homonymous_source_from_foreign_model_is_refused():
    frame = Rectangle("homonymous_balance_box", lower=(0., 0.),
                      upper=(1., 1.)).frame(Cartesian2D())
    first = pops.Model("first_owner", frame=frame)
    second = pops.Model("second_owner", frame=frame)
    first_state = first.state("U", components=("q",))
    second_state = second.state("U", components=("q",))
    first.source("drive", on=first_state, value=(first_state[0],))
    foreign = second.source("drive", on=second_state, value=(second_state[0],))
    with pytest.raises(ValueError, match="source.*this physics model"):
        first.rate("bad", equation=ddt(first_state) == foreign)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("expression,coefficient_sum", [
    ("double", 2.), ("signed", 1.),
])
def test_native_repeated_source_one_step_matches_occurrence_sum(
        expression, coefficient_sum, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _, _ = _case(expression)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.ones((1, CELLS, CELLS))
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=DT, max_steps=1, time_grid=(0., DT))
    assert report.accepted_steps == 1 and report.rejected_steps == 0
    actual = np.asarray(runtime.state_global("material")).reshape(initial.shape)
    np.testing.assert_allclose(actual, 1. + DT * 2. * coefficient_sum, rtol=0., atol=2.e-14)
