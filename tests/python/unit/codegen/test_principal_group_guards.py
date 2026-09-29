"""A pre-step stability query must not speculate on an inactive spatial body."""
import pops
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.numerics.variables import Conservative
from pops.time import AdaptiveCFL, FixedDt
from tests.python.unit.numerics.test_principal_grouping_contract import _principal_model


def _conditional_group(strategy):
    model, states, fluxes = _principal_model(2)
    rates = tuple(model.rate("R%d" % i, equation=ddt(state) == -div(flux))
                  for i, (state, flux) in enumerate(zip(states, fluxes)))
    case = pops.Case("conditional_principal")
    blocks = tuple(case.block("b%d" % i, model, states=(state,))
                   for i, state in enumerate(states))
    for block, state, flux, rate in zip(blocks, states, fluxes, rates):
        plan = DiscretizationPlan()
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=Conservative(state),
            reconstruction=FirstOrder(), riemann=Rusanov(),
            sampling=tuple(other for other in states if other != state)))
        case.numerics(plan, block=block)
    program = pops.Program("conditional_principal")
    values = tuple(program.state(block[state]) for block, state in zip(blocks, states))
    bindings = dict(zip(states, (value.n for value in values)))
    condition = program.norm_inf(values[0].n) < 0.

    def active(body):
        rhs = rates[0](values[0].n, bindings=bindings)
        return body.value("active", values[0].n + body.dt * rhs,
                          at=values[0].next.point)

    def inactive(body):
        return body.value("inactive", values[0].n, at=values[0].next.point)

    selected = program.branch(condition, active, inactive)
    program.commit(values[0].next, selected)
    program.commit(values[1].next,
        program.value("unchanged", values[1].n, at=values[1].next.point))
    program.step_strategy(strategy)
    case.program(program)
    layout = Uniform(CartesianGrid(frame=model.frame, cells=(8, 8),
                                   periodic=PeriodicAxes(model.frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return emit_cpp_program(resolved.time,
                            model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))


def test_prescribed_step_does_not_evaluate_an_inactive_group_to_find_a_bound():
    source = _conditional_group(FixedDt(.001))
    bound = source[source.index("pops_program_dt_bound("):]
    bound = bound[:bound.index("\n}")]
    assert "_evaluate(ctx," not in bound
    assert "_resource.publish(" in source
    assert "dt*principal_" in source


def test_conditional_adaptive_group_requires_its_authored_bound():
    with pytest.raises(NotImplementedError, match="authored Program.dt_bound"):
        _conditional_group(AdaptiveCFL(.3))
