"""Independent public SSP/path consumer proofs; source-only, no native mesh run."""
from fractions import Fraction
import re

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_partition_stability import (
    explicit_update_consumers, partition_stability_groups,
)
from pops.layouts import Uniform
from pops.lib.time import SSPRK2, SSPRK3, RungeKutta, ButcherTableau
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (
    DiscretizationPlan, PathConservativeFiniteVolume, reconstruction, riemann, variables,
)
from pops.time import FailRun, FixedDt
from tests.python.support.symbolic_path_case import declarations


def _case(method, *, guarded=False):
    model, state, flux, product, path = declarations()
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    plan = DiscretizationPlan()
    plan.rates.add(rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("public_ssp_path")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    if guarded:
        # Same acceptance topology as M17, on a small independent physical model.
        program = pops.Program("guarded_ssprk2")
        q = program.state(block[state])

        def guard(value, name):
            return program.guard(name, value, program.min(value) > 0., action=FailRun())

        current = guard(q.n, "current_domain")
        k0 = program.value("rhs_0", rate(current), at=program.stage("first", c=0))
        u1 = program.value("predictor", q.n + program.dt*k0,
                           at=program.stage("second", c=1))
        k1 = program.value("rhs_1", rate(guard(u1, "predictor_domain")), at=u1.point)
        endpoint = program.value("endpoint", q.n + Fraction(1, 2)*program.dt*(k0+k1),
                                 at=q.next.point)
        program.commit(q.next, guard(endpoint, "endpoint_domain"))
    else:
        program = method(block[state], rate=rate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=path.frame, cells=(8, 8),
                                   periodic=PeriodicAxes(path.frame.axes)))
    return case, layout


def _lower(method, *, guarded=False, target="system"):
    case, layout = _case(method, guarded=guarded)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    selected = resolved.blocks[0]
    emitter, _ = lower_and_validate(selected.model, state_space=selected.state_spaces[0],
        resolved_operations=selected.resolved_operations, numerics=selected.numerics)
    source = emit_cpp_program(resolved.time, model=emitter, target=target)
    return resolved.time, source


@pytest.mark.parametrize("method,expected", [
    (SSPRK2, ((Fraction(1), Fraction(1)), (Fraction(1, 2), Fraction(1, 2)))),
    (SSPRK3, ((Fraction(1), Fraction(1)), (Fraction(1, 4), Fraction(1, 4)),
              (Fraction(2, 3), Fraction(2, 3)))),
])
@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_public_butcher_ssp_has_one_budget_per_actual_stage(method, expected, target):
    program, source = _lower(method, target=target)
    consumers = explicit_update_consumers(program)
    certificates = []
    for value in program._values:
        if value.id in consumers:
            for _, alpha, rates in partition_stability_groups(value, include_transport=True):
                assert len(rates) == 1, "earlier stage rates must not be spent twice"
                certificates.append((alpha, rates[0][1]))
    assert tuple(certificates) == expected
    definitions = re.findall(r"const pops::Real user_face_update_frequency_\d+_\d+ = ([^;]+);", source)
    assert len(definitions) == len(expected)
    assert all(len(re.findall(r"path_frequency_\d+", line)) == 1 for line in definitions)
    assert source.count('"user_face_numerical_stability"') == len(expected)


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_guarded_m17_topology_keeps_domain_guards_and_both_ssp_budgets(target):
    program, source = _lower(SSPRK2, guarded=True, target=target)
    consumers = explicit_update_consumers(program)
    groups = [group for value in program._values if value.id in consumers
              for group in partition_stability_groups(value, include_transport=True)]
    assert [(alpha, tuple(beta for _, beta in rates)) for _, alpha, rates in groups] == [
        (Fraction(1), (Fraction(1),)),
        (Fraction(1, 2), (Fraction(1, 2),)),
    ]
    assert source.count('"user_face_numerical_stability"') == 2
    for name in ("current_domain", "predictor_domain", "endpoint_domain"):
        assert name in source, "proof transparency must preserve the physical guard"


def test_explicit_midpoint_is_not_misrepresented_as_ssp():
    # The final state has no positive budget for its midpoint evaluation. It is
    # second order but has no positive SSP coefficient for a general nonlinear RHS.
    def midpoint(state, *, rate):
        return RungeKutta(state, rate=rate, tableau=ButcherTableau(
            A=[[], [Fraction(1, 2)]], b=[0, 1], c=[0, Fraction(1, 2)], name="midpoint"))

    with pytest.raises(ValueError, match="stability|convex|Shu"):
        _lower(midpoint)


def test_accepted_rate_without_state_budget_cannot_skip_stability_proof():
    def rate_only(state, *, rate):
        program = pops.Program("rate_without_state_budget")
        q = program.state(state)
        rhs = program.value("rhs", rate(q.n), at=program.stage("sample", c=0))
        endpoint = program.value("endpoint", program.dt*rhs, at=q.next.point)
        program.commit(q.next, endpoint)
        return program

    with pytest.raises(ValueError, match="stability|convex|Shu"):
        _lower(rate_only)
