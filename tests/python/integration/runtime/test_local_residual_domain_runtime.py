"""Original residual domains through real local solvers, mesh state and rollback."""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, minimum, sqrt, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import (
    _assert_rejected, _compile, _root_check, _snapshot,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, .125


def _case(kind, captured):
    frame = Rectangle("residual_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("original_local_equation", frame=frame)
    state = model.state("density", components=("amount",))
    u = state[0]
    gate_handle = model.param(RuntimeParam("gate", default=2.)) if kind in ("lazy", "apply") else None
    gate = model.value(gate_handle) if gate_handle is not None else None
    dangerous = model.primitive("constitutive_root", sqrt(-u)) if kind == "primitive" else sqrt(-u)
    physical = minimum(dangerous, 0 * u)
    operator = None
    if kind == "lazy":
        physical = where(u > gate, lambda: physical, lambda: 0 * u)
    elif kind == "apply":
        coefficient = minimum(sqrt(-gate), 0.)
        physical = coefficient * u
        law = model.local_linear_operator("linear_constitutive", on=state,
                                           matrix=((coefficient,),))
        operator = model.operator("constitutive_apply", returns=law)
    source = model.source("constitutive_source", on=state, value=(physical,))
    rate = model.rate("physical_balance", equation=ddt(state) == source)
    case = pops.Case("local_residual_domain")
    block = case.block("material", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, StateStorage())
    case.numerics(plan, block=block)
    program = pops.Program("implicit_original_residual")
    q = program.state(block[state])
    seed = program.value("different_seed", .75 * q.n, at=q.next.point)
    program.store_history("trial", seed, depth=2)

    def residual(p, unknown, old):
        argument = old if captured else unknown
        value = (p.apply(operator, argument) if operator is not None else
                 p.source(model.module.operator_handle("constitutive_source"), argument))
        return p.value("original_equation", unknown - old - p.dt * value, at=unknown.point)

    candidate = program.solve(LocalResidual(residual, seed, captures={"old": q.n}),
                              solver=LocalNewton(tolerance=1.e-12, max_iterations=20)
                              ).consume(action=FailRun())
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes))
    return case, Uniform(grid), (case.resolve(block[gate_handle]) if gate_handle is not None else None)


@pytest.mark.parametrize("kind,captured", (
    ("source", False), ("source", True), ("primitive", False),
    ("apply", False), ("apply", True), ("lazy", False),
))
def test_local_residual_observes_invalid_intermediates_and_restores_real_history(
    isolated_native_cache, native_cxx, kokkos_root, kind, captured,
):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, gate = _case(kind, captured)
    artifact, world = _compile(case, layout, "local-residual-original-domain")
    initial = np.ones((1, N, N))
    settings = ((2., False),) if kind not in ("lazy", "apply") else (
        ((2., True), (.5, False)) if kind == "lazy" else ((-4., True), (2., False)))
    for value, accepted in settings:
        runtime = pops.bind(artifact, initial_state={"material": initial},
                            params={gate: value} if gate is not None else {},
                            resources={"execution_context": artifact_execution_context(artifact)})
        before = _snapshot(runtime, "material", 1, "uniform", world)
        old_history_count = runtime._executor.history_fill_count("trial")
        if accepted:
            assert pops.run(runtime, t_end=DT, max_steps=1, console=False).accepted_steps == 1
            assert runtime._executor.history_fill_count("trial") == old_history_count + 1
        else:
            _assert_rejected(runtime, world, DT)
            assert runtime._executor.history_fill_count("trial") == old_history_count
        after = _snapshot(runtime, "material", 1, "uniform", world)

        def check():
            np.testing.assert_array_equal(before, initial)
            if accepted:
                # The active physical source is exactly zero in these controls.
                np.testing.assert_allclose(after, initial, rtol=0., atol=2.e-12)
            else:
                np.testing.assert_array_equal(after, initial)

        _root_check(world, check)
