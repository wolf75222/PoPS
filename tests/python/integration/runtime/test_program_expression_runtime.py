"""Public native pipeline for reusable nonlinear expressions and frozen captures."""
import math
import numpy as np
import pytest
import pops

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.solvers.nonlinear import LocalNewton
from pops.time import FixedDt, LocalResidual, FailRun
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def accumulated(q):
    return q[0] + q[0] * q[0]


def make_case(*, implicit=False, seed=0.25, invalid=None):
    frame = Rectangle("expression_domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("expression_physics", frame=frame)
    state = model.state("U", components=("q",), representation=Conservative(),
                        space=CellState(frame=frame))
    flux = model.flux("zero", frame=frame, state=state,
                      components={axis: (0 * state[0],) for axis in frame.axes},
                      waves={axis: (0 * state[0],) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    # The physical expression and every temporal instantiation use this same body.
    assert accumulated(state) is not None
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                        reconstruction=reconstruction.FirstOrder(),
                                        riemann=riemann.Rusanov()))
    case = pops.Case("expression_case")
    block = case.block("fluid", model=model)
    case.numerics(numerics, block=block)
    program = pops.Program("expression_step")
    q = program.state(block[state])
    if invalid is not None:
        from pops.math import minimum
        if invalid == "division":
            zero = q.n[0] - q.n[0]
            expression = minimum(zero / zero, q.n[0])
        else:
            expression = minimum(q.n[0] * 1.e308, q.n[0])
        candidate = program.value("invalid", (expression,), at=q.next.point)
    elif implicit:
        guess = program.value("seed", (0 * q.n[0] + seed,), at=q.next.point)

        def residual(p, unknown, old):
            return (accumulated(unknown) - accumulated(old) - 1,)

        candidate = program.solve(LocalResidual(residual, guess, captures={"old": q.n}),
                                  solver=LocalNewton(tolerance=1.e-12, max_iterations=30)
                                  ).consume(action=FailRun())
    else:
        candidate = program.value("nonlinear", (accumulated(q.n),), at=q.next.point)
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout


@pytest.mark.parametrize("implicit,seed", [(False, 0.), (True, 0.25), (True, 2.)])
def test_public_pipeline_executes_expression_and_implicit_capture(
        isolated_native_cache, native_cxx, kokkos_root, implicit, seed):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = make_case(implicit=implicit, seed=seed)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial_value = 0. if implicit else 2.
    simulation = pops.bind(artifact, initial_state={"fluid": np.full((1, 4, 4), initial_value)},
                           resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(simulation, t_end=0.125, max_steps=1)
    assert report.accepted_steps == 1
    expected = (math.sqrt(5.) - 1.) / 2. if implicit else 6.
    actual = np.asarray(simulation.state_global("fluid"))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2.e-12)
    if implicit:
        np.testing.assert_allclose(actual + actual * actual, 1., rtol=0., atol=2.e-12)


@pytest.mark.parametrize("invalid", ["division", "overflow"])
def test_masked_invalid_intermediate_rejects_and_restores_state(
        isolated_native_cache, native_cxx, kokkos_root, invalid):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = make_case(invalid=invalid)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.full((1, 4, 4), 2.)
    simulation = pops.bind(artifact, initial_state={"fluid": initial},
                           resources={"execution_context": artifact_execution_context(artifact)})
    with pytest.raises(RuntimeError, match="pointwise_expression|non-finite"):
        pops.run(simulation, t_end=0.125, max_steps=1)
    np.testing.assert_array_equal(np.asarray(simulation.state_global("fluid")).reshape(initial.shape),
                                  initial)
