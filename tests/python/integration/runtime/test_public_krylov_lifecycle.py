"""Public Case -> resolve -> compile -> bind -> run witness for prepared Krylov."""

from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.physics import Density
from pops.representations import Conservative
from pops.solvers import GMRES
from pops.spaces import CellState
from pops.time import FailRun, FixedDt, SolveRequest, SolveUnknown
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _public_diagonal_krylov_case(*, general_request=False):
    frame = Rectangle(
        "public_krylov_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("public_krylov_tracer", frame=frame)
    state = model.state(
        "U",
        components=("inventory",),
        representation=Conservative(),
        space=CellState(frame=frame),
        roles={"inventory": Density()},
    )
    (inventory,) = state
    flux = model.flux(
        "zero_transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * inventory,), y_axis: (0.0 * inventory,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    rate = model.rate("inert_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )

    case = pops.Case("public_prepared_krylov")
    block = case.block("tracer", model=model)
    case.numerics(numerics, block=block)

    program = pops.Program("public_diagonal_gmres")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator(
        "double_identity", domain="state", range_="state", ncomp=1
    )
    program.set_apply(operator, lambda _program, _out, value: 2.0 * value)
    problem = LinearProblem(
            operator,
            temporal.n,
            properties=LinearOperatorProperties.symmetric_positive_definite(),
            nullspace=None,
        )
    if general_request:
        # This nonzero seed differs from both the RHS and the exact answer. It must
        # initialize the iteration without changing the frozen equation 2 I x = b.
        seed = program.value("independent_seed", 3.0 * temporal.n)
        problem = SolveRequest(
            problem, unknowns=(SolveUnknown("inventory", temporal.n),),
            equation_inputs={"operator": operator, "rhs": temporal.n},
            seeds={"inventory": seed},
            problem_metadata={"physical_equation": "2 inventory = inventory_n"})
    solution = program.solve(
        problem,
        solver=GMRES(max_iter=4, restart=2, rel_tol=1.0e-13),
        name="diagonal_solution",
    ).consume(action=FailRun())
    if general_request:
        solution = solution["inventory"]
    accepted = program.value("accepted", solution, at=temporal.next.point)
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(0.125))
    case.program(program)

    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(4, 4),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return case, layout


def test_public_case_resolve_bind_run_executes_prepared_gmres(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = _public_diagonal_krylov_case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.arange(16, dtype=np.float64).reshape(1, 4, 4) / 16.0
    runtime = pops.bind(artifact, initial_state={"tracer": initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=0.125, max_steps=1)

    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("tracer"), dtype=np.float64).reshape(1, 4, 4)
    # A copy of the RHS is not a solution of 2 I x = b.  This independently proves that the
    # built-in prepared GMRES provider executed rather than merely forwarding its input.
    np.testing.assert_allclose(actual, 0.5 * initial, rtol=0.0, atol=2.0e-15)


def test_general_request_validate_resolve_compile_bind_run_with_independent_seed(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = _public_diagonal_krylov_case(general_request=True)
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    artifact = pops.compile(resolved)
    initial = np.arange(16, dtype=np.float64).reshape(1, 4, 4) / 16.0
    runtime = pops.bind(artifact, initial_state={"tracer": initial.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=0.125, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("tracer"), dtype=np.float64).reshape(1, 4, 4)
    np.testing.assert_allclose(actual, 0.5 * initial, rtol=0.0, atol=2.0e-15)
