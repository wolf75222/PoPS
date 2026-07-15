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
from pops.time import FailRun, FixedDt


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _public_identity_krylov_case():
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

    program = pops.Program("public_identity_gmres")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator(
        "identity", domain="state", range_="state", ncomp=1
    )
    program.set_apply(operator, lambda _program, _out, value: value)
    solution = program.solve(
        LinearProblem(
            operator,
            temporal.n,
            properties=LinearOperatorProperties.symmetric_positive_definite(),
        ),
        solver=GMRES(max_iter=4, restart=2, rel_tol=1.0e-13),
        name="identity_solution",
    ).consume(action=FailRun())
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
    case, layout = _public_identity_krylov_case()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = np.arange(16, dtype=np.float64).reshape(1, 4, 4) / 16.0
    runtime = pops.bind(artifact, initial_state={"tracer": initial.copy()})
    report = pops.run(runtime, t_end=0.125, max_steps=1)

    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("tracer"), dtype=np.float64).reshape(1, 4, 4)
    # GMRES exercises real floating-point reductions even for the identity operator.  Preserve the
    # physical identity to one ULP rather than requiring a fictitious bitwise no-op.
    np.testing.assert_array_max_ulp(actual, initial, maxulp=1)
