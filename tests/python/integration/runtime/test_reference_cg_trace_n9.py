"""N9 lot 4, SV13 and SV-T2 at one rank: the native CG is a substitution of the Trace class of
the reference ``pops.lib.solve.cg``.

Each problem is u - c Lap u = b on a periodic square, solved once by the native prepared CG and
once by the reference, written with ``@pops.solver`` and lowered onto the Program. The Trace class
asks for iteration counts within 2% and the same failure variants; this file holds the counts
equal and the solutions within a few ulps, and checks the variants of the native's branches:
the iteration limit, a breakdown, an estimate reached at the cap and a start that meets the
request. The 2-rank run and the qualification table are in the PR (Q2).
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.physics import Density
from pops.representations import Conservative
from pops.solvers import CG
from pops.spaces import CellState
from pops.time import FailRun, FixedDt
from pops.time.solve_outcome import RejectAttempt
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]

_COMPILED: dict = {}


def _as_real(program, count, bits=12):
    """The Int64 count of the evidence as a Scalar, bit by bit, to record it."""
    total = None
    for j in range(bits):
        bit = program._int64_binop(program._int64_binop(count, 1 << j, "floordiv"), 2, "mod")
        value = program.branch(
            program._int64_compare(bit, 1, "=="),
            lambda p, j=j: p.scalar_expression("bit%d_on" % j, float(1 << j)),
            lambda p, j=j: p.scalar_expression("bit%d_off" % j, 0.0))
        total = value if total is None else program.scalar_expression("bits%d" % j, total + value)
    return total


def _artifact(route, *, cells, c, rtol, maxit):
    from pops.lib.solve import cg

    key = (route, cells, c, rtol, maxit)
    if key in _COMPILED:
        return _COMPILED[key]
    name = "trace_%s_%d_%d" % (route, cells, len(_COMPILED))
    frame = Rectangle(name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model(name + "_m", frame=frame)
    state = model.state("U", components=("u",), representation=Conservative(),
                        space=CellState(frame=frame), roles={"u": Density()},
                        sampling="cell_average")
    (u,) = state
    flux = model.flux("zero", frame=frame, state=state,
                      components={x_axis: (0.0 * u,), y_axis: (0.0 * u,)},
                      waves={x_axis: (0.0,), y_axis: (0.0,)})
    rate = model.rate("inert", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                          reconstruction=reconstruction.FirstOrder(),
                                          riemann=riemann.Rusanov()))
    case = pops.Case(name)
    block = case.block("blk", model=model)
    case.numerics(numerics, block=block)
    program = pops.Program(name + "_p")
    temporal = program.state(block[state])
    operator = program.matrix_free_operator("helmholtz", domain="state", range_="state", ncomp=1)

    def apply(p, out, x):
        return x - c * p.laplacian(p.scalar_field("lap", ncomp=1), x)

    program.set_apply(operator, apply)
    problem = LinearProblem(operator, temporal.n, nullspace=None,
                            properties=LinearOperatorProperties.symmetric_positive_definite())
    if route == "native":
        outcome = program.solve(problem, solver=CG(max_iter=maxit, rel_tol=rtol, abs_tol=0.0),
                                name="sol")
        solution = outcome.consume(action=FailRun())
    else:
        outcome = program.solve(problem, name="sol", solver=cg(max_iterations=maxit)
                                .with_request(rtol=rtol, atol=0.0))
        solution = outcome.consume(action=RejectAttempt())
    program.record_scalar("cg_iterations", _as_real(program, outcome.evidence.iterations))
    program.record_scalar("cg_residual", outcome.evidence.residual_norm)
    program.commit(temporal.next, program.value("accepted", solution, at=temporal.next.point))
    program.step_strategy(FixedDt(0.125))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)))
    _COMPILED[key] = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    return _COMPILED[key]


def _run(route, b, **controls):
    artifact = _artifact(route, cells=b.shape[-1], **controls)
    runtime = pops.bind(artifact, initial_state={"blk": b.copy()},
                        resources={"execution_context": artifact_execution_context(artifact)})
    try:
        pops.run(runtime, t_end=0.125, max_steps=1, console=False)
    except Exception as error:  # noqa: BLE001 - the failure is the observed variant
        return {"error": "%s: %s" % (type(error).__name__, error)}
    diagnostics = runtime.program_report().diagnostics
    return {"iterations": diagnostics["cg_iterations"], "residual": diagnostics["cg_residual"],
            "x": np.asarray(runtime.state_global("blk"), dtype=np.float64).reshape(b.shape)}


_PROBLEMS = [(16, 0.1, 1e-10), (24, 0.05, 1e-8)]


@pytest.mark.parametrize("cells, c, rtol", _PROBLEMS)
def test_the_native_cg_is_in_the_trace_class_of_the_reference(
        cells, c, rtol, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    rng = np.random.default_rng(2026 + cells)
    for _ in range(3):
        b = rng.standard_normal((1, cells, cells))
        native = _run("native", b, c=c, rtol=rtol, maxit=500)
        reference = _run("reference", b, c=c, rtol=rtol, maxit=500)
        assert native["iterations"] > 10
        assert abs(reference["iterations"] - native["iterations"]) <= 0.02 * native["iterations"]
        assert reference["iterations"] == native["iterations"]  # observed: equal counts
        np.testing.assert_allclose(reference["x"], native["x"], rtol=0.0,
                                   atol=64 * np.finfo(float).eps * np.abs(native["x"]).max())
        assert abs(reference["residual"] - native["residual"]) <= 1e-4 * native["residual"]


def test_the_variants_of_the_native_branches(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    b = np.random.default_rng(2026).standard_normal((1, 16, 16))
    controls = dict(c=0.1, rtol=1e-10)
    count = _run("native", b, maxit=500, **controls)["iterations"]

    # The iteration limit: BudgetExhausted on both routes (code 11, « iteration limit »).
    native, reference = _run("native", b, maxit=10, **controls), _run("reference", b, maxit=10,
                                                                     **controls)
    assert "iteration_limit" in native["error"]
    assert "StepAttemptRejected" in reference["error"] and "code=11" in reference["error"]

    # The estimate reached exactly at the cap: no confirmation in the recurrence, one check of
    # the candidate, and the same count (solve_cg l. 938-945, wrapper l. 2118).
    capped = int(count)
    native = _run("native", b, maxit=capped, **controls)
    reference = _run("reference", b, maxit=capped, **controls)
    assert native["iterations"] == reference["iterations"] == capped

    # A start that meets the request: 0 iterations on both routes.
    zero = np.zeros_like(b)
    assert _run("native", zero, maxit=500, **controls)["iterations"] == 0
    assert _run("reference", zero, maxit=500, **controls)["iterations"] == 0

    # Loss of definiteness: SolveFailed « breakdown » on both routes (code 21, D3).
    native = _run("native", b, c=-1.0, rtol=1e-10, maxit=200)
    reference = _run("reference", b, c=-1.0, rtol=1e-10, maxit=200)
    assert "breakdown" in native["error"]
    assert "StepAttemptRejected" in reference["error"] and "code=21" in reference["error"]


@pytest.mark.parametrize("scale", [1e160, 1e-160])
def test_the_recurrence_is_scaled_as_the_native(
        scale, isolated_native_cache, native_cxx, kokkos_root):
    """Review F1 of #775: data of large or small magnitude. The recurrence runs in the units of
    the cycle scale on both routes, so r.r neither overflows at 1e160 nor underflows at 1e-160;
    the counts, the variant and the solution are those of order one, scaled."""
    del isolated_native_cache, native_cxx, kokkos_root
    b = np.random.default_rng(2026).standard_normal((1, 16, 16)) * scale
    controls = dict(c=0.1, rtol=1e-10, maxit=500)
    native, reference = _run("native", b, **controls), _run("reference", b, **controls)
    assert "error" not in native and "error" not in reference, (native, reference)
    assert reference["iterations"] == native["iterations"]
    np.testing.assert_allclose(reference["x"], native["x"], rtol=0.0,
                               atol=64 * np.finfo(float).eps * np.abs(native["x"]).max())


def test_a_right_hand_side_that_is_not_finite_fails_the_solve(
        isolated_native_cache, native_cxx, kokkos_root):
    """A NaN in b: an invalid evaluation of the solve at 0 iterations on both routes, never a
    failure of the run (the scale-safe norm never pairs a field that is not finite)."""
    del isolated_native_cache, native_cxx, kokkos_root
    b = np.random.default_rng(2026).standard_normal((1, 16, 16))
    b[0, 3, 5] = np.nan
    controls = dict(c=0.1, rtol=1e-10, maxit=500)
    native, reference = _run("native", b, **controls), _run("reference", b, **controls)
    assert "invalid_evaluation" in native["error"], native
    assert "StepAttemptRejected" in reference["error"], reference
    assert "code=3" in reference["error"] or "code=22" in reference["error"], reference
