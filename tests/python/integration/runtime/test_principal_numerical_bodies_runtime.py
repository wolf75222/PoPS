"""Independent Uniform Dim2 receipt for mixed principal groups and User bodies.

The five-component oracle uses complete physical matrices and saved cell averages.
Only compile/MPI/gather plumbing is shared with the standalone User receipt.
"""
import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import (
    _assert_rejected, _compile, _root_check, _snapshot,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, .001
WIDTHS = (2, 3)
SLICES = (slice(0, 2), slice(2, 5))


def _matrices(permutation):
    row, col = np.indices((5, 5))
    x = np.diag(1. + .1 * np.arange(5)) + .025 * (row + col + 1.)
    y = np.diag(-.5 - .05 * np.arange(5)) - .015 * (row + col + 1.)
    return tuple(matrix[np.ix_(permutation, permutation)] for matrix in (x, y))


def _initial(permutation):
    x, y = np.meshgrid((np.arange(N) + .5) / N, (np.arange(N) + .5) / N)
    means = np.stack([2. + j + .13 * np.sinc(1 / N) * np.sin(2 * np.pi * x + .3 * j)
                      + .07 * np.sinc((j % 2 + 1) / N) * np.cos(2 * np.pi * (j % 2 + 1) * y)
                      for j in range(5)])
    return np.ascontiguousarray(means[list(permutation)])


def _oracle(initial, matrices, slopes, dissipations, effective_dt):
    slope = np.repeat(slopes, WIDTHS)[:, None, None]
    dissipation = np.repeat(dissipations, WIDTHS)[:, None, None]
    rhs = np.zeros_like(initial)
    for matrix, axis in zip(matrices, (2, 1), strict=True):
        adjacent = np.roll(initial, -1, axis=axis)
        left = initial + slope * (adjacent - np.roll(initial, 1, axis=axis))
        right = adjacent + slope * (initial - np.roll(initial, -2, axis=axis))
        physical_left = np.einsum("ij,jyx->iyx", matrix, left)
        physical_right = np.einsum("ij,jyx->iyx", matrix, right)
        speed = np.max(np.sum(np.abs(matrix), axis=1))
        face = .5 * (physical_left + physical_right) - dissipation * speed * (right - left)
        rhs += N * (np.roll(face, 1, axis=axis) - face)
    return initial + effective_dt * rhs


def _case(permutation, *, dt=DT, weight=1., diagnostic=False):
    frame = Rectangle("mixed_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("mixed_principal", frame=frame)
    states = tuple(model.species("row%d" % i,
        state=tuple("q%d" % j for j in permutation[part])) for i, part in enumerate(SLICES))
    parameters = tuple((model.param(RuntimeParam("slope%d" % i, default=.1)),
                        model.param(RuntimeParam("dissipation%d" % i, default=.75)))
                       for i in range(2))
    matrices = _matrices(permutation)
    speeds = tuple(float(np.max(np.sum(np.abs(matrix), axis=1))) for matrix in matrices)
    symbols = tuple(q for state in states for q in state)
    fluxes, rates = [], []
    for i, (state, part) in enumerate(zip(states, SLICES, strict=True)):
        flux = model.flux("F%d" % i, state=state, frame=frame,
            components={axis: tuple(sum(float(matrix[k, j]) * symbols[j] for j in range(5))
                                    for k in range(part.start, part.stop))
                        for axis, matrix in zip(frame.axes, matrices, strict=True)},
            waves=({axis: (speed,) * WIDTHS[0]
                    for axis, speed in zip(frame.axes, speeds, strict=True)} if i == 0 else None))
        fluxes.append(flux)
        rates.append(model.rate("R%d" % i, equation=ddt(state) == -div(flux)))
    order = (0, 1) if permutation == tuple(range(5)) else (1, 0)
    case = pops.Case("mixed_principal_receipt")
    blocks = {i: case.block("block%d" % i, model, states=(states[i],)) for i in order}
    for i in order:
        slope, dissipation = (model.value(parameter) for parameter in parameters[i])
        method = FiniteVolume(flux=fluxes[i], variables=variables.Conservative(states[i]),
            reconstruction=reconstruction.User(
                lambda sample, slope=slope: sample(0) + slope * (sample(1) - sample(-1)),
                formal_order=1),
            riemann=riemann.User(state=states[i],
                body=lambda left, right, fl, fr, speed, d=dissipation:
                    .5 * (fl + fr) - d * speed * (right - left),
                stability=lambda left, right, fl, fr, speed, d=dissipation: 2 * d * speed),
            sampling=(states[1-i],))
        plan = DiscretizationPlan()
        plan.rates.add(rates[i], method)
        case.numerics(plan, block=blocks[i])
        case.initials.add(InitialCondition(state=blocks[i][states[i]], value=BindArray(),
                                           projection=ConservativeCellAverage()))
    program = pops.Program("mixed_user_euler")
    quantities = {i: program.state(blocks[i][states[i]]) for i in order}
    bindings = {states[i]: quantities[i].n for i in order}
    for i in order:
        q = quantities[i]
        rhs = rates[i](q.n, bindings=bindings)
        value = q.n if diagnostic else q.n + weight * program.dt * rhs
        program.commit(q.next, program.value("accepted", value, at=q.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)))
    subjects = tuple(case.resolve(blocks[i][states[i]]) for i in range(2))
    handles = tuple(tuple(tuple(blocks[i][parameter] for parameter in pair)
                          for pair in parameters) for i in range(2))
    return case, layout, subjects, handles, matrices


def _bind(artifact, subjects, handles, initial, slopes, dissipations):
    params = {}
    for owner, table in enumerate(handles):
        for row, (slope_handle, dissipation_handle) in enumerate(table):
            # Unused slots are deliberately different: a wrong row/table is observable.
            params[slope_handle] = slopes[row] if row == owner else -.37 - owner
            params[dissipation_handle] = dissipations[row] if row == owner else 3.7 + owner
    return pops.bind(artifact, params=params,
        initial_values={subject: initial[part].copy()
                        for subject, part in zip(subjects, SLICES, strict=True)},
        resources={"execution_context": artifact_execution_context(artifact)})


def _gather(runtime, world):
    values = [_snapshot(runtime, "block%d" % i, width, "uniform", world)
              for i, width in enumerate(WIDTHS)]
    return np.concatenate(values) if world is None or world.rank == 0 else None


@pytest.mark.parametrize("permutation", (tuple(range(5)), (4, 1, 3, 0, 2)))
def test_mixed_group_captures_permutation_and_same_artifact_rebind(
        isolated_native_cache, native_cxx, kokkos_root, permutation):
    case, layout, subjects, handles, matrices = _case(permutation)
    artifact, world = _compile(case, layout, "principal-user-mixed")
    initial = _initial(permutation)
    previous = None
    for slopes, dissipations in (((.07, .19), (.6, .9)), ((.23, -.04), (1.1, .55))):
        runtime = _bind(artifact, subjects, handles, initial, slopes, dissipations)
        before = _gather(runtime, world)
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
        actual = _gather(runtime, world)

        def check():
            np.testing.assert_array_equal(before, initial)
            expected = _oracle(initial, matrices, slopes, dissipations, DT)
            np.testing.assert_allclose(actual, expected, rtol=3.e-12, atol=3.e-12)
            np.testing.assert_allclose(actual.sum(axis=(1, 2)), initial.sum(axis=(1, 2)), atol=2.e-11)
            for part in SLICES:
                assert np.max(np.abs(actual[part] - initial[part])) > 1.e-6
            if previous is not None:
                assert np.max(np.abs(actual - previous)) > 1.e-6
        _root_check(world, check)
        assert runtime.time() == pytest.approx(DT) and runtime.macro_step() == 1
        previous = actual


@pytest.mark.parametrize("mode", ("unsafe", "scaled", "diagnostic"))
def test_group_stability_restricts_the_explicit_consumer_and_rolls_back_jointly(
        isolated_native_cache, native_cxx, kokkos_root, mode):
    permutation = (4, 1, 3, 0, 2)
    matrices = _matrices(permutation)
    dt = .5 / (N * sum(np.max(np.sum(np.abs(a), axis=1)) for a in matrices))
    weight = .01 if mode == "scaled" else 1.
    case, layout, subjects, handles, matrices = _case(
        permutation, dt=dt, weight=weight, diagnostic=mode == "diagnostic")
    artifact, world = _compile(case, layout, "principal-user-consumer")
    initial = _initial(permutation)
    slopes, dissipations = (0., 0.), (10., 10.)
    runtime = _bind(artifact, subjects, handles, initial, slopes, dissipations)
    before = _gather(runtime, world)
    if mode == "unsafe":
        _assert_rejected(runtime, world, dt)
    else:
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
        assert runtime.time() == pytest.approx(dt) and runtime.macro_step() == 1
    actual = _gather(runtime, world)

    def check():
        np.testing.assert_array_equal(before, initial)
        if mode in ("unsafe", "diagnostic"):
            np.testing.assert_array_equal(actual, initial)
        else:
            np.testing.assert_allclose(actual,
                _oracle(initial, matrices, slopes, dissipations, weight * dt),
                rtol=3.e-12, atol=3.e-12)
    _root_check(world, check)


def test_mixed_oracle_has_cross_block_coupling_and_exact_periodic_conservation():
    permutation = (4, 1, 3, 0, 2)
    matrices, initial = _matrices(permutation), _initial(permutation)
    for matrix in matrices:
        assert np.all(matrix[:2, 2:] != 0.) and np.all(matrix[2:, :2] != 0.)
    result = _oracle(initial, matrices, (.07, .19), (.6, .9), DT)
    np.testing.assert_allclose(result.sum(axis=(1, 2)), initial.sum(axis=(1, 2)), atol=2.e-11)
    changed = initial.copy()
    changed[2] += .2 * np.sin(2 * np.pi * (np.arange(N) + .5) / N)[None, :]
    assert np.max(np.abs(_oracle(changed, matrices, (.07, .19), (.6, .9), DT)[:2] - result[:2])) > 1.e-6
    frequency = 20 * N * sum(np.max(np.sum(np.abs(a), axis=1)) for a in matrices)
    macro_dt = .5 / (N * sum(np.max(np.sum(np.abs(a), axis=1)) for a in matrices))
    assert macro_dt * frequency == pytest.approx(10.)
    assert .01 * macro_dt * frequency == pytest.approx(.1)
