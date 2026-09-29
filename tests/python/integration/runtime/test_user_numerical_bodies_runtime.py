"""Independent native User-body witnesses, using saved states and FV face oracles.

AMR1 exercises the actual hierarchy route with public transfer/initial contracts;
it does not qualify coarse/fine reflux. Positivity is asserted only for the scalar
diffusive pulse below, never inferred for an arbitrary authored face body.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, 1.e-3


def _world():
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator
    return _pops.mpi_world() if native_mpi_communicator(_pops) == "MPI_COMM_WORLD" else None


def _root_check(world, operation):
    failure = ""
    if world is None or world.rank == 0:
        try:
            operation()
        except Exception as exc:
            failure = "%s: %s" % (type(exc).__name__, exc)
    if world is not None:
        from pops._native_collectives import broadcast_value
        failure = broadcast_value(world, failure, root=0)
    assert not failure, failure


def _snapshot(runtime, name, width, layout_kind, world):
    data = (runtime.block_level_state_global(name, 0) if layout_kind == "amr1"
            else runtime.state_global(name))
    return (np.asarray(data).reshape(width, N, N).copy()
            if world is None or world.rank == 0 else None)


def _compile(case, layout, route):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    world = _world()
    if world is None:
        artifact = pops.compile(resolved)
    else:
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
        artifact = compile_resolved_plan_once(world, resolved, route=route,
                                              compile_artifact=pops.compile)
    assert artifact.resolved_dimension == 2
    return artifact, world


def _case(width, *, layout_kind="uniform", invalid=None, pulse=False, dt=DT):
    frame = Rectangle("user_square", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("authored_numerics", frame=frame)
    # Deliberately permute the scientific component names independently of storage width.
    state = model.state("U", components=tuple("q%d" % i for i in reversed(range(width))))
    reconstruction_parameter = model.param(RuntimeParam("slope", default=.2))
    dissipation_parameter = model.param(RuntimeParam("dissipation", default=.75))
    r, d = model.value(reconstruction_parameter), model.value(dissipation_parameter)
    gate_parameter = model.param(RuntimeParam("gate", default=2.)) if invalid else None
    gate = model.value(gate_parameter) if invalid else None
    velocity = (0., 0.) if pulse else (.7, -.4)
    spectral = (1., 0.) if pulse else tuple(abs(v) for v in velocity)
    flux = model.flux("transport", frame=frame, state=state,
        components={axis: tuple(v * q for q in state)
                    for axis, v in zip(frame.axes, velocity, strict=True)},
        waves={axis: (a,) * width for axis, a in zip(frame.axes, spectral, strict=True)})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))

    def stencil(sample):
        normal = sample(0) + r * (sample(1) - sample(-1))
        return (where(sample(0) > gate, lambda: 1 / (sample(0) - sample(0)), lambda: normal)
                if invalid == "reconstruction" else normal)

    def face(left, right, fl, fr, speed):
        normal = .5 * (fl + fr) - d * speed * (right - left)
        return (tuple(where(left[0] > gate, lambda: 1 / (left[0] - left[0]),
                            lambda item=item: item) for item in normal)
                if invalid == "body" else normal)

    def stability(left, right, fl, fr, speed):
        return (where(left[0] > gate, lambda: 1 / (left[0] - left[0]),
                      lambda: 2 * d * speed)
                if invalid == "bound" else 2 * d * speed)

    method = FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.User(stencil, formal_order=1),
        riemann=riemann.User(body=face, stability=stability, state=state))
    case = pops.Case("user_numerical_native")
    blocks = tuple(case.block(name, model) for name in (("pulse",) if pulse else ("first", "second")))
    program = pops.Program("authored_forward_euler")
    temporal = tuple(program.state(block[state]) for block in blocks)
    for block, q in zip(blocks, temporal, strict=True):
        plan = DiscretizationPlan()
        plan.rates.add(rate, method)
        case.numerics(plan, block=block)
        rhs = rate(q.n)
        program.commit(q.next, program.value("accepted", q.n + program.dt * rhs,
                                              at=q.next.point))
        case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                           projection=ConservativeCellAverage()))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes))
    if layout_kind == "uniform":
        layout = Uniform(grid)
    else:
        from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                              Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
        from pops.layouts import AMR
        from pops.lib.amr import StateTransfer
        from pops.time import every
        transfer = AMRTransfer()
        for block in blocks:
            transfer.state(block[state], StateTransfer())
        tag_threshold = case.param(RuntimeParam("refinement_threshold", default=1000.))
        layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=1, ratios=()),
            tagging=AMRTagging(rules=(Tag(ValueExpr(blocks[0][state])[state.components[0]] > case.value(tag_threshold)),
                                     Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.synchronous())
    handles = tuple((block[reconstruction_parameter], block[dissipation_parameter],
                     None if gate_parameter is None else block[gate_parameter]) for block in blocks)
    subjects = tuple(case.resolve(block[state]) for block in blocks)
    return case, layout, handles, subjects, velocity, spectral


def _initial(width, shift=0.):
    coordinate = (np.arange(N) + .5) / N
    x, y = np.meshgrid(coordinate, coordinate)
    # Exact FV means of smooth separable trigonometric fields, not center samples.
    return np.ascontiguousarray(np.stack([
        2. + j + shift + .13 * np.sinc(1 / N) * np.sin(2 * np.pi * x + .3 * j)
        + .07 * np.sinc((j % 2 + 1) / N) * np.cos(2 * np.pi * (j % 2 + 1) * y)
        for j in range(width)]))


def _oracle(initial, velocity, spectral, slope, dissipation, dt):
    rhs = np.zeros_like(initial)
    for v, a, axis in zip(velocity, spectral, (2, 1), strict=True):
        adjacent = np.roll(initial, -1, axis=axis)
        left = initial + slope * (adjacent - np.roll(initial, 1, axis=axis))
        # The right stencil is oriented towards the face, hence its reversed offsets.
        right = adjacent + slope * (initial - np.roll(initial, -2, axis=axis))
        density = .5 * v * (left + right) - dissipation * a * (right - left)
        rhs += N * (np.roll(density, 1, axis=axis) - density)
    return initial + dt * rhs


def _bind(artifact, handles, subjects, initials, coefficients, *, gate=None):
    params = {}
    for (r, d, g), (slope, dissipation) in zip(handles, coefficients, strict=True):
        params.update({r: slope, d: dissipation})
        if g is not None:
            params[g] = gate
    return pops.bind(artifact, params=params, initial_values=dict(zip(subjects, initials, strict=True)),
        resources={"execution_context": artifact_execution_context(artifact)})


def _assert_rejected(runtime, world, dt):
    failure = ""
    try:
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
    except RuntimeError as exc:
        failure = str(exc)
    failures = (failure,)
    if world is not None:
        from pops._native_collectives import allgather_value
        failures = allgather_value(world, failure)
    assert all(failures), "native rejection must reach every participating rank"
    assert runtime.time() == 0. and runtime.macro_step() == 0


@pytest.mark.parametrize("width,layout_kind", ((2, "uniform"), (5, "uniform"), (2, "amr1")))
def test_two_block_captures_and_rebind_match_independent_fv_oracle(
        isolated_native_cache, native_cxx, kokkos_root, width, layout_kind):
    case, layout, handles, subjects, velocity, spectral = _case(width, layout_kind=layout_kind)
    artifact, world = _compile(case, layout, "user-rebind-%d-%s" % (width, layout_kind))
    initials = (_initial(width), _initial(width, .37))
    outputs = []
    for coefficients in (((.17, .65), (-.11, 1.2)), ((-.08, 1.35), (.23, .55))):
        runtime = _bind(artifact, handles, subjects, initials, coefficients)
        assert runtime.n_levels() == 1
        report = pops.run(runtime, t_end=DT, max_steps=1, console=False)
        actual = tuple(_snapshot(runtime, name, width, layout_kind, world)
                       for name in ("first", "second"))
        def check():
            for initial, result, (slope, dissipation) in zip(initials, actual, coefficients, strict=True):
                np.testing.assert_allclose(result, _oracle(initial, velocity, spectral,
                    slope, dissipation, DT), rtol=0., atol=3.e-13)
                np.testing.assert_allclose(result.sum(axis=(1, 2)), initial.sum(axis=(1, 2)),
                                           rtol=0., atol=2.e-12)
        _root_check(world, check)
        assert report.accepted_steps == runtime.macro_step() == 1
        assert runtime.time() == pytest.approx(DT, rel=0., abs=1.e-16)
        outputs.append(actual)
    _root_check(world, lambda: np.testing.assert_array_less(
        1.e-5, max(np.max(np.abs(a - b)) for a, b in zip(*outputs, strict=True))))


@pytest.mark.parametrize("invalid", ("reconstruction", "body", "bound"))
@pytest.mark.parametrize("layout_kind", ("uniform", "amr1"))
def test_inactive_singularity_succeeds_and_active_failure_rolls_back(
        isolated_native_cache, native_cxx, kokkos_root, invalid, layout_kind):
    case, layout, handles, subjects, _, _ = _case(2, layout_kind=layout_kind, invalid=invalid)
    artifact, world = _compile(case, layout, "user-invalid-%s-%s" % (invalid, layout_kind))
    initials = (np.ones((2, N, N)), np.ones((2, N, N)))
    for active in (False, True):
        runtime = _bind(artifact, handles, subjects, initials, ((.2, .75), (.1, .6)),
                        gate=0. if active else 2.)
        if active:
            _assert_rejected(runtime, world, DT)
        else:
            assert pops.run(runtime, t_end=DT, max_steps=1, console=False).accepted_steps == 1
        actual = tuple(_snapshot(runtime, name, 2, layout_kind, world) for name in ("first", "second"))
        def check():
            for result, initial in zip(actual, initials, strict=True):
                np.testing.assert_array_equal(result, initial)
        _root_check(world, check)


@pytest.mark.parametrize("layout_kind", ("uniform", "amr1"))
def test_authored_dissipation_bound_rejects_unsafe_pulse_and_accepts_safe_step(
        isolated_native_cache, native_cxx, kokkos_root, layout_kind):
    unsafe, safe = .5 / N, .02 / N
    case, layout, handles, subjects, velocity, spectral = _case(
        1, layout_kind=layout_kind, pulse=True, dt=unsafe)
    artifact, world = _compile(case, layout, "user-dissipative-pulse-" + layout_kind)
    initial = np.zeros((1, N, N))
    initial[0, N // 2, N // 2] = 1.
    for dt in (unsafe, safe):
        runtime = _bind(artifact, handles, subjects, (initial,), ((0., 10.),))
        if dt == unsafe:
            _assert_rejected(runtime, world, dt)
        else:
            assert pops.run(runtime, t_end=dt, max_steps=1, console=False).accepted_steps == 1
        actual = _snapshot(runtime, "pulse", 1, layout_kind, world)
        def check():
            if dt == unsafe:
                np.testing.assert_array_equal(actual, initial)
            else:
                np.testing.assert_allclose(actual, _oracle(initial, velocity, spectral, 0., 10., dt),
                                           rtol=0., atol=3.e-14)
                assert actual.min() >= 0. and actual.max() <= 1.
                assert actual.sum() == pytest.approx(1., rel=0., abs=3.e-14)
        _root_check(world, check)


def test_independent_pulse_oracle_exposes_the_physical_only_bound_error():
    initial = np.zeros((1, N, N))
    initial[0, N // 2, N // 2] = 1.
    unsafe = _oracle(initial, (0., 0.), (1., 0.), 0., 10., .5 / N)
    safe = _oracle(initial, (0., 0.), (1., 0.), 0., 10., .02 / N)
    assert unsafe.min() == -9.
    assert safe.min() == 0. and safe.max() == pytest.approx(.6)
    assert unsafe.sum() == safe.sum() == 1.


def test_rank_one_owned_invalid_cell_rejects_collectively_without_publication(
        isolated_native_cache, native_cxx, kokkos_root):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    world = _world()
    if world is None or world.size != 2:
        pytest.skip("this witness requires exactly two native MPI ranks")
    from pops._native_collectives import allgather_value
    case, layout, handles, subjects, _, _ = _case(1, invalid="body")
    artifact, world = _compile(case, layout, "user-invalid-owned-rank-one")
    initials = (np.ones((1, N, N)), np.ones((1, N, N)))
    probe = _bind(artifact, handles, subjects, initials, ((0., .75), (0., .6)), gate=2.)
    owned = allgather_value(world, probe.local_boxes("first"))
    if not owned[1]:
        pytest.skip("the authored Uniform layout supplies no valid cell to rank one")
    lower, upper = owned[1][0]
    x, y = ((lo + hi - 1) // 2 for lo, hi in zip(lower, upper, strict=True))
    disturbed = initials[0].copy()
    disturbed[0, y, x] = 3.  # The active invalid branch is reached from a rank-one-owned cell.
    runtime = _bind(artifact, handles, subjects, (disturbed, initials[1]),
                    ((0., .75), (0., .6)), gate=2.)
    assert runtime.local_boxes("first") == tuple(owned[world.rank])
    before = tuple(_snapshot(runtime, name, 1, "uniform", world) for name in ("first", "second"))
    _assert_rejected(runtime, world, DT)
    after = tuple(_snapshot(runtime, name, 1, "uniform", world) for name in ("first", "second"))
    def check():
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(new, old)
    _root_check(world, check)
