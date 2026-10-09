"""Installed five-component joint User reconstruction against an independent face oracle."""

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div, sqrt, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import (
    _assert_rejected, _compile, _root_check,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, .0005
WIDTHS = (2, 3)


def _initial(permutation):
    y, x = np.meshgrid((np.arange(N) + .5) / N, (np.arange(N) + .5) / N,
                       indexing="ij")
    q = np.stack([2. + .3 * k + .11 * np.sin(2 * np.pi * x + .2 * k)
                  + .07 * np.cos(2 * np.pi * y - .3 * k) for k in range(5)])
    return np.ascontiguousarray(q[list(permutation)])


def _matrices(permutation):
    row, col = np.indices((5, 5))
    x = np.diag(.7 + .08 * np.arange(5)) + .02 * (1 + row + col)
    y = np.diag(-.4 - .05 * np.arange(5)) - .015 * (1 + row + col)
    return tuple(a[np.ix_(permutation, permutation)] for a in (x, y))


def _case(permutation, *, guarded=False, primitive=False, mixed=False):
    frame = Rectangle("joint_user_native", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("joint_user_native_model", frame=frame)
    states = (model.species("first", state=tuple("q%d" % k for k in permutation[:2])),
              model.species("second", state=tuple("q%d" % k for k in permutation[2:])))
    q = tuple(component for state in states for component in state)
    if primitive:
        coordinates = (q[0], *(model.primitive("p%d" % j,
                         q[j] / q[0] + .02 * q[0] * q[0]) for j in range(1, 5)))
        inverse = (coordinates[0], *(coordinates[0] *
                   (coordinates[j] - .02 * coordinates[0] * coordinates[0])
                   for j in range(1, 5)))
        model.primitive_state(*coordinates, states=states, conservative=inverse)
        model.recovery_admissibility(states=states, **{q[0].component: q[0] > 0})
    parameters = (model.param(RuntimeParam("alpha", default=.03)),
                  model.param(RuntimeParam("beta", default=-.02)))
    alpha, beta = (model.value(parameter) for parameter in parameters)
    matrices = _matrices(permutation)
    speeds = tuple(float(np.max(np.sum(np.abs(matrix), axis=1))) for matrix in matrices)
    fluxes, rates = [], []
    calls = []
    for row, state in enumerate(states):
        span = range(0, 2) if row == 0 else range(2, 5)
        flux = model.flux("F%d" % row, state=state, frame=frame,
            components={axis: tuple(sum(float(matrix[k, j]) * q[j] for j in range(5))
                                    for k in span)
                        for axis, matrix in zip(frame.axes, matrices, strict=True)},
            waves=({axis: (speed,) * 2 for axis, speed in zip(frame.axes, speeds, strict=True)}
                   if row == 0 else None))
        fluxes.append(flux)
        rates.append(model.rate("R%d" % row, equation=ddt(state) == -div(flux)))

    def first(sample):
        calls.append("first")
        own, other = sample(0), sample(3, states[1])
        first_value = own[0] + alpha * (other[2] - sample(-1, states[1])[2])
        if guarded:
            first_value = where(own[0] < 0, lambda: sqrt(own[0]), lambda: first_value)
        return (first_value,
                own[1] - alpha * (sample(1, states[1])[0] - sample(-2)[0]))

    def second(sample):
        calls.append("second")
        own = sample(0)
        return (own[0] + beta * (sample(1, states[0])[1] - sample(-1, states[0])[1]),
                own[1] + beta * (sample(1)[2] - sample(-1)[2]),
                own[2] + beta * (sample(2, states[0])[0] - sample(-1, states[0])[0]))

    case = pops.Case("joint_user_native_case")
    # Registration order differs from the physical declaration/packed component order.
    blocks = {row: case.block("block%d" % row, model, states=(states[row],)) for row in (1, 0)}
    for row, body in enumerate((first, second)):
        policy = (reconstruction.User(
            lambda sample: sample(0) + .1 * (sample(1) - sample(-1)), formal_order=1)
            if mixed and row == 1 else
            reconstruction.User(body, state=states[row], sampling=(states[1-row],),
                                formal_order=1))
        variable_policy = (variables.Primitive(states[row]) if primitive
                           else variables.Conservative(states[row]))
        method = FiniteVolume(flux=fluxes[row], variables=variable_policy,
                              reconstruction=policy, riemann=riemann.Rusanov(),
                              sampling=(states[1-row],))
        plan = DiscretizationPlan()
        plan.rates.add(rates[row], method)
        case.numerics(plan, block=blocks[row])
        case.initials.add(InitialCondition(state=blocks[row][states[row]], value=BindArray(),
                                           projection=ConservativeCellAverage()))
    program = pops.Program("joint_user_euler")
    temporal = {row: program.state(blocks[row][states[row]]) for row in (1, 0)}
    bindings = {states[row]: temporal[row].n for row in (1, 0)}
    for row in (1, 0):
        rhs = rates[row](temporal[row].n, bindings=bindings)
        program.commit(temporal[row].next,
            program.value("accepted%d" % row, temporal[row].n + program.dt * rhs,
                          at=temporal[row].next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)))
    subjects = tuple(case.resolve(blocks[row][states[row]]) for row in range(2))
    handles = tuple(tuple(blocks[row][parameter] for parameter in parameters)
                    for row in range(2))
    return case, layout, subjects, handles, matrices, calls


def _oracle(initial, matrices, alpha, beta, *, primitive=False, mixed=False):
    result = initial.copy()
    sampled = initial.copy()
    if primitive:
        sampled[1:] = initial[1:] / initial[0:1] + .02 * initial[0:1] ** 2

    def conservative(value):
        if not primitive:
            return value
        physical = value.copy()
        physical[1:] = value[0:1] * (value[1:] - .02 * value[0:1] ** 2)
        return physical

    for matrix, axis in zip(matrices, (2, 1), strict=True):
        def sample(offset, *, right=False):
            return np.roll(sampled, offset - 1 if right else -offset, axis=axis)
        def trace(right):
            q0, qm1, qp1 = sample(0, right=right), sample(-1, right=right), sample(1, right=right)
            qp2, qp3 = sample(2, right=right), sample(3, right=right)
            first = (q0[0] + alpha * (qp3[4] - qm1[4]),
                     q0[1] - alpha * (qp1[2] - sample(-2, right=right)[0]))
            second = (tuple(q0[k] + .1 * (qp1[k] - qm1[k]) for k in range(2, 5))
                      if mixed else
                      (q0[2] + beta * (qp1[1] - qm1[1]),
                       q0[3] + beta * (qp1[4] - qm1[4]),
                       q0[4] + beta * (qp2[0] - qm1[0])))
            return np.stack((*first, *second))
        left, right = conservative(trace(False)), conservative(trace(True))
        speed = float(np.max(np.sum(np.abs(matrix), axis=1)))
        face = .5 * np.einsum("ij,jyx->iyx", matrix, left + right) - .5 * speed * (right - left)
        result += DT * N * (np.roll(face, 1, axis=axis) - face)
    return result


def _bind(artifact, subjects, handles, initial, alpha, beta):
    params = {}
    for row, pair in enumerate(handles):
        params[pair[0]] = alpha if row == 0 else 3.7
        params[pair[1]] = beta if row == 1 else -4.2
    return pops.bind(artifact, params=params,
        initial_values={subjects[0]: initial[:2].copy(), subjects[1]: initial[2:].copy()},
        resources={"execution_context": artifact_execution_context(artifact)})


def _gather(runtime, world):
    rows = tuple(runtime.state_global("block%d" % row) for row in range(2))
    if world is not None and world.rank != 0:
        return None
    return np.concatenate((np.asarray(rows[0]).reshape(2, N, N),
                           np.asarray(rows[1]).reshape(3, N, N)))


@pytest.mark.parametrize("primitive,mixed,depths", ((False, False, (4, 3)),
                                                  (False, True, (4, 2)),
                                                  (True, False, (4, 3))))
def test_joint_cross_row_storage_halo_covers_every_sampled_component(primitive, mixed, depths):
    """A row with a narrow own stencil still supplies a wider neighbor's samples."""
    from pops.codegen.program_models import ProgramModelGraph

    case, layout, *_ = _case(tuple(range(5)), primitive=primitive, mixed=mixed)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    for row in range(2):
        model = graph.model_for_block("block%d" % row)._m
        group = model._principal_groups[0]["group"]
        assert tuple(method.ghost_depth for method in group.methods) == depths
        assert model._program_state_ghost_depth >= 4


@pytest.mark.parametrize("permutation,primitive,mixed", ((tuple(range(5)), False, False),
                                                         ((4, 2, 0, 3, 1), True, False),
                                                         (tuple(range(5)), False, True)))
def test_joint_native_face_matches_cross_component_oracle_and_live_rebind(
        isolated_native_cache, native_cxx, kokkos_root, permutation, primitive, mixed):
    case, layout, subjects, handles, matrices, calls = _case(
        permutation, primitive=primitive, mixed=mixed)
    artifact, world = _compile(case, layout, "joint-user-cross-component")
    authoring_calls = tuple(calls)
    initial = _initial(permutation)
    prior = None
    for alpha, beta in ((.03, -.02), (-.045, .06)):
        runtime = _bind(artifact, subjects, handles, initial, alpha, beta)
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
        actual = _gather(runtime, world)
        def check():
            expected = _oracle(initial, matrices, alpha, beta, primitive=primitive, mixed=mixed)
            np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=4e-12)
            np.testing.assert_allclose(actual.sum(axis=(1, 2)), initial.sum(axis=(1, 2)),
                                       rtol=0, atol=5e-11)
            if prior is not None:
                assert np.max(np.abs(actual - prior)) > 1e-8
        _root_check(world, check)
        assert tuple(calls) == authoring_calls, "Python body ran after native compilation"
        prior = actual


def test_joint_invalid_active_branch_refuses_all_rows_and_clock(
        isolated_native_cache, native_cxx, kokkos_root):
    permutation = tuple(range(5))
    case, layout, subjects, handles, _, calls = _case(permutation, guarded=True)
    artifact, world = _compile(case, layout, "joint-user-invalid-branch")
    authoring_calls = tuple(calls)
    initial = _initial(permutation)
    good = _bind(artifact, subjects, handles, initial, .03, -.02)
    pops.run(good, t_end=DT, max_steps=1, console=False)
    bad_initial = initial.copy()
    bad_initial[0, 2, 3] = -.25
    bad = _bind(artifact, subjects, handles, bad_initial, .03, -.02)
    before = _gather(bad, world)
    _assert_rejected(bad, world, DT)
    after = _gather(bad, world)
    _root_check(world, lambda: np.testing.assert_array_equal(after, before))
    assert tuple(calls) == authoring_calls, "Python body ran during native evaluation"
