"""Installed reception of a library body on stages and local unknowns, no recipe."""
import itertools
import math

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import affine_push_forward
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, FixedDt, LocalResidual
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.native_execution_context import artifact_execution_context

from tests.python.support.collective_checks import (
    collective_call, collective_check, state_snapshots,
)

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def affine_case(dimension, order, reverse):
    indices = tuple(index for index in itertools.product(range(order+1), repeat=dimension)
                    if sum(index) <= order)
    if reverse:
        indices = indices[::-1]
    matrix = ((.9,),) if dimension == 1 else (
        (math.cos(.2), math.sin(.2)), (-math.sin(.2), math.cos(.2)))
    offset = (.1,) if dimension == 1 else (.1, -.07)
    frame = Rectangle("affine_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("affine_moment_math", frame=frame)
    state = model.state("U", components=tuple("m"+"_".join(map(str, index)) for index in indices))
    case = pops.Case("affine_moment_body")
    block = case.block("matter", model)
    program = pops.Program("library_composition")
    q = program.state(block[state])

    def body(value):
        return affine_push_forward(value, indices=indices, matrix=matrix, offset=offset)

    target = program.value("mapped_stage", body(q.n), at=q.next.point)
    seed = program.value("distinct_seed", 1.1*q.n, at=q.next.point)

    def residual(p, unknown, target):
        image = body(unknown)
        return tuple(image[k]-target[k] for k in range(len(indices)))

    solved = program.solve(LocalResidual(residual, seed, captures={"target": target}),
                           solver=LocalNewton(tolerance=1e-12, max_iterations=8)).consume(action=FailRun())
    program.commit(q.next, program.value("accepted_image", body(solved), at=q.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout, indices, np.asarray(matrix), np.asarray(offset)


@pytest.mark.parametrize("dimension,order", [(1, 6), (2, 5)])
@pytest.mark.parametrize("reverse", [False, True])
def test_native_affine_body_on_stage_and_unknown_matches_discrete_measure(
        isolated_native_cache, native_cxx, kokkos_root, dimension, order, reverse):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, indices, matrix, offset = affine_case(dimension, order, reverse)
    points = np.random.default_rng(821).uniform(-.5, .6, (13, dimension))
    weights = np.linspace(.1, .7, 13)
    raw = np.array([np.dot(weights, np.prod(points ** index, axis=1)) for index in indices])
    moved = points @ matrix.T + offset
    expected = np.array([np.dot(weights, np.prod(moved ** index, axis=1)) for index in indices])
    initial = np.broadcast_to(raw[:, None, None], (len(indices), 4, 4)).copy()
    artifact, world = _compile(case, layout, "affine-body-%s-%s-%s" % (dimension, order, reverse))
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_state={"matter": initial},
                        resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=.01, max_steps=1))
    with collective_check(world):
        assert report.accepted_steps == 1
    gathered, = state_snapshots(runtime, world, ("matter",))
    # The committed image is exactly the original residual's lhs: this bound
    # tests that residual authority as well as the analytic affine push-forward.
    _root_check(world, lambda: np.testing.assert_allclose(
        gathered.reshape(initial.shape), np.broadcast_to(expected[:, None, None], initial.shape),
        atol=3e-12, rtol=0.))
