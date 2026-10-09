"""Independent FV oracle for an authored scalar, componentwise reconstruction.

This does not qualify vector/characteristic reconstruction or TVD behavior.
The oracle includes both face orientations and each spatial axis.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, where
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N = 8
DT = 1.e-3


def asymmetric_face(sample):
    return sample(0) + .3 * (sample(1) - sample(-1))


def guarded_face(sample):
    # Every cell is positive. Eager evaluation of this inactive singular branch
    # must not alter the actual numerical flux or cause an execution failure.
    return where(sample(0) > 0,
                 lambda: sample(0) + .3 * (sample(1) - sample(-1)),
                 lambda: 1 / (sample(0) - sample(0)))


def make_case(*, swapped_axes=False, guarded=False):
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("two_component_advection", frame=frame)
    state = model.state("U", components=("a", "b"), representation=Conservative(),
                        space=CellState(frame=frame))
    velocities = (-.4, .7) if swapped_axes else (.7, -.4)
    flux = model.flux(
        "transport", frame=frame, state=state,
        components={axis: tuple(speed * state[c] for c in range(2))
                    for axis, speed in zip(frame.axes, velocities)},
        waves={axis: tuple(speed + 0 * state[c] for c in range(2))
               for axis, speed in zip(frame.axes, velocities)})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    method = reconstruction.User(body=guarded_face if guarded else asymmetric_face,
                                 formal_order=1, name="asymmetric_componentwise")
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state), reconstruction=method,
        riemann=riemann.Rusanov()))
    case = pops.Case("authored_reconstruction_adversarial")
    block = case.block("fluid", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, N),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, velocities


def initial_state(*, swapped_axes=False, swapped_components=False):
    x = (np.arange(N) + .5) / N
    xx, yy = np.meshgrid(x, x)
    initial = np.stack((1. + .17 * np.sin(2 * np.pi * xx) + .09 * np.cos(4 * np.pi * yy),
                        2. + .11 * np.cos(4 * np.pi * xx + 2 * np.pi * yy)))
    if swapped_axes:
        initial = initial.transpose(0, 2, 1)
    if swapped_components:
        initial = initial[::-1]
    return np.ascontiguousarray(initial)


def oracle_step(initial, velocities):
    rhs = np.zeros_like(initial)
    for speed, array_axis in zip(velocities, (2, 1)):
        plus = np.roll(initial, -1, axis=array_axis)
        minus = np.roll(initial, 1, axis=array_axis)
        plus_two = np.roll(initial, -2, axis=array_axis)
        left = initial + .3 * (plus - minus)
        # The right source is the next cell; its local positive stencil
        # direction points towards the face, hence toward the previous cell.
        right = plus + .3 * (initial - plus_two)
        flux = .5 * speed * (left + right) - .5 * abs(speed) * (right - left)
        rhs += N * (np.roll(flux, 1, axis=array_axis) - flux)
    return initial + DT * rhs


@pytest.mark.parametrize("swapped_axes,swapped_components,guarded", [
    (False, False, False), (True, True, False), (False, True, True),
], ids=("both_orientations", "axes_and_components_permuted", "inactive_singularity"))
def test_authored_reconstruction_matches_two_component_two_axis_fv_oracle(
        isolated_native_cache, native_cxx, kokkos_root,
        swapped_axes, swapped_components, guarded):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, velocities = make_case(swapped_axes=swapped_axes, guarded=guarded)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial = initial_state(swapped_axes=swapped_axes, swapped_components=swapped_components)
    runtime = pops.bind(artifact, initial_state={"fluid": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=DT, max_steps=1)
    actual = np.asarray(runtime.state_global("fluid")).reshape(initial.shape)
    np.testing.assert_allclose(actual, oracle_step(initial, velocities), rtol=0., atol=5.e-14)
    np.testing.assert_allclose(actual.sum(axis=(1, 2)), initial.sum(axis=(1, 2)),
                               rtol=0., atol=5.e-13)
    assert np.max(np.abs(actual - initial)) > 1.e-4
    assert report.accepted_steps == runtime.macro_step() == 1
    assert runtime.time() == pytest.approx(DT, rel=0., abs=1.e-16)
