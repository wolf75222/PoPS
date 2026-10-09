"""Native reception prepared for exact zero diagonal diffusion coefficients.

Run only after the isolated checkout has been rebuilt into its own installed
environment. The tests below must never import an older native extension.
"""
from __future__ import annotations

import math

import numpy as np
import pops
import pytest

from pops.analytic import cos as analytic_cos
from pops.analytic import x as analytic_x
from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                      AMRTagging, AMRTransfer, Buffer, ConflictPolicy,
                      EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.lib.time import ForwardEuler
from pops.math import CoeffGradient, ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.params import RuntimeParam
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.amr_snapshots import composite_active_block_state
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _authored(n=16, dt=.12, *, coefficient="x_only", y_boundary="periodic",
              x_conormal=False, amr=False):
    frame = Rectangle("semidefinite_square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("semidefinite_diffusion", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    if coefficient == "x_only":
        dx = .01
    elif coefficient == "zero":
        dx = 0.
    elif coefficient == "dynamic":
        dx = u
    elif coefficient == "nonfinite":
        dx = 1 / (u - 1)
    elif coefficient == "overflow":
        dx = u
    else:
        raise ValueError(coefficient)
    physical = None
    if x_conormal:
        if y_boundary != "periodic":
            raise ValueError("overflow witness uses x conormal and periodic y")
        physical = (
            DiffusiveBoundary(0, "lower", "conormal", 0.),
            DiffusiveBoundary(0, "upper", "conormal", 0.),
            DiffusiveBoundary(1, "lower", "periodic"),
            DiffusiveBoundary(1, "upper", "periodic"),
        )
    elif y_boundary != "periodic":
        y_kind = "conormal" if y_boundary in ("conormal", "incompatible") else "value"
        upper_value = 1. if y_boundary == "incompatible" else 0.
        physical = (
            DiffusiveBoundary(0, "lower", "periodic"),
            DiffusiveBoundary(0, "upper", "periodic"),
            DiffusiveBoundary(1, "lower", y_kind, 0.),
            DiffusiveBoundary(1, "upper", y_kind, upper_value),
        )
    flux = model.diffusive_flux("diffusion", state=state,
                                value=CoeffGradient(0. * u if coefficient == "overflow" else u,
                                                    ((dx, 0.), (0., 0.))),
                                boundaries=physical)
    rate = model.rate("balance", equation=ddt(state) == div(flux))
    case = pops.Case("semidefinite_%s_%s" % (coefficient, y_boundary))
    block = case.block("heat", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=flux))
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    periodic = (PeriodicAxes((frame.y,)) if x_conormal else
                PeriodicAxes(frame.axes) if y_boundary == "periodic" else
                PeriodicAxes((frame.x,)))
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=periodic)
    if not amr:
        return pops.resolve(pops.validate(case), layout=Uniform(grid))
    if y_boundary != "periodic" or coefficient != "x_only":
        raise ValueError("AMR witness uses periodic exact x-only diffusion")
    case.initials.add(InitialCondition(
        state=block[state],
        value=Analytic(frame=frame, components=(1 + .2 * analytic_cos(2 * math.pi * analytic_x(frame)),)),
        projection=ConservativeCellAverage(),
    ))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("refine-threshold", default=1.04))
    layout = AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    return pops.resolve(pops.validate(case), layout=layout)


def _initial(n):
    x = (np.arange(n) + .5) / n
    xx, yy = np.meshgrid(x, x, indexing="xy")
    return np.ascontiguousarray((1 + .1 * np.cos(2 * np.pi * xx) +
                                 .07 * np.cos(2 * np.pi * yy))[None])


def _bind_uniform(resolved, initial):
    artifact = pops.compile(resolved)
    assert artifact.resolved_dimension == 2
    runtime = pops.bind(artifact, initial_state={"heat": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    return runtime


def test_periodic_x_only_diffusion_has_exact_one_axis_stencil_and_frequency():
    n, dt, dx = 16, .12, .01
    assert dt > 1 / (4 * dx * n * n) and dt < 1 / (2 * dx * n * n)
    initial = _initial(n)
    runtime = _bind_uniform(_authored(n, dt), initial)
    report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
    expected = initial + dt * dx * n * n * (
        np.roll(initial, 1, axis=-1) - 2 * initial + np.roll(initial, -1, axis=-1))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2.e-12)
    assert abs(actual.mean() - initial.mean()) < 2.e-13
    # The y-only Fourier mode must remain, while the x mode diffuses.
    np.testing.assert_allclose(actual.mean(axis=-1), initial.mean(axis=-1), rtol=0., atol=2.e-12)


def test_all_zero_tensor_has_zero_flux_and_zero_explicit_frequency():
    n, dt = 8, 1.
    initial = _initial(n)
    runtime = _bind_uniform(_authored(n, dt, coefficient="zero"), initial)
    report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
    np.testing.assert_array_equal(actual, initial)


@pytest.mark.parametrize("kind", ("value", "conormal"))
def test_zero_normal_value_or_zero_conormal_has_no_y_face_contribution(kind):
    n, dt, dx = 8, .05, .01
    initial = _initial(n)
    runtime = _bind_uniform(_authored(n, dt, y_boundary=kind), initial)
    report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
    expected = initial + dt * dx * n * n * (
        np.roll(initial, 1, axis=-1) - 2 * initial + np.roll(initial, -1, axis=-1))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2.e-12)


@pytest.mark.parametrize("coefficient, initial", (
    ("dynamic", "negative"), ("nonfinite", "singular")))
def test_negative_or_nonfinite_dynamic_coefficient_rolls_back(coefficient, initial):
    n, dt = 8, .01
    values = np.ones((1, n, n), dtype=np.float64)
    if initial == "negative":
        values[0, n // 2, n // 2] = -1.
    runtime = _bind_uniform(_authored(n, dt, coefficient=coefficient), values)
    before = np.asarray(runtime.state_global("heat")).reshape(values.shape).copy()
    with pytest.raises(RuntimeError, match="diffusive|pointwise|rejected"):
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
    after = np.asarray(runtime.state_global("heat")).reshape(values.shape)
    np.testing.assert_array_equal(before, values)
    np.testing.assert_array_equal(after, values)
    assert runtime.time() == 0 and runtime.macro_step() == 0


def test_nonzero_conormal_on_zero_normal_axis_rejects_before_acceptance():
    n, dt = 8, .01
    initial = _initial(n)
    runtime = _bind_uniform(_authored(n, dt, y_boundary="incompatible"), initial)
    before = np.asarray(runtime.state_global("heat")).reshape(initial.shape).copy()
    with pytest.raises(RuntimeError, match="diffusive|pointwise|rejected"):
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
    after = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
    np.testing.assert_array_equal(before, initial)
    np.testing.assert_array_equal(after, initial)
    assert runtime.time() == 0 and runtime.macro_step() == 0


def test_finite_cell_coefficients_overflowing_at_conormal_face_are_rejected():
    n, dt = 8, .01
    initial = np.zeros((1, n, n), dtype=np.float64)
    initial[0, :, 0] = 1.6e308
    assert np.isfinite(initial).all()
    assert not math.isfinite(1.5 * float(initial[0, 0, 0]))
    runtime = _bind_uniform(_authored(n, dt, coefficient="overflow", x_conormal=True), initial)
    before = np.asarray(runtime.state_global("heat")).reshape(initial.shape).copy()
    with pytest.raises(RuntimeError, match="diffusive|pointwise|rejected"):
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
    after = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
    np.testing.assert_array_equal(before, initial)
    np.testing.assert_array_equal(after, initial)
    assert runtime.time() == 0 and runtime.macro_step() == 0


def _composite_mass(runtime, n):
    return sum(float(composite_active_block_state(runtime, "heat", level,
                                                   refinement_ratio=2).sum()) /
               (n * 2**level)**2 for level in range(runtime.n_levels()))


def test_periodic_amr_x_only_diffusion_keeps_composite_mass_and_refines():
    n, dt = 16, .005
    artifact = pops.compile(_authored(n, dt, amr=True))
    runtime = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    initial_mass = _composite_mass(runtime, n)
    report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert report.accepted_steps == 1
    assert runtime.n_levels() == 2
    assert abs(_composite_mass(runtime, n) - initial_mass) < 2.e-11
