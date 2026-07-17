#!/usr/bin/env python3
"""ADC-514: bound per-block RUNTIME parameters on the AMR hierarchy.

A production AMR block (add_equation with a CompiledModel backend='production', target='amr_system')
whose model declares ``pops.params.RuntimeParam`` receives one complete parameter vector from the
artifact BindSchema at installation. The final runtime deliberately has no post-install
``AmrSystem.set_block_params`` mutation seam: values are authenticated bind inputs and flow into the
block's transport / source / elliptic bricks through the compiled package carrier.

This test asserts (Kokkos-gated, needs a compiler + a visible Kokkos to build + run the .so):

  1) the SAME compiled runtime-param AMR package RUNS with distinct complete bind vectors, and
     ``speed=4`` DIFFERS from ``speed=1`` (the speed enters the transport flux);
  2) BIT-IDENTITY: binding the same complete vector twice reproduces the trajectory byte-for-byte;
  3) the BindSchema rejects untyped string keys instead of accepting an ad-hoc parameter namespace.

Native prerequisites are expressed by pytest markers and fixtures. Any compile, bind or run failure
is a hard test failure rather than a self-skip.
"""
from pathlib import Path

import numpy as np
import pops
import pytest
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Gaussian
from pops.lib.time import SSPRK2
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every

ROOT = Path(__file__).resolve().parents[4]
N = 16
NSTEPS = 4
DT = 5.0e-4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _resolved_runtime_parameter_case(native_cxx):
    """Build the final typed Case/AMR plan and return its owner-qualified speed parameter."""
    frame = Rectangle(
        "adc514-runtime-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("adc514-runtime-advection", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    speed_declaration = model.param(RuntimeParam("speed", default=1.0))
    speed = model.value(speed_declaration)
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (speed * rho,), y_axis: (0.25 * rho,)},
        waves={
            x_axis: (speed + 0.0 * rho,),
            y_axis: (0.25 + 0.0 * rho,),
        },
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))

    case = pops.Case("adc514-runtime-case")
    block = case.block("gas", model)
    state_instance = block[state]
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
    case.numerics(numerics, block=block)
    program = SSPRK2(state_instance, rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(
        state=state_instance,
        value=Gaussian(
            frame=frame,
            center={x_axis: 0.4, y_axis: 0.5},
            background=1.0,
            amplitude=0.3,
            inverse_width=70.0,
        ),
        projection=ConservativeCellAverage(),
    ))
    refine_threshold = case.param(RuntimeParam("refine_threshold", default=1.1))
    transfer = AMRTransfer()
    transfer.state(state_instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(state_instance) > case.value(refine_threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    validated = pops.validate(case)
    bound_speed = validated.resolve(speed_declaration)
    plan = pops.resolve(
        validated,
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    return plan, bound_speed


def _run_bound(artifact, parameter, value):
    simulation = pops.bind(artifact, params={parameter: value})
    report = pops.run(simulation, t_end=NSTEPS * DT, max_steps=NSTEPS)
    assert report.accepted_steps == NSTEPS
    values = np.asarray(
        simulation.block_level_state_global("gas", 0), dtype=np.float64)
    return simulation, values


def test_amr_bound_params_change_trajectory_and_repeat_bit_identically(
    native_cxx, isolated_native_cache, kokkos_root,
):
    """One artifact bound with distinct typed values changes the AMR trajectory without recompiling."""
    del isolated_native_cache, kokkos_root
    plan, parameter = _resolved_runtime_parameter_case(native_cxx)
    artifact = pops.compile(plan)
    artifact.verify()

    first, base = _run_bound(artifact, parameter, 1.0)
    repeated, same = _run_bound(artifact, parameter, 1.0)
    changed_run, changed = _run_bound(artifact, parameter, 4.0)

    assert first.bind_identity == repeated.bind_identity
    assert first.bind_identity != changed_run.bind_identity
    np.testing.assert_array_equal(base, same)
    assert changed.shape == base.shape
    assert not np.array_equal(changed, base)
    assert np.all(np.isfinite(changed))
    assert first.n_levels() == repeated.n_levels() == changed_run.n_levels() == 2


def test_amr_runtime_params_reject_untyped_bind_keys(native_cxx):
    """BindSchema accepts owner-qualified ParamHandle keys, never ad-hoc parameter names."""
    plan, _ = _resolved_runtime_parameter_case(native_cxx)
    with pytest.raises((TypeError, ValueError)) as caught:
        plan.bind_schema.resolve_bind(
            {"speed": 1.0}, compile_values=plan.compile_values)
    message = str(caught.value)
    assert "ParamHandle" in message or "parameter" in message.lower()
