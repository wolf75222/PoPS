"""ADC-942 declared Dim2 CPU AMR transport-diffusion qualification matrix."""

from __future__ import annotations

import math as pmath
from pathlib import Path

import numpy as np
import pops
import pytest
from pops import math
from pops.analytic import cos as analytic_cos
from pops.analytic import sin as analytic_sin
from pops.analytic import x as analytic_x
from pops.analytic import y as analytic_y
from pops.amr import (
    AMRClockRelation,
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
from pops.lib.initial import Analytic
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics.diffusion import DiffusiveBoundary
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.amr_snapshots import (
    composite_active_block_state,
    composite_active_mask,
)
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
ROOT = Path(__file__).resolve().parents[4]
REFINEMENTS = (16, 32, 64)
VELOCITY = (0.35, -0.2)
DIFFUSIVITY = 0.05
FINAL_TIME = 1.0e-2


def _stable_dt(n):
    fine_n = 2 * n
    frequency = 4 * DIFFUSIVITY * fine_n**2 + sum(map(abs, VELOCITY)) * fine_n
    return 0.4 / frequency


def _author(n, dt, *, physical=False, cxx=None):
    frame = Rectangle("qualified-amr-diffusion", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("qualified-amr-diffusion-model", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    boundaries = None
    if physical:
        boundaries = tuple(
            DiffusiveBoundary(axis, side, "value", 1.0, (1.0, 0.0))
            if axis == 0
            else DiffusiveBoundary(axis, side, "conormal", 0.0)
            for axis in range(2)
            for side in ("lower", "upper")
        )
    diffusion = model.diffusive_flux(
        "diffusion", state=state, value=DIFFUSIVITY * math.grad(u), boundaries=boundaries
    )
    transport = None
    equation = div(diffusion)
    method = None
    if not physical:
        transport = model.flux(
            "transport",
            frame=frame,
            state=state,
            components={
                axis: (speed * u,) for axis, speed in zip(frame.axes, VELOCITY, strict=True)
            },
            waves={axis: (speed,) for axis, speed in zip(frame.axes, VELOCITY, strict=True)},
        )
        equation = -div(transport) + equation
        method = FiniteVolume(
            flux=transport,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        )
    rate = model.rate("transport-diffusion", equation=ddt(state) == equation)
    case = pops.Case("qualified-amr-diffusion-case")
    block = case.block("heat", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=diffusion, transport=method))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    if physical:
        profile = 1.0 + analytic_x(frame)
        threshold_value = 1.5
    else:
        profile = 1.0 + 0.2 * analytic_sin(2 * pmath.pi * analytic_x(frame)) * analytic_cos(
            2 * pmath.pi * analytic_y(frame)
        )
        threshold_value = 1.04
    case.initials.add(
        InitialCondition(
            state=block_state,
            value=Analytic(frame=frame, components=(profile,)),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("refine-threshold", default=threshold_value))
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame, cells=(n, n), periodic=None if physical else PeriodicAxes(frame.axes)
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block_state) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(4, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    compile_options = {"include": str(ROOT / "include")}
    if cxx is not None:
        compile_options["cxx"] = cxx
    return pops.resolve(
        pops.validate(case), layout=layout, backend=Production(), compile_options=compile_options
    )


def _bind(artifact):
    return pops.bind(
        artifact,
        resources={"execution_context": artifact_execution_context(artifact)},
    )


def _compile(resolved, *, route):
    """Publish once when this same matrix is launched under two MPI ranks."""
    from pops import _pops

    comm = _pops.mpi_world()
    if int(comm.size) == 1:
        return pops.compile(resolved)
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

    return compile_resolved_plan_once(comm, resolved, route=route, compile_artifact=pops.compile)


def _collective_path(path):
    from pops import _pops

    comm = _pops.mpi_world()
    if int(comm.size) == 1:
        return path
    from pops._native_collectives import broadcast_value

    shared = broadcast_value(comm, str(path) if int(comm.rank) == 0 else None, root=0)
    return Path(shared)


def _mass(runtime, n):
    return sum(
        float(composite_active_block_state(runtime, "heat", level, refinement_ratio=2).sum())
        / (n * 2**level) ** 2
        for level in range(runtime.n_levels())
    )


def _periodic_l2(runtime, n, time):
    squared = 0.0
    measure = 0.0
    amplitude = 0.2 * np.exp(-8 * np.pi**2 * DIFFUSIVITY * time)
    for level in range(runtime.n_levels()):
        level_n = n * 2**level
        coordinates = (np.arange(level_n) + 0.5) / level_n
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
        average = np.sinc(1 / level_n) ** 2
        exact = 1.0 + amplitude * average * np.sin(2 * np.pi * (xx - VELOCITY[0] * time)) * np.cos(
            2 * np.pi * (yy - VELOCITY[1] * time)
        )
        actual = np.asarray(runtime.block_level_state_global("heat", level), dtype=np.float64)
        actual = actual.reshape((1, level_n, level_n))[0]
        active = composite_active_mask(runtime, level, refinement_ratio=2)
        cell_measure = 1.0 / level_n**2
        squared += float(np.sum((actual[active] - exact[active]) ** 2)) * cell_measure
        measure += float(active.sum()) * cell_measure
    assert abs(measure - 1.0) < 2.0e-14
    return pmath.sqrt(squared)


@pytest.mark.parametrize("n", REFINEMENTS)
def test_periodic_full_refinement_restart_and_face_inventory(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
    tmp_path,
    n,
):
    del isolated_native_cache, kokkos_root
    dt = _stable_dt(n)
    resolved = _author(n, dt, cxx=native_cxx)
    artifact = _compile(resolved, route="periodic-n%d" % n)
    continuous = _bind(artifact)
    restarted_source = _bind(artifact)
    initial_mass = _mass(continuous, n)
    steps = pmath.ceil(FINAL_TIME / dt)
    half_steps = steps // 2
    half_time = half_steps * dt
    pops.run(continuous, t_end=FINAL_TIME, max_steps=steps + 1, console=False)
    pops.run(restarted_source, t_end=half_time, max_steps=half_steps, console=False)
    checkpoint = restarted_source.checkpoint(_collective_path(tmp_path / ("diffusion-n%d" % n)))
    restarted = _bind(artifact)
    restarted.restart(checkpoint)
    pops.run(restarted, t_end=FINAL_TIME, max_steps=steps + 1, console=False)
    assert abs(_mass(continuous, n) - initial_mass) < 8.0e-11
    assert abs(_mass(restarted, n) - initial_mass) < 8.0e-11
    assert _periodic_l2(continuous, n, FINAL_TIME) < 0.025 / n
    for level in range(continuous.n_levels()):
        np.testing.assert_array_equal(
            continuous.block_level_state_global("heat", level),
            restarted.block_level_state_global("heat", level),
        )
    manifest = "\n".join(
        "/".join(map(str, row)) for row in continuous._executor.program_flux_ledger_manifest()
    )
    assert "provider/1" in manifest and "provider/4" in manifest


def test_physical_diffusion_boundary_is_steady_across_refinement_and_subcycling(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    n = REFINEMENTS[0]
    resolved = _author(n, _stable_dt(n), physical=True, cxx=native_cxx)
    artifact = _compile(resolved, route="physical-n%d" % n)
    runtime = _bind(artifact)
    before = tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).copy()
        for level in range(runtime.n_levels())
    )
    pops.run(runtime, t_end=FINAL_TIME, max_steps=128, console=False)
    for level, expected in enumerate(before):
        np.testing.assert_allclose(
            runtime.block_level_state_global("heat", level), expected, rtol=0.0, atol=3.0e-12
        )


def test_combined_local_bound_rejection_leaks_no_accepted_state_and_retry_succeeds(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    n = REFINEMENTS[0]
    stable = _stable_dt(n)
    unstable = 3.0 * stable
    resolved = _author(n, unstable, cxx=native_cxx)
    runtime = _bind(_compile(resolved, route="rejection-n%d" % n))
    before = tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).copy()
        for level in range(runtime.n_levels())
    )
    with pytest.raises(RuntimeError, match="combined_transport_diffusion_stability|rejected"):
        pops.run(runtime, t_end=unstable, max_steps=1, console=False)
    assert runtime.time() == 0 and runtime.macro_step() == 0
    assert not runtime._executor.program_flux_ledger_manifest()
    assert not runtime._executor._program_exchange_records()
    for level, expected in enumerate(before):
        np.testing.assert_array_equal(runtime.block_level_state_global("heat", level), expected)
    report = pops.run(runtime, t_end=stable, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == 0
