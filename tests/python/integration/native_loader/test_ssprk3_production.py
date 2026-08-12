#!/usr/bin/env python3
"""Production SSPRK3 executes from the typed Case/Program authority.

Two independently authored public Programs -- the SSPRK3 convenience factory and the generic
``RungeKuttaRoute`` plus ``SSPRK3_TABLEAU`` form -- lower to the same executable graph and produce
bit-identical native trajectories.  SSPRK2 remains a numerical discriminator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


ROOT = Path(__file__).resolve().parents[4]
N = 32
DT = 1.0e-3
NSTEPS = 12
POPS_PROCESS_TIMEOUT = 900


def _resolved(method: str, cxx: str):
    frame = Rectangle("ssprk3-production-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = pops.Model("ssprk3-production-model", frame=frame)
    state = model.state("U", components=("rho", "tracer"))
    rho, tracer = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (0.5 * rho * rho, 0.45 * tracer),
            y_axis: (-0.2 * rho, 0.15 * tracer),
        },
        waves={
            x_axis: (rho, 0.45 + 0.0 * tracer),
            y_axis: (0.2 + 0.0 * rho, 0.15 + 0.0 * tracer),
        },
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    case = pops.Case("ssprk3-production-case")
    block = case.block("gas", model, states=(state,))
    instance = block[state]
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
    if method == "ssprk3":
        program = libtime.SSPRK3(instance, rate=rate)
    elif method == "ssprk3-route":
        program = libtime.RungeKutta(
            routes=(libtime.RungeKuttaRoute(instance, rate),),
            tableau=libtime.SSPRK3_TABLEAU,
        )
    elif method == "ssprk2":
        program = libtime.SSPRK2(instance, rate=rate)
    else:
        raise ValueError("unsupported typed method %r" % method)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=Uniform(
            CartesianGrid(
                frame=frame,
                cells=(N, N),
                periodic=PeriodicAxes(frame.axes),
            )
        ),
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": cxx},
    )
    return resolved, instance


def _initial_state() -> np.ndarray:
    axis = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(axis, axis, indexing="xy")
    rho = 1.0 + 0.3 * np.exp(-70.0 * ((x - 0.42) ** 2 + (y - 0.54) ** 2))
    tracer = 0.8 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    return np.ascontiguousarray(np.stack((rho, tracer)))


def _compile_and_run(method: str, cxx: str):
    resolved, instance = _resolved(method, cxx)
    artifact = pops.compile(resolved)
    artifact.verify()
    simulation = pops.bind(artifact, initial_values={instance: _initial_state()})
    report = pops.run(simulation, t_end=NSTEPS * DT, max_steps=NSTEPS)
    assert report.accepted_steps == simulation.macro_step() == NSTEPS
    assert simulation.program_report().program_hash == artifact.program_hash
    state = np.asarray(simulation.block_level_state_global("gas", 0), dtype=np.float64).reshape(
        (2, N, N)
    )
    assert np.isfinite(state).all() and float(state[0].min()) > 0.0
    return resolved, artifact, state


def main() -> None:
    cxx = default_cxx()
    missing = missing_native_compile_requirement(repo_include(), cxx)
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)

    preset, preset_artifact, preset_state = _compile_and_run("ssprk3", cxx)
    routed, routed_artifact, routed_state = _compile_and_run("ssprk3-route", cxx)
    _ssprk2, ssprk2_artifact, ssprk2_state = _compile_and_run("ssprk2", cxx)

    assert preset.time.ir_nodes() == routed.time.ir_nodes()
    assert preset_artifact.program_hash != ssprk2_artifact.program_hash
    assert routed_artifact.program_hash != ssprk2_artifact.program_hash
    np.testing.assert_array_equal(preset_state, routed_state)
    difference = float(np.max(np.abs(preset_state - ssprk2_state)))
    assert difference > 0.0, "compiled SSPRK3 trajectory unexpectedly equals SSPRK2"


if __name__ == "__main__":
    main()
