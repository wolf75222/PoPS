"""SSPRK3 order discrimination through an authored, compiled Program.

The real refined mono/multi-block conservation matrix lives in
``test_amr_explicit_family``.  This companion keeps the independent temporal-order evidence: the
public SSPRK3 factory is graph-identical to the generic SSPRK3 tableau route and its compiled
trajectory is closer than Forward Euler to an SSPRK3 small-step reference.
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
from pops.time._methods.properties import certify_program_graph
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


ROOT = Path(__file__).resolve().parents[4]
N = 32
POPS_PROCESS_TIMEOUT = 900


def _authoring(method: str, dt: float, *, cxx: str | None):
    frame = Rectangle("ssprk3-order-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    x_axis, y_axis = frame.axes
    model = pops.Model("ssprk3-order-transport", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.8 * rho,), y_axis: (-0.35 * rho,)},
        waves={x_axis: (0.8 + 0.0 * rho,), y_axis: (0.35 + 0.0 * rho,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("ssprk3-order-%s" % method)
    block = case.block("tracer", model, states=(state,))
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
    if method == "euler":
        program = libtime.ForwardEuler(instance, rate=rate)
    elif method == "ssprk3":
        program = libtime.SSPRK3(instance, rate=rate)
    else:
        raise ValueError("unsupported method %r" % method)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    options = None
    if cxx is not None:
        options = {"include": str(ROOT / "include"), "cxx": cxx}
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
        compile_options=options,
    )
    return resolved, instance, rate


def _initial_state() -> np.ndarray:
    axis = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(axis, axis, indexing="xy")
    rho = 1.0 + 0.25 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    return np.ascontiguousarray(rho[None, ...])


def _run(method: str, dt: float, nsteps: int, cxx: str) -> np.ndarray:
    resolved, instance, _rate = _authoring(method, dt, cxx=cxx)
    simulation = pops.bind(
        pops.compile(resolved),
        initial_values={instance: _initial_state()},
    )
    report = pops.run(simulation, t_end=nsteps * dt, max_steps=nsteps)
    assert report.accepted_steps == simulation.macro_step() == nsteps
    assert simulation.time() == pytest.approx(nsteps * dt, rel=0.0, abs=2.0e-15)
    result = np.asarray(simulation.block_level_state_global("tracer", 0), dtype=np.float64).reshape(
        (1, N, N)
    )[0]
    assert np.isfinite(result).all()
    return result


def _check_typed_graph_authority() -> None:
    resolved, instance, rate = _authoring("ssprk3", 1.0e-3, cxx=None)
    generic = libtime.RungeKutta(
        routes=(libtime.RungeKuttaRoute(instance, rate),),
        tableau=libtime.SSPRK3_TABLEAU,
    )
    generic.step_strategy(FixedDt(1.0e-3))
    assert generic.ir_nodes() == resolved.time.ir_nodes()
    certificate = certify_program_graph(resolved.time.to_graph())
    assert certificate.properties.order == 3
    assert len([node for node in resolved.time.ir_nodes() if node["op"] == "rhs"]) == 3


def _check_order(cxx: str) -> None:
    dt = 8.0e-3
    nsteps = 8
    refinement = 8
    euler = _run("euler", dt, nsteps, cxx)
    ssprk3 = _run("ssprk3", dt, nsteps, cxx)
    reference = _run("ssprk3", dt / refinement, nsteps * refinement, cxx)
    euler_error = float(np.mean(np.abs(euler - reference)))
    ssprk3_error = float(np.mean(np.abs(ssprk3 - reference)))
    assert euler_error > 1.0e-12
    assert ssprk3_error < euler_error, (
        "compiled SSPRK3 did not improve on Forward Euler: %.3e >= %.3e"
        % (ssprk3_error, euler_error)
    )


def main() -> None:
    _check_typed_graph_authority()
    cxx = default_cxx()
    missing = missing_native_compile_requirement(repo_include(), cxx)
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    _check_order(cxx)


if __name__ == "__main__":
    main()
