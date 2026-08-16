#!/usr/bin/env python3
"""Forward Euler is the exact stage operator of the typed SSPRK2 Program.

The semantic proof is authored through public symbolic Cases.  The SSPRK2 factory and the generic
``RungeKuttaRoute`` plus ``SSPRK2_TABLEAU`` form have the same executable graph, while a production
run retains the Shu--Osher identity and discriminates one Forward-Euler step from SSPRK2.
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
N = 24
DT = 2.0e-3
POPS_PROCESS_TIMEOUT = 900


def _resolved(method: str, *, cxx: str | None):
    frame = Rectangle("time-euler-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("time-euler-transport", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.5 * rho * rho,), y_axis: (-0.3 * rho,)},
        waves={x_axis: (rho,), y_axis: (0.3 + 0.0 * rho,)},
    )
    rate = model.rate("transport_rate", equation=ddt(state) == -div(flux))
    case = pops.Case("time-euler-case")
    block = case.block("ions", model, states=(state,))
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
    elif method == "ssprk2":
        program = libtime.SSPRK2(instance, rate=rate)
    elif method == "ssprk2-route":
        program = libtime.RungeKutta(
            routes=(libtime.RungeKuttaRoute(instance, rate),),
            tableau=libtime.SSPRK2_TABLEAU,
        )
    else:
        raise ValueError("unsupported method %r" % method)
    program.step_strategy(FixedDt(DT))
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
    return resolved, instance


def _initial_state() -> np.ndarray:
    axis = (np.arange(N, dtype=np.float64) + 0.5) / N
    x, y = np.meshgrid(axis, axis, indexing="xy")
    rho = 1.0 + 0.3 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    return np.ascontiguousarray(rho[None, ...])


def _run(method: str, steps: int, cxx: str) -> np.ndarray:
    resolved, instance = _resolved(method, cxx=cxx)
    simulation = pops.bind(
        pops.compile(resolved),
        initial_values={instance: _initial_state()},
    )
    report = pops.run(simulation, t_end=steps * DT, max_steps=steps)
    assert report.accepted_steps == simulation.macro_step() == steps
    return np.asarray(simulation.block_level_state_global("ions", 0), dtype=np.float64).reshape(
        (1, N, N)
    )[0]


def _check_graph_contract() -> None:
    preset, _instance = _resolved("ssprk2", cxx=None)
    routed, _routed_instance = _resolved("ssprk2-route", cxx=None)
    euler, _euler_instance = _resolved("euler", cxx=None)
    assert preset.time.ir_nodes() == routed.time.ir_nodes()
    assert certify_program_graph(preset.time.to_graph()).properties.order == 2
    assert certify_program_graph(euler.time.to_graph()).properties.order == 1
    assert len([node for node in preset.time.ir_nodes() if node["op"] == "rhs"]) == 2
    assert len([node for node in euler.time.ir_nodes() if node["op"] == "rhs"]) == 1


def _check_native_shu_osher_identity(cxx: str) -> None:
    initial = _initial_state()[0]
    ssprk2 = _run("ssprk2", 1, cxx)
    euler_once = _run("euler", 1, cxx)
    euler_twice = _run("euler", 2, cxx)
    reference = 0.5 * initial + 0.5 * euler_twice
    error = float(np.max(np.abs(ssprk2 - reference)))
    assert np.array_equal(ssprk2, reference) or error < 1.0e-15
    assert float(np.max(np.abs(euler_once - ssprk2))) > 1.0e-8


def main() -> None:
    _check_graph_contract()
    cxx = default_cxx()
    missing = missing_native_compile_requirement(repo_include(), cxx)
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    _check_native_shu_osher_identity(cxx)


if __name__ == "__main__":
    main()
