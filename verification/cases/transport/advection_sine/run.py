"""Public 1-d periodic scalar advection Case for TR-01.

Optional native compile/bind/run. Does not call ROMEO.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.lib.initial import Analytic
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import resolve_case, uniform_periodic_layout

A = 1.0
CFL = 0.45
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def _exact_module():
    path = Path(__file__).with_name("exact.py")
    spec = importlib.util.spec_from_file_location("tr01_advection_sine_exact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame():
    return CartesianDomain("unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _author(n_cells: int) -> _Authoring:
    count = int(n_cells)
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tr01_advection_sine", frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: A})
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (A * q,)},
        waves={x_axis: (A,)},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case("tr01_advection_sine")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    from pops.analytic import sin, x as analytic_x

    wave = 2.0 * np.pi * float(_exact_module().K)
    exact = _exact_module()
    case.initials.add(
        InitialCondition(
            state=instance,
            value=Analytic(
                frame=frame,
                components=(
                    float(exact.Q0) + float(exact.EPS) * sin(wave * analytic_x(frame)),
                ),
            ),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells) -> pops.Case:
    """Author the 1-d periodic scalar advection Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells, t_end=1.0):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    from verification.pops_verify.case_authoring import bind_public

    simulation = bind_public(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    return np.ravel(field)
