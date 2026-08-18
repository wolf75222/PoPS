"""Public 1-d periodic cyclotron-source Case for TM-04.

Keeps the in-memory exact rotation, explicit Euler, and implicit midpoint.
The public Case authors the cell-local cyclotron source
(dux/dt, duy/dt) = (ωc uy, -ωc ux) on a 1-d periodic interval. An inert
zero flux satisfies FiniteVolume; the rotation is the source, not a private
integrator. Implicit midpoint stays in memory: there is no public Cayley
factory. Optional native compile/bind/run. ``pops.run`` is used only inside
``run_native``. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.math import ddt, div
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import (
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

OMEGA_C = float(_exact.OMEGA_C)
N_CELLS = 1
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int, dt: float) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt


def exact_advance(u, t, *, omega_c=_exact.OMEGA_C):
    """Return the closed-form rotation of u by θ = ωc t."""
    return _exact.exact_advance(u, t, omega_c=omega_c)


def explicit_euler(u, dt, *, omega_c=_exact.OMEGA_C):
    """One explicit Euler step of du/dt = A u. Speed grows as sqrt(1+(ωc Δt)²)."""
    state = np.asarray(u, dtype=np.float64)
    return state + float(dt) * (_exact.cyclotron_matrix(omega_c=omega_c) @ state)


def implicit_midpoint(u, dt, *, omega_c=_exact.OMEGA_C):
    """One implicit midpoint (Cayley) step of du/dt = A u. Speed is conserved."""
    state = np.asarray(u, dtype=np.float64)
    half = 0.5 * float(dt)
    generator = _exact.cyclotron_matrix(omega_c=omega_c)
    identity = np.eye(2, dtype=np.float64)
    return np.linalg.solve(identity - half * generator, (identity + half * generator) @ state)


def _frame():
    return CartesianDomain("tm04_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(dt, *, n_cells: int = N_CELLS) -> _Authoring:
    count = int(n_cells)
    if count <= 0:
        raise ValueError("n_cells must be positive")
    step = float(dt)
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm04_larmor", frame=frame)
    state = model.state(
        "U",
        components=("ux", "uy"),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    ux, uy = state
    zero_flux = (0.0 * ux, 0.0 * uy)
    flux = model.flux(
        "inert_flux",
        frame=frame,
        state=state,
        components={x_axis: zero_flux},
        waves={x_axis: (0.0, 0.0)},
    )
    source = model.source("cyclotron", on=state, value=(OMEGA_C * uy, -OMEGA_C * ux))
    rate = model.rate("larmor_rate", equation=ddt(state) == -div(flux) + source)
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
    case = pops.Case("tm04_larmor")
    velocity = case.block("velocity", model=model, states=(state,))
    instance = velocity[state]
    case.numerics(numerics, block=velocity)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(FixedDt(step))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count, dt=step)


def build_case(dt, *, n_cells: int = N_CELLS) -> pops.Case:
    """Author the 1-d periodic cyclotron-source Case. Does not compile or run."""
    return _author(dt, n_cells=n_cells).case


def resolve_plan(dt, *, n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(dt, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(dt, t_end=1.0, *, n_cells: int = N_CELLS):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(dt, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.broadcast_to(
        np.asarray(_exact.U0, dtype=np.float64)[:, np.newaxis],
        (2, authored.n_cells),
    ).copy()
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("velocity"), dtype=np.float64)
    return np.ravel(field)
