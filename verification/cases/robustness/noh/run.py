"""1-d planar Noh authoring and native run.

Initial conditions come from ``primitives_1d(..., t=0)`` with a tiny
numerical pressure floor. ``build_case`` / ``resolve_plan`` author a
public 1-d Euler Case (Rusanov, FirstOrder, SSPRK2, ``positivity_floor``).
The origin is a public ``SlipWall`` and the outer face is ``Outflow``.
The oracle in ``exact.py`` stays p=0. ``run_native`` compiles, binds, and
advances the Case.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.case_authoring import wall_and_outflow_boundary_set

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

GAMMA = float(_EXACT.GAMMA)
N_CELLS = 64
DOMAIN_LOWER = -1.0
DOMAIN_UPPER = 1.0
CFL = 0.4
MAX_STEPS = 100_000
# Zhang-Shu density floor on reconstructed faces.
POSITIVITY_FLOOR = 1e-8
# Numerical IC only. exact.py keeps the cold p=0 oracle.
NUMERICAL_PRESSURE_FLOOR = 1e-8


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on [-1, 1]."""
    count = int(n_cells)
    width = (DOMAIN_UPPER - DOMAIN_LOWER) / count
    return DOMAIN_LOWER + (np.arange(count, dtype=np.float64) + 0.5) * width, width


def initial_primitives(n_cells: int = N_CELLS):
    """Numerical primitive IC W(x,0). Shape (3, n).

    ``exact.py`` keeps p=0. The native IC floors pressure so cell-centered
    faces stay strictly positive and ``sqrt(γ p / ρ)`` stays finite.
    """
    centers, _ = cell_centers(n_cells)
    primitives = np.array(_EXACT.primitives_1d(centers, 0.0), dtype=np.float64, copy=True)
    primitives[2] = np.maximum(primitives[2], NUMERICAL_PRESSURE_FLOOR)
    return primitives


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from the floored numerical primitives."""
    return _EXACT.primitives_to_conserved_1d(initial_primitives(n_cells))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain(
        "rb06-line", lower=(DOMAIN_LOWER,), upper=(DOMAIN_UPPER,)
    ).frame(Cartesian1D())


def _author(n_cells: int = N_CELLS) -> _Authoring:
    import pops
    import pops.lib.time as libtime
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.math import ddt, div, sqrt
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Energy, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("rb06-euler", frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "E"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "E": Energy(),
        },
    )
    rho, momentum, energy = state
    velocity = momentum / rho
    pressure = (GAMMA - 1.0) * (energy - 0.5 * rho * velocity * velocity)
    sound = sqrt(GAMMA * pressure / rho)
    flux = model.flux(
        "euler",
        frame=frame,
        state=state,
        components={
            x_axis: (
                momentum,
                momentum * velocity + pressure,
                velocity * (energy + pressure),
            ),
        },
        waves={x_axis: (velocity - sound, velocity, velocity + sound)},
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    case = pops.Case("rb06-noh")
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
            positivity_floor=POSITIVITY_FLOOR,
        ),
    )
    numerics.boundaries.add(
        wall_and_outflow_boundary_set(
            frame, instance, wall_faces=(frame.boundaries.x_min,)
        )
    )
    case.numerics(numerics, block=block)
    program = libtime.SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = N_CELLS):
    """Author a 1-d gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_open_layout,
        wall_and_outflow_boundary_set,
    )

    authored = _author(n_cells)
    layout = uniform_open_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int = N_CELLS, t_end: float = 0.6):
    """Compile, bind, and run the 1-d Noh case."""
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_open_layout,
        wall_and_outflow_boundary_set,
    )

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_open_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(initial_conserved(authored.n_cells), dtype=np.float64)
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return np.ascontiguousarray(np.reshape(field, (3, authored.n_cells)))
