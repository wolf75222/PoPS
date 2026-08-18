"""2-d periodic uniform-flow authoring and manufactured leftover.

Initial conditions come from ``exact_primitives(..., t=0)``. Manufactured
block-face / CF leftover is a 1-cell bump at the mid-domain face. L∞ versus
the uniform state equals the bump amplitude. ``build_case`` / ``resolve_plan``
author a public 2-d periodic Euler Case (Rusanov, MUSCL/VanLeer, SSPRK2).
``run_native`` is optional and raises ``NativeUnavailable`` when a compiler
is missing. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

GAMMA = float(_exact.GAMMA)
N_CELLS = int(_exact.N_CELLS)
BUMP_AMPLITUDE = 0.25
INTERFACE_KINDS = ("block_face", "cf")


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the periodic box [0, PERIOD]^2."""
    return _exact.cell_centers(n_cells)


def initial_primitives(n_cells: int = N_CELLS):
    """Primitive IC at t=0. Each field has shape (n, n)."""
    x, y = cell_centers(n_cells)
    return _exact.exact_primitives(x, y, 0.0)


def primitives_to_conserved(primitives) -> dict:
    """Convert primitive (rho, u, v, p) to conserved (rho, rho u, rho v, E)."""
    rho = np.asarray(primitives["rho"], dtype=np.float64)
    velocity_x = np.asarray(primitives["u"], dtype=np.float64)
    velocity_y = np.asarray(primitives["v"], dtype=np.float64)
    pressure = np.asarray(primitives["p"], dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return {
        "rho": rho,
        "rho_u": rho * velocity_x,
        "rho_v": rho * velocity_y,
        "E": energy,
    }


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from ``exact_primitives(..., t=0)``."""
    return primitives_to_conserved(initial_primitives(n_cells))


def one_cell_bump(field, index, amplitude):
    """Return a copy of field with amplitude added at one (j, i) cell."""
    bumped = np.asarray(field, dtype=np.float64).copy()
    bumped[index] = bumped[index] + float(amplitude)
    return bumped


def leftover_linf(field, oracle, volumes) -> float:
    """Return L∞ of field versus the uniform oracle."""
    return reference_errors(field, oracle, volumes).linf


def manufactured_interface_bump(
    n_cells: int = N_CELLS, *, amplitude: float = BUMP_AMPLITUDE, kind: str = "block_face"
):
    """Return density with a 1-cell bump at the mid-domain block-face / CF."""
    if kind not in INTERFACE_KINDS:
        raise ValueError(f"unknown interface kind {kind!r}")
    x, y = cell_centers(n_cells)
    density = _exact.exact_primitives(x, y, 0.0)["rho"]
    index = _exact.interface_cell_index(n_cells)
    return one_cell_bump(density, index, amplitude)


def one_cell_bump_leftover_linf(
    n_cells: int = N_CELLS, *, amplitude: float = BUMP_AMPLITUDE
) -> float:
    """Return L∞ leftover of the manufactured 1-cell block-face / CF bump."""
    bumped = manufactured_interface_bump(n_cells, amplitude=amplitude, kind="block_face")
    x, y = cell_centers(n_cells)
    oracle = _exact.exact_primitives(x, y, 0.0)["rho"]
    volumes = _exact.cell_volumes(n_cells)
    return leftover_linf(bumped, oracle, volumes)


def _box_frame():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D

    length = float(_exact.PERIOD)
    return Rectangle("eu06-box", (0.0, 0.0), (length, length)).frame(Cartesian2D())


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d periodic gamma-law Euler Case. Does not compile or run."""
    import pops
    import pops.lib.time as libtime
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.math import ddt, div, sqrt
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Energy, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    del n_cells
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("eu06-euler", frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "rho_v", "E"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "rho_v": Momentum(axis=y_axis),
            "E": Energy(),
        },
    )
    rho, momentum_x, momentum_y, energy = state
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
    )
    sound = sqrt(GAMMA * pressure / rho)
    flux = model.flux(
        "euler",
        frame=frame,
        state=state,
        components={
            x_axis: (
                momentum_x,
                momentum_x * velocity_x + pressure,
                momentum_x * velocity_y,
                velocity_x * (energy + pressure),
            ),
            y_axis: (
                momentum_y,
                momentum_y * velocity_x,
                momentum_y * velocity_y + pressure,
                velocity_y * (energy + pressure),
            ),
        },
        waves={
            x_axis: (velocity_x - sound, velocity_x, velocity_x, velocity_x + sound),
            y_axis: (velocity_y - sound, velocity_y, velocity_y, velocity_y + sound),
        },
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    case = pops.Case("eu06-uniform-flow")
    block = case.block("gas", model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiter=limiters.VanLeer()),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = libtime.SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.4))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    count = int(n_cells)
    case = build_case(count)
    layout = uniform_periodic_layout(_box_frame(), (count, count))
    return resolve_case(case, layout=layout)


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05):
    """Optional native path. Raises NativeUnavailable without a compiler.

    A full native free-stream campaign is optional in this worktree. ICs
    stay available from ``initial_primitives`` / ``exact_primitives``.
    """
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable("optional native EU-06 run not executed in this worktree")
