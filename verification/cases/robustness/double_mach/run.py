"""2-d Woodward–Colella DMR authoring and initial data.

Initial conditions come from ``primitives(..., t=0)``. ``build_case`` /
``resolve_plan`` author a public 2-d Euler Case (Rusanov, MUSCL/VanLeer,
SSPRK2). The physical DMR box is not periodic; the shared layout helper
is. ``run_native`` is optional and raises ``NativeUnavailable``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.case_authoring import transmissive_boundary_set

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

GAMMA = float(_exact.GAMMA)
N_CELLS = int(_exact.N_CELLS)


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the documented [0, 4] x [0, 1] box."""
    count = int(n_cells)
    lower_x, lower_y = (float(value) for value in _exact.DOMAIN_LOWER)
    upper_x, upper_y = (float(value) for value in _exact.DOMAIN_UPPER)
    width_x = (upper_x - lower_x) / count
    width_y = (upper_y - lower_y) / count
    x_centers = lower_x + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = lower_y + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x, y, width_x, width_y


def initial_primitives(n_cells: int = N_CELLS):
    """Primitive IC at t=0. Each field has shape (n, n)."""
    x, y, _, _ = cell_centers(n_cells)
    return _exact.primitives(x, y, 0.0)


def primitives_to_conserved(fields) -> dict:
    """Convert primitive (rho, u, v, p) to conserved (rho, rho u, rho v, E)."""
    return _exact.primitives_to_conserved(fields)


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from ``primitives(..., t=0)``."""
    return primitives_to_conserved(initial_primitives(n_cells))


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "rb08-box",
        _exact.DOMAIN_LOWER,
        _exact.DOMAIN_UPPER,
    ).frame(Cartesian2D())


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d gamma-law Euler Case. Does not compile or run."""
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
    model = pops.Model("rb08-euler", frame=frame)
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
    case = pops.Case("rb08-double-mach")
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
    numerics.boundaries.add(transmissive_boundary_set(frame, instance))
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
        uniform_open_layout,
        transmissive_boundary_set,
    )

    count = int(n_cells)
    case = build_case(count)
    layout = uniform_open_layout(_box_frame(), (count, count))
    return resolve_case(case, layout=layout)


def run_native(n_cells: int = N_CELLS, t_end: float = 0.2):
    """Optional native path. Raises NativeUnavailable without a compiler.

    A full native DMR campaign is optional in this worktree. ICs and the
    t=0 geometry helpers stay available from ``exact.py``.
    """
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable("optional native RB-08 run not executed in this worktree")
