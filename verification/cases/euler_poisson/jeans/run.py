"""1-d CP-10 Jeans ICs, public Euler Case, and optional native hook.

Initial data come from ``exact_state(..., t=0)`` via ``load_sibling_module``.
``build_case`` / ``resolve_plan`` author a public 1-d periodic Euler Case
(Rusanov, MUSCL/VanLeer, SSPRK2). Self-gravity stays on the oracle; it is not
injected into the Case. ``run_native`` is optional and raises
``NativeUnavailable``.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

GAMMA = 1.4
N_CELLS = 32
_EXACT = load_sibling_module(Path(__file__).with_name("exact.py"))
K_UNSTABLE = _EXACT.K_UNSTABLE
DOMAIN_LENGTH = _EXACT.DOMAIN_LENGTH


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the periodic interval of length 4π."""
    return _EXACT.uniform_cell_centers(n_cells)


def initial_state(n_cells: int = N_CELLS, *, k: float = K_UNSTABLE):
    """Jeans eigenmode IC at t=0."""
    centers, _ = cell_centers(n_cells)
    return _EXACT.exact_state(centers, 0.0, k=k)


def evolved_state(n_cells: int = N_CELLS, t: float = 1.0, *, k: float = K_UNSTABLE):
    """Closed-form time-advanced Jeans eigenmode."""
    centers, _ = cell_centers(n_cells)
    return _EXACT.exact_state(centers, t, k=k)


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain(
        "cp10-line", lower=(0.0,), upper=(float(DOMAIN_LENGTH),)
    ).frame(Cartesian1D())


def build_case(n_cells: int = N_CELLS):
    """Author a 1-d periodic gamma-law Euler Case. Does not compile or run."""
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
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("cp10-jeans-fluid", frame=frame)
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
    case = pops.Case("cp10-jeans")
    block = case.block("fluid", model, states=(state,))
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

    case = build_case(n_cells)
    layout = uniform_periodic_layout(_line_frame(), (int(n_cells),))
    return resolve_case(case, layout=layout)


def run_native(n_cells: int = N_CELLS, t_end: float = 1.0):
    """Optional native path. Raises NativeUnavailable without a compiler.

    A full native Jeans campaign is optional in this worktree. The
    dispersion oracle stays available from ``exact.py``.
    """
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable("optional native CP-10 run not executed in this worktree")
