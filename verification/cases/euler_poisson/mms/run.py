"""1-d periodic Euler–Poisson MMS authoring and initial data.

Initial conditions come from ``primitives_1d(..., t=0)`` via
``load_sibling_module``. ``build_case`` / ``resolve_plan`` author a public
1-d periodic Euler Case (Rusanov, MUSCL/VanLeer, SSPRK2). Poisson is
satisfied by the oracle identity; manufactured hyperbolic sources live in
``exact.sources_1d`` and are not yet injected into the Case. ``run_native``
is optional and raises ``NativeUnavailable`` when a compiler is missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

N_CELLS = 32
_EXACT = load_sibling_module(Path(__file__).with_name("exact.py"))
GAMMA = _EXACT.GAMMA


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / count
    return (np.arange(count, dtype=np.float64) + 0.5) * width, width


def initial_primitives(n_cells: int = N_CELLS):
    """Primitive IC W(x,0). Shape (3, n)."""
    centers, _ = cell_centers(n_cells)
    return _EXACT.primitives_1d(centers, 0.0)


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from ``primitives_1d(..., t=0)``."""
    return _EXACT.primitives_to_conserved_1d(initial_primitives(n_cells))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("cp01-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


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
    model = pops.Model("cp01-euler-poisson", frame=frame)
    state = model.state(
        "U",
        components=("n", "n_u", "E"),
        roles={
            "n": Density(),
            "n_u": Momentum(axis=x_axis),
            "E": Energy(),
        },
    )
    density, momentum, energy = state
    velocity = momentum / density
    pressure = (GAMMA - 1.0) * (energy - 0.5 * density * velocity * velocity)
    sound = sqrt(GAMMA * pressure / density)
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
    case = pops.Case("cp01-mms")
    block = case.block("electrons", model, states=(state,))
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


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05):
    """Optional native path. Raises NativeUnavailable without a compiler.

    A full native coupled MMS campaign is optional in this worktree. ICs and
    manufactured sources stay available from ``exact.py``.
    """
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable("optional native CP-01 run not executed in this worktree")
