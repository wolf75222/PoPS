"""2-d periodic oblique-wave ICs and public Case authoring.

Initial conditions come from the closed eigenmode via ``load_sibling_module``.
``build_case`` authors a public 2-d periodic pressureless Euler Case.
Poisson stays on the oracle. ``run_native`` is optional and raises
``NativeUnavailable``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.native_evidence import apply_campaign_request, require_bind_request
from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = _EXACT.N_CELLS


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_mesh(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the periodic unit square."""
    return _EXACT.uniform_cell_mesh(n_cells)


def initial_fields(n_cells: int = N_CELLS, t: float = 0.0):
    """Closed-form n, u, v, E, φ at one time on the uniform grid."""
    x, y, volumes = cell_mesh(n_cells)
    fields = _EXACT.exact_fields(x, y, float(t))
    fields["x"] = x
    fields["y"] = y
    fields["volumes"] = volumes
    return fields


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC (n, n u, n v) from the closed oracle at t=0."""
    sample = initial_fields(n_cells, 0.0)
    density = np.asarray(sample["n_e"], dtype=np.float64)
    velocity_x = np.asarray(sample["u_e"], dtype=np.float64)
    velocity_y = np.asarray(sample["v_e"], dtype=np.float64)
    return {
        "n": density,
        "n_u": density * velocity_x,
        "n_v": density * velocity_y,
    }


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "cp04-square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d periodic pressureless Euler Case. Does not compile or run."""
    import pops
    import pops.lib.time as libtime
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.physics import Density, Momentum
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    del n_cells
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("cp04-oblique-wave", frame=frame)
    state = model.state(
        "U",
        components=("n", "n_u", "n_v"),
        roles={
            "n": Density(),
            "n_u": Momentum(axis=x_axis),
            "n_v": Momentum(axis=y_axis),
        },
    )
    density, momentum_x, momentum_y = state
    velocity_x = momentum_x / density
    velocity_y = momentum_y / density
    flux = model.flux(
        "cold_electron",
        frame=frame,
        state=state,
        components={
            x_axis: (momentum_x, momentum_x * velocity_x, momentum_x * velocity_y),
            y_axis: (momentum_y, momentum_y * velocity_x, momentum_y * velocity_y),
        },
        waves={
            x_axis: (velocity_x, velocity_x, velocity_x),
            y_axis: (velocity_y, velocity_y, velocity_y),
        },
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    case = pops.Case("cp04-oblique-wave")
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

    count = int(n_cells)
    case = build_case(count)
    layout = uniform_periodic_layout(_box_frame(), (count, count))
    return resolve_case(case, layout=layout)


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, request=None):
    """Optional native path. Raises NativeUnavailable without a compiler.

    A full native 2-d oblique Langmuir campaign is optional in this worktree.
    ICs and the closed Poisson pack stay available from ``exact.py``.
    """
    n_cells = apply_campaign_request(
        n_cells, request, case_id='CP-04', allowed_dims=(2,), unavailable=NativeUnavailable
    )
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable(
        "CP-04 2-d Euler-Poisson field coupling is not on the public Case"
    )
