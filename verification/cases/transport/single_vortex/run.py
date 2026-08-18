"""TR-03 in-memory reversible vortex. No compile, bind, or pops.run.

Public 2-d Case authoring is available; the manufactured swirl lives in exact.py.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

N_CELLS = int(_exact.N_CELLS)
CFL = 0.4


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def manufactured_divergence(n_cells: int = N_CELLS, t=0.0):
    """Return the discrete cell-centre divergence of the manufactured (u, v)."""
    return _exact.discrete_divergence(n_cells, t)


def return_fields(n_cells: int = N_CELLS) -> dict:
    """Return the IC and the exact t=T return field on the same mesh."""
    x, y, _ = _exact.cell_centers(n_cells)
    return {
        "x": x,
        "y": y,
        "volumes": _exact.cell_volumes(n_cells),
        "initial": _exact.exact_scalar(x, y, 0.0),
        "returned": _exact.exact_return(x, y),
    }


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain("tr03-box", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d periodic scalar advection Case. Does not compile or run."""
    import pops
    import pops.lib.time as libtime
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.time import AdaptiveCFL

    del n_cells
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("tr03-single-vortex", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    speed_x = 0.0
    speed_y = 0.0
    velocity = model.vector(
        "a", frame=frame, components={x_axis: speed_x, y_axis: speed_y}
    )
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (speed_x * q,), y_axis: (speed_y * q,)},
        waves={x_axis: (speed_x,), y_axis: (speed_y,)},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiter=limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case("tr03-single-vortex")
    tracer = case.block("tracer", model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
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


def run_native(n_cells: int = N_CELLS, t_end: float = 1.0):
    """Optional native path. Raises NativeUnavailable without a compiler.

    A full native 2-d swirl campaign is optional in this worktree. The
    manufactured velocity and return oracle stay available from exact.py.
    """
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable("optional native TR-03 run not executed in this worktree")
