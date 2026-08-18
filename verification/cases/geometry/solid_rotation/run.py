"""GE-02 in-memory solid-body rotation. No compile, bind, or pops.run.

Public 2-d Cartesian Case authoring is available; the manufactured rotation
lives in exact.py. The public polar System is not active;
``refuse_public_polar_runtime`` returns the documented reason. ``run_native``
raises ``NativeUnavailable`` with that same string. There is no public
PolarMesh runtime.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

N_CELLS = int(_exact.N_CELLS)
CFL = 0.4
PERIOD = float(_exact.PERIOD)
POLAR_RUNTIME_REFUSAL = "public polar System not active"


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def refuse_public_polar_runtime() -> str:
    """Return the documented reason the public polar System is not active."""
    return POLAR_RUNTIME_REFUSAL


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


def quarter_turn_fields(n_cells: int = N_CELLS) -> dict:
    """Return the exact field at t=T/4 and the expected peak location."""
    x, y, width = _exact.cell_centers(n_cells)
    return {
        "x": x,
        "y": y,
        "width": width,
        "volumes": _exact.cell_volumes(n_cells),
        "field": _exact.exact_scalar(x, y, PERIOD / 4.0),
        "peak": _exact.peak_location(PERIOD / 4.0),
    }


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "ge02-box",
        _exact.DOMAIN_LOWER,
        _exact.DOMAIN_UPPER,
    ).frame(Cartesian2D())


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
    model = pops.Model("ge02-solid-rotation", frame=frame)
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
    case = pops.Case("ge02-solid-rotation")
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


def run_native(n_cells: int = N_CELLS, t_end: float = PERIOD):
    """Refuse native polar compile. Cartesian authoring stays in ``build_case``."""
    del n_cells, t_end
    raise NativeUnavailable(refuse_public_polar_runtime())
