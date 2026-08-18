"""GE-04 Cartesian authoring and polar→Cartesian interpolation.

The public Case is Cartesian only. The polar branch is in-memory sampling;
``refuse_public_polar_runtime`` returns the documented reason. ``run_native``
raises ``NativeUnavailable`` with that same string. There is no public
PolarMesh runtime.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

N_CELLS = int(_exact.N_CELLS)
N_R = int(_exact.N_R)
N_THETA = int(_exact.N_THETA)
LINF_BOUND = 0.05
POLAR_RUNTIME_REFUSAL = "public polar System not active"


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def refuse_public_polar_runtime() -> str:
    """Return the documented reason the public polar System is not active."""
    return POLAR_RUNTIME_REFUSAL


def field_to_field_errors(
    n_cells: int = N_CELLS,
    n_r: int = N_R,
    n_theta: int = N_THETA,
    *,
    method: str = "bilinear",
):
    """L1/L2/L∞ of polar→Cartesian interpolation vs Cartesian samples of φ."""
    cartesian = _exact.sample_cartesian(n_cells)
    polar = _exact.sample_polar(n_r, n_theta)
    interpolated = _exact.interpolate_polar_to_cartesian(
        polar["phi"],
        polar["r_centers"],
        polar["theta_centers"],
        cartesian["x"],
        cartesian["y"],
        method=method,
    )
    return reference_errors(interpolated, cartesian["phi"], cartesian["volumes"])


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "ge04-box",
        _exact.DOMAIN_LOWER,
        _exact.DOMAIN_UPPER,
    ).frame(Cartesian2D())


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d Cartesian scalar Case. Does not compile or run."""
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
    model = pops.Model("ge04-radial-ring", frame=frame)
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
    case = pops.Case("ge04-cartesian-polar-oracle")
    tracer = case.block("tracer", model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
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
    """Validate and resolve the authored Cartesian Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    count = int(n_cells)
    case = build_case(count)
    layout = uniform_periodic_layout(_box_frame(), (count, count))
    return resolve_case(case, layout=layout)


def run_native(n_cells: int = N_CELLS):
    """Refuse native polar compile. Cartesian authoring stays in ``build_case``."""
    del n_cells
    raise NativeUnavailable(refuse_public_polar_runtime())
