"""1-d reflecting-cavity Euler standing-wave authoring and initial data.

Initial conditions come from ``primitives_1d(..., t=0)``. ``build_case`` /
``resolve_plan`` author a public 1-d Euler Case (Rusanov, MUSCL/VanLeer,
SSPRK2). The layout helper is periodic; a native cavity needs reflecting
walls (u=0 at x=0,1) and is not executed here. ``run_native`` is optional
and raises ``NativeUnavailable`` when a compiler is missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_EXACT = load_sibling_module(Path(__file__).with_name("exact.py"))

GAMMA = float(_EXACT.GAMMA)
N_CELLS = int(_EXACT.N_CELLS)


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the unit interval."""
    count = int(n_cells)
    width = 1.0 / count
    return (np.arange(count, dtype=np.float64) + 0.5) * width, width


def initial_primitives(n_cells: int = N_CELLS):
    """Cell-average primitive IC W(x,0). Shape (3, n)."""
    from verification.pops_verify.cell_averages import analytic_cell_averages

    count = int(n_cells)
    width = 1.0 / count
    lo = np.arange(count, dtype=np.float64) * width
    hi = lo + width

    def _component(index):
        def _fn(x):
            samples = np.asarray(x, dtype=np.float64)
            field = _EXACT.primitives_1d(samples.reshape(-1), 0.0)
            return np.asarray(field[index]).reshape(samples.shape)

        return analytic_cell_averages(_fn, lo, hi)

    return np.stack((_component(0), _component(1), _component(2)))


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from ``primitives_1d(..., t=0)``."""
    return _EXACT.primitives_to_conserved_1d(initial_primitives(n_cells))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("eu04-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


def _author(n_cells: int = N_CELLS):
    """Author a 1-d reflecting Euler Case and return (case, instance, frame, n)."""
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

    count = int(n_cells)
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("eu04-euler", frame=frame)
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
    case = pops.Case("eu04-standing-acoustic")
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
    from pops.boundary import SlipWall, TransportBoundarySet

    numerics.boundaries.add(
        TransportBoundarySet(
            {boundary: SlipWall(state=instance) for boundary in frame.boundaries.all}
        )
    )
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
    return case, instance, frame, count


def build_case(n_cells: int = N_CELLS):
    """Author a 1-d gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells)[0]


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_open_layout,
    )

    case, _instance, frame, count = _author(n_cells)
    layout = uniform_open_layout(frame, (count,))
    return resolve_case(case, layout=layout)


def run_native(n_cells: int = N_CELLS, t_end: float = 2.0, *, request=None):
    """Compile, bind, and run the reflecting cavity when a compiler is present."""
    import pops

    from verification.pops_verify.native_toolchain import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_open_layout,
    )
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"EU-04 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    native = missing_native_compile_requirement(repo_include(), default_cxx())
    if native:
        raise NativeUnavailable(native)
    case, instance, frame, count = _author(n_cells)
    layout = uniform_open_layout(frame, (count,))
    plan = resolve_case(case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(initial_conserved(count), dtype=np.float64)
    simulation = pops.bind(artifact, initial_values={instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=100_000)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return maybe_campaign_payload(
        request,
        field,
        artifact=artifact,
        simulation=simulation,
        n_cells=count,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=1,
    )
