"""2-d periodic Euler Gresho-vortex authoring and initial data.

Initial conditions come from ``exact_gresho(..., t=0)``. ``build_case`` /
``resolve_plan`` author a public 2-d periodic Euler Case (Rusanov, MUSCL/VanLeer,
SSPRK2). 1-d is not applicable. ``run_native`` is optional and raises
``NativeUnavailable`` when a compiler is missing. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

GAMMA = float(_exact.GAMMA)
N_CELLS = 32


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the periodic box [0, PERIOD]^2."""
    count = int(n_cells)
    length = float(_exact.PERIOD)
    width = length / count
    centers = (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y, width


def initial_primitives(n_cells: int = N_CELLS):
    """Primitive IC at t=0. Each field has shape (n, n)."""
    x, y, _ = cell_centers(n_cells)
    return _exact.exact_gresho(x, y, 0.0)


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
    """Conserved IC from ``exact_gresho(..., t=0)``."""
    return primitives_to_conserved(initial_primitives(n_cells))


def _box_frame():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D

    length = float(_exact.PERIOD)
    return Rectangle("eu05-box", (0.0, 0.0), (length, length)).frame(Cartesian2D())


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case, instance, frame, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _author(n_cells: int = N_CELLS) -> _Authoring:
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
    frame = _box_frame()
    x_axis, y_axis = frame.axes
    model = pops.Model("eu05-euler", frame=frame)
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
    case = pops.Case("eu05-gresho")
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
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d periodic gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells, authored.n_cells))
    return resolve_case(authored.case, layout=layout)


def run_native(n_cells: int = N_CELLS, t_end: float = 1.0, *, request=None):
    """Compile, bind, and run the 2-d Gresho Case when a compiler is present."""
    import pops

    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )
    from verification.pops_verify.native_evidence import (
        maybe_campaign_payload,
        resolution_from_request,
    )

    if request is not None and int(request.pops_native_dim) != 2:
        raise NativeUnavailable(
            f"EU-05 requires pops_native_dim=2 (got {request.pops_native_dim}); "
            "no fallback"
        )
    n_cells = resolution_from_request(request, n_cells)
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    native = missing_native_compile_requirement(repo_include(), default_cxx())
    if native:
        raise NativeUnavailable(native)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(
        authored.frame, (authored.n_cells, authored.n_cells)
    )
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    conserved = initial_conserved(authored.n_cells)
    initial = np.ascontiguousarray(
        np.stack(
            [conserved["rho"], conserved["rho_u"], conserved["rho_v"], conserved["E"]]
        ),
        dtype=np.float64,
    )
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=100_000)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return maybe_campaign_payload(
        request,
        field,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=2,
    )
