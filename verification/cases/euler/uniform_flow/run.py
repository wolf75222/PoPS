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
    """Cell-average primitive IC at t=0. Each field has shape (n, n)."""
    from verification.pops_verify.cell_averages import analytic_cell_averages

    x, y = cell_centers(n_cells)
    count = int(n_cells)
    width = float(x[0, 1] - x[0, 0]) if count > 1 else 1.0
    axis_lo = np.arange(count, dtype=np.float64) * width
    axis_hi = axis_lo + width
    x_lo, y_lo = np.meshgrid(axis_lo, axis_lo, indexing="xy")
    x_hi, y_hi = np.meshgrid(axis_hi, axis_hi, indexing="xy")
    lo = np.stack((x_lo, y_lo), axis=-1)
    hi = np.stack((x_hi, y_hi), axis=-1)

    def _component(name):
        def _fn(xx, yy):
            return _exact.exact_primitives(xx, yy, 0.0)[name]

        return analytic_cell_averages(_fn, lo, hi)

    return {
        "rho": _component("rho"),
        "u": _component("u"),
        "v": _component("v"),
        "p": _component("p"),
    }


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


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, request=None):
    """Compile, bind, and run the 2-d uniform flow when a compiler is present."""
    import pops

    from verification.pops_verify.native_toolchain import (
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
            f"EU-06 requires pops_native_dim=2 (got {request.pops_native_dim}); "
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
    primitives = initial_primitives(authored.n_cells)
    conserved = {
        "rho": np.asarray(primitives["rho"], dtype=np.float64),
        "u": np.asarray(primitives["u"], dtype=np.float64),
        "v": np.asarray(primitives["v"], dtype=np.float64),
        "p": np.asarray(primitives["p"], dtype=np.float64),
    }
    rho = conserved["rho"]
    energy = conserved["p"] / (GAMMA - 1.0) + 0.5 * rho * (
        conserved["u"] ** 2 + conserved["v"] ** 2
    )
    initial = np.ascontiguousarray(
        np.stack([rho, rho * conserved["u"], rho * conserved["v"], energy]),
        dtype=np.float64,
    )
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=100_000)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return maybe_campaign_payload(
        request,
        field,
        artifact=artifact,
        simulation=simulation,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=0.4,
        dimension=2,
    )
