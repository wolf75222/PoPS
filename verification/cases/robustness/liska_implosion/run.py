"""2-d Liska–Wendroff implosion authoring and native run.

Initial conditions come from ``primitives(..., t=0)``. ``build_case`` /
``resolve_plan`` author a public 2-d Euler Case (Rusanov, MUSCL/VanLeer,
SSPRK2) with public ``SlipWall`` reflecting faces. ``run_native`` compiles,
binds, and advances the Case. 1-d is not applicable. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_EXACT = load_sibling_module(Path(__file__).with_name("exact.py"))

GAMMA = float(_EXACT.GAMMA)
N_CELLS = 32
CFL = 0.4
MAX_STEPS = 100_000
COMPONENT_ORDER = ("rho", "rho_u", "rho_v", "E")
REQUIRED_NATIVE_DIM = 2
DOMAIN_LOWER = _EXACT.DOMAIN_LOWER
DOMAIN_UPPER = _EXACT.DOMAIN_UPPER


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on [0, 0.3]²."""
    count = int(n_cells)
    length = float(DOMAIN_UPPER[0] - DOMAIN_LOWER[0])
    width = length / count
    origin = float(DOMAIN_LOWER[0])
    centers = origin + (np.arange(count, dtype=np.float64) + 0.5) * width
    x, y = np.meshgrid(centers, centers, indexing="xy")
    return x, y, width


def initial_primitives(n_cells: int = N_CELLS):
    """Primitive IC W(x,y,0). Each field has shape (n, n)."""
    x, y, _ = cell_centers(n_cells)
    return _EXACT.primitives(x, y, 0.0)


def initial_conserved(n_cells: int = N_CELLS):
    """Conserved IC from ``primitives(..., t=0)``."""
    return _EXACT.primitives_to_conserved(initial_primitives(n_cells))


def leftover_residual(q) -> dict:
    """Return the x=y leftover residual of an already-sampled field."""
    return _EXACT.leftover_residual(q)


def pack_conserved(conserved) -> np.ndarray:
    """Return a C-contiguous (4, n, n) array in COMPONENT_ORDER."""
    return np.ascontiguousarray(
        np.stack([conserved[name] for name in COMPONENT_ORDER], axis=0),
        dtype=np.float64,
    )


def unpack_conserved(field, n_cells: int | None = None) -> dict:
    """Split a (4, n, n) or flat native buffer into named conserved fields."""
    count = N_CELLS if n_cells is None else int(n_cells)
    array = np.ascontiguousarray(field, dtype=np.float64)
    if array.shape == (4, count, count):
        stacked = array
    else:
        stacked = np.reshape(array, (4, count, count))
    return {name: stacked[index] for index, name in enumerate(COMPONENT_ORDER)}


def conserved_to_primitives(conserved) -> dict:
    """Convert packed or named conserved fields to primitives."""
    if isinstance(conserved, np.ndarray):
        named = unpack_conserved(conserved, conserved.shape[-1])
    else:
        named = conserved
    rho = np.asarray(named["rho"], dtype=np.float64)
    momentum_x = np.asarray(named["rho_u"], dtype=np.float64)
    momentum_y = np.asarray(named["rho_v"], dtype=np.float64)
    energy = np.asarray(named["E"], dtype=np.float64)
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    pressure = (GAMMA - 1.0) * (
        energy - 0.5 * rho * (velocity_x * velocity_x + velocity_y * velocity_y)
    )
    return {"rho": rho, "u": velocity_x, "v": velocity_y, "p": pressure}


def _box_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D

    return CartesianDomain(
        "rb07-box",
        DOMAIN_LOWER,
        DOMAIN_UPPER,
    ).frame(Cartesian2D())


def _reflecting_layout(frame, n_cells: int):
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid

    count = int(n_cells)
    return Uniform(CartesianGrid(frame=frame, cells=(count, count)))


def _author(n_cells: int = N_CELLS) -> _Authoring:
    import pops
    import pops.lib.time as libtime
    from pops.boundary import SlipWall, TransportBoundarySet
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
    model = pops.Model("rb07-euler", frame=frame)
    state = model.state(
        "U",
        components=COMPONENT_ORDER,
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
    case = pops.Case("rb07-liska-implosion")
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
    numerics.boundaries.add(
        TransportBoundarySet(
            {boundary: SlipWall(state=instance) for boundary in frame.boundaries.all}
        )
    )
    case.numerics(numerics, block=block)
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
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = N_CELLS):
    """Author a 2-d gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import resolve_case

    authored = _author(n_cells)
    layout = _reflecting_layout(authored.frame, authored.n_cells)
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    import os

    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    launched = os.environ.get("POPS_NATIVE_DIM", "")
    if launched != str(REQUIRED_NATIVE_DIM):
        return (
            f"POPS_NATIVE_DIM={launched!r} does not match required dim "
            f"{REQUIRED_NATIVE_DIM}; no fallback to another native extension"
        )
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(n_cells: int = N_CELLS, t_end: float = 2.5):
    """Compile, bind, and run the 2-d implosion.

    Raises NativeUnavailable when ``POPS_NATIVE_DIM`` is not 2, or without
    Kokkos.
    """
    import pops

    from verification.pops_verify.case_authoring import resolve_case

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = _reflecting_layout(authored.frame, authored.n_cells)
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = pack_conserved(initial_conserved(authored.n_cells))
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return unpack_conserved(field, authored.n_cells)
