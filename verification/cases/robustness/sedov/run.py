"""2-d off-centre Sedov authoring and native run.

Samples the documented circular R(θ) about the off-centre blast origin.
Anisotropy uses Task 18 ``radial_anisotropy``. ``build_case`` /
``resolve_plan`` author a public 2-d periodic Euler Case (Rusanov, MUSCL/VanLeer,
SSPRK2). ``run_native`` compiles, binds, and advances the Case. 1-d is not
applicable. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module
from verification.pops_verify.case_authoring import transmissive_boundary_set
from verification.pops_verify.symmetry import radial_anisotropy

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

GAMMA = float(_exact.GAMMA)
N_CELLS = 32
CFL = 0.4
MAX_STEPS = 100_000
COMPONENT_ORDER = ("rho", "rho_u", "rho_v", "E")
REQUIRED_NATIVE_DIM = 2
P_AMBIENT = 1.0e-5
DEPOSIT_RADIUS_CELLS = 2.0
DOMAIN_LOWER = tuple(float(value) for value in _exact.DOMAIN_LOWER)
DOMAIN_UPPER = tuple(float(value) for value in _exact.DOMAIN_UPPER)


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def polar_radii(theta, t):
    """Return the self-similar shock radius sampled at the given angles."""
    return _exact.polar_shock_radius(theta, t)


def front_anisotropy(theta, t) -> float:
    """Return Task 18 radial anisotropy of the sampled circular front."""
    return radial_anisotropy(polar_radii(theta, t))


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell-center mesh on the documented box [0, 1]^2."""
    count = int(n_cells)
    length_x = DOMAIN_UPPER[0] - DOMAIN_LOWER[0]
    length_y = DOMAIN_UPPER[1] - DOMAIN_LOWER[1]
    width_x = length_x / count
    width_y = length_y / count
    x_centers = DOMAIN_LOWER[0] + (np.arange(count, dtype=np.float64) + 0.5) * width_x
    y_centers = DOMAIN_LOWER[1] + (np.arange(count, dtype=np.float64) + 0.5) * width_y
    x, y = np.meshgrid(x_centers, y_centers, indexing="xy")
    return x, y, width_x


def initial_primitives(n_cells: int = N_CELLS):
    """Primitive IC: uniform ambient plus off-centre energy deposit. Shape (n, n)."""
    x, y, width = cell_centers(n_cells)
    rho = np.full(x.shape, float(_exact.RHO0), dtype=np.float64)
    velocity_x = np.zeros_like(x)
    velocity_y = np.zeros_like(x)
    pressure = np.full(x.shape, P_AMBIENT, dtype=np.float64)
    radius = np.hypot(x - float(_exact.X0), y - float(_exact.Y0))
    mask = radius <= (DEPOSIT_RADIUS_CELLS * width)
    if not np.any(mask):
        nearest = np.unravel_index(int(np.argmin(radius)), radius.shape)
        mask = np.zeros(radius.shape, dtype=bool)
        mask[nearest] = True
    extra_energy = float(_exact.BLAST_ENERGY) / (
        float(np.count_nonzero(mask)) * width * width
    )
    pressure = np.array(pressure, copy=True)
    pressure[mask] = pressure[mask] + (GAMMA - 1.0) * extra_energy
    return {"rho": rho, "u": velocity_x, "v": velocity_y, "p": pressure}


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
    """Conserved IC from the off-centre discrete Sedov deposit."""
    return primitives_to_conserved(initial_primitives(n_cells))


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
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D

    return Rectangle("rb05-box", DOMAIN_LOWER, DOMAIN_UPPER).frame(Cartesian2D())


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
    model = pops.Model("rb05-euler", frame=frame)
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
    case = pops.Case("rb05-sedov")
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
    numerics.boundaries.add(transmissive_boundary_set(frame, instance))
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
    """Author a 2-d periodic gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_open_layout,
        transmissive_boundary_set,
    )

    authored = _author(n_cells)
    layout = uniform_open_layout(authored.frame, (authored.n_cells, authored.n_cells))
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


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05):
    """Compile, bind, and run the 2-d off-centre Sedov blast.

    Returns named conserved fields ``rho``, ``rho_u``, ``rho_v``, ``E``, each
    a C-contiguous ``(n_cells, n_cells)`` array (COMPONENT_ORDER). Raises
    NativeUnavailable when ``POPS_NATIVE_DIM`` is not 2, or without Kokkos.
    """
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_open_layout,
        transmissive_boundary_set,
    )

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_open_layout(authored.frame, (authored.n_cells, authored.n_cells))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = pack_conserved(initial_conserved(authored.n_cells))
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("gas"), dtype=np.float64)
    return unpack_conserved(field, authored.n_cells)
