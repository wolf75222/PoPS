"""1-d periodic Euler linear-wave authoring and native run.

Initial conditions come from ``exact_mode(..., t=0)``. ``build_case`` /
``resolve_plan`` author a public 1-d periodic Euler Case (Rusanov, MUSCL/VanLeer,
SSPRK2). ``run_native`` compiles, binds, and advances the Case.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

GAMMA = 1.4
N_CELLS = 32
CFL = 0.4
MAX_STEPS = 100_000
COMPONENT_ORDER = ("rho", "rho_u", "E")


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _exact_module():
    path = Path(__file__).with_name("exact.py")
    spec = importlib.util.spec_from_file_location("eu01_linear_waves_exact", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the periodic unit interval."""
    count = int(n_cells)
    width = 1.0 / count
    return (np.arange(count, dtype=np.float64) + 0.5) * width, width


def initial_primitives(n_cells: int = N_CELLS, *, mode="entropy", eps=1e-6, k=2.0 * math.pi):
    """Primitive IC W(x,0) = exact_mode(..., t=0). Shape (3, n)."""
    centers, _ = cell_centers(n_cells)
    return _exact_module().exact_mode(centers, 0.0, mode=mode, eps=eps, k=k)


def primitives_to_conserved(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    rho, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    momentum = rho * velocity
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * velocity * velocity
    return np.stack((rho, momentum, energy))


def initial_conserved(n_cells: int = N_CELLS, *, mode="entropy", eps=1e-6, k=2.0 * math.pi):
    """Conserved IC from ``exact_mode(..., t=0)``."""
    return primitives_to_conserved(initial_primitives(n_cells, mode=mode, eps=eps, k=k))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("eu01-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


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
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("eu01-euler", frame=frame)
    state = model.state(
        "U",
        components=COMPONENT_ORDER,
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
    case = pops.Case("eu01-linear-waves")
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
    """Author a 1-d periodic gamma-law Euler Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate and resolve the authored Case. Does not compile or run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _native_unavailable_reason() -> str | None:
    from tests.python.support.requirements import (
        default_cxx,
        missing_compiler_requirement,
        missing_native_compile_requirement,
        repo_include,
    )

    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def campaign_run_fields(n_cells: int, t_end: float, request) -> dict[str, object]:
    """Honest 1-d Serial campaign facts. Does not invent MPI or order."""
    count = int(n_cells)
    space = getattr(request, "execution_space", None) or "KokkosSerial"
    return {
        "compiler": os.environ.get("CXX", "c++"),
        "build_type": "native-dsl",
        "precision": "float64",
        "kokkos_execution_space": space,
        "mpi_enabled": False,
        "mpi_library": "none",
        "mpi_thread_level_requested": "none",
        "mpi_thread_level_provided": "none",
        "hdf5_collective_enabled": False,
        "mpi_ranks": 1,
        "omp_threads_per_rank": int(
            getattr(getattr(request, "resources", None), "omp_threads", None) or 1
        ),
        "gpus": 0,
        "resolution": [count],
        "block_size": [count],
        "amr_total_levels": 1,
        "refinement_ratio": 2,
        "subcycling": False,
        "time_program": "SSPRK2",
        "cfl": float(CFL),
        "final_time": float(t_end),
    }


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, mode="entropy", request=None):
    """Compile, bind, and run the 1-d linear wave. Raises NativeUnavailable without Kokkos."""
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"EU-01 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    if request is not None and request.min_resolution is not None:
        n_cells = int(request.min_resolution)
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(
        initial_conserved(authored.n_cells, mode=mode),
        dtype=np.float64,
    )
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.reshape(
        np.asarray(simulation.state_global("gas"), dtype=np.float64),
        (3, authored.n_cells),
    )
    if request is None:
        return field
    payload = campaign_run_fields(authored.n_cells, t_end, request)
    payload["result"] = field
    return payload
