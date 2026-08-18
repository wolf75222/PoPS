"""Public 1-d periodic CP-05 acoustic-toy authoring and native run.

The oracle is the 2-component linear system dt n + c dx u = 0,
dt u + c dx n = 0, with flux (c u, c n) and waves +/- c.
``build_case`` / ``resolve_plan`` author a public 1-d periodic Case
(Rusanov, MUSCL/VanLeer, SSPRK2). ``run_native`` compiles, binds
``exact_state(..., t=0, mode=)``, and advances the Case. The runtime
advance lives only inside ``run_native``. No Poisson.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_EXACT = load_sibling_module(_CASE_DIR / "exact.py")

N_CELLS = 32
WAVE_SPEED = float(_EXACT.WAVE_SPEED)
CFL = 0.4
MAX_STEPS = 100_000
COMPONENT_ORDER = ("n", "u")


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _require_mode(mode: str) -> str:
    name = str(mode)
    if name not in _EXACT.MODES:
        raise ValueError(f"unknown mode {mode!r}")
    return name


def cell_centers(n_cells: int = N_CELLS):
    """Uniform cell centers on the periodic unit interval."""
    return _EXACT.uniform_cell_centers(n_cells)


def initial_state(n_cells: int = N_CELLS, *, mode: str = "plus"):
    """Toy eigenmode IC at t=0. Shape (2, n)."""
    centers, _ = cell_centers(n_cells)
    return _EXACT.exact_state(centers, 0.0, mode=_require_mode(mode))


def evolved_state(n_cells: int = N_CELLS, t: float = 0.125, *, mode: str = "plus"):
    """Closed-form time-advanced eigenmode. Shape (2, n)."""
    centers, _ = cell_centers(n_cells)
    return _EXACT.exact_state(centers, t, mode=_require_mode(mode))


def _line_frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("cp05-line", lower=(0.0,), upper=(1.0,)).frame(Cartesian1D())


def _author(n_cells: int = N_CELLS) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.lib.time import SSPRK2
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.spaces import CellState
    from pops.time import AdaptiveCFL

    count = int(n_cells)
    frame = _line_frame()
    (x_axis,) = frame.axes
    model = pops.Model("cp05-acoustic-toy", frame=frame)
    state = model.state(
        "U",
        components=COMPONENT_ORDER,
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    density, velocity = state
    wavespeed = WAVE_SPEED
    flux = model.flux(
        "acoustic",
        frame=frame,
        state=state,
        components={x_axis: (wavespeed * velocity, wavespeed * density)},
        waves={x_axis: (-wavespeed, wavespeed)},
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    case = pops.Case("cp05-multifluid-modes")
    block = case.block("acoustic", model, states=(state,))
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
    program = SSPRK2(instance, rate=rate)
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


def build_case(n_cells: int = N_CELLS, *, mode: str = "plus"):
    """Author the 1-d periodic acoustic-toy Case. Does not compile or run."""
    _require_mode(mode)
    return _author(n_cells).case


def resolve_plan(n_cells: int = N_CELLS, *, mode: str = "plus"):
    """Validate and resolve the Case. Does not compile or execute a run."""
    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    _require_mode(mode)
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


def run_native(n_cells: int = N_CELLS, t_end: float = 0.05, *, mode: str = "plus"):
    """Compile, bind, and run the acoustic toy. Raises NativeUnavailable without Kokkos."""
    import pops

    from verification.pops_verify.case_authoring import (
        resolve_case,
        uniform_periodic_layout,
    )

    chosen = _require_mode(mode)
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(initial_state(authored.n_cells, mode=chosen), dtype=np.float64)
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("acoustic"), dtype=np.float64)
    return np.reshape(field, (2, authored.n_cells))
