"""Public 1-d periodic scalar advection Case for TM-01.

Fine grid N=64 is fixed. The time step is a FixedDt argument so a Δt series
can be authored. Optional native compile/bind/run. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
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
from pops.time import FixedDt
from verification.pops_verify.native_toolchain import (
    default_cxx,
    missing_compiler_requirement,
    missing_native_compile_requirement,
    repo_include,
)
from verification.pops_verify.case_authoring import (
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)

_CASE_DIR = Path(__file__).resolve().parent
_exact = load_sibling_module(_CASE_DIR / "exact.py")

A = 1.0
N_CELLS = 64
DT = float(_exact.DT)
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def _frame():
    return CartesianDomain("unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells", "dt")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int, dt: float) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells
        self.dt = dt


def _author(dt, *, n_cells: int = N_CELLS) -> _Authoring:
    count = int(n_cells)
    step = float(dt)
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("tm01_pure_temporal", frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    (q,) = state
    velocity = model.vector("a", frame=frame, components={x_axis: A})
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=state,
        components={x_axis: (A * q,)},
        waves={x_axis: (A,)},
    )
    rate = model.rate("advection_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=velocity),
        ),
    )
    case = pops.Case("tm01_pure_temporal")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(FixedDt(step))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count, dt=step)


def build_case(dt, *, n_cells: int = N_CELLS) -> pops.Case:
    """Author the 1-d periodic scalar advection Case. Does not compile or run."""
    return _author(dt, n_cells=n_cells).case


def resolve_plan(dt, *, n_cells: int = N_CELLS):
    """Validate and resolve the Case. Does not compile or call pops.run."""
    authored = _author(dt, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    return resolve_case(authored.case, layout=layout)


def _initial_field(n_cells: int = N_CELLS) -> np.ndarray:
    from verification.pops_verify.cell_averages import analytic_cell_averages

    centers, volumes = _exact.uniform_cell_centers(n_cells)
    width = float(volumes[0])
    lo = centers - 0.5 * width
    hi = centers + 0.5 * width
    return np.ascontiguousarray(
        analytic_cell_averages(lambda x: _exact.exact_sine(x, 0.0), lo, hi)
    )


def _native_unavailable_reason() -> str | None:
    missing = missing_compiler_requirement()
    if missing:
        return missing
    return missing_native_compile_requirement(repo_include(), default_cxx())


def run_native(dt=None, t_end=1.0, *, n_cells: int = N_CELLS, request=None):
    """Compile, bind, and run the Case. Raises NativeUnavailable without a compiler."""
    from verification.pops_verify.native_evidence import maybe_campaign_payload

    if request is not None and int(request.pops_native_dim) != 1:
        raise NativeUnavailable(
            f"TM-01 requires pops_native_dim=1 (got {request.pops_native_dim}); "
            "no fallback"
        )
    if request is not None and request.min_resolution is not None:
        if int(request.min_resolution) != N_CELLS:
            raise NativeUnavailable(
                f"TM-01 uses fixed N={N_CELLS}; refusing grid override "
                f"min_resolution={request.min_resolution}"
            )
    n_cells = N_CELLS
    if dt is None:
        dt = DT
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(dt, n_cells=n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    initial = np.ascontiguousarray(
        _initial_field(authored.n_cells)[np.newaxis, :],
        dtype=np.float64,
    )
    simulation = pops.bind(artifact, initial_values={authored.instance: initial})
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.ravel(np.asarray(simulation.state_global("tracer"), dtype=np.float64))
    return maybe_campaign_payload(
        request,
        field,
        artifact=artifact,
        simulation=simulation,
        n_cells=authored.n_cells,
        t_end=t_end,
        time_program="SSPRK2",
        cfl=float(authored.dt) * float(authored.n_cells),
        dimension=1,
    )
