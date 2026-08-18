"""IF-01 public Uniform advection under native PoPS MPI.

In-memory helpers sample the TR-01 sine on the four catalogued placements.
``run_native`` is the public pipeline only: Case → validate → resolve
(Uniform) → compile → bind (``ExecutionContext.mpi_world`` when the artifact
proves ``MPI_COMM_WORLD``) → ``pops.run``. Python does not launch ranks.
Launch this same entry under the campaign MPI launcher for a multi-rank run.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from verification.pops_verify.case_authoring import (
    attach_case_diagnostics,
    bind_public,
    load_sibling_module,
    resolve_case,
    uniform_periodic_layout,
)
from verification.pops_verify.reference_errors import reference_errors

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_TR01_RUN = (
    Path(__file__).resolve().parents[2] / "transport" / "advection_sine" / "run.py"
)
A = 1.0
CFL = 0.45
MAX_STEPS = 100_000


class NativeUnavailable(RuntimeError):
    """Raised when the optional native compile/run path cannot run."""


def four_block_splits():
    """Return the 1×4 / 4×1 1-d splits at 0.25 / 0.5 / 0.75."""
    return _exact.FOUR_BLOCK_SPLITS


def exact_fields_for_placements(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0):
    """Return exact fields keyed by the four MPI-style placements."""
    return {
        name: _exact.exact_on_placement(n_cells, name, t)
        for name in _exact.PLACEMENTS
    }


def max_decomposition_difference(n_cells: int = _exact.DEFAULT_N_CELLS, t=0.0) -> float:
    """Return the max pairwise L∞ between the four exact placements."""
    fields = exact_fields_for_placements(n_cells, t)
    volumes = _exact.cell_volumes(n_cells)
    linf = 0.0
    for left, right in combinations(fields.values(), 2):
        errors = reference_errors(left, right, volumes)
        linf = max(linf, errors.linf)
    return float(linf)


class _Authoring:
    __slots__ = ("case", "instance", "frame", "n_cells")

    def __init__(self, case: Any, instance: Any, frame: Any, n_cells: int) -> None:
        self.case = case
        self.instance = instance
        self.frame = frame
        self.n_cells = n_cells


def _sine_initial(frame):
    """Public analytic IC matching the TR-01 sine at t=0."""
    import math

    from pops.analytic import sin, x as analytic_x
    from pops.lib.initial import Analytic

    wave = 2.0 * math.pi * float(_exact._tr01.K)
    return Analytic(
        frame=frame,
        components=(
            float(_exact._tr01.Q0)
            + float(_exact._tr01.EPS) * sin(wave * analytic_x(frame)),
        ),
    )


def _frame():
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D

    return CartesianDomain("if01_unit_interval", (0.0,), (1.0,)).frame(Cartesian1D())


def _author(n_cells: int) -> _Authoring:
    import pops
    from pops.initial import InitialCondition
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
    frame = _frame()
    (x_axis,) = frame.axes
    model = pops.Model("if01_advection_sine", frame=frame)
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
    case = pops.Case("if01_mpi_invariance")
    tracer = case.block("tracer", model=model, states=(state,))
    instance = tracer[state]
    case.numerics(numerics, block=tracer)
    program = SSPRK2(instance, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=CFL))
    case.program(program)
    attach_case_diagnostics(case, tracer, program)
    case.initials.add(
        InitialCondition(
            state=instance,
            value=_sine_initial(frame),
            projection=ConservativeCellAverage(),
        )
    )
    return _Authoring(case=case, instance=instance, frame=frame, n_cells=count)


def build_case(n_cells: int = _exact.DEFAULT_N_CELLS):
    """Author the 1-d periodic Uniform Case. Does not compile or run."""
    return _author(n_cells).case


def resolve_plan(n_cells: int = _exact.DEFAULT_N_CELLS):
    """Validate and resolve Uniform. Decomposition is the communicator."""
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


def run_native(n_cells: int = _exact.DEFAULT_N_CELLS, t_end: float = 0.25):
    """Compile, bind, and run Uniform under the native communicator.

    Multi-rank invariance is obtained by launching this same function under
    MPI. This process does not spawn ranks.
    """
    import pops

    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    simulation = bind_public(artifact)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    return np.ravel(field)
