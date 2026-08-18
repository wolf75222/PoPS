"""IF-01 public Uniform advection under native PoPS MPI.

In-memory helpers sample the TR-01 sine on the four catalogued placements.
``run_native`` is the public pipeline only: Case → validate → resolve
(Uniform) → compile → bind (``ExecutionContext.mpi_world`` when the artifact
proves ``MPI_COMM_WORLD``) → ``pops.run``. Python does not launch ranks.
Launch this same entry under the campaign MPI launcher for a multi-rank run.
"""
from __future__ import annotations

from itertools import combinations
import os
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
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")
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

    wave = 2.0 * math.pi * float(_exact._tr01.KX)
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


def discovered_mpi_ranks() -> int:
    """Return the authenticated native communicator size. Does not spawn ranks.

    Launcher variables (OMPI/PMI/SLURM/POPS_CAMPAIGN_RANKS) are never a
    substitute for ``n_ranks()``. Missing, throwing, or unselected native
    worlds raise ``NativeUnavailable``.
    """
    from verification.pops_verify.mpi_world import NativeWorldError, native_world_size

    try:
        return int(native_world_size(required=True))
    except NativeWorldError as exc:
        raise NativeUnavailable(str(exc)) from exc


def _hdf5_collective_enabled() -> bool:
    try:
        from pops._native_selector import selected_native_module

        module = selected_native_module(required=False)
    except Exception:
        return False
    if module is None:
        return False
    return getattr(module, "__has_parallel_hdf5__", False) is True


def analytic_placements_are_not_mpi_proof() -> bool:
    """Exact-field identity across placements is not an MPI invariance result."""
    return True


def campaign_run_fields(n_cells: int, t_end: float, request) -> dict[str, object]:
    """Honest IF-01 campaign facts. MPI ranks come from the native communicator."""
    _v15.refuse_invalid_mode(request)
    count = int(n_cells)
    mpi_on = request is not None and getattr(request, "mpi_mode", "off") == "on"
    space = getattr(request, "execution_space", None) or "KokkosSerial"
    if mpi_on:
        ranks = discovered_mpi_ranks()
        if ranks < 2:
            raise NativeUnavailable(
                f"IF-01 mpi_mode=on discovered {ranks} rank(s); no serial fallback"
            )
        library = os.environ.get("POPS_MPI_LIBRARY") or "unknown"
        thread = "MPI_THREAD_SINGLE"
    else:
        ranks = 1
        library = "none"
        thread = "none"
    return {
        "compiler": os.environ.get("CXX", "c++"),
        "build_type": "native-dsl",
        "precision": "float64",
        "kokkos_execution_space": space,
        "mpi_enabled": mpi_on,
        "mpi_library": library,
        "mpi_thread_level_requested": thread,
        "mpi_thread_level_provided": thread,
        "hdf5_collective_enabled": _hdf5_collective_enabled() if mpi_on else False,
        "mpi_ranks": int(ranks),
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
        "comparison_artifacts": {
            "kind": "mpi_decomposition",
            "placements": list(_exact.PLACEMENTS),
            "note": "analytic placements are not MPI proof",
        },
    }


def run_native(n_cells: int = _exact.DEFAULT_N_CELLS, t_end: float = 0.25, request=None):
    """Compile, bind, and run Uniform under the native communicator.

    Multi-rank invariance is obtained by launching this same function under
    MPI. This process does not spawn ranks. ``request.mpi_mode`` is forwarded
    to ``bind_public``; omitted ``request`` binds serially.
    """
    import pops

    _v15.bind_campaign(request, NativeUnavailable)
    if request is not None:
        if request.min_resolution is not None:
            n_cells = int(request.min_resolution)
        mpi_mode = request.mpi_mode
    else:
        mpi_mode = "off"
    missing = _native_unavailable_reason()
    if missing:
        raise NativeUnavailable(missing)
    authored = _author(n_cells)
    layout = uniform_periodic_layout(authored.frame, (authored.n_cells,))
    plan = resolve_case(authored.case, layout=layout)
    artifact = pops.compile(plan)
    simulation = bind_public(artifact, mpi_mode=mpi_mode)
    pops.run(simulation, t_end=float(t_end), max_steps=MAX_STEPS)
    field = np.asarray(simulation.state_global("tracer"), dtype=np.float64)
    if request is not None:
        return campaign_run_fields(authored.n_cells, t_end, request)
    return np.ravel(field)
